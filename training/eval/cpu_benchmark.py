"""Reproducible CPU latency, throughput, and memory benchmark for Pocket TTS.

Measures steady-state streaming generation after one warmup. TTFA starts
immediately before requesting the first audio chunk; voice encoding is measured
separately. The project reports throughput as audio_seconds / wall_seconds
("Nx faster than real time"), while conventional RTF is its inverse; both are
written to avoid ambiguity.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from pocket_tts import TTSModel
from pocket_tts.utils.config import load_config
from pocket_tts.utils.utils import download_if_necessary

DEFAULT_TEXTS = [
    "Xin chào, đây là phép đo hiệu năng tổng hợp giọng nói trên bộ xử lý trung tâm.",
    "Trời hôm nay nắng đẹp, chúng tôi quyết định đi dạo quanh hồ và uống một ly cà phê.",
    "Mô hình nhỏ hơn cần phải tạo ra âm thanh nhanh, rõ ràng và giữ được đặc trưng của giọng mẫu.",
]


def current_rss_mib() -> float:
    """Resident set size from procfs, without an optional psutil dependency."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is unavailable in /proc/self/status")


def process_peak_rss_mib() -> float:
    # Linux reports ru_maxrss in KiB; this benchmark is Linux-only training tooling.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class PeakRssSampler:
    def __init__(self, interval_sec: float = 0.01):
        self.interval_sec = interval_sec
        self.peak_mib = current_rss_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self.peak_mib = max(self.peak_mib, current_rss_mib())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, current_rss_mib())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def voice_from_manifest(path: Path, min_sec: float) -> str:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("duration", 0) >= min_sec:
                return row["path"]
    raise ValueError(f"no voice at least {min_sec}s long in {path}")


def cpu_name() -> str:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def aggregate(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(q: float) -> float:
        pos = (len(ordered) - 1) * q
        low = int(pos)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": percentile(0.95),
    }


def generate_once(model, voice_state, text: str, seed: int) -> dict:
    torch.manual_seed(seed)
    gc.collect()
    rss_before = current_rss_mib()
    started = time.perf_counter()
    stream = iter(model.generate_audio_stream(voice_state, text))
    with PeakRssSampler() as memory:
        first = next(stream)
        first_at = time.perf_counter()
        samples = first.numel()
        chunks = 1
        for chunk in stream:
            samples += chunk.numel()
            chunks += 1
    finished = time.perf_counter()
    wall_sec = finished - started
    audio_sec = samples / model.sample_rate
    return {
        "seed": seed,
        "text": text,
        "text_characters": len(text),
        "ttfa_sec": first_at - started,
        "wall_sec": wall_sec,
        "audio_sec": audio_sec,
        "realtime_speedup_x": audio_sec / wall_sec,
        "rtf_wall_over_audio": wall_sec / audio_sec,
        "chunks": chunks,
        "samples": samples,
        "rss_before_mib": rss_before,
        "inference_peak_rss_mib": memory.peak_mib,
        "inference_peak_delta_mib": memory.peak_mib - rss_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--voice", default=None)
    parser.add_argument(
        "--voice-manifest", type=Path, default=Path("data/vi_combined/valid_aligned.jsonl")
    )
    parser.add_argument("--voice-sec", type=float, default=5.0)
    parser.add_argument("--text", action="append", dest="texts")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup_runs < 0 or args.torch_threads < 1:
        parser.error(
            "--runs and --torch-threads must be positive; --warmup-runs cannot be negative"
        )

    torch.set_num_threads(args.torch_threads)
    texts = args.texts or DEFAULT_TEXTS
    config = load_config(args.config)
    weights = Path(download_if_necessary(str(config.weights_path))).resolve()
    voice = args.voice or voice_from_manifest(args.voice_manifest, args.voice_sec)
    started_at = datetime.now(timezone.utc).isoformat()
    initial_rss = current_rss_mib()

    t0 = time.perf_counter()
    model = TTSModel.load_model(config=args.config)
    load_sec = time.perf_counter() - t0
    loaded_rss = current_rss_mib()
    assert model.device.type == "cpu", model.device

    t0 = time.perf_counter()
    voice_state = model.get_state_for_audio_prompt(voice)
    voice_prepare_sec = time.perf_counter() - t0
    ready_rss = current_rss_mib()

    for i in range(args.warmup_runs):
        generate_once(model, voice_state, texts[i % len(texts)], args.seed + 10_000 + i)

    records = [
        generate_once(model, voice_state, texts[i % len(texts)], args.seed + i)
        for i in range(args.runs)
    ]
    result = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "weights": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": sha256(weights),
        "voice": voice,
        "voice_manifest": str(args.voice_manifest),
        "voice_sec": args.voice_sec,
        "settings": {
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "base_seed": args.seed,
            "torch_threads_per_inference_thread": args.torch_threads,
            "sampler_decode_steps": model.sampler_decode_steps,
            "temperature": model.temp,
            "sample_rate": model.sample_rate,
        },
        "environment": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "load_sec": load_sec,
        "voice_prepare_sec": voice_prepare_sec,
        "initial_rss_mib": initial_rss,
        "loaded_rss_mib": loaded_rss,
        "ready_rss_mib": ready_rss,
        "process_peak_rss_mib": process_peak_rss_mib(),
        "summary": {
            "ttfa_sec": aggregate([r["ttfa_sec"] for r in records]),
            "realtime_speedup_x": aggregate([r["realtime_speedup_x"] for r in records]),
            "rtf_wall_over_audio": aggregate([r["rtf_wall_over_audio"] for r in records]),
            "inference_peak_rss_mib": max(r["inference_peak_rss_mib"] for r in records),
            "inference_peak_delta_mib": max(r["inference_peak_delta_mib"] for r in records),
        },
        "runs": records,
    }
    # Capture the high-water mark after hashing and result assembly too.
    result["process_peak_rss_mib"] = process_peak_rss_mib()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
