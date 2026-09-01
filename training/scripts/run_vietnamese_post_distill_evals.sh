#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$REPO_DIR/runs/vi_depth_distill"
FINAL_CHECKPOINT="$RUN_DIR/checkpoint_00200000.pt"
LOG_PATH="$RUN_DIR/post_eval.log"

mkdir -p "$RUN_DIR"
exec >>"$LOG_PATH" 2>&1
cd "$REPO_DIR"

echo "$(date --iso-8601=seconds) waiting for the atomic step-200000 checkpoint and export"
while true; do
    if [[ -s "$FINAL_CHECKPOINT" ]] \
        && [[ -s "$RUN_DIR/model.safetensors" ]] \
        && rg -q '"type": "checkpoint", "step": 200000' "$RUN_DIR/progress.jsonl"; then
        break
    fi
    if ! pgrep -f 'training/train.py training/configs/vietnamese_depth_distill.yaml' >/dev/null; then
        echo "$(date --iso-8601=seconds) training exited before the final export" >&2
        exit 1
    fi
    sleep 300
done

echo "$(date --iso-8601=seconds) running full Vietnamese FLEURS WER/CER evaluation"
env CUDA_VISIBLE_DEVICES=4 uv run python -m training.eval.fleurs_vi \
    "$RUN_DIR" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --use-ema \
    --prompts-from data/vi_combined/valid_aligned.jsonl \
    --batch-size 16 \
    --temp 0.3 \
    --cfg 1.0 \
    --n-steps 1 \
    --eos-threshold -1.0 \
    --out "$RUN_DIR/fleurs_vi_00200000.json"

echo "$(date --iso-8601=seconds) running steady-state CPU latency/throughput/memory evaluation"
env CUDA_VISIBLE_DEVICES='' uv run python -m training.eval.cpu_benchmark \
    --config pocket_tts/config/vietnamese.yaml \
    --voice-manifest data/vi_combined/valid_aligned.jsonl \
    --runs 5 \
    --warmup-runs 1 \
    --torch-threads 1 \
    --output "$RUN_DIR/cpu_benchmark.json"

echo "$(date --iso-8601=seconds) all post-distillation evaluations completed"
