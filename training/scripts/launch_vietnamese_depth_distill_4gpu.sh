#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$REPO_DIR/runs/vi_depth_distill"
LOG_PATH="$RUN_DIR/launcher.log"
GPU_IDS=(4 5 6 7)

mkdir -p "$RUN_DIR"
exec >>"$LOG_PATH" 2>&1

echo "$(date --iso-8601=seconds) waiting for physical GPUs ${GPU_IDS[*]}"
while true; do
    busy=()
    for gpu_id in "${GPU_IDS[@]}"; do
        pids="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits)"
        if [[ -n "$pids" ]]; then
            busy+=("$gpu_id:$pids")
        fi
    done
    if (( ${#busy[@]} == 0 )); then
        break
    fi
    echo "$(date --iso-8601=seconds) waiting; occupied ${busy[*]}"
    sleep 60
done

cd "$REPO_DIR"
echo "$(date --iso-8601=seconds) launching 200k-step Vietnamese 24L->6L distillation"
exec env CUDA_VISIBLE_DEVICES=4,5,6,7 \
    uv run torchrun --standalone --nproc-per-node 4 \
    training/train.py training/configs/vietnamese_depth_distill.yaml
