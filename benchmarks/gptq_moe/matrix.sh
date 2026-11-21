#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one GPU UUID or index}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
bench_dir=$(cd "$(dirname "$0")" && pwd)
out=${1:?Usage: bash matrix.sh OUTPUT_DIRECTORY [LOCAL_MODEL_DIRECTORY]}
mkdir -p "$out"
model_args=()
if [[ $# -ge 2 ]]; then
    model_args=(--model-dir "$2")
fi

run() {
    local label=$1 mode=$2 samples=$3
    python "$bench_dir/run.py" --mode "$mode" --samples "$samples" \
        "${model_args[@]}" --out "$out/$label.json" > "$out/$label.log" 2>&1 || {
        tail -n 30 "$out/$label.log" >&2
        return 1
    }
    tail -n 1 "$out/$label.log"
}

run smoke-auto auto 1
run smoke-none none 1
run 01-auto auto 8
run 02-none none 8
run 03-none none 8
run 04-auto auto 8
run 05-auto auto 8
run 06-none none 8
