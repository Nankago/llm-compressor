# Reproduce the calibration-forward ablation

Use a checkout of this branch and the pinned KL examples at `8e7e6ad781a416b967b3c4a498d8469d97224d61`. The production source in both timing arms is the fork at `c81852cbfc04fca46f5f85a9d88a60525efb4385`. Relative to upstream `f7d824c`, its only change under `src/` is the previously tested symmetric observer optimization. The driver verifies and reloads the original checkpoint in each subprocess. It retains raw data outside Git.

```bash
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT/benchmarks/gptq_moe"
PY=/tmp/moe-gptq-1121-venv/bin/python
export PYTHONPATH="$ROOT/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
MODEL=/tmp/imatrix-granite-408b6e9
TINY=/tmp/moe-gptq-1121-model
BANK="$ROOT/../llm-compressor-moe-calibration/tokens"
KL="$ROOT/../llm-compressor-qda-kl/examples/qda_kl_eval"
KA="$ROOT/../llm-compressor-qda-artifacts"
OUT=$(mktemp -d)
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv
# Set GPU, SMALL and optionally EVAL to idle physical UUIDs before launching.
"$PY" forward_trials.py --checkpoint granite --model-dir "$MODEL" --calibration-dir "$BANK" --gpu "$GPU" --out "$OUT/granite"
"$PY" forward_trials.py --checkpoint tinymoe --model-dir "$TINY" --gpu "$SMALL" --out "$OUT/tinymoe" --samples 8 --length 128 --pairs 2
```

The first pair (`r0`) primes the full workload and the private disk cache. `r1` exports the two packed checkpoints *after* timing. Trace and quality are independent invocations. The half-second NVML guard terminates its own process if a foreign GPU PID appears. To rebuild the Granite token bank and existing reference cache, use the [WikiText calibration commands](results/granite-calibration-load.md) and [paired KL commands at the pinned commit](https://github.com/Nankago/llm-compressor/blob/8e7e6ad781a416b967b3c4a498d8469d97224d61/examples/qda_kl_eval/PAIRED.md).

```bash
guard() {
    local gpu=$1 label=$2
    shift 2
    "$PY" forward_command.py --gpu "$gpu" --prefix "$OUT/$label" \
        --cache-root "$OUT/granite/compiled" --kl-tools "$KL" "$@"
}
ARC_REV=210d026faf9955653af8916fad021475a3f00453
curl -fL "https://huggingface.co/datasets/allenai/ai2_arc/resolve/$ARC_REV/ARC-Easy/test-00000-of-00001.parquet" -o "$OUT/arc-easy-test.parquet"
guard "${EVAL:-$GPU}" quality-granite forward_quality.py "$MODEL" "$OUT/granite/checkpoint-on" "$OUT/granite/checkpoint-off" "$KA/eval.jsonl" "$OUT/granite-quality.json" --granite-cache "$KA/pair-cli-hidden" --arc "$OUT/arc-easy-test.parquet"
"$PY" "$KL/inputs.py" "$KA/test.parquet" "$TINY" "$OUT/tinymoe-eval.jsonl"
guard "$SMALL" quality-tiny-v2 forward_quality.py "$TINY" "$OUT/tinymoe/checkpoint-on" "$OUT/tinymoe/checkpoint-off" "$OUT/tinymoe-eval.jsonl" "$OUT/tiny-quality.json"
guard "$SMALL" sqnr-check run.py --checkpoint tinymoe --model-dir "$TINY" --mode auto --samples 2 --seq-length 128 --no-propagate-error --log-sequential-error --coarse-timing --out "$OUT/sqnr-check.json"
guard "$GPU" trace-granite profile_forward.py --model-dir "$MODEL" --calibration-dir "$BANK" --out "$OUT/trace"
"$PY" summarize_forward_trace.py "$OUT/trace" "$OUT/trace-summary.json"
guard "$GPU" gram-replay gram_replay.py "$OUT/trace" "$OUT/gram-replay.json"
"$PY" record_forward_results.py "$OUT" "$OUT/forward-ablation.json"
```

The archived runs use `/gemini-0/space/private/lmx/temp-restore/Vllm-Omni/llm-compressor-forward-ablation-artifacts` for `OUT`. Chrome traces are under `trace/window-8.json` and `trace/window-40.json`; real intermediate tensors for the focused replay are under `trace/replay-*.safetensors`. The per-arm model weights are under `{granite,tinymoe}/checkpoint-{on,off}/`.

### Isolated Gram-reuse experiment

Use the same shell variables and guard. These trials keep error propagation on and use the already primed Granite cache from P0. Each run starts a fresh process. The first reuse run exports the representative checkpoint after timing.

```bash
mkdir -p "$OUT/reuse"
gargs=(--mode auto --coarse-timing --checkpoint granite --model-dir "$MODEL" --calibration-dir "$BANK" --samples 64 --seq-length 1024)
for label in r1-reuse r1-base r2-base r2-reuse r3-reuse r3-base r4-base r4-reuse; do
    driver=run.py
    extra=()
    if [[ $label == *-reuse ]]; then driver=try_gram_reuse.py; fi
    if [[ $label == r1-reuse ]]; then extra=(--save-checkpoint "$OUT/reuse/checkpoint"); fi
    guard "$GPU" "reuse/$label" "$driver" "${gargs[@]}" "${extra[@]}" --out "$OUT/reuse/$label.json"
done
guard "$SMALL" reuse/test-final test_gram_trial.py
guard "$SMALL" reuse/tiny-on try_gram_reuse.py --checkpoint tinymoe --model-dir "$TINY" --mode auto --coarse-timing --samples 8 --seq-length 128 --save-checkpoint "$OUT/reuse/tiny-checkpoint" --out "$OUT/reuse/tiny-on.json"
guard "$SMALL" reuse/granite-off try_gram_reuse.py "${gargs[@]}" --no-propagate-error --save-checkpoint "$OUT/reuse/off-checkpoint" --out "$OUT/reuse/granite-off.json"
"$PY" record_gram_trial.py "$OUT" "$OUT/gram-trial.json"
```
