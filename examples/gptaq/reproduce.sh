#!/usr/bin/env bash
# Run at the root of this checkout using the versions in README.md.
set -euo pipefail
: "${MODEL:?pinned Qwen3-0.6B snapshot}" "${TRAIN:?WikiText train parquet}"
: "${TEST:?WikiText test parquet}" "${ARC:?ARC-Easy test parquet}" "${OUT:?fresh output directory}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
here=examples/gptaq
build() {
    local label=$1 seed=$2 samples=$3 length=$4
    shift 4
    python "$here/build.py" --model "$MODEL" --train "$TRAIN" --seed "$seed" --samples "$samples" --length "$length" --out "$OUT/$label" "$@"
}
score() {
    local file=$1
    shift
    python "$here/score.py" --model "$MODEL" --test "$TEST" --out "$OUT/$file" --candidates "$@"
}
python -m pytest -q "$here/test_refit.py" "$here/test_coordinate.py"
if [[ -n "${QDA:-}" ]]; then python "$here/check_qda.py" "$QDA" > "$OUT/qda-check.json"; fi
build s42-stock 42 32 128 --alpha 0
build s42-gptaq 42 32 128 --alpha .3
build s43-gptaq 43 32 128 --alpha .3
build s43-stock 43 32 128 --alpha 0
score short-panel.json "$OUT/s42-stock" "$OUT/s42-gptaq" "$OUT/s43-stock" "$OUT/s43-gptaq"
build long-sweep 42 128 256 --alpha 0 --sweep
build long-stock 42 128 256 --alpha 0
build long-gptaq 42 128 256 --alpha .3
build rep-stock 43 128 256 --alpha 0
build rep-gptaq 43 128 256 --alpha .3
build rep-sweep 43 128 256 --alpha 0 --sweep
build long-float 42 128 256 --alpha 0 --float-prefix
build rep-float 43 128 256 --alpha 0 --float-prefix
for p in long rep; do score "$p-float-panel.json" "$OUT/$p-stock" "$OUT/$p-float"; done
for p in long rep; do score "$p-panel.json" "$OUT/$p-stock" "$OUT/$p-sweep" "$OUT/$p-gptaq"; done
python "$here/native_smoke.py" --model "$MODEL" --train "$TRAIN" --out "$OUT/native"
python "$here/native_smoke.py" --full --model "$MODEL" --train "$TRAIN" --out "$OUT/native-full"
score native-panel.json "$OUT/native-full/stock" "$OUT/native-full/sweep"
for name in bf16 long-stock long-float long-gptaq long-sweep rep-stock rep-float rep-gptaq rep-sweep; do
    checkpoint="$OUT/$name"
    if [[ $name == bf16 ]]; then checkpoint="$MODEL"; fi
    python "$here/arc.py" --model "$checkpoint" --tokenizer "$MODEL" --test "$ARC" --out "$OUT/arc-$name.json"
done
python "$here/summarize.py" "$OUT" "$OUT/results.json"
