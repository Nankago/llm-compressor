#!/usr/bin/env bash
# Run from any working directory. See README for environment versions and revisions.
set -euo pipefail
study=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$study/../.." && pwd)
: "${QPY:?quantization Python}" "${VPY:?serving Python}" "${QDA:?fixed QDA checkout}"
: "${M06:?Qwen3-0.6B}" "${M4:?Qwen3-4B}" "${M35:?Qwen3.5-0.8B-Base}"
: "${TRAIN:?WikiText train parquet}" "${TEST:?WikiText test parquet}" "${ARC:?ARC-Easy test parquet}"
: "${TINY:?TinyMoE checkpoint}" "${BENCH:?forward-ablation checkout}"
: "${GPU:?idle Hopper UUID for all W4AFP8 trials}" "${MOE_GPU:?idle Hopper UUID for MoE}"
: "${OUT:?new artifact directory}"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PYTHONPATH="$repo/src:$QDA/packages/qda-quant/src:$QDA/packages/qda-kl/src:$study${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=""
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
export TRITON_CACHE_DIR="$OUT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor"
export CUDA_CACHE_PATH="$OUT/cache/cuda" VLLM_CACHE_ROOT="$OUT/cache/vllm"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$GPU"; then
    echo "Selected W4AFP8 device is in use" >&2
    exit 1
fi
test "$(git -C "$QDA" rev-parse HEAD)" = dc39a02bfd87bb2790e34001708d54ce9131e076
test "$(git -C "$BENCH" rev-parse HEAD)" = 44cad4b8067d9de80a26c7a6399080d0a4601b4f
git -C "$repo" diff --quiet 18b10e2d7269e332c05eef4dda52adf74c08872e -- src
"$QPY" - "$M06" "$M4" "$M35" "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
small, medium, hybrid, out = map(Path,sys.argv[1:])
def check(folder,revision,files,manifest=None):
    for name,expected in files.items():
        with (folder/name).open("rb") as f:
            assert hashlib.file_digest(f,"sha256").hexdigest()==expected, name
    if manifest:
        (out/manifest).write_text(json.dumps({"sha":revision,"files":files}))
check(small,"c1899de289a04d12100db370d81485cdf75e47ca",{
    "model.safetensors":"f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"})
check(medium,"1cfa9a7208912126459214e8b04321603b3df60c",{
    "model-00001-of-00003.safetensors":"328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223",
    "model-00002-of-00003.safetensors":"6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5",
    "model-00003-of-00003.safetensors":"e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1"},"model4.json")
check(hybrid,"dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",{
    "model.safetensors-00001-of-00001.safetensors":"c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"},"model35.json")
PY
"$QPY" "$study/fixture.py" "$OUT/fixture.safetensors"
"$VPY" "$study/kernel_probe.py" "$OUT/fixture.safetensors" > "$OUT/probe-serve.log"
CUDA_VISIBLE_DEVICES="" "$QPY" "$study/mapping_probe.py" "$M35" > "$OUT/mapping-control.log"
"$QPY" "$study/tokens.py" "$M06" "$TEST" "$OUT/tokens.json"
"$QPY" "$study/tokens.py" "$M4" "$TEST" "$OUT/tokens4.json"
cmp "$OUT/tokens.json" "$OUT/tokens4.json"
"$QPY" "$study/build.py" "$M06" "$OUT/packed" --packed > "$OUT/packed.log"
for arm in scale product; do
    "$QPY" "$study/recode.py" "$M06" "$OUT/packed" "$OUT/final-$arm" --mode "$arm" > "$OUT/final-$arm.log"
done
for arm in main control aware; do
    "$QPY" "$study/gptq_ab.py" "$arm" "$M06" "$TRAIN" "$OUT/gptq-$arm" > "$OUT/gptq-$arm.log"
done
for arm in main aware; do
    "$QPY" "$study/gptq_ab.py" "$arm" "$M4" "$TRAIN" "$OUT/four-$arm" > "$OUT/four-$arm.log"
done
for arm in main aware; do
    "$QPY" "$study/gptq_ab.py" "$arm" "$M06" "$TRAIN" "$OUT/seed42-$arm" --calibration-seed 42 > "$OUT/seed42-$arm.log"
done
for arm in ref packed final-scale final-product gptq-main gptq-aware four-ref four-main four-aware seed42-main seed42-aware; do
    model="$OUT/$arm"
    base="$M06"
    tokens="$OUT/tokens.json"
    ref="$OUT/ref06"
    args=()
    if [[ $arm == four-* ]]; then base="$M4"; tokens="$OUT/tokens4.json"; ref="$OUT/ref4"; fi
    if [[ $arm == ref || $arm == four-ref ]]; then model="$base"; args=(--make-reference); fi
    "$VPY" "$study/all_tokens.py" "$model" "$tokens" "$ref" "$OUT/$arm-full.json" "${args[@]}"
    if [[ $arm != four-* && $arm != seed42-* ]]; then
        "$VPY" "$study/evaluate.py" "$model" "$tokens" "$OUT/anchor" "$OUT/$arm-score.json" "${args[@]}"
    fi
    if [[ $arm != four-ref ]]; then
        "$VPY" "$study/task_accuracy.py" "$model" "$base" "$ARC" "$OUT/$arm-arc.json"
    fi
done
# Use the fork's unmodified timing boundaries and NVML guard with the study's production source.
PYTHONPATH="$BENCH/benchmarks/gptq_moe:$PYTHONPATH" "$QPY" - "$repo" "$BENCH" "$MOE_GPU" "$TINY" "$OUT" <<'PY'
import sys
from pathlib import Path
from observer_trials import cache_env, launch, nvml
repo, bench, gpu, model, out = sys.argv[1:]
out=Path(out)
env=cache_env(out/"tiny-cache",Path(repo)/"src",gpu)
nvml.nvmlInit()
for trial in ("warm","r1","r2"):
    prefix=out/f"tiny-{trial}"
    launch([sys.executable,str(Path(bench)/"benchmarks/gptq_moe/run.py"),"--mode","auto",
            "--checkpoint","tinymoe","--model-dir",model,"--samples","8","--seq-length","128",
            "--coarse-timing","--out",str(prefix)+".json"],env,nvml.nvmlDeviceGetHandleByUUID(gpu),prefix)
PY
"$QPY" "$study/summarize.py" "$OUT" "$OUT/results.json"
