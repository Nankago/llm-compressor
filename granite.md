# Granite GPTQ with a longer WikiText calibration

**The symmetric-search prototype still saves about half a second of observer time when calibration grows 64-fold.** On 64 × 1,024 WikiText tokens, the four paired observer differences were 0.555–0.589 s (median 0.580 s). This accounted for 1.04% of baseline OneShot wall time. The measured end-to-end saving was 1.072 s (1.91%) at the median of the four pairs. [Per-process timings, device inventories and output digests](https://github.com/Nankago/llm-compressor/blob/673b778e30a1076e8f06782848327c0923848636/benchmarks/gptq_moe/results/granite-calibration-load.json)

| WikiText workload | Observer, base → prototype (s) | OneShot, base → prototype (s) | Paired observer saving (s) | Paired OneShot saving (s) | Paired OneShot saving (%) | Peak allocated, base / prototype (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 8 × 128 | 5.253 → 4.703 | 17.421 → 16.714 | 0.543 [0.513–0.658] | 0.751 [0.357–0.940] | 4.29 [2.06–5.35] | 1,198.94 / 1,198.94 |
| 64 × 1,024 | 5.329 → 4.760 | 56.128 → 54.951 | 0.580 [0.555–0.589] | 1.072 [0.673–1.937] | 1.91 [1.20–3.44] | 1,204.64 / 1,204.64 |

Each arrow compares arm medians. The savings columns take the median and full range of the four within-round differences or percentages. First calibration forward time on the baseline grew from 4.209 to 30.685 s, and the error-propagation forward grew from 1.197 to 10.033 s. The baseline observer's share of wall time fell from 30.2% to 9.5%. The absolute observer saving was similar because both workloads searched the same 2,400 weight tensors. Relative to baseline wall time, the paired observer saving fell from 3.12% to 1.04%.

On the longer workload, the part of the wall-time saving outside the observer varied from 0.097 to 1.348 s across the pairs. The observer timer directly supports the 0.580 s contribution; the remaining difference includes variation in forward execution and other phases. A defensible upstream claim on this H100 is a roughly 0.6 s observer saving with a smaller relative effect as calibration grows. The 1.91% describes the OneShot A/B for this measured workload.

All 2,400 target modules matched byte for byte between arms in every process within each workload, including final weights, scales, zero points, packed INT4 codes and shapes. The larger token set produced a different packed model from the small control. Both workloads exercised batched Cholesky on 2,352 modules and per-item Cholesky on 48, with zero RTN fallbacks. Across 20 processes (four primes and 16 timed runs), GPU 0 on the same H100 had no foreign PID in 2,315 occupancy samples. The tensor digests and memory measurements are in the [results JSON](https://github.com/Nankago/llm-compressor/blob/673b778e30a1076e8f06782848327c0923848636/benchmarks/gptq_moe/results/granite-calibration-load.json).

## Workload and controls

We quantized pretrained Granite 3.1 1B-A400M base with 64 extracts of 1,024 tokens from 64 different WikiText-2 train articles. This covers 65,536 tokens of English encyclopedic prose. The matched small control uses the first 128 tokens of the first eight extracts (1,024 tokens). Each window lies within a single article. Both arms read exactly the same offline token array for each workload. The smaller control uses the new corpus and timing probes, so it can be compared directly with the longer calibration.

The dataset preparation script verifies the raw train shard, shuffles article indices with seed 42, and takes one uniformly positioned window from each selected article with at least 1,024 tokens. It hashes the complete little-endian int64 bank before writing it outside Git. Run metadata includes the dataset and checkpoint revisions, per-workload token SHA-256, source rows and offsets. [Pinned train split](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1) · [Pinned model](https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/tree/408b6e90baab8cf24f4aa9f8e19703ffa0a53b29)

The production code comparison is upstream main `f7d824cee12ff15bfa719ba8a7ea8ace7566df67` as present in fork commit `6b4b677`, versus isolated prototype `1b49722`. The fork was at `af7993e` when the experiment started. Both arms use the same benchmark harness, W4A16 symmetric group size 128, `imatrix_mse`, GPTQ auto batching, block size 128, damping fraction 0.01, eager attention, batch size one, all experts calibrated, error propagation and one sequential target per subgraph. Every run verifies and reloads the original two checkpoint shards in a new process.

Each workload gets one excluded full-workload prime per arm. The next four pairs alternate baseline/prototype order (AB, BA, AB, BA). Each arm reuses its own disk caches after priming. The default Triton driver discovery runs in every process. Device sampling uses the same H100 UUID throughout and checks for foreign processes during each run. No profiler is attached.

OneShot wall time starts immediately before `oneshot` and ends after its final device synchronization. The observer timer uses the same CUDA-event boundaries around GPTQ's weight `observe` call as the earlier investigation. The first forward timer includes the input hooks, Hessian accumulation and model forward. The second forward timer covers error propagation with hooks disabled. We disabled the earlier per-hook timing to keep the overhead from scaling with the number of calibration samples. Stream elapsed times include host submission gaps. Model loading, token preparation, official INT4 packing and output hashing occur outside the OneShot timer. Peak memory is PyTorch's maximum allocated memory during that window. [Previous 8 × 128 benchmark](https://github.com/Nankago/llm-compressor/blob/673b778e30a1076e8f06782848327c0923848636/benchmarks/gptq_moe/results/imatrix-investigation.md)

## Reproduce

Use the environment from the [benchmark README](https://github.com/Nankago/llm-compressor/blob/673b778e30a1076e8f06782848327c0923848636/benchmarks/gptq_moe/README.md) and run the following from the benchmark checkout. The dataset reader used PyArrow 25.0.1. Set `gpu` to the UUID of an idle H100. Keep the raw directory outside the repository.

```bash
bench="$PWD/benchmarks/gptq_moe"
raw=$(mktemp -d)
git worktree add --detach "$raw/base" 6b4b6778e6954b065ce4f999f3cad63cc6f396fc
cp src/llmcompressor/version.py "$raw/base/src/llmcompressor/version.py"

hf download ibm-granite/granite-3.1-1b-a400m-base --revision 408b6e90baab8cf24f4aa9f8e19703ffa0a53b29 --local-dir "$raw/model" --include '*.json' '*.safetensors'
shard=wikitext-2-raw-v1/train-00000-of-00001.parquet
hf download Salesforce/wikitext "$shard" --repo-type dataset --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir "$raw/dataset"

python "$bench/wikitext_calibration.py" --parquet "$raw/dataset/$shard" --model-dir "$raw/model" --out "$raw/tokens"
python "$bench/calibration_load_trials.py" --gpu "$gpu" --model-dir "$raw/model" --calibration-dir "$raw/tokens" --base-src "$raw/base/src" --proto-src "$PWD/src" --out "$raw/trials"
python "$bench/summarize_calibration_load.py" "$raw/trials" "$raw/summary.json"
```

## Qwen3-30B-A3B availability

An H100 was available. A name search of the user's workspace, public and shared model locations, local caches and temporary directories found scripts, tests and documentation for this model. The matching example directory contained two shell scripts. A separate search of checkpoint configs in the model stores and Hugging Face caches found no `qwen3_moe` checkpoint there. The available Granite benchmark therefore supplies the full-model evidence for this run. Qwen3-30B-A3B timing and its observer branch remain unmeasured.
