# Reproduce on one idle H100

Run from this directory with the repository installed in an isolated environment. The tested server used `/tmp/moe-gptq-1121-venv/bin/python` and `PYTHONPATH` pointing to this worktree's `src`. Python, CUDA, driver and package versions are in [results.json](results.json). Model loading is local-only. Use a new output directory for every cache and benchmark run.

```bash
export PYTHONPATH="$(git rev-parse --show-toplevel)/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/tmp/moe-gptq-1121-venv/bin/python
MODEL=/tmp/imatrix-granite-408b6e9
WORK=/tmp/qda-reproduction
mkdir -p "$WORK"
REF=ibm-granite/granite-3.1-1b-a400m-base
REV=408b6e90baab8cf24f4aa9f8e19703ffa0a53b29
DATA=b08601e04326c79dfdd32d625aee71d232d685c3
SOURCE="https://huggingface.co/datasets/Salesforce/wikitext/resolve/$DATA/wikitext-2-raw-v1"
curl -fL "$SOURCE/test-00000-of-00001.parquet" -o "$WORK/test.parquet"
"$PY" inputs.py "$WORK/test.parquet" "$MODEL" "$WORK/eval.jsonl"
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest -v test_evaluate
```

On this server the verified training token bank is at `../llm-compressor-moe-calibration/tokens/tokens.npy` relative to the worktree root. To regenerate it elsewhere with the same tokenizer, run the pinned data preparation script.

```bash
curl -fL "$SOURCE/train-00000-of-00001.parquet" -o "$WORK/train.parquet"
curl -fL "https://raw.githubusercontent.com/Nankago/llm-compressor/c81852cbfc04fca46f5f85a9d88a60525efb4385/benchmarks/gptq_moe/wikitext_calibration.py" -o "$WORK/calibration.py"
"$PY" "$WORK/calibration.py" --parquet "$WORK/train.parquet" --model-dir "$MODEL" --out "$WORK/cal"
```

Select an idle GPU UUID before starting. The recorded trials used a separate NVML monitor that sampled every half second and stopped its own child if a foreign PID appeared. Candidate creation verifies the original weight shards and training-token slice against pinned SHA-256 digests. Skip candidate creation when reusing the two checkpoints listed in the report.

```bash
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
export CUDA_VISIBLE_DEVICES="${GPU_UUID:?set GPU_UUID to an idle physical GPU}"
"$PY" make_candidates.py "$MODEL" "$WORK/cal/tokens.npy" "$WORK/w4" W4A16
"$PY" make_candidates.py "$MODEL" "$WORK/cal/tokens.npy" "$WORK/w8" W8A16

common=(--inputs "$WORK/eval.jsonl" --reference-model "$REF" --reference-revision "$REV")
"$PY" evaluate.py prepare-reference --model "$MODEL" --cache "$WORK/ref" "${common[@]}"
for candidate in "$MODEL" "$WORK/w4" "$WORK/w8"; do
    "$PY" evaluate.py score --model "$candidate" --cache "$WORK/ref" "${common[@]}"
done
"$PY" benchmark.py "$MODEL" "$WORK/w4" "$WORK/w8" "$WORK/eval.jsonl" "$WORK/trials" --repeats 5
```

**Using other token sequences.** The UTF-8 JSONL file has one object per sequence. Supply a unique `id`, unpadded `input_ids`, and either `score_start` (inclusive target index, default 1) or strictly increasing `score_positions`. For example, `{"id":"a","input_ids":[11,22,33,44],"score_start":2}` scores the distributions at logit indices 1 and 2. For generated answers, retain the full prompt as context and set `score_start` to the index of the first answer token.

**Cache v1.** Each batch contains `ids` (int32), `logp` (FP64) and `tail_logmass` (FP64, natural log of the total complement mass, with zero encoded as negative infinity). The scorer normalizes each model over its entire vocabulary, gathers the candidate on the teacher's IDs and computes the complement by masked log-sum-exp. The manifest records the teacher ID, revision, file digests, tokenizer fingerprint, hashes of ordered inputs and target positions, batching, vocabulary and shard checksums. The model ID and revision are supplied by the caller for local checkpoints. Scoring verifies the expected identity, input and position hashes, candidate tokenizer, vocabulary, shard checksums and shapes. The candidate process needs the input file and cache; it never opens the teacher's weights.
