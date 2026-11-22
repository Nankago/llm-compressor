# MoE GPTQ module batching on one H100

On the pretrained TinyMoE-100M-2x8 checkpoint, the default GPTQ batcher reached batched Cholesky in all ten decoder layers. Each layer contributed one batch of **16** gate/up projections, one batch of **8** down projections, and two attention batches of **2**. Disabling batching increased median oneshot time from 13.91 to 16.00 seconds on this shared GPU.

| GPTQ mode | Module batch size × batches | Modules per-item / batched Cholesky | H accumulation (s) | H factorization (s) | Weight update (s) | OneShot wall (s) | Peak allocated (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| `auto` | 2 × 20, 8 × 10, 16 × 10 | 120 / 160 | 0.201 | 0.900 | 0.431 | 13.905 | 162.5 |
| `None` | 1 × 280 | 280 / 0 | 0.180 | 1.686 | 1.156 | 15.997 | 106.8 |

Values are per-column medians of three fresh-process runs per mode. Wall times were 13.851–13.924 s for auto and 15.930–16.004 s for None. Peak reserved memory was 210 and 132 MiB respectively. Including local checkpoint verification, loading, linearization and input preparation gave median setup-plus-oneshot times of 15.743 and 17.732 s. Downloads, interpreter startup, imports and artifact export are outside both measurements.

## Batching and follow-up

The pinned source defaults to `batched_quantization="auto"`. Within a sequential subgraph, the compatibility key includes the two-dimensional weight shape, dtype, execution device and serialized weight quantization arguments. Auto limits each compatible list to 75% of free device bytes divided by the estimated per-module peak. The observed limits ranged from 535 to 2,203 across the formal runs. The largest compatible list had 16 members. Raising the cap would produce the same batches in this setup. [Batch assignment](../../src/llmcompressor/modifiers/gptq/helpers.py)

Gate/up weights were [768, 384], with a stacked H of [16, 384, 384]. Their first Cholesky received that 3D tensor. Down weights were [384, 768], with H [8, 768, 768]. Their first Cholesky received eight separate [768, 768] matrices. Q/O and K/V each formed a same-shape pair. The threshold is inclusive at 16. The harness checks the rank and count of actual `torch.linalg.cholesky_ex` calls for every batch and checks coverage of all 280 targets. RTN fallback count was zero throughout. Formal auto used 150 fused GPTQ block-update calls per run and None used 1,080.

**Investigate the weight observer before changing the Cholesky threshold.** The comparison changed batch preparation, weight updates, writeback and decomposition together. Auto saved 2.09 s of median wall time and about 0.79 s in factorization. Roughly 10.9 s per run remained outside the instrumented CUDA-event intervals. In separate eight-sample event diagnostics, the upstream weight-observation call took **7.376 s with auto and 7.329 s with None**. These observations precede GPTQ batch preparation. In another eight-sample auto run, Python cProfile attributed 7.58 s inclusive to weight observation and 2.67 s inclusive to `_get_validated_importance` over 280 calls. The latter is inside the observer measurement. The validator performs device reductions followed by host decisions. A follow-up on an idle GPU with a warm process, CUDA API tracing and a larger checkpoint could distinguish synchronization costs from work already queued by the observer. The potential speedup requires a controlled test of that hypothesis. [Weight observation](../../src/llmcompressor/modifiers/quantization/calibration.py) · [Importance validation](../../src/llmcompressor/observers/imatrix.py)

## Workload

- Upstream main at `f7d824cee12ff15bfa719ba8a7ea8ace7566df67`. Only benchmark files were added. [Pinned source](https://github.com/vllm-project/llm-compressor/tree/f7d824cee12ff15bfa719ba8a7ea8ace7566df67)
- Full pretrained Mixtral-family model, 10 layers, 8 experts per layer, top-2 router. The publisher distributes FP32 weights. Every process verifies SHA-256 `feb1cc5a5140f3ec7c357ecb9b13f54b17575d76c792459f563ce061a1527c6a`, loads BF16 on the CPU, checks the load report, and calls the upstream `linearize_moe`. The harness uses all experts during calibration, eager attention and one decoder target per sequential subgraph, with quantization-error propagation enabled. [Checkpoint and training details](https://huggingface.co/FlameF0X/TinyMoE-100m-2x8-retrained/tree/0c1d097c964b949bc0092fa8e99c88c325421167)
- `GPTQModifier` with W4A16, symmetric INT4, group size 128, static activation ordering, `imatrix_mse` weight observer, block size 128 and damping 0.01. Targets include attention and all expert linears. Router and LM head are excluded. Each run quantizes 240 expert and 40 attention projections.
- Calibration uses eight disjoint 128-token windows of the upstream English README at the fixed SHA. Tokenization uses the checkpoint tokenizer without added special tokens. Python seed 42 shuffles the window offsets once. A fixed dataloader yields them in order, batch size 1. The little-endian int64 input digest is `b3d5839fee2d2adc20c570fe826ffd556a8f4a37c5b81fe0c9720681d4b400a0`. This is a timing workload. Model quality and sparse inference throughput were outside the experiment.
- NVIDIA H100 80GB HBM3, physical GPU 2, UUID `GPU-824ec5b7-85a1-18e4-4b08-d8f41f920b49`, driver 580.126.20. The existing vLLM workload was active and GPU utilization at both endpoints of all six runs was 100%. The benchmark used a process-local PyTorch allocator limit of 4.5% of total memory. The peaks in the table cover this process's PyTorch allocations during oneshot. CUDA context, driver allocations and other processes are outside those peaks.
- NVIDIA PyTorch environment 26.08 (build 406036884), Python 3.12.3, torch runtime `2.14.0a0+4fdf77b940.nv26.08`, CUDA 13.4, toolkit `nvcc V13.4.59`, cuDNN 92500, TF32 matmul enabled. [Runtime dependencies](results/environment.json)

## Timing boundaries

`instrument.py` wraps the pinned Python entry points at runtime. A CUDA event on the current stream closes one active interval and opens the next. Events are read after the final synchronization. The probes add no hook-level synchronization. Event spans measure elapsed stream time, including host submission gaps and scheduling delays. Upstream syncs, such as `info.item()`, remain in place.

- `accumulate` covers the upstream FP32 input conversion, sample count and Gram matrix accumulation. `hook_setup` covers the remainder of the GPTQ calibration hook, including first-use H allocation.
- `calib_forward_outside_gptq_hook` is the first subgraph forward excluding the whole GPTQ hook. It includes other observer hooks in the primary matrix. The separate `--observer-details` pair also isolates the imatrix input hook (0.364/0.343 s for auto/None, 2,240 calls apiece). The first-pass forward excluding both GPTQ and imatrix collection measured 0.970/1.032 s in that pair. `propagate_forward` covers the second pass with modifier hooks disabled.
- `prepare_batch` covers H normalization, stacking, and preparation of weights and qparams. `quant_setup` covers entry into `quantize_weight` through activation ordering and allocations before factorization.
- `factor` covers dead columns, damping, first Cholesky, inverse, final upper Cholesky and fallback handling. `weight_update` spans the end of factorization through return from `quantize_weight`, including block updates, cross-block error propagation, permutation restoration and final conversions. `writeback` covers `update_batch_qparams`.
- `oneshot_wall_s` runs from immediately before oneshot through device synchronization at return. `unattributed_wall_s` is this time less the disjoint event intervals. It includes tracing, observer work outside subgraph forwards, staging, inter-subgraph transfers, lifecycle bookkeeping, logging and instrumentation. Copies within a named interval are charged to that interval. Profiling totals are a separate diagnostic and include nested calls.

## Reproduce

Use a checkout of this benchmark branch with its parent commit accessible to Git. Start in the repository root, in a compatible NVIDIA PyTorch 26.08 environment. Keep the venv and Triton cache private to the run. The published NVIDIA torch and Triton build strings are in the environment record.

```bash
bench=benchmarks/gptq_moe
work=$(mktemp -d)
python3 -m venv --system-site-packages "$work/venv"
source "$work/venv/bin/activate"
python -m pip install -c "$bench/constraints.txt" setuptools-scm==8.2.0
python -m pip install --no-build-isolation -c "$bench/constraints.txt" -e .
export TRITON_CACHE_DIR="$work/triton"
export CUDA_VISIBLE_DEVICES=GPU-824ec5b7-85a1-18e4-4b08-d8f41f920b49
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

python "$bench/run.py" --mode auto --samples 1 --out "$work/raw/preflight-auto.json"
bash "$bench/matrix.sh" "$work/raw"
python "$bench/run.py" --mode auto --samples 8 --host-profile --out "$work/raw/profile-auto.json"
for mode in auto none; do
    python "$bench/run.py" --mode "$mode" --samples 8 --observer-details --out "$work/raw/observer-$mode.json"
done
python "$bench/summarize.py" "$work/raw" "$work/results"
```

On another host, choose one visible GPU. The model download is pinned. For an offline run, pass the local snapshot with `--model-dir` to `run.py` and as the second positional argument to `matrix.sh`. On the benchmark server all runs used this offline form. The matrix runs one single-sample warmup per mode, then six measurements in auto, None, None, auto, auto, None order. Each starts a fresh process and reloads the original checkpoint. Processes share the on-disk Triton cache. The initial path check, both warmups, the host-profile run and the observer diagnostic pair are recorded separately from the three formal repetitions per mode.

## Result files

- [summary.csv](results/summary.csv) contains the median for every disjoint stage in seconds, wall time and process peak memory.
- [runs.json](results/runs.json) contains the raw per-run values, token offsets, batch caps, endpoint GPU readings, fallback lists, call counts, and the diagnostic profile. Each record points to its exact batch layout. Identical layouts are stored once.
- [Auto batch records](results/batches-auto-5a3e325735cb.jsonl) and [None batch records](results/batches-none-2ae4e9bcc245.jsonl) give ordered module names, projection types, individual weight/H shapes, stacked H shape, subgraph, actual branch and number of first-Cholesky calls.

## Idle-H100 observer follow-up

Two preflight checks on this server found an existing vLLM process on every H100. Each process held 73,612 MiB. The six samples and the process inventory are recorded in [idle-observer-preflight.json](results/idle-observer-preflight.json). The baseline commit was `feea7730452f676ff159c56faf831a8bf7b5a959`. The 7.376 s shared-GPU observation and 2.67 s inclusive validation profile above remain the available comparison measurements at preflight.

From the same environment as the original benchmark, run the following once. The output directory must be new. Omit the local-model option to download the pinned checkpoint.

```bash
python benchmarks/gptq_moe/idle_observer.py --wait-hourly \
    --out /path/outside/git/observer-run --model-dir /path/to/pinned/snapshot
```

The runner samples availability once per hour while all devices are occupied. For an apparently idle H100 it confirms zero utilization and an empty compute and graphics process list twice more, two seconds apart. It fixes that UUID, records per-run before/after GPU inventories, and runs one 8×128 warmup followed by three unprofiled auto repetitions and one dedicated Nsight Systems capture. Every run reloads the checkpoint in a fresh process using the existing `--observer-details` timing boundaries. All five processes share one private on-disk Triton cache. Raw run JSON, summary medians and ranges, logs, the full `observer-trace.nsys-rep` and its SQLite export stay in the output directory.

The trace-only entry point wraps the imported GPTQ weight-observation function, the imatrix importance validator and the scale-search dispatch invoked by that observer. A CPU-only Nsight smoke on an actual imatrix observer verified all three range labels and the nesting of the validator and scale-search calls under weight observation. CUDA runtime/driver calls, kernels, copies, ATen NVTX, OS runtime, thread scheduling, GIL events and sampled Python stacks are requested for the separate GPU capture.

To interpret a wait, select the host thread inside `GPTQ/_get_validated_importance`, locate the scalar-read ATen range or synchronous API, and inspect kernels and DtoH copies overlapping the API interval. Follow each activity's process and correlation ID back to its launch, then check the launch timestamp against that invocation's validator range. An earlier launch can execute inside the validator's wall interval. Count kernel time as the union of clipped device spans, and report host waits separately. Use on-CPU samples and scheduling events to assess GPU gaps between launches. A gap alone does not identify Python overhead.

Once a complete capture is available, generate the compact result with:

```bash
python benchmarks/gptq_moe/record_idle_results.py /path/outside/git/observer-run \
    benchmarks/gptq_moe/results/idle-observer.json
```

The summary checks the model, calibration digest, batch histogram, probe calls and GPU endpoint inventories against the workload. It reports distinct host ranges for weight observation, validation and scale search. GPU active times use unions over streams within each host window. Per-kernel rankings count overlapping execution intervals and sum their clipped durations. Explicit synchronizations and synchronous copies form the host API residence metric. Async copy API durations are listed independently. The examples carry API and validator timestamps, enclosing scalar-read ATen labels, GPU stream and launch origin. Scheduled-out intervals during an API are evidence of host descheduling; a GPU dependency may also involve polling on CPU. The sampled CPU leaf symbols in GPU gaps can distinguish interpreter frames from native dispatch when enough samples are captured. Missing scheduling data or samples limits that attribution. The scalar validator is nested inside weight observation, and GPU/API overlap is reported on both timelines.

The [first-use and symmetric grid-search investigation](results/imatrix-investigation.md) compares cache lifetimes and validates a single observer change on two pretrained MoE checkpoints.
