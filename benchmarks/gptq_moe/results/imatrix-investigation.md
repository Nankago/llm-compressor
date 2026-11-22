# First-use cost and the symmetric iMatrix grid search

**The symmetric Triton grid-search change is ready for an upstream PR.** On pretrained Granite 3.1 1B-A400M, skipping unused base qparams reduced fresh-process OneShot from 19.563 to 18.843 s (3.7%) and the observer from 5.249 to 4.742 s (9.7%). The gain persisted on a second fresh model in the same process. All 2,400 packed weights and scales matched bit for bit. The change is isolated in [prototype commit 1b49722](https://github.com/Nankago/llm-compressor/commit/1b49722f38febe9f1fbb195f3bc14b354fbf8172). [Individual runs, environment and trace summaries](imatrix-investigation.json)

## Workload and measurement

The published branch started at `6b4b6778e6954b065ce4f999f3cad63cc6f396fc`. Upstream main still pointed to `f7d824cee12ff15bfa719ba8a7ea8ace7566df67`. Both source trees ran under the same NVIDIA H100 80GB HBM3 environment. The previous shared card gave 7.38 s for the auto observer. Its idle-card follow-up gave 3.31 s for the observer and 6.05 s for OneShot. [Previous report and full trace](idle-observer-report.md)

Every pass loads a verified pretrained checkpoint, resets the compression session and seed, uses the pinned README corpus and quantizes with W4A16 GPTQ, symmetric group size 128, `imatrix_mse`, auto batching and one sequential target per subgraph. Each calibration has eight fixed 128-token sequences, batch size one, all experts calibrated, and error propagation enabled. The model's tokenizer determines the token IDs. The original event boundaries, allocator limit and synchronization at OneShot return are retained. Peak memory is the maximum allocated by PyTorch during OneShot. Loading, output fingerprints and disk serialization are outside the wall timer.

### What costs three seconds?

| Process and kernel cache | Observer, original probes (s) | OneShot (s) | First search (ms) | Remaining 279 combined (ms) |
|---|---:|---:|---:|---:|
| New process, empty private caches | 4.111 [4.086–4.159] | 7.218 [7.119–7.261] | 3,874 [3,856–3,910] | 179.8 [179.4–188.3] |
| New process, reused disk caches | 3.296 [3.272–3.321] | 6.078 [6.050–6.100] | 3,104 [3,053–3,136] | 178.5 [177.2–179.0] |
| Same process, second fresh session | 0.236 [0.227–0.249] | 2.087 [2.057–2.107] | 0.84 [0.83–0.87] | 178.3 [178.2–180.3] |

Cells show the median and full range of three independent repetitions. Each row's first two columns come from the original benchmark. The final two come from a separate instrumented series. Additional CUDA events and host clocks surround each call to the actual scale search and are consumed after the existing synchronization. An event span includes device idle time while the host prepares the next launch. Host and stream intervals overlap. With the additional probes, observer medians were 4.163, 3.389 and 0.244 s respectively. The corresponding wall medians were 7.263, 6.165 and 2.067 s. The separate-run differences also include run variation. All 18 passes peaked at 162.49 MiB allocated.

Each cold repetition has new Triton, driver, extension and Inductor cache paths. Each disk-warm process reuses that cold process's directories. The second pass keeps CUDA libraries, the Triton driver, in-memory JIT callables, and ordinary Python module state in the same process. It explicitly clears the PyTorch allocator's free blocks. System page caches and checkpoint files are shared across all processes. Every pass rebuilds the calibration loader from the same specified tokens and reloads original weights.

A separate first-call cProfile run with warm disk caches spent 2.185 s inclusive in Triton's NVIDIA `libcuda_dirs()`, which on this image calls `subprocess.check_output(["/sbin/ldconfig", "-p"])`. Three standalone executions of that exact command produced 73,135 bytes and took 2.234, 2.223 and 2.241 s. A separate 0.776 s in `triton_key()` included 0.622 s of hashing. This Triton version reads and hashes its native library when constructing the per-process cache key. With cold caches, building the driver helper also took 0.547 s inclusive. These intervals are nested within the first search. The normal runs did not reproduce the 93.922 s host stall, and the old trace's file descriptor remains unidentified. Cache-key generation and library discovery are shared Triton startup costs.

A process-local discovery control ran the unmodified baseline in lookup, explicit, explicit, lookup order. Supplying `TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib` selects the first directory returned by this host's linker cache. First search took **3.106, 1.419, 0.894, 3.107 s**, respectively. OneShot took **6.177, 4.488, 3.979, 6.098 s**. The two explicit runs reused the same private directory, but changing the library-directory list also changes Triton's compiled driver-helper cache key. A second `cuda_utils` artifact exists under that root. The 0.894 s result is the warm explicit-path observation. This override was absent from all prototype comparisons.

The later 279 baseline calls spent 55–56 ms of host time submitting the base `calculate_qparams` operation, within about 179 ms of elapsed search time. The Python wrapper discarded its scale and supplied the zero point to a kernel with compile-time `HAS_ZP=False`. The prototype skips that calculation for symmetric Triton search and guards both packed and split zero-point loads. The asymmetric branch continues to derive its base zero point and the eager dispatch continues to compute candidate qparams.

### Warm CUDA timeline

Two Nsight Systems captures bracketed TinyMoE scale searches 29–56. The table sums the 28 disjoint NVTX host windows. GPU active time is the union of kernels, copies and memsets intersecting those windows. Scheduler events identify when the submitting thread was on CPU during a traced-process GPU gap. All columns can overlap across the host and device timelines.

| 28 invocations | Host inclusive (ms) | GPU active (ms) | Host on CPU in GPU gaps (ms) | Launch APIs | Host sync API residence (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline scale search | 24.842 | 5.673 | 19.169 | 1,092 | 0.717 |
| ↳ Baseline base-qparams range, nested | 8.925 | 0.717 | 8.208 | 476 | 0.158 |
| Prototype scale search | 15.444 | 4.758 | 8.436 | 588 | 0.563 |

The host submitted **39 → 21 launches per search**. Seventeen of the removed launches were inside each base-qparams range; the source also removes its FP32 zero-point cast. The packed search kernel still ran 28 times, taking 3.965 → 3.868 ms clipped to the windows. Baseline qparam kernels included 84 BF16 fills (0.083 ms), 56 maxima (0.084 ms) and 56 absolute-value kernels (0.069 ms). Within qparams, the 28 explicit synchronizations overlapped 0.020 ms of H→D copies from the current invocation. No earlier launch overlapped these waits. Within the full search windows, the corresponding overlaps were 0.171 ms for baseline and 0.152 ms for prototype. CPU samples in baseline GPU gaps included ATen TensorIterator and device-guard code. The trace supports a reduction in host submission work and small GPU operations. Its samples cannot give an exclusive Python-interpreter time.

Both profiled processes had much longer full OneShot times (109.7 and 108.4 s); the CUDA captures cover only the warmed window. Full reports and SQLite exports are on the server under `../llm-compressor-moe-benchmark-imatrix-investigation/extras/warm-{base,proto}.{nsys-rep,sqlite}`, relative to the checkout. The prior 94 s read is in `../llm-compressor-moe-benchmark-observer-artifacts/hourly-run/observer-trace.nsys-rep`. The JSON includes the absolute paths for both new captures.

### End-to-end A/B on pretrained models

Each implementation received one excluded cold-disk prime per model (r0). Four independent process pairs followed, in AB, BA, AB, BA order. Each process ran twice. Every timed observation used the original event probes without a profiler or per-search markers. Ranges are min–max across four processes. The paired saving is baseline wall time minus prototype wall time within the same round.

| Model and process state | Observer base → prototype (s) | OneShot base → prototype (s) | Paired OneShot saving, median [range] (s) | Peak allocated, both (MiB) |
|---|---:|---:|---:|---:|
| TinyMoE, fresh, warm disk | 3.417 [3.395, 3.449] → 3.264 [3.224, 3.446] | 6.297 [6.242, 6.379] → 6.141 [5.987, 6.289] | 0.164 [0.034, 0.298] | 162.49 |
| TinyMoE, second pass | 0.229 [0.226, 0.230] → 0.172 [0.166, 0.172] | 2.061 [2.051, 2.105] → 2.020 [1.974, 2.079] | 0.058 [-0.021, 0.090] | 162.49 |
| Granite, fresh, warm disk | 5.249 [5.211, 5.302] → 4.742 [4.700, 4.768] | 19.563 [19.272, 19.630] → 18.843 [18.673, 18.922] | 0.683 [0.423, 0.956] | 1198.94 |
| Granite, second pass | 2.158 [2.119, 2.166] → 1.610 [1.607, 1.624] | 15.397 [15.049, 15.505] → 14.637 [14.566, 14.689] | 0.730 [0.483, 0.875] | 1198.94 |

The validation checkpoint is the pretrained IBM Granite 3.1 1B-A400M base, with 24 decoder layers, 32 experts per layer and top-8 routing. The workload actually quantized 2,400 linear modules, including 2,304 expert projections. Expert weights have shapes 512 × 1,024 and 1,024 × 512. Batch sizes 2, 66 and 32 each occurred 24 times. A total of 2,352 modules used batched Cholesky. TinyMoE covered 280 modules. Neither model took the RTN fallback. Granite and TinyMoE each used a constant token digest across arms; full digests and exact model revisions are in the [per-process results](imatrix-investigation.json).

On Granite the fresh-process OneShot ranges for the two arms are disjoint. Every pair saved at least 0.423 s, larger than the baseline's 0.358 s run range. TinyMoE's smaller steady-state saving is clear in the observer itself, but one paired second pass took 0.021 s longer end to end. Startup variability also reversed one of the four fresh-process TinyMoE observer pairs. Granite provides the stronger end-to-end evidence.

All cache and A/B trials ran serially on GPU 0, UUID `GPU-25dc68ae-c0e6-49b1-23b5-c4da9d63e8ef`. Across 32 processes including primes, NVML recorded 2,086 samples with no foreign PID; all pre-run utilization values were zero, and all before/after memory readings were 480 MiB. The JSON contains individual process inventories, timestamps, memory peaks and every timed pass.

### Correctness and scope

Every A/B pass matched the baseline SHA-256 digests for all updated BF16 weights, scales and zero points. Separate post-timing verification runs also used the official `PackedQuantizationCompressor.compress` on every target module. Packed INT4 codes, scales and shapes were byte-identical across the 280 TinyMoE and 2,400 Granite modules. The hashes cover sorted module names, dtype, shape and tensor bytes.

All 16 explicit edge cases matched byte for byte, covering packed and split reductions, symmetric and asymmetric ranges, weighted and unweighted errors, and empty and nonempty candidate grids on data with zero and constant groups. The existing iMatrix and MSE test files reported **105 passed, 57 skipped** on this NVIDIA host, including importance fallback and eager parity coverage. No XPU measurement was made. The performance claim is limited to this H100, these pretrained BF16 checkpoints, and the 8 × 128 W4A16 task. The practical next step is to submit the isolated prototype with the warm trace and Granite paired measurements.

### Experimental decisions

| Hypothesis | Changed variable and observation | Decision |
|---|---|---|
| The prior 3.31 s primarily reflects first use. | Isolating disk and process reuse reduced the TinyMoE first call from 3.104 s to 0.84 ms, with the other calls near 179 ms throughout. | Profile first use separately. |
| Driver discovery accounts for much of the warm-disk penalty. | Naming this server's driver directory reduced the warm first call to 0.894 s; the ordinary lookup controls took 3.106–3.107 s. | Leave Triton startup handling to its own layer and inspect repeated observer work. |
| Base symmetry handling submits work whose result the kernel ignores. | Instrumented qparam preparation occupied about 55 ms over 279 later calls; the warmed A/B CUDA window eliminated 18 launches per call. | Test one guarded calculation on the full workloads. |
| The per-call saving matters beyond TinyMoE. | On Granite all four fresh and all four second-pass pairs completed sooner with identical allocated peaks. | Check numerical identity and isolate the prototype for upstream review. |

## Overlap with upstream

The merged shared Triton implementation introduced the relevant packed and split paths. Later commits changed patience handling and unified the error numerics. The tested main still computes base qparams unconditionally. The prototype touches the host preparation and the kernel argument used by these paths. [Shared kernels #3233](https://github.com/vllm-project/llm-compressor/pull/3233) · [Observer defaults #3454](https://github.com/vllm-project/llm-compressor/pull/3454) · [Error numerics #3489](https://github.com/vllm-project/llm-compressor/pull/3489)

A public issue/PR search on 2026-11-22 covered “zero point” and “symmetric” with “grid search”, “calculate_qparams” with “observer”, Triton observer titles, and open cache work. Related open changes address CUDA indexing, eager reduction precision, a new MXFP4 observer and microscaled defaults. Their published diffs contained no matching symmetric base-qparam bypass. The dependency already memoizes the range calculation. [CUDA indexing #2997](https://github.com/vllm-project/llm-compressor/pull/2997) · [Eager reduction #3326](https://github.com/vllm-project/llm-compressor/pull/3326) · [Ceil_x #3492](https://github.com/vllm-project/llm-compressor/pull/3492) · [Microscale defaults #3554](https://github.com/vllm-project/llm-compressor/pull/3554) · [Range memoization](https://github.com/vllm-project/compressed-tensors/pull/855)

## Reproduce

Start in the fork checkout with the benchmark environment from the [README](../README.md). Run GPU commands serially. `gpu` is the UUID of a quiescent H100. `raw` is a new directory outside Git. `python` below is the benchmark environment interpreter. The base worktree needs the generated package version file from that installation.

```bash
bench="$PWD/benchmarks/gptq_moe"
raw=$(mktemp -d)
git worktree add --detach "$raw/base" 6b4b6778e6954b065ce4f999f3cad63cc6f396fc
cp src/llmcompressor/version.py "$raw/base/src/llmcompressor/version.py"
base="$raw/base/src"
proto="$PWD/src"
tiny="$raw/tiny-model"
granite="$raw/granite-model"
hf download FlameF0X/TinyMoE-100m-2x8-retrained --revision 0c1d097c964b949bc0092fa8e99c88c325421167 --local-dir "$tiny" --include '*.json' '*.safetensors'
hf download ibm-granite/granite-3.1-1b-a400m-base --revision 408b6e90baab8cf24f4aa9f8e19703ffa0a53b29 --local-dir "$granite" --include '*.json' '*.safetensors'
unset TRITON_LIBCUDA_PATH
export CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
PYTHONPATH="$base" python "$bench/observer_trials.py" --gpu "$gpu" --model-dir "$tiny" --out "$raw/matrix"
for model in tinymoe granite; do
    if [ "$model" = tinymoe ]; then dir="$tiny"; series=tiny-ab; else dir="$granite"; series=granite-ab; fi
    python "$bench/compare_observer.py" --baseline-src "$base" --prototype-src "$proto" --checkpoint "$model" --model-dir "$dir" --gpu "$gpu" --out "$raw/$series" --repeats 4
done
```

The diagnostic controller runs the first-call profile, two short Nsight captures (searches 29–56), the independent library-discovery control, the observer test suites and bitwise edge checks. The driver directory below was the first match in this machine's linker cache. Set it to the corresponding directory on your system. Diagnostic boundary synchronizations are outside every formal timing trial.

```bash
python "$bench/observer_diagnostics.py" --raw "$raw" --baseline-src "$base" --prototype-src "$proto" --model-dir "$tiny" --granite-dir "$granite" --gpu "$gpu" --libcuda-dir /usr/local/cuda/compat/lib
python "$bench/record_investigation.py" "$raw" "$raw/imatrix-investigation.json"
```
