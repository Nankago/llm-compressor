# Idle H100 observer measurement

The three unprofiled idle-H100 runs had a median weight_observe of 3.311 s. Inside the separate weight_observe NVTX windows, kernels were active for 65.2 ms and synchronous CUDA API calls occupied 14.9 ms of host time. The nested validator's API residence was 8.0 ms.

| Unprofiled auto (s) | Median | Minimum | Maximum |
|---|---:|---:|---:|
| oneshot_s | 6.0546 | 6.0064 | 6.1735 |
| weight_observe_s | 3.3108 | 3.2955 | 3.3422 |

The earlier shared H100 weight-observation diagnostic measured 7.376 s for auto and 7.329 s for None. The dedicated trace uses separate inclusive host and GPU timelines. The validator and scale search are nested inside weight observation.

| Trace scope | Calls | Host ms | Kernel-active ms | Copy-active ms | Sync API host ms | GPU gap ms |
|---|---:|---:|---:|---:|---:|---:|
| GPTQ/weight_observe | 11 | 95861.73 | 65.16 | 4.27 | 14.85 | 95792.30 |
| GPTQ/_get_validated_importance | 280 | 141.97 | 5.02 | 2.55 | 7.96 | 134.40 |
| GPTQ/scale_search | 280 | 95664.43 | 58.79 | 1.71 | 6.90 | 95603.93 |

### First scale search in the profiled run

The first of 280 scale searches took **95.310 s**. The median across all 280 was **1.196 ms**. Inside the first range, the observer thread spent **93.922 s** in a single OS runtime `read()` (15,270,938,150 to 109,193,232,067 ns). The scheduler shows 93.922 s off CPU during that call. No traced process kernel, GPU copy or memset overlapped it. The host call chain includes `_Py_read` and `__read`. The file descriptor's source remains unresolved. [Reproduce the timeline query](../first_scale_search.sql)

The capture therefore localizes its largest stall to a blocking host read on first use. Its 95.862 s inclusive observer time is much longer than the 3.311 s unprofiled median. The capture cannot apportion those 3.311 s between initialization, scheduling and steady-state scale search. The most informative follow-up is to time the first scale search separately from the remaining 279, with lightweight timers in a fresh process and on a second pass in the same process.

### Kernels in the weight-observation host windows

- 40.151 ms across 280 window intersections, `_grid_search_observer_triton_packed_kernel`
- 3.300 ms across 1400 window intersections, `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)], std::array<char *, (unsigned long)2>, (int)4, TrivialOffsetCalculator<(int)1, unsigned int>, TrivialOffsetCalculator<(int)1, unsigned int>, at::native::memory::LoadWithCast<(int)1>, at::native::memory::StoreWithCast<(int)1>>(int, T1, T2, T4, T5, T6, T7)`
- 1.386 ms across 1400 window intersections, `void at::native::vectorized_elementwise_kernel<(int)8, at::native::FillFunctor<c10::BFloat16>, std::array<char *, (unsigned long)1>, (bool)0>(int, T2, T3)`
- 1.272 ms across 840 window intersections, `void at::native::vectorized_elementwise_kernel<(int)8, at::native::BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::maximum_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[lambda() (instance 4)]::operator ()() const::[lambda(c10::BFloat16, c10::BFloat16) (instance 1)]>, std::array<char *, (unsigned long)3>, (bool)0>(int, T2, T3)`
- 1.226 ms across 560 window intersections, `void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::and_kernel_cuda(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 12)]::operator ()() const::[lambda(bool, bool) (instance 1)]>, unsigned int, bool, (int)4, (int)4>>(T3)`

### Validator host waits

- `cudaStreamSynchronize_v3020` took 7.955 ms across 1120 API calls (inclusive host residence).
- `cudaMemcpyAsync_v3020` took 6.011 ms across 1120 API calls (inclusive host residence).

During validator synchronous APIs, 0.000 ms overlapped work submitted before that invocation, 2.523 ms overlapped work submitted by it, and 0.000 ms had ambiguous or missing launch correlation. Origins can overlap across streams. API residence may include active polling and CPU overhead.

The observer issued 14840 launch API calls. Its host thread was observed on CPU for 1361.965 ms during gaps in the recorded process GPU activity. Scheduling coverage available: True. Running-leaf samples in those gaps: 1411. Native dispatch, Python execution and profiler overhead can all contribute.

The longest recorded validator sync was enclosed by `aten::item` and `aten::_local_scalar_dense` and overlapped a Device-to-Host copy on stream 7 submitted by that invocation. Full timestamps, per-run device inventories and the local full-report path are in [idle-observer.json](idle-observer.json).

The [follow-up investigation](imatrix-investigation.md) separates the first call from steady state, identifies shared Triton initialization, and tests the repeated grid-search work on TinyMoE and Granite.
