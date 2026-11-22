# Cached top-k KL on a quantized Granite MoE

On one idle H100, caching the BF16 reference paid for itself on the **second candidate**. Across 3,732 held-out next-token positions, the cached score ranked GPTQ W8A16 closer to BF16 than W4A16. On a 501-position exact check, the default `k=128` understated both KL values by about **14%**. The magnitude of this bias matters when comparing nearby recipes.

## Scores

All scores are token-weighted means of KL(teacher ‖ candidate) in nats. Both models see identical unpadded token sequences, with the same right padding and attention mask inside each batch. At target index `t` the evaluator uses logits at `t−1`. It preserves the teacher's top-k identities and pools all other probabilities into one bucket on each side. The loss of information from pooling gives a lower bound on full-vocabulary KL. [QDA blog §2.1](https://mit-han-lab.github.io/qda-blog/#s2a) · [QDA paper §4.2, equation 6 and §5.2](https://arxiv.org/html/2611.23388v1#S4.SS2)

| Candidate | k=128, all 3,732 | k=128, first 501 | Exact, same 501 | Absolute gap | k=512, same 501 | k=2,048, same 501 |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 0 | 0 | 0 | 0 | 0 | 0 |
| GPTQ W4A16 | 0.218343 | 0.181450 | 0.210840 | 0.029390 | 0.197340 | 0.206609 |
| GPTQ W8A16 | 0.005052 | 0.004382 | 0.005114 | 0.000732 | 0.004770 | 0.004995 |

The exact subset comprises the first four fixed sequences. At k=128, mean teacher tail mass was **7.73%**. Relative underestimation was 13.94% for W4 and 14.32% for W8. At k=2,048 it fell to 2.01% and 2.33%. Every checked position satisfied the lower bound within 1e−11. BF16 also reproduced its first batch bit-for-bit in log probability after converting its expert weights to and from the checkpoint adapter's layout. Six CPU tests cover an independent full-KL oracle, zero and tiny tails, displaced candidate top tokens, masking, offset and cache rejection.

## Cost

Five paired trials, median seconds [min, max]. Each candidate and strategy took turns going first. The online baseline ran the teacher again for each candidate, formed the same top-k on device and passed it directly to the KL computation. Each cache trial built a fresh directory, then read and checksummed every shard for each candidate. The cached and online scores agreed across all five trials.

| Measured work | Fresh BF16 per candidate | Build once and reuse |
|---|---:|---:|
| Build reference only | — | 0.531 [0.517, 0.625] |
| Score W4 | 0.745 [0.741, 0.762] | 0.420 [0.404, 0.436] |
| Score W8 | 0.756 [0.745, 0.778] | 0.424 [0.405, 0.436] |
| Workflow through one candidate | 0.745 [0.741, 0.759] | 0.956 [0.937, 1.062] |
| Workflow through two candidates | 1.515 [1.487, 1.523] | 1.361 [1.351, 1.479] |

The first measured reference build took 0.625 s and occupied **5,767,815 bytes (5.50 MiB)** in 16 shards plus the manifest. Two-candidate savings ranged from 0.008 to 0.162 s across paired trials. All three models were resident and warmed on one batch; CUDA was synchronized at stage boundaries. The workflow clock includes forward passes, FP64 scoring, transfers, hashing reference shards and filesystem I/O. Model loading, input preparation, initial teacher weight-file hashing and exact diagnostics were timed separately or excluded. Filesystem page caches were left to the OS on the server's shared filesystem. Timings measure ordinary buffered writes. Standalone CLI invocations in fresh processes and loading costs are recorded separately in the JSON. Loading and unpacking the resident candidates took 15.4 s for W4 and 17.0 s for W8 before warmup. These timings describe this workload and backend.

## Inputs and backend

- Upstream main `f7d824cee12ff15bfa719ba8a7ea8ace7566df67`. Local BF16 Granite 3.1 1B-A400M base revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`. One H100 80GB HBM3, CUDA 13.4, driver 580.126.20, PyTorch 2.14.0a0+4fdf77b940.nv26.08, Transformers 5.21.0.
- No matching BF16-generated answer trajectories were available. From WikiText-2 raw **test**, shuffle paragraph indices with seed 42, keep the first 32 with at least 64 tokens and truncate each to 128. Score every token after the first. Calibration used the same 8×128 tokens from **train** for both GPTQ recipes. Dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The score here measures teacher-forced prose distributions.
- Each candidate was built from the original BF16 checkpoint and saved as a compressed-tensors checkpoint. Both contain 2,400 packed int32 projections, including 2,304 expert projections. The weights use symmetric groups of 128 at four or eight bits. The adapter uses the upstream decoder and strict state-dict loading, restores native Granite experts, and runs Transformers BF16 forwards with eager attention. This measures the distributions of dequantized weights. Production low-bit kernels and vLLM output parity remain untested; vLLM was absent from the environment.
- The server retained the two checkpoints (`w4/`, `w8/`), evaluation rows (`eval.jsonl`), independent CLI cache (`reference-v1/`) and paired trials (`final/`) under `/gemini-0/space/private/lmx/temp-restore/Vllm-Omni/llm-compressor-qda-artifacts/`. The pinned model, weight hashes, package versions and device occupancy are in [results.json](results.json). See [the reproduction commands and cache contract](REPRODUCE.md).

## Upstream discussion

Start with a reproducible evaluation example attached to the RFC discussion. The existing RFC requests both KL directions. A teacher-selected cache supports the forward direction tested here; a reverse-direction score would need its own approximation contract. The closed hook prototype reconstructed logits from vLLM hidden states and required eager execution. An open follow-up proposes a two-stage hidden-state/exact-KL tool. The vLLM scoring PR proposes on-device KL against stored reference logits. None of those descriptions establish parity for this sparse teacher-selected cache. [RFC #2031](https://github.com/vllm-project/llm-compressor/issues/2031) · [closed #2659](https://github.com/vllm-project/llm-compressor/pull/2659) · [open #2663](https://github.com/vllm-project/llm-compressor/pull/2663) · [vLLM #35961](https://github.com/vllm-project/vllm/pull/35961)

Before proposing an integrated scoring API, run adjacent quantization recipes on **BF16-generated answer spans** in the target vLLM backend. Check the per-position gather against this oracle and compare ranking, tail mass and bias against exact KL. The online reference baseline also provides a control for the concurrent-model approach raised in [#2667](https://github.com/vllm-project/llm-compressor/issues/2667). The two widely separated recipes here establish a working example. They leave the ranking resolution for close candidates open.
