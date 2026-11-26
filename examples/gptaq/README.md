# GPTAQ and QuantEase on pretrained Qwen3-0.6B

A QDA-style GPTAQ refit reduced held-out full-vocabulary KL by 13.6–15.0% relative to GPTQ at 32,768 calibration tokens across two seeds. It cost 61–90 additional CPU seconds per build. ARC-Easy moved in opposite directions. The existing floating-point-prefix control improved ARC at both seeds and built faster. These results support a focused discussion of upstream GPTAQ support and leave task-level benefit unresolved. One fixed-grid QuantEase sweep worked through the native OneShot recipe and exporter, but added cost without a consistent quality gain.

## Long calibration

Each seed selects 128 distinct WikiText-2 training articles and one 256-token window per article. Methods within a seed use identical tokens. KL is in nats per predicted token. Lower KL and perplexity (PPL) are better. ARC is the number correct out of 2,376. BF16 scored KL 0, PPL 45.746 and ARC 1,450.

| Calibration seed | Method | Exact KL | PPL | ARC correct | Quantization wall (s) | Peak process RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|
| 42 | GPTQ | 0.14597 | 48.984 | 1,347 | 127.9 | 3.62 |
| 42 | GPTQ, floating-point prefix | 0.14524 | 48.773 | 1,424 | 107.7 | 3.71 |
| 42 | GPTQ + QuantEase sweep | 0.14646 | 49.746 | 1,374 | 157.3 | 3.64 |
| 42 | GPTAQ refit + GPTQ | 0.12412 | 47.811 | 1,433 | 218.3 | 4.32 |
| 43 | GPTQ | 0.14505 | 49.142 | 1,365 | 125.2 | 3.69 |
| 43 | GPTQ, floating-point prefix | 0.14225 | 48.804 | 1,394 | 111.7 | 3.68 |
| 43 | GPTQ + QuantEase sweep | 0.14508 | 49.054 | 1,332 | 156.0 | 3.64 |
| 43 | GPTAQ refit + GPTQ | 0.12529 | 47.940 | 1,320 | 186.1 | 4.30 |

GPTAQ lowered KL on 45/48 and 43/48 held-out articles. The 95% paired article bootstrap intervals for the mean change against GPTQ were [−0.02714, −0.01688] and [−0.02589, −0.01377]. Against the floating-point prefix they were [−0.02721, −0.01560] and [−0.02279, −0.01135]. All intervals use 10,000 resamples of the same test articles. They measure article variability conditional on each checkpoint. Two calibration seeds give limited evidence about seed variability.

Against GPTQ, GPTAQ changed 224 wrong answers to right and 138 right to wrong at seed 42 (+86, two-sided exact McNemar p=0.0000072). At seed 43 the counts were 168 and 213 (−45, p=0.0241). Against the floating-point prefix GPTAQ gained 9 (p=0.666) and lost 74 (p=0.000180). These are descriptive, unadjusted comparisons on one benchmark. On a fresh run of the same 256-question subset, regrouped into batches, 256/256 and 255/256 correctness flags reproduced. One question changed at seed 43. The JSON includes the paired correctness bits and changed row ID.

## Short calibration and native OneShot

At 32 × 128 calibration tokens, GPTAQ's KL advantage disappeared at one seed.

| Seed | GPTQ KL → GPTAQ KL | GPTQ PPL → GPTAQ PPL | Build seconds |
|---|---|---|---|
| 42 | 0.18501 → 0.17798 | 51.187 → 49.636 | 67.1 → 99.1 |
| 43 | 0.17639 → 0.17872 | 50.356 → 50.417 | 66.7 → 82.9 |

The separate native OneShot trial uses identical tokens, the sequential pipeline, one sample per batch, and single-module quantization for both controls. On the full model at seed 42, GPTQ → QuantEase moved KL from 0.18351 to 0.19013, PPL from 50.688 to 51.374 and OneShot from 85.4 to 112.5 s. All 196 packed projection tensors changed. Every other checkpoint tensor, including every scale, stayed bitwise identical. Both checkpoints reloaded and produced finite logits. A separate first-block trial verified all seven projections. The native pipeline and research runner collect statistics through different entry points, so each comparison uses its own GPTQ control.

## Implementations and checks

- **GPTQ** uses the pinned upstream solver. The dense runner targets seven projections per decoder block, 28 blocks total. It uses symmetric W4A16, group and block size 128, memoryless min/max BF16 scales, static activation ordering, and damping of 0.01 times the mean Hessian diagonal. The embeddings and tied LM head stay BF16. The stock and QuantEase runs propagate each quantized block output downstream. The floating-point-prefix control propagates the saved pre-quantization output, corresponding to block-level `propagate_error=False`. [Upstream implementation](https://github.com/vllm-project/llm-compressor/tree/18b10e2d7269e332c05eef4dda52adf74c08872e/src/llmcompressor)
- **GPTAQ** runs aligned BF16 teacher and quantized-prefix student streams. For each projection input it accumulates `H = Xqᵀ Xq / N` and `C = (Xf − Xq)ᵀ Xq / N`. Before choosing scales and running the same GPTQ solver, it fits `W' = W + 0.3 W C (H + λI)⁻¹` with a damped Cholesky solve, then casts to BF16. Alpha 0.3 was fixed for all runs. This dense Qwen3 research runner is a prototype of QDA's refit. Its BF16 result matched the pinned QDA function bitwise on three fixtures. Tests also compare against independent FP64 least squares and verify cross-term orientation. The first block, where both streams see identical inputs, matched stock packed tensors and scales bitwise for every paired build. The prototype raises on failed factorization. All measured dense builds completed without solver fallback. [GPTAQ paper](https://proceedings.mlr.press/v267/li25dn.html) · [QDA refit](https://github.com/humanfia/qda/blob/dc39a02bfd87bb2790e34001708d54ce9131e076/packages/qda-quant/src/qda_quant/steps/gptaq.py)
- **QuantEase** uses GPTQ initialization followed by one input-column coordinate sweep at fixed group scales on the actual BF16 dequantization grid. Its scope is symmetric, grouped INT4 on BF16 weights. The experimental `QuantEaseModifier` resolves through `Recipe.from_dict`, processes one module at a time, and preserves GPTQ's round-to-nearest fallback. The kernel matches an independent exhaustive 16-code FP64 coordinate search on a fixture covering a dead column and a partial tile. Extra sweeps and sparse outlier variants are outside this experiment. [Modifier](../../src/llmcompressor/modifiers/experimental/quantease.py) · [QuantEase authors' implementation](https://github.com/linkedin/QuantEase)

Every scored candidate was loaded from a real packed checkpoint via Transformers `CompressedTensorsConfig(dequantize=True)`. For each of the 196 projections, the loaded BF16 weight was checked bitwise against compressed-tensors unpacking. These scores measure the exported weights. Serving throughput and low-bit inference kernels require a separate experiment. The dense runner wrote 538,459,960 bytes of model weights per checkpoint.

## Evaluation and timing

The WikiText-2 test panel contains 48 distinct held-out articles, one 128-token window each, seed 7651, totaling 6,096 next-token positions. Scores use full-vocabulary KL from the BF16 teacher to each candidate, FP32 log-softmax, FP64 summation and the same gold-token positions for PPL. ARC-Easy uses all 2,376 test questions and 9,501 choices. The zero-shot prompt is `Question: <question>\nAnswer:`, with no chat template. Choices are ranked by the sum of their answer-token log probabilities without length normalization. The loader checks the prompt/answer token boundary and the first batch checks selective LM-head output against a full causal forward. This explicitly specified protocol permits paired comparisons here. [WikiText revision](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3) · [ARC revision](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453)

All eight H100s had resident jobs before and after the experiment (about 61 GiB per card). Runs used CPU BF16 on Intel Xeon Platinum 8468V, six threads per process. Each dense build started from the original checkpoint in a fresh process, with local files and normal shared OS page cache. The build timer spans the decoder loop including statistics, refitting, quantization, and in-loop compression and round-trip checks. It excludes tokenization, model loading, embedding lookup and final disk serialization. Export seconds are separate in JSON. Peak RSS is for the full process including export. Native OneShot starts just before the API call and ends at return, before saving. Each build configuration was timed once and some independent CPU jobs overlapped. Timings establish observed cost at this model size.

## Reproduction

Use this branch with Python 3.12.3. Tested packages were PyTorch `2.14.0a0+4fdf77b940.nv26.8.63802676`, Transformers `5.21.0`, compressed-tensors `0.20.2a20261103`, safetensors `0.8.0`, PyArrow `25.0.1` and NumPy `2.1.0`. Install llm-compressor editable in that environment so its generated version module is available. Use the model snapshot and both dataset revisions linked here, with a fresh output directory and about 11 GiB free for artifacts. The scripts verify SHA-256 for the source weights and parquet inputs; the JSON records checkpoint and token digests. [Qwen3-0.6B revision](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)

```bash
# From the repository root, with all inputs already local.
export MODEL=/path/to/Qwen3-0.6B
export TRAIN=/path/to/wikitext-2-raw-v1/train-00000-of-00001.parquet
export TEST=/path/to/wikitext-2-raw-v1/test-00000-of-00001.parquet
export ARC=/path/to/ARC-Easy/test-00000-of-00001.parquet
export OUT=/path/to/fresh/results
# Optional comparison with QDA's implementation and its installed dependencies.
# export QDA=/path/to/qda-at-dc39a02
bash examples/gptaq/reproduce.sh
```

The focused and upstream regression selection gave **56 passed, 59 skipped** in the CPU environment. The optional regression run uses the cached `nm-testing/tinysmokellama-3.2` fixture at `d1abfb7465f27fd86fccb14bcb42e9145ba55d4c`.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q examples/gptaq/test_{refit,coordinate}.py \
  tests/llmcompressor/modifiers/gptq/test_{base,gptq_quantize,loss_mask}.py
```

## Upstream decision

The existing GPTAQ request closed after inactivity. The discussion identifies both the need for paired activations and the cost of retaining a floating-point stream. The floating-point-prefix option landed separately. The additional refit here beats that option on KL at both seeds, but does not consistently beat it on ARC. The most informative next experiment would keep all three controls, use an additional pretrained model, and measure peak device memory and task quality over multiple calibration seeds before proposing a native dual-stream modifier. The single-sweep QuantEase implementation already exercises the integration route. Its whole-model evidence does not justify proposing it as a default. [GPTAQ request and maintainer discussion](https://github.com/vllm-project/llm-compressor/issues/2004#issuecomment-3511947204) · [Memory discussion](https://github.com/vllm-project/llm-compressor/issues/2004#issuecomment-3524594856) · [Floating-point-prefix change](https://github.com/vllm-project/llm-compressor/pull/2008) · [Per-run results](results.json)
