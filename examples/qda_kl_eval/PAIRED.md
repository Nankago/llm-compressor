# Granite 上两种可复用 KL 参考缓存的配对实验

**结论。** 在 3,732 个完全相同的 next-token 位置上，QDA 的 `k=128` 尾桶保住了两份 GPTQ 候选的排序，缓存文件缩小 **49.75%**。两种方法都复用 BF16 参考时，八轮配对没有显示稳定的评分加速。一份和两份候选的总耗时都只有 2/8 轮由 top-k 胜出。[逐轮结果与设备记录](paired-results.json)

### 分数与正确性

| 候选 | A · 全词表 KL | B · top-128 + 尾桶 | 绝对差（nats/token） | 低估比例 |
|---|---:|---:|---:|---:|
| BF16 自评 | 0 | 0 | 0 | — |
| GPTQ W8A16 | 0.00568196 | 0.00505182 | 0.00063014 | 11.09% |
| GPTQ W4A16 | 0.25140527 | 0.21834330 | 0.03306197 | 13.15% |

两边各用自己的 final RMSNorm、BF16 输出头及 Granite 的 logits 除数 6。按 128 个位置分块投影，按全词表归一化，使用自然对数，按有效 token 求均值。三份模型分块重建的 logits 在全部位置和词表上与原生 Transformers 前向逐位一致。A 与直接取原生 logits 后用 FP64 算 KL 的最大均值误差为 2.33×10⁻⁸。八轮内各分数的波动为零；两个量化档位相距较远，尚无相邻配方的排序证据。

A 保存每个评分位置的 **post-norm BF16 hidden state**，候选评分时重新做参考侧投影；候选侧的隐藏状态按批直接送入输出头，两边的候选中间落盘字节均为零。B 保存 teacher 的 int32 ID、FP32 logprob 和 FP32 尾部总质量的自然对数。候选按 teacher ID 取值，尾桶用补集的 log-sum-exp，避免从 1 减去接近 1 的概率和。与此前 FP64 缓存原型相比，本次 B 的两个均分变化均小于 1.4×10⁻⁸。[QDA 博客 §2.1](https://mit-han-lab.github.io/qda-blog/#s2a) · [论文式 (6) 与 §5.2](https://arxiv.org/html/2611.23388v1#S4.SS2)

### 磁盘与时间

| 参考缓存（16 个分片和 manifest） | A · hidden | B · top-k |
|---|---:|---:|
| 文件总字节 | 7,646,808 | 3,842,530 |
| 文件字节/评分 token | 2,048.98 | 1,029.62 |
| 文件系统分配字节（各文件 st_blocks × 512） | 7,653,888 | 3,847,168 |
| 分配字节/评分 token | 2,050.88 | 1,030.86 |

八轮配对，表内为墙钟秒数的中位数 [最小值，最大值]。末列是先逐轮计算 B−A 再取中位数，正值表示 B 较慢。

| 工作 | A · hidden | B · top-k | 配对 B−A |
|---|---:|---:|---:|
| 建参考缓存，含写盘 | 0.526 [0.510, 0.536] | 0.525 [0.517, 0.552] | +4.4 ms |
| 只评 W4，含读盘 | 0.414 [0.404, 0.472] | 0.420 [0.406, 0.431] | +5.1 ms |
| 只评 W8，含读盘 | 0.411 [0.404, 0.424] | 0.421 [0.408, 0.431] | +6.7 ms |
| 建缓存并评完一份 | 0.947 [0.928, 1.009] | 0.953 [0.941, 0.976] | +3.9 ms |
| 建缓存并评完两份 | 1.356 [1.334, 1.414] | 1.367 [1.349, 1.384] | +14.0 ms |

第一轮建缓存 A 为 0.536 s，B 为 0.552 s。两份候选的配对总耗时差横跨 **B 快 30.6 ms 到 B 慢 33.9 ms**。实测的一至两份候选内没有可复现的回本点。

另跑一遍 CUDA event 诊断，候选 backbone 在两种格式中都占约 375–379 ms。A 的参考头投影及归一化约 3.0 ms，全词表 KL 与求和约 4.1 ms；B 省去参考投影，但当前补集 log-sum-exp 仍扫描候选全词表，尾桶 KL 与求和约 11.2 ms。主机端读取、校验及解码 A 为 19.8–20.6 ms，B 为 8.5–20.8 ms。这些区间来自单独的诊断，CPU 与 GPU 区间可能重叠。正式墙钟独立测量。

### 与 #2663 的对应关系

核对的 PR head 为 `1ac89a7526b221b5780d91ed2e2da83a3fdf5c03`。该提交的提取器指定层号 `num_hidden_layers`，离线代码将捕获值按 **pre-norm** 处理，FP32 补 norm 和输出投影，再按块算全词表 KL。这里 A 是同一后端上的 **#2663 式复现**：在 Transformers 返回的 post-norm 处保存，所以回放时直接投影。三个输出头的形状、dtype、权重 SHA、bias 和缩放都核对一致，独立评分进程才复用候选头；头不同时必须提供经校验的参考头。B 是 QDA 缓存与评分公式原型。两边均在单个 Transformers 进程的 GPU 上归约，未进入 vLLM worker，本环境没有 vLLM。[PR 固定版本源码](https://github.com/vllm-project/llm-compressor/tree/1ac89a7526b221b5780d91ed2e2da83a3fdf5c03/tools/kl_divergence)

另在前 247 个相同位置上，将 Transformers 捕获的 **pre-norm** 激活交给该提交未经修改的 `_compute_kl_for_chunk`。这一项只对照数值，未纳入上表计时。

| 候选 | 原生 logits 的精确 KL | PR 原函数，temperature=1 | PR 原函数，temperature=6 |
|---|---:|---:|---:|
| W8 | 0.006170 | 0.020196 | 0.004393 |
| W4 | 0.224056 | 1.564443 | 0.224275 |

PR 函数的温度参数可以表达 Granite 的除数，但原生 BF16 路径还有舍入。在 W8 上，保留原生 norm、仅将 head 和除法换为 FP32 就得到 0.004436；完整恢复原生 norm、head 和除法后，pre-norm 回放的 logits 与原模型逐位一致。**建议先提交可复现实验示例和 Granite 原生 logits 的回归用例。** 对 #2663 而言，先验证模型原生输出路径有直接的数值依据；对 top-k 格式而言，当前证据支持节省存储。是否集成，可再用 vLLM worker、BF16 生成的回答和相邻量化配方，测排序及端到端耗时。旧原型关闭时作者指出 eager hook 与维护者期望的架构有分歧；RFC 还提出双向 KL，本例验证的是 KL(P_teacher ‖ Q_candidate)。[旧 PR 的关闭说明](https://github.com/vllm-project/llm-compressor/pull/2659#issuecomment-4338693399) · [RFC #2031](https://github.com/vllm-project/llm-compressor/issues/2031)

### 负载与复现

- 复用 Granite 3.1 1B-A400M base，revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`。现有两个不同的量化 checkpoint 均来自同一份 WikiText-2 **train** 8×128 校准 token，W4/W8 对称分组 128。每份实含 2,400 个打包投影，包括 2,304 个专家投影；从磁盘严格重载，解包至 BF16 后运行。完整权重摘要见 JSON。
- 未找到匹配的 BF16 回答轨迹。复用同一份 WikiText-2 raw **test** 的 32 段留出文本，每段至多 128 个 token，评分位置 `t` 使用 logits `t−1`。两条路径的输入、位置、tokenizer 摘要完全一致。通过 10 项 CPU 测试，包括全词表极限、下界、padding、偏移、不同输出头、损坏或不匹配的缓存。
- 空闲 H100 80 GB HBM3，物理卡 1，driver 580.126.20，CUDA 13.4，PyTorch 2.14.0a0+4fdf77b940.nv26.08，Transformers 5.21.0，compressed-tensors 0.20.2a20261103。NVML 每 0.5 s 检查，未见外部进程。权重、token、两类缓存和运行日志留在 `/gemini-0/space/private/lmx/temp-restore/Vllm-Omni/llm-compressor-qda-artifacts/`，配对缓存在 `paired-h100/`。
- 同卡串行、同一进程内分阶段同步；三份模型各预热一个 batch，模型和输出头在显存中。每轮每种格式重建到新目录，方法和候选顺序交替。计入缓冲写盘、读取、分片哈希、传输、投影及归约；不清理共享文件系统页缓存。加载、首次权重及输出头指纹校验、预热、逐位验证、诊断均在配对计时之外。加载 BF16/W4/W8 分别为 0.79/15.26/17.05 s。

[模型与输入准备](REPRODUCE.md) 完成后，在本目录下运行以下命令。变量沿用该文档；本机现有候选为 `$ART/w4`、`$ART/w8`，输入为 `$ART/eval.jsonl`，`$ART` 为上述服务器目录。

```bash
ART=/gemini-0/space/private/lmx/temp-restore/Vllm-Omni/llm-compressor-qda-artifacts
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest -v test_evaluate test_paired
args=(--inputs "$ART/eval.jsonl" --reference-model "$REF" --reference-revision "$REV")
for fmt in hidden topk; do
    "$PY" paired.py prepare-reference --format "$fmt" --model "$MODEL" --cache "$WORK/ref-$fmt" "${args[@]}"
    "$PY" paired.py score --format "$fmt" --model "$ART/w4" --cache "$WORK/ref-$fmt" "${args[@]}"
done
"$PY" paired_benchmark.py "$MODEL" "$ART/w4" "$ART/w8" "$ART/eval.jsonl" "$WORK/paired" --repeats 8
```

运行 PR 原函数的可选数值核对时，脚本会先检查源码 SHA-256。

```bash
prsha=1ac89a7526b221b5780d91ed2e2da83a3fdf5c03
mkdir -p "$WORK/pr2663"
for file in compute_kl.py lm_head_utils.py; do
    curl -fL "https://raw.githubusercontent.com/vllm-project/llm-compressor/$prsha/tools/kl_divergence/$file" -o "$WORK/pr2663/$file"
done
"$PY" pr2663_check.py "$WORK/pr2663" "$MODEL" "$ART/w4" "$ART/w8" "$ART/eval.jsonl" "$WORK/pr-check.json"
```
