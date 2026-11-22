# Granite 长校准，第二次前向值不值得跑？

**若目标是保住本次 ARC-Easy 准确率，值得保留。** 在 Granite 3.1 1B-A400M 上，关闭 GPTQ 误差传播让 OneShot 的配对墙钟中位数减少 **10.781 s（19.38%）**。关闭后，WikiText 留出集上的全词表 KL 更接近 BF16，但同一套 ARC-Easy 零样本评测少答对 **53/2,376** 题。不能用这批文本的 KL 单独决定默认值。[逐轮原始数据、设备记录与轨迹摘要](forward-ablation.json) · [复现命令与服务器产物位置](../forward-reproduce.md)

## P0 · 同卡配对

原始 BF16 Granite revision `408b6e90baab8cf24f4aa9f8e19703ffa0a53b29`；WikiText-2 raw **train** 64 篇文章各取一个 1,024-token 窗口，合计 65,536 token。两组都用 W4A16 对称 GPTQ、group size 128、`imatrix_mse`、`batched_quantization="auto"`、block size 128、damping 0.01、batch size 1、全专家校准和每子图一个顺序目标。沿用既有对称 observer 补丁的同一份源码 `c81852cbfc04fca46f5f85a9d88a60525efb4385`。A 开启、B 关闭 `propagate_error`。[原长校准 benchmark](granite-calibration-load.md)

| Granite，秒 | A · 传播 | B · 关闭 |
|---|---:|---:|
| 第一次前向，含统计 hooks，中位数 | 30.444 | 31.071 |
| 第二次前向，中位数 | 10.044 | 0 |
| 权重 observer，中位数 | 4.726 | 4.766 |
| OneShot，中位数 | 55.474 | 44.757 |
| 峰值已分配显存，MiB | 1,204.64 | 1,202.64 |

| 对次 | 顺序 | OneShot A，s | OneShot B，s | 配对节省，s |
|---|---|---:|---:|---:|
| 1 | AB | 55.476 | 44.965 | 10.511 |
| 2 | BA | 55.759 | 44.709 | 11.050 |
| 3 | AB | 54.845 | 44.805 | 10.040 |
| 4 | BA | 55.471 | 43.909 | 11.562 |

每一遍都从原始 checkpoint 启动新进程。两臂各有一次排除在统计外的完整负载预跑，随后共享已预热的私有编译磁盘缓存。OneShot 前后同步 CUDA，模型载入、权重校验、保存 checkpoint 和质量评测均在计时外。子图前向与 weight observe 用当前流的 CUDA events，包含主机提交间隙；第一次前向内的 iMatrix **输入统计**和量化时的**权重 observer**是不同阶段。正式运行没有 profiler 或逐个输入 hook 的同步。剩余墙钟包含激活缓存的更新、量化、上下图搬运、调度等。独立计算每对 OneShot 差，节省范围为 **10.040–11.562 s（18.31%–20.84%）**。

两组第一次前向都是 **25 子图 × 64 = 1,600** 次。A 在子图量化后禁用 modifier hooks，再跑 1,600 次，把压缩权重产生的输出传给后续子图；B 直接缓存第一次前向的输出，第二次调用为零。两组各有 25 次子图级 weight observe，2,400 个目标投影，无 RTN fallback。正式两组均显式关闭 `log_sequential_error`。另用 TinyMoE 的 11 子图、2 个样本验证：关闭传播但开启该 SQNR 诊断时，第一次和第二次又各跑 **22 次**。开启诊断会改变对照。[顺序管线源码](https://github.com/Nankago/llm-compressor/blob/c81852cbfc04fca46f5f85a9d88a60525efb4385/src/llmcompressor/pipelines/sequential/pipeline.py)

### 概率分布与任务结果

| Granite | BF16 | A · 传播 | B · 关闭 | B−A |
|---|---:|---:|---:|---:|
| 全词表 KL(P_BF16 ‖ Q)，nats/token | 0 | 0.1985135 | 0.1853652 | −0.0131483 |
| WikiText 下一个 token 的困惑度 | 11.122 | 13.273 | 13.157 | −0.116 |
| ARC-Easy 正确数 / 2,376 | 1,663 | 1,613 | 1,560 | −53 |
| ARC-Easy 准确率 | 69.99% | 67.89% | 65.66% | −2.23 个百分点 |

使用第一对的两份真实打包 checkpoint；2,400 个投影中，**2,300** 个的 packed codes 不同，2,300 个的 scales 不同。用同一 Transformers 后端将权重解包到 BF16，在原 KL 实验的 WikiText-2 **test** 32 段、3,732 个固定 next-token 位置上复用同一个 BF16 隐藏状态缓存，逐块精确计算全词表 KL，并用原生前向 FP64 再核对一次。以段为单位配对 bootstrap、按 token 加权，B−A 的 95% 区间为 [−0.02230, −0.00454]。[精确评分代码](https://github.com/Nankago/llm-compressor/blob/8e7e6ad781a416b967b3c4a498d8469d97224d61/examples/qda_kl_eval/paired.py)

ARC-Easy 使用固定 test 分片全部题目。提示为 `Question: …\nAnswer:`，对每个答案文本逐 token 求条件 logprob 之和，取最高分；无示例，无采样，无长度归一化。A 单独答对 147 题，B 单独答对 94 题，二者都对 1,466 题。配对精确 McNemar 检验双侧 p=0.000775。这是该模板、该模型、这两份 checkpoint 的任务结果。[固定 ARC-Easy 数据](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453/ARC-Easy)

## P1 · 第一次前向在哪里花时间

完成 P0 后，在同一卡上单独跑完整长校准，抓取 `model.layers.11` 第一次前向的两段 Kineto trace。每段含三个 1,024-token 样本，起始样本索引为 8、40。通过发射该 kernel 的 CUDA runtime correlation 或 ATen external ID 归类。下表是每个三样本窗口的 GPU 活动区间并集，单位 ms。

| 提交来源 | 窗口 8 | 窗口 40 |
|---|---:|---:|
| 专家 gate/up 的 Hessian 累积 | 5.625 | 5.633 |
| 其他 Hessian 累积 | 2.098 | 2.098 |
| iMatrix 输入统计 | 5.212 | 5.228 |
| 模型 linear | 1.548 | 1.551 |
| 前向其他活动，含 routing 等 | 3.532 | 3.561 |
| 未在前向标记内提交的活动，主要为预取 H→D | 0.802 | 0.812 |

两个窗口各有 1,800 项归于 iMatrix 的 GPU 活动、1,596 项归于 Hessian、303 项归于模型 linear。gate/up Hessian 的单个最热 kernel 类型是 CUTLASS GEMM，两窗口各约 3.40 ms；iMatrix 最热的是归约，约 2.32 ms。主线程在标记范围内的 `cudaStreamSynchronize` 只有 0.013 ms。每窗口可见约 13.4 MB H→D 和 768 B D→H。后台预取可能与主前向重叠。带 profiler 的三次前向 CPU 区间约 133 ms，其中 GPU 忙碌并集约 18 ms；大量小操作的主机提交和 profiler 本身都能扩大间隙，不能把这个空闲比例外推到正式 30 s，也不能将 CPU 和 GPU 时间相加。

**优先拆 gate/up 重复的 Gram 矩阵乘。** 每个采样前向均观察到 32 个专家的 64 个 gate/up 调用使用同一个输入存储、相同 shape/stride，现有代码各算一次 Hessian 更新；每个模块的最终权重量化仍然独立。对六个真实 BF16 输入做最小回放，保留 64 份独立 H 与计数，只复用每个样本的 Gram 增量。四对原始/复用时间分别为 16.17/3.65、16.03/3.79、15.95/3.55、15.90/3.60 ms，**所有 H 与计数逐位一致**。此处只测统计更新。[GPTQ 累积实现](https://github.com/Nankago/llm-compressor/blob/c81852cbfc04fca46f5f85a9d88a60525efb4385/src/llmcompressor/modifiers/gptq/helpers.py)

### 单独的完整校准原型

保持传播开启，在一次线性化 MoE 全专家前向内，仅当两个线性层收到同一个未被原地修改的输入张量、使用同一 CUDA 流且没有 loss mask 时，复用该样本的 Gram 增量。各模块仍独立更新 Hessian 和计数。普通 GPTQ 入口沿用原有行为。每次 Granite 长校准，这组调用由 98,304 次 Gram 乘法降为 **1,536 次计算、96,768 次复用**。

| 同卡附加对照 | 原实现 OneShot，s | 复用 OneShot，s | 第一次前向节省，s | OneShot 节省，s |
|---|---:|---:|---:|---:|
| 1 | 53.507 | 53.264 | 0.959 | 0.243 |
| 2 | 54.895 | 53.716 | 1.817 | 1.179 |
| 3 | 54.913 | 52.777 | 2.337 | 2.136 |
| 4 | 54.285 | 52.971 | 1.298 | 1.314 |

第一次前向配对中位节省 **1.557 s**，OneShot 中位节省 **1.246 s（2.28%）**，范围 0.243–2.136 s。四对的两臂峰值已分配显存均为 1,204.64 MiB。此表是另跑的四对，和 P0 跳过传播的节省不能直接相加。

Granite 开传播、Granite 关传播、TinyMoE 开传播三组均与各自原产物的**全部七个导出文件逐字节一致**，包括 tokenizer、配方、配置和整份 safetensors。Granite 开传播的权重 SHA-256 为 `8c9f1a5e56dd9dc35f84ee78c73993b947580b32284b9c66e51dd0e9ce61c428`。CUDA 定向测试另覆盖原地修改、相同内容的新张量、带 mask 的回退和作用域释放。这个机制已具备**独立审阅实验补丁**的依据；库级接入位置、分布式或异构 offload、额外临时 Gram 对大模型显存预算的影响仍需验证。开启选项的研究驱动位于 [try_gram_reuse.py](../try_gram_reuse.py)，[配对计时和逐文件哈希](expert-gram-reuse.json) 可单独复查。

## P2 · 第二个预训练 MoE

在独立空闲 H100 上复用 TinyMoE-100M-2x8（Mixtral 架构），revision `0c1d097c964b949bc0092fa8e99c88c325421167`。缩小校准使用已有 benchmark 从固定上游 README 选取的 8×128 token；评测使用独立 WikiText test 文本，按该 tokenizer 得到 3,673 个位置。两次配对的 OneShot A/B 为 5.562/5.387、5.592/5.441 s，关传播中位节省 **0.163 s（2.93%）**；第一次调用两组各 88 次，第二次从 88 次降为零。精确 KL 从 **0.039901 降到 0.039674**；按段配对区间 [−0.00151, +0.00114] 跨零。耗时趋势复现，质量变化很小，不能据此推断 TinyMoE 的任务准确率。

Granite 和 TinyMoE 正式计时分别使用 H100 2 号和 3 号卡；Granite 质量评测在空闲 4 号卡完成。环境为 H100 80 GB HBM3、PyTorch 2.14.0a0+4fdf77b940.nv26.08、CUDA 13.4、Transformers 5.21.0。各运行有独立 NVML 进程记录，同卡串行；不同卡共享主机。以上结果可以补充“是否跳过传播”的路线图讨论；没有测试用模拟量化前向替换现有前向，也没有测试 SQNR/REAP 的收益。[路线图 #3268](https://github.com/vllm-project/llm-compressor/issues/3268) · [模拟量化前向 RFC #3508](https://github.com/vllm-project/llm-compressor/issues/3508)
