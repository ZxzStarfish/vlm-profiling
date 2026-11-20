# Native Qwen2-VL 三层瓶颈实验方案

## 研究范围与可行性

目标是解释原生 `Qwen/Qwen2-VL-2B-Instruct` 在 RTX 3070 上处理图文请求时，时延花在哪些阶段、模块和算子，以及随图像分辨率和 Token 数变化的原因。参考 [VLMCache 论文](https://doi.org/10.1145/3745756.3809243)的瓶颈分解思路；论文手机 CPU 的绝对时延不作为这台 GPU 服务器的复现目标。不引入 VLMCache 缓存路径、训练、剪枝、量化或跨请求复用。

环境、模型权重和单卡显存均已通过实测。[Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_vl)提供原生模型和 Processor，[PyTorch 2.7 Profiler](https://docs.pytorch.org/docs/2.7/profiler.html)用于采集算子和 CUDA Trace。实验沿用原生模型和单请求 KV Cache。VLMCache 的[公开仓库](https://github.com/yinyuanzhang/VLMCache)作为论文方法参考。

## 阶段与验收

| 阶段 | 本地工作 | 服务器工作 | 通过条件 |
| --- | --- | --- | --- |
| 1 环境 | 检查脚本、版本记录 | 安装与反馈 | **已通过**：CUDA、包一致性、类导入 |
| 2 Baseline | 官方权重、原生图文推理脚本、输入记录 | 校验模型并单卡运行 | 图像确实进入 ViT，输出正常，记录真实 Token 与峰值显存 |
| 3 阶段/模块 | 计时边界、模块映射、结果 CSV | 在空闲卡运行重复实验 | Stage 与 Module 的时延分布可信，可解释重叠与开销 |
| 4 算子/Kernel | 短窗口 Profiler、参数扫描、绘图与报告 | 收集 Trace/CSV 并反馈 | 三层瓶颈归因有原始记录支撑 |

环境和 Baseline 已通过。完整实验已按 [八组受控输入和三层计时方案](run_latency_campaign.md)运行，并采集 ViT、LLM Prefill 与短 Decode 轨迹。结果与适用范围见 [三层分析报告](../reports/latency_campaign.md)。

## 测量定义

1. **Stage**：CPU 图像预处理；GPU ViT（含 Patch Merger，归入视觉阶段）；首轮 LLM Prefill；后续逐 Token Decode。分别记录 GPU 耗时与请求端墙钟耗时。TTFT 定义为从开始处理该请求到首个输出 Token 计算完成；同时报告包含/不包含预处理的边界。TTFT 与阶段时间未必精确相加，因为 CPU/GPU 重叠、传输和调度存在开销。
2. **Module/Layer**：ViT Block、LLM Block，以及各自 Attention、MLP、Norm 的耗时和调用次数。对嵌套模块区分包含子模块的总时间与自身时间，避免重复相加。先固定推理路径和 Attention 实现，再解释不同模块差异。
3. **Operator/Kernel**：用 `torch.profiler` 的 CPU/CUDA 活动和标注窗口统计 MatMul、Attention、Norm 等算子及其关联 Kernel。PyTorch 算子与 CUDA Kernel 不是一一对应；融合 Kernel 要按实际 Trace 解释。`nsys`/`ncu` 缺失时，先完成 PyTorch 层证据，再决定是否需要更细工具。

Baseline 先用官方 Processor、Chat Template、`model.eval()`、`torch.inference_mode()`、batch 1、单 GPU、FP16，无量化和 CPU Offload。保留正常单请求 KV Cache。首先用小图和短回答证明图文推理，再测 500×375；若 OOM，记录失败并缩小输入，不悄悄改变模型。

## 变量、控制与结果

- 主变量：图像分辨率、实际视觉 Patch/Token 数、文本输入 Token 数、生成 Token 数。优先一次只变一个变量；将 500×375 设为论文参考点，同时记录 Processor 实际网格，不硬写论文 Token 数。
- 控制：模型 commit、软件锁定清单、dtype、Attention backend、batch、prompt 模板、Processor 参数、GPU 型号/编号、当前 GPU 占用、warmup 与 repeat。共享 GPU 忙时只做功能验证，不采纳计时结论。
- 基础指标：图像预处理、ViT、Prefill、Decode、TTFT、总生成时延、每 Token Decode 分布、峰值显存；报告中用中位数和离散度，并保留每次原始记录。
- 数据流：服务器输出 JSON/CSV/短 Trace，Codex 使用已授权的 SSH 连接把数据带回本地 `results/` 并写入 `reports/`。Warm Inference 边界从 CPU 请求预处理开始。

## 扩展实验与验收

首轮三层实验使用一张合成图。扩展实验检查换图后是否仍能看到相同趋势，并用相同请求研究 Decode 中两种计时的排序差异。模块区间和 Kernel 累计来自不同计时口径，解释时结合 CPU、CUDA API 与 GPU 时间线。

| 实验 | 状态 | 实际完成内容 |
| --- | --- | --- |
| 1 阶段趋势复测 | 完成 | GPU 1 的 A/B 各跑 8 组合成图和 20 组真实图，共 392 次正式测量。逐组核对 Token 和 GPU UUID |
| 2 Decode 计时差异及短期重复 | 完成 | 相同输入和 15 步 Decode，三组重新预热的对照共 15 次 Event 请求和 3 条短轨迹，另采 1 条 Attention 内部标注轨迹 |
| 3 热点 Kernel 归因 | 完成 | Nsight Systems 时间线和 Nsight Compute 三个热点硬件计数、两个补充调度采集 |
| 跨 GPU 一致性 | 取消 | 用户明确移出本轮范围。首轮 GPU 5 仅作为趋势参考 |
| 长期或跨日稳定性 | 本轮范围外 | A/B 是同一天两次独立启动，记录实际时间，不设固定等待间隔 |

各项沿用原生 Qwen2-VL-2B、单卡 FP16 和 batch 1。完整执行命令、判读方法和预期文件见 [复现性验证运行说明](run_reproduction.md)。若某个趋势只在部分照片或时段出现，先记录差异并检查 Token 网格和遥测。关于计算、访存与调度的结论限于实际采集到的计数器和被测 Kernel。

截至 2026-11-20，当前范围内的采集、校验、分析和图表均已完成。ViT 与 Prefill 的趋势在 A/B 和四张照片中都重现，合成小图 TTFT 的大幅波动在 B 再次出现。结果与仍无法回答的问题见 [三层分析报告](../reports/latency_campaign.md)和 [A/B 复测结果](run_reproduction.md)。
