# Native Qwen2-VL 三层瓶颈实验方案

## 研究范围与可行性

目标是解释原生 `Qwen/Qwen2-VL-2B-Instruct` 在 RTX 3070 上处理图文请求时，时延花在哪些阶段、模块和算子，以及随图像分辨率和 Token 数变化的原因。参考 [VLMCache 论文](https://doi.org/10.1145/3745756.3809243)的瓶颈分解思路；论文手机 CPU 的绝对时延不作为这台 GPU 服务器的复现目标。不引入 VLMCache 缓存路径、训练、剪枝、量化或跨请求复用。

实验技术上可行：环境已能导入 Qwen2-VL 并识别 CUDA；[Transformers 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_vl)提供原生模型和 Processor，[PyTorch 2.7 Profiler](https://docs.pytorch.org/docs/2.7/profiler.html)提供算子、CUDA 活动和 Trace。仍需权重与显存实测。VLMCache 的[公开仓库](https://github.com/yinyuanzhang/VLMCache)包含原生推理入口，但核心代码服务于背景/前景分离和缓存；本项目可参考其输入，不直接采用改造模型作为被测对象。

## 阶段与验收

| 阶段 | Codex 在本地完成 | 用户在服务器完成 | 通过条件 |
| --- | --- | --- | --- |
| 1 环境 | 检查脚本、版本记录 | 安装与反馈 | **已通过**：CUDA、包一致性、类导入 |
| 2 Baseline | 官方权重获取说明、原生图文推理脚本、输入记录 | 手动上传权重/代码，单卡运行 | 图像确实进入 ViT，输出正常，记录真实 Token 与峰值显存 |
| 3 阶段/模块 | 计时边界、模块映射、结果 CSV | 在空闲卡运行重复实验 | Stage 与 Module 的时延分布可信，可解释重叠与开销 |
| 4 算子/Kernel | 短窗口 Profiler、参数扫描、绘图与报告 | 收集 Trace/CSV 并反馈 | 三层瓶颈归因有原始记录支撑 |

当前位于**阶段 2 的上传与验证点**：软件环境通过，Windows 上固定 commit 的 11 个模型文件已齐全并完成权重 SHA-256 校验。服务器尚未加载模型，不能提前给出推理时间或显存结论。

## 测量定义

1. **Stage**：CPU 图像预处理；GPU ViT（含 Patch Merger，归入视觉阶段）；首轮 LLM Prefill；后续逐 Token Decode。分别记录 GPU 耗时与请求端墙钟耗时。TTFT 定义为从开始处理该请求到首个输出 Token 计算完成；同时报告包含/不包含预处理的边界。TTFT 与阶段时间未必精确相加，因为 CPU/GPU 重叠、传输和调度存在开销。
2. **Module/Layer**：ViT Block、LLM Block，以及各自 Attention、MLP、Norm 的耗时和调用次数。对嵌套模块区分包含子模块的总时间与自身时间，避免重复相加。先固定推理路径和 Attention 实现，再解释不同模块差异。
3. **Operator/Kernel**：用 `torch.profiler` 的 CPU/CUDA 活动和标注窗口统计 MatMul、Attention、Norm 等算子及其关联 Kernel。PyTorch 算子与 CUDA Kernel 不是一一对应；融合 Kernel 要按实际 Trace 解释。`nsys`/`ncu` 缺失时，先完成 PyTorch 层证据，再决定是否需要更细工具。

Baseline 先用官方 Processor、Chat Template、`model.eval()`、`torch.inference_mode()`、batch 1、单 GPU、FP16，无量化和 CPU Offload。保留正常单请求 KV Cache。首先用小图和短回答证明图文推理，再测 500×375；若 OOM，记录失败并缩小输入，不悄悄改变模型。

## 变量、控制与结果

- 主变量：图像分辨率、实际视觉 Patch/Token 数、文本输入 Token 数、生成 Token 数。优先一次只变一个变量；将 500×375 设为论文参考点，同时记录 Processor 实际网格，不硬写论文 Token 数。
- 控制：模型 commit、软件锁定清单、dtype、Attention backend、batch、prompt 模板、Processor 参数、GPU 型号/编号、当前 GPU 占用、warmup 与 repeat。共享 GPU 忙时只做功能验证，不采纳计时结论。
- 基础指标：图像预处理、ViT、Prefill、Decode、TTFT、总生成时延、每 Token Decode 分布、峰值显存；报告中用中位数和离散度，并保留每次原始记录。
- 数据流：服务器输出 JSON/CSV/短 Trace；用户手动带回本地 `results/`，Codex 在本地检查、绘图并写入 `reports/`。不把安装时间和模型加载时间计入 Warm Inference。

## 对旧提示词的修正

原 `gpt-prompts.md` 的总体目标仍适用，但路径和容量是假设：实际本地仓库是 `D:\LAB\vlm-profiling`，服务器是 `~/projects/vlm-profiling`，`/mnt/sdb` 不可直接写；根文件系统检查时可用约 385GB。示例 GPU 4 不能固定沿用，需按运行时空闲情况选择。先完成最小 Baseline 的正确性与显存验证，再增加重复计时；此后才开展模块和算子归因。
