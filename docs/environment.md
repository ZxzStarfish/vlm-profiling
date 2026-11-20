# 已确认的实验环境

| 项目 | 已确认结果 |
| --- | --- |
| Python | Linux x86-64，Python 3.12.3，项目 `.venv` 中使用 pip 24.0 |
| PyTorch | 2.7.1+cu126，运行时 CUDA 12.6 |
| GPU | RTX 3070，8192 MiB，NVIDIA 驱动 570.211.01 |
| 依赖 | Transformers 4.51.3、Accelerate 1.7.0、Pillow 11.2.1，见[完整锁定清单](../requirements.lock.txt) |
| Nsight | Systems 2024.6.2.225、Compute 2025.1.1.2，见[采集配置](run_latency_campaign.md) |

Baseline 已确认 CUDA 可用、依赖一致、官方 Processor 和模型类可加载，且原生图文推理在单卡 FP16 上通过。模型文件及 SHA-256 见[模型清单](../data/model_manifest.json)。正式样本的硬件和显存记录随各次结果保存，重跑前按[运行说明](run_latency_campaign.md)检查 GPU 的即时占用。
