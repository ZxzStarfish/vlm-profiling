# Native Qwen2-VL 三层性能瓶颈分析

研究目标：在原生 `Qwen/Qwen2-VL-2B-Instruct` 上分析 **阶段 → 模块/层 → 算子/Kernel** 的时延与输入规模关系。参考 VLMCache 的瓶颈问题，不采用其缓存优化或模型改造。

**当前状态：** 本轮范围已完成。包括原生图文 Baseline、三层时延实验、GPU 1 上的 A/B 复测、三组 Decode 重复和 Nsight 归因。跨 GPU 验证已取消。研究结论限于本轮模型、设备和输入条件。

- [实验方案与阶段验收](docs/research_plan.md)
- [已确认环境与版本](docs/environment.md)
- [本地开发和手动部署](docs/workflow.md)
- [原生 Baseline 运行说明](docs/run_baseline.md)
- [三层实验矩阵与复现命令](docs/run_latency_campaign.md)
- [三层结果与分析报告](reports/latency_campaign.md)
- [A/B 复测结果与运行说明](docs/run_reproduction.md)
- [Decode 计时与 Nsight 补充实验](reports/latency_campaign.md#补充实验一-decode-的两种计时为什么排序不同)
- [主要结果图](reports/figures/latency_summary.png)
- [完整依赖锁定清单](requirements.lock.txt)

`scripts/` 包含采集与汇总入口，`configs/` 定义实验配置，`data/` 保存输入和模型清单，`results/` 保存逐次测量与导出的计数表，`reports/` 保存汇总与图表。模型权重、采集日志、PyTorch 原始轨迹和 Nsight 二进制报告由运行者在本地保存。复查已发布结论可先读最终报告及其链接的结果表。
