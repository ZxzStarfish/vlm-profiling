# Native Qwen2-VL 三层性能瓶颈分析

研究目标：在原生 `Qwen/Qwen2-VL-2B-Instruct` 上分析 **阶段 → 模块/层 → 算子/Kernel** 的时延与输入规模关系。参考 VLMCache 的瓶颈问题，不采用其缓存优化或模型改造。

**当前状态：** 服务器软件环境已通过 CUDA、依赖和模型类导入检查；本地 11 个固定 commit 模型文件已齐全，两块权重的 SHA-256 已校验，`data/MODEL_READY.txt` 已生成。等待用户手动上传并在服务器运行图文 Baseline。

- [实验方案与阶段验收](docs/research_plan.md)
- [已确认环境与版本](docs/environment.md)
- [本地开发和手动部署](docs/workflow.md)
- [原生 Baseline 运行说明](docs/run_baseline.md)
- [完整依赖锁定清单](requirements.lock.txt)

本地仓库：`D:\LAB\vlm-profiling`。服务器目录：`~/projects/vlm-profiling`。Codex 只修改本地文件；用户通过 VS Code Remote SSH 手动上传并运行服务器命令。

目录用途：`scripts/` 为检查和运行入口，`src/` 为后续实现，`configs/` 为实验配置，`data/` 为输入与本地模型暂存，`results/` 为原始输出，`reports/` 为分析报告。
