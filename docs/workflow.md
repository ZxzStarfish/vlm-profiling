# 本地分析与服务器部署

实验在 Linux GPU 服务器运行，本地负责校验、汇总和生成图表。服务器模型目录由 `configs/baseline.json` 指定，Python 环境位于项目 `.venv`。准备[模型清单](../data/model_manifest.json)中的 11 个文件后，运行 `bash scripts/check_model_files.sh` 校验必需文件和两块权重的 SHA-256。

- [环境与软件版本](environment.md)
- [原生图文 Baseline](run_baseline.md)
- [三层采集与 Nsight 分析](run_latency_campaign.md)
- [A/B 复测与本地汇总](run_reproduction.md)

服务器脚本在 `results/` 下生成独立运行目录。将对应目录下载到本地同名路径后，可用运行说明中的命令重算汇总。仓库提供逐次测量和导出表，原始轨迹、Nsight 二进制文件和日志由运行者本地留存。
