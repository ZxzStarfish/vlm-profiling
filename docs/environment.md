# 已确认的实验环境（2026-11-18）

以下只记录用户在服务器上实际返回的检查结果；本地 Codex 没有连接服务器。

| 项目 | 已确认结果 |
| --- | --- |
| 服务器目录 | `/home/xingzhezhang/projects/vlm-profiling` |
| Python | 3.12.3；通过 `venv --without-pip` 建立 `.venv`，其中 pip 24.0 |
| PyTorch | `pip freeze` 为 `torch==2.7.1`；运行时 `torch.__version__ == 2.7.1+cu126`，`torch.version.cuda == 12.6` |
| CUDA | `torch.cuda.is_available() == True`，可见 8 张 RTX 3070，每张 8192 MiB；驱动 570.211.01 |
| 关键包 | Transformers 4.51.3、Accelerate 1.7.0、Pillow 11.2.1；完整清单见 [requirements.lock.txt](../requirements.lock.txt) |
| 包检查 | `python -m pip check` 无冲突；Qwen2-VL 模型类和 Processor 类导入成功 |
| 存储 | home 所在文件系统检查时可用约 385GB；`/mnt/sdb` 挂载点不可直接写入 |
| 工具 | `nsys`、`ncu` 未发现；后续可先用 PyTorch Profiler |
| 服务器网络 | 两次连接 `huggingface.co:443` 超时；清华 PyPI 镜像可访问 |
| 本地 Windows | D: 检查时可用约 45.9GB；官方模型配置与第一块权重 URL 均返回 HTTP 200；未安装 Python，已安装 Git LFS |

本地配置文件请求返回的 `X-Repo-Commit` 是 `895c3a49bc3fa70a340399125c650a463535e71c`，模型下载固定到这个 commit。11 个所需模型文件已下载，大小符合清单，两块权重的 SHA-256 校验通过。模型文件总量约 4.43GB，见[官方文件列表](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)。

最近一次 GPU 快照按索引列出的空闲显存为 `3999, 6426, 5858, 5162, 5372, 6502, 6508, 5452` MiB；对应利用率为 `94, 25, 32, 91, 89, 39, 19, 97`%。这是瞬时状态，不能据此指定固定实验卡或判定模型显存一定足够。

**结论：** 软件环境验收通过；权重获取、完整图文推理、峰值显存和稳定计时尚未验证。安装历史用 [requirements.lock.txt](../requirements.lock.txt) 与本文件保留。
