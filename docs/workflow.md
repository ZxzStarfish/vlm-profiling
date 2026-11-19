# 本地开发、手动部署流程

Codex 只在 Windows 本地仓库编写代码、处理本地文件与分析结果。用户通过 VS Code Remote SSH 手动上传所列文件、在 Linux 服务器运行所列命令，并反馈完整日志；Codex 不连接 SSH、不运行远程命令。

## 当前下一步：模型权重获取准备

本地 `config.json` 请求最终 HTTP 200，模型 commit 为 `895c3a49bc3fa70a340399125c650a463535e71c`；服务器到 Hugging Face 超时。用户已在 Windows CMD 验证第一块权重的 URL 返回 HTTP 200：

```cmd
cd /d D:\LAB\vlm-profiling
curl.exe -sS -I -L --connect-timeout 10 --max-time 30 -o NUL -w "HTTP=%{http_code}\n" "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/895c3a49bc3fa70a340399125c650a463535e71c/model-00001-of-00002.safetensors"
```

Codex 已在本地下载并校验固定 commit 的 11 个模型文件，`data/MODEL_READY.txt` 已生成。下一步由用户手动上传模型与 Baseline 文件，见 [运行说明](run_baseline.md)。不需要安装新依赖。

权重下载只取[官方模型仓库](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/tree/main)的同一 commit，放在本地 `data/models/`。本地大小和两个权重 SHA-256 已校验通过。用户按 [Baseline 运行说明](run_baseline.md)手动上传到服务器本人项目目录。模型体积约 4.43GB。上传和模型加载成功前，阶段 2 Baseline 仍未通过。

## 后续每轮交付格式

Codex 提供：本地新增/修改文件、需要上传的文件、按顺序可复制的 Linux 命令、预期输出与成功标准、失败时需要反馈的日志。用户手动执行后反馈；Codex依据真实结果继续修改本地代码。Windows 本地检查和可行的代码验证由 Codex 自行完成。

服务器实际工作目录为 `~/projects/vlm-profiling`，环境为 `.venv`；运行前使用 `cd ~/projects/vlm-profiling` 和 `source .venv/bin/activate`。正式 GPU 实验前重新检查 `nvidia-smi`，选择空闲卡，避免将其他用户负载计入瓶颈数据。
