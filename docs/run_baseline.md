# 第一轮原生图文 Baseline

当前代码只做功能和显存 Smoke Test：使用官方 Processor、本地原始 Qwen2-VL-2B-Instruct 权重、单 GPU FP16、batch 1。输出图文回答、实际视觉网格/Token 数、总生成墙钟时间及本进程峰值显存。**尚未测量独立 ViT、Prefill、Decode 或 TTFT**；这些在 Baseline 成功后实现。

## 先决条件

1. 等本地 `data/MODEL_READY.txt` 出现，表示 11 个模型文件已下载，两个权重分片完成 SHA-256 校验。可在 Windows CMD 运行 `type D:\LAB\vlm-profiling\data\MODEL_READY.txt` 检查；文件不存在就继续等待，查看 `data/model_weights.log` 和 `data/model_weights_error.log`。
2. 用户手动把本地 `data/models/Qwen2-VL-2B-Instruct/895c3a49bc3fa70a340399125c650a463535e71c/` 中的 11 个文件上传到服务器相同的项目相对路径。
3. 手动上传 `scripts/run_baseline.py`、`scripts/check_model_files.sh`、`configs/baseline.json`、`data/smoke.png`。不上传本地下载脚本、`.venv`、缓存或 `data/model_manifest.json`。
4. 在服务器上选一张尽可能空闲的 RTX 3070；其他用户任务可能导致 OOM 或不稳定计时。

## Linux 远程终端顺序

上传前先创建本人项目目录：

```bash
cd ~/projects/vlm-profiling
mkdir -p data/models/Qwen2-VL-2B-Instruct/895c3a49bc3fa70a340399125c650a463535e71c configs scripts results
```

上传完成后执行：

```bash
cd ~/projects/vlm-profiling
bash scripts/check_model_files.sh
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
```

第一条验证两个权重分片的 SHA-256 以及必需配置文件。第二条查看瞬时 GPU 状态。根据实际输出选择 GPU，并在下面命令的提示中输入编号；不应直接沿用历史快照。

```bash
read -rp '输入当前空闲 GPU 编号: ' GPU_ID
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU_ID" .venv/bin/python scripts/run_baseline.py --config configs/baseline.json --out results/baseline.json 2>&1 | tee results/baseline.log
```

运行前请确认 GPU_ID 对应的是当前允许使用的空闲卡。运行可能因显存不足失败；不要自动改用 CPU Offload、量化或多卡。输出文件 `results/baseline.json` 的 `status` 应为 `ok`，`image_placeholder_count` 应等于 `visual_token_count`，`generated_text` 应描述左侧红色方形和右侧蓝色圆形。请反馈完整日志和 JSON。

如果失败，请反馈 `results/baseline.log`、完整异常或 `cuda_oom` JSON、当时的 `nvidia-smi` 输出，以及模型文件校验输出。当前脚本的时延仅用于检查执行路径，不作为论文中的性能结论。
