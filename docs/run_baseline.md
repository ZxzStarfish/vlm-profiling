# 第一轮原生图文 Baseline

Baseline 使用官方 Processor、本地原始 Qwen2-VL-2B-Instruct 权重、单 GPU FP16 和 batch 1，检查图文回答、实际视觉网格与 Token 数、总生成时间和峰值显存。ViT、Prefill、Decode 和 TTFT 的独立计时见[三层时延实验](run_latency_campaign.md)。

## 先决条件

1. 按 [模型清单](../data/model_manifest.json) 准备固定版本的 11 个文件，并放在 `configs/baseline.json` 指定的模型目录。
2. 按 [环境记录](environment.md) 准备服务器 `.venv`，并在服务器进入仓库根目录。
3. 从允许使用的 GPU 中选一张空闲的 RTX 3070。

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
