# 三层时延实验运行说明

## 实验矩阵

复用固定版本的 Qwen2-VL-2B-Instruct 和 Baseline 软件环境。单张 RTX 3070，batch 1，FP16，使用原生贪心生成与单请求 KV Cache。慢速图像 Processor 显式固定。测试图先等比例放进 4:3 白底画布，再缩放到各尺寸。图像大小按 Processor 实际网格和 Token 数报告。

| 变量 | 取值 | 固定变量 |
| --- | --- | --- |
| 图像尺寸 | 224×168、336×252、500×375、672×504 | 同一短 prompt，输出 16 Token |
| 文本长度 | 基准短 prompt、增加 16 或 48 次相同句子 | 500×375，输出 16 Token |
| 输出长度 | 4、16、32 Token | 500×375，短 prompt |

三条轴共享 `anchor`，共 8 个 case。测试以相同的最少和最多新 Token 数限制输出长度，供受控时延比较。非视觉输入 Token 数含对话模板和特殊标记。保存实际输出 Token ID，并检查同一测法和其预热参考输出一致。

## 计时和验收

- **Level 1** 每个 case 预热 2 次、计时 7 次，按固定种子打乱各轮的 case 顺序。CPU 图像 Processor 与整个 CPU 预处理分别记录。TTFT 从处理内存图像开始，首 Token 处同步一次。ViT、LLM 主干 Prefill、逐步 Decode 使用 CUDA Event。完整请求墙钟含 CPU 预处理、输入传输和生成。
- **Level 2** 同一矩阵每个 case 运行 3 对测量。每对有一次阶段对照和一次模块探针，交替二者次序。保存每层 Block、Attention、MLP 和两个 Norm 的流时间。Block 包含子模块，Decode 总和跨所有 Decode 步。按对照检查探针开销。
- **Level 3** 在最小图、参考图、最大图及最长文本各采 2 条 PyTorch CPU/CUDA 轨迹，固定生成 5 个 Token，覆盖首次前向及 4 步 Decode。轨迹按 ViT、LLM 主干 Prefill、Decode 和外围工作标注，进一步标注 Attention、MLP、Norm 与 LM head。输出每个 ATen 算子的 self CPU、self CUDA 和 inclusive CUDA，以及 PyTorch 通过关联 ID 连接到对应 CPU 调用的 CUDA 活动。保存 gzip 压缩的 Chrome Trace，可用 Perfetto 打开。内存复制等 CUDA 活动和计算 Kernel 在轨迹中可区分。

三层时间口径分别是墙钟或 CUDA 流间隔、模块内流间隔和实际 CUDA 活动时长。它们不可直接相加。Profiler 与密集模块探针都会增加开销，延迟以 Level 1 为准。采样不包含图像打开、外部缩放和模型加载。GPU 使用情况在每部分起止采样，运行期间的短时争用仍可能存在。图像和文本都为控制输入，结果描述本模型在所测 GPU、软件版本和输入范围下的行为。

## 运行

将 `scripts/profile_stages.py`、`scripts/profile_operators.py`、`scripts/run_latency_campaign.sh` 和 `configs/latency_campaign.json` 放到服务器对应相对路径。选空闲 GPU 后，在 Linux 项目目录运行：

```bash
cd ~/projects/vlm-profiling
bash scripts/run_latency_campaign.sh 5
```

将 `5` 换成实时空闲卡的索引。脚本检查磁盘、空闲显存和活跃 GPU 计算进程，记录 GPU UUID 和输入文件 SHA-256。可附加 `stages`、`modules` 或 `trace` 单独重跑指定部分。接收的模型文件与依赖沿用 Baseline。

`CAMPAIGN_OK` 代表所有所选部分成功。每次运行创建独立的 `results/campaign.*` 目录，包含阶段和模块的原始样本、短轨迹、算子聚合、日志以及 GPU 快照。失败时优先检查对应部分的日志。

完整运行结果下载后，在本地用 Node.js 校验和汇总。脚本通过 Chrome Trace 的 CUDA 启动关联 ID 找到 Kernel 所处阶段，通过外部 ID 连接 ATen 调用。每条实际计算 Kernel 只计一次。

```cmd
node scripts/summarize_latency_campaign.mjs results/campaign.dGcY90WE reports/campaign.dGcY90WE
```

命令中的目录对应本次已完成的运行，汇总时需要相应运行目录里的本地 `.trace.json.gz`。仓库提供[已生成的汇总表](../reports/campaign.dGcY90WE/)，包括阶段中位数/IQR、模块类别归一化时长、ATen 算子、实际 CUDA Kernel 和轨迹覆盖率。Attention 的配置字段和实际模块类均应保留。本次实验视觉配置字段显示 `eager`，同配置复查确认实际视觉模块是 `VisionSdpaAttention`，轨迹也出现了对应 SDPA 算子。

## 后续 Decode 计时与 Nsight 复查

使用参考条件 `anchor`，即 500×375、234 个视觉 Token、总计 264 个输入 Token，固定输出 16 个 Token。输入预先放进 GPU，预热 3 次。主实验之外另用 672×504 采一条大图轨迹，为选测 ViT 注意力 Kernel 提供上下文。对照每次都从新请求开始，保留原生 `generate()` 和本请求的 KV Cache。

参考旧 VSR 项目通过 CUDA Profiler API 控制采集范围的做法。实验服务器使用 NVIDIA CUDA 12.8.1 官方分发清单中的 x86_64 Nsight Systems 2024.6.2.225、Nsight Compute 2025.1.1.2 归档，按清单校验 SHA-256 后解压到项目 `.cache/nsight/`。本次 CUDA/NVTX 轨迹关闭 CPU 采样和线程上下文切换采集。Nsight Compute 对选中的 Kernel 做硬件计数，关闭时钟和缓存控制，报告中的指标用于解释被采到的 Kernel 实例。[官方版本清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json) · [Nsight Systems CLI 指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [Nsight Compute CLI 指南](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

复现实验时先确认 GPU 对应索引、空闲显存和计算进程。本轮使用时段 A 的 GPU 1。以下命令从 Linux 项目目录执行，创建新的结果目录：

```bash
cd ~/projects/vlm-profiling
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
OUT=$(mktemp -d results/decode-study.XXXXXXXX)
NSYS="$PWD/.cache/nsight/nsight_systems-linux-x86_64-2024.6.2.225-archive/nsight-systems/2024.6.2/target-linux-x64/nsys"
NCU="$PWD/.cache/nsight/nsight_compute-linux-x86_64-2025.1.1.2-archive/ncu"
.venv/bin/python -u scripts/profile_decode_study.py --mode benchmark --repeats 7 --out "$OUT/benchmark.json"
.venv/bin/python -u scripts/profile_decode_study.py --mode torch-trace --repeats 2 --out "$OUT/torch-trace.json"
for ITEM in anchor.1 anchor.2 img_672.1; do
  "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop -o "$OUT/$ITEM" \
    .venv/bin/python -u scripts/profile_decode_study.py --mode nsight --case "${ITEM%.*}" --out "$OUT/$ITEM.json"
  "$NSYS" export --type sqlite --output "$OUT/$ITEM.sqlite" "$OUT/$ITEM.nsys-rep"
done
.venv/bin/python scripts/analyze_decode_timeline.py --trace "$OUT"/kineto.*.json.gz \
  --benchmark "$OUT/benchmark.json" --out "$OUT/kineto-tables"
.venv/bin/python scripts/analyze_decode_timeline.py --trace "$OUT"/anchor.*.sqlite \
  --benchmark "$OUT/benchmark.json" --out "$OUT/nsys-tables"
.venv/bin/python scripts/analyze_decode_timeline.py --trace "$OUT/img_672.1.sqlite" --out "$OUT/nsys-large-tables"
```

硬件计数每次调用只测一个匹配的 Kernel 启动。先采吞吐和占用，再单独补采两个 Kernel 的调度指标：

```bash
capture_kernel() {
  local name=$1 case_id=$2 label=$3 kernel=$4
  shift 4
  if (( $# == 0 )); then
    set -- --section SpeedOfLight --section LaunchStats --section Occupancy
  fi
  "$NCU" --target-processes application-only --profile-from-start off \
    --nvtx --nvtx-include "$label/" --kernel-name-base demangled --kernel-name "regex:$kernel" \
    --launch-count 1 --clock-control none --cache-control none "$@" --export "$OUT/$name" \
    .venv/bin/python -u scripts/profile_decode_study.py --mode nsight --case "$case_id" --out "$OUT/$name.json"
  "$NCU" --import "$OUT/$name.ncu-rep" --page details --csv > "$OUT/$name.ncu.csv"
}
capture_kernel decode-mlp-gemv anchor vlm_mlp_l14_s01 gemv2T_kernel_val
capture_kernel decode-attention-gemm anchor vlm_attention_l14_s01 cutlass_80_wmma_tensorop_f16_s161616gemm
capture_kernel vision-fmha img_672 vlm_vision_attention_l15_s00 fmha_cutlassF_f16_aligned_64x128_rf_sm80
capture_kernel vision-fmha-stalls img_672 vlm_vision_attention_l15_s00 fmha_cutlassF_f16_aligned_64x128_rf_sm80 \
  --section SchedulerStats --section WarpStateStats
capture_kernel decode-attention-gemm-stalls anchor vlm_attention_l14_s01 cutlass_80_wmma_tensorop_f16_s161616gemm \
  --section SchedulerStats --section WarpStateStats
```

成功时各请求打印 `DECODE_STUDY_OK`，分析打印 `DECODE_TIMELINE_OK`。检查两个工具产生的原始报告、CSV 和输出 Token。计时 JSON 中的 CUDA Event 区间、Profiler 的 GPU Kernel 执行时间与 CPU API 区间各自报告。若工具报错，保留完整工具输出、`gpu.csv` 和同批记录的 GPU 遥测。

## 最小 Decode 重复验证

本轮在 GPU 1 上只测 `anchor`。每组在新进程里预热 3 次，随后运行 5 次 Event 请求及对应对照。每组另启一个新进程预热并采一条相同配置的 Nsight Systems 轨迹，共 3 组。先检查 GPU 使用率、空闲显存和计算进程。

```bash
cd ~/projects/vlm-profiling
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,used_gpu_memory --format=csv
bash scripts/run_decode_repeats.sh 1
```

成功时最后打印 `DECODE_REPEATS_OK` 和结果目录。每组保留原始轨迹、按模块的 CSV、每秒的显存、频率、温度和其他计算进程数，以及开始和结束时的 CPU 负载。

下载成功目录后，在项目下用 Node.js 汇总。GPU 1 这次有完整数据，可以直接运行：

```cmd
node scripts/summarize_decode_repeats.mjs reports/decode_repeats results/decode-repeat.gpu1.ICDt8kGw
```

脚本检查模型、输入、脚本哈希、工具版本和输出 Token ID，按 GPU 分别输出组内和合并的中位数、P90、IQR。对于单次整轨迹的 Kernel 总量，P90 和 IQR 留空。本轮只用同卡结果验证短期重复性。跨卡项已取消。

如果还需查看 Attention 模块内部空档，可单独采一条细分轨迹。该轨迹有额外标注，不加入重复性统计。以下命令沿用上文的 `NSYS`、`CUDA_VISIBLE_DEVICES` 和离线设置：

```bash
DETAIL=$(mktemp -d results/attention-detail.XXXXXXXX)
"$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop -o "$DETAIL/attention" \
  .venv/bin/python -u scripts/trace_attention_detail.py --mode nsight --case anchor --out "$DETAIL/attention.json"
"$NSYS" export --type sqlite --output "$DETAIL/attention.sqlite" "$DETAIL/attention.nsys-rep"
.venv/bin/python scripts/analyze_attention_gaps.py "$DETAIL/attention.sqlite" "$DETAIL/tables"
```

报告的四幅结果图使用归档 CSV 绘制。可以在 Windows 项目目录运行：

```cmd
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File scripts/plot_latency_summary.ps1
```

生成文件为 [latency_summary.png](../reports/figures/latency_summary.png)。前三幅使用首轮阶段数据（GPU 5），第四幅使用短期 Decode 重复中位数（GPU 1）。
