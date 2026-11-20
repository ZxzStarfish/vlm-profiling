# 复现性验证运行说明

这一步验证首轮三层实验看到的趋势能否在不同时间和不同图像内容下重现。沿用原阶段计时脚本、Qwen2-VL-2B 模型版本、RTX 3070、FP16、batch 1、提示词和 Processor。运行期间每约 2 秒记录 GPU 使用率、空闲显存、SM 时钟、温度、功率和计算进程数，不保存进程 ID。阶段 JSON 中每次测量有时间戳，可与遥测对照。GPU 利用率包含本实验自己的工作，采样间隔内的短时负载可能漏掉。

## 两个时段，每个时段做两组对照

A/B 两次运行都使用 GPU 1，汇总按 UUID 核对。首轮三层实验用 GPU 5，保留为趋势参考。A/B 分别启动，记录会话标签和 UTC 实际时间。当前实验采用同日短期复测，无需人为设定等待间隔。每次运行均完成以下两个部分。

| 部分 | 输入 | 如何与首轮对比 |
| --- | --- | --- |
| 合成图重跑 | 原图、原八组矩阵、原提示词和原输出长度 | 对照首轮趋势，在同一张复现用 GPU 上逐组比较两时段的中位数和 IQR |
| 真实图验证 | 宇航员肖像、猫、咖啡和火箭发射，每张测 224×168、336×252、500×375、672×504 和 500×375 的长文本组，共 20 组 | 把照片按原长宽比放到同一大小的白底画布，再缩放到目标尺寸。每个尺寸都核对视觉网格、实际输入 Token 数和目标输出 Token 数与对应的首轮样本一致 |

照片的原始宽×高分别为 512×512、451×300、600×400、640×427，照片内容的变化与输入尺寸的变化分别比较。真实图验证仍使用原短提示词和重复句，目的是匹配 Token 预算，生成文本仅用于检查运行一致性。每组预热 2 次，正式计时 7 次。每轮按原配置的随机种子打乱各组执行顺序。照片来自 scikit-image 0.25.2 的公开样例，其 SHA-256 固定在 [照片清单](../configs/reproduction_images.json)。[样例来源与署名](../data/reproduction/README.md)

## 判读方法

- **ViT 随图像变大。** 每个时段分别看合成图和各张照片的四个尺寸，先看最大图与最小图的耗时差，再检查中间两组是否也符合递增趋势。保留逐图差异。
- **Prefill 随文本变长。** 比较同一张 500×375 图的短、长文本组，检查 Prefill 的差值和 ViT 对照的差值。输出长度固定。
- **复查小图 TTFT 波动。** 比较两个小图尺寸和参考图的 TTFT IQR、中位数以及 IQR 占中位数的比例，说明高波动出现在哪个输入和会话。若遥测出现额外计算进程或频率、温度变化，标记相应时段并保留原始样本。遥测用于排查线索，原因需单独验证。

合成图看每个会话的方向是否一致。真实图按“每个会话有几张支持该趋势”报告，例如 `3/4`，并列出例外。样本量按照片和时段说明。7 次请求是同一条件下的重复测量，不能当成 7 张照片。

## A/B 实测结果（2026-11-20）

GPU 1 的 A 和 B 各完成合成图 8 组、真实图 20 组，共 392 次正式测量。A 于北京时间 11:37 启动，B 于 12:38 启动。汇总核对了设备 UUID、输入清单、实际网格和 Token 数。A 和 B 的 GPU 遥测分别记录 85 和 73 个采样点，均只观察到最多一个计算进程。

| 观察指标 | A 合成图 | B 合成图 | A 四张真实图 | B 四张真实图 |
| --- | ---: | ---: | ---: | ---: |
| ViT 最大图减最小图的中位数差 | +103.12 ms | +107.90 ms | +102.58 至 +103.10 ms | +108.00 至 +108.63 ms |
| 500×375 长文本减短文本的 Prefill 中位数差 | +39.72 ms | +39.72 ms | +39.42 至 +39.73 ms | +39.82 至 +39.95 ms |
| 224×168 的 TTFT 四分位距 | 1.43 ms | 12.33 ms | 0.38–0.74 ms | 0.70–1.41 ms |

两个会话中，合成图和四张真实图的 ViT 中位数均随四个尺寸逐级增加。固定 500×375 图像后，长文本带来的 Prefill 增量约为 40 ms，ViT 变化不足 1 ms。两次合成图测试也都看到输出从 4 增至 32 个 Token 后总时延增加，TTFT 仍在约 123–125 ms。

合成小图的大波动在 B 再次出现。7 次 TTFT 分布在约 54.7–57.0 ms 和 66.9–67.8 ms 两段，ViT 和 LLM Prefill 耗时也一起变化。四张真实小图的组内波动仍较小，但其 TTFT 中位数从 A 的 67.42–67.78 ms 降到 B 的 51.46–51.91 ms。说明即使使用同一张 GPU，绝对时延仍会随运行条件变化。现有 GPU 遥测每约 2 秒采样一次，无法解释几十毫秒尺度的波动。A/B 只支持同日、同卡下的趋势复查。

[趋势对照 CSV](../reports/reproduction/reproduction_comparisons.csv) · [各组阶段汇总](../reports/reproduction/reproduction_stages.csv) · [小图 TTFT 逐次样本](../reports/reproduction/small_ttft_samples.csv) · [A 合成图](../results/repro-synthetic.session-a.AswQp2Ic/) · [A 真实图](../results/repro-real.session-a.7Yy2mUcX/) · [B 合成图](../results/repro-synthetic.session-b.xLNlbYPY/) · [B 真实图](../results/repro-real.session-b.mFDELn72/)

## Linux 服务器命令

源文件同步后进入服务器项目目录。本轮复现用 GPU 1。脚本会在每次开始前检查磁盘、空闲显存和当前计算进程。

```bash
cd ~/projects/vlm-profiling
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
bash scripts/run_latency_campaign.sh 1 repro-synthetic session-a &&
bash scripts/run_latency_campaign.sh 1 repro-real session-a
```

第二次用不同标签，并把两部分的顺序对调。

```bash
cd ~/projects/vlm-profiling
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
bash scripts/run_latency_campaign.sh 1 repro-real session-b &&
bash scripts/run_latency_campaign.sh 1 repro-synthetic session-b
```

每次成功时先出现 `STAGES_OK`，再出现 `CAMPAIGN_OK`。记下终端打印的 `RUN_DIR`。其中 `stages.json` 的合成图运行应有 8 组，真实图运行应有 20 组，每组各 7 条正式样本。照片、脚本或依赖出错时先保留终端完整输出。若已经创建运行目录，取回 `stages.log`、`gpu_telemetry.log`、`gpu_telemetry.csv` 和 `stages_gpu_before.csv`。Token 核验失败时同时提供该次的 `stages.json` 和 `input_sha256.txt`。

## 本地汇总

通过 SSH 将四个 `RUN_DIR` 下载到本地 `results/`。如果手动下载，可在 Windows PowerShell 执行以下命令。先把 `$remoteRun` 改成服务器打印的路径，每个运行目录执行一次。

```powershell
# 在本地仓库根目录运行。替换服务器登录目标和 RUN_DIR。
$remoteRun = 'results/替换为RUN_DIR中的目录名'
scp -r "user@gpu-server:projects/vlm-profiling/$remoteRun" results/
```

汇总命令中的四个值也要替换为实际目录，参数按每个会话的合成图、真实图顺序排列。

```cmd
set "SYN_A=results/填写合成图目录A"
set "REAL_A=results/填写真实图目录A"
set "SYN_B=results/填写合成图目录B"
set "REAL_B=results/填写真实图目录B"
node scripts/summarize_reproduction.mjs results/campaign.dGcY90WE reports/reproduction "%SYN_A%" "%REAL_A%" "%SYN_B%" "%REAL_B%"
```

汇总会核对原实验配置哈希、模型与软件版本、GPU 型号和驱动、照片哈希、尺寸、实际 Token 数和样本完整性。两个复现时段必须使用相同的 GPU UUID。若复现卡与首轮不同，CSV 会标明两张卡，并将相对首轮的绝对耗时差留空。成功时打印 `REPRODUCTION_SUMMARY_OK` 并生成阶段汇总、趋势对照、小图 TTFT 逐样本 CSV。只有一对运行时也可汇总，命令省略 B 的两个目录，结论留到第二个时段完成后再写。
