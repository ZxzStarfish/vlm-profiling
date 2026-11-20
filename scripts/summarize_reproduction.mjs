import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { csv, ensure, stageRow, validateSample } from "./latency_tables.mjs";

const [baselineDir, outDir, ...pairs] = process.argv.slice(2);
if (!baselineDir || !outDir || pairs.length < 2 || pairs.length % 2 !== 0) {
  throw new Error("Usage: node scripts/summarize_reproduction.mjs BASELINE_DIR OUT_DIR SYN1 REAL1 [SYN2 REAL2 ...]");
}

const matrix = JSON.parse(readFileSync(new URL("../configs/latency_campaign.json", import.meta.url)));
const manifest = JSON.parse(readFileSync(new URL("../configs/reproduction_images.json", import.meta.url)));
const text = (dir, file) => readFileSync(join(dir, file), "utf8").trim();
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const readStages = (dir) => JSON.parse(text(dir, "stages.json"));
const hashFor = (dir, file) => text(dir, "input_sha256.txt").split(/\r?\n/u)
  .find((line) => line.slice(66) === file)?.slice(0, 64);
const baseline = readStages(baselineDir);
const originalCases = new Map(baseline.cases.map((testCase) => [testCase.id, testCase]));
const stageRows = [];
const comparisons = [];
const ttftSamples = [];

function metadata(result) {
  return [
    result.model_revision, result.dtype, result.torch_version, result.transformers_version,
    result.gpu_name, result.prompt, result.max_new_tokens,
    result.vision_attention_backend, result.llm_attention_backend, result.warmup_per_size,
    result.repeats_per_size,
  ];
}

function readGpu(dir, result) {
  const [, row] = text(dir, "gpu.csv").split(/\r?\n/u);
  const [index, uuid, name, driver] = row?.split(",").map((field) => field.trim()) ?? [];
  ensure(/^\d+$/u.test(index) && uuid === result.cuda_visible_devices && name && driver,
    `Invalid GPU record: ${dir}`);
  return { index, uuid, name, driver };
}

const originalGpu = readGpu(baselineDir, baseline);

function validateRun(dir, result, real, gpu) {
  ensure(result.status === "ok" && !result.modules_enabled && !result.trace_enabled
    && same(metadata(result), metadata(baseline))
    && gpu.name === originalGpu.name && gpu.driver === originalGpu.driver, `Run settings differ: ${dir}`);
  for (const path of ["configs/latency_campaign.json", "configs/baseline.json", "data/smoke.png"]) {
    const hash = hashFor(baselineDir, path);
    ensure(hash && hashFor(dir, path) === hash, `Configuration or reference image changed: ${path}`);
  }
  const caseIds = real
    ? manifest.images.flatMap((image) => manifest.case_ids.map((id) => `${image.id}__${id}`))
    : matrix.cases.map((testCase) => testCase.id);
  ensure(same(result.cases.map((testCase) => testCase.id), caseIds), `Wrong case matrix: ${dir}`);
  ensure(result.run_order.length === 7 && result.run_order.every(
    (round) => same([...round].sort(), [...caseIds].sort()),
  ), `Incomplete measurement order: ${dir}`);
  for (const { id, sha256 } of real ? manifest.images : [{ id: "synthetic", sha256: hashFor(baselineDir, "data/smoke.png") }]) {
    ensure(result.images[id]?.sha256 === sha256, `Source image changed: ${id}`);
  }
  for (const testCase of result.cases) {
    const reference = originalCases.get(testCase.base_case_id);
    ensure(reference && same(testCase.image_size, reference.image_size)
      && testCase.samples.length === 7, `Incomplete case or size mismatch: ${testCase.id}`);
    const previous = reference.samples[0];
    for (const sample of testCase.samples) {
      validateSample(sample, testCase.samples[0], reference.requested_output_tokens);
      ensure(same(sample.image_grid_thw, previous.image_grid_thw)
        && sample.llm_input_tokens === previous.llm_input_tokens
        && sample.nonvisual_input_tokens === previous.nonvisual_input_tokens,
      `Token budget mismatch against the original: ${testCase.id}`);
      ensure(Number.isSafeInteger(sample.started_unix_ms)
        && Number.isSafeInteger(sample.finished_unix_ms)
        && sample.finished_unix_ms >= sample.started_unix_ms, `Invalid sample timestamp: ${testCase.id}`);
    }
  }
}

function collect(dir, session, result, kind, gpu) {
  const telemetry = kind === "original" ? [] : text(dir, "gpu_telemetry.csv").split(/\r?\n/u).slice(1);
  const processCounts = telemetry.map((line) => line.split(",").at(-1).trim()).filter(
    (value) => /^\d+$/u.test(value),
  ).map(Number);
  ensure(kind === "original" || (telemetry.length > 0 && processCounts.length === telemetry.length),
    `Missing GPU process samples: ${dir}`);
  const run = {
    session, kind, dir,
    started_utc: kind === "original" ? "" : text(dir, "started_utc.txt"),
    gpu_index: gpu.index, gpu_uuid: gpu.uuid,
    same_gpu_as_original: gpu.uuid === originalGpu.uuid,
    max_compute_processes_sampled: processCounts.length ? Math.max(...processCounts) : "",
  };
  const byImage = new Map();
  for (const testCase of result.cases) {
    const sourceId = testCase.image_id ?? "synthetic";
    const baseCaseId = testCase.base_case_id ?? testCase.id;
    const stats = stageRow(testCase);
    const reference = stageRow(originalCases.get(baseCaseId));
    const row = {
      ...run, image_id: sourceId, base_case_id: baseCaseId, ...stats,
      ttft_iqr_pct_of_median: 100 * stats.ttft_wall_ms_iqr / stats.ttft_wall_ms,
      ttft_delta_ms_from_original: run.same_gpu_as_original ? stats.ttft_wall_ms - reference.ttft_wall_ms : "",
      vit_delta_ms_from_original: run.same_gpu_as_original ? stats.vit_stream_ms - reference.vit_stream_ms : "",
      prefill_delta_ms_from_original: run.same_gpu_as_original
        ? stats.llm_backbone_prefill_stream_ms - reference.llm_backbone_prefill_stream_ms : "",
    };
    stageRows.push(row);
    if (!byImage.has(sourceId)) byImage.set(sourceId, new Map());
    byImage.get(sourceId).set(baseCaseId, row);
    if (["img_224", "img_336", "anchor"].includes(baseCaseId)) {
      for (const [index, sample] of testCase.samples.entries()) {
        ttftSamples.push({
          ...run, image_id: sourceId, base_case_id: baseCaseId, round: index + 1,
          position_in_round: result.run_order[index].indexOf(testCase.id) + 1,
          started_unix_ms: sample.started_unix_ms ?? "",
          finished_unix_ms: sample.finished_unix_ms ?? "",
          ttft_wall_ms: sample.ttft_wall_ms,
        });
      }
    }
  }
  for (const [imageId, rows] of byImage) {
    const small = rows.get("img_224");
    const middle = rows.get("img_336");
    const anchor = rows.get("anchor");
    const large = rows.get("img_672");
    const longText = rows.get("text_48");
    comparisons.push({
      ...run, image_id: imageId,
      vit_big_minus_small_ms: large.vit_stream_ms - small.vit_stream_ms,
      vit_medians_increase_at_each_size: small.vit_stream_ms < middle.vit_stream_ms
        && middle.vit_stream_ms < anchor.vit_stream_ms && anchor.vit_stream_ms < large.vit_stream_ms,
      prefill_long_minus_short_ms: longText.llm_backbone_prefill_stream_ms - anchor.llm_backbone_prefill_stream_ms,
      vit_long_minus_short_text_ms: longText.vit_stream_ms - anchor.vit_stream_ms,
      decode_out32_minus_out4_ms: rows.has("out_32")
        ? rows.get("out_32").decode_forward_total_stream_ms - rows.get("out_4").decode_forward_total_stream_ms
        : "",
      small_ttft_iqr_pct: small.ttft_iqr_pct_of_median,
      middle_ttft_iqr_pct: middle.ttft_iqr_pct_of_median,
      anchor_ttft_iqr_pct: anchor.ttft_iqr_pct_of_median,
    });
  }
}

ensure(baseline.status === "ok" && originalCases.size === 8, "Invalid original campaign");
collect(baselineDir, "original", baseline, "original", originalGpu);
const labels = new Set(["original"]);
let repetitionGpuUuid;
for (let i = 0; i < pairs.length; i += 2) {
  const [syntheticDir, realDir] = pairs.slice(i, i + 2);
  const label = text(syntheticDir, "session.txt");
  ensure(label && label === text(realDir, "session.txt") && !labels.has(label),
    "Each synthetic/real pair must have the same unique session label");
  labels.add(label);
  for (const [dir, kind] of [[syntheticDir, "synthetic"], [realDir, "real"]]) {
    const result = readStages(dir);
    const gpu = readGpu(dir, result);
    repetitionGpuUuid ??= gpu.uuid;
    ensure(gpu.uuid === repetitionGpuUuid, `Reproduction sessions used different GPU UUIDs: ${dir}`);
    validateRun(dir, result, kind === "real", gpu);
    collect(dir, label, result, kind, gpu);
  }
}
mkdirSync(outDir, { recursive: true });
csv(outDir, "reproduction_stages.csv", stageRows);
csv(outDir, "reproduction_comparisons.csv", comparisons);
csv(outDir, "small_ttft_samples.csv", ttftSamples);
console.log(`REPRODUCTION_SUMMARY_OK sessions=${pairs.length / 2} ${outDir}`);
