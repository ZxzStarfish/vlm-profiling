import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { csv, ensure } from "./latency_tables.mjs";

const [outDir, ...runDirs] = process.argv.slice(2);
if (!outDir || !runDirs.length) {
  throw new Error("Usage: node scripts/summarize_decode_repeats.mjs OUT_DIR GPU_RUN_DIR [GPU_RUN_DIR ...]");
}

const text = (dir, name) => readFileSync(join(dir, name), "utf8").trim();
const load = (dir, name) => JSON.parse(text(dir, name));

// These CUDA and nvidia-smi exports contain only simple identifiers and numbers.
function readCsv(dir, file) {
  const [header, ...body] = text(dir, file).split(/\r?\n/u);
  const names = header.split(",").map((name) => name.trim());
  return body.map((line) => {
    const fields = line.split(",").map((field) => field.trim());
    ensure(fields.length === names.length, `Malformed CSV: ${dir}/${file}`);
    return Object.fromEntries(names.map((name, idx) => [name, fields[idx]]));
  });
}

function percentile(values, p) {
  const sorted = [...values].sort((a, b) => a - b);
  ensure(sorted.length && sorted.every(Number.isFinite), "Missing or nonnumeric measurement");
  const position = (sorted.length - 1) * p;
  const lower = Math.floor(position);
  return sorted[lower] + (sorted[Math.ceil(position)] - sorted[lower]) * (position - lower);
}

function stats(values) {
  return {
    n: values.length, median_ms: percentile(values, 0.5),
    p90_ms: values.length > 1 ? percentile(values, 0.9) : "",
    iqr_ms: values.length > 1 ? percentile(values, 0.75) - percentile(values, 0.25) : "",
    min_ms: percentile(values, 0), max_ms: percentile(values, 1),
  };
}

const sum = (values) => values.reduce((a, b) => a + b, 0);
const metadata = (run) => JSON.stringify([
  run.case, run.image_size, run.input_tokens, run.visual_tokens, run.output_tokens, run.model_revision,
]);
const kinds = ["attention", "mlp"];
const measures = ["event_request_total", "event_step", "kernel_trace_total", "kernel_step"];
const metrics = [];
const ranks = [];
const telemetry = [];
const knownGpuUuids = new Set();
const hashes = text(runDirs[0], "input_sha256.txt");
const profilerVersion = text(runDirs[0], "nsys_version.txt");
const [targetGpu] = readCsv(runDirs[0], "gpu.csv");
let referenceShape;
let referenceTokens;

for (const runDir of runDirs) {
  const [gpu] = readCsv(runDir, "gpu.csv");
  ensure(!knownGpuUuids.has(gpu.uuid), `Repeated GPU UUID: ${gpu.uuid}`);
  knownGpuUuids.add(gpu.uuid);
  ensure(gpu.name === targetGpu.name && gpu.driver_version === targetGpu.driver_version
    && text(runDir, "input_sha256.txt") === hashes
    && text(runDir, "nsys_version.txt") === profilerVersion, `Tool, device, or input mismatch: ${runDir}`);
  const ident = { gpu_index: gpu.index, gpu_uuid: gpu.uuid, run_dir: runDir };
  const aggregate = Object.fromEntries(kinds.flatMap((kind) =>
    measures.map((measure) => [`${kind}/${measure}`, []]),
  ));

  for (const group of [1, 2, 3]) {
    const dir = join(runDir, `group-${group}`);
    const benchmark = load(dir, "benchmark.json");
    const capture = load(dir, "anchor.json");
    referenceShape ??= metadata(benchmark);
    referenceTokens ??= JSON.stringify(benchmark.samples[0].output_token_ids);
    for (const result of [benchmark, capture]) {
      ensure(result.status === "ok" && metadata(result) === referenceShape
        && result.samples.every((row) => JSON.stringify(row.output_token_ids) === referenceTokens),
      `Workload or output tokens changed: ${dir}`);
    }
    const timed = benchmark.samples.filter((sample) => sample.mode === "events");
    ensure(benchmark.input_tokens === 264 && benchmark.output_tokens === 16 && timed.length === 5
      && capture.samples.length === 1, `Wrong number of requests or tokens: ${dir}`);
    const summary = readCsv(dir, "timeline/decode_timeline.csv");
    ensure(summary.length === 2 && summary.every((row) => Number(row.calls) === 420
      && Number(row.api_linked_kernels) === Number(row.gpu_kernels) && Number(row.gpu_kernels) > 0),
    `Incomplete Nsight association: ${dir}`);
    const modules = readCsv(dir, "timeline/modules_timeline.csv").filter((row) => row.phase === "decode");
    const local = {};
    for (const kind of kinds) {
      const requestSteps = timed.map((sample) => {
        const layers = sample.modules.filter((layer) => layer.kind === kind);
        ensure(layers.length === 28 && layers.every((layer) => layer.decode_stream_ms.length === 15),
          `Incomplete CUDA Event samples: ${dir}`);
        return Array.from({ length: 15 }, (_, step) => sum(layers.map((layer) => layer.decode_stream_ms[step])));
      });
      const perStep = Array.from({ length: 15 }, (_, step) => {
        const scoped = modules.filter((row) => row.kind === kind && Number(row.step) === step + 1);
        ensure(scoped.length === 28, `Incomplete Nsight Decode steps: ${dir}`);
        return sum(scoped.map((row) => Number(row.gpu_kernel_us))) / 1000;
      });
      const trace = summary.find((row) => row.kind === kind);
      ensure(Math.abs(sum(perStep) - Number(trace.gpu_kernel_us) / 1000) < 0.01,
        `Kernel sums do not reconcile: ${dir}`);
      local[kind] = {
        event_request_total: requestSteps.map(sum),
        event_step: requestSteps.flat(),
        kernel_trace_total: [sum(perStep)],
        kernel_step: perStep,
      };
      for (const measure of measures) {
        const values = local[kind][measure];
        aggregate[`${kind}/${measure}`].push(...values);
        metrics.push({ ...ident, group, kind, measure, ...stats(values) });
      }
    }
    const attention = summary.find((row) => row.kind === "attention");
    const mlp = summary.find((row) => row.kind === "mlp");
    ranks.push({
      ...ident, group,
      event_attention_gt_mlp_count: local.attention.event_request_total.filter(
        (value, i) => value > local.mlp.event_request_total[i],
      ).length,
      event_requests: timed.length,
      kernel_mlp_gt_attention: Number(mlp.gpu_kernel_us) > Number(attention.gpu_kernel_us),
      attention_trace_idle_ms: Number(attention.gpu_idle_inside_span_us) / 1000,
      attention_idle_before_api_ms: Number(attention.gpu_idle_before_next_submission_us) / 1000,
      attention_idle_before_api_pct: 100 * Number(attention.gpu_idle_before_next_submission_us)
        / Number(attention.gpu_idle_inside_span_us),
      attention_launch_api_ms: Number(attention.cpu_launch_api_us) / 1000,
      mlp_trace_idle_ms: Number(mlp.gpu_idle_inside_span_us) / 1000,
      interstep_sync_ms: Number(attention.request_interstep_sync_or_blocking_copy_us) / 1000,
    });
    const env = readCsv(dir, "gpu_telemetry.csv");
    const active = env.filter((row) => Number(row.compute_processes) > Number(row.other_compute_processes));
    ensure(active.length > 0, `Telemetry missed the workload: ${dir}`);
    const estimate = (rows, key, fraction) => percentile(rows.map((row) => Number(row[key])), fraction);
    telemetry.push({
      ...ident, group, started_utc: text(dir, "started_utc.txt"), samples: env.length, active_samples: active.length,
      temperature_min_c: estimate(active, "temperature_c", 0),
      temperature_max_c: estimate(active, "temperature_c", 1),
      sm_clock_min_mhz: estimate(active, "sm_clock_mhz", 0),
      sm_clock_median_mhz: estimate(active, "sm_clock_mhz", 0.5),
      sm_clock_max_mhz: estimate(active, "sm_clock_mhz", 1),
      memory_clock_median_mhz: estimate(active, "memory_clock_mhz", 0.5),
      used_peak_mib: estimate(active, "used_mib", 1),
      other_compute_processes_max: estimate(env, "other_compute_processes", 1),
      other_compute_mib_max: estimate(env, "other_compute_mib", 1),
    });
  }
  for (const kind of kinds) {
    for (const measure of measures) {
      metrics.push({ ...ident, group: "all", kind, measure, ...stats(aggregate[`${kind}/${measure}`]) });
    }
  }
}

mkdirSync(outDir, { recursive: true });
csv(outDir, "decode_statistics.csv", metrics);
csv(outDir, "decode_rankings.csv", ranks);
csv(outDir, "gpu_environment.csv", telemetry);
console.log(`DECODE_REPEAT_SUMMARY_OK gpus=${runDirs.length} groups=${ranks.length} ${outDir}`);
