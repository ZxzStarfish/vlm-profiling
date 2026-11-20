import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { csv, ensure, identity, stageRow, validateSample } from "./latency_tables.mjs";

const [rawDir, outputDir] = process.argv.slice(2);
if (!rawDir || !outputDir) {
  throw new Error("Usage: node scripts/summarize_latency_campaign.mjs RAW_DIR REPORT_DIR");
}

function readJson(name) {
  return JSON.parse(readFileSync(join(rawDir, name), "utf8"));
}

const stages = readJson("stages.json");
const modules = readJson("modules.json");
const operators = readJson("operators.json");
for (const result of [stages, modules, operators]) {
  ensure(result.status === "ok" && result.model_revision === stages.model_revision
    && result.cuda_visible_devices === stages.cuda_visible_devices
    && result.dtype === stages.dtype && result.torch_version === stages.torch_version
    && result.transformers_version === stages.transformers_version
    && result.vision_attention_backend === stages.vision_attention_backend
    && result.llm_attention_backend === stages.llm_attention_backend,
  "Metadata mismatch between levels");
}

ensure(stages.cases.length === 8 && modules.cases.length === 8 && operators.cases.length === 4,
  "Unexpected matrix size");
const stageByCase = new Map(stages.cases.map((testCase) => [testCase.id, testCase]));
const anchor = stageByCase.get("anchor")?.samples[0];
ensure(anchor, "Missing anchor");
for (const testCase of stages.cases) {
  ensure(testCase.samples.length === 7, "Missing stage samples");
  for (const sample of testCase.samples) {
    validateSample(sample, testCase.samples[0], testCase.requested_output_tokens);
    if (testCase.axis !== "image") ensure(sample.visual_token_count === anchor.visual_token_count, "Image varies off axis");
    if (testCase.axis !== "text") ensure(sample.nonvisual_input_tokens === anchor.nonvisual_input_tokens, "Text varies off axis");
    if (testCase.axis !== "output") ensure(sample.generated_tokens === anchor.generated_tokens, "Output varies off axis");
  }
}
for (const testCase of modules.cases) {
  const reference = stageByCase.get(testCase.id)?.samples[0];
  ensure(reference && testCase.samples.length === 3 && testCase.stage_control_samples.length === 3,
    "Missing module samples or controls");
  for (const sample of [...testCase.samples, ...testCase.stage_control_samples]) {
    validateSample(sample, reference, testCase.requested_output_tokens);
  }
  for (const sample of testCase.samples) {
    for (const [phase, layers, calls] of [["vision", 32, 1], ["prefill", 28, 1], ["decode", 28, sample.decode_steps]]) {
      const rows = sample.module_calls.filter((row) => row.phase === phase);
      const keys = new Set(rows.map((row) => [row.layer, row.kind].join("/")));
      ensure(rows.length === 5 * layers && keys.size === rows.length
        && rows.every((row) => row.call_stream_ms.length === calls), "Module call geometry mismatch");
    }
  }
}

mkdirSync(outputDir, { recursive: true });
csv(outputDir, "stages.csv", stages.cases.map(stageRow));
csv(outputDir, "modules.csv", modules.cases.flatMap((testCase) =>
  Object.entries(testCase.module_medians_total_stream_ms).flatMap(([phase, kinds]) =>
    Object.entries(kinds).map(([kind, duration]) => ({
      ...identity(testCase, testCase.samples[0]),
      phase, kind, median_total_stream_ms: duration,
      median_stream_ms_per_step: duration / (phase === "decode" ? testCase.samples[0].decode_steps : 1),
      generation_wall_ratio_vs_stage_control: testCase.generation_wall_ratio_vs_stage_control,
    })),
  ),
));

const operatorRows = [];
const kernelRows = [];
const traceRows = [];

function linkKernels(events) {
  const timed = events.filter((event) => event.ph === "X");
  const kernels = timed.filter((event) => event.cat === "kernel");
  const ranges = timed.filter((event) => event.cat === "user_annotation"
    && event.name.startsWith("vlm.")).sort((a, b) => b.ts - a.ts || a.dur - b.dur);
  const launches = new Map(timed.filter((event) =>
    ["cuda_runtime", "cuda_driver"].includes(event.cat) && event.args?.correlation != null,
  ).map((event) => [event.args.correlation, event]));
  const cpuOps = new Map(timed.filter((event) =>
    event.cat === "cpu_op" && event.args?.["External id"] != null,
  ).map((event) => [event.args["External id"], event]));
  const rows = new Map();
  for (const kernel of kernels) {
    const launch = launches.get(kernel.args?.correlation);
    const cpu = cpuOps.get(kernel.args?.["External id"]);
    const site = launch ?? cpu;
    const containing = (prefix) => site && ranges.find((range) =>
      range.tid === site.tid && range.pid === site.pid && range.name.startsWith(prefix)
      && range.ts <= site.ts && site.ts < range.ts + range.dur,
    )?.name.slice(prefix.length);
    const phase = containing("vlm.phase/") ?? "unattributed";
    const module = containing("vlm.module/") ?? "other";
    const op = cpu?.name ?? "";
    const key = JSON.stringify([phase, module, op, kernel.name]);
    const result = rows.get(key) ?? { phase, module, op, name: kernel.name, calls: 0, cuda_us: 0 };
    result.calls += 1;
    result.cuda_us += kernel.dur;
    rows.set(key, result);
  }
  return { kernels, rows: [...rows.values()] };
}

for (const testCase of operators.cases) {
  ensure(testCase.traces.length === 2 && stageByCase.has(testCase.id), "Missing trace repetitions");
  for (const [index, trace] of testCase.traces.entries()) {
    const label = { id: testCase.id, repeat: index + 1 };
    const events = JSON.parse(gunzipSync(readFileSync(join(rawDir, trace.trace)))).traceEvents;
    const { kernels, rows } = linkKernels(events);
    ensure(kernels.length > 0, "Chrome trace has no CUDA kernels");
    const linked = rows.filter((row) => row.phase !== "unattributed");
    for (const phase of ["vision", "prefill", "decode"]) {
      ensure(linked.some((row) => row.phase === phase && row.cuda_us > 0), "Missing linked phase kernel");
    }
    traceRows.push({
      ...label,
      raw_kernel_count: kernels.length,
      linked_kernel_count: linked.reduce((sum, row) => sum + row.calls, 0),
      raw_kernel_us: kernels.reduce((sum, event) => sum + event.dur, 0),
      linked_kernel_us: linked.reduce((sum, row) => sum + row.cuda_us, 0),
    });
    operatorRows.push(...trace.operators.map((row) => ({ ...label, ...row })));
    kernelRows.push(...rows.map((row) => ({ ...label, ...row })));
  }
}
csv(outputDir, "operators.csv", operatorRows);
csv(outputDir, "kernels.csv", kernelRows);
csv(outputDir, "trace_coverage.csv", traceRows);
console.log(`CAMPAIGN_SUMMARY_OK ${outputDir}`);
