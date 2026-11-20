import { writeFileSync } from "node:fs";
import { join } from "node:path";

export function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

function quantile(values, fraction) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index);
  return sorted[lower] + (sorted[Math.ceil(index)] - sorted[lower]) * (index - lower);
}

export function csv(outputDir, file, rows) {
  const columns = Object.keys(rows[0]);
  const escape = (value) => {
    const field = String(value ?? "");
    return /[",\r\n]/u.test(field) ? '"' + field.replaceAll('"', '""') + '"' : field;
  };
  const lines = rows.map((row) => columns.map((column) => escape(row[column])).join(","));
  writeFileSync(join(outputDir, file), [columns.join(","), ...lines].join("\n") + "\n");
}

export function identity(testCase, sample) {
  return {
    id: testCase.id,
    axis: testCase.axis,
    width: testCase.image_size[0],
    height: testCase.image_size[1],
    patches: sample.vision_patch_count,
    visual_tokens: sample.visual_token_count,
    nonvisual_tokens: sample.nonvisual_input_tokens,
    llm_input_tokens: sample.llm_input_tokens,
    output_tokens: sample.generated_tokens,
    decode_steps: sample.decode_steps,
  };
}

export function validateSample(sample, reference, count) {
  ensure(sample.generated_tokens === count && sample.decode_steps === count - 1
    && sample.decode_forward_stream_ms.length === count - 1, "Output length mismatch");
  ensure(sample.generated_token_ids.join(",") === reference.generated_token_ids.join(","),
    "Generated token ID mismatch");
  ensure(sample.visual_token_count === reference.visual_token_count
    && sample.nonvisual_input_tokens === reference.nonvisual_input_tokens, "Input token mismatch");
  ensure(sample.visual_preprocess_wall_ms <= sample.cpu_preprocess_wall_ms
    && sample.ttft_wall_ms >= sample.ttft_from_generate_wall_ms
    && sample.ttft_wall_ms <= sample.request_wall_ms
    && sample.first_forward_stream_ms + 0.1 >= sample.vit_stream_ms + sample.llm_backbone_prefill_stream_ms,
  "Inconsistent timing boundaries");
}

export function stageRow(testCase) {
  const base = identity(testCase, testCase.samples[0]);
  for (const key of [
    "cpu_preprocess_wall_ms", "visual_preprocess_wall_ms", "host_to_device_wall_ms",
    "vit_stream_ms", "llm_backbone_prefill_stream_ms", "first_forward_stream_ms",
    "ttft_wall_ms", "ttft_from_generate_wall_ms", "decode_forward_total_stream_ms",
    "decode_forward_per_step_stream_ms", "request_wall_ms", "peak_allocated_mib",
  ]) {
    const values = testCase.samples.map((sample) => sample[key]);
    base[key] = quantile(values, 0.5);
    base[key + "_iqr"] = quantile(values, 0.75) - quantile(values, 0.25);
  }
  return base;
}
