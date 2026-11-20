"""Measure Qwen2-VL stages without replacing the native generation loop."""

import argparse
import json
import os
import re
from hashlib import sha256
from random import Random
from contextlib import ExitStack
from statistics import median
from time import perf_counter, time_ns
from unittest.mock import patch

from run_baseline import resolve_path


class ForwardEvents:
    def __init__(self, event_factory):
        self.event_factory = event_factory
        self.intervals = []

    def attach(self, module, cleanup):
        def before(*args):
            start = self.event_factory()
            start.record()
            self.intervals.append((start, self.event_factory()))

        def after(*args):
            self.intervals[-1][1].record()

        cleanup.callback(module.register_forward_pre_hook(before).remove)
        cleanup.callback(module.register_forward_hook(after).remove)

    def milliseconds(self):
        return [start.elapsed_time(end) for start, end in self.intervals]


def module_targets(model):
    for family, layers, members in (
        ("vision", model.visual.blocks, {
            "attention": "attn", "mlp": "mlp", "norm1": "norm1", "norm2": "norm2",
        }),
        ("llm", model.model.layers, {
            "attention": "self_attn", "mlp": "mlp", "norm1": "input_layernorm",
            "norm2": "post_attention_layernorm",
        }),
    ):
        if not layers:
            raise RuntimeError(f"No {family} blocks found")
        for layer, block in enumerate(layers):
            yield (family, layer, "block"), block
            for kind, attr in members.items():
                yield (family, layer, kind), getattr(block, attr)


def summarize_modules(timers, generated_count):
    rows = []
    totals = {}
    for (family, layer, kind), timer in timers.items():
        elapsed = timer.milliseconds()
        expected = 1 if family == "vision" else generated_count
        if len(elapsed) != expected:
            raise RuntimeError(f"{family}.{layer}.{kind}: {len(elapsed)} calls, expected {expected}")
        phases = {"vision": elapsed} if family == "vision" else {
            "prefill": elapsed[:1], "decode": elapsed[1:],
        }
        for phase, times in phases.items():
            rows.append({
                "phase": phase, "layer": layer, "kind": kind, "call_stream_ms": times,
            })
            phase_totals = totals.setdefault(phase, {})
            phase_totals[kind] = phase_totals.get(kind, 0.0) + sum(times)
    return rows, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modules", action="store_true")
    parser.add_argument("--campaign", help="JSON case matrix")
    parser.add_argument("--image-suite", help="Use a manifest of real images with selected campaign cases")
    parser.add_argument("--case", action="append", help="Run only these case IDs for a smoke test")
    parser.add_argument("--trace", action="store_true", help="Capture short CPU/CUDA traces for campaign anchors")
    parser.add_argument("--trace-repeats", type=int, default=2)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")
    if args.trace and (args.modules or not args.campaign or args.trace_repeats < 1 or not args.out):
        parser.error("--trace requires --campaign and --out, excludes --modules, and needs positive --trace-repeats")
    if args.image_suite and (not args.campaign or args.modules or args.trace or not args.out):
        parser.error("--image-suite requires --campaign and --out and currently supports stage timing only")
    if args.case and (not args.campaign or not args.out):
        parser.error("--case requires --campaign and --out")

    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import (
        AutoProcessor, Qwen2VLForConditionalGeneration, StoppingCriteria,
    )

    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Stage boundaries were checked against Transformers 4.51.3")
    config = json.loads(resolve_path("configs/baseline.json").read_text(encoding="utf-8"))
    if config["dtype"] != "float16":
        raise ValueError("Expected the Baseline float16 configuration")
    plan = None
    if args.campaign:
        plan = json.loads(resolve_path(args.campaign).read_text(encoding="utf-8"))
        case_specs = [{**plan["anchor"], **case} for case in plan["cases"]]
        if args.trace:
            case_specs = [
                {**case, "output_tokens": plan["trace_output_tokens"]}
                for case in case_specs if case["id"] in plan["trace_cases"]
            ]
    else:
        case_specs = [
            {"id": f"img_{width}x{height}", "axis": "pilot", "size": [width, height],
             "context_repeats": 0, "output_tokens": config["max_new_tokens"]}
            for width, height in ((224, 224), (500, 375))
        ]
    if not case_specs or len({case["id"] for case in case_specs}) != len(case_specs):
        raise ValueError("Campaign needs nonempty, unique case IDs")

    sources = [{"id": "synthetic", "path": config["image"]}]
    if args.image_suite:
        suite = json.loads(resolve_path(args.image_suite).read_text(encoding="utf-8"))
        sources = suite["images"]
        if not 3 <= len(sources) <= 5:
            raise ValueError("The real image suite needs three to five sources")
        suite_cases = set(suite["case_ids"])
        if not suite_cases <= {case["id"] for case in case_specs}:
            raise ValueError("Unknown case in the image suite")
        case_specs = [case for case in case_specs if case["id"] in suite_cases]
    if args.case:
        if not set(args.case) <= {case["id"] for case in case_specs}:
            raise ValueError("Unknown selected case")
        case_specs = [case for case in case_specs if case["id"] in args.case]
    source_ids = [source["id"] for source in sources]
    if not case_specs or len(set(source_ids)) != len(sources):
        raise ValueError("Need at least one case and unique image IDs")

    images = {}
    image_metadata = {}
    for source in sources:
        image_id = source["id"]
        if not re.fullmatch(r"[a-z0-9_]+", image_id):
            raise ValueError(f"Invalid image ID: {image_id}")
        image_path = resolve_path(source["path"])
        digest = sha256(image_path.read_bytes()).hexdigest()
        if args.image_suite and digest != source["sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {image_path}")
        with Image.open(image_path) as original:
            image = (ImageOps.exif_transpose(original) if args.image_suite else original).convert("RGB")
        image_metadata[image_id] = {
            "path": source["path"], "sha256": digest, "source_size": list(image.size),
        }
        # Fit each picture onto the same master canvas as the original campaign.
        images[image_id] = (
            ImageOps.pad(image, (672, 504), color="white", method=Image.Resampling.BICUBIC) if plan else image
        )

    device = torch.device("cuda:0")
    free_before_load, _ = torch.cuda.mem_get_info(device)
    model_dir = resolve_path(config["model_dir"])
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    path = resolve_path(args.out or ("results/modules.json" if args.modules else "results/stages.json"))
    raw_preprocess = processor.image_processor.preprocess
    image_preprocess_ms = []

    def timed_image_preprocess(*args, **kwargs):
        begin = perf_counter()
        result = raw_preprocess(*args, **kwargs)
        image_preprocess_ms.append((perf_counter() - begin) * 1000)
        return result

    class FirstToken(StoppingCriteria):
        wall_time = None

        def __call__(self, input_ids, scores, **kwargs):
            if self.wall_time is None:
                # The greedy loop appends the selected token before calling this callback.
                torch.cuda.current_stream(device).synchronize()
                self.wall_time = perf_counter()
            return torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

    def run_once(workload, measure, measure_modules=False):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        free_before_request, _ = torch.cuda.mem_get_info(device)
        started_unix_ms = time_ns() // 1_000_000
        start = perf_counter()
        image_preprocess_ms.clear()
        prompt = processor.apply_chat_template(workload["messages"], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[workload["frame"]], return_tensors="pt")
        preprocess_end = perf_counter()
        if len(image_preprocess_ms) != 1:
            raise RuntimeError(f"Expected one image preprocess call, got {len(image_preprocess_ms)}")
        grid = inputs.image_grid_thw[0].tolist()
        patches = grid[0] * grid[1] * grid[2]
        visual_tokens = patches // model.config.vision_config.spatial_merge_size**2
        prompt_tokens = inputs.input_ids.shape[1]
        if inputs.input_ids.eq(model.config.image_token_id).sum().item() != visual_tokens:
            raise RuntimeError("Visual token count does not match prompt placeholders")
        transfer_start = perf_counter()
        inputs = inputs.to(device)
        torch.cuda.synchronize(device)
        transfer_end = perf_counter()

        first_token = FirstToken()
        timers = {
            name: ForwardEvents(lambda: torch.cuda.Event(enable_timing=True))
            for name in ("forward", "vit", "llm")
        }
        module_timers = {}
        with ExitStack() as cleanup:
            criteria = {}
            if measure:
                for name, module in (("forward", model), ("vit", model.visual), ("llm", model.model)):
                    timers[name].attach(module, cleanup)
                criteria["stopping_criteria"] = [first_token]
            if measure_modules:
                for key, module in module_targets(model):
                    timer = ForwardEvents(lambda: torch.cuda.Event(enable_timing=True))
                    timer.attach(module, cleanup)
                    module_timers[key] = timer
            generation_start = perf_counter()
            sequence = model.generate(**inputs, **workload["generation"], **criteria)
            torch.cuda.synchronize(device)
            generation_end = perf_counter()
        finished_unix_ms = time_ns() // 1_000_000

        new_ids = sequence[0, prompt_tokens:].tolist()
        if plan and len(new_ids) != workload["output_tokens"]:
            raise RuntimeError(f"Expected {workload['output_tokens']} output tokens, got {len(new_ids)}")
        if not measure:
            return new_ids
        forward, vit, llm = (timers[name].milliseconds() for name in ("forward", "vit", "llm"))
        if (
            not new_ids or len(vit) != 1 or len(forward) != len(new_ids)
            or len(llm) != len(new_ids) or first_token.wall_time is None
        ):
            raise RuntimeError(
                f"Unexpected stage counts: vision={len(vit)}, forward={len(forward)}, "
                f"llm={len(llm)}, generated={len(new_ids)}"
            )
        sample = {
            "started_unix_ms": started_unix_ms,
            "finished_unix_ms": finished_unix_ms,
            "cpu_preprocess_wall_ms": (preprocess_end - start) * 1000,
            "visual_preprocess_wall_ms": image_preprocess_ms[0],
            "host_to_device_wall_ms": (transfer_end - transfer_start) * 1000,
            "vit_stream_ms": vit[0],
            "first_forward_stream_ms": forward[0],
            "llm_backbone_prefill_stream_ms": llm[0],
            "decode_forward_stream_ms": forward[1:],
            "decode_forward_total_stream_ms": sum(forward[1:]),
            "decode_forward_per_step_stream_ms": median(forward[1:]) if len(forward) > 1 else None,
            "decode_steps": len(forward) - 1,
            "ttft_wall_ms": (first_token.wall_time - start) * 1000,
            "ttft_from_generate_wall_ms": (first_token.wall_time - generation_start) * 1000,
            "generation_wall_ms": (generation_end - generation_start) * 1000,
            "request_wall_ms": (generation_end - start) * 1000,
            "free_before_request_mib": free_before_request / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "image_grid_thw": grid,
            "vision_patch_count": patches,
            "visual_token_count": visual_tokens,
            "llm_input_tokens": prompt_tokens,
            "nonvisual_input_tokens": prompt_tokens - visual_tokens,
            "generated_tokens": len(new_ids),
            "generated_token_ids": new_ids,
        }
        if measure_modules:
            sample["module_calls"], sample["module_totals_stream_ms"] = summarize_modules(
                module_timers, len(new_ids),
            )
        return sample

    workloads = []
    for image_id, spec in ((image_id, spec) for image_id in images for spec in case_specs):
        image = images[image_id]
        case_id = spec["id"]
        size = tuple(spec["size"])
        question = config["prompt"]
        if spec["context_repeats"]:
            question = ((plan["context_sentence"] + " ") * spec["context_repeats"]) + question
        generation = dict(max_new_tokens=spec["output_tokens"], do_sample=False, num_beams=1, use_cache=True)
        if plan:
            generation["min_new_tokens"] = spec["output_tokens"]
        workloads.append({
            **spec,
            "id": f"{image_id}__{case_id}" if args.image_suite else case_id,
            "base_case_id": case_id,
            "image_id": image_id,
            "frame": image if size == image.size else image.resize(size, Image.Resampling.BICUBIC),
            "messages": [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": question},
            ]}],
            "generation": generation,
            "samples": [],
            "controls": [],
        })

    cases = []
    with torch.inference_mode(), patch.object(processor.image_processor, "preprocess", timed_image_preprocess):
        for workload in workloads:
            for _ in range(args.warmup):
                workload["reference"] = run_once(workload, measure=False)

        if args.trace:
            from profile_operators import profile_generation

            for workload in workloads:
                traces = []
                for trace_index in range(args.trace_repeats):
                    trace_path = path.parent / f"{workload['id']}.{trace_index + 1}.trace.json.gz"
                    traces.append(profile_generation(
                        torch, model, lambda: run_once(workload, measure=False),
                        module_targets(model), trace_path, workload["reference"],
                    ))
                cases.append({
                    "id": workload["id"], "image_size": workload["size"],
                    "output_tokens": workload["output_tokens"],
                    "context_repeats": workload["context_repeats"], "traces": traces,
                })
                print(f"{workload['id']}: {len(traces)} CUDA traces", flush=True)

        rng = Random(plan["seed"]) if plan else None
        run_order = []
        for index in range(0 if args.trace else args.repeats):
            order = list(workloads)
            if rng:
                rng.shuffle(order)
            run_order.append([workload["id"] for workload in order])
            for workload in order:
                samples, controls = workload["samples"], workload["controls"]
                if args.modules and index % 2 == 0:
                    controls.append(run_once(workload, measure=True))
                samples.append(run_once(workload, measure=True, measure_modules=args.modules))
                if args.modules and index % 2 == 1:
                    controls.append(run_once(workload, measure=True))
                print(f"repeat {index + 1}/{args.repeats}: {workload['id']}", flush=True)

        for workload in ([] if args.trace else workloads):
            size = workload["size"]
            samples, stage_controls = workload["samples"], workload["controls"]
            reference = workload["reference"]
            if any(row["generated_token_ids"] != reference for row in samples + stage_controls):
                raise RuntimeError(f"Instrumented generation differs from warmup at {size}")
            medians = {
                key: median(row[key] for row in samples)
                for key in samples[0]
                if key.endswith(("_wall_ms", "_stream_ms", "_mib"))
                and isinstance(samples[0][key], (int, float))
            }
            case = {
                "id": workload["id"],
                "base_case_id": workload["base_case_id"],
                "image_id": workload["image_id"],
                "axis": workload["axis"],
                "context_repeats": workload["context_repeats"],
                "requested_output_tokens": workload["output_tokens"],
                "image_size": size,
                "generated_text": processor.decode(reference, skip_special_tokens=True),
                "medians": medians,
                "samples": samples,
            }
            if args.modules:
                case["stage_control_samples"] = stage_controls
                case["module_medians_total_stream_ms"] = {
                    phase: {
                        kind: median(sample["module_totals_stream_ms"][phase][kind] for sample in samples)
                        for kind in kinds
                    }
                    for phase, kinds in samples[0]["module_totals_stream_ms"].items()
                }
                control_ms = median(row["generation_wall_ms"] for row in stage_controls)
                case["stage_control_median_generation_wall_ms"] = control_ms
                case["generation_wall_ratio_vs_stage_control"] = medians["generation_wall_ms"] / control_ms
                print(f"{size}: module hooks / stage-only generation wall time "
                      f"{case['generation_wall_ratio_vs_stage_control']:.2f}x", flush=True)
            cases.append(case)
            print(f"{workload['id']} {size}: {samples[0]['visual_token_count']} visual tokens, "
                  f"median TTFT {medians['ttft_wall_ms']:.2f} ms", flush=True)

    output = {
        "status": "ok",
        "model_revision": config["model_revision"],
        "dtype": config["dtype"],
        "prompt": config["prompt"],
        "max_new_tokens": config["max_new_tokens"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "free_before_load_mib": free_before_load / 2**20,
        "llm_attention_backend": model.config._attn_implementation,
        "vision_attention_backend": model.config.vision_config._attn_implementation,
        "llm_attention_classes": sorted({type(layer.self_attn).__name__ for layer in model.model.layers}),
        "vision_attention_classes": sorted({type(block.attn).__name__ for block in model.visual.blocks}),
        "warmup_per_size": args.warmup,
        "repeats_per_size": args.repeats,
        "modules_enabled": args.modules,
        "trace_enabled": args.trace,
        "campaign": args.campaign,
        "image_suite": args.image_suite,
        "images": image_metadata,
        "run_order": run_order,
        "cases": cases,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    marker = "TRACE_OK" if args.trace else ("MODULES_OK" if args.modules else "STAGES_OK")
    print(f"{marker} {path}")


if __name__ == "__main__":
    main()
