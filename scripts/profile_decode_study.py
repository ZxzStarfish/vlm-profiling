"""Compare fixed-input Decode timings and capture matched PyTorch/Nsight traces."""

import argparse
import json
from contextlib import ExitStack
from random import Random
from time import perf_counter
from types import SimpleNamespace

from profile_operators import trace_module
from profile_stages import ForwardEvents, module_targets
from run_baseline import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "torch-trace", "nsight"), required=True)
    parser.add_argument("--case", choices=("anchor", "img_672"), default="anchor")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Use the campaign's Transformers 4.51.3 environment")
    baseline = json.loads(resolve_path("configs/baseline.json").read_text(encoding="utf-8"))
    campaign = json.loads(resolve_path("configs/latency_campaign.json").read_text(encoding="utf-8"))
    case = {**campaign["anchor"], **next(row for row in campaign["cases"] if row["id"] == args.case)}
    output_tokens = case["output_tokens"]
    path = resolve_path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model_dir = resolve_path(baseline["model_dir"])
    processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True,
    ).to(device).eval()
    with Image.open(resolve_path(baseline["image"])) as image:
        canvas = ImageOps.pad(image.convert("RGB"), (672, 504), color="white", method=Image.Resampling.BICUBIC)
    size = tuple(case["size"])
    frame = canvas if canvas.size == size else canvas.resize(size, Image.Resampling.BICUBIC)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": baseline["prompt"]},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[frame], return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]
    visual_tokens = inputs.input_ids.eq(model.config.image_token_id).sum().item()
    targets = [
        (family, layer, kind, module)
        for (family, layer, kind), module in module_targets(model)
        if kind in ("attention", "mlp")
    ]
    nvtx = SimpleNamespace(record_function=torch.cuda.nvtx.range)
    reference = None

    def markers(api, cleanup):
        trace_module(model, lambda call: f"vlm_step_{call:02}", api, cleanup)
        trace_module(model.visual, "vlm_vision", api, cleanup)
        for family, layer, kind, module in targets:
            prefix = ("vlm_vision_" if family == "vision" else "vlm_") + kind
            trace_module(module, lambda call, p=prefix, i=layer: f"{p}_l{i:02}_s{call:02}", api, cleanup)

    def request(mode="native", profile_path=None):
        torch.cuda.synchronize(device)
        module_events = {}
        with ExitStack() as cleanup:
            if mode == "events":
                forward_events = ForwardEvents(lambda: torch.cuda.Event(enable_timing=True))
                forward_events.attach(model, cleanup)
                for family, layer, kind, module in targets:
                    if family == "llm":
                        events = ForwardEvents(lambda: torch.cuda.Event(enable_timing=True))
                        events.attach(module, cleanup)
                        module_events[layer, kind] = events
            if mode == "nvtx" or mode == "nsight":
                markers(nvtx, cleanup)
            if mode == "torch-trace":
                markers(torch.profiler, cleanup)
                profiler = cleanup.enter_context(torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
                ]))
            if mode == "nsight":
                torch.cuda.profiler.start()
            try:
                begin = perf_counter()
                sequence = model.generate(
                    **inputs, min_new_tokens=output_tokens, max_new_tokens=output_tokens,
                    do_sample=False, use_cache=True, num_beams=1,
                )
                torch.cuda.synchronize(device)
                elapsed = (perf_counter() - begin) * 1000
            finally:
                if mode == "nsight":
                    torch.cuda.profiler.stop()

        if profile_path:
            profiler.export_chrome_trace(str(profile_path))
        tokens = sequence[0, input_tokens:].tolist()
        if len(tokens) != output_tokens or (reference is not None and tokens != reference):
            raise RuntimeError(f"Unexpected tokens in {mode}")
        row = {"mode": mode, "generation_ms": elapsed, "output_token_ids": tokens}
        if mode == "events":
            forward = forward_events.milliseconds()
            if len(forward) != output_tokens:
                raise RuntimeError("Unexpected number of forwards")
            row["decode_forward_stream_ms"] = forward[1:]
            row["modules"] = []
            for (layer, kind), events in module_events.items():
                times = events.milliseconds()
                if len(times) != output_tokens:
                    raise RuntimeError(f"Missing events in {kind} layer {layer}")
                row["modules"].append({"layer": layer, "kind": kind, "decode_stream_ms": times[1:]})
        return row

    samples = []
    with torch.inference_mode():
        for _ in range(3):
            reference = request()["output_token_ids"]
        if args.mode == "benchmark":
            rng = Random(campaign["seed"])
            for repeat in range(args.repeats):
                order = ["native", "events", "nvtx"]
                rng.shuffle(order)
                for mode in order:
                    samples.append({"repeat": repeat + 1, **request(mode)})
        elif args.mode == "torch-trace":
            for repeat in range(args.repeats):
                trace_path = path.with_name(f"kineto.{repeat + 1}.json.gz")
                samples.append({"repeat": repeat + 1, "trace": trace_path.name,
                                **request("torch-trace", trace_path)})
        else:
            samples.append({"repeat": 1, **request("nsight")})

    output = {
        "status": "ok", "mode": args.mode, "case": args.case, "image_size": size,
        "input_tokens": input_tokens, "visual_tokens": visual_tokens, "output_tokens": output_tokens,
        "model_revision": baseline["model_revision"], "samples": samples,
    }
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"DECODE_STUDY_OK {args.mode} {path}", flush=True)


if __name__ == "__main__":
    main()
