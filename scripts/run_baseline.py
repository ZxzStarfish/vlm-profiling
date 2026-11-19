"""Smoke test the unmodified Qwen2-VL model on one local image."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args()

    config = json.loads(resolve_path(args.config).read_text(encoding="utf-8"))
    model_dir = resolve_path(config["model_dir"])
    image_path = resolve_path(config["image"])
    output_path = resolve_path(args.out)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory missing: {model_dir}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing: {image_path}")
    if config["dtype"] != "float16":
        raise ValueError("This baseline is pinned to float16")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    print(f"GPU: {gpu_name}; free before load: {free_before // 2**20} MiB")
    print(f"Model: {model_dir}")

    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.float16,
            low_cpu_mem_usage=True, local_files_only=True,
        ).to(device)
    except torch.cuda.OutOfMemoryError as exc:
        failure = {
            "status": "cuda_oom_during_load",
            "error": str(exc),
            "model_dir": str(model_dir),
            "gpu_name": gpu_name,
            "free_before_load_mib": free_before / 2**20,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    model.eval()
    torch.cuda.synchronize()
    free_after_load, _ = torch.cuda.mem_get_info(device)
    print(f"Free after load: {free_after_load // 2**20} MiB")

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": config["prompt"]},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    start = time.perf_counter()
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    preprocess_s = time.perf_counter() - start
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("Processor did not produce image tensors")
    grid = inputs["image_grid_thw"].tolist()
    if len(grid) != 1:
        raise RuntimeError(f"Expected one image grid, got {grid}")
    patch_count = int(grid[0][0] * grid[0][1] * grid[0][2])
    merge_size = int(model.config.vision_config.spatial_merge_size)
    visual_tokens = patch_count // (merge_size * merge_size)
    input_ids = inputs["input_ids"]
    prompt_tokens = int(input_ids.shape[-1])
    image_token_id = int(model.config.image_token_id)
    placeholder_count = int((input_ids == image_token_id).sum().item())
    if placeholder_count != visual_tokens:
        raise RuntimeError(
            f"Image token mismatch: placeholders={placeholder_count}, "
            f"visual_tokens={visual_tokens}"
        )

    inputs = inputs.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = {
        "model_revision": config["model_revision"],
        "model_dir": str(model_dir),
        "image": str(image_path),
        "image_size": list(image.size),
        "prompt": config["prompt"],
        "dtype": config["dtype"],
        "gpu_name": gpu_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "free_before_load_mib": free_before / 2**20,
        "free_after_load_mib": free_after_load / 2**20,
        "total_gpu_memory_mib": total_memory / 2**20,
        "image_grid_thw": grid,
        "vision_patch_count": patch_count,
        "visual_token_count": visual_tokens,
        "image_placeholder_count": placeholder_count,
        "llm_input_tokens": prompt_tokens,
        "preprocess_wall_ms": preprocess_s * 1000,
    }
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=int(config["max_new_tokens"]),
                do_sample=False,
            )
        torch.cuda.synchronize()
        output["generation_wall_ms"] = (time.perf_counter() - start) * 1000
        new_ids = generated[0, prompt_tokens:]
        output["generated_tokens"] = int(new_ids.numel())
        output["generated_text"] = processor.decode(
            new_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        output["status"] = "cuda_oom"
        output["error"] = str(exc)
    finally:
        output["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        output["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
