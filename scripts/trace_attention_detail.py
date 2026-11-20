"""Add NVTX labels to the existing Attention operations for one diagnostic trace."""

from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import profile_decode_study as study


def main():
    import torch
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen2_vl import modeling_qwen2_vl as qwen

    annotate = study.trace_module
    profiling_attention = False
    nvtx_range = torch.cuda.nvtx.range

    def label_call(original, part):
        @wraps(original)
        def wrapped(*args, **kwargs):
            if profiling_attention:
                with nvtx_range(f"vlm_attn_part/{part}"):
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        return wrapped

    def enter(*args):
        nonlocal profiling_attention
        profiling_attention = True

    def leave(*args):
        nonlocal profiling_attention
        profiling_attention = False

    def add_ranges(module, name, api, cleanup):
        annotate(module, name, api, cleanup)
        if not isinstance(module, qwen.Qwen2VLSdpaAttention):
            return
        cleanup.callback(module.register_forward_pre_hook(enter).remove)
        cleanup.callback(module.register_forward_hook(leave, always_call=True).remove)
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            part = "output_projection" if attr == "o_proj" else "qkv_projection"
            annotate(getattr(module, attr), f"vlm_attn_part/{part}", api, cleanup)

    with ExitStack() as stack:
        for owner, attr, part in (
            (qwen, "apply_multimodal_rotary_pos_emb", "rope"),
            (DynamicCache, "update", "kv_cache_update"),
            (qwen, "repeat_kv", "kv_head_repeat"),
            (torch.nn.functional, "scaled_dot_product_attention", "sdpa"),
        ):
            stack.enter_context(patch.object(owner, attr, label_call(getattr(owner, attr), part)))
        stack.enter_context(patch.object(study, "trace_module", add_ranges))
        study.main()


if __name__ == "__main__":
    main()
