"""Summarize CPU operators and linked CUDA activities inside generation ranges."""

import gzip
import shutil
from collections import defaultdict
from contextlib import ExitStack


def trace_module(module, name, profiler, cleanup):
    active = []
    calls = 0

    def after(*args):
        active.pop().__exit__(None, None, None)

    # Forward hooks close normal ranges. A separate cleanup closes any remaining
    # range after an exception without closing an already finished range twice.
    def enter(*args):
        nonlocal calls
        label = name(calls) if callable(name) else name
        calls += 1
        context = profiler.record_function(label)
        context.__enter__()
        active.append(context)

    def close_remaining():
        while active:
            after()

    cleanup.callback(close_remaining)
    cleanup.callback(module.register_forward_pre_hook(enter).remove)
    cleanup.callback(module.register_forward_hook(after).remove)


def event_context(event, cache):
    key = id(event)
    if key not in cache:
        phase, module = ("outside", "other") if event.cpu_parent is None else event_context(
            event.cpu_parent, cache,
        )
        if event.name.startswith("vlm.phase/"):
            phase = event.name.split("/", 1)[1]
        if event.name.startswith("vlm.module/"):
            module = event.name.split("/", 1)[1]
        cache[key] = phase, module
    return cache[key]


def aggregate(prof, torch):
    operators = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    activities = defaultdict(lambda: [0, 0.0])
    context_cache = {}
    linked = 0
    device_events = 0
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            device_events += 1
        if event.device_type != torch.autograd.DeviceType.CPU or event.is_async:
            continue
        phase, module = event_context(event, context_cache)
        if event.name.startswith("aten::"):
            row = operators[phase, module, event.name]
            row[0] += 1
            row[1] += event.self_cpu_time_total
            row[2] += event.self_device_time_total
            row[3] += event.device_time_total
        for activity in event.kernels:
            row = activities[phase, module, activity.name]
            row[0] += 1
            row[1] += activity.duration
            linked += 1
    phases = {phase for (phase, _, _), row in activities.items() if row[1] > 0}
    if not device_events or not {"vision", "prefill", "decode"} <= phases:
        raise RuntimeError(f"Missing CUDA phase activity: events={device_events}, phases={sorted(phases)}")
    return {
        "cuda_device_events": device_events,
        "linked_cuda_activities": linked,
        "operators": [
            {"phase": p, "module": m, "name": n, "calls": count, "self_cpu_us": cpu,
             "self_cuda_us": cuda, "inclusive_cuda_us": inclusive}
            for (p, m, n), (count, cpu, cuda, inclusive) in operators.items()
        ],
        "cuda_activities": [
            {"phase": p, "module": m, "name": n, "calls": count, "cuda_us": duration}
            for (p, m, n), (count, duration) in activities.items()
        ],
    }


def profile_generation(torch, model, generate, targets, path, expected_tokens):
    profiler = torch.profiler
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]) as prof:
        with ExitStack() as cleanup:
            trace_module(model, lambda call: "vlm.phase/" + ("decode" if call else "first_forward"), profiler, cleanup)
            trace_module(model.visual, "vlm.phase/vision", profiler, cleanup)
            trace_module(model.model, lambda call: "vlm.phase/" + ("decode" if call else "prefill"), profiler, cleanup)
            trace_module(model.lm_head, "vlm.module/lm_head", profiler, cleanup)
            for (_, _, kind), module in targets:
                if kind != "block":
                    category = "norm" if kind in {"norm1", "norm2"} else kind
                    trace_module(module, "vlm.module/" + category, profiler, cleanup)
            with profiler.record_function("vlm.phase/generation_other"):
                tokens = generate()
            torch.cuda.synchronize()
    if tokens != expected_tokens:
        raise RuntimeError("Profiler generation differs from the reference token IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".json")
    prof.export_chrome_trace(str(raw))
    with raw.open("rb") as source, gzip.open(path, "wb") as destination:
        shutil.copyfileobj(source, destination)
    raw.unlink()
    result = aggregate(prof, torch)
    result["trace"] = path.name
    return result
