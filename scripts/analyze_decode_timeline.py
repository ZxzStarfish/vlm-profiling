"""Attribute traced GPU work and idle intervals to matched Decode module calls."""

import argparse
import csv
import gzip
import json
import re
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

MODULE = re.compile(r"vlm_(vision_)?(attention|mlp)_l(\d+)_s(\d+)")
STEP = re.compile(r"vlm_step_(\d+)")


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    name: str
    thread: object = None
    correlation: int | None = None
    kind: str = ""

    @property
    def duration(self):
        return self.end - self.start


class Coverage:
    """Length of the union of all intervals, with efficient window queries."""

    def __init__(self, spans):
        merged = []
        for span in sorted(spans, key=lambda item: item.start):
            if merged and span.start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], span.end)
            else:
                merged.append([span.start, span.end])
        self.starts = [start for start, _ in merged]
        self.ends = [end for _, end in merged]
        self.prefix = [0.0]
        for start, end in merged:
            self.prefix.append(self.prefix[-1] + end - start)

    def until(self, time):
        idx = bisect_right(self.starts, time) - 1
        if idx < 0:
            return 0.0
        return self.prefix[idx] + min(time, self.ends[idx]) - self.starts[idx]

    def inside(self, start, end):
        return max(0.0, self.until(end) - self.until(start)) if end > start else 0.0

    @property
    def total(self):
        return self.prefix[-1]


def from_kineto(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        raw = json.load(source)["traceEvents"]
    markers, apis, gpu = [], [], []
    for event in raw:
        if event.get("ph") != "X":
            continue
        kind, name = event.get("cat"), event["name"]
        start, end = event["ts"], event["ts"] + event["dur"]
        thread = (event.get("pid"), event.get("tid"))
        correlation = event.get("args", {}).get("correlation")
        row = Span(start, end, name, thread, correlation, kind)
        if kind == "user_annotation" and name.startswith("vlm_"):
            markers.append(row)
        elif kind in ("cuda_runtime", "cuda_driver"):
            apis.append(row)
        elif kind in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu.append(row)
    return markers, apis, gpu


def from_nsys(path):
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        names = dict(db.execute("SELECT id, value FROM StringIds"))
        markers = [
            Span(start / 1000, end / 1000, name, thread)
            for start, end, thread, label, label_id in db.execute(
                "SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
            )
            if (name := label or names.get(label_id, "")).startswith("vlm_")
        ]
        apis = []
        for suffix, kind in (("RUNTIME", "cuda_runtime"), ("DRIVER", "cuda_driver")):
            table = "CUPTI_ACTIVITY_KIND_" + suffix
            if table in tables:
                apis.extend(
                    Span(start / 1000, end / 1000, names.get(name, str(name)), thread, correlation, kind)
                    for start, end, thread, name, correlation in db.execute(
                        f"SELECT start,end,globalTid,nameId,correlationId FROM {table}"
                    )
                )
        gpu = []
        for suffix, kind in (("KERNEL", "kernel"), ("MEMCPY", "gpu_memcpy"), ("MEMSET", "gpu_memset")):
            table = "CUPTI_ACTIVITY_KIND_" + suffix
            if table in tables:
                name_column = "shortName" if kind == "kernel" else "NULL"
                gpu.extend(
                    Span(start / 1000, end / 1000, names.get(name, kind), stream, correlation, kind)
                    for start, end, stream, correlation, name in db.execute(
                        f"SELECT start,end,streamId,correlationId,{name_column} FROM {table}"
                    )
                )
    return markers, apis, gpu


def sync_api(api):
    """Report synchronization and blocking copies observable in CUDA API ranges."""
    return "Synchronize" in api.name or re.match(r"^(cuda|cu)Memcpy(?:_[a-zA-Z0-9]+)?$", api.name)


def analyze(path, decode_steps):
    markers, apis, gpu = from_nsys(path) if path.suffix == ".sqlite" else from_kineto(path)
    modules = [m for m in markers if MODULE.fullmatch(m.name)]
    steps = sorted((m for m in markers if STEP.fullmatch(m.name)), key=lambda m: m.start)
    nonvision = [m for m in modules if MODULE.fullmatch(m.name).group(1) is None]
    decode = [m for m in nonvision if int(MODULE.fullmatch(m.name).group(4)) > 0]
    if len(steps) != decode_steps + 1 or len(decode) != 28 * 2 * decode_steps:
        raise ValueError(f"{path.name}: incomplete step or module annotations")

    module_index = {}
    for thread in {row.thread for row in modules}:
        spans = sorted((m for m in modules if m.thread == thread), key=lambda m: m.start)
        module_index[thread] = ([m.start for m in spans], spans)

    def owner(api):
        starts, spans = module_index.get(api.thread, ([], []))
        idx = bisect_right(starts, api.start) - 1
        return spans[idx] if idx >= 0 and api.start < spans[idx].end else None

    api_by_id = {}
    module_apis = defaultdict(list)
    for api in apis:
        mod = owner(api)
        if mod is not None:
            module_apis[mod].append(api)
        if api.correlation is None:
            continue
        # Prefer a runtime entry over its nested driver call, where both exist.
        prev = api_by_id.get(api.correlation)
        if prev is None or (api.kind == "cuda_runtime" and prev.kind != "cuda_runtime"):
            api_by_id[api.correlation] = api
    work = defaultdict(list)
    match_counts = Counter()
    for activity in gpu:
        submit = api_by_id.get(activity.correlation)
        match_counts["gpu_kernels"] += int(activity.kind == "kernel")
        if submit is not None:
            match_counts["api_linked_kernels"] += int(activity.kind == "kernel")
            mod = owner(submit)
            if mod is not None:
                work[mod].append((activity, submit))
    gpu_coverage = Coverage(gpu)
    rows = []
    kernels = Counter()
    for mod in modules:
        vision, kind, layer, step = MODULE.fullmatch(mod.name).groups()
        phase = "vision" if vision else ("prefill" if step == "00" else "decode")
        launches = sorted(work[mod], key=lambda pair: pair[0].start)
        if not launches:
            raise ValueError(f"{path.name}: no linked GPU work in {mod.name}")
        start = launches[0][0].start
        end = max(activity.end for activity, _ in launches)
        span = end - start
        idle = max(0.0, span - gpu_coverage.inside(start, end))
        covered_until = start
        idle_before_submit = 0.0
        kernel_sum = 0.0
        other_sum = 0.0
        for activity, submit in launches:
            gap_end = min(activity.start, submit.start)
            if gap_end > covered_until:
                idle_before_submit += max(0.0, gap_end - covered_until - gpu_coverage.inside(covered_until, gap_end))
            covered_until = max(covered_until, activity.end)
            if activity.kind == "kernel":
                kernel_sum += activity.duration
                kernels[phase, kind, int(layer), int(step), mod.name, activity.name] += activity.duration
            else:
                other_sum += activity.duration
        api_ranges = module_apis[mod]
        rows.append({
            "trace": path.name, "phase": phase, "kind": kind, "layer": int(layer), "step": int(step),
            "host_us": mod.duration, "gpu_kernel_us": kernel_sum, "gpu_copy_set_us": other_sum,
            "gpu_span_us": span, "gpu_idle_inside_span_us": idle,
            "gpu_idle_before_next_submission_us": idle_before_submit,
            "cpu_launch_api_us": Coverage(row for row in api_ranges if "LaunchKernel" in row.name).total,
            "cpu_sync_or_blocking_copy_us": Coverage(row for row in api_ranges if sync_api(row)).total,
        })
    kernel_rows = [
        {"trace": path.name, "phase": phase, "kind": kind, "layer": layer, "step": step,
         "nvtx": label, "name": name, "kernel_us": duration}
        for (phase, kind, layer, step, label, name), duration in kernels.items()
    ]
    summary = []
    gaps = [Span(a.end, b.start, "inter_step", a.thread) for a, b in zip(steps, steps[1:])]
    interstep_sync = [
        api for api in apis if sync_api(api)
        and any(g.thread == api.thread and g.start <= api.start < g.end for g in gaps)
    ]
    for kind in ("attention", "mlp"):
        selected = [row for row in rows if row["phase"] == "decode" and row["kind"] == kind]
        summary.append({
            "trace": path.name, "kind": kind, "calls": len(selected), **match_counts,
            **{key: sum(row[key] for row in selected) for key in selected[0] if key.endswith("_us")},
            "request_interstep_host_us": sum(g.duration for g in gaps),
            "request_interstep_sync_or_blocking_copy_us": Coverage(interstep_sync).total,
            "request_interstep_sync_api_records": len(interstep_sync),
        })
    return summary, rows, kernel_rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_benchmark(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    native = median(row["generation_ms"] for row in samples if row["mode"] == "native")
    modes = []
    for mode in ("native", "events", "nvtx"):
        rows = [row for row in samples if row["mode"] == mode]
        info = {"mode": mode, "n": len(rows), "median_ms": median(row["generation_ms"] for row in rows)}
        info["ratio_to_native"] = info["median_ms"] / native
        for kind in ("attention", "mlp"):
            totals = [
                sum(sum(layer["decode_stream_ms"]) for layer in row["modules"] if layer["kind"] == kind)
                for row in rows if "modules" in row
            ]
            info[kind + "_median_decode_stream_ms"] = median(totals) if totals else ""
        modes.append(info)
    return modes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, modules, kernels = [], [], []
    for path in args.trace:
        summary, rows, kernel_rows = analyze(path.resolve(), decode_steps=15)
        summaries.extend(summary)
        modules.extend(rows)
        kernels.extend(kernel_rows)
    write_csv(args.out / "decode_timeline.csv", summaries)
    write_csv(args.out / "modules_timeline.csv", modules)
    write_csv(args.out / "kernels_timeline.csv", kernels)
    if args.benchmark:
        write_csv(args.out / "benchmark_summary.csv", analyze_benchmark(args.benchmark))
    print(f"DECODE_TIMELINE_OK traces={len(args.trace)} {args.out}")


if __name__ == "__main__":
    main()
