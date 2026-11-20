"""Break down visible inter-launch GPU gaps in the diagnostic NVTX trace."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from analyze_decode_timeline import Coverage, MODULE, from_nsys, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    markers, apis, gpu = from_nsys(args.trace.resolve())
    by_thread = defaultdict(list)
    for marker in markers:
        match = MODULE.fullmatch(marker.name)
        if match and match[1] is None and match[2] == "attention" and int(match[4]) > 0:
            by_thread[marker.thread].append(marker)
    if sum(map(len, by_thread.values())) != 420:
        raise ValueError("Expected 28 attention layers × 15 Decode steps")
    by_thread = {
        thread: (sorted(spans, key=lambda span: span.start), sorted(span.start for span in spans))
        for thread, spans in by_thread.items()
    }

    def module_at(span):
        ranges, starts = by_thread.get(span.thread, ([], []))
        i = bisect_right(starts, span.start) - 1
        return ranges[i] if i >= 0 and span.start < ranges[i].end else None

    apis_by_id = {}
    for api in apis:
        previous = apis_by_id.get(api.correlation)
        if api.correlation is not None and (previous is None or api.kind == "cuda_runtime"):
            apis_by_id[api.correlation] = api
    if any(row.kind == "kernel" and row.correlation not in apis_by_id for row in gpu):
        raise ValueError("Kernel launch correlation is incomplete")
    operations, parts = defaultdict(list), defaultdict(list)
    for row in gpu:
        submit = apis_by_id.get(row.correlation)
        scope = module_at(submit) if submit else None
        if scope:
            operations[scope].append((row, submit))
    for marker in markers:
        if marker.name.startswith("vlm_attn_part/") and (scope := module_at(marker)):
            parts[scope].append(marker)
    expected = {"rope", "kv_cache_update", "kv_head_repeat", "sdpa", "qkv_projection", "output_projection"}
    coverage = Coverage(gpu)
    gaps = []
    summary = defaultdict(lambda: [0.0, 0])

    def idle(start, end):
        return max(0.0, end - start - coverage.inside(start, end))

    def add(bucket, part, duration):
        if duration > 0:
            summary[bucket, part][0] += duration
            summary[bucket, part][1] += 1

    for scope, work in operations.items():
        regions = sorted(parts[scope], key=lambda x: x.start)
        if {r.name.split("/", 1)[1] for r in regions} != expected:
            raise ValueError(f"Incomplete subranges in {scope.name}")
        if any(a.end > b.start for a, b in zip(regions, regions[1:])):
            raise ValueError(f"Overlapping subranges in {scope.name}")
        ordered = sorted(work, key=lambda pair: pair[0].start)
        previous = ordered[0][0]
        for activity, api in ordered[1:]:
            left, right = previous.end, activity.start
            if left < right and (total := idle(left, right)) > 0:
                pre_end = max(left, min(api.start, right))
                api_end = max(pre_end, min(api.end, right))
                pre = idle(left, pre_end)
                in_api = idle(pre_end, api_end)
                post = idle(api_end, right)
                labelled = 0.0
                for region in regions:
                    start, end = max(left, region.start), min(pre_end, region.end)
                    overlap = idle(start, end)
                    add("before_cuda_api", region.name.split("/", 1)[1], overlap)
                    labelled += overlap
                add("before_cuda_api", "outside_subranges", max(0.0, pre - labelled))
                add("inside_cuda_api", "associated_launch", in_api)
                add("after_cuda_api", "already_submitted", post)
                gaps.append({
                    "module": scope.name, "previous_gpu_activity": previous.name,
                    "next_gpu_activity": activity.name, "cuda_api": api.name,
                    "gpu_idle_us": total, "before_api_us": pre, "inside_api_us": in_api,
                    "after_api_us": post,
                })
            if activity.end > previous.end:
                previous = activity
    if len(operations) != 420 or not gaps:
        raise ValueError("Missing linked Decode Attention calls or idle intervals")
    total = sum(row["gpu_idle_us"] for row in gaps)
    if abs(sum(duration for duration, _ in summary.values()) - total) > 0.01:
        raise ValueError("Gap categories did not reconcile")
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "attention_gaps.csv", gaps)
    write_csv(args.out / "attention_gap_breakdown.csv", [
        {"bucket": bucket, "host_subrange": area, "idle_us": duration,
         "percent_of_idle": 100 * duration / total, "overlapping_intervals": count}
        for (bucket, area), (duration, count) in sorted(summary.items(), key=lambda row: -row[1][0])
    ])
    print(f"ATTENTION_GAPS_OK intervals={len(gaps)} {args.out}")


if __name__ == "__main__":
    main()
