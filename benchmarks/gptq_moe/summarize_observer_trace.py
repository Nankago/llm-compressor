"""Summarize an idle_observer.py Nsight SQLite export in the observer host windows."""

import argparse
import json
import sqlite3
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path


class Timeline:
    """Index intervals, including activities that started before a queried window."""

    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r[0])
        self.starts = [r[0] for r in self.rows]
        self.ends = list(accumulate((r[1] for r in self.rows), max))

    def overlap(self, lo, hi):
        return [
            r
            for r in self.rows[
                bisect_right(self.ends, lo) : bisect_left(self.starts, hi)
            ]
            if r[1] > lo
        ]


def union(spans):
    merged = []
    for lo, hi in sorted(spans):
        if hi <= lo:
            continue
        if merged and merged[-1][1] >= lo:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def duration(spans):
    return sum(hi - lo for lo, hi in union(spans))


def add(stats, name, ns):
    stats[name][0] += 1
    stats[name][1] += ns


def top(stats):
    return [
        {"name": key, "calls": n, "sum_ms": t / 1e6}
        for key, (n, t) in sorted(stats.items(), key=lambda item: -item[1][1])[:10]
    ]


def analyze(path, expected=None):
    db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    tables = {
        n for (n,) in db.execute("select name from sqlite_master where type='table'")
    }
    nvtx = """select n.start,n.end,coalesce(n.text,s.value),n.globalTid
              from NVTX_EVENTS n left join StringIds s on n.textId=s.id
              where n.end is not null and """
    ranges = list(
        db.execute(nvtx + "coalesce(n.text,s.value) like 'GPTQ/%' order by n.start")
    )
    counts = Counter(r[2] for r in ranges)
    if expected is None:
        expected = {
            "GPTQ/weight_observe": 11,
            "GPTQ/_get_validated_importance": 280,
            "GPTQ/scale_search": 280,
        }
    if counts != expected or len({r[3] for r in ranges}) != 1:
        raise ValueError(f"Incomplete observer NVTX coverage: {counts}")
    tid = ranges[0][3]
    pid = tid & ~0xFFFFFF
    atos = Timeline(
        db.execute(
            nvtx + """n.globalTid=? and
        (coalesce(n.text,s.value) like 'aten::item%' or coalesce(n.text,s.value) like 'aten::is_nonzero%'
         or coalesce(n.text,s.value) like 'aten::_local_scalar_dense%')""",
            (tid,),
        )
    )
    apis = list(
        db.execute(
            """select r.start,r.end,s.value,r.correlationId,r.globalTid
        from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
        where (r.globalTid & ~16777215)=? order by r.start""",
            (pid,),
        )
    )
    launches = defaultdict(list)
    for start, end, name, correlation, thread in apis:
        launches[correlation].append((start, thread))
    api = Timeline(r for r in apis if r[4] == tid)
    ops = list(
        db.execute(
            """select k.start,k.end,s.value,k.correlationId,k.streamId,'kernel'
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.globalPid=?""",
            (pid,),
        )
    )
    if not ops:
        raise ValueError("Missing correlated process kernel trace")
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        ops.extend(
            db.execute(
                """select m.start,m.end,'Memcpy ' || e.label,m.correlationId,m.streamId,'memcpy'
            from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id
            where m.globalPid=?""",
                (pid,),
            )
        )
    if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
        ops.extend(
            db.execute(
                """select start,end,'Memset',correlationId,streamId,'memset'
            from CUPTI_ACTIVITY_KIND_MEMSET where globalPid=?""",
                (pid,),
            )
        )
    gpu = Timeline(ops)
    on_cpu = off_cpu = None
    if "SCHED_EVENTS" in tables:
        events = list(
            db.execute(
                "select start,isSchedIn from SCHED_EVENTS where globalTid=? order by start",
                (tid,),
            )
        )
        if events:
            off_cpu = Timeline(
                (x[0], y[0]) for x, y in zip(events, events[1:]) if not x[1] and y[1]
            )
            on_cpu = Timeline(
                (x[0], y[0]) for x, y in zip(events, events[1:]) if x[1] and not y[1]
            )

    leaf_samples = Timeline([])
    if {"COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS"} <= tables:
        leaf_samples = Timeline(
            db.execute(
                """
            select e.start,e.start+1,s.value from COMPOSITE_EVENTS e
            join SAMPLING_CALLCHAINS c on c.id=e.id and c.stackDepth=0
            join StringIds s on s.id=c.symbol
            where e.globalTid=? and e.threadState=1
        """,
                (tid,),
            )
        )
    report = {}
    for label in expected:
        windows = [(r[0], r[1]) for r in ranges if r[2] == label]
        if sum(b - a for a, b in windows) != duration(windows):
            raise ValueError("Overlapping ranges with the same label")
        totals = Counter()
        origins = defaultdict(list)
        wait_intersections = defaultdict(list)
        kernel_stats = defaultdict(lambda: [0, 0])
        api_stats = defaultdict(lambda: [0, 0])
        samples = []
        gap_leaves = Counter()
        copy_stats = defaultdict(lambda: [0, 0])
        for start, end in windows:
            activities = gpu.overlap(start, end)
            spans = union((max(start, x), min(end, y)) for x, y, *_ in activities)
            totals["inclusive_host"] += end - start
            totals["gpu_active_in_window"] += duration(spans)
            gaps = [
                (a, b)
                for a, b in zip(
                    [start] + [s[1] for s in spans], [s[0] for s in spans] + [end]
                )
                if b > a
            ]
            totals["no_gpu_activity"] += duration(gaps)
            for x, y in gaps:
                gap_leaves.update(symbol for _, _, symbol in leaf_samples.overlap(x, y))
            for kind in ("kernel", "memcpy", "memset"):
                totals[kind + "_active"] += duration(
                    (max(start, x), min(end, y))
                    for x, y, _, _, _, k in activities
                    if k == kind
                )
            if on_cpu is not None:
                for x, y in gaps:
                    totals["on_cpu_in_gpu_gap"] += duration(
                        (max(x, a), min(y, b)) for a, b in on_cpu.overlap(x, y)
                    )
            for x, y, op, _, _, kind in activities:
                if kind == "kernel":
                    add(kernel_stats, op, min(end, y) - max(start, x))
                if kind == "memcpy":
                    add(copy_stats, op, min(end, y) - max(start, x))
            waits = []
            for x, y, name, corr, _ in api.overlap(start, end):
                lo, hi = max(start, x), min(end, y)
                totals["launch_api_count"] += int("launch" in name.lower())
                lower = name.lower()
                if "synchronize" not in lower and "memcpy" not in lower:
                    continue
                # MemcpyAsync API residence is recorded separately from explicit waits.
                add(api_stats, name, hi - lo)
                if "async" in lower:
                    continue
                waits.append((lo, hi))
                overlapping = gpu.overlap(lo, hi)
                origin_set = set()
                for gs, ge, kernel, gcorr, stream, _ in overlapping:
                    submissions = launches.get(gcorr, [])
                    if len(submissions) == 1:
                        submit, submitting_tid = submissions[0]
                        if submit < start:
                            origin = "earlier_launch"
                        elif submit < end and submitting_tid == tid:
                            origin = "this_invocation"
                        else:
                            origin = "other_launch"
                    else:
                        origin = "unknown_launch"
                    clip = (max(lo, gs), min(hi, ge))
                    origins[origin].append(clip)
                    wait_intersections[(origin, kernel)].append(clip)
                    origin_set.add((origin, kernel, stream))
                if label == "GPTQ/_get_validated_importance":
                    scalar = sorted(
                        {
                            op.split(",")[0]
                            for a, b, op, _ in atos.overlap(lo, hi)
                            if a <= x and y <= b
                        }
                    )
                    samples.append(
                        {
                            "api": name,
                            "start_ns": x,
                            "end_ns": y,
                            "correlation": corr,
                            "aten": scalar,
                            "validator_start_ns": start,
                            "overlapping_work": [
                                list(s) for s in sorted(origin_set)[:8]
                            ],
                        }
                    )
            totals["sync_api_residence"] += duration(waits)
            for lo, hi in union(waits):
                if on_cpu is not None:
                    totals["off_cpu_in_sync_api"] += duration(
                        (max(lo, a), min(hi, b)) for a, b in off_cpu.overlap(lo, hi)
                    )
        values = {
            k + "_ms": v / 1e6 for k, v in totals.items() if k != "launch_api_count"
        }
        values.update(
            {
                "calls": len(windows),
                "launch_api_count": totals["launch_api_count"],
                "scheduler_available": on_cpu is not None,
                "top_kernels_clipped": top(kernel_stats),
                "copies_clipped": top(copy_stats),
                "gpu_gap_sampled_leaf_count": sum(gap_leaves.values()),
                "gpu_gap_top_cpu_sample_leaves": gap_leaves.most_common(10),
                "top_sync_and_copy_apis": top(api_stats),
                "gpu_overlap_with_sync_by_launch_ms": {
                    k: duration(v) / 1e6 for k, v in origins.items()
                },
                "top_work_during_sync": [
                    {"origin": k[0], "name": k[1], "union_ms": duration(v) / 1e6}
                    for k, v in sorted(
                        wait_intersections.items(), key=lambda item: -duration(item[1])
                    )[:10]
                ],
            }
        )
        if samples:
            values["longest_validator_wait_examples"] = sorted(
                samples, key=lambda r: r["start_ns"] - r["end_ns"]
            )[:5]
        report[label] = values
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--warm-window", choices=["base", "proto"])
    args = parser.parse_args()
    expected = None
    if args.warm_window:
        expected = {"GPTQ/scale_search": 28}
        if args.warm_window == "base":
            expected["GPTQ/base_qparams"] = 28
    args.out.write_text(json.dumps(analyze(args.sqlite, expected), indent=2) + "\n")
