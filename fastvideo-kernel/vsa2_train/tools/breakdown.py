"""Where a solution's time goes: per workload, the kernels one call launches, their time, the gaps between them, and,
when the task gives bounds, how far the call is from its bound.

    python breakdown.py [--solution DIR] [--workload W ...] [--calls N] [--out DIR] [--spec spec.py]

Runs without special rights, on one GPU, in this process (CUPTI; no profiler daemons). The solution is loaded with
loader.py as bench.py loads it; spec.py gives WORKLOADS, make_inputs and, for destination passing, make_outputs, and
optionally bounds(workload) (see bounds.py). Defaults, for a check run as `python -I $CHECKS/breakdown.py`: the
solution $WORK/solution, $CHECKS/spec.py (else spec.py next to this file), every workload, and $OUT.

Per workload, after warm-up calls, N calls (default 5) are recorded, each with fresh input copies and a flushed L2,
as bench.py times them. A call spans from the start of its first GPU operation to the end of its last. The call of
median span stands for the workload: per kernel name, its time is the summed duration of its launches in that
call; gaps are the time in the span when no operation runs (launch overhead, host work, synchronization), so the
kernel, copy, set and gap rows add up to the span. Library kernels (cuBLAS/cuBLASLt, cuDNN, ATen, Triton) are found
as audit.py finds them, from the launch's call stack; without audit.py, by name.

The bound of a workload is the slower of its FLOPs at the GPU's 16-bit tensor-core peak and its bytes at the GPU's
memory bandwidth (bounds.py: estimated peaks at the GPU's SM clock); the gap is the call's span / its bound. Where
bounds() also names the call's stages, each gets a line: the time of the kernels whose names its pattern matches,
against the stage's tensor, SFU and memory floors (bounds.stage_floors), and how far it is from the largest of them.

Writes OUT/result.json:

    {"passed": bool, "summary": "per workload: span, bound and gap, the largest kernels and the gaps",
     "table": {"columns": ["workload", "kernel", "calls", "ms", "share", "library"] (+ ["bound_ms", "gap"]),
               "rows": [...]}}

The table is complete, every kernel of every workload with its full name; only the summary is cut to SUMMARY_LIMIT
characters. Per workload, the first row, "(call)", is the whole call: its launches, its span (ms) and, with bounds,
bound_ms and gap. Then one row per kernel (calls: launches per call, ms: time per call, share: fraction of the span),
"(memcpy)" and "(memset)" rows if any, and "(gaps)".
"""

import argparse
import collections
import json
import os
import re
import sys
import traceback

# Expandable segments, as bench.py sets them; set before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["workload", "kernel", "calls", "ms", "share", "library"]
BOUND_COLUMNS = ["bound_ms", "gap"]
WARMUP = 3
SUMMARY_LIMIT = 6000
MARKER = "spin_kernel"  # torch.cuda._sleep, which brackets each recorded call

# Used when audit.py is not next to this file: library kernels by name.
NAMES = [
    (r"(void )?at::.*|.*at::native::.*|_ZN2at.*|_ZN3c10.*", "ATen"),
    (r".*(nvjet|cublas|gemv2?[NT]?_|gemmSN|splitKreduce|xmma|sgemm|hgemm|dgemm).*|cutlass3x_.*|cutlass_\d.*",
     "cuBLAS"),
    (r".*cudnn.*", "cuDNN"),
    (r"triton_.*", "Triton"),
]


def load_spec(path):
    import importlib.util

    module_spec = importlib.util.spec_from_file_location("spec", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    sys.modules["spec"] = module
    return module


def demangler():
    try:
        import audit

        return audit.demangle
    except ImportError:
        return lambda name: name


def short_name(name, limit=60):
    """A kernel's demangled name without return type and parameter list, its template arguments collapsed from the
    innermost out until it fits in `limit` (None: the whole name); a short wrapper (Kernel2<...>) keeps the start of
    its argument."""
    full = re.sub(r"^void ", "", name.strip())
    if full.endswith(")"):
        depth = 0
        for i in range(len(full) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(full[i], 0)
            if depth == 0:
                full = full[:i]
                break
    if limit is None:
        return full
    short = full
    while len(short) > limit:
        collapsed = re.sub(r"<(?!…>)(?:[^<>]|<…>)+>", "<…>", short, count=1)
        if collapsed == short:
            break
        short = collapsed
    if len(short) < limit // 2 and short.endswith("<…>"):
        short = full
    return short if len(short) <= limit else short[: limit - 1] + "…"


class Recorder:
    """CUPTI activity records (kernels, copies, sets) of the calls made in between start() and stop()."""

    def __init__(self):
        from cupti import cupti

        self.cupti = cupti
        self.kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET)
        self.records = []

    def start(self):
        cupti = self.cupti

        def requested():
            return 8 * 1024 * 1024, 0

        def completed(activities):
            for a in activities:
                if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                    self.records.append((a.start, a.end, "kernel", a.name))
                elif a.kind == cupti.ActivityKind.MEMCPY:
                    self.records.append((a.start, a.end, "memcpy", ""))
                elif a.kind == cupti.ActivityKind.MEMSET:
                    self.records.append((a.start, a.end, "memset", ""))

        for k in self.kinds:
            cupti.activity_enable(k)
        cupti.activity_register_callbacks(requested, completed)

    def stop(self):
        try:
            self.cupti.activity_flush_all(1)
        finally:
            for k in self.kinds:
                self.cupti.activity_disable(k)
            self.cupti.finalize()


def busy_time(ops):
    """Time covered by at least one of the (start, end) intervals."""
    total, until = 0, None
    for s, e in sorted((op[0], op[1]) for op in ops):
        if until is None or s > until:
            total += e - s
            until = e
        elif e > until:
            total += e - until
            until = e
    return total


def clock_reader():
    """bench.py's reader of this GPU's SM clock (NVML in this process), or None where it cannot be had."""
    import torch

    try:
        import bench

        uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "") or "").lower().removeprefix("gpu-")
        return bench.sm_clock(uuid)
    except Exception:  # noqa: BLE001
        return None


def breakdown(sol, spec, workload, calls, seed, demangle):
    """Rows of one workload's call of median span: (kernel, launches, ms, share of the span, library), kernels in
    launch order, then copies, sets and gaps; that call's span, the min and max span over the calls (ms), and the mean
    SM clock in MHz while the recorded calls ran (None where NVML cannot be read)."""
    import torch
    from cuda.bindings import runtime

    inputs = spec.make_inputs(workload, seed, "cuda")
    read_clock, clocks = clock_reader(), []
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")

    def prepare():
        args = [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]
        outs = sol.make_outputs(args) if sol.dps else None
        runtime.cudaCtxResetPersistingL2Cache()
        flush.zero_()
        return args, outs

    def call(args, outs, sample=False):
        torch.cuda._sleep(0)
        if sol.dps:
            sol.fn(*args, *outs)
        else:
            sol.fn(*args)
        torch.cuda._sleep(0)
        if sample and read_clock:
            # The clock while the call runs, read from the host until its last operation is done; reading it puts
            # nothing on the GPU, so the call's span is the same.
            done = torch.cuda.Event()
            done.record()
            while not done.query():
                mhz = read_clock()
                if mhz:
                    clocks.append(mhz)
        torch.cuda.synchronize()

    for _ in range(WARMUP):
        call(*prepare())
    traced = libraries(sol, prepare)
    rec = Recorder()
    rec.start()
    try:
        for _ in range(calls):
            call(*prepare(), sample=True)
    finally:
        rec.stop()
    records = sorted(rec.records)
    marks = [i for i, r in enumerate(records) if MARKER in r[3]]
    if len(marks) != 2 * calls:
        raise RuntimeError(f"expected {2 * calls} marker kernels, found {len(marks)}")
    per_call = [records[marks[j] + 1: marks[j + 1]] for j in range(0, len(marks), 2)]
    spans = [max(op[1] for op in ops) - min(op[0] for op in ops) if ops else 0 for ops in per_call]
    # The call of median span stands for the workload, so that its rows add up to its span.
    pick = sorted(range(calls), key=lambda i: spans[i])[calls // 2]
    ops, span = per_call[pick], max(spans[pick], 1) / 1e6
    times, counts = collections.Counter(), collections.Counter()
    for s, e, kind, name in ops:
        times[(kind, name)] += e - s
        counts[(kind, name)] += 1
    rows = []
    for key in sorted(counts, key=lambda k: k[0] != "kernel"):  # kernels in launch order, then copies and sets
        kind, name = key
        ms = times[key] / 1e6
        label = name if kind == "kernel" else f"({kind})"
        library = "" if kind != "kernel" else traced[name] if name in traced else by_name(demangle(name))
        rows.append([label, counts[key], ms, ms / span, library])
    gap = span - busy_time(ops) / 1e6 if ops else 0.0
    rows.append(["(gaps)", 0, gap, gap / span, ""])
    held = sum(clocks) / len(clocks) if clocks else None
    return rows, span, (min(spans) / 1e6, max(spans) / 1e6), held


def libraries(sol, prepare):
    """{kernel name: library or ""} of one call traced by audit.py; {} without audit.py."""
    try:
        import audit
    except ImportError:
        return {}
    args, outs = prepare()
    ops = audit.trace(sol.fn, *args, *(outs or []))
    return {op.name: op.library or "" for op in ops if op.kind == "kernel"}


def by_name(name):
    """The library of a kernel by its name alone (see NAMES)."""
    for pattern, label in NAMES:
        if re.fullmatch(pattern, name):
            return label
    return ""


def stage_lines(kernels, floors, demangle, held=None, mhz=None):
    """One line per stage: the time of its kernels, its floors at the bound's clock and the stage's time over the
    largest of them; with `held`, the same floors at the clock the calls held (mhz), and the time over their largest."""
    def named(f):
        return [(label, f[key]) for key, label in (("tensor", "tensor"), ("sfu", "SFU"), ("memory", "memory"))
                if f[key] is not None]

    lines = []
    for i, f in enumerate(floors):
        mine = [r for r in kernels if re.search(f["match"], demangle(r[0]))]
        ms = sum(r[2] for r in mine)
        line = f"  stage {f['name']}: {ms:.3f} ms in {len(mine)} kernel{'' if len(mine) == 1 else 's'}"
        for prefix, these in (("; floors ", named(f)), (f"; at the {mhz or 0:.0f} MHz the calls held ",
                                                         named(held[i]) if held and mhz else [])):
            if these:
                line += prefix + ", ".join(f"{label} {v:.3f}" for label, v in these) + " ms"
                top = max(v for _, v in these)
                if mine and top > 0:
                    line += f", x{ms / top:.2f} of the largest"
        lines.append(line)
    return lines


def summarize(results, errors, calls, demangle, limits=None, bound_text="", stages=None):
    """Per workload: span, bound and gap, gaps, library share, its stages against their floors (stages: {workload:
    {"floors": bounds.stage_floors(...), "held": the same at the clock the calls held, or None, "mhz": that clock}}),
    and its largest kernels; as many as fit in SUMMARY_LIMIT."""
    limits, stages = limits or {}, stages or {}
    head = [f"One call per workload, the one of median span among {calls} (min-max in parentheses), with fresh "
            "inputs and a flushed L2: ms and % of its span. [lib]: a library kernel."]
    if limits:
        head.append(f"Bound: {bound_text}; gap = span / bound.")
    if any(s.get("floors") for s in stages.values()):
        head.append("Stages: their kernels' time against the stage's floors at the bound's clock: tensor (its FLOPs "
                    "at the tensor-core peak), SFU (every exponential on the MUFU, which work moved to the FMA pipe can "
                    "undercut) and memory; and the same floors at the mean SM clock the recorded calls held, where "
                    "NVML reads it (a GPU at its power cap holds less than the bound's clock).")

    def block(w, k):
        if w in errors:
            return [f"{w}: ERROR {errors[w].strip().splitlines()[-1][:200]}"]
        rows, span, (lo, hi) = results[w]
        kernels = [r for r in rows if not r[0].startswith("(")]
        gap = next(r for r in rows if r[0] == "(gaps)")
        lib = [r for r in kernels if r[4]]
        line = f"{w}: {span:.3f} ms ({lo:.3f}-{hi:.3f})"
        if w in limits:
            line += f", bound {limits[w]:.3f} ms, gap x{span / limits[w]:.2f}"
        line += (f", {sum(int(r[1]) for r in kernels)} launches of {len(kernels)} kernels, gaps {gap[2]:.3f} ms "
                 f"({100 * gap[3]:.0f}%)")
        if lib:
            line += f", library {sum(r[2] for r in lib):.3f} ms in {len(lib)} kernels"
        copies = [r for r in rows if r[0] in ("(memcpy)", "(memset)")]
        if copies:
            line += ", " + ", ".join(f"{r[0][1:-1]} {r[2]:.3f} ms" for r in copies)
        s = stages.get(w, {})
        lines = [line] + stage_lines(kernels, s.get("floors", []), demangle, s.get("held"), s.get("mhz"))
        for r in sorted(kernels, key=lambda r: -r[2])[:k]:
            lib_mark = f" [{r[4]}]" if r[4] else ""
            lines.append(f"  {r[2]:8.3f} {100 * r[3]:4.0f}%  x{int(r[1]):<3} {short_name(demangle(r[0]))}{lib_mark}")
        if len(kernels) > k:
            rest = sorted(kernels, key=lambda r: -r[2])[k:]
            lines.append(f"  {sum(r[2] for r in rest):8.3f} {100 * sum(r[3] for r in rest):4.0f}%  "
                         f"{len(rest)} more kernels")
        return lines

    names = list(results) + [w for w in errors if w not in results]
    tail = []
    gaps = [results[w][1] / limits[w] for w in results if w in limits]
    if gaps and len(gaps) == len(names):
        tail.append(f"Geomean gap to the bound: x{bound_geomean(gaps):.2f} over {len(gaps)} workloads.")
    for k in (8, 6, 5, 4, 3, 2, 1, 0):
        text = "\n".join(head + [line for w in names for line in block(w, k)] + tail)
        if len(text) <= SUMMARY_LIMIT:
            return text
    return text[: SUMMARY_LIMIT - 20] + "\n[summary cut]"


def bound_geomean(xs):
    import math

    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solution", default=os.path.join(os.environ.get("WORK", "."), "solution"))
    p.add_argument("--workload", nargs="*")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--out", default=os.environ.get("OUT", "."))
    p.add_argument("--spec", default=os.path.join(os.environ.get("CHECKS", HERE), "spec.py"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    spec = load_spec(a.spec)
    workloads = a.workload or list(spec.WORKLOADS)
    unknown = [w for w in workloads if w not in spec.WORKLOADS]
    if unknown:
        p.error(f"unknown workloads {unknown}; known: {list(spec.WORKLOADS)}")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    sys.path.insert(0, HERE)

    import torch

    import bounds
    import loader

    limits, dev, peak = bounds.workload_bounds(spec, workloads)
    stages = {}
    columns = COLUMNS + (BOUND_COLUMNS if limits else [])
    result = {"passed": False, "summary": "", "table": {"columns": columns, "rows": []}}
    results, errors = {}, {}
    try:
        sol = loader.load(os.path.abspath(a.solution), make_outputs=getattr(spec, "make_outputs", None))
    except Exception as e:  # noqa: BLE001
        result["summary"] = f"FAILED: the solution did not load: {e}"[:SUMMARY_LIMIT]
        sol = None
    if sol is not None:
        demangle = demangler()
        for w in workloads:
            try:
                rows, span, spread, mhz = breakdown(sol, spec, w, a.calls, a.seed, demangle)
                results[w] = (rows, span, spread)
                if limits:
                    held = bounds.stage_floors(spec.bounds(w), bounds.peaks({**dev, "clock_mhz": mhz})) if mhz else None
                    stages[w] = {"floors": bounds.stage_floors(spec.bounds(w), peak), "held": held, "mhz": mhz}
                launches = sum(int(r[1]) for r in rows if not r[0].startswith("("))
                call = [w, "(call)", launches, round(span, 6), 1.0, ""]
                if limits:
                    call += [round(limits[w], 6), round(span / limits[w], 4)]
                # Rows of copies, sets and gaps keep their "(...)" labels; kernels get their full names.
                table = [call] + [[w, r[0] if r[0].startswith("(") else short_name(demangle(r[0]), None), r[1],
                                   round(r[2], 6), round(r[3], 4), r[4]] + ([None, None] if limits else [])
                                  for r in rows]
                result["table"]["rows"] += table
            except Exception:  # noqa: BLE001
                errors[w] = traceback.format_exc(limit=3)[-1500:]
            torch.cuda.empty_cache()
        result["passed"] = not errors
        result["summary"] = summarize(results, errors, a.calls, demangle, limits,
                                      bounds.describe(dev, peak) if limits else "", stages)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(result, f)
    print(result["summary"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
