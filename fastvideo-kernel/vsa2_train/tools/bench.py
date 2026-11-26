"""Correctness, timing and library-kernel audit of a solution; a task's CI runs this same file.

It needs loader.py, audit.py and bounds.py (the harness) next to it, and spec.py, which describes the task:

    OUTPUTS: list[str]            names of the outputs, in order
    WORKLOADS: dict[str, dict]    workload name -> {"atol": float, "rtol": float, "match": float, <axes>...}; names
                                  are short identifiers (letters, digits, _), e.g. "B1"; the axes (e.g. "batch_size")
                                  carry the deployment's axis names and values; an optional "outputs" maps an output's
                                  name to the atol, rtol or match it is judged with instead, for outputs of different
                                  scales (a log-sum-exp beside an attention output, gradients beside both)
    make_inputs(workload, seed, device) -> list
                                  inputs in the order of run()'s parameters, outputs aside (tensors and scalars)
    make_outputs(inputs) -> list  zero-filled output tensors for a call on these inputs, in OUTPUTS order; a
                                  destination-passing solution gets them after its inputs, as the last arguments
    bounds(workload) -> dict      optional: {"flops", "bytes"}, lower bounds for one call (see bounds.py)
    RULES                         optional: changes to the stack rules (see audit.py)

The solution is a directory (default solution/, see loader.py), built and loaded exactly as it is deployed; a
development copy (tools/) takes a build its node already made of the same sources, flags and toolchain from the node's
cache (see loader.py), unless --no-cache. Run it in a GPU job from the repo root:

    python tools/bench.py                          # every workload: check, audit, time
    python tools/bench.py --workload B1 B56        # a subset
    python tools/bench.py --no-time                # correctness only
    python tools/bench.py --kernels                # also list what one call launched, per workload
    python tools/bench.py --solution other         # another solution directory

Correctness: one call per workload, compared with reference.run on the same inputs. An output passes when at least
`match` of the elements the reference judges satisfy |out - ref| <= atol + rtol * |ref|, it has no NaN or Inf, and it
is not all zeros where the reference is not. A reference judges every element it does not return as NaN: one too slow
to compute whole at a large workload computes a random sample of it, which the solution cannot know. A reference whose
run takes a keyword argument `outputs` is given copies of the solution's outputs there, in OUTPUTS order, after the
inputs, so that what it returns can depend on a choice the solution made: one can check the solution's top-k block
selection against its own scores within a rounding margin, and compute the attention under that selection. A
reference without the parameter is called with the inputs alone.

Timing, per workload: before every call the inputs are copied to fresh addresses (destination-passing outputs are
zeroed there too) and the L2 cache is flushed; a call's time runs from the start of the first GPU operation it
launched to the end of the last one, as CUPTI records them (CPU work before the first launch is not counted, gaps
between launches are). 10 warm-up calls, one audited call, then 3 trials of 50 calls. A trial's time is the median
of its calls; the latency of record is the mean of the 3 trials, and the fastest trial is shown too. A call so long
that this would take minutes is repeated less: the first warm-up call's time sets the warm-up to the calls that fill
1 s and a trial to those that fill 2 s, within the counts above and at least one each, so that a call of a few
milliseconds is measured as above and one of seconds is warmed once and timed once per trial. Every timed
call must launch the same GPU operations as the audited one, and the solution may start no threads. Which
processes hold the GPU is asked (nvidia-smi) before every trial and after the last: a workload timed while another
process held it is marked SHARED, since its times do not compare with those of another job. The SM clock is read
(NVML, in this process) while each timed call runs, and the mean of those reads is shown with the latency: where the
clock cannot be locked, it says at what clock the times were taken. It is not shown where NVML cannot be read.

Audit: the audited call runs under CUPTI, and each kernel is attributed to the library that launched it (see
audit.py). Library kernels (cuBLAS/cuBLASLt, cuDNN, ATen) are allowed as placeholders; the count per call and their
share of the call's GPU time show where the solution still depends on them. --kernels lists them.

Bound: when spec.py defines bounds(), each workload's latency is shown with its bound (the slower of its FLOPs at the
GPU's tensor-core peak and its bytes at its memory bandwidth, see bounds.py) and the gap, latency / bound.
"""

import os

# Expandable segments, as in deployment; set before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import collections  # noqa: E402
import ctypes  # noqa: E402
import functools  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402
import bounds  # noqa: E402
import loader  # noqa: E402
import spec  # noqa: E402

WARMUP, TRIALS, REPS = 10, 3, 50
WARMUP_S, TRIAL_S = 1.0, 2.0
NVML_CLOCK_SM = 1  # the nvmlClockType_t of the SM clock


def seed_of(workload, seed):
    """Each workload draws its inputs from its own seed."""
    return seed + list(spec.WORKLOADS).index(workload)


def load_reference(path):
    """run() of reference.py."""
    module_spec = importlib.util.spec_from_file_location("_reference", os.path.abspath(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.run


def fresh(inputs):
    """Copies of the tensors among these arguments, and the other arguments as they are."""
    return [t.clone() if isinstance(t, torch.Tensor) else t for t in inputs]


def expected(reference, inputs, outs):
    """What reference.run returns for these inputs, which the solution's outputs are compared with; a reference that
    takes a keyword argument `outputs` is also given copies of the solution's (see the module docstring)."""
    takes = inspect.signature(reference).parameters.get("outputs")
    if takes is not None and takes.kind in (takes.POSITIONAL_OR_KEYWORD, takes.KEYWORD_ONLY):
        return reference(*fresh(inputs), outputs=fresh(outs))
    return reference(*fresh(inputs))


def tolerance(tol, name):
    """The tolerance output `name` is judged with: the workload's atol, rtol and match, with what its "outputs" entry
    gives that output instead."""
    return {"atol": tol["atol"], "rtol": tol["rtol"], "match": tol["match"], **tol.get("outputs", {}).get(name, {})}


def compare(outs, refs, tol):
    """One row per output: (name, passed, matched ratio, max abs error, note). refs is what reference.run returned: a
    tuple of tensors, or the one tensor of a reference with a single output; tol is the workload's entry of WORKLOADS."""
    if isinstance(refs, torch.Tensor):
        refs = (refs,)
    if not isinstance(outs, (tuple, list)) or len(outs) != len(spec.OUTPUTS):
        return [("outputs", False, 0.0, math.inf, f"expected a tuple of {len(spec.OUTPUTS)} tensors")]
    rows = []
    for name, out, ref in zip(spec.OUTPUTS, outs, refs):
        if type(out) is not torch.Tensor:
            rows.append((name, False, 0.0, math.inf, f"{type(out).__name__}, not a plain torch.Tensor"))
            continue
        if out.shape != ref.shape or out.dtype != ref.dtype:
            rows.append((name, False, 0.0, math.inf, f"{tuple(out.shape)} {out.dtype}, expected {tuple(ref.shape)} {ref.dtype}"))
            continue
        x, y = out.to(ref.device, torch.float32), ref.float()
        if not torch.isfinite(x).all():
            rows.append((name, False, 0.0, math.inf, "NaN or Inf"))
            continue
        judged = ~torch.isnan(y)
        if not judged.any():
            rows.append((name, False, 0.0, math.inf, "the reference judges no element"))
            continue
        if not judged.all():
            x, y = x[judged], y[judged]
        if y.abs().sum() > 0 and x.abs().sum() == 0:
            rows.append((name, False, 0.0, math.inf, "all zeros"))
            continue
        err = (x - y).abs()
        own = tolerance(tol, name)
        # A count, not a float32 mean: over 1e8 elements a mean of all ones can round below 1.
        ratio = (err <= own["atol"] + own["rtol"] * y.abs()).sum().item() / err.numel()
        rows.append((name, ratio >= own["match"], ratio, err.max().item(), ""))
    return rows


def gpu_holders(uuid):
    """The processes that hold the GPU now, as host process ids (a job's own namespace numbers them otherwise), or None
    when nvidia-smi cannot say. Only those of the GPU with this UUID when one matches: a machine of one GPU is asked
    whole even if the UUIDs are spelled differently."""
    try:
        got = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if got.returncode != 0:
        return None
    rows = []
    for line in got.stdout.splitlines():
        gpu, _, pid = line.partition(",")
        if pid.strip().isdigit():
            rows.append((gpu.strip().lower().removeprefix("gpu-"), int(pid)))
    mine = [pid for gpu, pid in rows if uuid and gpu == uuid]
    return set(mine or [pid for _, pid in rows])


@functools.cache
def sm_clock(uuid):
    """A function that reads the SM clock of the GPU with this UUID now, in MHz (None when a read fails), or None where
    NVML cannot be loaded or does not know the GPU. It asks NVML in this process, which takes microseconds where
    nvidia-smi takes a tenth of a second, so that a timed call can be read while it runs. As gpu_holders does, a
    machine of one GPU is asked whole when no GPU has this UUID."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return None
        handle, count, mhz = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_uint()
        found = bool(uuid) and nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode(), ctypes.byref(handle)) == 0
        if not found and not (nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) == 0 and count.value == 1
                              and nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0):
            return None
    except (OSError, AttributeError):
        return None

    def read():
        ok = nvml.nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, ctypes.byref(mhz)) == 0
        return mhz.value if ok and mhz.value else None

    return read if read() else None


class Pool:
    """Copies of the inputs, and zeroed outputs, at a new address for every call, shifted by multiples of 256 bytes."""

    ALIGN, MAX_SHIFT = 256, 2048

    def __init__(self, inputs, outputs, calls, seed):
        rng = random.Random(seed)
        self.offsets = [0]
        for _ in range(calls - 1):
            self.offsets.append(self.offsets[-1] + rng.randint(1, self.MAX_SHIFT // self.ALIGN))
        self.args = [(t, False) for t in inputs] + [(t, True) for t in outputs]
        self.pools = []
        for t, _ in self.args:
            if isinstance(t, torch.Tensor):
                block = max(1, self.ALIGN // t.element_size())
                self.pools.append((torch.empty(t.numel() + self.offsets[-1] * block, dtype=t.dtype, device=t.device), block))
            else:
                self.pools.append(None)
        self.call = 0

    def next(self):
        off = self.offsets[self.call]
        self.call += 1
        args = []
        for (t, is_output), pool in zip(self.args, self.pools):
            if pool is None:
                args.append(t)
                continue
            buf, block = pool
            view = buf[off * block: off * block + t.numel()].view(t.shape)
            view.zero_() if is_output else view.copy_(t)
            args.append(view)
        return args


def measure(sol, inputs, seed, timed=True):
    """Warm-up calls, the audited call, and the timed trials (see the module docstring).

    Returns (the audited call's operations, each trial's time in ms, how many processes besides this one held the GPU
    while it was timed, or None when that could not be asked, and the mean SM clock in MHz while the timed calls ran,
    or None where it cannot be read)."""
    from cuda.bindings import runtime
    from cupti import cupti

    outputs = sol.make_outputs(inputs) if sol.dps else []
    pool = Pool(inputs, outputs, WARMUP + 1 + (TRIALS * REPS if timed else 0), seed)
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.int8, device="cuda")
    fn = sol.fn

    def prepare():
        args = pool.next()
        runtime.cudaCtxResetPersistingL2Cache()
        flush.zero_()
        return args

    def call(args, sample=None):
        # A marker kernel before and after the call brackets its GPU work; the CPU does not wait for the copies
        # and the flush, so launches queue up behind them as they would in a training step.
        torch.cuda._sleep(0)
        fn(*args)
        torch.cuda._sleep(0)
        if sample is not None:
            # Everything the call launches is queued and the GPU is still at it (or at the copies and the flush
            # before it): the clock now is the call's, and reading it adds nothing to the call's span.
            sample()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    # The first call says how long one takes, which sizes the warm-up and the trials (see the module docstring).
    start = time.perf_counter()
    call(prepare())
    took = time.perf_counter() - start
    warmup = max(1, min(WARMUP, math.ceil(WARMUP_S / took)))
    reps = max(1, min(REPS, math.ceil(TRIAL_S / took)))
    for _ in range(warmup - 1):
        call(prepare())
    args = prepare()
    ops = audit.trace(fn, *args)
    del args
    if not timed:
        return ops, [], None, None
    uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", "") or "").lower().removeprefix("gpu-")
    holders = []
    clock, clocks = sm_clock(uuid), []

    def sample():
        mhz = clock()
        if mhz:
            clocks.append(mhz)

    records = []

    def requested():
        return 8 * 1024 * 1024, 0

    def completed(activities):
        for a in activities:
            if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                records.append((a.start, a.end, "kernel", a.name, 0))
            elif a.kind in (cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET):
                records.append((a.start, a.end, "memcpy" if a.kind == cupti.ActivityKind.MEMCPY else "memset", "", a.bytes))

    kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMCPY, cupti.ActivityKind.MEMSET)
    for k in kinds:
        cupti.activity_enable(k)
    cupti.activity_register_callbacks(requested, completed)
    try:
        for i in range(TRIALS * reps):
            if i % reps == 0:
                holders.append(gpu_holders(uuid))
            call(prepare(), sample if clock else None)
        cupti.activity_flush_all(1)
    finally:
        for k in kinds:
            cupti.activity_disable(k)
        cupti.finalize()

    holders.append(gpu_holders(uuid))
    # This process is one of them; a process seen at any of the questions besides it shared the GPU.
    others = None if any(h is None for h in holders) else max(0, len(set().union(*holders)) - 1)

    records.sort()
    marks = [i for i, r in enumerate(records) if "spin_kernel" in r[3]]
    if len(marks) != 2 * TRIALS * reps:
        raise RuntimeError(f"expected {2 * TRIALS * reps} marker kernels, found {len(marks)}")
    expected = audit.signature(ops)
    times = []
    for j in range(0, len(marks), 2):
        inner = records[marks[j] + 1: marks[j + 1]]
        got = collections.Counter((r[2], r[3], r[4]) for r in inner)
        if got != expected:
            raise RuntimeError(f"timed call {j // 2 + 1} launched other GPU operations than the audited call "
                               f"(missing: {_ops_text(expected - got)}; extra: {_ops_text(got - expected)})")
        times.append((max(r[1] for r in inner) - min(r[0] for r in inner)) / 1e6 if inner else 0.0)
    trials = [statistics.median(times[i * reps:(i + 1) * reps]) for i in range(TRIALS)]
    return ops, trials, others, statistics.mean(clocks) if clocks else None


def _ops_text(counter, limit=4):
    items = [f"{n}x {kind} {audit.short_name(name, 60) if name else nbytes}" for (kind, name, nbytes), n in counter.items()]
    return ", ".join(items[:limit]) + (" ..." if len(items) > limit else "") if items else "none"


def ops_lines(ops):
    """What a call launched, one line per operation."""
    lines = []
    for kind, name, nbytes, launcher, library, ns in ops:
        if kind == "kernel":
            lines.append(f"    {'[' + library + ']' if library else '':<14} {ns / 1e3:9.1f} us  {audit.short_name(name, 140)}"
                         f"  ({launcher or 'untraced'})")
        else:
            lines.append(f"    {'':<14} {ns / 1e3:9.1f} us  {kind} {nbytes / 2**20:.2f} MiB")
    return lines


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solution", default="solution", help="the solution directory (default: solution)")
    p.add_argument("--reference", default=os.path.join(here, "..", "reference.py"))
    p.add_argument("--workload", nargs="*", default=list(spec.WORKLOADS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--kernels", action="store_true", help="list what one call launched, per workload")
    p.add_argument("--build-dir", help="write and build the solution here; reused when it holds the same build")
    p.add_argument("--no-cache", action="store_true", help="build fresh, without the node's cache of builds")
    p.add_argument("--save", help="CI: save each workload's outputs here instead of checking them")
    p.add_argument("--json", help="write the results to this file")
    a = p.parse_args()
    unknown = [w for w in a.workload if w not in spec.WORKLOADS]
    if unknown:
        p.error(f"unknown workloads {unknown}; known: {list(spec.WORKLOADS)}")
    # Absolute paths: loading the solution makes its working directory the current one.
    solution = os.path.abspath(a.solution)
    if os.path.isfile(solution):
        solution = os.path.dirname(solution)
    a.reference, a.build_dir, a.save, a.json = (x and os.path.abspath(x) for x in (a.reference, a.build_dir, a.save, a.json))
    report = {"workloads": {}}

    def finish():
        if a.json:
            with open(a.json, "w") as f:
                json.dump(report, f)

    rules = audit.rule_table(getattr(spec, "RULES", None))
    report["rules"] = audit.check(solution, rules)
    for v in report["rules"]:
        print(f"rule violated: {v}")
    print(f"stack rules: {'ok' if not report['rules'] else 'FAIL'} ({len(rules)} rules)")
    limits, dev, peak = bounds.workload_bounds(spec, a.workload)
    if limits:
        print(bounds.describe(dev, peak))

    reference = None if a.no_check or a.save else load_reference(a.reference)
    try:
        sol = loader.load(solution, a.build_dir, make_outputs=getattr(spec, "make_outputs", None),
                          cache=None if a.no_cache else loader.cache_dir())
    except Exception as e:
        report["load_error"] = str(e) if isinstance(e, loader.SolutionError) else traceback.format_exc(limit=6)
        print(f"FAILED to build or load {solution}:\n{report['load_error']}", flush=True)
        finish()
        sys.exit(1)
    report |= {"language": sol.language, "dps": sol.dps, "build_s": sol.build_seconds}
    built = f", built in {sol.build_seconds:.1f} s" if sol.build_seconds is not None else ", a build reused"
    print(f"loaded {sol.entry} ({', '.join(sol.languages)}, destination passing: {sol.dps}){built}", flush=True)

    width = max(len(w) for w in spec.WORKLOADS)
    for w in a.workload:
        res = report["workloads"][w] = {}
        try:
            inputs = spec.make_inputs(w, seed_of(w, a.seed), "cuda")
            outs = sol(*fresh(inputs))
            torch.cuda.synchronize()
            if a.save:
                torch.save([o.cpu() if isinstance(o, torch.Tensor) else o for o in outs], os.path.join(a.save, f"{w}.pt"))
            elif reference is not None:
                rows = compare(outs, expected(reference, inputs, outs), spec.WORKLOADS[w])
                res["passed"] = all(r[1] for r in rows)
                res["rows"] = rows
            del outs
            threads = threading.active_count()
            if not a.no_time or a.kernels:
                ops, trials, others, clock = measure(sol, inputs, seed_of(w, a.seed), timed=not a.no_time)
                res["ops"] = [list(op) for op in ops]
                res["lib"] = sum(1 for op in ops if op.library)
                res["lib_share"] = audit.library_share(ops)
                if trials:
                    res |= {"trials": trials, "ms": statistics.mean(trials), "min_ms": min(trials)}
                if clock:
                    res["clock_mhz"] = clock
                if others:
                    res["shared"] = others
            if threading.active_count() > threads:
                raise RuntimeError(f"the solution started threads ({threading.active_count() - threads} more)")
        except Exception:
            res["error"] = traceback.format_exc(limit=4)
        inputs = None
        torch.cuda.empty_cache()
        line = f"{w:<{width}}"
        if "ms" in res:
            at = f", SM clock {res['clock_mhz']:.0f} MHz" if "clock_mhz" in res else ""
            line += f" {res['ms']:9.4f} ms (fastest trial {res['min_ms']:.4f}{at})"
            if w in limits:
                res |= {"bound_ms": limits[w], "gap": res["ms"] / limits[w]}
                line += f"  bound {limits[w]:.4f} ms, gap x{res['gap']:.2f}"
        if "passed" in res:
            worst = min(res["rows"], key=lambda r: r[2])
            line += f"  {'ok  ' if res['passed'] else 'FAIL'} (lowest match {worst[2]:.4f} in {worst[0]}{', ' + worst[4] if worst[4] else ''})"
        if "lib" in res:
            line += f"  library kernels {res['lib']} ({100 * res['lib_share']:.0f}% of GPU time)"
        if "shared" in res:
            line += f"  SHARED: {res['shared']} other process(es) held the GPU while it was timed"
        if "error" in res:
            line += "  ERROR\n" + res["error"]
        print(line, flush=True)
        if a.kernels and "ops" in res:
            print("\n".join(ops_lines(audit.Op(*op) for op in res["ops"])), flush=True)
    results = report["workloads"].values()
    timed = [r["ms"] for r in results if "ms" in r]
    if len(timed) == len(a.workload):
        gaps = [r["gap"] for r in results if "gap" in r]
        gap = f", gap to the bound x{bounds.geomean(gaps):.2f}" if len(gaps) == len(timed) else ""
        clocks = [r["clock_mhz"] for r in results if "clock_mhz" in r]
        at = f", mean SM clock {statistics.mean(clocks):.0f} MHz" if len(clocks) == len(timed) else ""
        print(f"geomean {math.exp(statistics.mean(math.log(t) for t in timed)):.4f} ms over {len(timed)} workloads "
              f"(mean of trials){gap}{at}")
    shared = [w for w in a.workload if "shared" in report["workloads"][w]]
    if shared:
        print(f"SHARED GPU: another process held it while {', '.join(shared)} {'was' if len(shared) == 1 else 'were'} "
              "timed, so those times do not compare with another job's")
    audited = [r for r in results if "ops" in r]
    if audited:
        libs = collections.Counter(op[4] for r in audited for op in r["ops"] if op[4])
        detail = ", ".join(f"{k} {v}" for k, v in libs.most_common())
        share = statistics.mean(r["lib_share"] for r in audited)
        print(f"library kernels: {sum(libs.values())} launches in {len(audited)} calls{' (' + detail + ')' if detail else ''}, "
              f"{100 * share:.0f}% of GPU time on average")
    for r in results:
        r.pop("rows", None)
    finish()


if __name__ == "__main__":
    main()
