"""Correctness, timing and library-kernel audit of a solution: the Wan2.2 VAE decode of latents into video, on one, two
or four GB200 GPUs of one node. The task's CI runs this same file (see ci.py).

The task (spec.py has the details)
--------
A solution decodes latents z [1, 48, T, 44, 78] (bf16) into the video [1, 3, 4T-3, 704, 1248] (bf16, in [-1, 1]) with
the Wan2.2 VAE's decoder, as diffusers' AutoencoderKLWan decodes them (reference.py in the kernel repo). It is a
directory (default solution/, see loader.py) compiled into one extension module that exports

    run(z, weights, rank, world, pg, out)                    positional, destination passing
    prepare(weights, rank, world, pg) -> list[Tensor]       optional

called in one process per GPU: z the full latents on every rank; weights the decoder's 110 bf16 tensors on the rank's
GPU, in the order of spec.WEIGHTS (or what prepare returned); pg torch.distributed's NCCL ProcessGroup over the ranks;
out the video on rank 0, from torch.empty and never zeroed (every element must be written), and an empty tensor on the
other ranks. When run has returned on every rank, rank 0's out holds the video. prepare, if exported, runs once per
process before any timed call and is not timed; its result replaces the weights in every run.

The workloads (spec.WORKLOADS) are clips decoded at a world size, named <clip>x<world>. A clip is the first T latent
frames of one of three real 20 s clips, named by its video frames: f1, f5, f17 (T = 1, 2, 5) and f121, f241, f481
(T = 31, 61, 121). The long clips at 1, 2 and 4 GPUs are the nine timed workloads (f121x1 ... f481x4), whose geomean
is the objective; the short clips at 1 and 4 GPUs (f1x1 ... f17x4) are checked and never timed.

Running it
----------
In a job with as many GPUs as the largest world size it runs, with the task's data at $DATA (default /data), from the
repo root:

    python tools/bench.py                             every workload this job has the GPUs for: check, audit, time
    python tools/bench.py --workload f5x1 f17x1       a subset; the short clips take seconds
    python tools/bench.py --workload f121x1 --no-time correctness only
    python tools/bench.py --kernels                   also list the audited calls' kernels by GPU time
    python tools/bench.py --solution other            another solution directory

A run
-----
The solution is built once, by this process, exactly as it is deployed (loader.py; a development copy, tools/, takes a
build its node already made of the same sources, flags and toolchain from the node's cache, unless --no-cache). Then,
per world size in turn (1, 2, 4), torch.multiprocessing spawns one process per rank, rank r on GPU r of the job's
CUDA_VISIBLE_DEVICES. Each initializes NCCL over 127.0.0.1 with its device (the default group, the solution's pg) and
a gloo group for the harness's own barriers, imports the build (in its working directory, as deployed), moves the
weights ($DATA/decoder_weights_bf16.pt, checked against spec.WEIGHTS) to its GPU and calls prepare, if the module
exports it. The ranks then run that world size's workloads in spec.WORKLOADS' order. A rank that raises, dies or
hangs (WORKLOAD_SEC per workload, SESSION_START_SEC before the first) fails the workload it was at: all the ranks'
processes are stopped, and the workloads after it run in new ones. This process runs no solution code.

Every call gets z at a fresh address (a copy of the clip's latents) and out at a fresh address, 256-byte aligned and
filled with NaN on rank 0, so that an element run leaves unwritten fails the check. run must not write z, nor the
weights (compared after each workload with what prepare left), nor start threads: each fails the workload.

Correctness
-----------
The first call of a workload, its warm-up, is checked on rank 0 against the clip's fp32 decode ($DATA/ref_<clip>.pt):
every element finite and within [-1, 1]; and with d = out - ref in fp32, max |d| <= min(2, 3 x bf16_max_abs) and
rms(d) <= 3 x bf16_rms, the error of the release's bf16 decode of that clip ($DATA/calibration.json; spec.py,
Correctness). The line shows max |d| and rms(d), each with its allowance and the share of it used, and the PSNR,
10 log10(4 / mse). Every later call's output (the audited call's, the trials') must agree with the checked one:
lose none of the elements it wrote (none left NaN or overflowed), and be within the same allowance of it everywhere
else. A race, or a state carried from call to call, fails the workload.

Timing (the timed workloads)
------
After the warm-up come the trials: 3 when the warm-up took under 1.5 s, else 2. A trial is one call timed by wall
clock between barriers on all ranks: synchronize the device, barrier, start, run, synchronize, barrier, stop. The
time of record is rank 0's; the latency of record the mean of the trials, and the fastest trial is shown too. While
the trials run, this process reads the SM clock of the world's GPUs through NVML every CLOCK_PERIOD_S (a rank inside
run may hold its GIL for the whole call, so the ranks do not read it themselves); the mean of rank 0's GPU is shown
with the latency, and the range over the world's GPUs where they differ by 2% or more: the clocks are not locked
alike on every GPU, and this says at what clock the times were taken. Which processes hold the world's GPUs is
asked (nvidia-smi) before every trial and after the last: a workload timed while another process held one of them is
marked SHARED, since its times do not compare with those of another job.

Audit
-----
At world 1, one more call per timed clip runs under CUPTI, and each kernel is attributed to the library that launched
it (see audit.py). Library kernels (cuBLAS/cuBLASLt, cuDNN, ATen) are allowed as placeholders: their count per call
and their share of the call's GPU time show where the solution still depends on them. NCCL's kernels are
communication, reported apart and not counted as placeholders. --kernels lists the audited calls' kernels.

Bound
-----
Each timed workload's latency is shown with its bound (bounds.py with spec.bounds: the decode's FLOPs spread evenly
over the world's GPUs at the 16-bit tensor-core peak, or the video and the weights at the memory bandwidth, whichever
is slower) and the gap, latency / bound. The peak assumes the SM clock GUILD_SM_CLOCK_MHZ where the environment sets
it (the guild's: 2062 MHz, the GB200's highest), else the GPU's highest.

In CI, and from tools/
----------------------
ci.py runs this file in child processes: --save DIR has rank 0 save each workload's checked output there (node-local
scratch) instead of checking it, so that CI judges it in a process that runs no solution code; --no-check skips the
check (the timing runs), --no-audit the audit, and --json writes the report. The kernel repo holds copies of this
file, spec.py, loader.py, audit.py and bounds.py in tools/, which run as they do here: they read the data at $DATA
(default /data) and the solution from solution/ under the current directory.
"""

import argparse
import collections
import ctypes
import functools
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import timedelta
from multiprocessing import connection

# The caching allocator stays as torch configures it by default: without expandable segments, whose memory
# cudaIpcGetMemHandle cannot export, since a solution may share its buffers between ranks by CUDA IPC.
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit  # noqa: E402
import bounds  # noqa: E402
import loader  # noqa: E402
import spec  # noqa: E402

# Trials per timed workload, fewer when one call takes long.
TRIALS, LONG_TRIALS, LONG_CALL_S = 3, 2, 1.5
# How often this process reads rank 0's SM clock while ranks run.
CLOCK_PERIOD_S = 0.02
NVML_CLOCK_SM = 1  # the nvmlClockType_t of the SM clock
# Limits that stop a runaway: a session's start (NCCL, the module, the weights, prepare), and one workload (its calls,
# its check or its save); a call of the release's bf16 decode takes 15 s at the longest clip.
SESSION_START_SEC = 900
WORKLOAD_SEC = 1800
STOP_GRACE_SEC = 15
# Elements of a video compared at once, in fp32 (1 GiB per operand).
CHUNK_ELEMENTS = 1 << 28


def gpu_holders():
    """{GPU UUID (lower case, without "GPU-"): ids of the processes that hold it} now, as nvidia-smi gives them, or
    None when it cannot say."""
    try:
        got = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if got.returncode != 0:
        return None
    out = collections.defaultdict(set)
    for line in got.stdout.splitlines():
        gpu, _, pid = line.partition(",")
        if pid.strip().isdigit():
            out[gpu.strip().lower().removeprefix("gpu-")].add(int(pid))
    return out


def others(questions, uuids, ours):
    """How many processes besides this run's held one of these GPUs at any of the questions (gpu_holders' answers), or
    None when one could not be asked. This run's processes are known by their ids where nvidia-smi gives the ids this
    process sees (one PID namespace, the usual case: then a rank that also holds another rank's GPU is still one of
    ours); otherwise each GPU is taken to hold one of ours, the rank on it."""
    if not questions or any(q is None for q in questions):
        return None
    total = 0
    for uuid in uuids:
        seen = set().union(*(q.get(uuid, set()) for q in questions))
        total += len(seen - ours) if seen & ours else max(0, len(seen) - 1)
    return total


@functools.cache
def sm_clock(uuid):
    """A function that reads the SM clock of the GPU with this UUID now, in MHz (None when a read fails), or None where
    NVML cannot be loaded or does not know the GPU. NVML in this process takes microseconds per read."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            return None
        handle, mhz = ctypes.c_void_p(), ctypes.c_uint()
        if not uuid or nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode(), ctypes.byref(handle)) != 0:
            return None
    except (OSError, AttributeError):
        return None

    def read():
        ok = nvml.nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, ctypes.byref(mhz)) == 0
        return mhz.value if ok and mhz.value else None

    return read if read() else None


class Clock:
    """The SM clocks of the world's GPUs (rank order), read every CLOCK_PERIOD_S by a thread of this process, as
    (monotonic time, [MHz or None per GPU])."""

    def __init__(self, uuids):
        self.reads, self.samples, self.done = [sm_clock(u) for u in uuids], [], threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True) if any(self.reads) else None
        if self.thread:
            self.thread.start()

    def _run(self):
        while not self.done.wait(CLOCK_PERIOD_S):
            self.samples.append((time.monotonic(), [read() if read else None for read in self.reads]))

    def stop(self):
        self.done.set()
        if self.thread:
            self.thread.join()

    def means(self, windows):
        """Per GPU, the mean of its reads within these (start, stop) windows of time.monotonic(), or None without
        any."""
        inside = [mhz for t, mhz in list(self.samples) if any(a <= t <= b for a, b in windows)]
        out = []
        for i in range(len(self.reads)):
            got = [m[i] for m in inside if m[i]]
            out.append(statistics.mean(got) if got else None)
        return out


class Pool:
    """A workload's buffers for one tensor: each call's at a fresh address, shifted by multiples of 256 bytes."""

    ALIGN, MAX_SHIFT = 256, 2048

    def __init__(self, shape, dtype, device, calls, seed):
        rng = random.Random(seed)
        self.offsets = [0]
        for _ in range(calls - 1):
            self.offsets.append(self.offsets[-1] + rng.randint(1, self.MAX_SHIFT // self.ALIGN))
        self.shape, self.numel = tuple(shape), math.prod(shape)
        self.block = max(1, self.ALIGN // torch.empty(0, dtype=dtype).element_size())
        self.buffer = torch.empty(self.numel + self.offsets[-1] * self.block, dtype=dtype, device=device)
        self.calls = 0

    def next(self):
        start = self.offsets[self.calls] * self.block
        self.calls += 1
        return self.buffer[start:start + self.numel].view(self.shape)


def fingerprint(tensors):
    """Each tensor's bytes summed as integers: whether run() wrote the weights."""
    out = []
    for t in tensors:
        raw = t.detach().contiguous().view(-1).view(torch.uint8)
        try:
            raw = raw.view(torch.int32)
        except RuntimeError:
            pass
        out.append(int(raw.sum(dtype=torch.int64)))
    return out


def judge(out, ref, allowed):
    """A video against the reference, or a later call's video against the checked one's: {"passed", "nonfinite",
    "outside" (elements beyond [-1, 1]), "max_abs", "rms", "psnr_db", "max_allowed", "rms_allowed"}, d = out - ref where
    both are finite (rms and PSNR over those elements, None where there is none), in fp32 on out's GPU (the current one
    for a video on the CPU), in chunks of frames, the squares summed in fp64."""
    device = out.device if out.is_cuda else torch.device("cuda", torch.cuda.current_device())
    step = max(1, CHUNK_ELEMENTS // max(1, out[:, :, :1].numel()))
    nonfinite = outside = compared = unmatched = 0
    max_abs = squares = 0.0
    for f in range(0, out.shape[2], step):
        x = out[:, :, f:f + step].to(device, torch.float32)
        y = ref[:, :, f:f + step].to(device, torch.float32)
        finite = torch.isfinite(x)
        nonfinite += int((~finite).sum())
        outside += int((x.abs() > 1).sum())
        # Where the checked call itself left an element unfinite, a later call is not compared there (the check said).
        both = finite & torch.isfinite(y)
        compared += int(both.sum())
        unmatched += int((~finite & torch.isfinite(y)).sum())
        d = torch.where(both, x - y, 0.0)
        max_abs = max(max_abs, float(d.abs().max()))
        squares += float(d.square().sum(dtype=torch.float64))
        del x, y, finite, both, d
    mse = squares / compared if compared else None
    rms = math.sqrt(mse) if compared else None
    passed = (nonfinite == 0 and outside == 0 and compared == out.numel() and max_abs <= allowed["max_abs"]
              and rms <= allowed["rms"])
    return {"passed": passed, "nonfinite": nonfinite, "unmatched": unmatched, "outside": outside,
            "max_abs": max_abs if compared else None, "rms": rms, "psnr_db": 10 * math.log10(4 / mse) if mse else None,
            "max_allowed": allowed["max_abs"], "rms_allowed": allowed["rms"]}


def verdict(c):
    """A check (judge's result) in a few words."""
    parts = []
    if c["nonfinite"]:
        parts.append(f"{c['nonfinite']} elements not finite (NaN: never written, or overflowed)")
    if c["outside"]:
        parts.append(f"{c['outside']} elements outside [-1, 1]")
    if c["max_abs"] is None:
        parts.append("no element to compare")
    else:
        parts.append(f"max |d| {c['max_abs']:.4g} of {c['max_allowed']:.4g} (x{c['max_abs'] / c['max_allowed']:.2f})")
        parts.append(f"rms {c['rms']:.4g} of {c['rms_allowed']:.4g} (x{c['rms'] / c['rms_allowed']:.2f})")
        parts.append(f"PSNR {c['psnr_db']:.1f} dB" if c["psnr_db"] is not None else "identical")
    return f"{'ok  ' if c['passed'] else 'FAIL'} ({', '.join(parts)})"


def run_workload(w, sol, weights, marks, rank, world, pg, harness, ranks, cfg):
    """One workload on one rank, in step with the other ranks (see the module docstring); rank 0's result."""
    entry = spec.WORKLOADS[w]
    clip, device = entry["clip"], torch.device("cuda", rank)
    timed = entry["timed"] and cfg["time"]
    audited = entry["timed"] and cfg["audit"] and world == 1
    calls = 1 + audited + (TRIALS if timed else 0)
    seed = list(spec.WORKLOADS).index(w)
    z0 = spec.make_inputs(w, device)[0]
    zs = Pool(z0.shape, z0.dtype, device, calls, seed)
    outs = Pool(spec.output_shape(w), torch.bfloat16, device, calls, seed + 1) if rank == 0 else None
    allowed = spec.allowance(clip)
    res = {"clip": clip, "world": world, "timed": entry["timed"]}
    problems = []
    threads = threading.active_count()

    def arguments():
        z = zs.next()
        z.copy_(z0)
        out = outs.next().fill_(math.nan) if outs else torch.empty(0, dtype=torch.bfloat16, device=device)
        return z, out

    def unchanged(z):
        if not torch.equal(z, z0) and "run() wrote its input z" not in problems:
            problems.append("run() wrote its input z")

    def call():
        z, out = arguments()
        torch.cuda.synchronize(device)
        dist.barrier(group=harness)
        start = time.monotonic()
        sol.fn(z, weights, rank, world, pg, out)
        torch.cuda.synchronize(device)
        dist.barrier(group=harness)
        stop = time.monotonic()
        unchanged(z)
        return out, start, stop

    # The warm-up, which is the call checked.
    out, start, stop = call()
    warm = stop - start
    first, worst = None, 0.0
    if rank == 0:
        if cfg["save"]:
            path = os.path.join(cfg["save"], f"{w}.pt")
            torch.save(out.cpu(), path)
            res["saved"] = path
        elif cfg["check"]:
            res["check"] = judge(out, spec.load_reference(clip), allowed)
        if calls > 1:
            first = out.clone()

    def agree(out, what):
        nonlocal worst
        # It agrees when it loses no element the checked call wrote, and is within the allowance of it elsewhere;
        # the checked call's own faults are its check's to report.
        c = judge(out, first, allowed)
        worst = max(worst, c["max_abs"] or 0.0)
        close = c["max_abs"] is None or (c["max_abs"] <= allowed["max_abs"] and c["rms"] <= allowed["rms"])
        if c["unmatched"] or not close:
            lost = f"{c['unmatched']} elements not finite where it was; " if c["unmatched"] else ""
            problems.append(f"{what}'s output does not agree with the checked call's: {lost}"
                            f"{verdict(c).removeprefix('FAIL').removeprefix('ok').strip()}")

    # World 1 alone: one call under CUPTI, which attributes every kernel.
    if audited:
        z, out = arguments()
        ops = audit.trace(sol.fn, z, weights, rank, world, pg, out)
        unchanged(z)
        agree(out, "the audited call")
        res |= {"kernels": audit.kernels(ops), "launches": sum(1 for op in ops if op.kind == "kernel"),
                "lib": sum(1 for op in ops if audit.placeholder(op.library)), "lib_share": audit.library_share(ops),
                "comm": sum(1 for op in ops if op.library in audit.COMMUNICATION), "comm_share": audit.comm_share(ops)}
        del ops

    if timed:
        n = [TRIALS if warm < LONG_CALL_S else LONG_TRIALS]
        dist.broadcast_object_list(n, src=0, group=harness)
        questions, trials, windows = [], [], []
        for k in range(n[0]):
            if rank == 0:
                questions.append(gpu_holders())
            out, start, stop = call()
            trials.append(1e3 * (stop - start))
            windows.append((start, stop))
            if rank == 0:
                agree(out, f"trial {k + 1}")
        if rank == 0:
            questions.append(gpu_holders())
            res |= {"trials": trials, "ms": statistics.mean(trials), "min_ms": min(trials), "windows": windows}
            held = others(questions, [uuid for _, uuid in ranks], {pid for pid, _ in ranks} | {os.getppid()})
            if held:
                res["shared"] = held
    if first is not None:
        res["agree_max_abs"] = worst
    if fingerprint(weights) != marks:
        problems.append("run() wrote its weights (what prepare returned is read-only to run: keep scratch elsewhere)")
    if threading.active_count() > threads:
        problems.append(f"the solution started threads ({threading.active_count() - threads} more)")

    # What every rank found, so that rank 0 reports it.
    found = [None] * world
    dist.all_gather_object(found, problems, group=harness)
    text = [p if world == 1 else f"rank {r}: {p}" for r, ps in enumerate(found) for p in ps]
    if text and rank == 0:
        res["error"] = "; ".join(text)
    return res


def worker(rank, world, port, cfg, conn):
    """The process of one rank (see A run in the module docstring): it reports to the process that spawned it through
    conn, ("ready", rank, uuid, prepare seconds), ("start", workload) and ("result", workload, result) on rank 0,
    ("done", rank), or ("error", rank, workload or None, phase, traceback), after which it exits at once."""
    phase, current = "start", None
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        timeout = timedelta(seconds=WORKLOAD_SEC)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                                device_id=device, timeout=timeout)
        harness = dist.new_group(backend="gloo", timeout=timeout)
        pg = dist.group.WORLD
        phase = "load"
        sol = loader.import_built(cfg["workdir"], cfg["spec"])
        phase = "weights"
        weights = spec.load_weights(device)
        phase = "prepare"
        prepare_s = None
        if sol.prepare is not None:
            torch.cuda.synchronize(device)
            start = time.monotonic()
            got = sol.prepare(weights, rank, world, pg)
            torch.cuda.synchronize(device)
            prepare_s = time.monotonic() - start
            if not isinstance(got, (list, tuple)) or not all(isinstance(t, torch.Tensor) for t in got):
                raise TypeError(f"prepare returned {type(got).__name__}, not a list of tensors")
            weights = list(got)
        uuid = str(torch.cuda.get_device_properties(device).uuid).lower().removeprefix("gpu-")
        ranks = [None] * world
        dist.all_gather_object(ranks, (os.getpid(), uuid), group=harness)
        conn.send(("ready", rank, uuid, prepare_s))
        phase = "workload"
        marks = fingerprint(weights)
        for w in cfg["workloads"]:
            current = w
            if rank == 0:
                conn.send(("start", w))
            res = run_workload(w, sol, weights, marks, rank, world, pg, harness, ranks, cfg)
            if rank == 0:
                conn.send(("result", w, res))
            current = None
            torch.cuda.empty_cache()
        phase = "end"
        dist.barrier(group=harness)
        conn.send(("done", rank))
        dist.destroy_process_group()
    except BaseException:
        try:
            conn.send(("error", rank, current, phase, traceback.format_exc(limit=8)))
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            # At once: NCCL's teardown would wait for the other ranks, which may wait for this one.
            os._exit(1)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stop(procs, grace=0.0):
    """Ends rank processes: those that do not exit within `grace` seconds get SIGTERM, then SIGKILL after
    STOP_GRACE_SEC."""
    deadline = time.monotonic() + grace
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.terminate()
    deadline = time.monotonic() + STOP_GRACE_SEC
    for p in procs:
        p.join(max(0.1, deadline - time.monotonic()))
        if p.is_alive():
            p.kill()
            p.join(10)


def exit_text(code):
    if code is None:
        return "did not exit"
    if code < 0:
        import signal

        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


def session(world, workloads, cfg, report, show):
    """One set of rank processes running these workloads at one world size, in order; results go into report as they
    arrive, and show(w) prints each. Returns the workloads it did not reach: after a failure at one workload (which gets
    the error), those after it; [] when every one ran, or when the failure came before any (every one gets it)."""
    ctx = torch.multiprocessing.get_context("spawn")
    port = free_port()
    pipes = [ctx.Pipe(duplex=False) for _ in range(world)]
    procs = [ctx.Process(target=worker, args=(r, world, port, dict(cfg, workloads=workloads), pipes[r][1]),
                         daemon=True) for r in range(world)]
    for p in procs:
        p.start()
    for _, send in pipes:
        send.close()
    receivers = {pipes[r][0]: r for r in range(world)}
    done, errors, finished, uuids = set(), [], [], {}
    current, clock, failure = None, None, None
    deadline = time.monotonic() + SESSION_START_SEC
    started = time.monotonic()

    def handle(msg):
        nonlocal current, clock, deadline
        kind = msg[0]
        if kind == "ready":
            _, rank, uuid, prepare_s = msg
            uuids[rank] = uuid
            if rank == 0:
                report.setdefault("prepare_s", {})[str(world)] = prepare_s
            if len(uuids) == world:
                clock = Clock([uuids[r] for r in range(world)])
                ready = f", prepare {report['prepare_s'][str(world)]:.1f} s" if prepare_s is not None else ""
                print(f"world {world}: {world} rank{'s' if world > 1 else ''} up in {time.monotonic() - started:.0f} s"
                      f"{ready}", flush=True)
        elif kind == "start":
            current = msg[1]
            deadline = time.monotonic() + WORKLOAD_SEC
        elif kind == "result":
            _, w, res = msg
            windows = res.pop("windows", None)
            means = clock.means(windows) if clock and windows else []
            if means and means[0]:
                res["clock_mhz"] = means[0]
            if world > 1 and means and all(means):
                res["world_clock_mhz"] = [min(means), max(means)]
            report["workloads"][w] = res
            finished.append(w)
            current = None
            deadline = time.monotonic() + WORKLOAD_SEC
            show(w)
        elif kind == "error":
            errors.append(msg[1:])
        elif kind == "done":
            done.add(msg[1])

    try:
        while len(done) < world:
            # A rank that exited cleanly may still have its last messages in its pipe: it fails once that is read.
            bad = [(r, p.exitcode) for r, p in enumerate(procs) if r not in done and p.exitcode is not None
                   and (p.exitcode != 0 or pipes[r][0] not in receivers)]
            if bad:
                failure = "; ".join(f"rank {r} {exit_text(code)}" for r, code in bad)
                break
            left = deadline - time.monotonic()
            if left <= 0:
                failure = (f"timed out after {WORKLOAD_SEC} s" if current else
                           f"the ranks did not start within {SESSION_START_SEC} s")
                break
            waits = list(receivers) + [p.sentinel for p in procs if p.exitcode is None]
            for obj in connection.wait(waits, timeout=min(left, 5.0)):
                if obj in receivers:
                    try:
                        handle(obj.recv())
                    except (EOFError, OSError):
                        del receivers[obj]
        # What the other ranks still say, such as the error that made one exit.
        drain = time.monotonic() + 2.0
        while failure and receivers and time.monotonic() < drain:
            for obj in connection.wait(list(receivers), timeout=max(0.0, drain - time.monotonic())):
                try:
                    handle(obj.recv())
                except (EOFError, OSError):
                    del receivers[obj]
    finally:
        stop(procs, 0.0 if failure else STOP_GRACE_SEC)
        if clock:
            clock.stop()
    if not failure:
        return []

    # The first error a rank reported says what failed; else how a rank ended.
    if errors:
        rank, at, phase, text = errors[0]
        if phase == "load":
            report["load_error"] = text.strip().splitlines()[-1]
        text = (f"rank {rank}: " if world > 1 else "") + text.strip()
    else:
        text = failure
    rest = [w for w in workloads if w not in finished]
    at = current or (errors and errors[0][1])
    if at not in rest:
        at = rest[0] if rest and finished else None
    if not at or report.get("load_error"):
        for w in rest:
            report["workloads"][w] = {"error": text, "world": world, "clip": spec.WORKLOADS[w]["clip"],
                                      "timed": spec.WORKLOADS[w]["timed"]}
            show(w)
        return []
    report["workloads"][at] = {"error": text, "world": world, "clip": spec.WORKLOADS[at]["clip"],
                               "timed": spec.WORKLOADS[at]["timed"]}
    show(at)
    return rest[rest.index(at) + 1:]


def kernel_lines(res, limit=25):
    """The audited call's kernels by GPU time, one line each."""
    rows = res.get("kernels", [])
    total = sum(r[3] for r in rows) or 1
    lines = [f"    {'[' + lib + ']' if lib else '':<9} {ns / 1e6:10.2f} ms {100 * ns / total:5.1f}%  x{n:<6} "
             f"{name if name.startswith('(') else audit.short_name(name, 110)}" for lib, name, n, ns in rows[:limit]]
    if len(rows) > limit:
        rest = rows[limit:]
        lines.append(f"    {'':<9} {sum(r[3] for r in rest) / 1e6:10.2f} ms "
                     f"{100 * sum(r[3] for r in rest) / total:5.1f}%  {len(rest)} more kinds")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solution", default="solution", help="the solution directory (default: solution)")
    p.add_argument("--workload", nargs="*", help="workloads to run (default: every one this job has the GPUs for)")
    p.add_argument("--no-time", action="store_true", help="correctness only: no audit, no trials")
    p.add_argument("--no-check", action="store_true", help="do not check the outputs (CI's timing runs)")
    p.add_argument("--no-audit", action="store_true", help="no call under CUPTI")
    p.add_argument("--kernels", action="store_true", help="list the audited calls' kernels by GPU time")
    p.add_argument("--build-dir", help="write and build the solution here; reused when it holds the same build")
    p.add_argument("--no-cache", action="store_true", help="build fresh, without the node's cache of builds")
    p.add_argument("--save", help="CI: save each workload's checked output here instead of checking it")
    p.add_argument("--json", help="write the results to this file")
    a = p.parse_args()
    unknown = [w for w in a.workload or [] if w not in spec.WORKLOADS]
    if unknown:
        p.error(f"unknown workloads {unknown}; known: {list(spec.WORKLOADS)}")
    # Absolute paths, the data's too: the ranks run in the solution's working directory.
    solution = os.path.abspath(a.solution)
    if os.path.isfile(solution):
        solution = os.path.dirname(solution)
    a.build_dir, a.save, a.json = (x and os.path.abspath(x) for x in (a.build_dir, a.save, a.json))
    os.environ["DATA"] = os.path.abspath(spec.data_dir())
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

    # The workloads: those named, or every one this job has the GPUs for.
    gpus = torch.cuda.device_count()
    chosen = a.workload or [w for w, e in spec.WORKLOADS.items() if e["world"] <= gpus]
    skipped = [w for w in spec.WORKLOADS if w not in chosen and not a.workload]
    if skipped:
        print(f"skipped {', '.join(skipped)}: this job has {gpus} GPU{'s' if gpus != 1 else ''}")
    order = [w for w in spec.WORKLOADS if w in chosen]
    timed = [w for w in order if spec.WORKLOADS[w]["timed"] and not a.no_time]
    limits, dev, peak = bounds.workload_bounds(spec, timed)
    if limits:
        print(bounds.describe(dev, peak))

    try:
        workdir, build_spec, seconds = loader.stage(solution, a.build_dir,
                                                    cache=None if a.no_cache else loader.cache_dir())
    except Exception as e:
        report["load_error"] = str(e) if isinstance(e, loader.SolutionError) else traceback.format_exc(limit=6)
        print(f"FAILED to build {solution}:\n{report['load_error']}", flush=True)
        finish()
        sys.exit(1)
    report |= {"language": build_spec["languages"][0], "build_s": seconds}
    built = f"built in {seconds:.1f} s" if seconds is not None else "a build reused"
    print(f"{build_spec['entry_point']} ({', '.join(build_spec['languages'])}): {built}", flush=True)

    width = max(len(w) for w in spec.WORKLOADS)

    def show(w):
        res = report["workloads"][w]
        line = f"{w:<{width}}"
        if "ms" in res:
            at = f", SM clock {res['clock_mhz']:.0f} MHz" if "clock_mhz" in res else ""
            low, high = res.get("world_clock_mhz", (None, None))
            if low and "clock_mhz" in res and high - low >= 0.02 * res["clock_mhz"]:
                at += f", the world's GPUs {low:.0f}-{high:.0f} MHz"
            line += f" {res['ms']:9.1f} ms (fastest trial {res['min_ms']:.1f}{at})"
            if w in limits:
                res |= {"bound_ms": limits[w], "gap": res["ms"] / limits[w]}
                line += f"  bound {limits[w]:.1f} ms, gap x{res['gap']:.2f}"
        if "check" in res:
            line += "  " + verdict(res["check"])
        elif "saved" in res:
            line += "  saved"
        if not res.get("timed", True):
            line += "  (correctness only)"
        if "lib" in res:
            line += f"  library kernels {res['lib']} of {res['launches']} ({100 * res['lib_share']:.0f}% of GPU time)"
        if res.get("comm"):
            line += f", NCCL {res['comm']} ({100 * res['comm_share']:.0f}%)"
        if "shared" in res:
            line += f"  SHARED: {res['shared']} other process(es) held a GPU while it was timed"
        if "error" in res:
            line += "  ERROR\n" + res["error"]
        print(line, flush=True)
        if a.kernels and "kernels" in res:
            print("\n".join(kernel_lines(res)), flush=True)
        finish()

    cfg = {"workdir": workdir, "spec": build_spec, "time": not a.no_time, "check": not a.no_check,
           "audit": not a.no_audit and (not a.no_time or a.kernels), "save": a.save}
    for world in sorted({spec.WORKLOADS[w]["world"] for w in order}):
        todo = [w for w in order if spec.WORKLOADS[w]["world"] == world]
        if world > gpus or report.get("load_error"):
            why = report.get("load_error") or f"needs {world} GPUs; this job has {gpus}"
            for w in todo:
                report["workloads"][w] = {"error": why, "world": world, "clip": spec.WORKLOADS[w]["clip"],
                                          "timed": spec.WORKLOADS[w]["timed"]}
                show(w)
            continue
        while todo:
            todo = session(world, todo, cfg, report, show)

    results = [report["workloads"][w] for w in order if w in report["workloads"]]
    times = [report["workloads"].get(w, {}).get("ms") for w in timed]
    if timed and all(times):
        gaps = [report["workloads"][w]["gap"] for w in timed if "gap" in report["workloads"][w]]
        gap = f", gap to the bound x{bounds.geomean(gaps):.2f}" if len(gaps) == len(timed) else ""
        clocks = [report["workloads"][w].get("clock_mhz") for w in timed]
        at = f", mean SM clock {statistics.mean(clocks):.0f} MHz" if all(clocks) else ""
        print(f"geomean {bounds.geomean(times):.1f} ms over {len(timed)} timed workloads (mean of trials){gap}{at}")
    shared = [w for w in order if "shared" in report["workloads"].get(w, {})]
    if shared:
        print(f"SHARED GPU: another process held one while {', '.join(shared)} "
              f"{'was' if len(shared) == 1 else 'were'} timed, so those times do not compare with another job's")
    checked = [r for r in results if "check" in r]
    if checked:
        ok = sum(1 for r in checked if r["check"]["passed"] and "error" not in r)
        print(f"correctness: ok at {ok} of {len(checked)} workloads checked")
    failed = [w for w in order if "error" in report["workloads"].get(w, {})]
    if failed:
        print(f"errors at {', '.join(failed)}")
    audited = [r for r in results if "kernels" in r]
    if audited:
        libs = collections.Counter()
        for r in audited:
            for lib, _, n, _ in r["kernels"]:
                if audit.placeholder(lib):
                    libs[lib] += n
        detail = ", ".join(f"{k} {v}" for k, v in libs.most_common())
        share = statistics.mean(r["lib_share"] for r in audited)
        print(f"library kernels: {sum(libs.values())} launches in {len(audited)} audited calls"
              f"{' (' + detail + ')' if detail else ''}, {100 * share:.0f}% of GPU time on average")
    finish()


if __name__ == "__main__":
    main()
