"""The physical lower bound of a call's time, from the work it must do and what the GPUs can do per second.

A task's spec.py may define

    bounds(workload) -> {"flops": float, "bytes": float, "gpus": int}    optionally also "math": one of PEAKS ("f16")

lower bounds for one call of run() at that workload: the floating-point operations any correct algorithm performs
(its convolutions and matrix products) and the bytes the busiest GPU must move to or from its memory at least once,
and the GPUs the call runs on (default 1). The bound time is then

    bound = max(flops / (gpus x peak FLOP/s of the chosen math), bytes / memory bandwidth)

with the peaks of the GPU it runs on at its SM clock: a call cannot be faster than the slower of the two, even with
its FLOPs spread evenly over its GPUs and nothing else to do, so the gap, measured / bound, says how much of the way
remains. The peaks are estimates, derived as SM count x SM clock x per-SM rate per clock (DEVICES), not measured; they
are the same for every solution, so gaps compare. The SM clock is the one GUILD_SM_CLOCK_MHZ names, where a task's
environment sets it, else the one DEVICES gives the GPU, else the GPU's highest SM clock: GB200 clocks are not locked,
and read low while they idle.

    python tools/bounds.py [--spec spec.py]      the bound of every workload on this GPU (or the reference device)
"""

import argparse
import math
import os
import subprocess

# Per compute capability: dense per-SM throughput per clock, and the memory bandwidth. Estimates from the
# architecture's published figures; the memory clock does not follow the SM clock. `sm_clock_mhz`, where given, is the
# clock to assume instead of the GPU's.
DEVICES = {
    (10, 0): {
        "name": "Blackwell, compute capability 10.0",
        # FLOP per SM per clock, dense (no structured sparsity), fp32 accumulation.
        "per_sm_clk": {
            "f16": 8192,  # FP16/BF16 tensor cores (tcgen05.mma kind::f16)
            "tf32": 4096,  # TF32 tensor cores (kind::tf32)
            "f32": 256,  # FP32 FMA on the CUDA cores: 128 lanes x 2
        },
        "bytes_per_s": 8.0e12,  # HBM3e, about 8 TB/s per GPU
    },
}
PEAKS = {"f16": "16-bit tensor cores", "tf32": "TF32 tensor cores", "f32": "FP32 CUDA cores"}
# Where no GPU is visible (on a laptop, say): a GB200 GPU at its highest SM clock.
REFERENCE_DEVICE = {"capability": (10, 0), "sms": 152, "clock_mhz": 2062}


def sm_clock_mhz(uuid=None):
    """The highest SM clock of the GPU with this UUID, or the reference device's."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits"]
        if uuid:
            cmd.insert(1, f"--id={uuid}")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.split()
        return float(out[0]) if out else REFERENCE_DEVICE["clock_mhz"]
    except (OSError, ValueError, subprocess.SubprocessError):
        return REFERENCE_DEVICE["clock_mhz"]


def assumed_clock_mhz():
    """The SM clock a task's environment says the bound assumes (GUILD_SM_CLOCK_MHZ), or None where it says none."""
    try:
        value = float(os.environ.get("GUILD_SM_CLOCK_MHZ", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def device():
    """{"capability", "sms", "clock_mhz"} of CUDA device 0, or REFERENCE_DEVICE where there is none. It asks the GPU's
    properties only, which makes no CUDA context: a process that times others must hold none."""
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            uuid = str(getattr(p, "uuid", "") or "")
            uuid = uuid if not uuid or uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"
            capability = (p.major, p.minor)
            clock = assumed_clock_mhz() or DEVICES.get(capability, {}).get("sm_clock_mhz") or sm_clock_mhz(uuid)
            return {"capability": capability, "sms": p.multi_processor_count, "clock_mhz": clock}
    except ImportError:
        pass
    return dict(REFERENCE_DEVICE)


def peaks(dev):
    """{"f16": FLOP/s, "tf32": ..., "f32": ..., "bytes_per_s": B/s} of one GPU, or None when DEVICES lacks it."""
    table = DEVICES.get(tuple(dev["capability"]))
    if table is None:
        return None
    hz = dev["clock_mhz"] * 1e6
    out = {k: v * dev["sms"] * hz for k, v in table["per_sm_clk"].items()}
    out["bytes_per_s"] = table["bytes_per_s"]
    return out


def bound_ms(work, peak):
    """The bound time in ms of one call doing `work` ({"flops", "bytes"[, "gpus", "math"]}) on GPUs with these peaks."""
    math_kind = work.get("math", "f16")
    if math_kind not in PEAKS:
        raise ValueError(f"bounds(): math must be one of {sorted(PEAKS)}, not {math_kind!r}")
    flops, nbytes = float(work.get("flops", 0.0)), float(work.get("bytes", 0.0))
    gpus = int(work.get("gpus", 1))
    return 1e3 * max(flops / (gpus * peak[math_kind]), nbytes / peak["bytes_per_s"])


def describe(dev, peak):
    """One line on how the bound is computed here."""
    return (f"bound = max(FLOPs / (GPUs x {peak['f16'] / 1e15:.2f} PFLOP/s) ({PEAKS['f16']}, dense), bytes / "
            f"{peak['bytes_per_s'] / 1e12:.1f} TB/s), estimated for {dev['sms']} SMs per GPU at "
            f"{dev['clock_mhz']:.0f} MHz")


def workload_bounds(spec, workloads, dev=None):
    """{workload: bound ms} for a spec that defines bounds(), with the device and its peaks; ({}, dev, None) when the
    spec has no bounds() or the device no peak figures."""
    dev = dev or device()
    peak = peaks(dev)
    fn = getattr(spec, "bounds", None)
    if fn is None or peak is None:
        return {}, dev, peak
    return {w: bound_ms(fn(w), peak) for w in workloads}, dev, peak


def geomean(xs):
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else math.nan


def main():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", default=os.path.join(os.environ.get("CHECKS", here), "spec.py"))
    a = p.parse_args()
    module_spec = importlib.util.spec_from_file_location("spec", a.spec)
    spec = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(spec)
    workloads = [w for w, e in spec.WORKLOADS.items() if e.get("timed", True)]
    bounds, dev, peak = workload_bounds(spec, workloads)
    if peak is None:
        raise SystemExit(f"no peak figures for compute capability {dev['capability']}")
    if not bounds:
        raise SystemExit(f"{a.spec} defines no bounds(workload)")
    print(describe(dev, peak))
    for w, ms in bounds.items():
        work = spec.bounds(w)
        print(f"{w:<10} {ms:10.1f} ms   {work['flops'] / 1e15:8.4f} PFLOP over {work.get('gpus', 1)} GPU(s)   "
              f"{work['bytes'] / 1e9:6.2f} GB")
    print(f"geomean {geomean(bounds.values()):.1f} ms over {len(bounds)} workloads")


if __name__ == "__main__":
    main()
