"""The physical lower bound of a call's time, from the work it must do and what the GPU can do per second.

A task's spec.py may define

    bounds(workload) -> {"flops": float, "bytes": float}      optionally also "math": one of PEAKS (default "f16")

lower bounds for one call of run() at that workload: the floating-point operations any correct algorithm performs
(usually its matrix multiplications) and the bytes it must move to or from GPU memory at least once (its inputs read
and its outputs written). The bound time is then

    bound = max(flops / peak FLOP/s of the chosen math, bytes / memory bandwidth)

with the peaks of the GPU it runs on at its SM clock: a call cannot be faster than the slower of the two, so the gap,
measured / bound, says how much of the way remains. The peaks are estimates, derived as SM count x SM clock x
per-SM rate per clock (DEVICES), not measured; they are the same for every solution, so gaps compare. The SM clock is
the one GUILD_SM_CLOCK_MHZ names, where a task's environment sets it (for GPUs whose clocks are not locked, which read
low while they idle), else the one DEVICES gives the GPU, else the GPU's current clock (the locked one on the nodes).

A call whose stages run one after another, each bound by a different unit, is not described by one bound over its
totals. bounds() may also give them, under "stages": [{"name", "match", "flops", "exps", "bytes"}], match a regular
expression on the names of the stage's kernels and exps the exponentials it evaluates. stage_floors() turns each into
its floors at the same clock: the tensor floor (its FLOPs at the tensor-core peak), the SFU floor (every exponential on
the MUFU, which a kernel can undercut by computing some on the FMA pipe instead, so it is a floor of that unit and not
of the stage) and the memory floor. A breakdown reads each stage's kernels against them.

    python bounds.py [--spec spec.py]      the bound of every workload on this GPU (or the reference device)
"""

import argparse
import math
import os
import subprocess

# Per compute capability: dense per-SM throughput per clock, and the memory bandwidth. Estimates from the
# architecture's published figures; the memory clock does not follow the SM clock lock. `sm_clock_mhz`, where given,
# is the clock to assume instead of the one the GPU reports.
DEVICES = {
    (10, 0): {
        "name": "Blackwell, compute capability 10.0",
        # FLOP per SM per clock, dense (no structured sparsity), fp32 accumulation.
        "per_sm_clk": {
            "f16": 8192,  # FP16/BF16 tensor cores (tcgen05.mma kind::f16)
            "tf32": 4096,  # TF32 tensor cores (kind::tf32)
            "f32": 256,  # FP32 FMA on the CUDA cores: 128 lanes x 2
        },
        "ex2_per_sm_clk": 16,  # MUFU.EX2 on the SFU: 4 per sub-partition per clock
        "bytes_per_s": 8.0e12,  # HBM3e, about 8 TB/s per GPU
    },
    (12, 1): {
        "name": "GB10 (DGX Spark), compute capability 12.1",
        "per_sm_clk": {
            "f16": 1024,  # FP16/BF16 tensor cores (mma.sync), fp32 accumulation at the full 16-bit rate
            "tf32": 512,  # TF32 tensor cores, half the 16-bit rate
            "f32": 256,  # FP32 FMA on the CUDA cores: 128 lanes x 2
        },
        "ex2_per_sm_clk": 16,  # MUFU.EX2 on the SFU: 4 per sub-partition per clock
        "bytes_per_s": 2.73e11,  # LPDDR5X-8533 on a 256-bit bus, about 273 GB/s, shared with the CPUs
        # Its SM clock cannot be locked and reads low while it idles: the clock its tensor cores hold under load, at
        # which a measured BF16 mma.sync loop ran 126 TFLOP/s.
        "sm_clock_mhz": 2570,
    },
}
PEAKS = {"f16": "16-bit tensor cores", "tf32": "TF32 tensor cores", "f32": "FP32 CUDA cores"}
# The SM clock the guild's nodes lock every GPU at (setup/node.sh), used when the GPU's clock cannot be read.
LOCKED_SM_CLOCK_MHZ = 1500
# Where no GPU is visible (on a laptop, say): the GPU of the nodes.
REFERENCE_DEVICE = {"capability": (10, 0), "sms": 152, "clock_mhz": LOCKED_SM_CLOCK_MHZ}


def sm_clock_mhz(uuid=None):
    """The current SM clock of the GPU with this UUID (the locked clock on the nodes), or LOCKED_SM_CLOCK_MHZ."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]
        if uuid:
            cmd.insert(1, f"--id={uuid}")
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.split()
        return float(out[0]) if out else LOCKED_SM_CLOCK_MHZ
    except (OSError, ValueError, subprocess.SubprocessError):
        return LOCKED_SM_CLOCK_MHZ


def assumed_clock_mhz():
    """The SM clock a task's environment says the bound assumes (GUILD_SM_CLOCK_MHZ), or None where it says none."""
    try:
        value = float(os.environ.get("GUILD_SM_CLOCK_MHZ", ""))
    except ValueError:
        return None
    return value if value > 0 else None


def device():
    """{"capability", "sms", "clock_mhz"} of CUDA device 0, or REFERENCE_DEVICE where there is none."""
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
    """{"f16": FLOP/s, "tf32": ..., "f32": ..., "bytes_per_s": B/s, "ex2_per_s": exponentials/s} of a device, or None
    when DEVICES lacks it; ex2_per_s only where DEVICES gives the SFU's rate."""
    table = DEVICES.get(tuple(dev["capability"]))
    if table is None:
        return None
    hz = dev["clock_mhz"] * 1e6
    out = {k: v * dev["sms"] * hz for k, v in table["per_sm_clk"].items()}
    out["bytes_per_s"] = table["bytes_per_s"]
    if "ex2_per_sm_clk" in table:
        out["ex2_per_s"] = table["ex2_per_sm_clk"] * dev["sms"] * hz
    return out


def bound_ms(work, peak):
    """The bound time in ms of one call doing `work` ({"flops", "bytes"[, "math"]}) on a device with these peaks."""
    math_kind = work.get("math", "f16")
    if math_kind not in PEAKS:
        raise ValueError(f"bounds(): math must be one of {sorted(PEAKS)}, not {math_kind!r}")
    flops, nbytes = float(work.get("flops", 0.0)), float(work.get("bytes", 0.0))
    return 1e3 * max(flops / peak[math_kind], nbytes / peak["bytes_per_s"])


def stage_floors(work, peak):
    """The floors in ms of each stage `work` names (see the module docstring): [{"name", "match", "tensor", "sfu",
    "memory"}], a floor None where the stage gives no count for it or the device no rate."""
    math_kind = work.get("math", "f16")
    out = []
    for stage in work.get("stages", []):
        flops, exps, nbytes = (stage.get(k) for k in ("flops", "exps", "bytes"))
        out.append({
            "name": stage["name"], "match": stage["match"],
            "tensor": None if flops is None else 1e3 * float(flops) / peak[math_kind],
            "sfu": None if exps is None or "ex2_per_s" not in peak else 1e3 * float(exps) / peak["ex2_per_s"],
            "memory": None if nbytes is None else 1e3 * float(nbytes) / peak["bytes_per_s"],
        })
    return out


def describe(dev, peak):
    """One line on how the bound is computed here."""
    return (f"bound = max(FLOPs / {peak['f16'] / 1e15:.2f} PFLOP/s ({PEAKS['f16']}, dense), bytes / "
            f"{peak['bytes_per_s'] / 1e12:.1f} TB/s), estimated for {dev['sms']} SMs at {dev['clock_mhz']:.0f} MHz")


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
    bounds, dev, peak = workload_bounds(spec, list(spec.WORKLOADS))
    if peak is None:
        raise SystemExit(f"no peak figures for compute capability {dev['capability']}")
    if not bounds:
        raise SystemExit(f"{a.spec} defines no bounds(workload)")
    print(describe(dev, peak))
    for w, ms in bounds.items():
        work = spec.bounds(w)
        print(f"{w:<12} {ms:9.4f} ms   {work['flops'] / 1e9:10.2f} GFLOP  {work['bytes'] / 1e6:10.1f} MB")
    print(f"geomean {geomean(bounds.values()):.4f} ms")


if __name__ == "__main__":
    main()
