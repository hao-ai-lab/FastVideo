"""The verifier — tiered gates with a typed evidence ledger.

Verification is a product surface, not a test suite: every run appends typed
``GateResult`` records (card digest + environment fingerprint + metrics) to
``fastvideo2/evidence/ledger.jsonl``, so "which artifact passed what, where"
is queryable data.

Tiers (each subsumes nothing — they answer different questions):

  T0  contract     CPU, no weights. Cards validate, digests round-trip,
                   pipelines have sound edges. Seconds.
  T1  components   Load every component; fingerprint parameter identity
                   against the blessed baseline (``--bless`` to seed it).
                   Catches load/mapping/dtype drift. ~1 GPU-minute.
  T2  trajectory   Run the production pipeline and ``reference.py`` from the
                   same seed; per-step latent parity. The tolerance is
                   calibrated against the reference's own run-to-run noise, so
                   this gate also *measures* the card's determinism class.
  T3  decode       Decode both trajectories; SSIM between production and
                   reference video, plus anti-degeneracy anchors (a static or
                   black video cannot pass on consistency alone).

Gates fail closed: a tier that cannot run (missing baseline, missing weights)
is a failure, not a skip.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "evidence")
LEDGER = os.path.join(EVIDENCE_DIR, "ledger.jsonl")

# The pinned T2/T3 probe: small enough for minutes-scale gating, large enough
# to exercise the real geometry (T'=5 latent frames at 480p).
PROBE = dict(prompt="A red panda eating bamboo in a sunlit forest, cinematic.",
             seed=1234, num_steps=8, num_frames=17, height=480, width=832,
             guidance_scale=5.0, shift=3.0)
T2_TOL_FLOOR = 1e-3   # relative L2 on final latents; raised by measured self-noise


@dataclass
class GateResult:
    gate: str
    status: str            # pass | fail | blessed
    model_id: str
    card_digest: str
    metrics: dict[str, Any] = field(default_factory=dict)
    tolerances: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "blessed")


def env_fingerprint() -> dict:
    """Where did this evidence come from — enough to refuse cross-environment
    baseline comparisons instead of silently trusting them."""
    import platform
    import subprocess
    env: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine()}
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=os.path.dirname(__file__), timeout=5)
        env["commit"] = out.stdout.strip() or "unknown"
    except Exception:
        env["commit"] = "unknown"
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    for mod in ("diffusers", "transformers"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            pass
    from fastvideo2.layers.attention import available_backends, backend_policy
    env["attention"] = {"policy": backend_policy(), "available": available_backends()}
    return env


def append_ledger(results: list[GateResult], path: str = LEDGER) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in results:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                **asdict(r)}) + "\n")
    return path


# --------------------------------------------------------------------------- #
# T0 — contracts (CPU, no weights, no torch)                                   #
# --------------------------------------------------------------------------- #
def gate_t0_contract(card: Any, pipeline: Any) -> GateResult:
    from fastvideo2.card import CardError, ModelCard, derive
    metrics: dict[str, Any] = {}
    try:
        card.validate()
        rt = ModelCard.from_dict(json.loads(card.to_json()))
        if rt.digest() != card.digest():
            raise CardError("JSON round-trip changed the card digest")
        metrics["digest"] = card.digest()
        variant = derive(card, model_id=card.model_id + "-t0probe")
        if variant.digest() == card.digest():
            raise CardError("derive() produced an identical digest for a changed card")
        try:  # the teeth: mismatched loop semantics must be rejected
            derive(card, provenance={"assumes_loop": "bogus.semantics/v0"})
            raise CardError("assumes_loop mismatch was NOT rejected")
        except CardError as e:
            if "NOT rejected" in str(e):
                raise
        pipeline.validate()
        metrics["stages"] = len(pipeline.stages)
        return GateResult("T0.contract", "pass", card.model_id, card.digest(), metrics=metrics)
    except Exception as e:
        return GateResult("T0.contract", "fail", card.model_id, card.digest(),
                          metrics=metrics, detail=f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# T1 — component fingerprints                                                  #
# --------------------------------------------------------------------------- #
def _baseline_path(model_id: str) -> str:
    return os.path.join(EVIDENCE_DIR, f"{model_id}.fingerprints.json")


def gate_t1_components(instance: Any, *, bless: bool = False) -> GateResult:
    from fastvideo2.loading import component_fingerprint
    card = instance.card
    prints = {cid: component_fingerprint(instance.component(cid), spec)
              for cid, spec in card.components.items()}
    path = _baseline_path(card.model_id)
    if bless:
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"card_digest": card.digest(), "env": env_fingerprint(),
                       "fingerprints": prints}, f, indent=2, sort_keys=True)
        return GateResult("T1.components", "blessed", card.model_id, card.digest(),
                          metrics={"components": len(prints)}, detail=f"baseline -> {path}")
    if not os.path.exists(path):
        return GateResult("T1.components", "fail", card.model_id, card.digest(),
                          detail=f"no blessed baseline at {path} (run with --bless once)")
    with open(path) as f:
        base = json.load(f)
    mismatched = [cid for cid in prints if prints[cid] != base["fingerprints"].get(cid)]
    metrics = {"components": len(prints), "mismatched": mismatched,
               "baseline_card_digest": base.get("card_digest")}
    if base.get("card_digest") != card.digest():
        return GateResult("T1.components", "fail", card.model_id, card.digest(), metrics=metrics,
                          detail="baseline was blessed for a different card digest")
    status = "pass" if not mismatched else "fail"
    return GateResult("T1.components", status, card.model_id, card.digest(), metrics=metrics,
                      detail="" if status == "pass" else f"fingerprint drift in {mismatched}")


# --------------------------------------------------------------------------- #
# T2 — trajectory parity against the reference oracle                          #
# --------------------------------------------------------------------------- #
def _rel_l2(a: Any, b: Any) -> float:
    import torch
    a32, b32 = a.to(torch.float32), b.to(torch.float32)
    denom = torch.linalg.vector_norm(b32).item() or 1.0
    return torch.linalg.vector_norm(a32 - b32).item() / denom


def gate_t2_trajectory(instance: Any, pipeline: Any) -> tuple[GateResult, Any, Any]:
    """Returns the gate result plus (production_output, reference_result) so
    T3 can reuse the same runs instead of paying for new ones."""
    from fastvideo2.engine import Request, run
    from fastvideo2.wan21 import reference
    card = instance.card

    # Share the already-resident modules with the reference: T2 isolates the
    # *execution* path (loop/pipeline/runtime); T1 owns load-path identity.
    models = tuple(instance.component(c) for c in ("tokenizer", "text_encoder", "transformer", "vae"))
    kw = dict(PROBE)
    ref1 = reference.generate(kw.pop("prompt"), models=models, device=instance.device,
                              capture_trajectory=True, **kw)
    kw = dict(PROBE)
    ref2 = reference.generate(kw.pop("prompt"), models=models, device=instance.device,
                              capture_trajectory=True, **kw)
    self_noise = max((_rel_l2(a, b) for a, b in zip(ref1.trajectory, ref2.trajectory)), default=0.0)

    req = Request(request_id="t2", capture_trajectory=True, **{
        "prompt": PROBE["prompt"], "seed": PROBE["seed"], "num_steps": PROBE["num_steps"],
        "num_frames": PROBE["num_frames"], "height": PROBE["height"], "width": PROBE["width"],
        "guidance_scale": PROBE["guidance_scale"], "shift": PROBE["shift"]})
    out = run(instance, pipeline, req)
    prod_traj = out.outputs["latents"]["trajectory"]

    tol = max(T2_TOL_FLOOR, 10.0 * self_noise)
    per_step = [_rel_l2(p, r) for p, r in zip(prod_traj, ref1.trajectory)]
    final = per_step[-1] if per_step else float("inf")
    metrics = {"probe": PROBE, "self_noise": self_noise, "per_step_rel_l2": per_step,
               "final_rel_l2": final, "steps": len(per_step),
               "step_seconds": [t["seconds"] for t in out.trace if "/denoise." in t["label"]]}
    status = "pass"
    detail = ""
    if len(prod_traj) != len(ref1.trajectory):
        status, detail = "fail", (f"trajectory length {len(prod_traj)} != reference "
                                  f"{len(ref1.trajectory)}")
    elif final > tol:
        status, detail = "fail", f"final rel L2 {final:.3e} > tol {tol:.3e}"
    elif card.determinism == "bitwise" and self_noise > 0.0:
        status, detail = "fail", f"card claims bitwise determinism but self-noise is {self_noise:.3e}"
    return (GateResult("T2.trajectory", status, card.model_id, card.digest(), metrics=metrics,
                       tolerances={"final_rel_l2": tol, "floor": T2_TOL_FLOOR}, detail=detail),
            out, ref1)


# --------------------------------------------------------------------------- #
# T3 — decoded output parity + anti-degeneracy anchors                         #
# --------------------------------------------------------------------------- #
def _ssim(a: Any, b: Any) -> float:
    """Mean SSIM over frames, grayscale, uniform 7x7 window. Plain numpy —
    good enough for a parity gate between two decodes of near-identical
    latents; not a perceptual quality metric."""
    import numpy as np
    x = np.asarray(a, dtype=np.float64).mean(axis=-1)  # [T,H,W] grayscale
    y = np.asarray(b, dtype=np.float64).mean(axis=-1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = 7
    vals = []
    for xf, yf in zip(x, y):
        # box-filter via cumulative sums
        def box(img):
            cs = np.cumsum(np.cumsum(img, axis=0), axis=1)
            cs = np.pad(cs, ((1, 0), (1, 0)))
            return (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)
        mx, my = box(xf), box(yf)
        mxx, myy, mxy = box(xf * xf), box(yf * yf), box(xf * yf)
        vx, vy, cov = mxx - mx * mx, myy - my * my, mxy - mx * my
        s = ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx**2 + my**2 + c1) * (vx + vy + c2))
        vals.append(float(s.mean()))
    return float(sum(vals) / max(len(vals), 1))


def gate_t3_decode(card: Any, prod_out: Any, ref_res: Any) -> GateResult:
    import numpy as np
    prod, ref = prod_out.outputs["video"], ref_res.video
    metrics: dict[str, Any] = {"shape": list(prod.shape)}
    status, detail = "pass", ""
    if prod.shape != ref.shape:
        return GateResult("T3.decode", "fail", card.model_id, card.digest(), metrics=metrics,
                          detail=f"video shape {prod.shape} != reference {ref.shape}")
    ssim = _ssim(prod, ref)
    dyn = float(np.abs(np.diff(ref.astype(np.float32), axis=0)).mean())
    bright = float(ref.astype(np.float32).mean())
    metrics.update(ssim_vs_reference=ssim, temporal_dynamics=dyn, mean_brightness=bright)
    if ssim < 0.98:
        status, detail = "fail", f"SSIM vs reference {ssim:.4f} < 0.98"
    elif dyn < 0.5:  # anchor: a frozen video must not pass on similarity alone
        status, detail = "fail", f"degenerate output: temporal dynamics {dyn:.3f} < 0.5"
    elif not (5.0 < bright < 250.0):
        status, detail = "fail", f"degenerate output: mean brightness {bright:.1f}"
    return GateResult("T3.decode", status, card.model_id, card.digest(), metrics=metrics,
                      tolerances={"ssim": 0.98, "temporal_dynamics": 0.5}, detail=detail)


# --------------------------------------------------------------------------- #
# Anchor — certification against OFFICIAL goldens                              #
# --------------------------------------------------------------------------- #
def gate_anchor(instance: Any) -> list[GateResult]:
    """Compare this implementation's components against goldens captured from
    the official Wan-Video/Wan2.1 repo. The authority ordering is official >
    reference > production: diffusers-backed components are a *port* whose
    fidelity these records certify, never assume. Fails closed if no goldens
    have been captured."""
    from fastvideo2.wan21 import anchor as A
    from fastvideo2.wan21 import goldens as G
    card = instance.card
    gdir = G.golden_dir()
    if not os.path.exists(os.path.join(gdir, "manifest.json")):
        return [GateResult("anchor", "fail", card.model_id, card.digest(),
                           detail=f"no goldens at {gdir} — run capture_official.py first")]
    adapter = A.fastvideo2_adapter(instance.root, instance.device)
    out: list[GateResult] = []
    for rec in A.run_anchor(adapter, gdir) + A.schedule_records(gdir):
        name = rec.pop("name")
        status = rec.pop("status")
        detail = rec.pop("detail", "")
        out.append(GateResult(f"anchor.{name}", "pass" if status == "info" else status,
                              card.model_id, card.digest(), metrics=rec,
                              tolerances={"rel_l2": rec.get("tol_rel")}, detail=detail))
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def verify(model_id: str, *, tier: int = 3, root: str | None = None,
           device: str | None = None, bless: bool = False,
           anchor: bool = False) -> list[GateResult]:
    from fastvideo2.registry import resolve
    card, build_pipeline = resolve(model_id)
    pipeline = build_pipeline()
    env = env_fingerprint()
    results = [gate_t0_contract(card, pipeline)]
    if tier >= 1 and results[-1].ok:
        from fastvideo2.engine import load
        instance = load(card, root=root, device=device)
        results.append(gate_t1_components(instance, bless=bless))
        if anchor and results[-1].ok:
            results.extend(gate_anchor(instance))
        if tier >= 2 and all(r.ok for r in results):
            t2, prod_out, ref_res = gate_t2_trajectory(instance, pipeline)
            results.append(t2)
            if tier >= 3 and t2.ok:
                results.append(gate_t3_decode(card, prod_out, ref_res))
    for r in results:
        r.env = env
    append_ledger(results)
    return results
