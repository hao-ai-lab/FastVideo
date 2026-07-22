"""Anchor adapters — run each implementation's components on the official
golden probes and produce typed comparison records.

Authority ordering (the project rule): the official Wan-Video/Wan2.1
implementation is the numerics ground truth. Everything else — fastvideo2's
diffusers-backed components, fastvideo main's native components — is a port
whose fidelity is *measured* against the captured goldens, never assumed.
Ports (diffusers included) are known to drift from official implementations
in small ways that surface later as training/serving skew; when a convention
conflicts, the official one wins.

Each adapter exposes the same three callables (text_encode / dit_forward /
vae_decode) over numpy arrays; ``run_anchor`` drives any adapter over the
goldens and returns comparison records. Heavy imports are lazy and guarded so
this module imports anywhere.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

import numpy as np

from fastvideo2.wan21 import goldens as G

# Certification thresholds (relative L2). Two DiT rows make two claims:
# ``dit`` (bf16) certifies the deployment numerics band — measured decomposition
# (probe_dit): ~5.6e-3 attention backend (flash vs SDPA) + precision placement
# + kernel accumulation ≈ 1.6-2.1e-2 total. ``dit_fp32`` certifies math
# fidelity in true fp32 against the official SDPA-path golden: measured
# diffusers delta is 2.6e-3 — a real implementation difference (suspected
# RoPE formulation; official builds rotary phases via fp64/complex-polar).
# The bound sits just above that measurement so *growth* fails the gate;
# layer-wise bisection is the documented follow-up if training-grade
# exactness is ever required. A convention drift (e.g. truncating the
# load-bearing 512 zero-pad context: rel 0.21) fails everything.
TOLS = {"text_encoder": 2e-2, "dit": 5e-2, "dit_fp32": 5e-3, "vae": 5e-2}


def _vae_stats(root: str) -> tuple[np.ndarray, np.ndarray]:
    """latents_mean/std straight from the checkpoint's vae/config.json — one
    source for every port, so the anchor measures the model, not the config."""
    with open(os.path.join(root, "vae", "config.json")) as f:
        cfg = json.load(f)
    mean = np.asarray(cfg["latents_mean"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    std = np.asarray(cfg["latents_std"], dtype=np.float32).reshape(1, -1, 1, 1, 1)
    return mean, std


# --------------------------------------------------------------------------- #
# Adapter: fastvideo2 (diffusers-backed components via our loading path)       #
# --------------------------------------------------------------------------- #
def fastvideo2_adapter(root: str, device: str) -> dict[str, Callable]:
    import torch

    from fastvideo2.engine import Instance
    from fastvideo2.wan21.card import WAN21_T2V_1_3B
    from fastvideo2.wan21.pipeline import _encode_one

    inst = Instance(WAN21_T2V_1_3B, root=root, device=device)
    mean, std = _vae_stats(inst.root)

    def text_encode(text: str) -> np.ndarray:
        emb = _encode_one(inst.component("tokenizer"), inst.component("text_encoder"), text)
        return emb[0].to(torch.float32).cpu().numpy()          # [512, 4096], zero-padded

    def dit_forward(x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray:
        from fastvideo2.wan21.loop import WanForwardInputs
        dit = inst.component("transformer")
        xt = torch.from_numpy(x).to(device, torch.bfloat16)
        ct = torch.from_numpy(context).to(device, torch.bfloat16)[None]
        tt = torch.tensor([t], device=device, dtype=torch.float32)
        with torch.no_grad():
            out = WanForwardInputs(xt, tt, ct).forward(dit)
        return out.to(torch.float32).cpu().numpy()

    def vae_decode(z: np.ndarray) -> np.ndarray:
        vae = inst.component("vae")
        zt = torch.from_numpy(z * std + mean).to(device, torch.float32)
        with torch.no_grad():
            video = vae.decode(zt, return_dict=False)[0]
        return video[0].to(torch.float32).cpu().numpy()        # [3, T, H, W] in [-1, 1]

    def dit_forward_fp32(x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray:
        import copy
        from fastvideo2.wan21.loop import WanForwardInputs
        dit32 = copy.deepcopy(inst.component("transformer")).float()
        out = WanForwardInputs(torch.from_numpy(x).to(device, torch.float32),
                               torch.tensor([t], device=device, dtype=torch.float32),
                               torch.from_numpy(context).to(device, torch.float32)[None])
        with torch.no_grad():
            res = out.forward(dit32).to(torch.float32).cpu().numpy()
        del dit32
        torch.cuda.empty_cache()
        return res

    return {"text_encode": text_encode, "dit_forward": dit_forward, "vae_decode": vae_decode,
            "dit_forward_fp32": dit_forward_fp32}


# --------------------------------------------------------------------------- #
# Adapter: fastvideo main (native components via their loaders)                #
# --------------------------------------------------------------------------- #
def fastvideo_main_adapter(root: str, device: str) -> dict[str, Callable]:
    """Guarded: requires `fastvideo` importable (the main-branch environment).
    Mirrors fastvideo's own component tests: single-GPU distributed init,
    component loaders, forward under set_forward_context."""
    import torch
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29512")
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines import PipelineConfig, WanT2V480PConfig
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.loader.component_loader import (TextEncoderLoader, TransformerLoader,
                                                          VAELoader)
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from transformers import AutoTokenizer

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    mean, std = _vae_stats(root)
    cache: dict[str, Any] = {}

    def _dit():
        if "dit" not in cache:
            path = os.path.join(root, "transformer")
            args = FastVideoArgs(model_path=path,
                                 pipeline_config=PipelineConfig(dit_config=WanVideoConfig(),
                                                                dit_precision="bf16"))
            args.device = torch.device(device)
            cache["dit"] = TransformerLoader().load(path, args).to(dtype=torch.bfloat16).eval()
        return cache["dit"]

    def _t5():
        if "t5" not in cache:
            path = os.path.join(root, "text_encoder")
            args = FastVideoArgs(model_path=path, pipeline_config=WanT2V480PConfig(),
                                 pin_cpu_memory=False)
            cache["t5"] = (TextEncoderLoader().load(path, args).to(torch.bfloat16).eval(),
                           AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer")))
        return cache["t5"]

    def _vae():
        if "vae" not in cache:
            path = os.path.join(root, "vae")
            args = FastVideoArgs(model_path=path, pipeline_config=WanT2V480PConfig())
            args.device = torch.device(device)
            # their loader can leave the module on CPU regardless of args.device
            cache["vae"] = VAELoader().load(path, args).to(torch.device(device),
                                                           torch.float32).eval()
        return cache["vae"]

    def text_encode(text: str) -> np.ndarray:
        model, tok = _t5()
        batch = tok([text], padding="max_length", max_length=512, truncation=True,
                    add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids = batch.input_ids.to(device)
        mask = batch.attention_mask.to(device)
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            emb = model(input_ids=ids, attention_mask=mask).last_hidden_state
        emb = emb.to(torch.float32)
        emb[:, int(mask[0].sum()):] = 0
        return emb[0].cpu().numpy()

    def dit_forward(x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray:
        dit = _dit()
        xt = torch.from_numpy(x).to(device, torch.bfloat16)
        ct = torch.from_numpy(context).to(device, torch.bfloat16)[None]
        tt = torch.tensor([t], device=device, dtype=torch.float32)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16), \
                set_forward_context(current_timestep=0, attn_metadata=None,
                                    forward_batch=ForwardBatch(data_type="dummy")):
            out = dit(hidden_states=xt, encoder_hidden_states=ct, timestep=tt)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.to(torch.float32).cpu().numpy()

    def vae_decode(z: np.ndarray) -> np.ndarray:
        vae = _vae()
        zt = torch.from_numpy(z * std + mean).to(device, torch.float32)
        with torch.no_grad():
            out = vae.decode(zt)
        out = getattr(out, "sample", out)
        out = out[0] if isinstance(out, (tuple, list)) else out
        if out.ndim == 5:
            out = out[0]
        return out.to(torch.float32).cpu().numpy()

    def dit_forward_fp32(x: np.ndarray, t: float, context: np.ndarray) -> np.ndarray:
        import copy
        dit32 = copy.deepcopy(_dit()).float()
        xt = torch.from_numpy(x).to(device, torch.float32)
        ct = torch.from_numpy(context).to(device, torch.float32)[None]
        tt = torch.tensor([t], device=device, dtype=torch.float32)
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None,
                                                  forward_batch=ForwardBatch(data_type="dummy")):
            out = dit32(hidden_states=xt, encoder_hidden_states=ct, timestep=tt)
        out = out[0] if isinstance(out, (tuple, list)) else out
        res = out.to(torch.float32).cpu().numpy()
        del dit32
        torch.cuda.empty_cache()
        return res

    return {"text_encode": text_encode, "dit_forward": dit_forward, "vae_decode": vae_decode,
            "dit_forward_fp32": dit_forward_fp32}


# --------------------------------------------------------------------------- #
# The generic anchor run                                                       #
# --------------------------------------------------------------------------- #
def run_anchor(adapter: dict[str, Callable], goldens_dir: str) -> list[dict]:
    """Drive one implementation over the goldens; return comparison records.
    Each probe is isolated: a crash becomes a fail record, not a lost run."""
    records: list[dict] = []

    def probe(name: str, fn: Callable[[], dict]) -> None:
        try:
            records.append(fn())
        except Exception as e:
            records.append({"name": name, "status": "fail",
                            "detail": f"{type(e).__name__}: {e}"})

    t5 = G.load_golden(goldens_dir, "text_encoder")
    for key, text in (("prompt", G.PROMPT), ("negative", G.NEGATIVE)):
        def _t5(key=key, text=text) -> dict:
            ours = adapter["text_encode"](text)                 # [512, 4096] zero-padded
            length = int(t5[f"{key}_len"])
            rec = G.compare(f"text_encoder.{key}", ours[:length], t5[f"{key}_embeds"][:length],
                            TOLS["text_encoder"])
            rec["pad_max_abs"] = (float(np.abs(ours[length:]).max())
                                  if length < ours.shape[0] else 0.0)
            return rec
        probe(f"text_encoder.{key}", _t5)

    dit = G.load_golden(goldens_dir, "dit")
    context = dit["context"]                                    # [512, 4096], canonical padded form
    for t in dit["timesteps"].tolist():
        probe(f"dit.t{int(t)}", lambda t=t: G.compare(
            f"dit.t{int(t)}", adapter["dit_forward"](dit["x"], float(t), context),
            dit[f"out_t{int(t)}"], TOLS["dit"]))

    if "dit_forward_fp32" in adapter:
        try:
            f32 = G.load_golden(goldens_dir, "dit_fp32")
        except FileNotFoundError:
            f32 = None
        if f32 is not None:
            probe("dit.fp32", lambda: G.compare(
                "dit.fp32",
                adapter["dit_forward_fp32"](f32["x"], float(f32["timestep"]), f32["context"]),
                f32["out"], TOLS["dit_fp32"]))

    vae = G.load_golden(goldens_dir, "vae")
    probe("vae.decode", lambda: G.compare(
        "vae.decode", adapter["vae_decode"](vae["z"]),
        vae["video"].astype(np.float32), TOLS["vae"]))

    return records


def schedule_records(goldens_dir: str) -> list[dict]:
    """Informational: our Euler sigma grid vs the official UniPC grid for each
    captured (steps, shift). The solver semantics differ by declaration
    (`wan.flow_euler.cfg/v1`), so this records the grid delta, it does not gate."""
    from fastvideo2.wan21.loop import flow_sigmas
    sched = G.load_golden(goldens_dir, "schedule")
    out = []
    for steps, shift in G.SCHEDULES:
        official = sched[f"sigmas_s{steps}_f{shift}"]
        ours = np.asarray(flow_sigmas(steps, shift), dtype=np.float64)
        n = min(len(ours), len(official))
        out.append({"name": f"schedule.s{steps}_f{shift}", "status": "info",
                    "rel_l2": G.rel_l2(ours[:n], official[:n]),
                    "ours_len": len(ours), "official_len": len(official)})
    return out


def report_markdown(all_records: dict[str, list[dict]], manifest: dict) -> str:
    """One table: rows = probes, columns = implementations, cells = rel L2 vs
    the official goldens."""
    impls = list(all_records)
    by_name = {impl: {r["name"]: r for r in recs} for impl, recs in all_records.items()}
    names: list[str] = []
    for recs in all_records.values():                 # union, first-seen order
        for r in recs:
            if r["name"] not in names:
                names.append(r["name"])
    lines = ["# Wan2.1 numerics vs official implementation",
             "",
             f"Goldens: `{manifest.get('capture', {}).get('repo', '?')}` @ "
             f"`{manifest.get('capture', {}).get('commit', '?')}` — see `manifest.json`.",
             "Cells are relative L2 vs the official output (lower is closer; `FAIL` = over tolerance).",
             "",
             "| probe | " + " | ".join(impls) + " |",
             "|---|" + "---|" * len(impls)]
    for name in names:
        cells = []
        for impl in impls:
            r = by_name[impl].get(name)
            if r is None:
                cells.append("—")
                continue
            v = r.get("rel_l2")
            cell = f"{v:.2e}" if v is not None else (r.get("detail", "n/a")[:40] or "n/a")
            if r["status"] == "fail":
                cell = f"**FAIL** {cell}"
            cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
