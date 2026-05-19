#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FastVideo↔AnyFlow numerical parity verification.

Two checks, both on a single H200:

  (A) Forward parity — load nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers into
      FastVideo's WanTransformer3DModel (with r_embedder enabled +
      param_names_mapping handling the delta_embedder rename), forward
      it on identical inputs against AnyFlow's reference loader, and
      compare. Expectation: rel-mean diff < 10% in bf16 (bf16 kernel noise).

  (B) Any-step end-to-end sampling — run the new
      FlowMapEulerDiscreteScheduler through 4 Euler-flow steps on the
      same loaded weights, confirm the final latent is finite and
      well-scaled.

Run via::

    srun --jobid=304 bash -c '
      source ~/.venvs/anyflow/bin/activate
      export CPATH=/home/guian/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10
      cd ~/projects/anyflow/FastVideo
      PYTHONPATH=$PWD python scripts/verify_anyflow_fastvideo_parity.py
    '
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


ANYFLOW_LOCAL = Path("/home/guian/projects/anyflow/anyflow-1.3b")
ANYFLOW_REF = Path("/home/guian/projects/anyflow/anyflow-ref")
sys.path.insert(0, str(ANYFLOW_REF))

SEED = 1234
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Distributed bootstrap (single-rank). Required by WanTransformer3DModel
# which calls get_sp_world_size() and uses ReplicatedLinear.
# ---------------------------------------------------------------------------


def init_single_rank() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 backend="nccl")
    initialize_model_parallel(
        tensor_model_parallel_size=1,
        sequence_model_parallel_size=1,
        data_parallel_size=1,
    )
    print("  single-rank distributed environment + TP/SP/DP groups initialized")


# ---------------------------------------------------------------------------
# Translate AnyFlow HF safetensor keys onto FastVideo's WanTransformer3DModel
# internal layout, applying the regex from WanVideoArchConfig.param_names_mapping.
# ---------------------------------------------------------------------------


def translate_keys(
    raw: dict[str, torch.Tensor],
    *,
    mapping: dict[str, str],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        new_k = k
        for pat, repl in mapping.items():
            new_k = re.sub(pat, repl, new_k)
        out[new_k] = v
    return out


# ---------------------------------------------------------------------------
# Build FastVideo WanTransformer3DModel with AnyFlow weights loaded.
# ---------------------------------------------------------------------------


def build_fastvideo_transformer():
    banner("(1) Build FastVideo WanTransformer3DModel + load AnyFlow weights")
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    cfg = WanVideoConfig()
    arch = cfg.arch_config
    # AnyFlow Wan2.1-T2V-1.3B arch dims.
    arch.num_attention_heads = 12
    arch.attention_head_dim = 128
    arch.num_layers = 30
    arch.ffn_dim = 8960
    # AnyFlow dual-timestep.
    arch.r_embedder = True
    arch.r_embedder_fusion = "gated"
    arch.r_embedder_gate_value = 0.25
    arch.r_embedder_deltatime_type = "r"
    arch.__post_init__()

    # WanTransformer3DModel takes (config, hf_config). hf_config is
    # diffusers-style — we provide a minimal dict; only fields the model
    # actually reads matter.
    hf_config: dict = {}
    t0 = time.time()
    model = WanTransformer3DModel(config=cfg, hf_config=hf_config)
    model = model.to(DEVICE, dtype=DTYPE).eval()
    print(f"  Wan transformer built in {time.time() - t0:.1f}s; "
          f"params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # Load AnyFlow checkpoint.
    af_path = ANYFLOW_LOCAL / "transformer" / "diffusion_pytorch_model.safetensors"
    af_raw = load_file(str(af_path), device="cpu")
    print(f"  AnyFlow checkpoint: {len(af_raw)} tensors")

    translated = translate_keys(af_raw, mapping=arch.param_names_mapping)
    info = model.load_state_dict(translated, strict=False)
    miss, unex = info.missing_keys, info.unexpected_keys
    print(f"  missing_keys    : {len(miss)} (first 5: {miss[:5]})")
    print(f"  unexpected_keys : {len(unex)} (first 5: {unex[:5]})")
    return model, len(miss), len(unex)


# ---------------------------------------------------------------------------
# Build AnyFlow reference net.
# ---------------------------------------------------------------------------


def build_anyflow_reference():
    banner("(2) Build AnyFlow reference loader")
    from far.models import build_model

    af_net = build_model("FAR_Wan_Transformer3DModel").from_pretrained(
        str(ANYFLOW_LOCAL),
        subfolder="transformer",
        chunk_partition=None,
        full_chunk_limit=0,
        compressed_patch_size=[1, 4, 4],
    ).to(DEVICE, dtype=DTYPE).eval()
    print(f"  AnyFlow {type(af_net).__name__} ready")
    return af_net


# ---------------------------------------------------------------------------
# Forward parity test.
# ---------------------------------------------------------------------------


def forward_compare(fv_model, af_net):
    banner("(3) Forward output comparison")
    B, C, F, H, W = 1, 16, 21, 60, 104
    SEQ, DIM = 32, 4096
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    x = torch.randn(B, C, F, H, W, device=DEVICE, dtype=DTYPE, generator=g)
    enc = torch.randn(B, SEQ, DIM, device=DEVICE, dtype=DTYPE, generator=g)

    # AnyFlow expects per-frame (t, r) [B, F]; FastVideo T2V expects a
    # single scalar per sample [B] (timestep.dim()==2 is reserved for
    # Wan2.2 ti2v's per-token timestep schedule). For shared-t T2V the two
    # are semantically equivalent.
    t_per_frame = torch.full((B, F), 500.0, device=DEVICE, dtype=DTYPE)
    r_per_frame = torch.full((B, F), 200.0, device=DEVICE, dtype=DTYPE)
    t_per_sample = torch.full((B,), 500.0, device=DEVICE, dtype=DTYPE)
    r_per_sample = torch.full((B,), 200.0, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        # AnyFlow native: takes [B, F, C, H, W] + [B, F] (t, r).
        x_af = x.permute(0, 2, 1, 3, 4).contiguous()
        af_out = af_net(
            x_af,
            timestep=t_per_frame,
            r_timestep=r_per_frame,
            encoder_hidden_states=enc,
            return_dict=False,
            is_causal=False,
        )[0]
        if af_out.shape[1] != C and af_out.shape[2] == C:
            af_out = af_out.permute(0, 2, 1, 3, 4).contiguous()

        # FastVideo: [B, C, F, H, W] + [B] (t, r). Must wrap in
        # set_forward_context so the attention layer can pick up the
        # current timestep / attn_metadata.
        from fastvideo.forward_context import set_forward_context
        with set_forward_context(
            current_timestep=t_per_sample,
            attn_metadata=None,
        ):
            fv_out = fv_model(
                hidden_states=x,
                encoder_hidden_states=enc,
                timestep=t_per_sample,
                r_timestep=r_per_sample,
            )

    print(f"  AnyFlow out: shape={tuple(af_out.shape)} dtype={af_out.dtype}")
    print(f"  FastVideo  : shape={tuple(fv_out.shape)} dtype={fv_out.dtype}")
    if af_out.shape != fv_out.shape:
        print("  ❌ shape mismatch")
        return False
    diff = (af_out.float() - fv_out.float()).abs()
    ref = af_out.float().abs().mean().item() + 1e-12
    print(f"  max abs diff : {diff.max().item():.3e}")
    print(f"  mean abs diff: {diff.mean().item():.3e}")
    print(f"  rel mean diff: {diff.mean().item() / ref:.3e}")
    ok = diff.mean().item() / ref < 0.10
    print(f"  >>> {'PASS' if ok else 'FAIL'} (target rel diff < 10%, bf16 noise)")
    return ok


# ---------------------------------------------------------------------------
# Any-step sampling smoke via FlowMapEulerDiscreteScheduler.
# ---------------------------------------------------------------------------


def sample_anystep(fv_model):
    banner("(4) Any-step 4-step Euler-flow sampling")
    from fastvideo.models.schedulers.scheduling_flow_map_euler_discrete import (
        FlowMapEulerDiscreteScheduler, )

    scheduler = FlowMapEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)
    scheduler.set_timesteps(num_inference_steps=4, device=DEVICE)
    timesteps = scheduler.timesteps.to(dtype=DTYPE)

    B, C, F, H, W = 1, 16, 21, 60, 104
    SEQ, DIM = 32, 4096
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    x = torch.randn(B, C, F, H, W, device=DEVICE, dtype=DTYPE, generator=g)
    enc = torch.randn(B, SEQ, DIM, device=DEVICE, dtype=DTYPE, generator=g)

    from fastvideo.forward_context import set_forward_context
    t0 = time.time()
    with torch.no_grad():
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            t_in = t_cur.expand(B).to(DTYPE)
            r_in = t_next.expand(B).to(DTYPE)
            with set_forward_context(current_timestep=t_in, attn_metadata=None):
                v = fv_model(
                    hidden_states=x,
                    encoder_hidden_states=enc,
                    timestep=t_in,
                    r_timestep=r_in,
                )
            x = scheduler.step(
                v, sample=x,
                timestep=t_cur.repeat(B),
                r_timestep=t_next.repeat(B),
            )
    elapsed = time.time() - t0
    xf = x.float()
    print(f"  elapsed: {elapsed:.1f}s")
    print(f"  final latent: mean={xf.mean().item():+.3f} std={xf.std().item():.3f} "
          f"range=[{xf.min().item():+.2f}, {xf.max().item():+.2f}] "
          f"finite={torch.isfinite(xf).all().item()}")
    ok = torch.isfinite(xf).all().item() and 0.01 < xf.std().item() < 30
    print(f"  >>> {'PASS' if ok else 'FAIL'}")
    return ok


NUM_TRAIN_TIMESTEPS = 1000
EPSILON = 5.0  # AnyFlow paper default


@torch.no_grad()
def training_step_compare(fv_model, af_net) -> bool:
    """Inline replica of AnyFlow's train_bidirection central-difference loss
    on both code paths with identical synthetic (real, noise, t, r) inputs.

    Compares scalar loss + intermediate flow_pred / target tensors.
    """
    banner("(5) Training-step loss comparison (central-difference target)")
    from fastvideo.forward_context import set_forward_context

    B, C, F, H, W = 1, 16, 21, 60, 104
    SEQ, DIM = 32, 4096
    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    real = torch.randn(B, C, F, H, W, device=DEVICE, dtype=DTYPE, generator=g)
    enc = torch.randn(B, SEQ, DIM, device=DEVICE, dtype=DTYPE, generator=g)
    noise = torch.randn_like(real)
    t_abs_pf = torch.full((B, F), 500.0, device=DEVICE, dtype=DTYPE)
    r_abs_pf = torch.full((B, F), 200.0, device=DEVICE, dtype=DTYPE)
    t_abs_ps = torch.full((B,), 500.0, device=DEVICE, dtype=DTYPE)
    r_abs_ps = torch.full((B,), 200.0, device=DEVICE, dtype=DTYPE)

    # AnyFlow inline replica: uses [B, F, C, H, W] layout.
    real_btchw = real.permute(0, 2, 1, 3, 4).contiguous()
    noise_btchw = noise.permute(0, 2, 1, 3, 4).contiguous()
    t_norm_pf = (t_abs_pf / NUM_TRAIN_TIMESTEPS).view(B, F, 1, 1, 1).to(DTYPE)
    noisy_btchw = t_norm_pf * noise_btchw + (1 - t_norm_pf) * real_btchw

    def u_func_af(x_in, t_in, r_in):
        return af_net(
            x_in, timestep=t_in, r_timestep=r_in,
            encoder_hidden_states=enc, return_dict=False, is_causal=False)[0]

    v_pred = noise_btchw - real_btchw
    eps = EPSILON
    F_plus = u_func_af(noisy_btchw + v_pred * (eps / NUM_TRAIN_TIMESTEPS),
                       t_abs_pf + eps, r_abs_pf)
    F_minus = u_func_af(noisy_btchw - v_pred * (eps / NUM_TRAIN_TIMESTEPS),
                        t_abs_pf - eps, r_abs_pf)
    dF_dt_af = (F_plus - F_minus) / (2 * eps)
    target_af = ((noise_btchw - real_btchw)
                 - (t_abs_pf - r_abs_pf).view(B, F, 1, 1, 1) * dF_dt_af)
    flow_af = u_func_af(noisy_btchw, t_abs_pf, r_abs_pf)
    loss_af = (flow_af.float() - target_af.float()).pow(2).reshape(B, -1).mean(-1)

    # FastVideo inline replica: uses [B, C, F, H, W] layout + [B] t/r.
    real_bcfhw = real
    noise_bcfhw = noise
    t_norm_ps = (t_abs_ps / NUM_TRAIN_TIMESTEPS).view(B, 1, 1, 1, 1).to(DTYPE)
    noisy_bcfhw = t_norm_ps * noise_bcfhw + (1 - t_norm_ps) * real_bcfhw

    def u_func_fv(x_in, t_in, r_in):
        with set_forward_context(current_timestep=t_in, attn_metadata=None):
            return fv_model(
                hidden_states=x_in,
                encoder_hidden_states=enc,
                timestep=t_in,
                r_timestep=r_in,
            )

    v_pred_fv = noise_bcfhw - real_bcfhw
    F_plus_fv = u_func_fv(
        noisy_bcfhw + v_pred_fv * (eps / NUM_TRAIN_TIMESTEPS),
        t_abs_ps + eps, r_abs_ps)
    F_minus_fv = u_func_fv(
        noisy_bcfhw - v_pred_fv * (eps / NUM_TRAIN_TIMESTEPS),
        t_abs_ps - eps, r_abs_ps)
    dF_dt_fv = (F_plus_fv - F_minus_fv) / (2 * eps)
    target_fv = ((noise_bcfhw - real_bcfhw)
                 - (t_abs_ps - r_abs_ps).view(B, 1, 1, 1, 1) * dF_dt_fv)
    flow_fv = u_func_fv(noisy_bcfhw, t_abs_ps, r_abs_ps)
    loss_fv = (flow_fv.float() - target_fv.float()).pow(2).reshape(B, -1).mean(-1)

    # Compare. Align AnyFlow's [B, F, C, H, W] → [B, C, F, H, W].
    flow_af_aligned = flow_af.permute(0, 2, 1, 3, 4)
    target_af_aligned = target_af.permute(0, 2, 1, 3, 4)
    flow_diff = (flow_fv.float() - flow_af_aligned.float()).abs()
    target_diff = (target_fv.float() - target_af_aligned.float()).abs()

    af_loss_v = loss_af.mean().item()
    fv_loss_v = loss_fv.mean().item()
    abs_diff = abs(af_loss_v - fv_loss_v)
    rel_diff = abs_diff / abs(af_loss_v + 1e-12)
    print(f"  AnyFlow loss : {af_loss_v:.6f}")
    print(f"  FastVideo loss: {fv_loss_v:.6f}")
    print(f"  abs diff     : {abs_diff:.3e}")
    print(f"  rel diff     : {rel_diff:.3e}")
    print(f"  flow_pred  : max abs {flow_diff.max().item():.3e}  "
          f"mean {flow_diff.mean().item():.3e}")
    print(f"  target     : max abs {target_diff.max().item():.3e}  "
          f"mean {target_diff.mean().item():.3e}")
    ok = rel_diff < 0.20
    print(f"  >>> {'PASS' if ok else 'FAIL'} (target rel loss diff < 20%)")
    return ok


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    init_single_rank()
    fv_model, n_miss, n_unex = build_fastvideo_transformer()
    af_net = build_anyflow_reference()
    forward_ok = forward_compare(fv_model, af_net)
    sample_ok = sample_anystep(fv_model)
    train_ok = training_step_compare(fv_model, af_net)
    banner(
        f"SUMMARY: missing_keys={n_miss} unexpected_keys={n_unex} "
        f"forward_parity={forward_ok} sample_smoke={sample_ok} "
        f"training_parity={train_ok}"
    )
    sys.exit(0 if (forward_ok and sample_ok and train_ok) else 1)


if __name__ == "__main__":
    main()
