# SPDX-License-Identifier: Apache-2.0
"""Per-block divergence debugger for MagiHumanDiT vs upstream DiTModel.

Not a pytest test (filename starts with `_`). Run directly:

    python tests/local_tests/transformers/_debug_magi_human_block_parity.py

Mirrors the inputs / loader of `test_magi_human_dit_parity` but adds
forward hooks on:

  * `model.adapter` (post-embedding)
  * each `model.block.layers[i]` (per-block output, 40 blocks)
  * model output (post-final-norms)

Logs (idx, label, abs_mean, sum) for both sides side-by-side, and
prints the first block where |abs_mean diff| or |sum diff| exceeds a
threshold so we know where to drill in.
"""
from __future__ import annotations

import gc
import glob
import os
import sys
from pathlib import Path

import torch

# Match the parity test: FA on both sides.
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _find_base_shard_dir() -> Path | None:
    override = os.getenv("MAGI_HUMAN_BASE_SHARD_DIR")
    if override:
        p = Path(override)
        return p if p.is_dir() else None
    try:
        from huggingface_hub import hf_hub_download
        idx = hf_hub_download(
            repo_id="GAIR/daVinci-MagiHuman",
            filename="base/model.safetensors.index.json",
        )
        return Path(idx).parent
    except Exception:
        return None


def _stat(name: str, t: torch.Tensor) -> dict:
    f = t.detach().float()
    return {
        "name": name,
        "shape": tuple(t.shape),
        "abs_mean": f.abs().mean().item(),
        "sum": f.sum().item(),
        "min": f.min().item(),
        "max": f.max().item(),
    }


def _attach_block_hooks(model, label: str, log: list[dict],
                        tensors: dict[str, torch.Tensor] | None = None,
                        drill_layer: int | None = None):
    """Attach forward hooks to adapter + each block.layers[i].

    If ``drill_layer`` is set, also hooks the submodules of
    ``block.layers[drill_layer]`` (attention, mlp, attn_post_norm,
    mlp_post_norm if present), letting us pinpoint which submodule
    introduces the first measurable drift.
    """
    handles = []

    def _hook(name):
        def fn(_module, _inputs, outputs):
            t = outputs[0] if isinstance(outputs, tuple) else outputs
            if not torch.is_tensor(t):
                return
            log.append({"side": label, **_stat(name, t)})
            if tensors is not None:
                tensors[name] = t.detach().float().cpu()
        return fn

    def _pre_hook(name):
        def fn(_module, inputs):
            t = inputs[0] if isinstance(inputs, tuple) else inputs
            if not torch.is_tensor(t):
                return
            label_in = f"{name}<in>"
            log.append({"side": label, **_stat(label_in, t)})
            if tensors is not None:
                tensors[label_in] = t.detach().float().cpu()
        return fn

    handles.append(model.adapter.register_forward_hook(_hook("adapter")))
    for i, layer in enumerate(model.block.layers):
        handles.append(layer.register_forward_hook(_hook(f"block[{i:02d}]")))
        if drill_layer is not None and i == drill_layer:
            tag = f"L{i:02d}"
            handles.append(layer.attention.register_forward_hook(
                _hook(f"{tag}.attention")))
            handles.append(layer.mlp.pre_norm.register_forward_hook(
                _hook(f"{tag}.mlp.pre_norm")))
            handles.append(layer.mlp.up_gate_proj.register_forward_hook(
                _hook(f"{tag}.mlp.up_gate_proj")))
            # Pre-hook on down_proj captures the post-activation tensor
            # (the activation func is a free function, not a module, so
            # we observe its output by intercepting down_proj's input).
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                _pre_hook(f"{tag}.mlp.down_proj")))
            handles.append(layer.mlp.down_proj.register_forward_hook(
                _hook(f"{tag}.mlp.down_proj")))
            handles.append(layer.mlp.register_forward_hook(
                _hook(f"{tag}.mlp")))
            if hasattr(layer, "attn_post_norm"):
                handles.append(layer.attn_post_norm.register_forward_hook(
                    _hook(f"{tag}.attn_post_norm")))
            if hasattr(layer, "mlp_post_norm"):
                handles.append(layer.mlp_post_norm.register_forward_hook(
                    _hook(f"{tag}.mlp_post_norm")))
    return handles


def main() -> None:
    if not torch.cuda.is_available():
        print("Need CUDA. Skipping.")
        return

    upstream_src = REPO_ROOT / "daVinci-MagiHuman"
    if not upstream_src.exists():
        print(f"daVinci-MagiHuman/ not present under {REPO_ROOT}.")
        return

    base_shard_dir = _find_base_shard_dir()
    if base_shard_dir is None or not base_shard_dir.is_dir():
        print("Upstream base/ shards missing.")
        return

    converted_dir = Path(os.getenv(
        "MAGI_HUMAN_DIFFUSERS_PATH",
        REPO_ROOT / "converted_weights" / "magi_human_base",
    ))
    transformer_dir = converted_dir / "transformer"
    if not transformer_dir.is_dir():
        print(f"Converted transformer dir missing at {transformer_dir}")
        return

    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs, load_upstream_dit,
    )
    install_stubs()

    # Optional: monkey-patch PackedExpertLinear.forward to mirror upstream's
    # explicit-cast torch.matmul pattern (`_BF16ComputeLinear.apply`).
    # Toggled via env var so the experiment is reproducible.
    if os.getenv("MAGI_DEBUG_PATCH_LINEAR") == "1":
        from fastvideo.models.dits import magi_human as _mh

        def _patched_forward(self, x, modality_dispatcher=None):
            def _bf16_linear(inp, w, b):
                inp_c = inp.to(torch.bfloat16)
                w_c = w.to(torch.bfloat16)
                out = torch.matmul(inp_c, w_c.t())
                if b is not None:
                    out = out + b.to(torch.bfloat16)
                return out.to(inp.dtype)

            if self.num_experts == 1:
                return _bf16_linear(x, self.weight, self.bias)
            assert modality_dispatcher is not None
            parts = modality_dispatcher.dispatch(x)
            w_chunks = self.weight.chunk(self.num_experts, dim=0)
            b_chunks = (
                self.bias.chunk(self.num_experts, dim=0)
                if self.bias is not None else [None] * self.num_experts
            )
            for i in range(self.num_experts):
                parts[i] = _bf16_linear(parts[i], w_chunks[i], b_chunks[i])
            return modality_dispatcher.undispatch(*parts)

        _mh.PackedExpertLinear.forward = _patched_forward
        print("[debug] Patched PackedExpertLinear.forward to mirror "
              "upstream's _BF16ComputeLinear pattern.")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    z_dim = 48
    pT, pH, pW = 1, 2, 2
    lat_T, lat_H, lat_W = 2, 6, 6
    video_latent = torch.randn((1, z_dim, lat_T, lat_H, lat_W), dtype=torch.float32, device=device)
    num_video = (lat_T // pT) * (lat_H // pH) * (lat_W // pW)
    num_audio = 4
    num_text = 8
    audio_latent = torch.randn((1, num_audio, 64), dtype=torch.float32, device=device)
    text_feat = torch.randn((1, num_text, 3584), dtype=torch.float32, device=device)

    from fastvideo.pipelines.basic.magi_human.stages.latent_preparation import (
        build_packed_inputs,
    )
    x, coords, mm = build_packed_inputs(
        video_latent=video_latent,
        audio_latent=audio_latent,
        audio_feat_len=num_audio,
        txt_feat=text_feat,
        txt_feat_len=num_text,
        patch_size=(pT, pH, pW),
        coords_style="v2",
    )
    total_tokens = x.shape[0]

    # --- Upstream ---
    print("Loading upstream DiTModel...")
    upstream = load_upstream_dit(base_shard_dir, device=device, dtype=None)
    from inference.common import VarlenHandler
    cu = torch.tensor([0, total_tokens], dtype=torch.int32, device=device)
    varlen = VarlenHandler(
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=total_tokens, max_seqlen_k=total_tokens,
    )
    drill_layer = int(os.getenv("MAGI_DEBUG_DRILL_LAYER", "0"))
    up_log: list[dict] = []
    up_tensors: dict[str, torch.Tensor] = {}
    _attach_block_hooks(upstream, "up", up_log, tensors=up_tensors, drill_layer=drill_layer)
    print("Running upstream forward (with hooks)...")
    with torch.inference_mode():
        ref_out = upstream(
            x=x.clone(), coords_mapping=coords.clone(),
            modality_mapping=mm.clone(),
            varlen_handler=varlen, local_attn_handler=None,
        ).detach().float().cpu()
    del upstream
    gc.collect(); torch.cuda.empty_cache()

    # --- FastVideo ---
    from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
    from fastvideo.models.dits.magi_human import MagiHumanDiT
    from safetensors.torch import load_file
    print("Loading FastVideo MagiHumanDiT...")
    fv = MagiHumanDiT(MagiHumanVideoConfig())
    state = {}
    for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))):
        state.update(load_file(shard))
    fv.load_state_dict(state, strict=False)
    fv = fv.to(device).eval()
    fv_log: list[dict] = []
    fv_tensors: dict[str, torch.Tensor] = {}
    _attach_block_hooks(fv, "fv", fv_log, tensors=fv_tensors, drill_layer=drill_layer)
    print("Running FastVideo forward (with hooks)...")
    with torch.inference_mode():
        fv_out = fv(x.clone(), coords.clone(), mm.clone()).detach().float().cpu()

    # --- Side-by-side per-block comparison ---
    # Group by name; each name should appear once on each side.
    by_name: dict[str, dict] = {}
    for entry in up_log + fv_log:
        d = by_name.setdefault(entry["name"], {})
        d[entry["side"]] = entry

    print()
    print(f"{'name':<14} {'up_shape':<22} {'up_absmean':>12} {'fv_absmean':>12} "
          f"{'absmean_diff':>14} {'rel%':>8} {'up_sum':>14} {'fv_sum':>14} {'sum_diff':>12}")
    print("-" * 145)

    first_div_idx = None
    rel_threshold = 0.005  # 0.5% drift in abs_mean per block

    # Print in canonical order: adapter, drilled-layer submodules
    # (intermixed with their parent block), then remaining blocks.
    def _sort_key(n: str):
        if n == "adapter":
            return (0, "")
        if n.startswith(f"L{drill_layer:02d}."):
            # Submodule snapshots — sort to appear right before
            # block[NN] so they read as "what fed into block[NN]'s
            # output". Order: attention, attn_post_norm, mlp, mlp_post_norm.
            sub_order = {
                "attention": 0,
                "attn_post_norm": 1,
                "mlp.pre_norm": 2,
                "mlp.up_gate_proj": 3,
                "mlp.down_proj": 4,
                "mlp": 5,
                "mlp_post_norm": 6,
            }.get(n.split(".", 1)[1], 9)
            return (1, f"block[{drill_layer:02d}]", sub_order)
        if n.startswith("block["):
            return (1, n, 99)
        return (2, n, 0)

    Path("/tmp/opencode").mkdir(parents=True, exist_ok=True)
    up_log_path = Path("/tmp/opencode/magi_dit_up_layers.log")
    fv_log_path = Path("/tmp/opencode/magi_dit_fv_layers.log")

    def _log_lines(entries: list[dict]) -> list[str]:
        lines = []
        for entry in sorted(entries, key=lambda e: _sort_key(e["name"])):
            lines.append(
                f"{entry['name']}\t{entry['shape']}\t{entry['abs_mean']:.6f}\t"
                f"{entry['sum']:.6f}\t{entry['min']:.6f}\t{entry['max']:.6f}"
            )
        return lines

    up_log_path.write_text("\n".join(_log_lines(up_log)) + "\n")
    fv_log_path.write_text("\n".join(_log_lines(fv_log)) + "\n")

    ordered_names = sorted(by_name.keys(), key=_sort_key)
    for name in ordered_names:
        d = by_name[name]
        up = d.get("up")
        fv = d.get("fv")
        if up is None or fv is None:
            continue
        am_diff = abs(up["abs_mean"] - fv["abs_mean"])
        am_rel = am_diff / max(up["abs_mean"], 1e-9)
        sum_diff = abs(up["sum"] - fv["sum"])
        flag = ""
        if name.startswith("block[") and am_rel > rel_threshold:
            flag = " <<< DIVERGE"
            if first_div_idx is None:
                first_div_idx = int(name[len("block["):-1])
        print(f"{name:<14} {str(up['shape']):<22} {up['abs_mean']:>12.6f} {fv['abs_mean']:>12.6f} "
              f"{am_diff:>14.6f} {am_rel*100:>7.3f}% {up['sum']:>14.4f} {fv['sum']:>14.4f} {sum_diff:>12.4f}{flag}")

    print()
    if first_div_idx is not None:
        print(f"First block exceeding {rel_threshold*100:.2f}% abs_mean rel drift: block[{first_div_idx:02d}]")
    else:
        print(f"No block exceeded {rel_threshold*100:.2f}% — divergence is amortized across blocks.")

    print("[debug] Per-side logs: /tmp/opencode/magi_dit_up_layers.log + /tmp/opencode/magi_dit_fv_layers.log  (diff with: diff /tmp/opencode/magi_dit_up_layers.log /tmp/opencode/magi_dit_fv_layers.log)")

    # Final output diff
    diff = (ref_out - fv_out).abs()
    print()
    print(f"Final  ref_abs={ref_out.abs().mean():.6f}  fv_abs={fv_out.abs().mean():.6f}  "
          f"diff_max={diff.max():.6f}  diff_mean={diff.mean():.6f}")

    # Element-wise diff stats for drilled submodules.
    print()
    print(f"Element-wise diffs for drilled L{drill_layer:02d} submodules:")
    print(f"{'name':<30} {'shape':<22} {'diff_max':>12} {'diff_mean':>12} {'diff_rel%':>10}")
    print("-" * 95)
    common_names = set(up_tensors.keys()) & set(fv_tensors.keys())
    for name in sorted(common_names):
        a, b = up_tensors[name], fv_tensors[name]
        if a.shape != b.shape:
            continue
        d = (a - b).abs()
        ref_abs = a.abs().mean().item()
        rel = (d.mean().item() / max(ref_abs, 1e-9)) * 100
        print(f"{name:<30} {str(tuple(a.shape)):<22} {d.max().item():>12.6f} {d.mean().item():>12.6f} {rel:>9.4f}%")


if __name__ == "__main__":
    main()
