# SPDX-License-Identifier: Apache-2.0
"""Per-block divergence debugger template for FastVideo model ports.

Run directly (not a pytest test):
    python tests/local_tests/transformers/_debug_<family>_<component>_parity.py

Generalizes: tests/local_tests/transformers/_debug_magi_human_block_parity.py

Fill FAMILY, COMPONENT, and the two loader functions. Run once for the initial
drift table, then set <FAMILY>_DEBUG_DRILL_LAYER=NN to drill into submodules.
Add <FAMILY>_DEBUG_PATCH_<HYPOTHESIS>=1 to A/B test a suspect implementation.

CLEANUP: all hooks removed in try/finally; monkey-patches restored in
try/finally; source edits tracked in a named git stash. Zero source residue.
See add-model-08-trace/SKILL.md for the full cleanup gate checklist.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch

FAMILY: str = "<family>"       # e.g. "magi_human", "ltx2", "wan"
COMPONENT: str = "<component>" # e.g. "dit", "vae", "encoder"
DRILL_LAYER_ENV: str = "<FAMILY>_DEBUG_DRILL_LAYER"
HYPOTHESIS_ENV: str = "<FAMILY>_DEBUG_PATCH_<HYPOTHESIS>"
REL_THRESHOLD: float = 0.005  # 0.5% abs_mean drift flags a block as divergent
LOG_DIR: Path = Path("/tmp/opencode")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def load_official(device: torch.device) -> torch.nn.Module:
    """Load the official upstream model. TODO: implement for your family.

    Example (magi-human):
        from tests.local_tests.helpers.magi_human_upstream import install_stubs, load_upstream_dit
        install_stubs()
        return load_upstream_dit(base_shard_dir, device=device, dtype=None)
    """
    raise NotImplementedError(f"Fill load_official() for {FAMILY}/{COMPONENT}.")


def load_fastvideo(device: torch.device) -> torch.nn.Module:
    """Load the FastVideo-native model. TODO: implement for your family.

    Example (magi-human):
        from fastvideo.configs.models.dits.magi_human import MagiHumanVideoConfig
        from fastvideo.models.dits.magi_human import MagiHumanDiT
        from safetensors.torch import load_file; import glob
        fv = MagiHumanDiT(MagiHumanVideoConfig())
        state = {}
        for shard in sorted(glob.glob(str(transformer_dir / "*.safetensors"))): state.update(load_file(shard))
        fv.load_state_dict(state, strict=False); return fv.to(device).eval()
    """
    raise NotImplementedError(f"Fill load_fastvideo() for {FAMILY}/{COMPONENT}.")


def build_inputs(device: torch.device) -> dict[str, Any]:
    """Return deterministic inputs shared by both sides. TODO: replace.

    Both sides must receive the SAME tensors (clone before each forward call).
    Non-identical inputs cause non-zero drift everywhere.
    """
    torch.manual_seed(0)
    return {"x": torch.randn(64, 1024, dtype=torch.bfloat16, device=device)}


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


def _attach_block_hooks(
    model: torch.nn.Module,
    label: str,
    log: list[dict],
    tensors: dict[str, torch.Tensor] | None = None,
    drill_layer: int | None = None,
) -> list[Any]:
    """Return hook handles. Caller MUST remove them in try/finally."""
    handles: list[Any] = []

    def _hook(name: str):
        def fn(_module, _inputs, outputs):
            t = outputs[0] if isinstance(outputs, tuple) else outputs
            if not torch.is_tensor(t):
                return
            log.append({"side": label, **_stat(name, t)})
            if tensors is not None:
                tensors[name] = t.detach().float().cpu()
        return fn

    def _pre_hook(name: str):
        # Pre-hooks observe a free function's output by intercepting the next
        # module's input (useful when the activation is not an nn.Module).
        def fn(_module, inputs):
            t = inputs[0] if isinstance(inputs, tuple) else inputs
            if not torch.is_tensor(t):
                return
            key = f"{name}<in>"
            log.append({"side": label, **_stat(key, t)})
            if tensors is not None:
                tensors[key] = t.detach().float().cpu()
        return fn

    # TODO: adapt attribute paths to your model. Remove adapter block if absent.
    if hasattr(model, "adapter"):
        handles.append(model.adapter.register_forward_hook(_hook("adapter")))

    # TODO: adapt model.block.layers to your block container.
    # Alternatives: model.transformer.layers, model.blocks, model.layers
    block_layers = model.block.layers  # type: ignore[attr-defined]
    for i, layer in enumerate(block_layers):
        handles.append(layer.register_forward_hook(_hook(f"block[{i:02d}]")))
        if drill_layer is not None and i == drill_layer:
            tag = f"L{i:02d}"
            # TODO: adapt submodule names to your layer's attributes.
            # magi-human uses: attention, mlp.pre_norm, mlp.up_gate_proj,
            # mlp.down_proj (pre+post), mlp, attn_post_norm, mlp_post_norm.
            if hasattr(layer, "attention"):
                handles.append(
                    layer.attention.register_forward_hook(_hook(f"{tag}.attention"))
                )
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                if hasattr(mlp, "pre_norm"):
                    handles.append(
                        mlp.pre_norm.register_forward_hook(_hook(f"{tag}.mlp.pre_norm"))
                    )
                if hasattr(mlp, "up_gate_proj"):
                    handles.append(
                        mlp.up_gate_proj.register_forward_hook(
                            _hook(f"{tag}.mlp.up_gate_proj")
                        )
                    )
                if hasattr(mlp, "down_proj"):
                    handles.append(
                        mlp.down_proj.register_forward_pre_hook(
                            _pre_hook(f"{tag}.mlp.down_proj")
                        )
                    )
                    handles.append(
                        mlp.down_proj.register_forward_hook(_hook(f"{tag}.mlp.down_proj"))
                    )
                handles.append(mlp.register_forward_hook(_hook(f"{tag}.mlp")))
            if hasattr(layer, "attn_post_norm"):
                handles.append(
                    layer.attn_post_norm.register_forward_hook(
                        _hook(f"{tag}.attn_post_norm")
                    )
                )
            if hasattr(layer, "mlp_post_norm"):
                handles.append(
                    layer.mlp_post_norm.register_forward_hook(
                        _hook(f"{tag}.mlp_post_norm")
                    )
                )
    return handles


def _apply_hypothesis_patch() -> bool:
    """Apply an optional monkey-patch gated by HYPOTHESIS_ENV. TODO: implement.

    Pattern: save original on the class, patch, restore in _restore_hypothesis_patch().
    """
    if os.getenv(HYPOTHESIS_ENV) != "1":
        return False
    # TODO: import FastVideo class, save original, apply patch.
    print(f"[debug] Hypothesis patch {HYPOTHESIS_ENV}=1 applied.")
    return True


def _restore_hypothesis_patch() -> None:
    if os.getenv(HYPOTHESIS_ENV) != "1":
        return
    # TODO: restore original, e.g.: _mod.TargetClass.method = _mod._ORIGINAL_METHOD


def _write_log(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(
                f"{e['name']} {e['shape']} "
                f"{e['abs_mean']:.8f} {e['sum']:.4f} "
                f"{e['min']:.6f} {e['max']:.6f}\n"
            )


def _sort_key(name: str, drill_layer: int) -> tuple:
    if name == "adapter":
        return (0, "")
    if name.startswith(f"L{drill_layer:02d}."):
        sub_order = {
            "attention": 0, "attn_post_norm": 1, "mlp.pre_norm": 2,
            "mlp.up_gate_proj": 3, "mlp.down_proj<in>": 4,
            "mlp.down_proj": 5, "mlp": 6, "mlp_post_norm": 7,
        }.get(name.split(".", 1)[1], 9)
        return (1, f"block[{drill_layer:02d}]", sub_order)
    if name.startswith("block["):
        return (1, name, 99)
    return (2, name, 0)


def _print_table(by_name: dict[str, dict], drill_layer: int) -> int | None:
    hdr = (
        f"{'name':<18} {'up_shape':<22} {'up_absmean':>12} {'fv_absmean':>12} "
        f"{'absmean_diff':>14} {'rel%':>8} {'up_sum':>14} {'fv_sum':>14} {'sum_diff':>12}"
    )
    print(f"\n{hdr}\n{'-' * len(hdr)}")
    first_div: int | None = None
    for name in sorted(by_name.keys(), key=lambda n: _sort_key(n, drill_layer)):
        d = by_name[name]
        up, fv = d.get("up"), d.get("fv")
        if up is None or fv is None:
            continue
        am_diff = abs(up["abs_mean"] - fv["abs_mean"])
        am_rel = am_diff / max(up["abs_mean"], 1e-9)
        sum_diff = abs(up["sum"] - fv["sum"])
        flag = ""
        if name.startswith("block[") and am_rel > REL_THRESHOLD:
            flag = " <<< DIVERGE"
            if first_div is None:
                first_div = int(name[len("block["):-1])
        print(
            f"{name:<18} {str(up['shape']):<22} {up['abs_mean']:>12.6f} "
            f"{fv['abs_mean']:>12.6f} {am_diff:>14.6f} {am_rel * 100:>7.3f}% "
            f"{up['sum']:>14.4f} {fv['sum']:>14.4f} {sum_diff:>12.4f}{flag}"
        )
    return first_div


def _print_elementwise(up_t: dict[str, torch.Tensor], fv_t: dict[str, torch.Tensor], drill_layer: int) -> None:
    common = set(up_t.keys()) & set(fv_t.keys())
    if not common:
        return
    hdr = f"{'name':<30} {'shape':<22} {'diff_max':>12} {'diff_mean':>12} {'diff_rel%':>10}"
    print(f"\nElement-wise diffs for drilled L{drill_layer:02d} submodules:\n{hdr}\n{'-' * len(hdr)}")
    for name in sorted(common):
        a, b = up_t[name], fv_t[name]
        if a.shape != b.shape:
            continue
        diff = (a - b).abs()
        rel = (diff.mean().item() / max(a.abs().mean().item(), 1e-9)) * 100
        print(
            f"{name:<30} {str(tuple(a.shape)):<22} "
            f"{diff.max().item():>12.6f} {diff.mean().item():>12.6f} {rel:>9.4f}%"
        )


def main() -> None:
    if not torch.cuda.is_available():
        print("Need CUDA. Skipping.")
        return

    # TODO: add precondition checks (official clone present, weights available).

    drill_layer = int(os.getenv(DRILL_LAYER_ENV, "0"))
    device = torch.device("cuda:0")
    patched = _apply_hypothesis_patch()
    try:
        inputs = build_inputs(device)

        print("Loading official model...")
        official = load_official(device)
        up_log: list[dict] = []
        up_t: dict[str, torch.Tensor] = {}
        up_handles = _attach_block_hooks(official, "up", up_log, up_t, drill_layer)
        print("Running official forward (with hooks)...")
        try:
            with torch.inference_mode():
                # TODO: adapt forward call signature to your component.
                ref_out = official(**{k: v.clone() for k, v in inputs.items()})
            if isinstance(ref_out, dict):
                sample = ref_out.get("sample")
                ref_out = sample if sample is not None else ref_out.get("x")
            elif hasattr(ref_out, "sample"):
                ref_out = ref_out.sample
            elif isinstance(ref_out, tuple):
                ref_out = ref_out[0]
            assert torch.is_tensor(ref_out), f"official output is not tensor: {type(ref_out)}"
            ref_out = ref_out.detach().float().cpu()
        finally:
            for h in up_handles:
                h.remove()
        del official
        gc.collect()
        torch.cuda.empty_cache()

        print("Loading FastVideo model...")
        fv = load_fastvideo(device)
        fv_log: list[dict] = []
        fv_t: dict[str, torch.Tensor] = {}
        fv_handles = _attach_block_hooks(fv, "fv", fv_log, fv_t, drill_layer)
        print("Running FastVideo forward (with hooks)...")
        try:
            with torch.inference_mode():
                # TODO: adapt forward call signature to your component.
                fv_out = fv(**{k: v.clone() for k, v in inputs.items()})
            if isinstance(fv_out, dict):
                sample = fv_out.get("sample")
                fv_out = sample if sample is not None else fv_out.get("x")
            elif hasattr(fv_out, "sample"):
                fv_out = fv_out.sample
            elif isinstance(fv_out, tuple):
                fv_out = fv_out[0]
            assert torch.is_tensor(fv_out), f"FastVideo output is not tensor: {type(fv_out)}"
            fv_out = fv_out.detach().float().cpu()
        finally:
            for h in fv_handles:
                h.remove()
    finally:
        _restore_hypothesis_patch()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    up_path = LOG_DIR / f"{FAMILY}_{COMPONENT}_up_layers.log"
    fv_path = LOG_DIR / f"{FAMILY}_{COMPONENT}_fv_layers.log"
    _write_log(up_log, up_path)
    _write_log(fv_log, fv_path)
    print(f"\nLogs: {up_path}  {fv_path}\nDiff: diff {up_path} {fv_path}")

    by_name: dict[str, dict] = {}
    for entry in up_log + fv_log:
        by_name.setdefault(entry["name"], {})[entry["side"]] = entry
    first_div = _print_table(by_name, drill_layer)

    print()
    if first_div is not None:
        print(f"First block exceeding {REL_THRESHOLD * 100:.2f}% drift: block[{first_div:02d}]")
        print(f"Re-run with {DRILL_LAYER_ENV}={first_div} to drill submodules.")
    else:
        print(f"No block exceeded {REL_THRESHOLD * 100:.2f}% -- divergence is amortized or pre-block.")

    diff = (ref_out - fv_out).abs()
    print(f"\nFinal  ref_abs={ref_out.abs().mean():.6f}  fv_abs={fv_out.abs().mean():.6f}  "
          f"diff_max={diff.max():.6f}  diff_mean={diff.mean():.6f}")
    _print_elementwise(up_t, fv_t, drill_layer)
    if patched:
        print(f"\n[debug] Hypothesis {HYPOTHESIS_ENV}=1 was active this run.")


if __name__ == "__main__":
    main()
