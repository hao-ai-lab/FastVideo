# SPDX-License-Identifier: Apache-2.0
"""Stubs to make upstream daVinci-MagiHuman code importable in-process.

The upstream DiT (daVinci-MagiHuman/inference/model/dit/dit_module.py) hard-
imports SandAI's internal `magi_compiler` + a distributed-runtime init
that requires `torchrun`. Neither is available in a single-process
parity test. This module installs the minimum stubs to let the upstream
DiT load and run on a single GPU with cp_world_size == 1 (which makes
Ulysses's scatter/gather a no-op).

Use:
    from tests.local_tests.helpers.magi_human_upstream import (
        install_stubs, load_upstream_dit,
    )
    install_stubs()
    model = load_upstream_dit(base_shard_dir, device=torch.device("cuda"))
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# magi_compiler stubs — identity decorators.
# ---------------------------------------------------------------------------


def _install_magi_compiler_stub() -> None:
    """Stub magi_compiler + register the `torch.ops.infra.*` ops upstream
    calls via `torch.ops`.

    Upstream code decorates plain Python fns with
    `@magi_register_custom_op(name="infra::flash_attn_func", ...)` and
    then calls them as `torch.ops.infra.flash_attn_func(...)`. Our stub
    decorator has to both (a) preserve the decorated fn for direct call
    sites and (b) register the fn under the advertised torch.ops
    namespace so `torch.ops.infra.*` resolves.

    For parity testing we route `infra::flash_attn_func` through
    `F.scaled_dot_product_attention`, matching the FastVideo DiT's
    kernel choice so drift measured in this test is architectural,
    not kernel-dependent.
    """
    if "magi_compiler" in sys.modules:
        return

    import torch
    import torch.nn.functional as F

    pkg = types.ModuleType("magi_compiler")

    def magi_compile(config_patch=None):
        def decorator(cls_or_fn):
            return cls_or_fn
        return decorator

    # Create one Library per (namespace, schema) pair. Track by namespace
    # so we don't define the same op twice on re-import.
    _libs: dict[str, torch.library.Library] = {}
    _defined: set[tuple[str, str]] = set()

    def _sdpa_flash_attn_func(q, k, v):
        # Upstream shape: [batch=1, L, H, D]. SDPA expects [B, H, L, D]
        # and no native GQA; expand K/V to match Q heads.
        num_heads_q = q.shape[2]
        num_heads_kv = k.shape[2]
        if num_heads_q != num_heads_kv:
            assert num_heads_q % num_heads_kv == 0
            repeat = num_heads_q // num_heads_kv
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        q2 = q.transpose(1, 2).contiguous()
        k2 = k.transpose(1, 2).contiguous()
        v2 = v.transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q2, k2, v2)
        return out.transpose(1, 2).contiguous()

    def _not_implemented_flex(q, k, v, q_ranges, k_ranges):
        # Only hit when local_attn_layers is non-empty; base model has none.
        raise NotImplementedError(
            "flex_flash_attn_func stub: local-attn layers are out of "
            "scope for the base-model parity test."
        )

    def magi_register_custom_op(name=None, mutates_args=(), infer_output_meta_fn=None, is_subgraph_boundary=False, **kwargs):
        def decorator(fn):
            if not name:
                return fn
            namespace, op_name = name.split("::", 1)
            if namespace not in _libs:
                _libs[namespace] = torch.library.Library(namespace, "FRAGMENT")
            if (namespace, op_name) in _defined:
                # Already registered in a previous test run — reuse.
                return fn
            # Route known ops through SDPA; leave unknown ones as direct fn.
            schema_name = f"{op_name}({_infer_schema(fn)}) -> Tensor"
            try:
                _libs[namespace].define(schema_name)
            except Exception:
                pass
            if op_name == "flash_attn_func":
                torch.library.impl(
                    _libs[namespace], op_name, "CUDA"
                )(_sdpa_flash_attn_func)
                torch.library.impl(
                    _libs[namespace], op_name, "CPU"
                )(_sdpa_flash_attn_func)
            elif op_name == "flex_flash_attn_func":
                torch.library.impl(
                    _libs[namespace], op_name, "CUDA"
                )(_not_implemented_flex)
            else:
                # For ops we don't care about (compile-only wrappers), the
                # Python fn path inside the module body is used directly —
                # we just need `torch.ops.<ns>.<op>` to exist so module-
                # load-time attribute lookups succeed.
                torch.library.impl(
                    _libs[namespace], op_name, "CUDA"
                )(fn)
            _defined.add((namespace, op_name))
            return fn
        return decorator

    pkg.magi_compile = magi_compile
    sys.modules["magi_compiler"] = pkg

    api = types.ModuleType("magi_compiler.api")
    api.magi_register_custom_op = magi_register_custom_op
    sys.modules["magi_compiler.api"] = api
    pkg.api = api

    config_mod = types.ModuleType("magi_compiler.config")

    class CompileConfig:
        class offload_config:  # pragma: no cover - pass-through
            gpu_resident_weight_ratio = 1.0
    config_mod.CompileConfig = CompileConfig
    sys.modules["magi_compiler.config"] = config_mod
    pkg.config = config_mod


def _infer_schema(fn) -> str:
    """Return a minimal torch.library schema string for the given fn.

    For our stub we just need *something* parseable; all real ops we
    care about take `(q, k, v)` or `(q, k, v, q_ranges, k_ranges)` or
    variants. Use generic `Tensor a, Tensor b, ...` arg names.
    """
    import inspect
    sig = inspect.signature(fn)
    parts = []
    for i, _ in enumerate(sig.parameters):
        parts.append(f"Tensor a{i}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# distributed / CP stubs — single-GPU, cp_world_size == 1.
# ---------------------------------------------------------------------------


def _install_distributed_stubs() -> None:
    """Monkey-patch upstream distributed + parallelism modules for cp=1."""
    # inference.infra.distributed.*
    # The real module requires NCCL / parallel_state to be initialized
    # from torchrun; here we short-circuit the handful of getters the DiT
    # actually calls.
    import inference.infra.distributed as dist_mod

    dist_mod.get_cp_world_size = lambda: 1
    dist_mod.get_cp_group = lambda: None
    dist_mod.get_cp_rank = lambda: 0
    dist_mod.get_tp_rank = lambda: 0
    dist_mod.get_pp_rank = lambda: 0

    # inference.infra.parallelism.*
    # At cp_world_size=1, scatter/gather are trivially no-ops.
    import inference.infra.parallelism.gather_scatter_primitive as gs

    def _scatter_noop(x, cp_split_sizes, group=None):
        return x

    def _gather_noop(x, cp_split_sizes, group=None):
        return x

    gs.scatter_to_context_parallel_region = _scatter_noop
    gs.gather_from_context_parallel_region = _gather_noop

    # Re-import ulysses_scheduler with patched scatter/gather in place.
    import inference.infra.parallelism.ulysses_scheduler as us
    us.scatter_to_context_parallel_region = _scatter_noop
    us.gather_from_context_parallel_region = _gather_noop
    us.get_cp_world_size = lambda: 1
    us.get_cp_group = lambda: None

    # all-to-all primitives used by flash_attn_with_cp. At cp=1 they are
    # entered only as a no-op path (the `if cp_world_size > 1` branch is
    # skipped), so no stubs needed there.


def install_stubs() -> None:
    """Install all stubs. Idempotent."""
    _install_magi_compiler_stub()
    repo_root = Path(__file__).resolve().parents[3]
    upstream = repo_root / "daVinci-MagiHuman"
    path_s = str(upstream)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)
    # Reload inference.* after sys.path mutation so it picks up the real
    # upstream package (not a stale one).
    for name in list(sys.modules):
        if name == "inference" or name.startswith("inference."):
            del sys.modules[name]
    import inference  # noqa: F401
    _install_distributed_stubs()


# ---------------------------------------------------------------------------
# Upstream DiTModel loader — instantiate + load base shards.
# ---------------------------------------------------------------------------


def _base_arch_dict() -> dict:
    """Return the upstream `ModelConfig`-equivalent dict for the base variant.
    Matches `inference/common/config.py::ModelConfig` defaults for base.
    """
    import torch
    return dict(
        num_layers=40,
        hidden_size=5120,
        head_dim=128,
        num_query_groups=8,
        video_in_channels=48 * 4,
        audio_in_channels=64,
        text_in_channels=3584,
        checkpoint_qk_layernorm_rope=False,
        params_dtype=torch.float32,
        tread_config=dict(
            selection_rate=0.5, start_layer_idx=2, end_layer_idx=25,
        ),
        mm_layers=[0, 1, 2, 3, 36, 37, 38, 39],
        local_attn_layers=[],
        enable_attn_gating=True,
        activation_type="swiglu7",
        gelu7_layers=[0, 1, 2, 3],
        # derived
        num_heads_q=40,
        num_heads_kv=8,
        post_norm_layers=[],
    )


def load_upstream_dit(base_shard_dir, device=None, dtype=None):
    """Instantiate upstream `DiTModel` and load the base shards into it.

    Args:
        base_shard_dir: path to `base/` (contains `model-0000*-of-00007.safetensors`
            and `model.safetensors.index.json`).
        device: torch device (default cuda if available).
        dtype: dtype cast (default: leave checkpoint dtypes as-is).

    Returns:
        An upstream `DiTModel` in `.eval()` mode with weights loaded.
    """
    import glob
    import json
    import types as _types

    import torch
    from safetensors.torch import load_file

    from inference.common.config import ModelConfig  # upstream pydantic class
    from inference.model.dit.dit_module import DiTModel

    arch_dict = _base_arch_dict()
    # ModelConfig is a pydantic BaseModel — build via kwargs.
    model_config = ModelConfig(**arch_dict)

    model = DiTModel(model_config=model_config)

    # Load all base shards into a single state dict.
    base_shard_dir = Path(base_shard_dir)
    shard_paths = sorted(base_shard_dir.glob("*.safetensors"))
    state = {}
    for p in shard_paths:
        state.update(load_file(str(p)))

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Upstream DiT missing {len(missing)} keys: {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"Upstream DiT unexpected {len(unexpected)} keys: {unexpected[:5]}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    if dtype is not None:
        model = model.to(dtype=dtype)
    model.eval()
    return model
