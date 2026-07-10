# SPDX-License-Identifier: Apache-2.0
"""Trace Klein Flux2 dense-vs-TP2 transformer drift.

Run only on Modal/GPU, for example:

    FLUX2_MODEL_DIR=/root/data/official_weights/black-forest-labs__FLUX.2-klein-4B \
    python -m torch.distributed.run --nproc_per_node=2 \
        tests/local_tests/flux2/debug_flux2_klein_tp_trace.py

Rank 0 runs a dense Diffusers reference forward, then both ranks run the
FastVideo TP2 forward with identical tensors. Rank 0 compares captured top-level
module outputs and writes diff-friendly summaries to /tmp/opencode.
"""

from __future__ import annotations

from collections.abc import Iterable
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


TRACE_DIR = Path("/tmp/opencode")
REF_LOG = TRACE_DIR / "flux2_klein_tp2_ref_layers.log"
FV_LOG = TRACE_DIR / "flux2_klein_tp2_fv_layers.log"
DIFF_LOG = TRACE_DIR / "flux2_klein_tp2_diff.log"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_safetensors_paths(model_dir: Path) -> list[str]:
    paths = sorted(str(path) for path in model_dir.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")
    return paths


def _collect_safetensors_keys(paths: Iterable[str]) -> set[str]:
    from safetensors.torch import safe_open

    keys: set[str] = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys


def _load_tensor_from_safetensors(paths: Iterable[str], key: str) -> torch.Tensor:
    from safetensors.torch import safe_open

    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    raise KeyError(f"Could not find {key!r} in safetensors files")


class _DenseLinearTuple(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return torch.nn.functional.linear(x, self.weight, self.bias), None


def _record_tensor(trace: dict[str, torch.Tensor], name: str, tensor: torch.Tensor) -> None:
    trace[name] = tensor.detach().float().cpu()


def _record_output(trace: dict[str, torch.Tensor], name: str, output: Any) -> None:
    if torch.is_tensor(output):
        _record_tensor(trace, name, output)
        return
    if isinstance(output, tuple) and len(output) == 2 and torch.is_tensor(output[0]) and output[1] is None:
        _record_tensor(trace, name, output[0])
        return
    if isinstance(output, (tuple, list)):
        for index, item in enumerate(output):
            _record_output(trace, f"{name}[{index}]", item)


def _get_submodule(model: torch.nn.Module, name: str) -> torch.nn.Module | None:
    current: torch.nn.Module = model
    for part in name.split("."):
        if part.isdigit() and isinstance(current, torch.nn.ModuleList):
            current = current[int(part)]
            continue
        child = getattr(current, part, None)
        if not isinstance(child, torch.nn.Module):
            return None
        current = child
    return current


def _set_submodule(model: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    child_name = parts[-1]
    parent = _get_submodule(model, parent_name) if parent_name else model
    if parent is None:
        raise AttributeError(f"Could not find parent module for {name!r}")
    if child_name.isdigit() and isinstance(parent, torch.nn.ModuleList):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _patch_dense_linear_tuple(
    model: torch.nn.Module,
    weight_paths: Iterable[str],
    available_keys: set[str],
    module_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    weight_key = f"{module_name}.weight"
    bias_key = f"{module_name}.bias"
    weight = _load_tensor_from_safetensors(weight_paths, weight_key).to(device=device, dtype=dtype)
    bias = None
    if bias_key in available_keys:
        bias = _load_tensor_from_safetensors(weight_paths, bias_key).to(device=device, dtype=dtype)
    _set_submodule(model, module_name, _DenseLinearTuple(weight, bias).to(device=device))
    print(f"[debug] Patched {module_name} to replicated dense F.linear")


def _attach_hooks(
    model: torch.nn.Module,
    names: Iterable[str],
    trace: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name in names:
        module = _get_submodule(model, name)
        if module is None:
            print(f"[trace] missing module {name}")
            continue

        def _hook(_module, _inputs, output, *, hook_name=name):
            _record_output(trace, hook_name, output)

        handles.append(module.register_forward_hook(_hook))
    return handles


def _trace_names(num_double: int, num_single: int) -> list[str]:
    names = [
        "time_guidance_embed",
        "double_stream_modulation_img",
        "double_stream_modulation_txt",
        "single_stream_modulation",
        "x_embedder",
        "context_embedder",
    ]
    names.extend(f"transformer_blocks.{idx}" for idx in range(num_double))
    names.extend(f"single_transformer_blocks.{idx}" for idx in range(num_single))
    names.extend(["norm_out", "proj_out"])
    drill_double = os.getenv("FLUX2_KLEIN_TP_TRACE_DRILL_DOUBLE_BLOCK", "")
    if drill_double:
        base = f"transformer_blocks.{int(drill_double)}"
        names.extend(
            f"{base}.{suffix}"
            for suffix in (
                "norm1",
                "norm1_context",
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.norm_q",
                "attn.norm_k",
                "attn.norm_added_q",
                "attn.norm_added_k",
                "attn.to_add_out",
                "attn.to_out.0",
                "attn",
                "norm2",
                "ff.linear_in",
                "ff.act_fn",
                "ff.linear_out",
                "ff",
                "norm2_context",
                "ff_context.linear_in",
                "ff_context.act_fn",
                "ff_context.linear_out",
                "ff_context",
            )
        )
    return names


def _write_trace(path: Path, trace: dict[str, torch.Tensor]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for name, tensor in trace.items():
            t = tensor.float()
            f.write(
                f"{name} {tuple(t.shape)} "
                f"{t.abs().mean().item():.8f} {t.sum().item():.8f} "
                f"{t.min().item():.8f} {t.max().item():.8f}\n"
            )


def _compare_traces(
    ref_trace: dict[str, torch.Tensor],
    fv_trace: dict[str, torch.Tensor],
) -> None:
    first = None
    with DIFF_LOG.open("w", encoding="utf-8") as f:
        for name, ref_tensor in ref_trace.items():
            fv_tensor = fv_trace.get(name)
            if fv_tensor is None:
                f.write(f"{name} missing_on_fastvideo\n")
                if first is None:
                    first = (name, "missing")
                continue
            if tuple(ref_tensor.shape) != tuple(fv_tensor.shape):
                f.write(f"{name} shape ref={tuple(ref_tensor.shape)} fv={tuple(fv_tensor.shape)}\n")
                if first is None:
                    first = (name, "shape")
                continue
            diff = (ref_tensor - fv_tensor).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            median_diff = diff.median().item()
            f.write(
                f"{name} max={max_diff:.8f} mean={mean_diff:.8f} "
                f"median={median_diff:.8f} ref_abs={ref_tensor.abs().mean().item():.8f} "
                f"fv_abs={fv_tensor.abs().mean().item():.8f}\n"
            )
            if first is None and max_diff > 0:
                first = (name, f"max={max_diff:.8f} mean={mean_diff:.8f}")
    if first is None:
        print("[trace] no divergence across captured tensors")
    else:
        print(f"[trace] first divergence: {first[0]} {first[1]}")
    print(f"[trace] wrote {REF_LOG}")
    print(f"[trace] wrote {FV_LOG}")
    print(f"[trace] wrote {DIFF_LOG}")
    if DIFF_LOG.exists():
        print(DIFF_LOG.read_text(encoding="utf-8"))


def _build_inputs(
    *,
    batch: int,
    img_h: int,
    img_w: int,
    txt_len: int,
    in_channels: int,
    joint_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    seq_len = img_h * img_w
    hidden = torch.randn(batch, seq_len, in_channels, dtype=torch.float32)
    encoder = torch.randn(batch, txt_len, joint_dim, dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    guidance = torch.zeros_like(timestep)
    txt_ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(1),
        torch.arange(1),
        torch.arange(txt_len),
    )
    img_ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(img_h),
        torch.arange(img_w),
        torch.arange(1),
    )
    return hidden, encoder, timestep, guidance, txt_ids, img_ids


def _run_reference(
    transformer_dir: Path,
    cfg: dict[str, Any],
    names: list[str],
    dtype: torch.dtype,
    device: torch.device,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from diffusers import Flux2Transformer2DModel as RefTransformer

    trace: dict[str, torch.Tensor] = {}
    ref = RefTransformer.from_pretrained(
        str(transformer_dir),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).eval().to(device)

    class _ZeroGuidance(torch.nn.Module):
        def __init__(self, embedding_dim: int):
            super().__init__()
            self.embedding_dim = embedding_dim

        def forward(self, guidance_proj: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                guidance_proj.shape[0],
                self.embedding_dim,
                device=guidance_proj.device,
                dtype=guidance_proj.dtype,
            )

    ref.time_guidance_embed.guidance_embedder = _ZeroGuidance(int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"]))
    handles = _attach_hooks(ref, names, trace)
    hidden, encoder, timestep, guidance, txt_ids, img_ids = inputs
    try:
        with torch.no_grad():
            output = ref(
                hidden_states=hidden.to(device=device, dtype=dtype),
                encoder_hidden_states=encoder.to(device=device, dtype=dtype),
                timestep=timestep.to(device=device, dtype=dtype),
                img_ids=img_ids.to(device=device),
                txt_ids=txt_ids.to(device=device),
                guidance=guidance.to(device=device, dtype=dtype),
                return_dict=False,
            )[0]
        _record_tensor(trace, "output", output)
    finally:
        for handle in handles:
            handle.remove()
        del ref
        gc.collect()
        torch.cuda.empty_cache()
    _ = cfg
    return trace


def _run_fastvideo_tp(
    transformer_dir: Path,
    cfg: dict[str, Any],
    names: list[str],
    dtype: torch.dtype,
    device: torch.device,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int,
) -> dict[str, torch.Tensor]:
    from fastvideo.configs.models.dits.flux_2 import Flux2Config
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model
    from fastvideo.models.registry import ModelRegistry

    fv_cls, _ = ModelRegistry.resolve_model_cls("Flux2Transformer2DModel")
    dit_cfg = Flux2Config()
    dit_cfg.update_model_arch(dict(cfg))
    update_fn = getattr(dit_cfg.arch_config, "update_from_weight_keys", None)
    weight_paths = _collect_safetensors_paths(transformer_dir)
    weight_keys = _collect_safetensors_keys(weight_paths)
    if callable(update_fn):
        update_fn(weight_keys)

    fv = maybe_load_fsdp_model(
        model_cls=fv_cls,
        init_params={"config": dit_cfg, "hf_config": dict(cfg)},
        weight_dir_list=weight_paths,
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        strict=True,
        cpu_offload=False,
        fsdp_inference=False,
        default_dtype=dtype,
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        output_dtype=None,
        training_mode=False,
        pin_cpu_memory=False,
    ).eval()

    dense_modules: list[str] = []
    if os.getenv("FLUX2_KLEIN_TP_TRACE_PATCH_CONTEXT_DENSE", "0") == "1":
        dense_modules.append("context_embedder")
    if os.getenv("FLUX2_KLEIN_TP_TRACE_PATCH_BLOCK0_FF_CONTEXT_OUT_DENSE", "0") == "1":
        dense_modules.append("transformer_blocks.0.ff_context.linear_out")
    dense_modules.extend(
        name.strip()
        for name in os.getenv("FLUX2_KLEIN_TP_TRACE_DENSE_LINEAR_MODULES", "").split(",")
        if name.strip()
    )
    for module_name in dict.fromkeys(dense_modules):
        _patch_dense_linear_tuple(fv, weight_paths, weight_keys, module_name, device, dtype)

    trace: dict[str, torch.Tensor] = {}
    handles = _attach_hooks(fv, names, trace) if rank == 0 else []
    hidden, encoder, timestep, guidance, txt_ids, img_ids = inputs
    try:
        with torch.no_grad():
            with set_forward_context(current_timestep=0, attn_metadata=None):
                output = fv(
                    hidden_states=hidden.to(device=device, dtype=dtype),
                    encoder_hidden_states=encoder.to(device=device, dtype=dtype),
                    timestep=timestep.to(device=device, dtype=dtype),
                    img_ids=img_ids.to(device=device),
                    txt_ids=txt_ids.to(device=device),
                    guidance=guidance.to(device=device, dtype=dtype),
                )
        if rank == 0:
            _record_tensor(trace, "output", output)
    finally:
        for handle in handles:
            handle.remove()
        del fv
        gc.collect()
        torch.cuda.empty_cache()
    return trace


def main() -> None:
    model_dir = Path(os.getenv("FLUX2_MODEL_DIR", ""))
    if not model_dir.exists():
        raise RuntimeError("Set FLUX2_MODEL_DIR to the Klein checkpoint directory")
    transformer_dir = model_dir / "transformer"
    cfg = _load_json(transformer_dir / "config.json")
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)

    num_double = int(cfg.get("num_layers", 19))
    num_single = int(cfg.get("num_single_layers", 38))
    names = _trace_names(num_double, num_single)
    inputs = _build_inputs(
        batch=1,
        img_h=8,
        img_w=8,
        txt_len=16,
        in_channels=int(cfg["in_channels"]),
        joint_dim=int(cfg["joint_attention_dim"]),
    )

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ref_trace: dict[str, torch.Tensor] = {}
    if rank == 0:
        ref_trace = _run_reference(transformer_dir, cfg, names, dtype, device, inputs)
        _write_trace(REF_LOG, ref_trace)

    dist.barrier()
    fv_trace = _run_fastvideo_tp(transformer_dir, cfg, names, dtype, device, inputs, rank)
    dist.barrier()

    if rank == 0:
        _write_trace(FV_LOG, fv_trace)
        _compare_traces(ref_trace, fv_trace)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
