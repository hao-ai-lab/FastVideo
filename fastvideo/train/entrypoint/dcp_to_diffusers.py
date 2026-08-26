# SPDX-License-Identifier: Apache-2.0
"""Convert a DCP training checkpoint to a diffusers-style model directory.

Works on a single GPU regardless of how many GPUs were used for training
(DCP handles resharding automatically).

Usage (no torchrun needed)::

    python -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint /path/to/checkpoint-1000 \
        --output-dir /path/to/diffusers_output

Or with torchrun (also fine)::

    torchrun --nproc_per_node=1 \
        -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint ... --output-dir ...

The checkpoint must contain ``metadata.json`` (written by
``CheckpointManager``).  If the checkpoint predates metadata
support, pass ``--config`` explicitly to provide the training
YAML.

Pass ``--verify`` to strictly reload the exported transformer immediately
after writing it, so a key-mapping bug fails here instead of deep inside a
later training/inference launch that loads the exported directory.

Model plugins select the physical component name (for example H3 Ref2VA uses
``transformer_ref``). Training LoRA wrappers are exported as merged native
weights because the public inference pipeline cannot consume those wrappers or
a standalone H3 adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _transformer_component_name(model: Any) -> str:
    """Return the physical Diffusers component owned by a training model."""
    component_name = str(getattr(model, "transformer_module_type", "transformer"))
    if not component_name or component_name in {".", ".."} or os.path.basename(component_name) != component_name:
        raise ValueError(f"Invalid transformer component name: {component_name!r}")
    return component_name


def _native_export_state_dict(
    module: Any,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], bool]:
    """Merge training LoRA wrappers into the native transformer key surface.

    H3 inference does not expose a public adapter-loading API.  Its supported
    training handoff is therefore a normal Diffusers-style component with each
    adapter merged into ``weight`` and every ``base_layer``/``lora_*`` key
    removed.  The returned alias map lets reverse HF-name mapping still consult
    the checkpoint mapping recorded for ``base_layer.weight``.
    """
    import torch

    from fastvideo.layers.lora.linear import BaseLayerWithLoRA

    # Consume the gathered state in place. H3's target projections account for
    # tens of GiB; keeping the original base tensors alive in a second mapping
    # while materializing every merged tensor would nearly duplicate them.
    exported = state_dict
    reverse_lookup_aliases: dict[str, str] = {}
    wrappers = [(name, child) for name, child in module.named_modules()
                if name and isinstance(child, BaseLayerWithLoRA)]
    for prefix, wrapper in wrappers:
        lora_a_key = f"{prefix}.lora_A"
        lora_b_key = f"{prefix}.lora_B"
        base_weight_key = f"{prefix}.base_layer.weight"
        native_weight_key = f"{prefix}.weight"
        missing = [key for key in (lora_a_key, lora_b_key, base_weight_key) if key not in exported]
        if missing:
            raise KeyError(f"LoRA export is missing state for {prefix!r}: {missing}")

        lora_a = exported.pop(lora_a_key)
        lora_b = exported.pop(lora_b_key)
        base_weight = exported.pop(base_weight_key)
        if not all(isinstance(value, torch.Tensor) for value in (lora_a, lora_b, base_weight)):
            raise TypeError(f"LoRA export expected tensor state for {prefix!r}")
        if wrapper.lora_rank is None or wrapper.lora_rank <= 0:
            raise ValueError(f"LoRA export has invalid rank for {prefix!r}: {wrapper.lora_rank!r}")
        alpha = wrapper.lora_alpha if wrapper.lora_alpha is not None else wrapper.lora_rank
        scale = (float(alpha) / float(wrapper.lora_rank)) * float(wrapper.lora_strength)
        exported[native_weight_key] = torch.addmm(
            base_weight,
            wrapper.slice_lora_b_weights(lora_b.to(base_weight)),
            wrapper.slice_lora_a_weights(lora_a.to(base_weight)),
            alpha=scale,
        )
        reverse_lookup_aliases[native_weight_key] = base_weight_key

        base_prefix = f"{prefix}.base_layer."
        for key in [name for name in exported if name.startswith(base_prefix)]:
            native_key = f"{prefix}.{key[len(base_prefix):]}"
            if native_key in exported:
                raise ValueError(f"LoRA export key collision at {native_key!r}")
            exported[native_key] = exported.pop(key)
            reverse_lookup_aliases[native_key] = key

    unexpected_wrapper_keys = [
        name for name in exported if ".base_layer." in name or name.endswith((".lora_A", ".lora_B"))
    ]
    if unexpected_wrapper_keys:
        raise ValueError(f"LoRA export left wrapper-only keys: {unexpected_wrapper_keys[:10]}")
    return exported, reverse_lookup_aliases, bool(wrappers)


def _ensure_distributed() -> None:
    """Set up a single-process distributed env if needed.

    When running under ``torchrun`` the env vars are already set.
    For plain ``python`` we fill in the minimum required vars so
    that ``init_process_group`` succeeds with world_size=1.
    """
    for key, default in [
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ]:
        os.environ.setdefault(key, default)


def _save_role_pretrained(
    *,
    role: str,
    base_model_path: str,
    output_dir: str,
    module_names: list[str] | None = None,
    overwrite: bool = False,
    model: Any,
) -> str:
    """Export a role's modules into a diffusers-style model dir.

    Produces a ``model_path`` loadable by
    ``PipelineComponentLoader`` (``model_index.json``,
    ``transformer/``, ``vae/``, etc. copied from
    ``base_model_path``).
    """
    import shutil
    from pathlib import Path

    import torch
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    from fastvideo.utils import maybe_download_model

    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    local_base = Path(maybe_download_model(str(base_model_path))).resolve()
    dst = Path(os.path.expanduser(str(output_dir))).resolve()

    if _rank() == 0:
        if dst.exists():
            if overwrite:
                shutil.rmtree(dst, ignore_errors=True)
            else:
                raise FileExistsError(f"Refusing to overwrite existing "
                                      f"directory: {dst}. "
                                      "Pass --overwrite to replace it.")

        def _copy_or_link(src: str, dest: str) -> None:
            # Resolve symlinks ourselves: os.link's follow_symlinks=True
            # default isn't honored on all filesystems (e.g. some
            # network/overlay mounts), which can silently hard-link to
            # the symlink itself instead of its target.
            real_src = os.path.realpath(src)
            try:
                os.link(real_src, dest)
            except OSError:
                shutil.copy2(real_src, dest)

        logger.info(
            "Creating pretrained export dir at %s "
            "(base=%s)",
            dst,
            local_base,
        )
        shutil.copytree(
            local_base,
            dst,
            symlinks=False,
            copy_function=_copy_or_link,
        )

    _barrier()

    transformer_component = _transformer_component_name(model)
    modules: dict[str, torch.nn.Module] = {}
    if model.transformer is not None:
        modules[transformer_component] = model.transformer

    if module_names is None:
        module_names = sorted(modules.keys())

    for module_name in module_names:
        if module_name not in modules:
            raise KeyError(f"Role {role!r} does not have module "
                           f"{module_name!r}. "
                           f"Available: {sorted(modules.keys())}")

        module_dir = dst / module_name
        if not module_dir.is_dir():
            raise FileNotFoundError(f"Export directory missing component "
                                    f"dir {module_name!r}: {module_dir}")

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        state_dict = get_model_state_dict(
            modules[module_name],
            options=options,
        )

        if _rank() == 0:
            for path in module_dir.glob("*.safetensors"):
                path.unlink(missing_ok=True)

            # Convert internal parameter names back to HF format.
            # load_model_from_full_model_state_dict builds reverse_param_names_mapping
            # (internal_key → hf_key) and stores it on the module.  Without this,
            # the exported safetensors would have internal keys (e.g.
            # "patch_embedding.proj.bias") and the next load would double-map them
            # (e.g. → "patch_embedding.proj.proj.bias").
            reverse_mapping: dict = getattr(modules[module_name], "reverse_param_names_mapping", {})

            native_state, reverse_lookup_aliases, merged_lora = _native_export_state_dict(
                modules[module_name],
                state_dict,
            )
            tensor_state: dict[str, torch.Tensor] = {}
            for key, value in native_state.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expected tensor in state_dict "
                                    f"for {module_name}.{key}, "
                                    f"got {type(value).__name__}")
                reverse_key = reverse_lookup_aliases.get(key, key)
                if reverse_key in reverse_mapping:
                    hf_key, merge_index, _ = reverse_mapping[reverse_key]
                    if merge_index is not None:
                        logger.warning(
                            "Skipping reverse-mapping for merged param %s "
                            "(merge_index=%s); saving under internal key.",
                            key,
                            merge_index,
                        )
                        hf_key = key
                    key = hf_key
                tensor_state[key] = value.detach().cpu()

            from safetensors.torch import save_file

            out_path = module_dir / "model.safetensors"
            logger.info(
                "Saving %s weights to %s (%s tensors)",
                module_name,
                out_path,
                len(tensor_state),
            )
            save_file(tensor_state, str(out_path))

            metadata_path = dst / "fastvideo_training_export.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "role": role,
                        "transformer_component": transformer_component,
                        "lora": "merged" if merged_lora else "none",
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )

        _barrier()

    return str(dst)


def _strict_reload_verify(
    *,
    output_dir: str,
    training_config: Any,
    module_type: str,
) -> None:
    """Reload the just-exported transformer from disk and fail loudly on
    any key mismatch.

    ``TransformerLoader.load()`` already loads strictly (``strict=True``)
    for every non-Cosmos25 model, so this doesn't add leniency -- it moves
    the failure to right after export, with a clear "the export is broken"
    error, instead of surfacing deep inside a later training/inference
    launch that happens to load this directory.
    """
    from fastvideo.train.utils.moduleloader import load_module_from_path

    logger.info("Verifying export: strictly reloading %s from %s", module_type, output_dir)
    load_module_from_path(
        model_path=output_dir,
        module_type=module_type,
        training_config=training_config,
    )
    logger.info("Strict reload verification passed.")


def convert(
    *,
    checkpoint_dir: str,
    output_dir: str,
    config_path: str | None = None,
    role: str = "student",
    overwrite: bool = False,
    verify: bool = False,
) -> str:
    """Load a DCP checkpoint and export as a diffusers model.

    Returns the path to the exported model directory.
    """
    _ensure_distributed()

    from fastvideo.distributed import (
        maybe_init_distributed_environment_and_model_parallel, )
    from fastvideo.train.utils.builder import build_from_config
    from fastvideo.train.utils.checkpoint import (
        CheckpointManager,
        _resolve_resume_checkpoint,
    )
    from fastvideo.train.utils.config import (
        RunConfig,
        load_run_config,
    )

    import torch.distributed.checkpoint as dcp

    # -- Resolve checkpoint directory --
    resolved = _resolve_resume_checkpoint(
        checkpoint_dir,
        output_dir=checkpoint_dir,
    )
    dcp_dir = resolved / "dcp"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"Missing dcp/ under {resolved}")

    # -- Obtain config --
    cfg: RunConfig
    if config_path is not None:
        cfg = load_run_config(config_path)
    else:
        metadata = CheckpointManager.load_metadata(resolved)
        raw_config = metadata.get("config")
        if raw_config is None:
            raise ValueError("Checkpoint metadata.json does not "
                             "contain 'config'. Pass --config "
                             "explicitly.")
        cfg = _run_config_from_raw(raw_config)

    tc = cfg.training

    # -- Init distributed (1 GPU is enough; DCP reshards) --
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
    )

    # Override distributed config so model loading uses 1 GPU.
    tc.distributed.tp_size = 1
    tc.distributed.sp_size = 1
    tc.distributed.num_gpus = 1
    tc.distributed.hsdp_replicate_dim = 1
    tc.distributed.hsdp_shard_dim = 1

    # -- Build model (loads pretrained weights + FSDP) --
    _, method, _, _ = build_from_config(cfg)

    # -- Load DCP weights into the model --
    states = method.checkpoint_state()
    logger.info(
        "Loading DCP checkpoint from %s",
        resolved,
    )
    dcp.load(states, checkpoint_id=str(dcp_dir))

    # -- Export to diffusers format --
    model = method._role_models[role]
    base_model_path = str(tc.model_path)
    if not base_model_path:
        raise ValueError("Cannot determine base_model_path from "
                         "config. Ensure models.student.init_from "
                         "is set.")

    logger.info(
        "Exporting role=%s to %s (base=%s)",
        role,
        output_dir,
        base_model_path,
    )
    transformer_component = _transformer_component_name(model)
    result = _save_role_pretrained(
        role=role,
        base_model_path=base_model_path,
        output_dir=output_dir,
        overwrite=overwrite,
        model=model,
    )
    logger.info("Export complete: %s", result)

    if verify:
        _strict_reload_verify(
            output_dir=result,
            training_config=tc,
            module_type=transformer_component,
        )

    return result


def _run_config_from_raw(raw: dict[str, Any], ) -> Any:
    """Reconstruct a RunConfig from a raw config dict.

    This mirrors ``load_run_config`` but operates on an
    already-parsed dict (from metadata.json) instead of
    reading from a YAML file.
    """
    from fastvideo.train.utils.config import (
        RunConfig,
        _build_training_config,
        _parse_pipeline_config,
        _require_mapping,
        _require_str,
    )

    models_raw = _require_mapping(
        raw.get("models"),
        where="models",
    )
    models: dict[str, dict[str, Any]] = {}
    for role_key, model_cfg_raw in models_raw.items():
        role_str = _require_str(
            role_key,
            where="models.<role>",
        )
        model_cfg = _require_mapping(
            model_cfg_raw,
            where=f"models.{role_str}",
        )
        models[role_str] = dict(model_cfg)

    method_raw = _require_mapping(
        raw.get("method"),
        where="method",
    )
    method = dict(method_raw)

    callbacks_raw = raw.get("callbacks")
    callbacks: dict[str, dict[str, Any]] = (_require_mapping(
        callbacks_raw,
        where="callbacks",
    ) if callbacks_raw is not None else {})

    pipeline_config = _parse_pipeline_config(
        raw,
        models=models,
    )

    training_raw = _require_mapping(
        raw.get("training"),
        where="training",
    )
    t = dict(training_raw)
    training = _build_training_config(
        t,
        models=models,
        pipeline_config=pipeline_config,
    )

    return RunConfig(
        models=models,
        method=method,
        training=training,
        callbacks=callbacks,
        raw=raw,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=("Convert a DCP training checkpoint to a "
                                                  "diffusers-style model directory. "
                                                  "Only 1 GPU needed (DCP reshards "
                                                  "automatically)."), )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=("Path to checkpoint-<step> dir, its dcp/ "
              "subdir, or an output_dir (auto-picks "
              "latest)."),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Destination for the diffusers model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Training YAML config. If omitted, read "
              "from checkpoint metadata.json."),
    )
    parser.add_argument(
        "--role",
        type=str,
        default="student",
        help="Role to export (default: student).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=("After exporting, strictly reload the transformer from "
              "the exported directory to catch key-mapping bugs "
              "immediately."),
    )
    args = parser.parse_args(sys.argv[1:])

    convert(
        checkpoint_dir=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        role=args.role,
        overwrite=args.overwrite,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
