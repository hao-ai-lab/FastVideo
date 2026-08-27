# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtune
# Copyright 2024 The TorchTune Authors.
# Copyright 2025 The FastVideo Authors.

from __future__ import annotations
import os
import contextlib
import re
from collections.abc import Callable, Generator
from itertools import chain
from typing import Any

import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import distribute_tensor
from torch.distributed.fsdp import (CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy, fully_shard)
from torch.nn.modules.module import _IncompatibleKeys

from fastvideo.logger import init_logger
from fastvideo.models.loader.lora_patch import DenseLoRAPatch
from fastvideo.models.loader.utils import (get_param_names_mapping, hf_to_custom_state_dict)
from fastvideo.models.loader.weight_utils import safetensors_weights_iterator
from fastvideo.utils import set_mixed_precision_policy, is_pin_memory_available

logger = init_logger(__name__)


def _summarize_param_names(names: set[str]) -> str:
    """Collapse per-layer parameter names into one ``blocks.*.suffix xN`` entry each."""
    families: dict[str, int] = {}
    for name in names:
        families[re.sub(r"\.\d+\.", ".*.", name)] = families.get(re.sub(r"\.\d+\.", ".*.", name), 0) + 1
    return ", ".join(f"{family} x{count}" if count > 1 else family for family, count in sorted(families.items()))


def _maybe_quantize_model(model: nn.Module) -> None:
    """Quantize NVFP4- or FP8-tagged linear layers in-place after weights are loaded.

    Walks the module tree once, looking for layers whose ``quant_method``
    is an :class:`NVFP4QuantizeMethod` or :class:`FP8QuantizeMethod` (attached
    at construction time by the respective ``get_quant_method``). When at least
    one such layer exists, calls the matching conversion function to register
    quantized weight buffers on each targeted layer.

    The walk returns on the first quantized layer found so unquantized callers
    pay only an ``isinstance`` check per module. Both imports are deferred so
    this is a no-op on hosts without the relevant backends.

    QAT-*train* linears (``nvfp4_qat_train``) need no weight conversion — the
    master weight stays full precision — but their attachment is otherwise
    silent, so the same walk emits a one-line receipt with the count of
    linears that actually carry the QAT method.
    """
    # Defer imports: these modules pull in heavy symbols at module-load time.
    from fastvideo.layers.linear import LinearBase
    from fastvideo.layers.quantization.nvfp4_config import (
        NVFP4QuantizeMethod,
        convert_model_to_nvfp4,
    )
    from fastvideo.layers.quantization.nvfp4_qat_config import (
        NVFP4QATQuantizeMethod,
        convert_model_to_fp4,
    )
    from fastvideo.layers.quantization.nvfp4_qat_train_config import (
        NVFP4QATTrainQuantizeMethod, )
    from fastvideo.layers.quantization.fp8_config import (
        FP8QuantizeMethod,
        convert_model_to_fp8,
    )

    qat_train_attached = 0
    qat_train_skipped = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, NVFP4QuantizeMethod):
            logger.info("Converting loaded model weights for NVFP4 linear layers")
            convert_model_to_nvfp4(model)
            return
        if isinstance(qm, NVFP4QATQuantizeMethod):
            logger.info("Converting loaded model weights for NVFP4-QAT linear layers")
            convert_model_to_fp4(model)
            return
        if isinstance(qm, FP8QuantizeMethod):
            logger.info("Converting loaded model weights for FP8 linear layers")
            convert_model_to_fp8(model)
            return
        # QAT-train configs are mutually exclusive with the inference schemes
        # above (one quant_config per model), so when they're active the loop
        # always runs to completion and the counts below are model-wide.
        if isinstance(qm, NVFP4QATTrainQuantizeMethod):
            qat_train_attached += 1
        elif isinstance(mod, LinearBase):
            qat_train_skipped += 1
    if qat_train_attached:
        logger.info("NVFP4 QAT: attached %d linears (%d skipped by prefix filter)", qat_train_attached,
                    qat_train_skipped)


# TODO(PY): move this to utils elsewhere
@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def _prepare_model_for_compile(model: nn.Module, *, regional: bool) -> str | None:
    """Run a model compile hook, preferring its regional specialization."""
    prepare = None
    hook_name = "prepare_for_compile"
    if regional:
        prepare = getattr(model, "prepare_for_regional_compile", None)
        hook_name = "prepare_for_regional_compile"
    if not callable(prepare):
        prepare = getattr(model, "prepare_for_compile", None)
        hook_name = "prepare_for_compile"
    if not callable(prepare):
        return None
    logger.info("Running %s for %s", hook_name, type(model).__name__)
    unsupported = prepare()
    return unsupported if isinstance(unsupported, str) and unsupported else None


# Supports optional torch.compile for FSDP-wrapped models during training
def maybe_load_fsdp_model(
    model_cls: type[nn.Module],
    init_params: dict[str, Any],
    weight_dir_list: list[str],
    device: torch.device,
    hsdp_replicate_dim: int,
    hsdp_shard_dim: int,
    default_dtype: torch.dtype,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    strict: bool = True,
    cpu_offload: bool = False,
    fsdp_inference: bool = False,
    output_dtype: torch.dtype | None = None,
    training_mode: bool = True,
    pin_cpu_memory: bool = True,
    enable_torch_compile: bool = False,
    torch_compile_kwargs: dict[str, Any] | None = None,
    inference_regional_compile: bool = False,
    inference_vsa_tile_size: int | None = None,
    lora_path: str | None = None,
    lora_strength: float = 1.0,
) -> torch.nn.Module:
    """
    Load the model with FSDP if is training, else load the model without FSDP.

    ``lora_path`` is consulted only for the part of an adapter that addresses whole
    parameters (``.diff`` / ``.set_weight``); see
    :mod:`fastvideo.models.loader.lora_patch`. The low-rank half is merged later by
    ``LoRAPipeline``. Passing it here is what lets an adapter contribute a parameter the
    base checkpoint does not contain, which has to happen while the tensor is still
    unsharded.
    """
    # NOTE(will): cast_forward_inputs=True shouldn't be needed as we are
    # manually casting the inputs to the model
    mp_policy = MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype, cast_forward_inputs=False)

    set_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        mp_policy=mp_policy,
    )

    logger.info("Loading model with default_dtype: %s", default_dtype)
    with set_default_dtype(default_dtype), torch.device("meta"):
        model = model_cls(**init_params)

    dtype_selector = getattr(model, "_get_parameter_dtype", None)
    has_mixed_parameter_dtypes = callable(dtype_selector) and any(
        dtype_selector(name, param_dtype) != param_dtype for name, _ in model.named_parameters())
    if training_mode and has_mixed_parameter_dtypes:
        raise NotImplementedError("FSDP training with model-selected mixed parameter dtypes requires "
                                  "separate gradient synchronization for replicated parameters.")

    # Check if we should use FSDP
    use_fsdp = training_mode or fsdp_inference

    # Disable FSDP for MPS as it's not compatible
    from fastvideo.platforms import current_platform
    if current_platform.is_mps():
        use_fsdp = False
        logger.info("Disabling FSDP for MPS platform as it's not compatible")

    if use_fsdp:
        pin_cpu_memory = pin_cpu_memory and is_pin_memory_available()
        world_size = hsdp_replicate_dim * hsdp_shard_dim
        if not training_mode and not fsdp_inference:
            hsdp_replicate_dim = world_size
            hsdp_shard_dim = 1

        if current_platform.is_npu():
            with torch.device("cpu"):
                device_mesh = init_device_mesh(
                    "npu",
                    # (Replicate(), Shard(dim=0))
                    mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
                    mesh_dim_names=("replicate", "shard"),
                )
        else:
            device_mesh = init_device_mesh(
                "cuda",
                # (Replicate(), Shard(dim=0))
                mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
                mesh_dim_names=("replicate", "shard"),
            )
        shard_model(model,
                    cpu_offload=cpu_offload,
                    reshard_after_forward=True,
                    mp_policy=mp_policy,
                    mesh=device_mesh,
                    fsdp_shard_conditions=model._fsdp_shard_conditions,
                    pin_cpu_memory=pin_cpu_memory)

    weight_iterator = safetensors_weights_iterator(weight_dir_list, to_cpu=True)
    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
    dense_lora_patch = DenseLoRAPatch.from_adapter(
        lora_path,
        param_names_mapping_fn,
        strength=lora_strength,
    )
    if dense_lora_patch is not None:
        # H3's compression gate is created only by the VSA attention backend. Loading a
        # VSA student under dense attention would otherwise warn about 50 unmatched
        # replacements and continue with a silently incomplete model.
        model_parameter_names = {name for name, _ in model.named_parameters()}
        missing_vsa_gates = sorted(name for name in dense_lora_patch.replacement_parameters
                                   if "gate_compress" in name and name not in model_parameter_names)
        if missing_vsa_gates:
            raise ValueError(
                "This LoRA adapter provides MiniMax H3 VSA compression gates, but the selected attention backend "
                "did not construct them. Use attention_backend='VIDEO_SPARSE_ATTN_H3'. Missing parameters: "
                + ", ".join(missing_vsa_gates[:3]) + (" ..." if len(missing_vsa_gates) > 3 else ""))
    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        device,
        default_dtype,
        strict=strict,
        cpu_offload=cpu_offload,
        param_names_mapping=param_names_mapping_fn,
        dense_lora_patch=dense_lora_patch,
    )
    if hasattr(model, "materialize_non_persistent_buffers"):
        model.materialize_non_persistent_buffers(device=device, dtype=default_dtype)
    for n, p in chain(model.named_parameters(), model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {n} on meta device.")
        # Avoid unintended computation graph accumulation during inference
        if isinstance(p, torch.nn.Parameter):
            p.requires_grad = False

    # Post-load weight quantization. We detect the active scheme by the
    # ``quant_method`` attached to each linear layer at construction time
    # (via ``QuantizationConfig.get_quant_method``). The loader's
    # responsibility is just to materialize the quantized weight buffers
    # from the freshly-loaded bf16 weights. No-op when no quantized layers
    # are present (lazy imports inside the helper).
    _maybe_quantize_model(model)

    compile_in_loader = enable_torch_compile and training_mode
    if compile_in_loader:
        unsupported = _prepare_model_for_compile(model, regional=False)
        if unsupported is not None:
            logger.warning("Training torch.compile requested but disabled: %s. Model stays eager.", unsupported)
        else:
            compile_kwargs = torch_compile_kwargs or {}
            logger.info("Enabling torch.compile for FSDP training module with kwargs=%s", compile_kwargs)
            model = torch.compile(model, **compile_kwargs)
            logger.info("torch.compile enabled for %s", type(model).__name__)
    elif inference_regional_compile and not training_mode:
        # Inference-side counterpart of the #1718 training regional compile:
        # per-block fullgraph compile right after the transformer loads, no
        # user kwargs needed (fullgraph + emulate_precision_casts injected).
        unsupported = _regional_compile_unsupported_reason(
            init_params,
            vsa_tile_size=inference_vsa_tile_size,
        )
        if unsupported is None:
            unsupported = _prepare_model_for_compile(model, regional=True)
        if unsupported is not None:
            logger.warning(
                "inference_torch_compile requested but disabled: %s. "
                "Inference continues in eager mode.", unsupported)
        else:
            attention_count = _enable_regional_attention_compile(model)
            logger.info("Enabled attention tracing for %d modules in %s", attention_count, type(model).__name__)
            _compile_model_regions(model, torch_compile_kwargs or {})
    return model


def _regional_compile_unsupported_reason(
    init_params: dict[str, Any],
    *,
    vsa_tile_size: int | None = None,
) -> str | None:
    """Return why regional fullgraph compile cannot run, or None if it can.

    Dense FA2, FA3, and FA4 inference all route through compile-visible
    custom-op boundaries. FA3's raw autograd.Function carve-out applies only
    to grad-enabled calls, outside this inference-only loader path.

    The legacy VSA backend remains outside the fullgraph support envelope.
    MiniMax H3's VSA backend is supported only through the inference-only
    sm_100a tile-64 route; its regional hook resolves loaded compression
    gates and probes the kernel before block capture.
    """
    try:
        from fastvideo.attention.layer import _attention_compile_explicitly_disabled
    except Exception:  # pragma: no cover - attention stack not importable
        pass
    else:
        if _attention_compile_explicitly_disabled():
            # The escape hatch wraps attention forwards in
            # torch.compiler.disable, which is a hard dynamo error inside a
            # fullgraph region ("Skip inlining `torch.compiler.disable()`d
            # function"). Degrade to eager instead, matching the hatch's
            # debugging intent.
            return ("FASTVIDEO_DISABLE_ATTENTION_COMPILE=1 keeps attention "
                    "forwards out of compiled graphs via torch.compiler."
                    "disable, which fullgraph regional compile cannot trace; "
                    "this model stays eager")
    config = init_params.get("config")
    resolved = getattr(config, "_resolved_attention_backend", None)
    resolved_name = getattr(resolved, "name", "")
    if resolved_name == "VIDEO_SPARSE_ATTN_H3":
        if os.environ.get("FASTVIDEO_H3_VSA_PROBE"):
            return ("FASTVIDEO_H3_VSA_PROBE records tensors and files from the VSA-H3 attention body, which "
                    "regional fullgraph compile cannot capture; this model stays eager")
        if os.environ.get("FASTVIDEO_VSA_SM100A", "0") != "1":
            return ("VIDEO_SPARSE_ATTN_H3 regional compile requires the compile-safe sm_100a route "
                    "(FASTVIDEO_VSA_SM100A=1); Triton/CuTe VSA stays eager")
        if vsa_tile_size != 64:
            return ("VIDEO_SPARSE_ATTN_H3 regional compile requires VSA_tile_size=64; "
                    f"got {vsa_tile_size!r}, so tile-256/CuTe VSA stays eager")
    if resolved_name == "VIDEO_SPARSE_ATTN":
        return (f"attention backend resolved to {resolved_name}, whose Triton "
                "kernels, sequence-parallel collectives, and sync metadata "
                "guard graph-break (incompatible with fullgraph regional "
                "compile); this model stays eager")
    return None


def _enable_regional_attention_compile(model: nn.Module) -> int:
    """Opt in distributed-attention instances owned by ``model`` only."""
    from fastvideo.attention.layer import DistributedAttention

    enabled_count = 0
    for submodule in model.modules():
        if isinstance(submodule, DistributedAttention):
            submodule._set_compile_forward_enabled(True)
            enabled_count += 1
    return enabled_count


def _compile_model_regions(model: nn.Module, compile_kwargs: dict[str, Any]) -> int:
    """Compile repeated mathematical regions of a loaded model.

    Only the selected module ``forward`` is replaced. This keeps activation
    checkpoint wrappers structurally transparent while any module-level hooks
    (FSDP pre/post, layerwise offload) execute outside the compiled region.
    """
    compile_conditions = getattr(model, "_compile_conditions", None)
    if not compile_conditions:
        raise ValueError(f"{type(model).__name__} does not declare _compile_conditions")

    if compile_kwargs.get("fullgraph", True) is not True:
        raise ValueError("Regional compile requires fullgraph=True")
    if "mode" in compile_kwargs:
        # torch.compile forbids passing both `mode` and `options`, and
        # regional compile always injects options (emulate_precision_casts)
        # to match the training-side regional-compile configuration. Fail here
        # with an actionable message instead of letting torch raise a
        # mode/options conflict about an `options` key the user never wrote.
        raise ValueError("Regional compile sets inductor options "
                         "(emulate_precision_casts) and cannot be combined "
                         "with torch_compile_kwargs['mode']. Remove 'mode' or "
                         "express its effect via torch_compile_kwargs['options'].")
    kwargs = {**compile_kwargs, "fullgraph": True}
    options = {"emulate_precision_casts": True}
    options.update(kwargs.get("options") or {})
    kwargs["options"] = options
    compiled_count = 0
    for name, submodule in list(model.named_modules()):
        if not name:
            continue
        if any(condition(name, submodule) for condition in compile_conditions):
            # Activation checkpoint wrappers are control-flow boundaries, not
            # mathematical regions. Keep their saved-tensor/recompute logic
            # eager and compile only the repeated block they own.
            compile_target = getattr(submodule, "_checkpoint_wrapped_module", submodule)
            compile_target.forward = torch.compile(compile_target.forward, **kwargs)
            compiled_count += 1

    if compiled_count == 0:
        raise ValueError(f"No submodules in {type(model).__name__} matched _compile_conditions")
    logger.info(
        "Enabled regional torch.compile for %d submodules in %s with kwargs=%s",
        compiled_count,
        type(model).__name__,
        kwargs,
    )
    return compiled_count


def shard_model(
    model,
    *,
    cpu_offload: bool,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = MixedPrecisionPolicy(),  # noqa
    mesh: DeviceMesh | None = None,
    fsdp_shard_conditions: list[Callable[[str, nn.Module], bool]] = [],  # noqa
    pin_cpu_memory: bool = True,
) -> None:
    """
    Utility to shard a model with FSDP using the PyTorch Distributed fully_shard API.

    This method will over the model's named modules from the bottom-up and apply shard modules
    based on whether they meet any of the criteria from shard_conditions.

    Args:
        model (TransformerDecoder): Model to shard with FSDP.
        shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP. Each function should take module name (relative to root)
            and the module itself, returning True if FSDP should shard the module and False otherwise.
            If any of shard_conditions return True for a given module, it will be sharded by FSDP.
        cpu_offload (bool): If set to True, FSDP will offload parameters, gradients, and optimizer
            states to CPU.
        reshard_after_forward (bool): Whether to reshard parameters and buffers after
            the forward pass. Setting this to True corresponds to the FULL_SHARD sharding strategy
            from FSDP1, while setting it to False corresponds to the SHARD_GRAD_OP sharding strategy.
        mesh (Optional[DeviceMesh]): Device mesh to use for FSDP sharding under multiple parallelism.
            Default to None.
        fsdp_shard_conditions (List[Callable[[str, nn.Module], bool]]): A list of functions to determine
            which modules to shard with FSDP.
        pin_cpu_memory (bool): If set to True, FSDP will pin the CPU memory of the offloaded parameters.

    Raises:
        ValueError: If no layer modules were sharded, indicating that no shard_condition was triggered.
    """
    # Check if we should use size-based filtering
    use_size_filtering = os.environ.get("FASTVIDEO_FSDP2_AUTOWRAP", "0") == "1"

    if not fsdp_shard_conditions:
        logger.warning("No FSDP shard conditions provided; nothing will be sharded.")
        return

    default_param_dtype = getattr(mp_policy, "param_dtype", None)
    dtype_selector = getattr(model, "_get_parameter_dtype", None)
    ignored_params: set[nn.Parameter] = set()
    if callable(dtype_selector) and default_param_dtype is not None:
        ignored_params = {
            parameter
            for name, parameter in model.named_parameters()
            if dtype_selector(name, default_param_dtype) != default_param_dtype
        }
    named_modules = list(model.named_modules())
    ignored_params_by_module = {
        id(module): ignored_params.intersection(set(module.parameters()))
        for _, module in named_modules
    }

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=pin_cpu_memory)

    # iterating in reverse to start with
    # lowest-level modules first
    num_layers_sharded = 0

    if use_size_filtering:
        # Size-based filtering mode
        min_params = int(os.environ.get("FASTVIDEO_FSDP2_MIN_PARAMS", "10000000"))
        logger.info("Using size-based filtering with threshold: %.2fM", min_params / 1e6)

        for n, m in reversed(named_modules):
            if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):
                # Count all parameters
                param_count = sum(p.numel() for p in m.parameters(recurse=True))

                # Skip small modules
                if param_count < min_params:
                    logger.info("Skipping module %s (%.2fM params < %.2fM threshold)", n, param_count / 1e6,
                                min_params / 1e6)
                    continue

                # Shard this module
                logger.info("Sharding module %s (%.2fM params)", n, param_count / 1e6)
                module_kwargs = fsdp_kwargs
                local_ignored_params = ignored_params_by_module[id(m)]
                if local_ignored_params:
                    module_kwargs = {**fsdp_kwargs, "ignored_params": local_ignored_params}
                fully_shard(m, **module_kwargs)
                num_layers_sharded += 1
    else:
        # Shard all modules matching conditions
        for n, m in reversed(named_modules):
            if any([shard_condition(n, m) for shard_condition in fsdp_shard_conditions]):
                module_kwargs = fsdp_kwargs
                local_ignored_params = ignored_params_by_module[id(m)]
                if local_ignored_params:
                    module_kwargs = {**fsdp_kwargs, "ignored_params": local_ignored_params}
                fully_shard(m, **module_kwargs)
                num_layers_sharded += 1

        if num_layers_sharded == 0:
            raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")

    # Finally shard the entire model to account for any stragglers
    root_kwargs = fsdp_kwargs
    if ignored_params:
        root_kwargs = {**fsdp_kwargs, "ignored_params": ignored_params}
    fully_shard(model, **root_kwargs)


# TODO(PY): device mesh for cfg parallel
def load_model_from_full_model_state_dict(
    model: FSDPModule | torch.nn.Module,
    full_sd_iterator: Generator[tuple[str, torch.Tensor], None, None],
    device: torch.device,
    param_dtype: torch.dtype,
    strict: bool = False,
    cpu_offload: bool = False,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]] | None = None,
    training_mode: bool = True,
    dense_lora_patch: DenseLoRAPatch | None = None,
) -> _IncompatibleKeys:
    """
    Converting full state dict into a sharded state dict
    and loading it into FSDP model (if training) or normal huggingface model
    Args:
        model (Union[FSDPModule, torch.nn.Module]): Model to generate fully qualified names for cpu_state_dict
        full_sd_iterator (Generator): an iterator yielding (param_name, tensor) pairs
        device (torch.device): device used to move full state dict tensors
        param_dtype (torch.dtype): dtype used to move full state dict tensors
        strict (bool): flag to check if to load the model in strict mode
        cpu_offload (bool): flag to check if FSDP offload is enabled
        param_names_mapping (Optional[Callable[[str], str]]): a function that maps full param name to sharded param name
        training_mode (bool): apply FSDP only for training
        dense_lora_patch (Optional[DenseLoRAPatch]): whole-parameter adapter payload,
            folded into each full tensor before it is sharded so that FSDP/TP placement
            is inherited from this function rather than reimplemented downstream
    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    Raises:
        NotImplementedError: If got FSDP with more than 1D.
    """
    meta_sd = model.state_dict()
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    sharded_sd = {}
    custom_param_sd, reverse_param_names_mapping = hf_to_custom_state_dict(full_sd_iterator,
                                                                           param_names_mapping)  # type: ignore
    # Drain rather than iterate. Production safetensors values may retain
    # memory-mapped shard storage, while mapped or merged parameters can own
    # ordinary allocations. Keeping the dict retains all of that source
    # storage until loading finishes; popping releases each reference as soon
    # as its conversion completes and lowers the host/unified-memory working
    # set.
    for target_param_name in list(custom_param_sd):
        full_tensor = custom_param_sd.pop(target_param_name)
        meta_sharded_param = meta_sd.get(target_param_name)
        if meta_sharded_param is None:
            # Some checkpoints include extra entries that are not part of the
            # instantiated model's state_dict (e.g. `_extra_state` keys from
            # some FSDP checkpoint formats). These can be safely skipped.
            if (target_param_name.endswith("._extra_state") or target_param_name.endswith("_extra_state")):
                logger.warning(
                    "Skipping non-parameter checkpoint key: %s",
                    target_param_name,
                )
                continue

            # For non-strict loads, treat this as an "unexpected key" and skip it
            # (mirrors torch.nn.Module.load_state_dict(strict=False)).
            if not strict:
                logger.warning(
                    "Skipping unexpected checkpoint key (not present in model): %s",
                    target_param_name,
                )
                continue

            raise ValueError(
                f"Parameter {target_param_name} not found in custom model state dict. The hf to custom mapping may be incorrect."
            )
        target_dtype = param_dtype
        dtype_selector = getattr(model, "_get_parameter_dtype", None)
        if callable(dtype_selector):
            target_dtype = dtype_selector(target_param_name, param_dtype)
        if dense_lora_patch is not None:
            # Returns float32 when a delta was added, so the cast below is what lands
            # the parameter in its storage dtype.
            full_tensor = dense_lora_patch.apply_to(target_param_name, full_tensor)
        if not hasattr(meta_sharded_param, "device_mesh"):
            full_tensor = full_tensor.to(device=device, dtype=target_dtype)
            target_param = named_parameters.get(target_param_name)
            weight_loader = getattr(target_param, "weight_loader", None)
            # Gated on a shape mismatch: only fused/stacked params with a custom
            # weight_loader (e.g. Qwen3's merged QKV/gate-up) take this path.
            # Existing models whose unsharded params match the checkpoint shape
            # fall through to the original `sharded_tensor = full_tensor` below.
            if target_param is not None and callable(weight_loader) and tuple(target_param.shape) != tuple(
                    full_tensor.shape):
                loaded_param = nn.Parameter(torch.empty(tuple(target_param.shape), device=device, dtype=target_dtype),
                                            requires_grad=False)
                for attr_name, attr_value in vars(target_param).items():
                    setattr(loaded_param, attr_name, attr_value)
                weight_loader(loaded_param, full_tensor)
                sharded_tensor = loaded_param.data
            else:
                # In cases where parts of the model aren't sharded, some parameters will be plain tensors.
                sharded_tensor = full_tensor
        else:
            full_tensor = full_tensor.to(device=device, dtype=target_dtype)
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
        if target_param_name in named_buffers:
            sharded_sd[target_param_name] = sharded_tensor
        else:
            sharded_sd[target_param_name] = nn.Parameter(sharded_tensor)

    model.reverse_param_names_mapping = reverse_param_names_mapping
    unused_keys = set(meta_sd.keys()) - set(sharded_sd.keys())
    if unused_keys:
        # Say which of these the adapter is about to fill in. Reporting all of them as
        # "unloaded" was accurate when zero-init was the only outcome; with an adapter
        # supplying real values it reads as a problem that is not one. Names are
        # summarized because a 50-layer model prints 50 near-identical lines otherwise.
        from_adapter = ({key for key in unused_keys if dense_lora_patch.provides(key)} if dense_lora_patch is not None
                        else set())
        zero_init = unused_keys - from_adapter
        if from_adapter:
            logger.info("Parameters absent from the checkpoint and supplied by the LoRA adapter: %d (%s)",
                        len(from_adapter), _summarize_param_names(from_adapter))
        if zero_init:
            logger.warning("Found unloaded parameters in meta state dict, zero-initializing: %d (%s)", len(zero_init),
                           _summarize_param_names(zero_init))

    # List of allowed parameter name patterns
    ALLOWED_NEW_PARAM_PATTERNS = ["gate_compress", "proj_l"]  # Can be extended as needed
    for new_param_name in unused_keys:
        # An adapter that ships the parameter outright both supplies the value and
        # authorizes it: the allowlist exists to catch a checkpoint silently missing a
        # weight, which is not the case when something deliberately provides one.
        adapter_value = (dense_lora_patch.replacement_for(new_param_name) if dense_lora_patch is not None else None)
        if adapter_value is None and not any(pattern in new_param_name for pattern in ALLOWED_NEW_PARAM_PATTERNS):
            logger.error("Unsupported new parameter: %s. Allowed patterns: %s", new_param_name,
                         ALLOWED_NEW_PARAM_PATTERNS)
            raise ValueError(f"New parameter '{new_param_name}' is not supported. "
                             f"Currently only parameters containing {ALLOWED_NEW_PARAM_PATTERNS} are allowed.")
        meta_sharded_param = meta_sd.get(new_param_name)
        target_dtype = param_dtype
        dtype_selector = getattr(model, "_get_parameter_dtype", None)
        if callable(dtype_selector):
            target_dtype = dtype_selector(new_param_name, param_dtype)
        if adapter_value is not None:
            if tuple(adapter_value.shape) != tuple(meta_sharded_param.shape):
                raise ValueError(f"LoRA set_weight for {new_param_name} has shape {tuple(adapter_value.shape)}, "
                                 f"but the parameter is {tuple(meta_sharded_param.shape)}")
            full_tensor = adapter_value.to(device=device, dtype=target_dtype)
            if not hasattr(meta_sharded_param, "device_mesh"):
                sharded_tensor = full_tensor
            else:
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    meta_sharded_param.device_mesh,
                    meta_sharded_param.placements,
                )
                if cpu_offload:
                    sharded_tensor = sharded_tensor.cpu()
        elif not hasattr(meta_sharded_param, "device_mesh"):
            # Initialize with zeros
            sharded_tensor = torch.zeros_like(meta_sharded_param, device=device, dtype=target_dtype)
        else:
            # Initialize with zeros and distribute
            full_tensor = torch.zeros_like(meta_sharded_param, device=device, dtype=target_dtype)
            sharded_tensor = distribute_tensor(
                full_tensor,
                meta_sharded_param.device_mesh,
                meta_sharded_param.placements,
            )
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
        sharded_sd[new_param_name] = nn.Parameter(sharded_tensor)

    if dense_lora_patch is not None:
        dense_lora_patch.report_unapplied()

    # choose `assign=True` since we cannot call `copy_` on meta tensor
    return model.load_state_dict(sharded_sd, strict=strict, assign=True)
