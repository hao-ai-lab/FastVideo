# SPDX-License-Identifier: Apache-2.0
"""Default-off H3 student policy, installed after FSDP and checkpointing."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, CheckpointWrapper
from torch.distributed.fsdp import FSDPModule

from fastvideo.logger import init_logger
from fastvideo.train.utils.compile_cache import CheckpointCompileCache
from fastvideo.train.utils.student_compile import (
    NativeBackwardBlockSizeGuard,
    StudentCompileRegions,
    compile_options,
    record_compile_cleanup_error,
)

logger = init_logger(__name__)


@contextmanager
def _dynamic_dense_capacity() -> Iterator[None]:
    """Keep packed FA4 capacity symbolic across teacher/critic grad modes."""
    source = (r"G\['__import_fastvideo_dot_forward_context'\]\._forward_context\.attn_metadata\."
              r"packed_sequence\.sequence_capacity$")
    previous = torch.compiler.config.dynamic_sources
    sources = ",".join(value for value in (previous, source) if value)
    with torch.compiler.config.patch(dynamic_sources=sources):
        yield


def _validate_policy(cfg: Any, method: Any) -> dict[str, Any]:
    """Check structural requirements without binding training to a launch site."""
    from fastvideo.attention.selector import component_attention_backend

    tc = cfg.training
    student = getattr(method, "student", None)
    if student is None or getattr(student, "attention_backend_name", None) != "VIDEO_SPARSE_ATTN_H3":
        raise ValueError("H3 student compilation requires a VSA-H3 student")
    if tc.vsa_tile_size != 128 or tc.vsa_cache_tile_buf or tc.dit_precision != "fp32":
        raise ValueError("H3 student compilation requires tile-128, uncached tiles and FP32 master weights")
    kwargs = tc.model.torch_compile_kwargs
    if kwargs.get("dynamic") is not True or kwargs.get("fullgraph", True) is not True:
        raise ValueError("H3 student compilation requires dynamic=True and fullgraph=True")
    limit = kwargs.get("recompile_limit")
    if type(limit) is not int or limit <= 0:
        raise ValueError("H3 student compilation requires a positive integer recompile_limit")
    receipt: dict[str, Any] = {}
    for role, model in method._role_models.items():
        transformer = model.transformer
        if not isinstance(transformer, FSDPModule):
            raise ValueError(f"H3 {role} must use FSDP")
        if getattr(getattr(model, "_lora_config", None), "enable", False):
            raise ValueError("H3 full-block student compilation does not support LoRA roles")
        if role == "student":
            backend = component_attention_backend(transformer)
            if getattr(backend, "name", None) != "VIDEO_SPARSE_ATTN_H3":
                raise ValueError("The loaded student does not use VSA-H3")
        if model._trainable:
            if any(p.dtype != torch.float32 for p in transformer.parameters() if p.requires_grad):
                raise ValueError(f"H3 {role} compilation requires FP32 master weights")
            for block in transformer.transformer_blocks:
                if (not isinstance(block, CheckpointWrapper) or block.checkpoint_impl != CheckpointImpl.NO_REENTRANT
                        or "context_fn" in block.checkpoint_fn.keywords):
                    raise ValueError(f"H3 {role} requires full non-reentrant checkpoint wrappers")
        receipt[f"{role}_backend"] = model.attention_backend_name
    return receipt


@contextmanager
def minimax_h3_student_compile(cfg: Any, method: Any) -> Iterator[None]:
    """Install the selected H3 compile policy, restoring only owned replacements."""
    policy = cfg.training.model.minimax_h3_student_compile
    if policy == "off":
        yield
        return
    if policy != "grad_only_full_block":
        raise ValueError("training.model.minimax_h3_student_compile must be off or grad_only_full_block")
    receipt = _validate_policy(cfg, method)
    guard = None
    regions = None
    cache = None
    primary = None
    mc = cfg.training.model
    limit = mc.torch_compile_kwargs["recompile_limit"]
    attention_flags: list[tuple[Any, bool]] = []
    try:
        from fastvideo.attention.layer import _attention_compile_explicitly_disabled
        if _attention_compile_explicitly_disabled():
            raise ValueError("H3 student compilation requires attention compilation to be enabled")
        for module in method.student.transformer.modules():
            setter = getattr(module, "_set_compile_forward_enabled", None)
            if callable(setter):
                attention_flags.append((module, bool(module._compile_forward_enabled)))
                setter(True)
        guard = NativeBackwardBlockSizeGuard()
        if mc.minimax_h3_compile_limit_fallback:
            cache = CheckpointCompileCache()
            dense_kwargs = {**mc.torch_compile_kwargs, "fullgraph": True}
            dense_kwargs["options"] = {"emulate_precision_casts": True, **(dense_kwargs.get("options") or {})}
            for role, model in method._role_models.items():
                cache.install_checkpoints(model.transformer)
                if role != "student":
                    count = cache.install_dense(model.transformer, dense_kwargs, role)
                    logger.info("H3 %s compile recovery wrapped %d dense regions", role, count)
        regions = StudentCompileRegions(method.student, "block", True, None, limit, cudagraphs=False, cache=cache)
        logger.info(
            "H3 student compile installed: policy=%s blocks=%d fullgraph=True dynamic=True "
            "recompile_limit=%d limit_fallback=%s accumulated_recompile_limit=%d "
            "options=%s no_grad=original_eager native_module=%s native_module_id=%d roles=%s", policy,
            regions.count, limit, cache is not None, mc.minimax_h3_accumulated_recompile_limit,
            compile_options(True, cudagraphs=False), getattr(guard.module, "__file__", None), id(guard.module), receipt)
        with (_dynamic_dense_capacity(),
              torch._dynamo.config.patch(accumulated_recompile_limit=mc.minimax_h3_accumulated_recompile_limit)
              if cache is not None else nullcontext()):
            yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures: list[BaseException] = []
        restored = 0
        guard_restored = False
        if regions is not None:
            try:
                regions.close()
            except BaseException as cleanup:
                failures.append(cleanup)
            restored = regions.restored_count
        if cache is not None:
            try:
                cache.close()
            except BaseException as cleanup:
                failures.append(cleanup)
        if guard is not None:
            try:
                guard_owned = guard.module._block_size is guard.guarded
                guard.close()
                guard_restored = guard_owned
            except BaseException as cleanup:
                failures.append(cleanup)
        for module, original_flag in reversed(attention_flags):
            try:
                module._set_compile_forward_enabled(original_flag)
            except BaseException as cleanup:
                failures.append(cleanup)
        try:
            logger.info(
                "H3 student compile scope closed: policy=%s restored_blocks=%d restored_native_guard=%s "
                "cleanup_failures=%d", policy, restored, guard_restored, len(failures))
        except BaseException as cleanup:
            failures.append(cleanup)
        if failures:
            if primary is not None:
                for secondary in failures:
                    record_compile_cleanup_error(primary, secondary)
            else:
                for secondary in failures[1:]:
                    record_compile_cleanup_error(failures[0], secondary)
                raise failures[0]
