# SPDX-License-Identifier: Apache-2.0
"""Shared opt-in bound-forward compilation for checkpointed student regions.

Only bound forwards change. Checkpoint/FSDP modules and parameters retain their
owners; no-grad prewalks stay eager. Inductor owns capture and lifetime checks.
The benchmark default remains strict; production can supply a checkpoint-aware
cache-exhaustion policy. Native/FSDP numerical qualification remains separate.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.utils.compile_cache import CheckpointCompileCache

logger = init_logger(__name__)

FFN_COMPILE_OPTIONS = {"emulate_precision_casts": True, "triton.cudagraphs": True}

# These are row-dependent layout inputs, not model constants. Leaving global
# metadata integers specialized forces a graph per sample despite dynamic=True.
# Keep the allowlist restricted to H3's actual forward-context source and leave
# architecture, sparsity, parameters, and process-global cache limits alone.
H3_DYNAMIC_METADATA_SOURCES = (r"G\['__import_fastvideo_dot_forward_context'\]\._forward_context\.attn_metadata\."
                               r"(total_seq_length|pad_tokens|num_prefix_tiles|num_video_tiles|video_tile_spans|"
                               r"variable_block_sizes|untile_combined_index|block_mask_plan)(?:$|\[)")


def compile_options(division_rounding: bool = False, *, cudagraphs: bool = True) -> dict[str, bool]:
    options = dict(FFN_COMPILE_OPTIONS)
    options["triton.cudagraphs"] = cudagraphs
    if division_rounding:
        options["eager_numerics.division_rounding"] = True
    return options


class NativeBackwardBlockSizeGuard:
    """Owned fix for native backward support dispatch hashing a SymInt.

    Ordinary int conversion specializes the quotient with a shape equality guard;
    it does not read an unguarded hint. All original zero/nonintegral and native
    support checks remain in their original functions. No installed bytes change.
    """

    def __init__(self, module: Any = None) -> None:
        if module is None:
            from fastvideo_kernel import block_sparse_attn_bwd_ptx_blackwell
            module = block_sparse_attn_bwd_ptx_blackwell
        self.module = module
        self.original = original = module._block_size
        if getattr(original, "_fastvideo_block_size_guard", False):
            raise ValueError("Native backward block-size guard is already installed")

        @wraps(original)
        def guarded(q: torch.Tensor, variable_block_sizes: torch.Tensor) -> int:
            return int(original(q, variable_block_sizes))

        vars(guarded)["_fastvideo_block_size_guard"] = True
        self.guarded = guarded
        module._block_size = guarded

    def close(self) -> None:
        if self.module._block_size is self.guarded:
            self.module._block_size = self.original


class CompileDiagnostics:
    """Scalar compile provenance; never retain tensors or infer backward FQNs."""

    def __init__(self, emit: Any) -> None:
        self.emit = emit
        self.context: dict[str, Any] = {}
        self.active_region: str | None = None
        self.calls: Counter = Counter()

    def event(self, event: str, **fields: Any) -> None:
        self.emit(event, **self.context, active_region=self.active_region, **fields)

    def call(self, fqn: str, function: Any, args: Any, kwargs: Any) -> Any:
        from torch.utils.checkpoint import _StopRecomputationError
        previous = self.active_region
        self.active_region = fqn
        key = (fqn, self.context.get("execution_phase", "unknown"), torch.is_grad_enabled())
        self.calls[(*key, "attempted")] += 1
        self.event("region_call_started")
        try:
            result = function(*args, **kwargs)
            self.calls[(*key, "returned")] += 1
            return result
        except _StopRecomputationError:
            # Ordinary non-reentrant checkpoint early-stop, not a graph break.
            self.calls[(*key, "checkpoint_early_stop")] += 1
            raise
        except BaseException as exc:
            self.calls[(*key, "failed")] += 1
            self.event("region_call_failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.active_region = previous

    def take_calls(self) -> list[dict[str, Any]]:
        result = [
            dict(fqn=fqn, execution_phase=phase, grad=grad, outcome=outcome, count=count)
            for (fqn, phase, grad, outcome), count in sorted(self.calls.items())
        ]
        self.calls.clear()  # Scalar diagnostics only, not compiler/graph/carry state.
        return result


def validate_recompile_limit(recompile_limit: int | None) -> None:
    if recompile_limit is not None and (type(recompile_limit) is not int or recompile_limit <= 0):
        raise ValueError("student_recompile_limit must be a positive integer or None (unset)")


def compile_student_ffns(student: Any,
                         *,
                         division_rounding: bool = False,
                         diagnostics: CompileDiagnostics | None = None,
                         recompile_limit: int | None = None) -> tuple[int, Counter]:
    return _compile_student_regions(student, "ffn", division_rounding, diagnostics, recompile_limit, cudagraphs=True)


def compile_student_blocks(student: Any,
                           *,
                           diagnostics: CompileDiagnostics,
                           division_rounding: bool = False,
                           recompile_limit: int | None = None,
                           cudagraphs: bool = True) -> tuple[int, Counter]:
    if diagnostics is None:
        raise ValueError("Full-block compilation requires installation/phase receipts, not numerical qualification")
    return _compile_student_regions(student,
                                    "block",
                                    division_rounding,
                                    diagnostics,
                                    recompile_limit,
                                    cudagraphs=cudagraphs)


@contextmanager
def _dynamic_h3_metadata() -> Iterator[None]:
    """Scope symbolic layout sources to an owned grad-enabled H3 block call."""
    from fastvideo import forward_context
    metadata = getattr(forward_context._forward_context, "attn_metadata", None)
    if metadata is None:
        yield
        return
    from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadata
    if not isinstance(metadata, MiniMaxH3VSAMetadata):
        yield
        return
    previous = torch.compiler.config.dynamic_sources
    sources = ",".join(value for value in (previous, H3_DYNAMIC_METADATA_SOURCES) if value)
    with torch.compiler.config.patch(dynamic_sources=sources):
        yield


@contextmanager
def _isolated_block_runtime_routes() -> Iterator[None]:
    """Keep diagnostic set history out of compiled guards, without inventing routes."""
    from fastvideo import forward_context
    metadata = getattr(forward_context._forward_context, "attn_metadata", None)
    original = getattr(metadata, "runtime_routes", None)
    if original is None:
        yield
        return
    scratch: set[str] = set()
    metadata.runtime_routes = scratch
    try:
        yield
    finally:
        metadata.runtime_routes = original
    # Only a normally returned call contributes its actual backend markers.
    # Exceptions, including checkpoint early-stop, restore identity but merge nothing.
    original.update(scratch)


def _compile_student_regions(student: Any, scope: str, division_rounding: bool, diagnostics: CompileDiagnostics | None,
                             recompile_limit: int | None, *, cudagraphs: bool) -> tuple[int, Counter]:
    installed = StudentCompileRegions(student,
                                      scope,
                                      division_rounding,
                                      diagnostics,
                                      recompile_limit,
                                      cudagraphs=cudagraphs)
    return installed.count, installed.calls


def record_compile_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    """Retain secondary exceptions on Python 3.10 too, even if reporting fails."""
    errors = vars(primary).setdefault("_fastvideo_compile_cleanup_errors", [])
    errors.append(cleanup)
    try:
        logger.error("Student compile cleanup failed; preserving the primary %s",
                     type(primary).__name__,
                     exc_info=(type(cleanup), cleanup, cleanup.__traceback__))
    except BaseException as reporting_error:
        errors.append(reporting_error)


class StudentCompileRegions:
    """Install original-bound-forward dispatch; roll back only replacements we own."""

    def __init__(self,
                 student: Any,
                 scope: str,
                 division_rounding: bool,
                 diagnostics: CompileDiagnostics | None,
                 recompile_limit: int | None,
                 *,
                 cudagraphs: bool,
                 cache: CheckpointCompileCache | None = None) -> None:
        self.calls: Counter = Counter()
        self.originals: list[tuple[Any, Any, Any, bool, Any]] = []
        self.count = 0
        self.restored_count = 0
        self.scope = scope
        self.cache = cache
        try:
            self._install(student, scope, division_rounding, diagnostics, recompile_limit, cudagraphs=cudagraphs)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                record_compile_cleanup_error(primary, cleanup)
            raise

    def _install(self, student: Any, scope: str, division_rounding: bool, diagnostics: CompileDiagnostics | None,
                 recompile_limit: int | None, *, cudagraphs: bool) -> None:
        validate_recompile_limit(recompile_limit)
        calls = self.calls
        targets = []
        for index, block in enumerate(student.transformer.transformer_blocks):
            inner = getattr(block, "_checkpoint_wrapped_module", None)
            if inner is None or not hasattr(inner, "ff"):
                raise ValueError("Student compilation requires existing full block checkpoint wrappers")
            if (getattr(inner, "_fastvideo_student_compile_scope", None) is not None
                    or getattr(inner, "_fastvideo_student_compile_owner", None) is not None
                    or hasattr(inner.forward, "_torchdynamo_orig_callable")):
                raise ValueError("Student FFN/block compile scopes are mutually exclusive; never nest compilation")
            fqn = f"transformer.transformer_blocks.{index}" + (".ff" if scope == "ffn" else "")
            targets.append((inner, inner.ff if scope == "ffn" else inner, fqn))
        self.count = len(targets)
        for inner, target, fqn in targets:
            eager = target.forward
            compile_kwargs: dict[str, Any] = dict(fullgraph=True,
                                                  dynamic=True,
                                                  options=compile_options(division_rounding, cudagraphs=cudagraphs))
            # Public per-call allowance, not an Inductor option or a global cache-policy change.
            # Omit when unset so existing callers retain the installed Torch default.
            if recompile_limit is not None:
                compile_kwargs["recompile_limit"] = recompile_limit
            compiled = torch.compile(eager, **compile_kwargs)
            if self.cache is not None:
                compiled = self.cache.wrap(eager, compiled, compile_kwargs, fqn)

            @wraps(eager)
            def forward(*args: Any, _eager=eager, _compiled=compiled, _fqn=fqn, **kwargs: Any) -> Any:
                grad = torch.is_grad_enabled()
                calls["grad_compiled_calls" if grad else "nograd_eager_calls"] += 1
                function = _compiled if grad else _eager
                with (_isolated_block_runtime_routes() if grad and scope == "block" else nullcontext(),
                      _dynamic_h3_metadata() if grad and scope == "block" else nullcontext()):
                    if diagnostics is not None:
                        return diagnostics.call(_fqn, function, args, kwargs)
                    return function(*args, **kwargs)

            self.originals.append((inner, target, eager, "forward" in vars(target), forward))
            target.forward = forward
            inner._fastvideo_student_compile_owner = self
            inner._fastvideo_student_compile_scope = scope
            if diagnostics is not None:
                diagnostics.event("student_region_compile_installed",
                                  fqn=fqn,
                                  scope=scope,
                                  compile_kwargs=compile_kwargs)

    def close(self) -> int:
        self.restored_count = 0
        pending = []
        failures = []
        for original in reversed(self.originals):
            inner, target, eager, had_forward, forward = original
            try:
                if target.forward is forward:
                    if had_forward:
                        target.forward = eager
                    else:
                        del target.forward
                    self.restored_count += 1
                if getattr(inner, "_fastvideo_student_compile_owner", None) is self:
                    # Keep ownership until both the forward and scope are restored.
                    if getattr(inner, "_fastvideo_student_compile_scope", None) == self.scope:
                        del inner._fastvideo_student_compile_scope
                    del inner._fastvideo_student_compile_owner
            except BaseException as cleanup:
                pending.append(original)
                failures.append(cleanup)
        self.originals = list(reversed(pending))
        if failures:
            primary = failures[0]
            # Constructor rollback may have no returned handle; retain failed ownership.
            vars(primary)["_fastvideo_student_compile_regions"] = self
            for secondary in failures[1:]:
                record_compile_cleanup_error(primary, secondary)
            raise primary
        return self.restored_count
