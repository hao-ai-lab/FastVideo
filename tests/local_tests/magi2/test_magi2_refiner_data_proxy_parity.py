# SPDX-License-Identifier: Apache-2.0
"""Strict MAGI-2 refiner data-proxy implementation parity.

Coverage scope: implementation_subcomponent. The tests load the pinned official
``Magi2RefinerDataProxy`` source and compare every packed or depacked tensor
with the FastVideo-owned component. A scheduler check verifies that uneven
context-parallel splitting remains owned by ``UlyssesScheduler``.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import typing
from typing import Any

import pytest
import torch
from torch.testing import assert_close
from torch.utils._pytree import tree_map

from fastvideo.pipelines.basic.magi2.stages.refiner_data_proxy import (
    Magi2RefinerDataProxy,
    Magi2RefinerDataProxyConfig,
    Modality,
    RefinerModelInput,
    VarlenHandler,
    WindowLocalAttnHandler,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO_ROOT = Path(
    os.environ.get(
        "MAGI2_OFFICIAL_REF_DIR",
        REPO_ROOT.parent / "MAGI-2-preview",
    )
)
OFFICIAL_PROXY_PATH = OFFICIAL_REPO_ROOT / "inference" / "pipeline" / "refiner_data_proxy.py"
OFFICIAL_MODEL_PATH = OFFICIAL_REPO_ROOT / "inference" / "model" / "magi2_refiner.py"
OFFICIAL_SCHEDULER_PATH = (
    OFFICIAL_REPO_ROOT
    / "inference"
    / "infra"
    / "parallelism"
    / "context_parallel"
    / "ulysses_scheduler.py"
)
PARITY_SCOPE = "implementation_subcomponent"


@dataclass(frozen=True)
class _ParallelStateStub:
    """Expose one fixed context-parallel size and process-group marker."""

    cp_size: int

    def get_world_size(self, dimension: str = "") -> int:
        """Return the configured context-parallel world size."""
        return self.cp_size if dimension == "cp" else 1

    def get_parallel_group(self, dimension: str) -> str:
        """Return an identifiable process-group marker for call assertions."""
        return f"{dimension}-group"


def _load_official_refiner_model_utilities() -> types.ModuleType:
    """Load the official frame-local attention helper without model dependencies."""
    if not OFFICIAL_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Official MAGI-2 refiner source is missing: {OFFICIAL_MODEL_PATH}")
    source_module = ast.parse(
        OFFICIAL_MODEL_PATH.read_text(encoding="utf-8"),
        filename=str(OFFICIAL_MODEL_PATH),
    )
    symbol_names = {
        "FFAHandler",
        "calc_local_qk_range",
        "calc_local_attn_ffa_handler",
    }
    symbol_nodes = [
        node
        for node in source_module.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in symbol_names
    ]
    found_names = {node.name for node in symbol_nodes}
    if found_names != symbol_names:
        raise AssertionError(
            f"Official refiner utility definitions differ: expected {symbol_names}, found {found_names}"
        )

    module = types.ModuleType("inference.model.magi2_refiner")
    module.__file__ = str(OFFICIAL_MODEL_PATH)
    module.__dict__.update(
        {
            "Optional": typing.Optional,
            "dataclass": dataclass,
            "torch": torch,
        }
    )
    module_name = module.__name__
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(
            compile(
                ast.Module(body=symbol_nodes, type_ignores=[]),
                str(OFFICIAL_MODEL_PATH),
                "exec",
            ),
            module.__dict__,
        )
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    return module


@lru_cache(maxsize=1)
def _load_official_proxy_module() -> types.ModuleType:
    """Import the official proxy while isolating unrelated refiner model code."""
    if not OFFICIAL_PROXY_PATH.is_file():
        raise FileNotFoundError(f"Official MAGI-2 refiner proxy is missing: {OFFICIAL_PROXY_PATH}")
    if importlib.util.find_spec("unfoldNd") is None:
        pytest.skip("Official MAGI-2 refiner data-proxy parity requires unfoldNd==0.2.3.")

    model_module_name = "inference.model.magi2_refiner"
    missing_module = object()
    previous_model_module = sys.modules.get(model_module_name, missing_module)
    sys.modules[model_module_name] = _load_official_refiner_model_utilities()
    official_path_entry = str(OFFICIAL_REPO_ROOT)
    added_path = official_path_entry not in sys.path
    if added_path:
        sys.path.insert(0, official_path_entry)
    module_name = "magi2_official_refiner_data_proxy"
    try:
        spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_PROXY_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create an import spec for {OFFICIAL_PROXY_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if added_path:
            sys.path.remove(official_path_entry)
        if previous_model_module is missing_module:
            sys.modules.pop(model_module_name, None)
        else:
            sys.modules[model_module_name] = previous_model_module
    return module


def _load_official_scheduler_class() -> type[Any]:
    """Load the official Ulysses scheduler class with injectable collectives."""
    if not OFFICIAL_SCHEDULER_PATH.is_file():
        raise FileNotFoundError(f"Official Ulysses scheduler is missing: {OFFICIAL_SCHEDULER_PATH}")
    source_module = ast.parse(
        OFFICIAL_SCHEDULER_PATH.read_text(encoding="utf-8"),
        filename=str(OFFICIAL_SCHEDULER_PATH),
    )
    class_nodes = [
        node
        for node in source_module.body
        if isinstance(node, ast.ClassDef) and node.name == "UlyssesScheduler"
    ]
    if len(class_nodes) != 1:
        raise AssertionError("Official source must define exactly one UlyssesScheduler class")
    module = types.ModuleType("magi2_official_ulysses_scheduler")
    module.__dict__.update(
        {
            "Generic": typing.Generic,
            "List": list,
            "Optional": typing.Optional,
            "T": typing.TypeVar("T"),
            "torch": torch,
            "tree_map": tree_map,
        }
    )
    exec(
        compile(
            ast.Module(body=class_nodes, type_ignores=[]),
            str(OFFICIAL_SCHEDULER_PATH),
            "exec",
        ),
        module.__dict__,
    )
    return module.UlyssesScheduler


def _cuda_device() -> torch.device:
    """Return the CUDA device required by the pinned official proxy."""
    if not torch.cuda.is_available():
        pytest.skip("Official MAGI-2 refiner data-proxy parity requires CUDA.")
    return torch.device("cuda", torch.cuda.current_device())


def _build_proxy_pair(
    official_module: types.ModuleType,
    config: Magi2RefinerDataProxyConfig,
) -> tuple[Any, Magi2RefinerDataProxy]:
    """Construct official and FastVideo proxies from identical config values."""
    official_config = official_module.Magi2RefinerDataProxyConfig(**asdict(config))
    return (
        official_module.Magi2RefinerDataProxy(official_config),
        Magi2RefinerDataProxy(config),
    )


def _build_model_input_pair(
    input_tensors: dict[str, torch.Tensor],
) -> tuple[types.SimpleNamespace, RefinerModelInput]:
    """Construct both proxy inputs over the same immutable test tensors."""
    return (
        types.SimpleNamespace(**input_tensors),
        RefinerModelInput(**input_tensors),
    )


def _make_input_tensors(
    *,
    device: torch.device,
    batch_size: int,
    ref_video_token_limit: int,
    distinguish_cfg_samples: bool = False,
) -> dict[str, torch.Tensor]:
    """Create deterministic refiner video, audio, text, and reference features."""
    video = torch.arange(
        batch_size * 3 * 3 * 6 * 8,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 3, 3, 6, 8)
    audio = torch.arange(
        batch_size * 5 * 4,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 5, 4)
    text = torch.arange(
        batch_size * 4 * 6,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 4, 6)
    reference_audio = torch.arange(
        batch_size * 3 * 4,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 3, 4)
    reference_video = torch.arange(
        batch_size * 3 * 3 * 6 * 8,
        device=device,
        dtype=torch.float32,
    ).reshape(batch_size, 3, 3, 6, 8)
    if distinguish_cfg_samples and batch_size > 1:
        video[1].add_(1000)
        audio[1].add_(2000)
        text[1].add_(3000)
        reference_audio[1].add_(4000)
        reference_video[1].add_(5000)
    return {
        "x_t": video,
        "audio_x_t": audio,
        "audio_feat_len": torch.full((batch_size,), 4, dtype=torch.int64),
        "txt_feat": text,
        "txt_feat_len": torch.full((batch_size,), 3, dtype=torch.int64),
        "ref_audio_feat": reference_audio,
        "ref_audio_feat_len": torch.full((batch_size,), 2, dtype=torch.int64),
        "ref_video_feat": reference_video,
        "ref_video_feat_len": torch.full(
            (batch_size,),
            ref_video_token_limit,
            dtype=torch.int64,
        ),
    }


def _assert_tensor_exact(
    fastvideo_tensor: torch.Tensor,
    official_tensor: torch.Tensor,
) -> None:
    """Require identical tensor metadata and values."""
    assert fastvideo_tensor.shape == official_tensor.shape
    assert fastvideo_tensor.dtype == official_tensor.dtype
    assert fastvideo_tensor.stride() == official_tensor.stride()
    assert_close(fastvideo_tensor, official_tensor, atol=0, rtol=0)


def _assert_scalar_or_tensor_exact(
    fastvideo_value: int | torch.Tensor,
    official_value: int | torch.Tensor,
) -> None:
    """Compare attention-length metadata without discarding tensor dtype."""
    if isinstance(official_value, torch.Tensor):
        assert isinstance(fastvideo_value, torch.Tensor)
        _assert_tensor_exact(fastvideo_value, official_value)
    else:
        assert fastvideo_value == official_value


def _assert_local_attn_handler_exact(
    fastvideo_handler: WindowLocalAttnHandler | None,
    official_handler: Any,
) -> None:
    """Compare every attention-range field exposed by the official handler."""
    if official_handler is None:
        assert fastvideo_handler is None
        return
    assert fastvideo_handler is not None
    _assert_tensor_exact(fastvideo_handler.q_ranges, official_handler.q_ranges)
    _assert_tensor_exact(fastvideo_handler.k_ranges, official_handler.k_ranges)
    _assert_tensor_exact(fastvideo_handler.attn_type_map, official_handler.attn_type_map)
    assert fastvideo_handler.max_seqlen_q == official_handler.max_seqlen_q
    assert fastvideo_handler.max_seqlen_k == official_handler.max_seqlen_k
    assert fastvideo_handler.softmax_scale == official_handler.softmax_scale
    for field_name in (
        "bwd_q_ranges",
        "bwd_k_ranges",
        "bwd_attn_type_map",
    ):
        official_value = getattr(official_handler, field_name, None)
        fastvideo_value = getattr(fastvideo_handler, field_name)
        if official_value is None:
            assert fastvideo_value is None
        else:
            assert fastvideo_value is not None
            _assert_tensor_exact(fastvideo_value, official_value)
    assert fastvideo_handler.auto_range_merge == official_handler.auto_range_merge
    assert fastvideo_handler.sparse_load == official_handler.sparse_load


def _assert_packed_inputs_exact(
    fastvideo_output: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        VarlenHandler,
        WindowLocalAttnHandler | None,
    ],
    official_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any],
) -> None:
    """Compare all packed tensors and attention metadata."""
    (
        fastvideo_tokens,
        fastvideo_coords,
        fastvideo_modalities,
        fastvideo_varlen,
        fastvideo_local_attn,
    ) = fastvideo_output
    (
        official_tokens,
        official_coords,
        official_modalities,
        official_varlen,
        official_local_attn,
    ) = official_output
    _assert_tensor_exact(fastvideo_tokens, official_tokens)
    _assert_tensor_exact(fastvideo_coords, official_coords)
    _assert_tensor_exact(fastvideo_modalities, official_modalities)
    _assert_tensor_exact(fastvideo_varlen.cu_seqlens_q, official_varlen.cu_seqlens_q)
    _assert_tensor_exact(fastvideo_varlen.cu_seqlens_k, official_varlen.cu_seqlens_k)
    _assert_scalar_or_tensor_exact(fastvideo_varlen.max_seqlen_q, official_varlen.max_seqlen_q)
    _assert_scalar_or_tensor_exact(fastvideo_varlen.max_seqlen_k, official_varlen.max_seqlen_k)
    _assert_local_attn_handler_exact(fastvideo_local_attn, official_local_attn)


def _assert_process_output_exact(
    official_proxy: Any,
    fastvideo_proxy: Magi2RefinerDataProxy,
    packed_token_count: int,
    output_channel: int,
    device: torch.device,
) -> None:
    """Compare depacked video and audio after deterministic model-like output."""
    model_output = torch.arange(
        packed_token_count * output_channel,
        device=device,
        dtype=torch.float32,
    ).reshape(packed_token_count, output_channel)
    official_video, official_audio = official_proxy.process_output(model_output)
    fastvideo_video, fastvideo_audio = fastvideo_proxy.process_output(model_output)
    _assert_tensor_exact(fastvideo_video, official_video)
    _assert_tensor_exact(fastvideo_audio, official_audio)


def test_magi2_refiner_data_proxy_config_defaults_match_official() -> None:
    """Keep FastVideo refiner proxy defaults aligned with the official config."""
    official_module = _load_official_proxy_module()
    official_defaults = official_module.Magi2RefinerDataProxyConfig().model_dump()

    assert asdict(Magi2RefinerDataProxyConfig()) == official_defaults


def test_process_input_default_frame_local_attention_matches_official() -> None:
    """Match default patching, coordinates, varlen metadata, and frame ranges."""
    device = _cuda_device()
    official_module = _load_official_proxy_module()
    config = Magi2RefinerDataProxyConfig()
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    official_input, fastvideo_input = _build_model_input_pair(
        _make_input_tensors(
            device=device,
            batch_size=1,
            ref_video_token_limit=0,
        )
    )

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, _, modalities, varlen_handler, local_attn_handler = fastvideo_output
    assert tokens.shape == (45, 12)
    assert modalities.tolist() == (
        [int(Modality.VIDEO)] * 36
        + [int(Modality.AUDIO)] * 4
        + [int(Modality.TEXT)] * 3
        + [int(Modality.AUDIO)] * 2
    )
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 45]
    assert local_attn_handler is not None
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=tokens.shape[0],
        output_channel=tokens.shape[1],
        device=device,
    )


def test_process_input_cfg_boundaries_and_no_time_features_match_official() -> None:
    """Preserve adjacent CFG samples without adding timestep feature tokens."""
    device = _cuda_device()
    official_module = _load_official_proxy_module()
    config = Magi2RefinerDataProxyConfig(
        patch_size=2,
        frame_receptive_field=-1,
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    input_tensors = _make_input_tensors(
        device=device,
        batch_size=2,
        ref_video_token_limit=1,
        distinguish_cfg_samples=True,
    )
    official_input, fastvideo_input = _build_model_input_pair(input_tensors)

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, _, _, varlen_handler, local_attn_handler = fastvideo_output
    assert tokens.shape == (92, 12)
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 46, 92]
    assert local_attn_handler is None
    assert not torch.equal(tokens[:36], tokens[46:82])
    assert "t" not in {input_field.name for input_field in fields(RefinerModelInput)}
    assert len(fastvideo_output) == 5
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=tokens.shape[0],
        output_channel=tokens.shape[1],
        device=device,
    )


def test_process_input_shipping_block_window_and_depacking_match_official() -> None:
    """Match the release block-grid token order, ranges, and inverse depacking."""
    device = _cuda_device()
    official_module = _load_official_proxy_module()
    config = Magi2RefinerDataProxyConfig(
        t_patch_size=1,
        patch_size=1,
        frame_receptive_field=11,
        spatial_rope_interpolation="extra",
        coords_style="v1",
        text_offset=0,
        attn_config={
            "mode": "window",
            "block_t_size": 8,
            "block_size": 4,
            "window": {
                "level": "block",
                "block_mode": "grid",
                "block_t_radius": 2,
                "block_h_radius": 2,
                "block_w_radius": 2,
                "win_size": 384,
                "frame_receptive_field": -1,
                "auto_range_merge": True,
                "sparse_load": False,
                "full_attn_layers": [],
            },
        },
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    official_input, fastvideo_input = _build_model_input_pair(
        _make_input_tensors(
            device=device,
            batch_size=1,
            ref_video_token_limit=4,
        )
    )

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, coords, modalities, varlen_handler, local_attn_handler = fastvideo_output
    assert tokens.shape == (157, 6)
    assert coords.shape == (157, 9)
    assert modalities.shape == (157,)
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 157]
    assert local_attn_handler is not None
    assert local_attn_handler.auto_range_merge is True
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=tokens.shape[0],
        output_channel=tokens.shape[1],
        device=device,
    )


def test_process_input_distributed_context_keeps_unpadded_official_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep distributed reduction and uneven splitting outside the refiner proxy."""
    device = _cuda_device()
    official_module = _load_official_proxy_module()
    config = Magi2RefinerDataProxyConfig(
        patch_size=2,
        frame_receptive_field=-1,
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    official_input, fastvideo_input = _build_model_input_pair(
        _make_input_tensors(
            device=device,
            batch_size=1,
            ref_video_token_limit=1,
        )
    )

    def reject_all_reduce(*args: Any, **kwargs: Any) -> None:
        """Fail if preview-style distributed maximum padding enters this proxy."""
        del args, kwargs
        raise AssertionError("The refiner proxy must not perform all_reduce")

    monkeypatch.setattr(torch.distributed, "all_reduce", reject_all_reduce)
    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)
    packed_tokens = fastvideo_output[0]
    assert packed_tokens.shape[0] == 46
    assert packed_tokens.shape[0] % 48 != 0
    assert "pad_size" not in fastvideo_proxy._saved_data
    assert "pad_size" not in official_proxy._saved_data

    official_scheduler_class = _load_official_scheduler_class()
    target_scheduler_module = importlib.import_module(
        "fastvideo.models.dits.magi2_runtime.context_parallel.ulysses_scheduler"
    )
    parallel_state = _ParallelStateStub(cp_size=3)
    official_capture: list[tuple[list[int], Any]] = []
    target_capture: list[tuple[list[int], Any]] = []

    def official_scatter(
        tensor: torch.Tensor,
        split_sizes: list[int],
        group: Any,
    ) -> torch.Tensor:
        """Capture the official scheduler's uneven split without communication."""
        official_capture.append((split_sizes, group))
        return tensor

    def target_scatter(
        tensor: torch.Tensor,
        split_sizes: list[int],
        group: Any,
    ) -> torch.Tensor:
        """Capture the FastVideo scheduler's uneven split without communication."""
        target_capture.append((split_sizes, group))
        return tensor

    official_scheduler_class._dispatch.__globals__["psm"] = parallel_state
    official_scheduler_class._dispatch.__globals__["scatter_to_context_parallel_region"] = official_scatter
    monkeypatch.setattr(target_scheduler_module, "psm", parallel_state)
    monkeypatch.setattr(target_scheduler_module, "scatter_to_context_parallel_region", target_scatter)
    official_scheduler = official_scheduler_class()
    target_scheduler = target_scheduler_module.UlyssesScheduler()

    official_dispatched = official_scheduler.dispatch(packed_tokens)
    target_dispatched = target_scheduler.dispatch(packed_tokens)
    _assert_tensor_exact(target_dispatched, official_dispatched)
    assert official_scheduler.cp_split_sizes == [16, 15, 15]
    assert target_scheduler.cp_split_sizes == [16, 15, 15]
    assert official_capture == [([16, 15, 15], "cp-group")]
    assert target_capture == official_capture
