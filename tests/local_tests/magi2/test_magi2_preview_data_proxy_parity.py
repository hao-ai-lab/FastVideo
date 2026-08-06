# SPDX-License-Identifier: Apache-2.0
"""Strict MAGI-2 preview data-proxy implementation parity.

Coverage scope: implementation_subcomponent. The tests load the pinned official
``Magi2DataProxy`` source and compare it with the FastVideo-owned component on
deterministic text-to-video (T2V) and image-to-video (I2V) tensors.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import IntEnum
from functools import lru_cache
import importlib.util
import math
import os
from pathlib import Path
import sys
import types
import typing
from typing import Any

import pytest
import torch
from torch.testing import assert_close

import fastvideo.pipelines.basic.magi2.stages.preview_data_proxy as fastvideo_proxy_module
from fastvideo.pipelines.basic.magi2.stages.preview_data_proxy import (
    Magi2DataProxy,
    Magi2DataProxyConfig,
    ModelInput,
    Modality,
    VarlenHandler,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REPO_ROOT = Path(
    os.environ.get(
        "MAGI2_OFFICIAL_REF_DIR",
        REPO_ROOT.parent / "MAGI-2-preview",
    )
)
OFFICIAL_PROXY_PATH = (
    OFFICIAL_REPO_ROOT / "inference" / "pipeline" / "preview_data_proxy.py"
)
OFFICIAL_MODEL_PATH = (
    OFFICIAL_REPO_ROOT / "inference" / "model" / "magi2_preview.py"
)
PARITY_SCOPE = "implementation_subcomponent"


@dataclass(frozen=True)
class _ParallelStateStub:
    """Expose fixed parallel sizes and identifiable process-group handles."""

    cp_size: int = 1
    ep_size: int = 1

    def get_world_size(self, dimension: str = "") -> int:
        """Return the configured size for one parallel dimension."""
        if dimension == "cp":
            return self.cp_size
        if dimension == "ep":
            return self.ep_size
        return max(self.cp_size, self.ep_size)

    def get_parallel_group(self, dimension: str) -> str:
        """Return a stable process-group marker for distributed call assertions."""
        return f"{dimension}-group"


def _load_official_model_utility_module() -> types.ModuleType:
    """Load official coordinate, modality, and time-embedding definitions."""
    if not OFFICIAL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Official MAGI-2 model source is missing: {OFFICIAL_MODEL_PATH}"
        )
    source = OFFICIAL_MODEL_PATH.read_text(encoding="utf-8")
    source_module = ast.parse(source, filename=str(OFFICIAL_MODEL_PATH))
    symbol_names = {
        "Modality",
        "VarlenHandler",
        "get_coords",
        "sinusoidal_embedding_1d",
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
            f"Official model utility definitions differ: expected {symbol_names}, "
            f"found {found_names}"
        )

    module = types.ModuleType("inference.model.magi2_preview")
    module.__file__ = str(OFFICIAL_MODEL_PATH)
    module.__dict__.update(
        {
            "Coords": tuple[int, int, int],
            "IntEnum": IntEnum,
            "Optional": typing.Optional,
            "dataclass": dataclass,
            "math": math,
            "torch": torch,
        }
    )
    utility_module = ast.Module(body=symbol_nodes, type_ignores=[])
    module_name = module.__name__
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(
            compile(utility_module, str(OFFICIAL_MODEL_PATH), "exec"),
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
    """Import the official proxy while replacing unrelated model dependencies."""
    if not OFFICIAL_PROXY_PATH.is_file():
        raise FileNotFoundError(
            f"Official MAGI-2 data-proxy source is missing: {OFFICIAL_PROXY_PATH}"
        )
    if importlib.util.find_spec("unfoldNd") is None:
        pytest.skip(
            "Official MAGI-2 data-proxy parity requires unfoldNd==0.2.3."
        )

    model_stub = _load_official_model_utility_module()
    distributed_stub = types.ModuleType("inference.infra.distributed")
    distributed_stub.psm = _ParallelStateStub()
    replacement_modules = {
        "inference.model.magi2_preview": model_stub,
        "inference.infra.distributed": distributed_stub,
    }
    missing_module = object()
    previous_modules = {
        name: sys.modules.get(name, missing_module)
        for name in replacement_modules
    }
    sys.modules.update(replacement_modules)

    official_path_entry = str(OFFICIAL_REPO_ROOT)
    added_path = official_path_entry not in sys.path
    if added_path:
        sys.path.insert(0, official_path_entry)
    module_name = "magi2_official_preview_data_proxy"
    try:
        spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_PROXY_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot create an import spec for {OFFICIAL_PROXY_PATH}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if added_path:
            sys.path.remove(official_path_entry)
        for name, previous_module in previous_modules.items():
            if previous_module is missing_module:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
    return module


def _build_proxy_pair(
    official_module: types.ModuleType,
    config: Magi2DataProxyConfig,
) -> tuple[Any, Magi2DataProxy]:
    """Construct official and FastVideo proxies from identical config values."""
    official_config = official_module.DataProxyConfig(**asdict(config))
    return (
        official_module.Magi2DataProxy(official_config),
        Magi2DataProxy(config),
    )


def _build_model_input_pair(
    official_module: types.ModuleType,
    input_tensors: dict[str, Any],
) -> tuple[Any, ModelInput]:
    """Construct both input dataclasses over the same immutable test tensors."""
    return (
        official_module.ModelInput(**input_tensors),
        ModelInput(**input_tensors),
    )


def _assert_tensor_exact(
    fastvideo_tensor: torch.Tensor,
    official_tensor: torch.Tensor,
) -> None:
    """Require identical tensor metadata and values."""
    assert fastvideo_tensor.shape == official_tensor.shape
    assert fastvideo_tensor.dtype == official_tensor.dtype
    assert fastvideo_tensor.stride() == official_tensor.stride()
    assert_close(fastvideo_tensor, official_tensor, atol=0, rtol=0)


def _assert_packed_inputs_exact(
    fastvideo_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, VarlenHandler, torch.Tensor],
    official_output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor],
) -> None:
    """Compare all packed tensors and variable-length attention boundaries."""
    fastvideo_tokens, fastvideo_coords, fastvideo_modalities, fastvideo_varlen, fastvideo_time = fastvideo_output
    official_tokens, official_coords, official_modalities, official_varlen, official_time = official_output
    _assert_tensor_exact(fastvideo_tokens, official_tokens)
    _assert_tensor_exact(fastvideo_coords, official_coords)
    _assert_tensor_exact(fastvideo_modalities, official_modalities)
    _assert_tensor_exact(fastvideo_time, official_time)
    _assert_tensor_exact(fastvideo_varlen.cu_seqlens_q, official_varlen.cu_seqlens_q)
    _assert_tensor_exact(fastvideo_varlen.cu_seqlens_k, official_varlen.cu_seqlens_k)
    assert fastvideo_varlen.max_seqlen_q == official_varlen.max_seqlen_q
    assert fastvideo_varlen.max_seqlen_k == official_varlen.max_seqlen_k


def _make_t2v_input_tensors() -> dict[str, Any]:
    """Create two distinguishable CFG samples with video, audio, and text."""
    video = torch.arange(2 * 3 * 2 * 4 * 4, dtype=torch.float32).reshape(
        2, 3, 2, 4, 4
    )
    video[1].add_(1000)
    audio = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    audio[1].add_(2000)
    text = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    text[1].add_(3000)
    per_token_video_time = torch.linspace(
        0.05,
        0.85,
        steps=2 * 2 * 4 * 4,
        dtype=torch.float32,
    ).reshape(2, 1, 2, 4, 4)
    per_token_audio_time = torch.linspace(
        0.1,
        0.9,
        steps=2 * 5,
        dtype=torch.float32,
    ).reshape(2, 5, 1)
    return {
        "x_t": video,
        "audio_x_t": audio,
        "audio_feat_len": torch.tensor([5, 3], dtype=torch.int32),
        "txt_feat": text,
        "txt_feat_len": torch.tensor([4, 2], dtype=torch.int32),
        "t": torch.tensor([750, 750], dtype=torch.int64),
        "per_token_video_t": per_token_video_time,
        "per_token_audio_t": per_token_audio_time,
    }


def _make_i2v_input_tensors() -> dict[str, Any]:
    """Create two CFG samples with two conditioning images per sample."""
    generator = torch.Generator(device="cpu").manual_seed(314159)
    return {
        "x_t": torch.randn(
            (2, 3, 2, 4, 4),
            generator=generator,
            dtype=torch.float32,
        ),
        "audio_x_t": torch.randn(
            (2, 4, 4),
            generator=generator,
            dtype=torch.float32,
        ),
        "audio_feat_len": torch.tensor([4, 2], dtype=torch.int32),
        "txt_feat": torch.randn(
            (2, 3, 6),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16),
        "txt_feat_len": torch.tensor([3, 2], dtype=torch.int32),
        "t": torch.tensor([0.6, 0.6], dtype=torch.float32),
        "per_token_video_t": torch.full(
            (2, 1, 2, 4, 4),
            0.6,
            dtype=torch.float32,
        ),
        "per_token_audio_t": torch.full(
            (2, 4, 1),
            0.6,
            dtype=torch.float32,
        ),
        "ref_image_feat": torch.randn(
            (2, 2, 3, 1, 4, 4),
            generator=generator,
            dtype=torch.float32,
        ),
        "ref_image_feat_len": torch.tensor(
            [[[2, 2], [2, 2]], [[2, 2], [2, 2]]],
            dtype=torch.int32,
        ),
        "ref_image_special_token_embedding": torch.randn(
            (2, 2, 6),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16),
    }


def _assert_process_output_exact(
    official_proxy: Any,
    fastvideo_proxy: Magi2DataProxy,
    packed_token_count: int,
    output_channel: int,
) -> None:
    """Compare depacked video and audio after deterministic model-like output."""
    model_output = torch.arange(
        packed_token_count * output_channel,
        dtype=torch.float32,
    ).reshape(packed_token_count, output_channel)
    official_video, official_audio = official_proxy.process_output(model_output)
    fastvideo_video, fastvideo_audio = fastvideo_proxy.process_output(model_output)
    _assert_tensor_exact(fastvideo_video, official_video)
    _assert_tensor_exact(fastvideo_audio, official_audio)


def test_magi2_data_proxy_config_defaults_match_official() -> None:
    """Keep FastVideo proxy defaults aligned with the official configuration."""
    official_module = _load_official_proxy_module()
    official_defaults = official_module.DataProxyConfig().model_dump()

    assert asdict(Magi2DataProxyConfig()) == official_defaults


@pytest.mark.parametrize(
    ("parallel_dimension", "cp_size", "ep_size"),
    [("cp", 2, 1), ("ep", 1, 2)],
    ids=("context_parallel", "expert_parallel"),
)
def test_process_input_distributed_remote_max_padding_matches_official(
    monkeypatch: pytest.MonkeyPatch,
    parallel_dimension: str,
    cp_size: int,
    ep_size: int,
) -> None:
    """Match distributed maximum reduction and multiple-of-48 padding."""
    official_module = _load_official_proxy_module()
    parallel_state = _ParallelStateStub(cp_size=cp_size, ep_size=ep_size)
    monkeypatch.setattr(official_module, "psm", parallel_state)
    monkeypatch.setattr(fastvideo_proxy_module, "psm", parallel_state)
    reduced_groups: list[Any] = []

    def fake_all_reduce(
        tensor: torch.Tensor,
        op: Any = None,
        group: Any = None,
    ) -> None:
        """Simulate a remote rank whose packed sequence contains 49 tokens."""
        del op
        reduced_groups.append(group)
        tensor.fill_(49)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = Magi2DataProxyConfig(
        t_patch_size=2,
        patch_size=2,
        spatial_rope_interpolation="extra",
        add_time_token=False,
        time_channel_dim=1,
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    official_input, fastvideo_input = _build_model_input_pair(
        official_module,
        _make_t2v_input_tensors(),
    )

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, _, modalities, varlen_handler, time_features = fastvideo_output
    assert tokens.shape == (96, 24)
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 13, 22, 96]
    assert varlen_handler.cu_seqlens_k.tolist() == [0, 13, 22, 96]
    assert varlen_handler.max_seqlen_q == 74
    assert varlen_handler.max_seqlen_k == 74
    assert modalities[22:].tolist() == [int(Modality.TEXT)] * 74
    assert torch.count_nonzero(tokens[22:]) == 0
    assert torch.count_nonzero(time_features[22:]) == 0
    assert reduced_groups == [
        f"{parallel_dimension}-group",
        f"{parallel_dimension}-group",
    ]
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=96,
        output_channel=24,
    )


def test_process_input_t2v_cfg_and_cp_padding_match_official(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match T2V packing, CFG boundaries, time features, and CP padding."""
    official_module = _load_official_proxy_module()
    parallel_state = _ParallelStateStub(cp_size=2, ep_size=1)
    monkeypatch.setattr(official_module, "psm", parallel_state)
    monkeypatch.setattr(fastvideo_proxy_module, "psm", parallel_state)
    config = Magi2DataProxyConfig(
        t_patch_size=2,
        patch_size=2,
        spatial_rope_interpolation="extra",
        add_time_token=False,
        time_channel_dim=7,
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    official_input, fastvideo_input = _build_model_input_pair(
        official_module,
        _make_t2v_input_tensors(),
    )

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, _, modalities, varlen_handler, time_features = fastvideo_output
    assert tokens.shape == (48, 24)
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 13, 22, 48]
    assert varlen_handler.cu_seqlens_k.tolist() == [0, 13, 22, 48]
    expected_modalities = (
        [int(Modality.VIDEO)] * 4
        + [int(Modality.AUDIO)] * 5
        + [int(Modality.TEXT)] * 4
        + [int(Modality.VIDEO)] * 4
        + [int(Modality.AUDIO)] * 3
        + [int(Modality.TEXT)] * 2
        + [int(Modality.TEXT)] * 26
    )
    assert modalities.tolist() == expected_modalities
    assert not torch.equal(tokens[:4], tokens[13:17])
    assert_close(
        time_features[9:13],
        torch.tensor([1, 1, 1, 0, 0, 0, 0], dtype=torch.float32).expand(4, -1),
        atol=0,
        rtol=0,
    )
    assert torch.count_nonzero(time_features[22:]) == 0
    assert official_proxy._saved_data.keys() == fastvideo_proxy._saved_data.keys()
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=48,
        output_channel=24,
    )


def test_process_input_i2v_token_order_and_coordinates_match_official(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match I2V special-token order, image coordinates, times, and depacking."""
    official_module = _load_official_proxy_module()
    parallel_state = _ParallelStateStub(cp_size=1, ep_size=1)
    monkeypatch.setattr(official_module, "psm", parallel_state)
    monkeypatch.setattr(fastvideo_proxy_module, "psm", parallel_state)
    config = Magi2DataProxyConfig(
        t_patch_size=1,
        patch_size=2,
        spatial_rope_interpolation="extra",
        add_time_token=False,
        time_channel_dim=1,
    )
    official_proxy, fastvideo_proxy = _build_proxy_pair(official_module, config)
    input_tensors = _make_i2v_input_tensors()
    official_input, fastvideo_input = _build_model_input_pair(
        official_module,
        input_tensors,
    )

    official_output = official_proxy.process_input(official_input)
    fastvideo_output = fastvideo_proxy.process_input(fastvideo_input)
    _assert_packed_inputs_exact(fastvideo_output, official_output)

    tokens, coords, modalities, varlen_handler, time_features = fastvideo_output
    assert tokens.shape == (47, 12)
    assert varlen_handler.cu_seqlens_q.tolist() == [0, 25, 47]
    first_sample_modalities = (
        [int(Modality.VIDEO)] * 8
        + [int(Modality.AUDIO)] * 4
        + [int(Modality.TEXT)] * 3
        + [int(Modality.TEXT)]
        + [int(Modality.VIDEO)] * 4
        + [int(Modality.TEXT)]
        + [int(Modality.VIDEO)] * 4
    )
    second_sample_modalities = (
        [int(Modality.VIDEO)] * 8
        + [int(Modality.AUDIO)] * 2
        + [int(Modality.TEXT)] * 2
        + [int(Modality.TEXT)]
        + [int(Modality.VIDEO)] * 4
        + [int(Modality.TEXT)]
        + [int(Modality.VIDEO)] * 4
    )
    assert modalities.tolist() == first_sample_modalities + second_sample_modalities

    first_special_offset = 8 + 4 + 3
    first_special = input_tensors["ref_image_special_token_embedding"][0, 0]
    _assert_tensor_exact(
        tokens[first_special_offset, :6],
        first_special.to(tokens.dtype),
    )
    first_image_patch = input_tensors["ref_image_feat"][0, 0, :, :, :2, :2].flatten()
    _assert_tensor_exact(tokens[first_special_offset + 1], first_image_patch)
    assert coords[first_special_offset].tolist() == [4, -1, -1, 1, 2, 2, 1, 2, 2]
    assert coords[first_special_offset + 5].tolist() == [5, -1, -1, 1, 2, 2, 1, 2, 2]
    assert torch.count_nonzero(time_features[12:25]) == 0
    assert official_proxy._saved_data.keys() == fastvideo_proxy._saved_data.keys()
    _assert_process_output_exact(
        official_proxy,
        fastvideo_proxy,
        packed_token_count=47,
        output_channel=12,
    )
