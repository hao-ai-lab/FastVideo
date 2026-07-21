# SPDX-License-Identifier: Apache-2.0
"""Exact one-layer Cosmos3 omni-forward fingerprint on the pinned L40S stack.

Routine CI loads one real decoder layer plus only the top-level weights needed
by fixed T2V, T2VS, action2world, and deepstack-reasoning forwards from the
pinned public checkpoint.  Set
``FASTVIDEO_SEED_COSMOS3_FINGERPRINT=1`` to print replacement hashes.  When
``COSMOS3_OFFICIAL_REPO`` points at the pinned NVIDIA ``cosmos-framework``
checkout, seed mode also proves every captured generation decoder-layer input
and output matches NVIDIA's framework MoT layer bit-for-bit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Iterator

from huggingface_hub import hf_hub_download
import pytest
from safetensors import safe_open
import torch
import torch.nn.functional as F

from fastvideo.configs.models.dits.cosmos3 import Cosmos3ArchConfig, Cosmos3VideoConfig
from fastvideo.models.dits.cosmos3 import (
    Cosmos3VFMTransformer,
    EntryClass,
    compute_mrope_position_ids_text,
    compute_mrope_position_ids_vision,
)
from fastvideo.models.loader.fsdp_load import set_default_dtype


MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
OFFICIAL_REVISION = "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
INDEX_FILE = "transformer/diffusion_pytorch_model.safetensors.index.json"

# Keep production hidden/head/MLP/latent/modality dimensions.  Only depth,
# vocabulary, and the action embodiment table's unused rows are reduced.
VOCAB_SIZE = 64
NUM_TEXT_TOKENS = 7
VISION_GRID = (2, 2, 2)  # temporal, patch-height, patch-width
VISION_LATENT_SHAPE = (48, 2, 4, 4)
ACTION_TOKEN_SHAPE = (2, )
ACTION_DOMAIN_ID = 3
FINGERPRINT_NUM_EMBODIMENT_DOMAINS = ACTION_DOMAIN_ID + 1
SOUND_TOKEN_SHAPE = (2, 1, 1)
REASON_NUM_TOKENS = 5
REASON_VISUAL_INDEXES = (1, 3)
INPUT_SEED = 20260721

# Golden values seeded on the exact Modal L40S image asserted below.
EXPECTED_ENVIRONMENT_SHA256 = "ad3d22373224999f0a0520c96eb406335ad5b58bdf4fd2f3e4ec29de2f6e1dda"
EXPECTED_CONTRACT_SHA256 = "5043ad8a281ddfd2595317c45943c2c06d71cf8012525db7efed1db7af7e31e3"
EXPECTED_INPUT_SHA256 = "4c2f1879c1847f65372fbb08df981bbb00e2e4b7ad6176c22db5515e69b2846f"
EXPECTED_WEIGHTS_SHA256 = "480b5a497bb0c9ef71edabeee62c057b2c54fce753362e8938d073d6bf7cfa62"
EXPECTED_OUTPUT_SHA256 = {
    "action2world": "4cf042b91e1a0c4131efbb454e23d679b73542abda9b36af18cd81bb694b1b34",
    "reason": "112b9ecd765b0e1afc7d21d169b4f3e941901d975c27f59df6062dedef60bc35",
    "t2v": "2b15c03d936bace07acea7aba55fe850dd3cadc344b361dce8da68acd94f9cb7",
    "t2vs": "1bffb714b7f316665e0176ddfe2b930df70d6dae1f75b3f761e92b5b9d8c26dd",
}
EXPECTED_OUTPUT_SHAPES = {
    "action2world": {
        "last_hidden_state": (17, 4096),
        "preds_action.0": (2, 64),
        "preds_vision.0": (1, 48, 2, 4, 4),
    },
    "reason": {
        "last_hidden_state": (5, 4096),
        "logits": (5, 64),
    },
    "t2v": {
        "last_hidden_state": (15, 4096),
        "preds_vision.0": (1, 48, 2, 4, 4),
    },
    "t2vs": {
        "last_hidden_state": (17, 4096),
        "preds_sound.0": (64, 2),
        "preds_vision.0": (1, 48, 2, 4, 4),
    },
}

SEED_MODE = os.environ.get("FASTVIDEO_SEED_COSMOS3_FINGERPRINT") == "1"


def _update_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu()
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(value.view(torch.uint8).numpy()))


def _tensor_hashes(tensors: dict[str, torch.Tensor], metadata: dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\0")
    for name in sorted(tensors):
        _update_hash(digest, name, tensors[name])
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "container_image": os.environ.get("FASTVIDEO_CONTAINER_IMAGE_REF", ""),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "pytorch": torch.__version__,
        "sdpa_backend": "MATH",
        "cudnn_sdpa": False,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


@contextmanager
def _fixed_math_profile() -> Iterator[None]:
    old_cudnn_sdpa = torch.backends.cuda.cudnn_sdp_enabled()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Cosmos3 fingerprint requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.enable_cudnn_sdp(old_cudnn_sdpa)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_matmul_precision)
        torch.use_deterministic_algorithms(old_deterministic)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _contract() -> tuple[dict[str, Any], str]:
    names = (
        "layers.0",
        "model.layers.35",
        "layers",
        "layers.0.self_attn",
        "transformer_blocks.0",
        "action_proj_in",
    )

    def matches(conditions) -> dict[str, bool]:
        return {name: any(condition(name, None) for condition in conditions) for name in names}

    expected_matches = {
        "layers.0": True,
        "model.layers.35": True,
        "layers": False,
        "layers.0.self_attn": False,
        "transformer_blocks.0": False,
        "action_proj_in": False,
    }
    fsdp_matches = matches(Cosmos3VFMTransformer._fsdp_shard_conditions)
    compile_matches = matches(Cosmos3VFMTransformer._compile_conditions)
    assert fsdp_matches == expected_matches
    assert compile_matches == expected_matches
    contract = {
        "entry_class": _jsonable(EntryClass),
        "fingerprint_config": _jsonable(Cosmos3VideoConfig(arch_config=_arch())),
        "production_config": _jsonable(Cosmos3VideoConfig()),
        "model_class": {
            "compile_conditions": _jsonable(Cosmos3VFMTransformer._compile_conditions),
            "compile_matches": compile_matches,
            "fsdp_shard_conditions": _jsonable(Cosmos3VFMTransformer._fsdp_shard_conditions),
            "fsdp_shard_matches": fsdp_matches,
            "param_names_mapping": _jsonable(Cosmos3VFMTransformer.param_names_mapping),
            "reverse_param_names_mapping": _jsonable(Cosmos3VFMTransformer.reverse_param_names_mapping),
        },
    }
    sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return contract, sha256


def _arch() -> Cosmos3ArchConfig:
    return Cosmos3ArchConfig(
        num_hidden_layers=1,
        vocab_size=VOCAB_SIZE,
        num_embodiment_domains=FINGERPRINT_NUM_EMBODIMENT_DOMAINS,
    )


def _new_meta_model() -> Cosmos3VFMTransformer:
    with set_default_dtype(torch.bfloat16), torch.device("meta"):
        return Cosmos3VFMTransformer(Cosmos3VideoConfig(arch_config=_arch()), hf_config={})


def _load_checkpoint_state(expected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    index_path = Path(
        hf_hub_download(repo_id=MODEL_ID, revision=MODEL_REVISION, filename=INDEX_FILE)
    )
    weight_map = json.loads(index_path.read_text())["weight_map"]
    missing = sorted(set(expected) - set(weight_map))
    assert not missing, f"Pinned Cosmos3 checkpoint is missing fingerprint keys: {missing}"

    by_shard: dict[str, list[str]] = {}
    for name in expected:
        by_shard.setdefault(weight_map[name], []).append(name)

    state: dict[str, torch.Tensor] = {}
    for shard_name, names in sorted(by_shard.items()):
        shard_path = hf_hub_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            filename=f"transformer/{shard_name}",
        )
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in sorted(names):
                if name in {"embed_tokens.weight", "lm_head.weight"}:
                    state[name] = shard.get_slice(name)[:VOCAB_SIZE].contiguous()
                elif name.startswith(("action_proj_in.", "action_proj_out.")):
                    state[name] = shard.get_slice(name)[:FINGERPRINT_NUM_EMBODIMENT_DOMAINS].contiguous()
                else:
                    state[name] = shard.get_tensor(name)

    for name, expected_tensor in expected.items():
        actual = state[name]
        assert actual.dtype == torch.bfloat16, f"{name}: expected bf16 checkpoint weight, got {actual.dtype}"
        assert actual.shape == expected_tensor.shape, (
            f"{name}: checkpoint shape {tuple(actual.shape)} != model shape {tuple(expected_tensor.shape)}"
        )
    return state


def _load_native_model(device: torch.device) -> tuple[Cosmos3VFMTransformer, str, dict[str, torch.Tensor]]:
    model = _new_meta_model()
    expected = model.state_dict()
    state = _load_checkpoint_state(expected)
    weights_sha256 = _tensor_hashes(state, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "num_hidden_layers": 1,
        "num_embodiment_domains": FINGERPRINT_NUM_EMBODIMENT_DOMAINS,
        "vocab_size": VOCAB_SIZE,
    })
    model.load_state_dict(state, strict=True, assign=True)
    model.materialize_non_persistent_buffers(device)
    model = model.to(device).eval()
    return model, weights_sha256, state


def _video_case(case: str) -> dict[str, Any]:
    arch = _arch()
    grid_t, grid_h, grid_w = VISION_GRID
    num_vision_tokens = grid_t * grid_h * grid_w

    text_position_ids, offset = compute_mrope_position_ids_text(NUM_TEXT_TOKENS, temporal_offset=0)
    vision_temporal_offset = offset + arch.temporal_modality_margin
    vision_position_ids, _ = compute_mrope_position_ids_vision(
        grid_t,
        grid_h,
        grid_w,
        temporal_offset=vision_temporal_offset,
        fps=24.0,
        base_fps=arch.base_fps,
        temporal_compression_factor=arch.temporal_compression_factor,
        enable_fps_modulation=arch.enable_fps_modulation,
    )
    position_blocks = [text_position_ids, vision_position_ids]

    # Integer arithmetic plus one fp32 division is stable and avoids relying on
    # a particular torch RNG implementation for the fixed input bytes.
    latent_elements = 1
    for size in VISION_LATENT_SHAPE:
        latent_elements *= size
    vision = ((torch.arange(latent_elements, dtype=torch.float32) % 257) - 128).reshape(
        VISION_LATENT_SHAPE
    ) / 127.0
    text_ids = torch.tensor([1, 3, 7, 15, 31, 47, 63], dtype=torch.long)
    text_indexes = torch.arange(NUM_TEXT_TOKENS, dtype=torch.long)
    vision_indexes = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + num_vision_tokens, dtype=torch.long)
    vision_timesteps = torch.full((num_vision_tokens,), 500.0, dtype=torch.float32)
    noisy_frames = torch.arange(grid_t, dtype=torch.long)
    kwargs: dict[str, Any] = {
        "text_ids": text_ids,
        "text_indexes": text_indexes,
        "split_lens": [NUM_TEXT_TOKENS, num_vision_tokens],
        "attn_modes": ["causal", "full"],
        "vision_tokens": [vision],
        "vision_token_shapes": [VISION_GRID],
        "vision_sequence_indexes": vision_indexes,
        "vision_timesteps": vision_timesteps,
        "vision_mse_loss_indexes": vision_indexes,
        "vision_noisy_frame_indexes": [noisy_frames],
        "fps_vision": torch.tensor([24.0], dtype=torch.float32),
    }

    if case == "t2vs":
        sound_t = SOUND_TOKEN_SHAPE[0]
        sound = ((torch.arange(arch.sound_dim * sound_t, dtype=torch.float32) % 61) - 30).reshape(
            arch.sound_dim, sound_t
        ) / 31.0
        sound_indexes = torch.arange(
            NUM_TEXT_TOKENS + num_vision_tokens,
            NUM_TEXT_TOKENS + num_vision_tokens + sound_t,
            dtype=torch.long,
        )
        sound_position_ids, _ = compute_mrope_position_ids_vision(
            sound_t,
            1,
            1,
            temporal_offset=vision_temporal_offset,
            fps=arch.sound_latent_fps,
            base_fps=arch.base_fps,
            temporal_compression_factor=arch.temporal_compression_factor_sound,
            enable_fps_modulation=arch.enable_fps_modulation,
        )
        position_blocks.append(sound_position_ids)
        kwargs.update(
            sound_tokens=[sound],
            sound_token_shapes=[SOUND_TOKEN_SHAPE],
            sound_sequence_indexes=sound_indexes,
            sound_timesteps=torch.full((sound_t,), 500.0, dtype=torch.float32),
            sound_mse_loss_indexes=sound_indexes,
            sound_noisy_frame_indexes=[torch.arange(sound_t, dtype=torch.long)],
            fps_sound=torch.tensor([arch.sound_latent_fps], dtype=torch.float32),
        )
        kwargs["split_lens"][1] += sound_t
    elif case == "action2world":
        action_t = ACTION_TOKEN_SHAPE[0]
        action = ((torch.arange(action_t * arch.action_dim, dtype=torch.float32) % 71) - 35).reshape(
            action_t, arch.action_dim
        ) / 36.0
        action_indexes = torch.arange(
            NUM_TEXT_TOKENS + num_vision_tokens,
            NUM_TEXT_TOKENS + num_vision_tokens + action_t,
            dtype=torch.long,
        )
        action_position_ids, _ = compute_mrope_position_ids_vision(
            action_t,
            1,
            1,
            temporal_offset=vision_temporal_offset,
            fps=arch.base_fps,
            base_fps=arch.base_fps,
            temporal_compression_factor=1,
            base_temporal_compression_factor=arch.temporal_compression_factor,
            enable_fps_modulation=arch.enable_fps_modulation,
            start_frame_offset=1,
        )
        position_blocks.append(action_position_ids)
        kwargs.update(
            action_tokens=[action],
            action_token_shapes=[ACTION_TOKEN_SHAPE],
            action_sequence_indexes=action_indexes,
            action_timesteps=torch.full((action_t,), 500.0, dtype=torch.float32),
            action_mse_loss_indexes=action_indexes,
            action_noisy_frame_indexes=[torch.arange(action_t, dtype=torch.long)],
            action_domain_id=[torch.tensor([ACTION_DOMAIN_ID], dtype=torch.long)],
        )
        kwargs["split_lens"][1] += action_t
    elif case != "t2v":
        raise ValueError(f"Unknown Cosmos3 fingerprint case: {case}")

    kwargs["position_ids"] = torch.cat(position_blocks, dim=1)
    kwargs["sequence_length"] = sum(kwargs["split_lens"])
    return kwargs


def _reason_inputs() -> dict[str, Any]:
    hidden_size = _arch().hidden_size
    inputs_embeds = ((torch.arange(REASON_NUM_TOKENS * hidden_size, dtype=torch.float32) % 251) - 125).reshape(
        REASON_NUM_TOKENS, hidden_size
    ) / 127.0
    deepstack = ((torch.arange(len(REASON_VISUAL_INDEXES) * hidden_size, dtype=torch.float32) % 97) - 48).reshape(
        len(REASON_VISUAL_INDEXES), hidden_size
    ) / 49.0
    visual_pos_mask = torch.zeros(REASON_NUM_TOKENS, dtype=torch.bool)
    visual_pos_mask[list(REASON_VISUAL_INDEXES)] = True
    return {
        "inputs_embeds": inputs_embeds.to(torch.bfloat16),
        "position_ids": torch.tensor(
            [[0, 1, 2, 3, 4], [0, 1, 7, 3, 8], [0, 1, 9, 3, 10]],
            dtype=torch.long,
        ),
        "deepstack_embeds": [deepstack.to(torch.bfloat16)],
        "visual_pos_mask": visual_pos_mask,
    }


def _collect_tensors(name: str, value: Any, output: dict[str, torch.Tensor]) -> None:
    if torch.is_tensor(value):
        output[name] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_tensors(f"{name}.{key}", item, output)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_tensors(f"{name}.{index}", item, output)


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _inputs(device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    cpu_cases = {case: _video_case(case) for case in ("t2v", "t2vs", "action2world")}
    cpu_reason = _reason_inputs()
    tensors: dict[str, torch.Tensor] = {}
    _collect_tensors("cases", cpu_cases, tensors)
    _collect_tensors("reason", cpu_reason, tensors)
    metadata = {
        "action_domain_id": ACTION_DOMAIN_ID,
        "action_token_shapes": [list(ACTION_TOKEN_SHAPE)],
        "cases": {
            case: {
                "attn_modes": kwargs["attn_modes"],
                "sequence_length": kwargs["sequence_length"],
                "split_lens": kwargs["split_lens"],
            }
            for case, kwargs in cpu_cases.items()
        },
        "input_seed": INPUT_SEED,
        "reason_visual_indexes": list(REASON_VISUAL_INDEXES),
        "sound_token_shapes": [list(SOUND_TOKEN_SHAPE)],
        "vision_token_shapes": [list(VISION_GRID)],
    }
    input_sha256 = _tensor_hashes(tensors, metadata)
    return (
        {case: _to_device(kwargs, device) for case, kwargs in cpu_cases.items()},
        _to_device(cpu_reason, device),
        input_sha256,
    )


def _run_native(
    model: Cosmos3VFMTransformer,
    kwargs: dict[str, Any],
    *,
    capture_layer: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: dict[str, Any] = {}
    handles = []
    if capture_layer:
        def capture_input(_module, args):
            captured["inputs"] = tuple(value.detach().cpu() for value in args)

        def capture_output(_module, _args, output):
            captured["outputs"] = tuple(value.detach().cpu() for value in output)

        handles = [
            model.layers[0].register_forward_pre_hook(capture_input),
            model.layers[0].register_forward_hook(capture_output),
        ]
    try:
        with torch.inference_mode():
            output = model(**kwargs)
        return output, captured
    finally:
        for handle in handles:
            handle.remove()


def _official_layer_parity(
    repo: Path,
    layer_state: dict[str, torch.Tensor],
    captured_by_case: dict[str, dict[str, Any]],
    device: torch.device,
) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != OFFICIAL_REVISION:
        raise RuntimeError(
            f"Official Cosmos3 checkout must be {OFFICIAL_REVISION}, got {revision}"
        )
    if not (repo / "cosmos_framework").is_dir():
        raise RuntimeError(f"Missing official cosmos_framework package under {repo}")
    sys.path.insert(0, str(repo))
    try:
        from cosmos_framework.data.generator.sequence_packing.runtime import (
            from_und_gen_splits,
            get_gen_seq,
            get_und_seq,
            sequence_pack_from_packed_sequence,
        )
        from cosmos_framework.model.generator.mot.unified_mot import (
            LayerTypes,
            MoTDecoderLayer,
        )
    finally:
        sys.path.pop(0)

    arch = _arch()
    config = SimpleNamespace(
        hidden_size=arch.hidden_size,
        intermediate_size=arch.intermediate_size,
        num_attention_heads=arch.num_attention_heads,
        num_key_value_heads=arch.num_key_value_heads,
        head_dim=arch.head_dim,
        attention_bias=arch.attention_bias,
        attention_dropout=0.0,
        rms_norm_eps=arch.rms_norm_eps,
        hidden_act=arch.hidden_act,
    )
    with set_default_dtype(torch.bfloat16), torch.device("meta"):
        block = MoTDecoderLayer(
            config=config,
            layer_idx=0,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=True,
            qk_norm_for_diffusion=True,
        )

    native_to_official = {
        "self_attn.to_q.": "self_attn.q_proj.",
        "self_attn.to_k.": "self_attn.k_proj.",
        "self_attn.to_v.": "self_attn.v_proj.",
        "self_attn.to_out.": "self_attn.o_proj.",
        "self_attn.norm_q.": "self_attn.q_norm.",
        "self_attn.norm_k.": "self_attn.k_norm.",
        "self_attn.add_q_proj.": "self_attn.q_proj_moe_gen.",
        "self_attn.add_k_proj.": "self_attn.k_proj_moe_gen.",
        "self_attn.add_v_proj.": "self_attn.v_proj_moe_gen.",
        "self_attn.to_add_out.": "self_attn.o_proj_moe_gen.",
        "self_attn.norm_added_q.": "self_attn.q_norm_moe_gen.",
        "self_attn.norm_added_k.": "self_attn.k_norm_moe_gen.",
    }

    def remap(name: str) -> str:
        for source, target in native_to_official.items():
            if name.startswith(source):
                return target + name.removeprefix(source)
        return name

    official_state = {remap(name): tensor for name, tensor in layer_state.items()}
    block.load_state_dict(official_state, strict=True, assign=True)
    block = block.to(device).eval()

    def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, is_causal: bool) -> torch.Tensor:
        query = query.permute(1, 0, 2).unsqueeze(0)
        key = key.permute(1, 0, 2).unsqueeze(0)
        value = value.permute(1, 0, 2).unsqueeze(0)
        if key.shape[1] != query.shape[1]:
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            scale=query.shape[-1]**-0.5,
        )
        return output.squeeze(0).permute(1, 0, 2).flatten(1)

    def exact_dispatch(query_pack, key_pack, value_pack, _attention_mask, **_kwargs):
        query_und = get_und_seq(query_pack)
        key_und = get_und_seq(key_pack)
        value_und = get_und_seq(value_pack)
        query_gen = get_gen_seq(query_pack)
        key_gen = get_gen_seq(key_pack)
        value_gen = get_gen_seq(value_pack)
        und_output = attend(query_und, key_und, value_und, is_causal=True)
        gen_output = attend(
            query_gen,
            torch.cat([key_und, key_gen], dim=0),
            torch.cat([value_und, value_gen], dim=0),
            is_causal=False,
        )
        return from_und_gen_splits(und_output, gen_output, query_pack), None

    block.self_attn.dispatch_attention_fn = exact_dispatch
    for case, captured in sorted(captured_by_case.items()):
        und, gen, cos_und, sin_und, cos_gen, sin_gen = (
            value.to(device) for value in captured["inputs"]
        )
        packed = torch.cat([und, gen], dim=0)
        pack = sequence_pack_from_packed_sequence(
            packed_sequence=packed,
            attn_modes=["causal", "full"],
            split_lens=[und.shape[0], gen.shape[0]],
            sample_lens=[packed.shape[0]],
            packed_und_token_indexes=torch.arange(und.shape[0], device=device),
            packed_gen_token_indexes=torch.arange(und.shape[0], packed.shape[0], device=device),
        )
        rope = (
            from_und_gen_splits(cos_und, cos_gen, pack),
            from_und_gen_splits(sin_und, sin_gen, pack),
        )
        with torch.inference_mode():
            official_output, lbl_metadata, kv_to_store = block(pack, None, rope)
        assert not lbl_metadata, case
        assert kv_to_store is None, case
        expected_und, expected_gen = captured["outputs"]
        torch.testing.assert_close(get_und_seq(official_output).cpu(), expected_und, atol=0, rtol=0, msg=case)
        torch.testing.assert_close(get_gen_seq(official_output).cpu(), expected_gen, atol=0, rtol=0, msg=case)


def _flatten_outputs(output: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in output.items():
        if torch.is_tensor(value):
            tensors[name] = value
        elif isinstance(value, (list, tuple)):
            for index, tensor in enumerate(value):
                assert torch.is_tensor(tensor), f"{name}.{index} is not a tensor"
                tensors[f"{name}.{index}"] = tensor
        else:
            raise TypeError(f"Unsupported Cosmos3 fingerprint output {name}: {type(value).__name__}")
    return tensors


def _run_reason(model: Cosmos3VFMTransformer, kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
    with torch.inference_mode():
        hidden = model.reason_forward(**kwargs)
        logits = model.lm_head(hidden)
    return {"last_hidden_state": hidden, "logits": logits}


def test_cosmos3_transformer_contract_fingerprint() -> None:
    _contract_value, contract_sha256 = _contract()
    if not SEED_MODE:
        assert contract_sha256 == EXPECTED_CONTRACT_SHA256


def test_cosmos3_one_layer_omni_forward_fingerprint() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 exact fingerprint requires the pinned Modal L40S CUDA profile.")

    device = torch.device("cuda:0")
    with _fixed_math_profile():
        environment = _environment()
        environment_sha256 = hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        contract, contract_sha256 = _contract()
        model, weights_sha256, state = _load_native_model(device)
        case_inputs, reason_inputs, input_sha256 = _inputs(device)
        output_hashes: dict[str, str] = {}
        output_summaries: dict[str, Any] = {}
        captured_by_case: dict[str, dict[str, Any]] = {}

        for case, kwargs in case_inputs.items():
            first, captured = _run_native(model, kwargs, capture_layer=SEED_MODE)
            second, _ = _run_native(model, kwargs)
            first_tensors = _flatten_outputs(first)
            second_tensors = _flatten_outputs(second)
            assert first_tensors.keys() == second_tensors.keys()
            for name in first_tensors:
                torch.testing.assert_close(second_tensors[name], first_tensors[name], atol=0, rtol=0, msg=case)
            shapes = {name: tuple(tensor.shape) for name, tensor in first_tensors.items()}
            assert shapes == EXPECTED_OUTPUT_SHAPES[case]
            assert all(tensor.dtype == torch.bfloat16 for tensor in first_tensors.values())
            output_hashes[case] = _tensor_hashes(first_tensors)
            output_summaries[case] = {
                "sha256": output_hashes[case],
                "tensors": {
                    name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
                    for name, tensor in sorted(first_tensors.items())
                },
            }
            if SEED_MODE:
                captured_by_case[case] = captured

        first_reason = _run_reason(model, reason_inputs)
        second_reason = _run_reason(model, reason_inputs)
        for name in first_reason:
            torch.testing.assert_close(second_reason[name], first_reason[name], atol=0, rtol=0, msg="reason")
        reason_shapes = {name: tuple(tensor.shape) for name, tensor in first_reason.items()}
        assert reason_shapes == EXPECTED_OUTPUT_SHAPES["reason"]
        assert all(tensor.dtype == torch.bfloat16 for tensor in first_reason.values())
        output_hashes["reason"] = _tensor_hashes(first_reason)
        output_summaries["reason"] = {
            "sha256": output_hashes["reason"],
            "tensors": {
                name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
                for name, tensor in sorted(first_reason.items())
            },
        }

        official_repo = os.environ.get("COSMOS3_OFFICIAL_REPO")
        if SEED_MODE and not official_repo:
            raise RuntimeError(
                "Seed mode requires COSMOS3_OFFICIAL_REPO at the pinned NVIDIA cosmos-framework checkout."
            )
        if SEED_MODE:
            layer_state = {
                name.removeprefix("layers.0."): tensor
                for name, tensor in state.items()
                if name.startswith("layers.0.")
            }
            _official_layer_parity(Path(official_repo), layer_state, captured_by_case, device)

    seeded = {
        "contract": contract,
        "contract_sha256": contract_sha256,
        "environment": environment,
        "environment_sha256": environment_sha256,
        "input_sha256": input_sha256,
        "weights_sha256": weights_sha256,
        "outputs": output_summaries,
        "model_revision": MODEL_REVISION,
        "official_revision": OFFICIAL_REVISION,
        "official_layer_parity": sorted(captured_by_case) if official_repo else [],
    }
    if SEED_MODE:
        print("COSMOS3_FINGERPRINT_SEED=" + json.dumps(seeded, sort_keys=True))
        return

    assert environment_sha256 == EXPECTED_ENVIRONMENT_SHA256, seeded
    assert contract_sha256 == EXPECTED_CONTRACT_SHA256, seeded
    assert input_sha256 == EXPECTED_INPUT_SHA256, seeded
    assert weights_sha256 == EXPECTED_WEIGHTS_SHA256, seeded
    assert output_hashes == EXPECTED_OUTPUT_SHA256, seeded

    del model, state, case_inputs, reason_inputs, first_reason, second_reason
    gc.collect()
    torch.cuda.empty_cache()
