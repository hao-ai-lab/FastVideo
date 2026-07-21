# SPDX-License-Identifier: Apache-2.0
"""Exact one-layer Cosmos3 video-forward fingerprint on the pinned L40S stack.

Routine CI loads one real decoder layer plus only the video-path top-level
weights from the pinned public checkpoint.  Set
``FASTVIDEO_SEED_COSMOS3_FINGERPRINT=1`` to print replacement hashes.  When
``COSMOS3_OFFICIAL_REPO`` points at the pinned NVIDIA ``cosmos-framework``
checkout, seed mode also proves the captured decoder-layer inputs and outputs
match NVIDIA's framework MoT layer bit-for-bit.
"""
from __future__ import annotations

from contextlib import contextmanager
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
    compute_mrope_position_ids_text,
    compute_mrope_position_ids_vision,
)
from fastvideo.models.loader.fsdp_load import set_default_dtype


MODEL_ID = "nvidia/Cosmos3-Nano"
MODEL_REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
OFFICIAL_REVISION = "ed8287fd7477113f8ac4f6b84290514d55cf0cdc"
INDEX_FILE = "transformer/diffusion_pytorch_model.safetensors.index.json"

# Keep production hidden/head/MLP/latent dimensions.  Only depth, vocabulary,
# and unused output modalities are reduced.
VOCAB_SIZE = 64
NUM_TEXT_TOKENS = 7
VISION_GRID = (2, 2, 2)  # temporal, patch-height, patch-width
VISION_LATENT_SHAPE = (48, 2, 4, 4)
INPUT_SEED = 20260721

# Seed these on the exact Modal L40S image before enabling routine CI.
EXPECTED_ENVIRONMENT_SHA256 = "TO_BE_SEEDED"
EXPECTED_INPUT_SHA256 = "TO_BE_SEEDED"
EXPECTED_WEIGHTS_SHA256 = "TO_BE_SEEDED"
EXPECTED_LAST_HIDDEN_STATE_SHA256 = "TO_BE_SEEDED"
EXPECTED_PREDS_VISION_SHA256 = "TO_BE_SEEDED"
EXPECTED_LAST_HIDDEN_STATE_SHAPE = (15, 4096)
EXPECTED_PREDS_VISION_SHAPE = (1, 48, 2, 4, 4)

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


def _arch() -> Cosmos3ArchConfig:
    return Cosmos3ArchConfig(
        num_hidden_layers=1,
        vocab_size=VOCAB_SIZE,
        action_gen=False,
        sound_gen=False,
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
        "vocab_size": VOCAB_SIZE,
    })
    model.load_state_dict(state, strict=True, assign=True)
    model.materialize_non_persistent_buffers(device)
    model = model.to(device).eval()
    return model, weights_sha256, state


def _inputs(device: torch.device) -> tuple[dict[str, Any], str]:
    arch = _arch()
    grid_t, grid_h, grid_w = VISION_GRID
    num_vision_tokens = grid_t * grid_h * grid_w
    sequence_length = NUM_TEXT_TOKENS + num_vision_tokens

    text_position_ids, offset = compute_mrope_position_ids_text(NUM_TEXT_TOKENS, temporal_offset=0)
    vision_position_ids, _ = compute_mrope_position_ids_vision(
        grid_t,
        grid_h,
        grid_w,
        temporal_offset=offset + arch.temporal_modality_margin,
        fps=24.0,
        base_fps=arch.base_fps,
        temporal_compression_factor=arch.temporal_compression_factor,
        enable_fps_modulation=arch.enable_fps_modulation,
    )
    position_ids = torch.cat([text_position_ids, vision_position_ids], dim=1)

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
    vision_indexes = torch.arange(NUM_TEXT_TOKENS, sequence_length, dtype=torch.long)
    vision_timesteps = torch.full((num_vision_tokens,), 500.0, dtype=torch.float32)
    noisy_frames = torch.arange(grid_t, dtype=torch.long)

    cpu_tensors = {
        "position_ids": position_ids,
        "text_ids": text_ids,
        "text_indexes": text_indexes,
        "vision": vision,
        "vision_indexes": vision_indexes,
        "vision_timesteps": vision_timesteps,
        "noisy_frames": noisy_frames,
    }
    metadata = {
        "input_seed": INPUT_SEED,
        "sequence_length": sequence_length,
        "split_lens": [NUM_TEXT_TOKENS, num_vision_tokens],
        "attn_modes": ["causal", "full"],
        "vision_token_shapes": [list(VISION_GRID)],
    }
    input_sha256 = _tensor_hashes(cpu_tensors, metadata)
    kwargs = {
        "text_ids": text_ids.to(device),
        "text_indexes": text_indexes.to(device),
        "position_ids": position_ids.to(device),
        "sequence_length": sequence_length,
        "split_lens": metadata["split_lens"],
        "attn_modes": metadata["attn_modes"],
        "vision_tokens": [vision.to(device)],
        "vision_token_shapes": [VISION_GRID],
        "vision_sequence_indexes": vision_indexes.to(device),
        "vision_timesteps": vision_timesteps.to(device),
        "vision_mse_loss_indexes": vision_indexes.to(device),
        "vision_noisy_frame_indexes": [noisy_frames.to(device)],
        "fps_vision": torch.tensor([24.0], dtype=torch.float32, device=device),
    }
    return kwargs, input_sha256


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
    captured: dict[str, Any],
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
    assert not lbl_metadata
    assert kv_to_store is None
    expected_und, expected_gen = captured["outputs"]
    torch.testing.assert_close(get_und_seq(official_output).cpu(), expected_und, atol=0, rtol=0)
    torch.testing.assert_close(get_gen_seq(official_output).cpu(), expected_gen, atol=0, rtol=0)


def test_cosmos3_one_layer_video_forward_fingerprint() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Cosmos3 exact fingerprint requires the pinned Modal L40S CUDA profile.")

    device = torch.device("cuda:0")
    with _fixed_math_profile():
        environment = _environment()
        environment_sha256 = hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        model, weights_sha256, state = _load_native_model(device)
        kwargs, input_sha256 = _inputs(device)
        first, captured = _run_native(model, kwargs, capture_layer=SEED_MODE)
        second, _ = _run_native(model, kwargs)

        first_hidden = first["last_hidden_state"].detach().cpu()
        first_vision = first["preds_vision"][0].detach().cpu()
        second_hidden = second["last_hidden_state"].detach().cpu()
        second_vision = second["preds_vision"][0].detach().cpu()
        torch.testing.assert_close(second_hidden, first_hidden, atol=0, rtol=0)
        torch.testing.assert_close(second_vision, first_vision, atol=0, rtol=0)

        hidden_sha256 = _tensor_hashes({"last_hidden_state": first_hidden})
        vision_sha256 = _tensor_hashes({"preds_vision": first_vision})
        assert first_hidden.dtype == torch.bfloat16
        assert first_vision.dtype == torch.bfloat16
        assert tuple(first_hidden.shape) == EXPECTED_LAST_HIDDEN_STATE_SHAPE
        assert tuple(first_vision.shape) == EXPECTED_PREDS_VISION_SHAPE

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
            _official_layer_parity(Path(official_repo), layer_state, captured, device)

    seeded = {
        "environment": environment,
        "environment_sha256": environment_sha256,
        "input_sha256": input_sha256,
        "weights_sha256": weights_sha256,
        "last_hidden_state": {
            "dtype": str(first_hidden.dtype),
            "shape": list(first_hidden.shape),
            "sha256": hidden_sha256,
        },
        "preds_vision": {
            "dtype": str(first_vision.dtype),
            "shape": list(first_vision.shape),
            "sha256": vision_sha256,
        },
        "model_revision": MODEL_REVISION,
        "official_revision": OFFICIAL_REVISION,
        "official_layer_parity": bool(official_repo),
    }
    if SEED_MODE:
        print("COSMOS3_FINGERPRINT_SEED=" + json.dumps(seeded, sort_keys=True))
        return

    assert environment_sha256 == EXPECTED_ENVIRONMENT_SHA256, seeded
    assert input_sha256 == EXPECTED_INPUT_SHA256, seeded
    assert weights_sha256 == EXPECTED_WEIGHTS_SHA256, seeded
    assert hidden_sha256 == EXPECTED_LAST_HIDDEN_STATE_SHA256, seeded
    assert vision_sha256 == EXPECTED_PREDS_VISION_SHA256, seeded

    del model, state, first, second
    gc.collect()
    torch.cuda.empty_cache()
