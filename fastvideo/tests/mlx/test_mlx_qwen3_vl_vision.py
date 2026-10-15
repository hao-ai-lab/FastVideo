# SPDX-License-Identifier: Apache-2.0
"""Parity gate: MLX Qwen3-VL vision tower + MRoPE ids vs the torch reference.

The H3 Ref2VA track conditions on images and sampled reference-video frames,
so the MLX runtime needs the Qwen3-VL visual half and the mixed
text/image/video MRoPE index builder that the CUDA path uses. Tiny random-weight
models run in ``transformers`` and in
``fastvideo.mlx_runtime.minimax_h3_vision`` with identical weights; the merged
tokens, the deepstack blocks, the rope indices and the HF processor geometry
must all match within pinned tolerances.

Runs anywhere MLX is installed:

    pytest fastvideo/tests/mlx/test_mlx_qwen3_vl_vision.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for the Qwen3-VL vision parity test")
torch = pytest.importorskip("torch", reason="PyTorch supplies the independent vision reference")
safetensors_numpy = pytest.importorskip("safetensors.numpy", reason="Writing the tiny vision checkpoint")

from transformers.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionModel  # noqa: E402

from fastvideo.mlx_runtime.minimax_h3_conditioner import (  # noqa: E402
    MLX_NVFP4_ENCODER_MANIFEST,
    ConditionerConfig,
    StreamedMiniMaxH3TextConditioner,
)
from fastvideo.mlx_runtime.minimax_h3_vision import (  # noqa: E402
    MLXQwen3VLVisionModel,
    Qwen3VLVisionConfig as MLXVisionConfig,
    prepare_vision_inputs,
    qwen3_vl_rope_index,
)

SEED = 991
# fp32 vision forward with erf-geLU and softmax attention; the tiny grid keeps
# the accumulation short, so this comfortably binds.
FP32_ATOL = 2e-5
FP32_RTOL = 2e-5

TINY = dict(
    hidden_size=32,
    depth=4,
    num_heads=4,
    intermediate_size=48,
    in_channels=3,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    out_hidden_size=16,
    num_position_embeddings=64,  # side 8, keeps grids on the position table
    deepstack_visual_indexes=(1, 3),
    hidden_act="gelu_pytorch_tanh",
)

IMAGE_TOKEN = 151655
VIDEO_TOKEN = 151656
VISION_START = 151652


def _torch_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(**TINY)


def _mlx_config() -> MLXVisionConfig:
    return MLXVisionConfig(**TINY)


def _seeded_torch_model() -> Qwen3VLVisionModel:
    torch.manual_seed(SEED)
    model = Qwen3VLVisionModel(_torch_config())
    model.eval()
    return model


def _write_checkpoint(directory, state_dict) -> None:
    """Persist the torch vision tower as ``model.visual.*`` fp32 safetensors."""
    (directory / "config.json").write_text(json.dumps({"vision_config": TINY}))
    tensors = {f"model.visual.{key}": value.detach().cpu().numpy().astype(np.float32)
               for key, value in state_dict.items()}
    safetensors_numpy.save_file(tensors, str(directory / "model.safetensors"))


def _seeded_inputs() -> tuple[np.ndarray, np.ndarray]:
    grid_thw = np.asarray([[1, 4, 6], [2, 4, 6]], dtype=np.int64)
    width = TINY["in_channels"] * TINY["temporal_patch_size"] * TINY["patch_size"]**2
    rows = int(np.prod(grid_thw, axis=1).sum())
    generator = np.random.default_rng(SEED)
    pixels = generator.standard_normal((rows, width), dtype=np.float32)
    return pixels, grid_thw


def test_encode_matches_transformers_reference(tmp_path) -> None:
    """Merged tokens and every deepstack block must match the HF vision tower."""
    model = _seeded_torch_model()
    _write_checkpoint(tmp_path, model.state_dict())
    pixels, grid_thw = _seeded_inputs()

    with torch.no_grad():
        reference = model(torch.from_numpy(pixels), torch.from_numpy(grid_thw))
    reference_merged = reference.pooler_output.numpy()
    reference_deepstack = [block.numpy() for block in reference.deepstack_features]
    assert reference_deepstack, "the reference must emit deepstack features for this index set"

    mlx_model = MLXQwen3VLVisionModel.from_safetensors_dir(tmp_path)
    merged, deepstack = mlx_model.encode(pixels, grid_thw)

    assert merged.shape == reference_merged.shape
    assert len(deepstack) == len(reference_deepstack)
    np.testing.assert_allclose(merged, reference_merged, atol=FP32_ATOL, rtol=FP32_RTOL)
    for index, (actual, expected) in enumerate(zip(deepstack, reference_deepstack, strict=True)):
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL,
                                   err_msg=f"deepstack block {index} diverged")


def test_rope_index_matches_reference_on_mixed_sequence() -> None:
    """Mixed text+image+video position ids must equal the torch builder's."""
    from types import SimpleNamespace

    from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner

    tokens = ([1, 2] + [VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN] + [3] +
              [VISION_START, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN,
               VIDEO_TOKEN, VIDEO_TOKEN] + [4, 5])
    input_ids = np.asarray(tokens, dtype=np.int64)
    image_grid_thw = np.asarray([[1, 2, 2]], dtype=np.int64)
    video_grid_thw = np.asarray([[2, 2, 2]], dtype=np.int64)

    config = SimpleNamespace(vision_spatial_merge_size=2, vision_start_token_id=VISION_START,
                             image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN)
    reference = MiniMaxH3Qwen3VLConditioner._get_rope_index(
        SimpleNamespace(config=config),
        torch.from_numpy(input_ids)[None],
        torch.from_numpy(image_grid_thw),
        torch.from_numpy(video_grid_thw),
        None,
    )[:, 0].numpy()

    actual = qwen3_vl_rope_index(input_ids,
                                 image_grid_thw,
                                 video_grid_thw,
                                 vision_start_token_id=VISION_START,
                                 image_token_id=IMAGE_TOKEN,
                                 video_token_id=VIDEO_TOKEN,
                                 spatial_merge_size=2)
    np.testing.assert_array_equal(actual, reference)


def test_prepare_vision_inputs_matches_hf_video_processor() -> None:
    """The numpy processor must reproduce HF geometry and pixel values."""
    processor_cls = pytest.importorskip("transformers.models.qwen3_vl").Qwen3VLVideoProcessor
    processor = processor_cls()
    # Wide bounds keep smart-resize in the round-only regime, where the image
    # and video processors agree, so the comparison isolates the pixel path.
    min_pixels, max_pixels = 16 * 16 * 4, 32 * 32 * 512
    generator = np.random.default_rng(7)

    image = generator.integers(0, 256, (45, 61, 3), dtype=np.uint8)
    video = generator.integers(0, 256, (4, 41, 37, 3), dtype=np.uint8)

    pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = prepare_vision_inputs(
        [image], [video], config=_mlx_config(), min_pixels=min_pixels, max_pixels=max_pixels)

    reference_image = processor(videos=[image[None]], do_sample_frames=False, return_tensors="np", size={
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels
    })
    reference_video = processor(videos=[video],
                                do_sample_frames=False,
                                return_tensors="np",
                                size={
                                    "shortest_edge": min_pixels,
                                    "longest_edge": max_pixels
                                })

    np.testing.assert_array_equal(image_grid_thw, reference_image["video_grid_thw"])
    np.testing.assert_array_equal(video_grid_thw, reference_video["video_grid_thw"])
    # Resize residual is at most one uint8 level, i.e. 2/255 after normalisation.
    level = 2.0 / 255.0 + 1e-6
    np.testing.assert_allclose(pixel_values, reference_image["pixel_values_videos"].astype(np.float32),
                               atol=level)
    np.testing.assert_allclose(pixel_values_videos, reference_video["pixel_values_videos"].astype(np.float32),
                               atol=level)


# --- encode_tokens_with_vision integration ---------------------------------

VISION_TINY = dict(TINY, hidden_size=8, num_heads=2, intermediate_size=8, out_hidden_size=8, depth=2)

TEXT_HIDDEN = 8
NUM_TEXT_LAYERS = 50  # TEXT_ENCODER_LAYER


class _FakeIndex:
    """Minimal conditioner weight map: an embedding table plus decoder layers."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights

    def get_row(self, key: str, row: int) -> np.ndarray:
        return self.weights[key][row]

    def get_mlx(self, key: str):
        import mlx.core as mx
        return mx.array(self.weights[key].astype(np.float32))

    def close(self) -> None:
        pass


def _build_fake_conditioner(tmp_path, input_dim: int = 32) -> StreamedMiniMaxH3TextConditioner:
    rng = np.random.default_rng(SEED)
    cfg = ConditionerConfig(hidden_size=TEXT_HIDDEN, num_layers=NUM_TEXT_LAYERS, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=4, intermediate_size=8, rms_norm_eps=1e-6,
                            mrope_section=(1, 1, 0), vocab_size=32, vision_start_token_id=10,
                            image_token_id=11, video_token_id=12, vision_spatial_merge_size=2)
    weights: dict[str, np.ndarray] = {
        "model.language_model.embed_tokens.weight": (rng.standard_normal((cfg.vocab_size, TEXT_HIDDEN)) * 0.1)
    }
    for layer in range(NUM_TEXT_LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = np.ones(TEXT_HIDDEN, dtype=np.float32)
        weights[prefix + "post_attention_layernorm.weight"] = np.ones(TEXT_HIDDEN, dtype=np.float32)
        weights[prefix + "self_attn.q_norm.weight"] = np.ones(4, dtype=np.float32)
        weights[prefix + "self_attn.k_norm.weight"] = np.ones(4, dtype=np.float32)
        for name, shape in (("q_proj", (8, TEXT_HIDDEN)), ("k_proj", (8, TEXT_HIDDEN)), ("v_proj",
                                                                                         (8, TEXT_HIDDEN)),
                            ("o_proj", (TEXT_HIDDEN, 8))):
            weights[prefix + f"self_attn.{name}.weight"] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
        for name, shape in (("gate_proj", (8, TEXT_HIDDEN)), ("up_proj", (8, TEXT_HIDDEN)),
                            ("down_proj", (TEXT_HIDDEN, 8))):
            weights[prefix + f"mlp.{name}.weight"] = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    del input_dim

    conditioner = StreamedMiniMaxH3TextConditioner.__new__(StreamedMiniMaxH3TextConditioner)
    conditioner.component_dir = tmp_path
    conditioner.config = cfg
    conditioner.index = _FakeIndex(weights)
    conditioner._vision = None
    return conditioner


def test_encode_tokens_with_vision_splices_and_keeps_text_path(tmp_path) -> None:
    """The Ref2VA encode must splice vision tokens, add deepstack, and match text."""
    # A tiny vision tower whose out_hidden_size equals the language hidden size.
    (tmp_path / "config.json").write_text(json.dumps({"vision_config": VISION_TINY}))
    torch.manual_seed(SEED)
    vision = Qwen3VLVisionModel(Qwen3VLVisionConfig(**VISION_TINY))
    vision.eval()
    tensors = {f"model.visual.{key}": value.detach().cpu().numpy().astype(np.float32)
               for key, value in vision.state_dict().items()}
    safetensors_numpy.save_file(tensors, str(tmp_path / "model.safetensors"))

    conditioner = _build_fake_conditioner(tmp_path)
    grid_thw = np.asarray([[1, 4, 4]], dtype=np.int64)
    image_tokens = int(np.prod(grid_thw, axis=1).sum()) // VISION_TINY["spatial_merge_size"]**2
    width = VISION_TINY["in_channels"] * VISION_TINY["temporal_patch_size"] * VISION_TINY["patch_size"]**2
    generator = np.random.default_rng(SEED)
    pixels = generator.standard_normal((int(np.prod(grid_thw, axis=1).sum()), width), dtype=np.float32)

    tokens = [1, 2, 10] + [11] * image_tokens + [3]
    tags = np.array([0, 0, 0] + [1] * image_tokens + [0], dtype=np.int64)

    hidden, out_tags = conditioner.encode_tokens_with_vision(tokens, tags,
                                                             pixel_values=pixels,
                                                             image_grid_thw=grid_thw)
    assert hidden.shape == (len(tokens), TEXT_HIDDEN)
    assert hidden.dtype == np.float32
    np.testing.assert_array_equal(out_tags, tags)

    # Text-only through the vision entry point is the same stack as encode_tokens.
    text_only, _ = conditioner.encode_tokens_with_vision(tokens, tags)
    baseline, _ = conditioner.encode_tokens(tokens)
    np.testing.assert_array_equal(text_only, baseline)

    # Deepstack must actually move the visual rows relative to the text inputs.
    assert not np.allclose(hidden[[3, 4, 5, 6]], text_only[[3, 4, 5, 6]])

    # Determinism: a second run reproduces the first.
    repeat, _ = conditioner.encode_tokens_with_vision(tokens, tags,
                                                      pixel_values=pixels,
                                                      image_grid_thw=grid_thw)
    np.testing.assert_array_equal(repeat, hidden)


def test_rotary_positions_are_frame_major() -> None:
    """A two-patch-frame grid repeats the whole spatial grid, not each coordinate.

    ``np.repeat`` instead runs each coordinate's frames together, which leaves
    the end-to-end result only ~1e-5 off - inside the parity tolerance - so the
    ordering is asserted structurally here.
    """
    mlx_model = MLXQwen3VLVisionModel(_mlx_config())
    grid_thw = np.asarray([[2, 4, 6]], dtype=np.int64)
    positions = np.asarray(mlx_model._rotary_positions(grid_thw))

    per_frame = (4 // TINY["spatial_merge_size"]) * (6 // TINY["spatial_merge_size"]) * TINY["spatial_merge_size"]**2
    assert positions.shape[0] == 2 * per_frame
    np.testing.assert_array_equal(positions[:per_frame], positions[per_frame:])


def test_prepare_vision_inputs_counts_padded_video_frames() -> None:
    """An odd frame count pads up to a whole temporal patch, and the grid says so.

    The placeholder count is derived from ``video_grid_thw``; reporting the
    unpadded count makes a real 124-frame reference (11 samples, 6 padded
    blocks) label six blocks and declare five.
    """
    generator = np.random.default_rng(11)
    video = generator.integers(0, 256, (5, 41, 37, 3), dtype=np.uint8)

    _, _, pixel_values_videos, video_grid_thw = prepare_vision_inputs(
        None, [video], config=_mlx_config(), min_pixels=16 * 16 * 4, max_pixels=32 * 32 * 512)

    temporal_patch = TINY["temporal_patch_size"]
    assert video_grid_thw.shape == (1, 3)
    assert video_grid_thw[0][0] == 6 // temporal_patch == 3
    assert pixel_values_videos.shape[0] == int(np.prod(video_grid_thw[0]))

    processor = pytest.importorskip("transformers.models.qwen3_vl").Qwen3VLVideoProcessor
    reference = processor()(videos=[video],
                           do_sample_frames=False,
                           return_tensors="np",
                           size={
                               "shortest_edge": 16 * 16 * 4,
                               "longest_edge": 32 * 32 * 512
                           })
    np.testing.assert_array_equal(video_grid_thw, reference["video_grid_thw"])


def test_vision_source_dir_prefers_the_checkpoint_over_an_encoder_cache(tmp_path) -> None:
    """A native MLX encoder cache keeps no biases, so the tower reads the source."""
    source = tmp_path / "checkpoint"
    source.mkdir()
    conditioner = StreamedMiniMaxH3TextConditioner.__new__(StreamedMiniMaxH3TextConditioner)
    conditioner.component_dir = tmp_path / "cache"
    conditioner.component_dir.mkdir()
    assert conditioner._vision_source_dir() == conditioner.component_dir

    manifest = conditioner.component_dir / MLX_NVFP4_ENCODER_MANIFEST
    manifest.write_text(json.dumps({"source_dir": str(source)}))
    assert conditioner._vision_source_dir() == source

    manifest.write_text(json.dumps({"source_dir": str(tmp_path / "gone")}))
    with pytest.raises(FileNotFoundError, match="is gone"):
        conditioner._vision_source_dir()
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="records no source_dir"):
        conditioner._vision_source_dir()
