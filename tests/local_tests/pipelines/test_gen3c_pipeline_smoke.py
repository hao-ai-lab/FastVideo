# SPDX-License-Identifier: Apache-2.0
"""
GEN3C pipeline smoke test.

Verifies the Gen3CPipeline can be instantiated and its stages execute in the
correct order with matching input/output shapes.  This is a lightweight test
that uses small tensor dimensions and random weights so no real checkpoint is
required.

Usage:
    # Quick smoke test (no weights required, CPU or CUDA)
    pytest tests/local_tests/pipelines/test_gen3c_pipeline_smoke.py -v

    # Full pipeline smoke test with converted weights
    GEN3C_DIFFUSERS_PATH=./gen3c_fastvideo \
        pytest tests/local_tests/pipelines/test_gen3c_pipeline_smoke.py -v -k full
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.models.dits.gen3c import Gen3CArchConfig, Gen3CVideoConfig
from fastvideo.configs.pipelines.gen3c import Gen3CConfig, t5_large_postprocess_text
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.models.dits.gen3c import Gen3CTransformer3DModel
from fastvideo.pipelines.basic.gen3c.gen3c_pipeline import Gen3CCFGPolicyStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    t = tensor.float()
    print(
        f"[GEN3C SMOKE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={t.min().item():.6f} max={t.max().item():.6f} "
        f"mean={t.mean().item():.6f}"
    )


def _create_small_config(num_layers: int = 2) -> Gen3CVideoConfig:
    """Create a minimal GEN3C config for smoke testing."""
    config = Gen3CVideoConfig()
    config.arch_config = Gen3CArchConfig()
    config.arch_config.num_layers = num_layers
    config.arch_config.max_size = (16, 32, 32)
    config.arch_config.__post_init__()
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGen3CPostprocessText:
    """Validate the T5 postprocess helper used by Gen3CConfig."""

    def test_zeroes_beyond_sequence_length(self):
        """Tokens beyond the actual sequence length should be zeroed."""
        batch_size, seq_len, dim = 2, 8, 16
        hidden = torch.randn(batch_size, seq_len, dim)
        # First sample has length 3, second has length 6
        attn_mask = torch.zeros(batch_size, seq_len)
        attn_mask[0, :3] = 1
        attn_mask[1, :6] = 1

        outputs = BaseEncoderOutput(
            last_hidden_state=hidden,
            attention_mask=attn_mask,
        )
        result = t5_large_postprocess_text(outputs)

        assert result.shape == hidden.shape
        # Everything beyond the real lengths should be zero
        assert (result[0, 3:] == 0).all()
        assert (result[1, 6:] == 0).all()
        # Within the lengths, values should be preserved
        assert torch.equal(result[0, :3], hidden[0, :3])
        assert torch.equal(result[1, :6], hidden[1, :6])

    def test_nan_replacement(self):
        """NaN values should be replaced with 0."""
        hidden = torch.randn(1, 4, 8)
        hidden[0, 2, 3] = float("nan")

        outputs = BaseEncoderOutput(
            last_hidden_state=hidden,
            attention_mask=None,
        )
        result = t5_large_postprocess_text(outputs)
        assert not torch.isnan(result).any()


class TestGen3CPipelineConfig:
    """Validate Gen3CConfig defaults and cross-validation."""

    def test_default_config_instantiation(self):
        """Gen3CConfig should instantiate with valid defaults."""
        config = Gen3CConfig()
        assert config.frame_buffer_max == 2
        assert config.fps == 24
        assert config.num_frames == 121
        # Official GEN3C/Cosmos inference defaults are 704x1280.
        assert config.video_resolution == (704, 1280)

    def test_frame_buffer_max_mismatch_raises(self):
        """Pipeline and DiT frame_buffer_max should be consistent."""
        config = Gen3CConfig()
        config.frame_buffer_max = 3  # mismatch with default DiT (2)
        with pytest.raises(ValueError, match="frame_buffer_max mismatch"):
            config.__post_init__()


class TestGen3CCFGPolicy:
    """Validate explicit CFG policy behavior for GEN3C."""

    def test_official_unity_cfg_fills_default_negative_prompt(self):
        config = Gen3CConfig()
        config.cfg_behavior = "official_uncond_at_unity"
        stage = Gen3CCFGPolicyStage()
        batch = ForwardBatch(
            data_type="video",
            prompt="test prompt",
            guidance_scale=1.0,
        )

        out = stage.forward(batch, SimpleNamespace(pipeline_config=config))

        assert out.do_classifier_free_guidance is True
        assert out.negative_prompt == config.default_negative_prompt

    def test_official_unity_cfg_keeps_existing_negative_embeds(self):
        config = Gen3CConfig()
        config.cfg_behavior = "official_uncond_at_unity"
        stage = Gen3CCFGPolicyStage()
        batch = ForwardBatch(
            data_type="video",
            prompt="test prompt",
            guidance_scale=1.0,
            negative_prompt_embeds=[torch.zeros(1, 1, 1)],
        )

        out = stage.forward(batch, SimpleNamespace(pipeline_config=config))

        assert out.do_classifier_free_guidance is True
        assert out.negative_prompt is None


class TestGen3CModelSmoke:
    """Lightweight model instantiation checks (no GPU required)."""

    def test_model_instantiation(self):
        config = _create_small_config()
        model = Gen3CTransformer3DModel(config=config, hf_config={})
        assert len(model.transformer_blocks) == 2
        assert model.hidden_size == 4096

    def test_buffer_channels(self):
        for n_buffers in (1, 2, 3):
            config = _create_small_config()
            config.arch_config.frame_buffer_max = n_buffers
            config.arch_config.__post_init__()
            expected = n_buffers * config.arch_config.CHANNELS_PER_BUFFER
            assert config.arch_config.buffer_channels == expected

    def test_patch_embed_channels(self):
        config = _create_small_config()
        model = Gen3CTransformer3DModel(config=config, hf_config={})
        expected = (
            config.arch_config.in_channels  # 16
            + 1  # condition_video_input_mask
            + config.arch_config.buffer_channels  # 64
            + (1 if config.arch_config.concat_padding_mask else 0)
        )
        actual = model.patch_embed.dim // (
            model.patch_size[0] * model.patch_size[1] * model.patch_size[2]
        )
        assert actual == expected


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GEN3C full pipeline smoke test requires CUDA.",
)
@pytest.mark.skipif(
    not os.getenv("GEN3C_DIFFUSERS_PATH"),
    reason="Set GEN3C_DIFFUSERS_PATH to run the full pipeline smoke test.",
)
def test_gen3c_full_pipeline_smoke():
    """End-to-end smoke test using converted weights.

    Requires:
        - Converted weights at GEN3C_DIFFUSERS_PATH (with model_index.json)
        - A CUDA GPU
    """
    from fastvideo import VideoGenerator

    diffusers_path = os.getenv("GEN3C_DIFFUSERS_PATH")
    if not os.path.isdir(diffusers_path):
        pytest.skip(f"Missing GEN3C diffusers repo at {diffusers_path}")
    if not os.path.isfile(os.path.join(diffusers_path, "model_index.json")):
        pytest.skip(f"Missing model_index.json in {diffusers_path}")

    device = torch.device("cuda:0")
    prompt = "A camera slowly orbits around a vase of flowers on a table."
    seed = 42
    height = 240  # Small for smoke test
    width = 424
    num_frames = 25  # Short clip
    steps = 4  # Minimal steps

    generator = VideoGenerator.from_pretrained(
        diffusers_path,
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    result = generator.generate_video(
        prompt=prompt,
        output_path="outputs_video/gen3c_smoke",
        save_video=False,
        height=height,
        width=width,
        num_frames=num_frames,
        fps=24,
        num_inference_steps=steps,
        guidance_scale=6.0,
        seed=seed,
    )
    generator.shutdown()

    fastvideo_out = result["samples"]
    fastvideo_out = fastvideo_out.to(device=device, dtype=torch.float32)
    _log_tensor_stats("gen3c_output", fastvideo_out)

    # Basic sanity checks
    assert fastvideo_out.ndim == 5, f"Expected 5D output, got {fastvideo_out.ndim}D"
    assert fastvideo_out.shape[0] == 1, "Batch size should be 1"
    assert not torch.isnan(fastvideo_out).any(), "Output contains NaNs"
    assert not torch.isinf(fastvideo_out).any(), "Output contains Infs"
    print(f"[GEN3C SMOKE] Full pipeline smoke test passed: {fastvideo_out.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
