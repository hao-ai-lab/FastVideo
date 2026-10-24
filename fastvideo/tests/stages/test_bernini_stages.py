# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.bernini.stages import projected_guidance, sample_frame_indices


@pytest.mark.parametrize("total,source_fps,fps,limit,expected", [
    (9, 24, 16, 81, [0, 2, 4, 6, 8]),
    (9, 16, 16, 5, [0, 1, 2, 3, 4]),
    (1, 16, 16, 5, [0, 0, 0, 0, 0]),
    (8, 16, 16, 81, [0, 1, 2, 3, 4, 5, 6, 7, 7]),
])
def test_bernini_frame_sampling(total, source_fps, fps, limit, expected):
    assert sample_frame_indices(total, source_fps, fps, limit) == expected


@pytest.mark.parametrize("total,source_fps,fps", [
    (0, 16, 16), (9, 0, 16), (9, 16, 0), (9, None, 16), (9, 16, float("nan")), (9, float("inf"), 16),
])
def test_bernini_bad_video_metadata(total, source_fps, fps):
    with pytest.raises(ValueError):
        sample_frame_indices(total, source_fps, fps, 9)


@pytest.mark.parametrize("changes,match", [
    ({"video_path": None}, "video_path"),
    ({"image_path": "image.png"}, "without reference images"),
    ({"prompt": ["a", "b"]}, "single prompt"),
    ({"num_videos_per_prompt": 2}, "single prompt"),
    ({"references": [{"type": "image", "path": "reference.png"}]}, "conditioning uses video_path"),
    ({"max_sequence_length": 513}, "between 1 and 512"),
    ({"max_sequence_length": 0}, "between 1 and 512"),
    ({"num_frames": 8}, "4\\*k\\+1"),
    ({"num_frames": 0}, "4\\*k\\+1"),
    ({"height": None}, "divisible by 16"),
    ({"height": -16}, "divisible by 16"),
    ({"width": 24}, "divisible by 16"),
])
def test_bernini_rejects_unsupported_inputs_before_video_io(changes, match):
    from fastvideo.pipelines.basic.bernini.stages import BerniniInputStage
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    fields = dict(data_type="video", prompt="edit", video_path="not-opened.mp4", height=32, width=48, num_frames=9)
    fields.update(changes)
    with pytest.raises(ValueError, match=match):
        BerniniInputStage().forward(ForwardBatch(**fields), SimpleNamespace())


def test_bernini_vae_uses_mode_and_preserves_rng(monkeypatch):
    import fastvideo.pipelines.basic.bernini.stages as stages
    from fastvideo.models.vaes.common import DiagonalGaussianDistribution
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    means = torch.arange(16*2*4*6, dtype=torch.float32).reshape(1,16,2,4,6)

    class VAE:
        config = SimpleNamespace(latents_mean=[1.0]*16, latents_std=[2.0]*16)

        def to(self, device):
            return self

        def encode(self, pixels):
            assert pixels.dtype == torch.float32
            return DiagonalGaussianDistribution(torch.cat([means, torch.zeros_like(means)], dim=1))

    batch = SimpleNamespace(video_latent=torch.zeros(1,3,5,32,48, dtype=torch.bfloat16))
    before = torch.random.get_rng_state().clone()
    stages.BerniniVideoEncodingStage(VAE()).forward(batch, SimpleNamespace(vae_cpu_offload=True))
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(batch.video_latent, (means-1)/2, rtol=0, atol=0)


def test_bernini_vae_exception_restores_cpu_placement(monkeypatch):
    import fastvideo.pipelines.basic.bernini.stages as stages
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    moves = []

    class VAE:
        def to(self, device):
            moves.append(str(device))
            return self

        def encode(self, pixels):
            raise RuntimeError("encoder failure")

    batch = SimpleNamespace(video_latent=torch.zeros(1,3,5,32,48))
    with pytest.raises(RuntimeError, match="encoder failure"):
        stages.BerniniVideoEncodingStage(VAE()).forward(batch, SimpleNamespace(vae_cpu_offload=True))
    assert moves == ["cpu", "cpu"]


def test_bernini_denoise_exception_restores_cpu_placement(monkeypatch):
    import fastvideo.pipelines.basic.bernini.stages as stages
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    moves = []

    class Model:
        def to(self, device):
            moves.append(str(device))
            return self

    args = SimpleNamespace(dit_cpu_offload=True, pipeline_config=SimpleNamespace(dit_precision="bf16"))
    batch = SimpleNamespace(video_latent=torch.zeros(1,16,2,4,6), generator=None,
                            latents=torch.zeros(1,16,1,4,6))
    with pytest.raises(ValueError, match="initial latents"):
        stages.BerniniDenoisingStage(Model(), None).forward(batch, args)
    assert moves == ["cpu", "cpu"]


def test_bernini_apg_per_frame_norm_and_zero_delta():
    torch.manual_seed(1)
    sample = torch.randn(1, 16, 3, 4, 6)
    cond, uncond = torch.randn_like(sample) * 30, torch.randn_like(sample) * 30
    sigma = torch.tensor(0.7)
    x_cond, x_uncond = sample - sigma * cond, sample - sigma * uncond
    delta = x_cond - x_uncond
    clipped = torch.stack([frame * min(1, 50 / frame.norm().item()) for frame in delta.unbind(2)], dim=2)
    reference = torch.nn.functional.normalize(x_cond.double(), dim=(1, 3, 4))
    parallel = (clipped.double() * reference).sum(dim=(1, 3, 4), keepdim=True) * reference
    guided_delta = (clipped.double() - parallel).float() + 0.5 * parallel.float()
    expected = (sample - (x_uncond + 4 * guided_delta)) / sigma
    torch.testing.assert_close(projected_guidance(cond, uncond, sample, sigma, 4), expected)
    torch.testing.assert_close(projected_guidance(cond, cond, sample, sigma, 4), cond)


@pytest.mark.parametrize("scale", [1.0, 3.0, 4.0])
def test_bernini_apg_small_delta_reduces_to_cfg(scale):
    sample = torch.zeros(1, 16, 2, 2, 2)
    cond = torch.ones_like(sample) * 0.1
    uncond = torch.ones_like(sample) * -0.1
    torch.testing.assert_close(projected_guidance(cond, uncond, sample, torch.tensor(0.5), scale, eta=1.0),
                               uncond + scale * (cond - uncond))


@pytest.mark.parametrize("steps,scale", [(3, 1.0), (3, 4.0), (10, 4.0)])
def test_bernini_denoising_trajectory_and_request_reset(monkeypatch, steps, scale):
    from diffusers import UniPCMultistepScheduler
    import fastvideo.pipelines.basic.bernini.stages as stages
    from fastvideo.models.dits.bernini import pack_latents
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, x, text, timestep, source_latents):
            self.calls.append((x.clone(), text.clone(), timestep.clone(), source_latents.clone()))
            return x * 0.1 + text.mean() + source_latents * 0.05

    model = Model()
    config = dict(num_train_timesteps=1000, solver_order=2, prediction_type="flow_prediction",
                  use_flow_sigmas=True, flow_shift=3.0, timestep_spacing="linspace")
    stage = stages.BerniniDenoisingStage(model, UniPCMultistepScheduler(**config))
    torch.manual_seed(22)
    source, noise = torch.randn(1, 16, 2, 4, 6), torch.randn(1, 16, 2, 4, 6)
    args = SimpleNamespace(dit_cpu_offload=True, pipeline_config=SimpleNamespace(flow_shift=3.0, dit_precision="fp32"))

    def batch():
        return SimpleNamespace(video_latent=source.clone(), latents=noise.clone(), generator=None,
                               prompt_embeds=[torch.ones(1, 5, 3)], negative_prompt_embeds=[torch.zeros(1, 5, 3)],
                               guidance_scale=scale, do_classifier_free_guidance=True, num_inference_steps=steps)

    actual = stage.forward(batch(), args).latents
    repeated = stage.forward(batch(), args).latents
    assert torch.equal(actual, repeated)
    assert len(model.calls) == 4 * steps
    assert all(torch.equal(c[3], source) for c in model.calls)
    # An independent spatial CFG/threshold calculation, stepped in the official
    # patch order. The small predictions stay below APG's threshold.
    expected = noise.clone()
    scheduler = UniPCMultistepScheduler(**config)
    scheduler.set_timesteps(steps)
    for t in scheduler.timesteps:
        cond = expected * 0.1 + 1 + source * 0.05
        uncond = expected * 0.1 + source * 0.05
        sigma = scheduler.sigmas[scheduler.step_index or 0]
        x_c, x_u = expected - sigma * cond, expected - sigma * uncond
        delta = x_c - x_u
        clipped = delta * (50.0 / delta.norm(p=2, dim=(1,3,4), keepdim=True)).clamp(max=1)
        reference = torch.nn.functional.normalize(x_c.double(), dim=(1,3,4))
        parallel = (clipped.double() * reference).sum(dim=(1,3,4), keepdim=True) * reference
        guided_delta = (clipped.double() - parallel).float() + 0.5 * parallel.float()
        prediction = (expected - (x_u + scale * guided_delta)) / sigma
        result = scheduler.step(pack_latents(prediction), t, pack_latents(expected), return_dict=False)[0]
        expected = stages.unpack_latents(result, tuple(noise.shape))
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_bernini_denoising_uses_configured_precision_with_fp32_first_parameter(monkeypatch):
    from diffusers import UniPCMultistepScheduler
    import fastvideo.pipelines.basic.bernini.stages as stages

    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.scale_shift_table = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.projection = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.calls = 0

        def forward(self, latents, text, timestep, source_latents):
            assert latents.dtype == text.dtype == source_latents.dtype == self.projection.dtype
            self.calls += 1
            return latents * 0.1 + text.mean() + source_latents * 0.05

    model = Model()
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    batch = SimpleNamespace(video_latent=torch.ones(1, 16, 2, 4, 6), latents=None,
                            generator=[torch.Generator("cpu").manual_seed(42)],
                            prompt_embeds=[torch.ones(1, 5, 3)], negative_prompt_embeds=[torch.zeros(1, 5, 3)],
                            guidance_scale=4.0, do_classifier_free_guidance=True, num_inference_steps=3)
    args = SimpleNamespace(dit_cpu_offload=True,
                           pipeline_config=SimpleNamespace(flow_shift=3.0, dit_precision="bf16"))

    result = stages.BerniniDenoisingStage(model, scheduler).forward(batch, args)

    assert next(model.parameters()).dtype == torch.float32
    assert model.calls == 6
    assert result.latents.dtype == torch.float32
    assert torch.isfinite(result.latents).all()
