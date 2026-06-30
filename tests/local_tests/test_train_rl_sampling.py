import torch
import pytest

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.rl.common import (
    DiffusionSampler,
    RLValidationConfig,
    SamplingConfig,
    distributed_k_repeat_indices,
    media_to_video_array,
    validation_caption,
    validation_shard_indices,
)
from fastvideo.train.utils.config import load_run_config


class _FakeScheduler:

    def __init__(self):
        self.num_train_timesteps = 1000
        self.set_timesteps_calls = []
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self.step_calls = 0

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, sigmas=None):
        self.set_timesteps_calls.append({
            "num_inference_steps": num_inference_steps,
            "timesteps": timesteps,
            "sigmas": sigmas,
        })
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, device=device, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1000, 0, int(num_inference_steps), device=device)
        if sigmas is not None:
            self.sigmas = torch.tensor(sigmas, device=device, dtype=torch.float32)
        else:
            self.sigmas = torch.cat([self.timesteps / 1000.0, torch.zeros(1, device=device)])

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep
        self.step_calls += 1
        prev = sample + model_output
        return (prev, ) if not return_dict else {"prev_sample": prev}


class _FakeModel:

    def __init__(self):
        self.noise_scheduler = _FakeScheduler()
        self.add_noise_calls = 0
        self.timestep_shapes = []
        self.conditional_calls = []

    def predict_noise(self, noisy_latents, timestep, batch, *, conditional, attn_kind):
        del attn_kind
        self.conditional_calls.append(bool(conditional))
        self.timestep_shapes.append(tuple(timestep.shape))
        assert batch.timesteps is timestep
        value = 1.0 if conditional else -1.0
        return torch.full_like(noisy_latents, value)

    def predict_x0(self, noisy_latents, timestep, batch, *, conditional, attn_kind):
        del attn_kind
        self.conditional_calls.append(bool(conditional))
        self.timestep_shapes.append(tuple(timestep.shape))
        assert batch.timesteps is timestep
        value = 1.0 if conditional else -1.0
        return noisy_latents + value

    def add_noise(self, clean_latents, noise, timestep):
        del timestep
        self.add_noise_calls += 1
        return clean_latents + noise


def _batch():
    batch = TrainingBatch()
    batch.latents = torch.zeros(2, 1, 3, 4, 4)
    return batch


def test_sampler_preserves_latent_dtype():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))
    batch = _batch()
    batch.latents = batch.latents.to(torch.bfloat16)

    result = sampler.sample(model, batch, generator=torch.Generator().manual_seed(0))

    assert result.latents.dtype is torch.bfloat16


def test_sampler_uses_scheduler_generated_timesteps_by_default():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [1000.0, 666.6666259765625, 333.3333435058594, 0.0]


def test_sampler_honors_explicit_timestep_override():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=3, timesteps=[900, 300, 10]))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [900.0, 300.0, 10.0]


def test_sampler_honors_explicit_timesteps_without_matching_num_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(timesteps=[900, 300, 10]))

    result = sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert result.timesteps.tolist() == [900.0, 300.0, 10.0]
    assert model.noise_scheduler.set_timesteps_calls == []


def test_sampling_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unsupported method.sampling key"):
        SamplingConfig.from_mapping({"solver": "dpm2"})


def test_sampling_config_accepts_flow_unipc_and_rejects_explicit_timesteps():
    cfg = SamplingConfig.from_mapping({"scheduler": "flow_unipc", "num_steps": 50, "flow_shift": 8})

    assert cfg.scheduler == "flow_unipc"
    assert cfg.num_steps == 50
    assert cfg.flow_shift == 8.0

    with pytest.raises(ValueError, match="timesteps is not supported with flow_unipc"):
        SamplingConfig.from_mapping({"scheduler": "flow_unipc", "timesteps": [1000, 500, 0]})


def test_sampler_restores_original_batch_timestep_after_sampling():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=2))
    batch = _batch()
    original_timesteps = torch.tensor([123.0])
    batch.timesteps = original_timesteps

    sampler.sample(batch=batch, model=model, generator=torch.Generator().manual_seed(0))

    assert batch.timesteps is original_timesteps


def test_euler_sampler_does_not_renoise_between_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 0
    assert model.timestep_shapes == [(2,), (2,), (2,), (2,)]
    assert model.conditional_calls == [True, True, True, True]


def test_euler_sampler_applies_cfg_guidance_when_requested():
    baseline_model = _FakeModel()
    guided_model = _FakeModel()
    baseline_sampler = DiffusionSampler(SamplingConfig(num_steps=1))
    guided_sampler = DiffusionSampler(SamplingConfig(num_steps=1, guidance_scale=3.0))

    baseline = baseline_sampler.sample(
        baseline_model,
        _batch(),
        generator=torch.Generator().manual_seed(0),
    )
    guided = guided_sampler.sample(
        guided_model,
        _batch(),
        generator=torch.Generator().manual_seed(0),
    )

    # Fake cond prediction is +1 and uncond is -1, so guidance=3 changes
    # the one-step scheduler update from +1 to +5.
    assert torch.allclose(guided.latents - baseline.latents, torch.full_like(guided.latents, 4.0))
    assert baseline_model.conditional_calls == [True]
    assert guided_model.conditional_calls == [True, False]


def test_sde_reflow_sampler_renoises_between_steps():
    model = _FakeModel()
    sampler = DiffusionSampler(SamplingConfig(num_steps=4, trajectory="sde_reflow"))

    sampler.sample(model, _batch(), generator=torch.Generator().manual_seed(0))

    assert model.add_noise_calls == 3


def test_diffusion_nft_config_uses_rl_sampler_not_dmd_pipeline():
    config_path = "examples/train/configs/rl/wan/diffusion_nft_pick_clip.yaml"

    cfg = load_run_config(config_path)
    raw_text = open(config_path, encoding="utf-8").read()

    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.diffusion_nft.DiffusionNFTMethod"
    assert cfg.training.optimizer.learning_rate == 3.0e-5
    assert cfg.training.data.num_latent_t == 1
    assert cfg.training.data.num_frames == 1
    assert "sampling_timesteps" not in raw_text
    assert "WanDMDPipeline" not in raw_text
    assert "solver" not in cfg.method["sampling"]
    assert cfg.method["sampling"]["scheduler"] == "flow_match_euler"
    assert cfg.method["sampling"]["trajectory"] == "ode"
    assert cfg.method["sampling"]["flow_shift"] == "inherit"
    assert "deterministic" not in cfg.method["sampling"]
    assert "noise_level" not in cfg.method["sampling"]
    assert cfg.method["validation"]["every_steps"] == 10
    assert cfg.method["validation"]["num_steps"] == 40
    assert cfg.method["validation"]["num_prompts"] == 16
    assert cfg.method["validation"]["log_samples"] is True


def test_diffusion_nft_video_config_uses_genrl_rewards_in_clean_layout():
    config_path = "examples/train/configs/rl/wan/diffusion_nft_videoalign.yaml"

    cfg = load_run_config(config_path)
    raw_text = open(config_path, encoding="utf-8").read()

    assert cfg.method["_target_"] == "fastvideo.train.methods.rl.diffusion_nft.DiffusionNFTMethod"
    assert cfg.method["reward_backend"] == "genrl"
    assert cfg.method["reward_fn"]["rewards"] == {
        "videoalign_vq": 1.0,
        "videoalign_mq": 1.0,
        "videoalign_ta": 1.0,
    }
    assert cfg.method["sampling"]["scheduler"] == "flow_unipc"
    assert cfg.method["sampling"]["trajectory"] == "ode"
    assert cfg.method["sampling"]["num_steps"] == 50
    assert cfg.method["sampling"]["flow_shift"] == 8.0
    assert cfg.method["sampling"]["guidance_scale"] == 6.0
    assert cfg.method["validation"]["num_steps"] == 50
    assert cfg.method["validation"]["num_prompts"] == 16
    assert cfg.method["validation"]["batch_size"] == 4
    assert cfg.method["validation"]["max_samples"] == 4
    assert cfg.method["validation"]["fps"] == 16
    assert cfg.method["beta"] == 0.1
    assert cfg.method["kl_beta"] == 0.0001
    assert cfg.training.loop.gradient_accumulation_steps == 24
    assert cfg.training.data.num_latent_t == 20
    assert cfg.training.data.num_frames == 77
    assert "rl/reward/" not in raw_text
    assert "WanDMDPipeline" not in raw_text
    assert "solver" not in cfg.method["sampling"]


def test_validation_shard_indices_are_stable_and_padded():
    rank0 = validation_shard_indices(5, rank=0, world_size=2)
    rank1 = validation_shard_indices(5, rank=1, world_size=2)

    assert rank0 == [(0, True), (2, True), (4, True)]
    assert rank1 == [(1, True), (3, True), (0, False)]


def test_rl_validation_config_parses_video_fps():
    config = RLValidationConfig.from_mapping({"fps": 16, "max_samples": 2})

    assert config.fps == 16
    assert config.max_samples == 2


def test_distributed_k_repeat_indices_repeats_prompts_globally():
    rank0 = distributed_k_repeat_indices(
        dataset_length=100,
        batch_size=6,
        repeats_per_prompt=24,
        world_size=4,
        rank=0,
        seed=123,
    )
    all_indices = []
    for rank in range(4):
        sample = distributed_k_repeat_indices(
            dataset_length=100,
            batch_size=6,
            repeats_per_prompt=24,
            world_size=4,
            rank=rank,
            seed=123,
        )
        all_indices.extend(sample.local_indices)

    assert rank0.unique_prompt_count == 1
    assert len(all_indices) == 24
    assert len(set(all_indices)) == 1


def test_validation_caption_puts_rewards_first():
    caption = validation_caption(
        "a small blue cube",
        {
            "avg": 0.75,
            "pickscore": 0.5,
        },
    )

    assert caption.startswith("avg: 0.7500 | pickscore: 0.5000 | ")
    assert caption.endswith("a small blue cube")


def test_media_to_video_array_treats_frame_as_single_frame_video():
    frame = torch.ones(3, 4, 5)

    video = media_to_video_array(frame)

    assert video.shape == (1, 3, 4, 5)
    assert video.dtype.name == "uint8"


def test_media_to_video_array_preserves_video_frames():
    media = torch.ones(3, 2, 4, 5)

    video = media_to_video_array(media)

    assert video.shape == (2, 3, 4, 5)
