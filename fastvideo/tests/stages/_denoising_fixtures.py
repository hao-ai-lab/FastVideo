# SPDX-License-Identifier: Apache-2.0
"""Weight-free fixtures for shared and Wan denoising contract tests."""

from contextlib import nullcontext
from types import SimpleNamespace

import torch


class RecordingLogger:

    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


class NullProgressBar:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self):
        pass


class TinyScheduler:
    order = 1
    num_train_timesteps = 1000

    def scale_model_input(self, latents, timestep):
        return latents

    def step(self, noise_pred, timestep, latents, return_dict=False):
        return (latents.float() - 0.01 * noise_pred.float(), )


class TinyDenoiser(torch.nn.Module):
    hidden_size = 1
    num_attention_heads = 1

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_meanflow=False)
        self.calls = []

    def forward(self, latent_model_input, prompt_embeds, timestep, guidance=None):
        prompt_value = prompt_embeds[0].to(latent_model_input.device, torch.float32)
        is_uncond = bool(prompt_value.item() < 0)
        self.calls.append("uncond" if is_uncond else "cond")

        prompt_term = prompt_value.reshape((1, ) * latent_model_input.ndim)
        timestep_term = timestep.to(latent_model_input.device,
                                    torch.float32).reshape(latent_model_input.shape[0],
                                                           *([1] * (latent_model_input.ndim - 1)))
        return latent_model_input.float() * 0.2 + prompt_term * 0.5 + timestep_term * 0.001


def _tiny_args():
    return SimpleNamespace(
        disable_autocast=True,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        moba_config={},
        VSA_sparsity=0.0,
        model_loaded={"transformer": True},
        model_paths={"transformer": "unused"},
        pipeline_config=SimpleNamespace(
            embedded_cfg_scale=None,
            ti2v_task=False,
            lucy_edit_task=False,
            dit_config=SimpleNamespace(boundary_ratio=None, patch_size=(1, 1, 1)),
        ),
    )


def _tiny_batch():
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    return ForwardBatch(
        data_type="video",
        latents=torch.tensor([[[[[0.125]]]]], dtype=torch.float32),
        prompt_embeds=[torch.tensor([1.0], dtype=torch.float32)],
        negative_prompt_embeds=[torch.tensor([-1.0], dtype=torch.float32)],
        timesteps=torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32),
        num_inference_steps=4,
        guidance_scale=2.0,
        height=1,
        width=1,
        num_frames=1,
        raw_latent_shape=(1, 1, 1, 1, 1),
        save_video=False,
    )


def _patch_denoising_module(monkeypatch, cfg_gate_step):
    if cfg_gate_step is None:
        monkeypatch.delenv("FASTVIDEO_CFG_GATE_STEP", raising=False)
        expected_gate_step = 1.0
    else:
        monkeypatch.setenv("FASTVIDEO_CFG_GATE_STEP", str(cfg_gate_step))
        expected_gate_step = float(cfg_gate_step)

    import fastvideo.pipelines.stages.denoising as denoising
    from fastvideo.pipelines.stages.base import PipelineStage

    # envs.py evaluates FASTVIDEO_CFG_GATE_STEP lazily via __getattr__, so the
    # stage sees monkeypatched values without reloading the module.
    assert denoising.envs.FASTVIDEO_CFG_GATE_STEP == expected_gate_step

    logger = RecordingLogger()
    monkeypatch.setattr(denoising, "logger", logger)
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(denoising, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(denoising, "get_attn_backend", lambda **kwargs: object())
    monkeypatch.setattr(denoising, "set_forward_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(PipelineStage, "device", property(lambda self: torch.device("cpu")))
    return denoising, logger


class RecordingDenoiser(TinyDenoiser):

    def __init__(self, offset=0.0):
        super().__init__()
        self.offset = offset
        self.inputs = []

    def forward(self, hidden_states, prompt_embeds, timestep, guidance=None, encoder_hidden_states_image=None):
        self.inputs.append((hidden_states.clone(), timestep.clone()))
        prompt = prompt_embeds[0].float().mean()
        self.calls.append("uncond" if prompt < 0 else "cond")
        time = timestep.float().reshape(hidden_states.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1, 1)
        return hidden_states[:, :2].float() * 0.125 + prompt * 0.25 + time * 0.0001 + self.offset


class TinyVAE(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.shift_factor = torch.tensor([0.25, -0.25]).reshape(1, 2, 1, 1, 1)
        self.scaling_factor = torch.tensor([2.0, 0.5]).reshape(1, 2, 1, 1, 1)
        self.encode_calls = 0

    def encode(self, image):
        self.encode_calls += 1
        return SimpleNamespace(mean=torch.full((1, 2, 1, 2, 4), 0.5, dtype=torch.float32))


def _batch(steps=4, cfg=True):
    batch = _tiny_batch()
    batch.latents = torch.linspace(-0.5, 0.5, 48).reshape(1, 2, 3, 2, 4)
    batch.num_inference_steps = steps
    batch.timesteps = torch.linspace(1000, 1, steps)
    batch.do_classifier_free_guidance = cfg
    batch.num_frames, batch.height, batch.width = 9, 16, 32
    batch.raw_latent_shape = tuple(batch.latents.shape)
    batch.return_trajectory_latents = True
    return batch


def _args():
    args = _tiny_args()
    args.vae_cpu_offload = True
    arch = SimpleNamespace(patch_size=(1, 2, 2))
    args.pipeline_config.dit_config.arch_config = arch
    args.pipeline_config.vae_config = SimpleNamespace(
        arch_config=SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8))
    return args
