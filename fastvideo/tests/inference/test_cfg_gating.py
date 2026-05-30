# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
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
        timestep_term = timestep.to(latent_model_input.device, torch.float32).reshape(
            latent_model_input.shape[0], *([1] * (latent_model_input.ndim - 1)))
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

    # envs.py evaluates FASTVIDEO_CFG_GATE_STEP lazily via __getattr__, so the
    # stage sees monkeypatched values without reloading the module.
    assert denoising.envs.FASTVIDEO_CFG_GATE_STEP == expected_gate_step

    logger = RecordingLogger()
    monkeypatch.setattr(denoising, "logger", logger)
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(denoising, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(denoising, "get_attn_backend", lambda **kwargs: object())
    monkeypatch.setattr(denoising, "set_forward_context", lambda **kwargs: nullcontext())
    return denoising, logger


def _run_stage(monkeypatch, cfg_gate_step):
    denoising, logger = _patch_denoising_module(monkeypatch, cfg_gate_step)
    model = TinyDenoiser()
    stage = denoising.DenoisingStage(model, TinyScheduler())
    stage.progress_bar = lambda iterable=None, total=None: NullProgressBar()

    result = stage.forward(_tiny_batch(), _tiny_args())
    return result.latents, model, logger


def _run_legacy_two_pass():
    batch = _tiny_batch()
    model = TinyDenoiser()
    scheduler = TinyScheduler()
    assert batch.latents is not None
    assert batch.timesteps is not None
    latents = batch.latents.clone()

    for timestep in batch.timesteps:
        latent_model_input = scheduler.scale_model_input(latents.to(torch.bfloat16), timestep)
        timestep_expand = timestep.repeat(latent_model_input.shape[0])
        noise_pred_text = model(latent_model_input, batch.prompt_embeds, timestep_expand)
        noise_pred_uncond = model(latent_model_input, batch.negative_prompt_embeds, timestep_expand)
        noise_pred = noise_pred_uncond + batch.guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    return latents


@pytest.mark.parametrize("cfg_gate_step", [None, "1.0"])
def test_cfg_gating_default_off_matches_legacy_two_pass(monkeypatch, cfg_gate_step):
    out, model, logger = _run_stage(monkeypatch, cfg_gate_step)
    legacy_out = _run_legacy_two_pass()

    assert torch.equal(out, legacy_out)
    assert model.calls == ["cond", "uncond"] * 4
    assert not any("CFG gating enabled" in msg for msg in logger.infos)
    assert any("gate_step=-1/4" in msg and "reused=0" in msg for msg in logger.infos)


def test_cfg_gating_reuses_cached_delta_after_gate(monkeypatch):
    out, model, logger = _run_stage(monkeypatch, "0.5")
    legacy_out = _run_legacy_two_pass()

    assert model.calls == ["cond", "uncond", "cond", "uncond", "cond", "cond"]
    assert any("CFG gating enabled: fraction=0.500, gate_step=2/4" in msg for msg in logger.infos)
    assert any("fresh_uncond=2 reused=2 invalidations=0" in msg for msg in logger.infos)
    assert torch.allclose(out, legacy_out, atol=1e-3, rtol=0.0)
