"""
Equivalence test for batched-CFG vs. sequential-CFG in
``DenoisingStage.forward``.

The transformer stub is a pure deterministic function of
``(hidden_states, encoder_hidden_states[0].mean(), timestep)``. With no
batch-coupled layers and a no-op scheduler step, running the same
denoise loop with ``use_batched_cfg=True`` and ``use_batched_cfg=False``
must produce identical final latents.

This catches the easy ways to break the port — wrong cat order
(neg vs pos), wrong chunk(2) split direction, broken CFG combine math,
list-of-encoders convention drops — without needing a real DiT.
"""
import types
from typing import Any

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage


class _StubTransformer(torch.nn.Module):
    """Deterministic per-sample noise prediction.

    Output element-wise depends only on the per-batch-item slice of
    ``hidden_states``, the per-batch-item ``encoder_hidden_states`` mean,
    and the per-batch-item ``timestep``. No cross-batch coupling, so
    ``f(cat([uncond, cond]))`` chunked back equals
    ``[f(uncond), f(cond)]`` bit-for-bit.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(use_meanflow=False)
        self.register_parameter("_p", torch.nn.Parameter(torch.zeros(1)))

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.Tensor,
                guidance: torch.Tensor | None = None,
                encoder_hidden_states_2: Any = None,
                encoder_attention_mask: Any = None,
                encoder_hidden_states_image: Any = None,
                mask_strategy: Any = None,
                mouse_cond: torch.Tensor | None = None,
                keyboard_cond: torch.Tensor | None = None,
                c2ws_plucker_emb: torch.Tensor | None = None,
                camera_states: torch.Tensor | None = None,
                timestep_r: torch.Tensor | None = None) -> torch.Tensor:
        if encoder_hidden_states is not None and not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        ctx = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        ts = timestep.reshape(-1, 1, 1, 1, 1).to(hidden_states.dtype)
        return hidden_states * 0.1 + ctx.to(hidden_states.dtype) * 0.5 + ts * 1e-4


class _IdentityScheduler:
    """No-op scheduler: ``step`` returns the input ``latents`` unchanged.

    Keeps the per-step noise_pred isolated so the final-latents
    comparison reduces to: did the cond/uncond combine math match?
    """

    num_train_timesteps = 1000
    order = 1

    def scale_model_input(self, sample: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return sample

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor, latents: torch.Tensor,
             return_dict: bool = True) -> tuple[torch.Tensor]:
        # Return latents - noise_pred so the per-step contribution is
        # observable in the final latents (otherwise CFG differences
        # would be invisible).
        return (latents - noise_pred, )


def _make_fastvideo_args(use_batched_cfg: bool) -> Any:
    return types.SimpleNamespace(
        model_loaded={"transformer": True},
        pipeline_config=types.SimpleNamespace(
            embedded_cfg_scale=None,
            ti2v_task=False,
            vae_config=types.SimpleNamespace(arch_config=types.SimpleNamespace(scale_factor_temporal=1,
                                                                               scale_factor_spatial=1)),
            dit_config=types.SimpleNamespace(boundary_ratio=None,
                                             arch_config=types.SimpleNamespace(patch_size=(1, 1, 1))),
        ),
        disable_autocast=True,
        use_batched_cfg=use_batched_cfg,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        VSA_sparsity=0.0,
        moba_config={},
    )


def _make_batch(do_cfg: bool) -> ForwardBatch:
    torch.manual_seed(0)
    latents = torch.randn(1, 4, 2, 4, 4)
    # Distinct pos / neg embeddings so the CFG combine actually matters.
    prompt_embeds = [torch.randn(1, 6, 8)]
    negative_prompt_embeds = [torch.randn(1, 6, 8)]
    timesteps = torch.tensor([100, 50, 10], dtype=torch.float32)
    return ForwardBatch(
        data_type="video",
        latents=latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds if do_cfg else None,
        do_classifier_free_guidance=do_cfg,
        timesteps=timesteps,
        num_inference_steps=len(timesteps),
        guidance_scale=7.5,
        guidance_scale_2=7.5,
        guidance_rescale=0.0,
    )


def _make_stage() -> DenoisingStage:
    """Bypass ``__init__`` (which calls into ``get_attn_backend``) and
    set the fields ``forward`` actually reads."""
    stage = DenoisingStage.__new__(DenoisingStage)
    torch.nn.Module.__init__(stage)
    stage.transformer = _StubTransformer()
    stage.transformer_2 = None
    stage.scheduler = _IdentityScheduler()
    stage.vae = None
    stage.pipeline = None
    # attn_backend is checked only against VSA / VMOBA backend types in
    # forward; any sentinel object that doesn't match those works.
    stage.attn_backend = object()
    return stage


def _run(use_batched_cfg: bool, do_cfg: bool = True) -> torch.Tensor:
    stage = _make_stage()
    batch = _make_batch(do_cfg=do_cfg)
    fastvideo_args = _make_fastvideo_args(use_batched_cfg=use_batched_cfg)
    out = stage.forward(batch, fastvideo_args)
    assert out.latents is not None
    return out.latents


def test_batched_cfg_matches_sequential_cfg() -> None:
    """Core equivalence: batched and sequential paths must produce
    identical final latents on a pure deterministic transformer."""
    latents_batched = _run(use_batched_cfg=True, do_cfg=True)
    latents_sequential = _run(use_batched_cfg=False, do_cfg=True)
    assert torch.equal(latents_batched, latents_sequential), (
        f"batched-CFG diverged from sequential-CFG; "
        f"max abs diff = {(latents_batched - latents_sequential).abs().max()}")


def test_batched_cfg_flag_off_is_legacy_path() -> None:
    """With CFG off, both paths reduce to a single forward and must
    match regardless of the ``use_batched_cfg`` flag."""
    a = _run(use_batched_cfg=True, do_cfg=False)
    b = _run(use_batched_cfg=False, do_cfg=False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("disabling_field, value", [
    ("video_latent", torch.zeros(1, 4, 2, 4, 4)),
    ("image_latent", torch.zeros(1, 4, 2, 4, 4)),
    ("image_embeds", [torch.zeros(1, 4)]),
    ("mouse_cond", torch.zeros(1, 2, 2)),
    ("keyboard_cond", torch.zeros(1, 2, 4)),
    ("c2ws_plucker_emb", torch.zeros(1, 6, 2, 4, 4)),
    ("camera_states", torch.zeros(1, 2, 6, 4, 4)),
])
def test_batched_cfg_autodetect_disables_on_conditioning(disabling_field: str, value: Any) -> None:
    """When V2V/I2V/action/camera conditioning is present, the batched
    path must auto-disable. We verify by checking that the result with
    ``use_batched_cfg=True`` equals the result with the flag forced off
    — i.e. the autodetect routed us through the sequential path."""
    stage = _make_stage()
    fastvideo_args_batched = _make_fastvideo_args(use_batched_cfg=True)
    fastvideo_args_seq = _make_fastvideo_args(use_batched_cfg=False)

    def _batch_with_field() -> ForwardBatch:
        b = _make_batch(do_cfg=True)
        setattr(b, disabling_field, value)
        return b

    # The V2V / I2V latent paths trigger extra cat()s in the
    # transformer call that the stub can't simulate (the stub returns a
    # noise tensor shaped like the input, so a wider latent_model_input
    # would just propagate). For the autodetect test we only care that
    # the *gate* picks the sequential branch — we run with the flag in
    # both states and assert equality, even when video/image latents
    # trip the inner cat (the cat is the same on both runs).
    out_a = stage.forward(_batch_with_field(), fastvideo_args_batched)
    # Fresh stage to avoid state carry-over.
    stage_b = _make_stage()
    out_b = stage_b.forward(_batch_with_field(), fastvideo_args_seq)
    assert out_a.latents is not None and out_b.latents is not None
    assert torch.equal(out_a.latents, out_b.latents)
