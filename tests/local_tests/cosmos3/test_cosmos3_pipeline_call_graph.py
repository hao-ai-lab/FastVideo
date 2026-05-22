# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline call-graph parity (Tier A scaffold).

Reference:
  * ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:883-1033`` —
    ``Cosmos3OmniDiffusersPipeline.diffuse``: 3-mode CFG denoising loop
    with UND-cache management.
  * ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:1037-1067`` —
    ``forward``: parses the request, selects T2I/T2V/I2V mode, dispatches
    defaults.
  * Reference test invariants at
    ``tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py:126-156,196-251``.

The Tier A scaffold uses the stubs from ``conftest.py`` (StubScheduler,
StubCosmos3VAE, StubCosmos3Transformer) so the call-graph can be tested
without a real DiT or VAE. The 4 invariants under test:

  1. ``diffuse(...)`` calls ``transformer.reset_cache()`` exactly once
     before iterating timesteps;
  2. With ``do_cfg=True`` and ``guidance_interval=None``, each timestep
     invokes the transformer twice — once with ``cond_ids`` and once with
     ``uncond_ids`` — and reuses the cached UND K/V from step 1 onward
     (asserted via ``StubCosmos3Transformer.calls`` order);
  3. I2V mode: ``velocity_mask`` zeros frame-0 noise predictions, and
     ``image_latent`` is re-injected into frame 0 after each scheduler step;
  4. ``forward`` selects T2I vs T2V mode from ``prompt["modalities"]``
     and applies the per-mode default ``flow_shift`` / ``num_frames``.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.local]


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def test_diffuse_resets_cache_and_calls_cfg_in_order(make_cosmos3_pipeline) -> None:
    """Asserts the 3-mode CFG call graph: ``reset_cache`` first, then
    interleaved ``cond / uncond`` transformer calls with cache reuse.

    With timesteps ``[900, 100]`` and ``guidance_scale=3.0``, the upstream
    reference (test_cosmos3_pipeline.py:126-142) asserts the call order is
    ``[2, 1, 2]`` — i.e. step 0 calls cond (token=2) then uncond (token=1),
    step 1 reuses the cached UND K/V and only calls cond again (token=2).
    """
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )
    assert pipeline.transformer.reset_calls == 1
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2]
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))


def test_diffuse_i2v_velocity_mask_zeros_frame_zero(make_cosmos3_pipeline) -> None:
    """Asserts I2V velocity-mask + image-latent re-injection contract.

    Cross-check: pipeline_cosmos3.py:937-951. The velocity_mask zeroes
    noise predictions on conditioning frames before stepping, and
    ``image_latent`` is overwritten into frame 0 of the output latents
    after each scheduler step.
    """
    pipeline = make_cosmos3_pipeline()
    result = pipeline.diffuse(
        latents=torch.zeros(1, 2, 2, 1, 1),
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
        velocity_mask=torch.tensor([[[[[0.0]], [[1.0]]]]]),
        image_latent=torch.full((1, 2, 1, 1, 1), 7.0),
    )
    torch.testing.assert_close(result[:, :, 0:1], torch.full((1, 2, 1, 1, 1), 7.0))


@pytest.mark.parametrize(
    ("modalities", "expected_is_t2i", "expected_default_flow_shift", "expected_default_frames"),
    [
        (["image"], True, 3.0, 1),
        (["video"], False, 1.0, 189),
    ],
)
def test_forward_mode_dispatch_t2i_vs_t2v(
    make_cosmos3_pipeline,
    modalities: list[str],
    expected_is_t2i: bool,
    expected_default_flow_shift: float,
    expected_default_frames: int,
) -> None:
    """Asserts forward() routes to T2I vs T2V mode based on prompt modalities,
    and applies per-mode defaults.

    Cross-check: pipeline_cosmos3.py:1069-1093. T2I defaults:
    ``num_frames=1``, ``flow_shift=3.0``, ``num_inference_steps=50``,
    ``guidance_interval=[400, 1000]``. T2V defaults: ``num_frames=189``,
    ``flow_shift=engine_init`` (1.0 here), ``num_inference_steps=35``,
    no guidance_interval.
    """
    pipeline = make_cosmos3_pipeline()

    from types import SimpleNamespace

    captured: dict[str, object] = {"flow_shifts": [], "format_calls": []}

    def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
        captured["format_calls"].append(
            {
                "is_t2i": kwargs.get("is_t2i"),
                "num_frames": num_frames,
            }
        )
        return _ids(2), _mask(), _ids(1), _mask()

    pipeline._format_and_tokenize_prompts = fake_format
    pipeline._prepare_latents = lambda *a, **kw: torch.zeros(1, 2, 1, 1, 1)
    pipeline._set_flow_shift = lambda target: captured["flow_shifts"].append(target)
    pipeline._set_scheduler_timesteps = lambda steps: setattr(
        pipeline.scheduler, "timesteps", torch.tensor([7])
    )
    pipeline.diffuse = lambda **kw: kw["latents"]
    pipeline._decode_latents = lambda latents: latents

    output = pipeline.forward(
        SimpleNamespace(
            prompts=[{"prompt": "test", "modalities": modalities}],
            sampling_params=SimpleNamespace(
                height=None,
                width=None,
                num_frames=None,
                num_inference_steps=None,
                guidance_scale=None,
                generator=None,
                seed=123,
                num_outputs_per_prompt=1,
                frame_rate=None,
                resolved_frame_rate=None,
                max_sequence_length=None,
                extra_args={},
            ),
        )
    )
    assert captured["format_calls"][-1]["is_t2i"] is expected_is_t2i
    assert captured["format_calls"][-1]["num_frames"] == expected_default_frames
    assert captured["flow_shifts"] == [expected_default_flow_shift]
    expected_output_key = "image" if expected_is_t2i else "video"
    assert expected_output_key in output.output


def test_forward_rejects_both_image_and_video_modalities(make_cosmos3_pipeline) -> None:
    """Asserts ``_is_t2i_request`` raises ValueError when a prompt
    requests both image and video modalities simultaneously.

    Cross-check: pipeline_cosmos3.py:490-496.
    """
    pipeline = make_cosmos3_pipeline()

    from types import SimpleNamespace

    with pytest.raises(ValueError, match="both image and video"):
        pipeline.forward(
            SimpleNamespace(
                prompts=[{"prompt": "x", "modalities": ["image", "video"]}],
                sampling_params=SimpleNamespace(
                    height=None,
                    width=None,
                    num_frames=None,
                    num_inference_steps=None,
                    guidance_scale=None,
                    generator=None,
                    seed=123,
                    num_outputs_per_prompt=1,
                    frame_rate=None,
                    resolved_frame_rate=None,
                    max_sequence_length=None,
                    extra_args={},
                ),
            )
        )
