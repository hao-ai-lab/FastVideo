# SPDX-License-Identifier: Apache-2.0
"""Check P-JiT's sampler against release arithmetic without loading a checkpoint."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.pipelines.pjit import PJiTPipelineConfig
from fastvideo.pipelines.basic.pjit import stages
from fastvideo.pipelines.basic.pjit.noise import pixel_noise, randn_as_on_sm_count
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _Predictor:

    def __call__(self, *, x_t, t, text, text_lens):
        assert x_t.dtype == torch.float32 and torch.equal(x_t, x_t.bfloat16().float())
        assert torch.equal(x_t[0], x_t[1])
        assert text.shape == (2, 4, 2)
        assert text_lens.dtype == torch.int32 and text_lens.tolist() == [2, 1]
        assert not text[:, 2:].count_nonzero() and not text[1, 1:].count_nonzero()
        offset = (text.float().sum(dim=(1, 2)) / 20)[:, None, None, None, None]
        channel = torch.tensor([-1., 0.5, 2.]).view(1, 3, 1, 1, 1)
        return (.13 * x_t + offset * channel + (t.float() / 1000).view(-1, 1, 1, 1, 1) * .01).bfloat16()


@pytest.mark.parametrize("supply_state", [False, True])
def test_euler_apg_matches_release_sampler(monkeypatch, supply_state):
    monkeypatch.setattr(stages, "get_local_torch_device", lambda: torch.device("cpu"))
    config = PJiTPipelineConfig(apg_eta=.2, apg_rescale=.4, noise_sm_count=None)
    config.dit_config.arch_config.text_len = 4
    args = SimpleNamespace(pipeline_config=config, dit_cpu_offload=False, enable_stage_verification=True)
    batch = ForwardBatch(data_type="image",
                         height=2,
                         width=3,
                         seed=19,
                         num_inference_steps=4,
                         guidance_scale=1.6,
                         prompt_embeds=[torch.tensor([[.5, 1.5], [-.5, .3]], dtype=torch.bfloat16)],
                         negative_prompt_embeds=[torch.tensor([[.7, -.8]], dtype=torch.bfloat16)],
                         return_trajectory_latents=True,
                         return_trajectory_decoded=True)
    if supply_state:
        batch.latents = 4 * torch.randn((1, 3, 1, 2, 3), generator=torch.Generator().manual_seed(19))
        batch.seed = None
    out = stages.PJiTDenoisingStage(_Predictor())(batch, args)
    # Linum sampler.py @ 9540393 with the same toy predictor and native CPU seeded noise.
    release = [144, 125, 141, 89, 124, 95, 159, 152, 166, 153, 121, 131, 174, 174, 148, 157, 151, 160]
    assert out.output.shape == (1, 3, 1, 2, 3)
    assert out.output.dtype == torch.uint8
    assert out.output.flatten().tolist() == release
    assert out.trajectory_timesteps.tolist() == [1000, 750, 500, 250]
    assert out.trajectory_latents.shape == (1, 4, 3, 1, 2, 3)
    assert torch.equal(out.trajectory_decoded[-1], out.output.float() / 255)


def test_cpu_noise_obeys_reference_seed_modulo():
    shape = (1, 3, 1, 8, 9)
    device = torch.device("cpu")
    assert torch.equal(pixel_noise(shape, 5, device, 132), pixel_noise(shape, (1 << 32) + 5, device, None))
    assert not torch.equal(pixel_noise(shape, 5, device, None), pixel_noise(shape, 6, device, None))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA")
def test_philox_noise_matches_local_cuda_launch():
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    # Hopper and datacenter Blackwell allow 2048 resident threads/SM.
    # A smaller uncapped launch also exercises the kernel on other NVIDIA SM versions.
    shape = (3, 1, 1024, 1024) if props.major in (9, 10) else (3, 1, 32, 32)
    ref_gen = torch.Generator(device).manual_seed(42)
    rng = torch.Generator(device).manual_seed(42)
    expected = torch.randn(shape, dtype=torch.float32, device=device, generator=ref_gen)
    actual = randn_as_on_sm_count(shape, rng, props.multi_processor_count)
    assert torch.equal(actual, expected)
    assert rng.get_offset() == ref_gen.get_offset()


def test_multiple_image_request_fails_before_encoding():
    args = SimpleNamespace()
    batch = ForwardBatch(data_type="image", height=32, width=32, prompt="coast", num_videos_per_prompt=2)
    with pytest.raises(ValueError, match="one image"):
        stages.PJiTTextEncodingStage(None, None).forward(batch, args)


@pytest.mark.parametrize("option, message", [("parallel", "single-device"), ("lora", "LoRA"),
                                             ("quant", "quantization")])
def test_unsupported_model_options_fail_before_loading(option, message):
    from fastvideo.pipelines.basic.pjit.pjit_pipeline import PJiTPipeline

    config = PJiTPipelineConfig()
    args = SimpleNamespace(num_gpus=1,
                           tp_size=1,
                           sp_size=1,
                           use_fsdp_inference=False,
                           lora_path=None,
                           pipeline_config=config)
    if option == "parallel":
        args.num_gpus = 2
    elif option == "lora":
        args.lora_path = "adapter"
    else:
        config.dit_config.quant_config = object()
    with pytest.raises(ValueError, match=message):
        PJiTPipeline.__new__(PJiTPipeline).load_modules(args)
