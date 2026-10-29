# SPDX-License-Identifier: Apache-2.0
"""PAN2 end-to-end pipeline parity on the public tiny checkpoint.

The FastVideo pipeline (built through the real registry and component loaders) is compared against
``tests/local_tests/pan2/reference.py::reference_generate``, which follows the vLLM-Omni ``PAN2Pipeline.forward``
step by step with transformers/diffusers components: prompt template and crop, fp32 noise, first-frame
conditioning, the uniform sigma schedule with shift 7, sequential CFG, fp32 Euler steps and the VAE decode.

Both sides run the DiT in bf16 (FastVideo's denoiser always feeds it bf16), so the comparison allows bf16 rounding.
Runs on CPU when no GPU is present. Set ``PAN2_TINY_MODEL_DIR`` to a local copy of the tiny checkpoint.
"""

from __future__ import annotations

import socket
from contextlib import ExitStack

import PIL.Image
import pytest
import torch

from fastvideo import envs
from tests.local_tests.pan2.reference import load_reference_components, reference_generate, tiny_model_dir

HEIGHT, WIDTH, NUM_FRAMES = 64, 96, 9
NUM_INFERENCE_STEPS = 3
# A large scale makes the guided steps visible through the tiny random-weight VAE.
GUIDANCE_SCALE = 30.0
SEED = 42
PROMPT = "a cat walks on the grass"
NEGATIVE_PROMPT = "blurry"
# Not the output aspect ratio, so the cover-resize and center crop are exercised.
CONDITION_IMAGE = PIL.Image.radial_gradient("L").resize((120, 64)).convert("RGB")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def pan2_runtime():
    model_dir = tiny_model_dir()
    if model_dir is None or not (model_dir / "modular_model_index.json").is_file():
        pytest.skip("wuqing157/tiny-pan2-modular-pipe is not available locally; set PAN2_TINY_MODEL_DIR")
    from fastvideo.configs.pipelines.pan2 import PAN2PipelineConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    import fastvideo.distributed.parallel_state as parallel_state
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.pipelines import build_pipeline

    with ExitStack() as stack:
        stack.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
        monkeypatch = stack.enter_context(pytest.MonkeyPatch.context())
        if not torch.cuda.is_available():
            # Reach modules that imported the device helper by name on a GPU-less Linux host.
            def _cpu_device() -> torch.device:
                return torch.device("cpu")

            monkeypatch.setattr(parallel_state.get_local_torch_device, "__code__", _cpu_device.__code__)
        for name, value in (("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(_free_port())), ("RANK", "0"),
                            ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
            stack.enter_context(envs.override_external(name, value))
        stack.callback(cleanup_dist_env_and_memory)
        maybe_init_distributed_environment_and_model_parallel(1, 1)

        # The parity checkpoint has random weights, so its videos are noise: compare without the guardrails.
        pipeline_config = PAN2PipelineConfig(dit_precision="bf16",
                                             vae_precision="fp32",
                                             text_encoder_precisions=("fp32", ),
                                             enable_guardrails=False)
        args = FastVideoArgs(model_path=str(model_dir),
                             num_gpus=1,
                             pipeline_config=pipeline_config,
                             dit_cpu_offload=False,
                             dit_layerwise_offload=False,
                             text_encoder_cpu_offload=False,
                             vae_cpu_offload=False,
                             use_fsdp_inference=False)
        pipeline = build_pipeline(args)
        device = parallel_state.get_local_torch_device()
        yield pipeline, args, model_dir, device


def _fastvideo_generate(pipeline, args, image, guidance_scale: float = GUIDANCE_SCALE):
    from fastvideo.pipelines import ForwardBatch

    batch = ForwardBatch(data_type="video",
                         prompt=PROMPT,
                         negative_prompt=NEGATIVE_PROMPT,
                         seed=SEED,
                         height=HEIGHT,
                         width=WIDTH,
                         num_frames=NUM_FRAMES,
                         num_inference_steps=NUM_INFERENCE_STEPS,
                         guidance_scale=guidance_scale,
                         pil_image=image)
    with torch.inference_mode():
        batch = pipeline.forward(batch, args)
    return batch.latents.float().cpu(), batch.output.float().cpu()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().clamp_min(1e-12)
    return float(10 * torch.log10(1.0 / mse))


@pytest.mark.parametrize(
    ("image", "skip_first_latent_cfg"),
    [(None, True), (CONDITION_IMAGE, True), (CONDITION_IMAGE, False)],
    ids=["t2v", "i2v", "i2v-guided-first-frame"],
)
def test_pan2_pipeline_matches_reference(pan2_runtime, monkeypatch, image, skip_first_latent_cfg):
    pipeline, args, model_dir, device = pan2_runtime
    monkeypatch.setattr(args.pipeline_config, "i2v_skip_first_latent_cfg", skip_first_latent_cfg)
    latents, video = _fastvideo_generate(pipeline, args, image)

    components = load_reference_components(model_dir, torch.bfloat16)
    components = {
        name: module.to(device) if isinstance(module, torch.nn.Module) else module
        for name, module in components.items()
    }
    ref_latents, ref_video = reference_generate(components,
                                                prompt=PROMPT,
                                                negative_prompt=NEGATIVE_PROMPT,
                                                image=image,
                                                height=HEIGHT,
                                                width=WIDTH,
                                                num_frames=NUM_FRAMES,
                                                num_inference_steps=NUM_INFERENCE_STEPS,
                                                guidance_scale=GUIDANCE_SCALE,
                                                seed=SEED,
                                                dit_dtype=torch.bfloat16,
                                                device=device,
                                                skip_first_latent_cfg=skip_first_latent_cfg)
    ref_latents, ref_video = ref_latents.float().cpu(), ref_video.float().cpu()

    assert latents.dtype == torch.float32
    assert latents.shape == ref_latents.shape == (1, 48, 3, HEIGHT // 16, WIDTH // 16)
    assert video.shape == ref_video.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH)
    latent_rel = float((latents - ref_latents).norm() / ref_latents.norm())
    cosine = float(torch.nn.functional.cosine_similarity(latents.flatten(), ref_latents.flatten(), dim=0))
    psnr = _psnr(video, ref_video)
    print(f"latents rel_l2={latent_rel:.3e} cosine={cosine:.6f}; video psnr={psnr:.1f} dB")
    assert latent_rel < 2e-2
    assert cosine > 0.9995
    assert psnr > 40.0


def test_pan2_image_conditioning_and_guidance_change_every_frame(pan2_runtime):
    pipeline, args, _, _ = pan2_runtime
    _, t2v = _fastvideo_generate(pipeline, args, None)
    _, i2v = _fastvideo_generate(pipeline, args, CONDITION_IMAGE)
    _, unguided = _fastvideo_generate(pipeline, args, None, guidance_scale=1.0)

    def per_frame_max_diff(a, b):
        return (a - b).abs().amax(dim=(0, 1, 3, 4))

    assert per_frame_max_diff(t2v, i2v).max() > 1e-3
    assert per_frame_max_diff(t2v, unguided).min() > 1e-3
