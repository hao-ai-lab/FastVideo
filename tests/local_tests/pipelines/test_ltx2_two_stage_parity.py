# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch
from torch.testing import assert_close

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fastvideo import VideoGenerator


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.float()
    print(
        f"[LTX2 TWO STAGE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f} sum={tensor_f32.sum().item():.6f}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LTX-2 two-stage parity test requires CUDA.",
)
def test_ltx2_two_stage_parity():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
    if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
        sys.path.insert(0, str(ltx_core_path))
    ltx_pipelines_path = repo_root / "LTX-2" / "packages" / "ltx-pipelines" / "src"
    if ltx_pipelines_path.exists() and str(ltx_pipelines_path) not in sys.path:
        sys.path.insert(0, str(ltx_pipelines_path))

    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("LTX2_REFERENCE_ATTN", "pytorch")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    diffusers_path = os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    gemma_model_path = os.path.join(diffusers_path, "text_encoder", "gemma")
    official_path = os.getenv(
        "LTX2_OFFICIAL_PATH",
        "official_ltx_weights/ltx-2-19b-distilled.safetensors",
    )
    upsampler_official_path = os.getenv(
        "LTX2_UPSAMPLER_OFFICIAL_PATH",
        "official_ltx_weights/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    )
    upsampler_path = os.getenv(
        "LTX2_UPSAMPLER_PATH",
        "converted/ltx2_spatial_upscaler",
    )
    lora_path = os.getenv(
        "LTX2_REFINER_LORA_PATH",
        "official_ltx_weights/ltx-2-19b-distilled-lora-384.safetensors",
    )

    if not os.path.isdir(diffusers_path):
        pytest.skip(f"Missing LTX-2 diffusers repo at {diffusers_path}")
    if not os.path.isfile(os.path.join(diffusers_path, "model_index.json")):
        pytest.skip("Missing model_index.json in diffusers path")
    if not gemma_model_path or not os.path.isdir(gemma_model_path):
        pytest.skip("Gemma weights not found in text_encoder/gemma.")
    if not os.path.isfile(official_path):
        pytest.skip(f"Missing LTX-2 official weights at {official_path}")
    if not os.path.isfile(upsampler_official_path):
        pytest.skip(f"Missing official upsampler at {upsampler_official_path}")
    if not os.path.isdir(upsampler_path):
        pytest.skip(f"Missing FastVideo upsampler at {upsampler_path}")
    if not os.path.isfile(lora_path):
        pytest.skip(f"Missing distilled LoRA at {lora_path}")

    try:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_core.model.transformer import attention as ltx_attention
        from ltx_core.model.transformer.attention import Attention, AttentionFunction
    except ImportError as exc:
        pytest.skip(f"LTX-2 pipeline import failed: {exc}")

    # Force reference attention to use PyTorch SDPA
    ltx_attention.memory_efficient_attention = None
    ltx_attention.flash_attn_interface = None

    device = torch.device("cuda:0")
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers."
    negative_prompt = "low quality, blurry, distorted, artifacts, jpeg compression"
    seed = 42
    height = 64
    width = 64
    num_frames = 9
    fps = 12.0
    steps = 4
    guidance_scale = 4.0

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    def _env_flag(name: str) -> bool:
        return os.getenv(name, "0").lower() in ("1", "true", "yes")

    debug_kwargs = {}
    if _env_flag("FASTVIDEO_DEBUG_STAGE_SUMS"):
        debug_kwargs["debug_stage_sums"] = True
        debug_kwargs["debug_stage_sums_path"] = os.getenv(
            "FASTVIDEO_DEBUG_STAGE_SUMS_PATH")
    if _env_flag("FASTVIDEO_DEBUG_MODULE_SUMS"):
        debug_kwargs["debug_module_sums"] = True
        debug_kwargs["debug_module_sums_path"] = os.getenv(
            "FASTVIDEO_DEBUG_MODULE_SUMS_PATH")
        include = os.getenv("FASTVIDEO_DEBUG_MODULE_SUMS_INCLUDE")
        exclude = os.getenv("FASTVIDEO_DEBUG_MODULE_SUMS_EXCLUDE")
        if include:
            debug_kwargs["debug_module_sums_include"] = include.split(",")
        if exclude:
            debug_kwargs["debug_module_sums_exclude"] = exclude.split(",")
    if _env_flag("FASTVIDEO_DEBUG_MODEL_SUMS"):
        debug_kwargs["debug_model_sums"] = True
        debug_kwargs["debug_model_sums_path"] = os.getenv(
            "FASTVIDEO_DEBUG_MODEL_SUMS_PATH")
    if _env_flag("FASTVIDEO_DEBUG_MODEL_DETAIL"):
        debug_kwargs["debug_model_detail"] = True
        debug_kwargs["debug_model_detail_path"] = os.getenv(
            "FASTVIDEO_DEBUG_MODEL_DETAIL_PATH")

    ref_debug = _env_flag("FASTVIDEO_DEBUG_REF_SUMS")
    ref_debug_path = os.getenv("FASTVIDEO_DEBUG_REF_SUMS_PATH")

    def _ref_log(line: str) -> None:
        if ref_debug_path:
            os.makedirs(os.path.dirname(ref_debug_path), exist_ok=True)
            with open(ref_debug_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            print(line)

    with tempfile.TemporaryDirectory() as tmpdir:
        # FastVideo two-stage
        generator = VideoGenerator.from_pretrained(
            diffusers_path,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
            ltx2_vae_tiling=False,
            ltx2_refine_enabled=True,
            ltx2_refine_upsampler_path=upsampler_path,
            ltx2_refine_lora_path=lora_path,
            ltx2_refine_num_inference_steps=3,
            ltx2_refine_guidance_scale=1.0,
            ltx2_refine_add_noise=True,
            **debug_kwargs,
        )
        # Disable decoder noise for parity if possible.
        if hasattr(generator, "executor") and hasattr(generator.executor, "pipeline"):
            pipeline = generator.executor.pipeline
            if hasattr(pipeline, "modules") and "vae" in pipeline.modules:
                decoder = getattr(pipeline.modules["vae"], "decoder", None)
                if decoder is not None and hasattr(decoder, "decode_noise_scale"):
                    decoder.decode_noise_scale = 0.0

        result = generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path=os.path.join(tmpdir, "fastvideo_two_stage"),
            save_video=False,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        generator.shutdown()

        fastvideo_out = result["samples"].to(dtype=torch.float32).cpu()
        _log_tensor_stats("fastvideo_video", fastvideo_out)
        del result
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Reference two-stage
        if ref_debug:
            from ltx_pipelines.utils import helpers as ltx_helpers
            import ltx_pipelines.ti2vid_two_stages as ltx_two_stage
            from ltx_core.model import upsampler as ltx_upsampler
            from ltx_core.tools import AudioLatentTools, VideoLatentTools
            from ltx_core.text_encoders import gemma as ltx_gemma

            original_denoise = ltx_helpers.denoise_audio_video
            original_upsample = ltx_upsampler.upsample_video
            original_create_noised_state = ltx_helpers.create_noised_state
            original_encode_text = ltx_gemma.encode_text

            def _ref_sum(tensor: torch.Tensor | None) -> float:
                if tensor is None:
                    return 0.0
                return float(tensor.detach().sum(dtype=torch.float32).item())

            def _wrapped_upsample(*args, **kwargs):
                out = original_upsample(*args, **kwargs)
                _ref_log(
                    f"reference:upsample_latent_sum={_ref_sum(out):.6f} "
                    f"shape={tuple(out.shape)}")
                return out

            def _wrapped_denoise(
                *,
                output_shape,
                conditionings,
                noiser,
                sigmas,
                stepper,
                denoising_loop_fn,
                components,
                dtype,
                device,
                noise_scale=1.0,
                initial_video_latent=None,
                initial_audio_latent=None,
                skip_video_noiser=False,
                skip_audio_noiser=False,
            ):
                stage = "stage2" if output_shape.width == width else "stage1"
                _ref_log(
                    f"reference:{stage}:noise_scale={float(noise_scale):.6f} "
                    f"init_video_sum={_ref_sum(initial_video_latent):.6f} "
                    f"init_audio_sum={_ref_sum(initial_audio_latent):.6f}")
                video_state, audio_state = original_denoise(
                    output_shape=output_shape,
                    conditionings=conditionings,
                    noiser=noiser,
                    sigmas=sigmas,
                    stepper=stepper,
                    denoising_loop_fn=denoising_loop_fn,
                    components=components,
                    dtype=dtype,
                    device=device,
                    noise_scale=noise_scale,
                    initial_video_latent=initial_video_latent,
                    initial_audio_latent=initial_audio_latent,
                    skip_video_noiser=skip_video_noiser,
                    skip_audio_noiser=skip_audio_noiser,
                )
                _ref_log(
                    f"reference:{stage}:video_latent_sum={_ref_sum(video_state.latent):.6f} "
                    f"audio_latent_sum={_ref_sum(audio_state.latent):.6f}")
                return video_state, audio_state

            def _wrapped_create_noised_state(
                tools,
                conditionings,
                noiser,
                dtype,
                device,
                noise_scale=1.0,
                initial_latent=None,
                skip_noiser=False,
            ):
                state = original_create_noised_state(
                    tools=tools,
                    conditionings=conditionings,
                    noiser=noiser,
                    dtype=dtype,
                    device=device,
                    noise_scale=noise_scale,
                    initial_latent=initial_latent,
                    skip_noiser=skip_noiser,
                )
                stage = "stage2" if initial_latent is not None else "stage1"
                kind = "video" if isinstance(tools, VideoLatentTools) else "audio"
                _ref_log(
                    f"reference:{stage}:{kind}:noised_latent_sum={_ref_sum(state.latent):.6f} "
                    f"noise_scale={float(noise_scale):.6f}")
                return state

            def _wrapped_encode_text(text_encoder, prompts):
                context_p, context_n = original_encode_text(text_encoder, prompts=prompts)
                v_context_p, a_context_p = context_p
                v_context_n, a_context_n = context_n
                _ref_log(
                    "reference:text:"
                    f"v_pos_sum={_ref_sum(v_context_p):.6f} "
                    f"a_pos_sum={_ref_sum(a_context_p):.6f} "
                    f"v_neg_sum={_ref_sum(v_context_n):.6f} "
                    f"a_neg_sum={_ref_sum(a_context_n):.6f}")
                return context_p, context_n

            ltx_helpers.denoise_audio_video = _wrapped_denoise
            ltx_upsampler.upsample_video = _wrapped_upsample
            ltx_helpers.create_noised_state = _wrapped_create_noised_state
            ltx_two_stage.denoise_audio_video = _wrapped_denoise
            ltx_two_stage.upsample_video = _wrapped_upsample
            ltx_gemma.encode_text = _wrapped_encode_text
            ltx_two_stage.encode_text = _wrapped_encode_text

        ref_pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=official_path,
            distilled_lora=[
                LoraPathStrengthAndSDOps(
                    lora_path,
                    1.0,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ],
            spatial_upsampler_path=upsampler_official_path,
            gemma_root=gemma_model_path,
            loras=[],
            device=device,
            fp8transformer=False,
        )

        original_text_encoder = ref_pipeline.stage_1_model_ledger.text_encoder
        original_transformer = ref_pipeline.stage_1_model_ledger.transformer
        original_video_decoder = ref_pipeline.stage_2_model_ledger.video_decoder

        def _patched_text_encoder():
            encoder = original_text_encoder()
            if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
                if hasattr(encoder.model.config, "attn_implementation"):
                    encoder.model.config.attn_implementation = "sdpa"
                if hasattr(encoder.model.config, "_attn_implementation"):
                    encoder.model.config._attn_implementation = "sdpa"
            for module in encoder.modules():
                if isinstance(module, Attention):
                    module.attention_function = AttentionFunction.PYTORCH
            return encoder

        def _patched_video_decoder():
            decoder = original_video_decoder()
            if hasattr(decoder, "decode_noise_scale"):
                decoder.decode_noise_scale = 0.0
            return decoder

        ref_pipeline.stage_1_model_ledger.text_encoder = _patched_text_encoder
        ref_pipeline.stage_2_model_ledger.video_decoder = _patched_video_decoder
        ref_pipeline.stage_1_model_ledger.transformer = original_transformer

        with torch.no_grad():
            ref_video_iter, _ = ref_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=steps,
                cfg_guidance_scale=guidance_scale,
                images=[],
                enhance_prompt=False,
            )
            ref_chunks = list(ref_video_iter)
        ref_video = torch.cat(
            [chunk if torch.is_tensor(chunk) else torch.from_numpy(chunk) for chunk in ref_chunks],
            dim=0,
        )
        ref_video = ref_video.to(torch.float32) / 255.0
        ref_video = ref_video.permute(3, 0, 1, 2).unsqueeze(0)
        ref_video = ref_video.cpu()
        _log_tensor_stats("reference_video", ref_video)

        assert ref_video.shape == fastvideo_out.shape
        assert_close(ref_video, fastvideo_out, atol=2 / 255, rtol=1e-3)
