# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from torch.testing import assert_close

from fastvideo import VideoGenerator
from fastvideo.models.dits.ltx2 import (
    AudioLatentShape,
    DEFAULT_LTX2_AUDIO_CHANNELS,
    DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
    DEFAULT_LTX2_AUDIO_HOP_LENGTH,
    DEFAULT_LTX2_AUDIO_MEL_BINS,
    DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
)
from fastvideo.models.loader.component_loader import PipelineComponentLoader


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.float()
    print(
        f"[LTX2 SMOKE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f} "
        f"mean={tensor_f32.mean().item():.6f} sum={tensor_f32.sum().item():.6f}"
    )

def _truncate_debug_logs() -> None:
    for env_var in (
        "LTX2_PIPELINE_DEBUG_PATH",
        "LTX2_REFERENCE_DEBUG_PATH",
        "LTX2_PIPELINE_DEBUG_DETAIL_PATH",
        "LTX2_REFERENCE_DEBUG_DETAIL_PATH",
    ):
        log_path = os.getenv(env_var, "")
        if not log_path:
            continue
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")


def _run_audio_decode_smoke(
    diffusers_path: str,
    fastvideo_args,
    device: torch.device,
    num_frames: int,
    fps: float,
) -> None:
    audio_decoder_path = os.path.join(diffusers_path, "audio_vae")
    vocoder_path = os.path.join(diffusers_path, "vocoder")
    if not os.path.isdir(audio_decoder_path):
        pytest.skip(f"Missing LTX-2 audio decoder at {audio_decoder_path}")
    if not os.path.isdir(vocoder_path):
        pytest.skip(f"Missing LTX-2 vocoder at {vocoder_path}")

    audio_decoder = PipelineComponentLoader.load_module(
        module_name="audio_decoder",
        component_model_path=audio_decoder_path,
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    )
    vocoder = PipelineComponentLoader.load_module(
        module_name="vocoder",
        component_model_path=vocoder_path,
        transformers_or_diffusers="diffusers",
        fastvideo_args=fastvideo_args,
    )

    duration = float(num_frames) / float(fps)
    audio_shape = AudioLatentShape.from_duration(
        batch=1,
        duration=duration,
        channels=DEFAULT_LTX2_AUDIO_CHANNELS,
        mel_bins=DEFAULT_LTX2_AUDIO_MEL_BINS,
        sample_rate=DEFAULT_LTX2_AUDIO_SAMPLE_RATE,
        hop_length=DEFAULT_LTX2_AUDIO_HOP_LENGTH,
        audio_latent_downsample_factor=DEFAULT_LTX2_AUDIO_DOWNSAMPLE,
    )
    audio_dtype = next(audio_decoder.parameters()).dtype
    audio_latents = torch.randn(
        audio_shape.to_torch_shape(),
        device=device,
        dtype=audio_dtype,
    )

    with torch.no_grad():
        decoded_spec = audio_decoder(audio_latents)
        audio_wave = vocoder(decoded_spec)

    assert audio_wave.ndim == 3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="LTX-2 pipeline smoke test requires CUDA.",
)
def test_ltx2_pipeline_smoke():
    repo_root = Path(__file__).resolve().parents[3]
    debug_dir = repo_root / "ltx2_debug"
    os.environ.setdefault(
        "LTX2_PIPELINE_DEBUG_PATH",
        str(debug_dir / "fastvideo_pipeline.log"),
    )
    os.environ.setdefault(
        "LTX2_REFERENCE_DEBUG_PATH",
        str(debug_dir / "reference_pipeline.log"),
    )
    os.environ.setdefault(
        "LTX2_PIPELINE_DEBUG_DETAIL_PATH",
        str(debug_dir / "fastvideo_pipeline_detail.log"),
    )
    os.environ.setdefault(
        "LTX2_REFERENCE_DEBUG_DETAIL_PATH",
        str(debug_dir / "reference_pipeline_detail.log"),
    )
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("LTX2_REFERENCE_ATTN", "pytorch")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    _truncate_debug_logs()

    ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
    if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
        sys.path.insert(0, str(ltx_core_path))
    ltx_pipelines_path = repo_root / "LTX-2" / "packages" / "ltx-pipelines" / "src"
    if ltx_pipelines_path.exists() and str(ltx_pipelines_path) not in sys.path:
        sys.path.insert(0, str(ltx_pipelines_path))
    os.environ["PYTHONPATH"] = str(repo_root)

    diffusers_path = os.getenv("LTX2_DIFFUSERS_PATH", "converted/ltx2_diffusers")
    gemma_model_path = os.path.join(diffusers_path, "text_encoder", "gemma")
    official_path = os.getenv(
        "LTX2_OFFICIAL_PATH",
        "official_ltx_weights/ltx-2-19b-distilled.safetensors",
    )

    if not os.path.isdir(diffusers_path):
        pytest.skip(f"Missing LTX-2 diffusers repo at {diffusers_path}")
    if not os.path.isfile(os.path.join(diffusers_path, "model_index.json")):
        pytest.skip(f"Missing model_index.json in {diffusers_path}")

    if not gemma_model_path or not os.path.isdir(gemma_model_path):
        pytest.skip("Gemma weights not found in text_encoder/gemma.")
    if not os.path.isfile(official_path):
        pytest.skip(f"Missing LTX-2 official weights at {official_path}")

    try:
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        from ltx_core.model.transformer import attention as ltx_attention
    except ImportError as exc:
        pytest.skip(f"LTX-2 pipeline import failed: {exc}")
    ltx_attention.memory_efficient_attention = None
    ltx_attention.flash_attn_interface = None

    device = torch.device("cuda:0")
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers."
    negative_prompt = "low quality, blurry, distorted, artifacts, jpeg compression"
    seed = 42
    height = 64
    width = 96
    num_frames = 9
    fps = 12.0
    steps = 4
    guidance_scale = 4.0

    with tempfile.TemporaryDirectory() as tmpdir:
        latent_path = str(Path(tmpdir) / "ltx2_initial_latent.pt")

        generator = VideoGenerator.from_pretrained(
            diffusers_path,
            num_gpus=1,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            pin_cpu_memory=False,
            ltx2_vae_tiling=False,
            ltx2_initial_latent_path=latent_path,
        )
        result = generator.generate_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_path="outputs_video/ltx2_smoke",
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
        _run_audio_decode_smoke(
            diffusers_path=diffusers_path,
            fastvideo_args=generator.fastvideo_args,
            device=device,
            num_frames=num_frames,
            fps=fps,
        )

        fastvideo_out = result["samples"]
        fastvideo_out = fastvideo_out.to(device=device, dtype=torch.float32)
        _log_tensor_stats("fastvideo_video", fastvideo_out)

        ref_pipeline = TI2VidOneStagePipeline(
            checkpoint_path=official_path,
            gemma_root=gemma_model_path,
            loras=[],
            device=device,
            fp8transformer=False,
        )
        original_text_encoder = ref_pipeline.model_ledger.text_encoder
        original_transformer = ref_pipeline.model_ledger.transformer
        original_video_decoder = ref_pipeline.model_ledger.video_decoder

        def _patched_text_encoder():
            encoder = original_text_encoder()
            try:
                from ltx_core.model.transformer.attention import (  # type: ignore
                    Attention,
                    AttentionFunction,
                )
            except ImportError:
                return encoder
            if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
                if hasattr(encoder.model.config, "attn_implementation"):
                    encoder.model.config.attn_implementation = "sdpa"
                if hasattr(encoder.model.config, "_attn_implementation"):
                    encoder.model.config._attn_implementation = "sdpa"
            for module in encoder.modules():
                if isinstance(module, Attention):
                    module.attention_function = AttentionFunction.PYTORCH
            return encoder

        ref_pipeline.model_ledger.text_encoder = _patched_text_encoder
        if os.getenv("LTX2_DISABLE_VAE_NOISE", "1") == "1":
            def _patched_video_decoder():
                decoder = original_video_decoder()
                if hasattr(decoder, "decode_noise_scale"):
                    decoder.decode_noise_scale = 0.0
                return decoder

            ref_pipeline.model_ledger.video_decoder = _patched_video_decoder
        if os.getenv("LTX2_DEBUG_DETAIL", "0") == "1":
            from ..transformers.test_ltx2 import (
                _attach_block_detail_logging,
            )

            def _patched_transformer():
                model = original_transformer()
                core = getattr(model, "velocity_model", model)
                _attach_block_detail_logging(
                    core,
                    Path(os.environ["LTX2_REFERENCE_DEBUG_DETAIL_PATH"]),
                    "reference",
                    True,
                )
                return model

            ref_pipeline.model_ledger.transformer = _patched_transformer
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
                initial_video_latent_path=latent_path,
            )
            ref_chunks = list(ref_video_iter)
        ref_video = torch.cat(
            [chunk if torch.is_tensor(chunk) else torch.from_numpy(chunk) for chunk in ref_chunks],
            dim=0,
        )
        ref_video = ref_video.to(torch.float32) / 255.0
        ref_video = ref_video.permute(3, 0, 1, 2).unsqueeze(0)
        _log_tensor_stats("reference_video", ref_video)

        assert ref_video.shape == fastvideo_out.shape
        assert_close(ref_video, fastvideo_out, atol=2 / 255, rtol=1e-3)
