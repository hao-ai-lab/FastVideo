#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manual parity runner: FastVideo vs official LTX2 one-stage with no CFG."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
from dataclasses import dataclass

import torch

from fastvideo import VideoGenerator


def _setup_ltx_paths(repo_root: Path) -> None:
    ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
    ltx_pipelines_path = repo_root / "LTX-2" / "packages" / "ltx-pipelines" / "src"
    if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
        sys.path.insert(0, str(ltx_core_path))
    if ltx_pipelines_path.exists() and str(ltx_pipelines_path) not in sys.path:
        sys.path.insert(0, str(ltx_pipelines_path))


def _run_reference_pipeline(
    ref_pipeline,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: float,
    steps: int,
    guidance_scale: float,
):
    from ltx_core.components.guiders import MultiModalGuiderParams

    sig = inspect.signature(ref_pipeline.__call__)
    params = set(sig.parameters)
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "enhance_prompt": False,
    }

    if "frame_rate" in params:
        kwargs["frame_rate"] = fps
    if "num_inference_steps" in params:
        kwargs["num_inference_steps"] = steps
    if "images" in params:
        kwargs["images"] = []

    if "video_guider_params" in params and "audio_guider_params" in params:
        kwargs["video_guider_params"] = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=0.0,
            rescale_scale=0.0,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[],
        )
        kwargs["audio_guider_params"] = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=0.0,
            rescale_scale=0.0,
            modality_scale=1.0,
            skip_step=0,
            stg_blocks=[],
        )
    elif "cfg_guidance_scale" in params:
        kwargs["cfg_guidance_scale"] = guidance_scale

    with torch.no_grad():
        res = ref_pipeline(**kwargs)
    return res


def _to_bcthw(video_iter) -> torch.Tensor:
    chunks = list(video_iter)
    ref_video = torch.cat(
        [
            chunk if torch.is_tensor(chunk) else torch.from_numpy(chunk)
            for chunk in chunks
        ],
        dim=0,
    )
    ref_video = ref_video.to(torch.float32) / 255.0
    return ref_video.permute(3, 0, 1, 2).unsqueeze(0)


def _to_fhwc_uint8(video_iter) -> torch.Tensor:
    chunks = list(video_iter)
    video = torch.cat(
        [
            chunk if torch.is_tensor(chunk) else torch.from_numpy(chunk)
            for chunk in chunks
        ],
        dim=0,
    )
    if video.dtype != torch.uint8:
        video = video.to(torch.float32).clamp(0, 255).to(torch.uint8)
    return video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusers-path",
                        default="converted_weights/ltx2-base")
    parser.add_argument(
        "--official-path",
        default="official_ltx_weights/ltx-2-19b-dev.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_video/ltx2_compare",
        help="Directory to save both generated mp4 files.",
    )
    return parser.parse_args()


PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic.")

official_negative_prompt = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)

@dataclass
class Setting:
    height: int
    width: int
    guidance_scale: float
    negative_prompt: str
    output_name: str
    fps: int = 24
    num_frames: int = 121
    steps: int = 40
    seed: int = 10
    prompt: str = PROMPT


SETTINGS = [
    Setting(
        height=1088,
        width=1920,
        guidance_scale=1.0,
        negative_prompt="",
        output_name="ltx2_compare_1088_1920_1.0.mp4",
    ),
    Setting(
        height=1088,
        width=1920,
        guidance_scale=3.0,
        negative_prompt="blurry, low quality, distorted, noise",
        output_name="ltx2_compare_1088_1920_3.0.mp4",
    ),
    Setting(
        height=736,
        width=1280,
        guidance_scale=1.0,
        negative_prompt="",
        output_name="ltx2_compare_736_1280_1.0.mp4",
    ),
    Setting(
        height=736,
        width=1280,
        guidance_scale=3.0,
        negative_prompt="blurry, low quality, distorted, noise",
        output_name="ltx2_compare_736_1280_3.0.mp4",
    ),
    Setting(
        height=480,
        width=640,
        guidance_scale=1.0,
        negative_prompt="",
        output_name="ltx2_compare_480_640_1.0.mp4",
    ),
    Setting(
        height=480,
        width=640,
        guidance_scale=3.0,
        negative_prompt="blurry, low quality, distorted, noise",
        output_name="ltx2_compare_480_640_3.0.mp4",
    ),
]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    # os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "FLASH_ATTN")
    # torch.backends.cuda.enable_flash_sdp(False)
    # torch.backends.cuda.enable_mem_efficient_sdp(False)
    # torch.backends.cuda.enable_math_sdp(True)

    repo_root = Path(__file__).resolve().parent
    _setup_ltx_paths(repo_root)

    from ltx_core.model.transformer import attention as ltx_attention
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
    from ltx_pipelines.utils.media_io import encode_video

    ltx_attention.memory_efficient_attention = None
    ltx_attention.flash_attn_interface = None

    diffusers_path = Path(args.diffusers_path).resolve()
    official_path = Path(args.official_path).resolve()
    gemma_root = diffusers_path / "text_encoder" / "gemma"
    device = torch.device("cuda:0")

    if not diffusers_path.is_dir():
        raise FileNotFoundError(f"Missing diffusers path: {diffusers_path}")
    if not official_path.is_file():
        raise FileNotFoundError(f"Missing official checkpoint: {official_path}")
    if not gemma_root.is_dir():
        raise FileNotFoundError(f"Missing gemma root: {gemma_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup generators
    generator = VideoGenerator.from_pretrained(
        str(diffusers_path),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        ltx2_vae_tiling=False,
    )

    ref_pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(official_path),
        gemma_root=str(gemma_root),
        loras=[],
        device=device,
        fp8transformer=False,
    )

    # Generate videos
    for setting in SETTINGS:
        generator.generate_video(
            prompt=setting.prompt,
            negative_prompt=setting.negative_prompt,
            output_path=str(output_dir / ("fv_" + setting.output_name)),
            save_video=True,
            height=setting.height,
            width=setting.width,
            num_frames=setting.num_frames,
            fps=setting.fps,
            num_inference_steps=setting.steps,
            guidance_scale=setting.guidance_scale,
            seed=setting.seed,
        )
    generator.shutdown()

    with torch.no_grad():
        for setting in SETTINGS:
            ref_video_iter, ref_audio = _run_reference_pipeline(
                ref_pipeline=ref_pipeline,
                prompt=setting.prompt,
                negative_prompt=setting.negative_prompt,
                seed=setting.seed,
                height=setting.height,
                width=setting.width,
                num_frames=setting.num_frames,
                fps=setting.fps,
                steps=setting.steps,
                guidance_scale=setting.guidance_scale,
            )
            ref_video_u8 = _to_fhwc_uint8(ref_video_iter)
            encode_video(
                video=ref_video_u8,
                fps=int(setting.fps),
                audio=ref_audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_dir / ("ref_" + setting.output_name)),
                video_chunks_number=1,
            )


if __name__ == "__main__":
    main()
