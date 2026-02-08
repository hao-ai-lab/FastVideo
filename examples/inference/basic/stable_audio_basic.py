#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal example: generate audio from a text prompt using Stable Audio Open.

Requires: pip install .[stable-audio] (no stable-audio-tools clone needed).

Usage:
  python examples/inference/basic/stable_audio_basic.py
  python examples/inference/basic/stable_audio_basic.py --prompt "A gentle rain" --duration 8
  python examples/inference/basic/stable_audio_basic.py --no-cpu-offload  # higher GPU utilization
"""
import argparse
import os

import numpy as np
import torch

from fastvideo import VideoGenerator


def save_audio_wav(audio: torch.Tensor, sample_rate: int, path: str) -> None:
    """Save audio tensor (B, C, T) to WAV file. Output is stereo interleaved."""
    import wave

    if audio.ndim == 3:
        audio = audio[0]
    audio_np = audio.detach().cpu().float().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    if audio_int16.ndim == 1:
        audio_int16 = audio_int16[:, None]
    num_channels = audio_int16.shape[0]
    num_frames = audio_int16.shape[1]
    frames_bytes = audio_int16.T.tobytes()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable Audio text-to-audio generation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="stabilityai/stable-audio-open-1.0",
        help="Path to model or HuggingFace model ID (e.g. stabilityai/stable-audio-open-1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful piano arpeggio",
        help="Text description of the audio to generate",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stable_audio_output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of denoising steps (default: 100)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=6.0,
        help="Classifier-free guidance scale (default: 6.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable CPU offload for higher GPU utilization (requires more VRAM)",
    )
    args = parser.parse_args()

    offload_kwargs = {}
    if args.no_cpu_offload:
        offload_kwargs = dict(
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
            vae_cpu_offload=False,
        )

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=1,
        **offload_kwargs,
    )

    result = generator.generate_audio(
        prompt=args.prompt,
        duration_seconds=args.duration,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    generator.shutdown()

    save_audio_wav(result["audio"], result["sample_rate"], args.output)
    print(f"Saved audio to {args.output}")
    print(f"  Shape: {result['audio'].shape}, sample_rate: {result['sample_rate']} Hz")
    if result.get("generation_time"):
        print(f"  Generation time: {result['generation_time']:.1f}s")


if __name__ == "__main__":
    main()
