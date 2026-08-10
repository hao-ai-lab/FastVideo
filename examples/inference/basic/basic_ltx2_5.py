# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 (sft/base) text-to-video from a single-file bundle.

Loads LTX-2.5 from a single-file bundle: ONE ``.safetensors`` file carrying
every component (transformer, video VAE, audio VAE, vocoder, text
projection), so ``--model-path`` takes the bundle FILE, not a repo
directory. (The split-pack Hugging Face release loads through the standard
repo path instead.)

The Gemma text encoder is NOT in the bundle: pass its root directory with
``--gemma-root`` (or set ``FASTVIDEO_LTX_ENCODER_ROOT``). Use the encoder
root shipped WITH this transformer variant -- the published roots share
weights but differ in prompt templating, so a mismatched root silently
changes prompting.

A non-distilled bundle resolves to the standard ``ltx2_base`` preset
(40 steps, CFG 3.0 with STG); run without sampling flags to use it as-is.
For the 8-step distilled recipe, see ``basic_ltx2_5_distilled.py``.

Audio: the bundle carries an audio VAE + vocoder, so generated videos get
an audio track. A bundle that declares no audio decoder skips audio
decoding and still produces video.

Decoder caveat: bundles that declare a *diffusion* VAE decoder
(``CausalDiffusionVAE``) are not implemented yet -- loading such a bundle
currently fails at VAE build time with an unsupported-architecture error
(no classic-decoder fallback is wired). Bundles with the classic
``CausalVideoAutoencoder`` decode end to end.
"""
import argparse
import os

from fastvideo import VideoGenerator

PROMPT = ("A warm sunny backyard. The camera starts in a tight cinematic close-up "
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

# Sampling flags default to None: unset flags are NOT passed to
# `generate_video`, so the bundle's preset supplies them via
# `SamplingParam.from_pretrained` (ltx2_base: 40 steps, cfg 3.0, 512x768,
# 121 frames).
_SAMPLING_FLAGS = ("height", "width", "num_frames", "num_inference_steps", "guidance_scale", "seed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LTX-2.5 (sft/base) inference from a single-file bundle.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the LTX-2.5 bundle (.safetensors FILE).",
    )
    parser.add_argument(
        "--gemma-root",
        default=None,
        help="Gemma text-encoder root directory. Must be the root paired with "
        "this transformer variant; the roots differ in prompt templating.",
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output-path", default="outputs_video/ltx2_5/output_ltx2_5_t2v.mp4")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def sampling_overrides(args: argparse.Namespace) -> dict:
    """Only sampling flags the user actually passed; the preset supplies the rest."""
    return {name: getattr(args, name) for name in _SAMPLING_FLAGS if getattr(args, name) is not None}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gemma_root:
        os.environ["FASTVIDEO_LTX_ENCODER_ROOT"] = args.gemma_root

    generator = VideoGenerator.from_pretrained(
        args.model_path,
        num_gpus=args.num_gpus,
    )
    generator.generate_video(
        prompt=args.prompt,
        output_path=args.output_path,
        save_video=True,
        **sampling_overrides(args),
    )
    generator.shutdown()


if __name__ == "__main__":
    main()
