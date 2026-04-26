"""SSIM regression test template for FastVideo model ports.

Copy to fastvideo/tests/ssim/test_<model_name>_ssim.py and fill in the TODOs.

Requires:
    pip install scikit-image imageio
    GPU with enough VRAM to run the pipeline.

Reference video is generated once with the official implementation and stored
in assets/videos/<model_name>_reference.mp4.  The test generates a new video
with the FastVideo pipeline using identical prompt + seed, then computes
per-frame SSIM.

Run:
    pytest fastvideo/tests/ssim/test_<model_name>_ssim.py -vs
"""
import os
from pathlib import Path

import numpy as np
import pytest

REFERENCE_VIDEO = Path("assets/videos/<model_name>_reference.mp4")
PROMPT = "TODO: insert the exact prompt used to generate the reference video"
SEED = 42
SSIM_THRESHOLD = 0.85  # adjust per model; 0.85 is conservative for T2V

# TODO: replace with actual model path (HF id or local directory)
MODEL_PATH = "fastvideo/<model_name>"  # or "official_weights/<model_name>/"


def load_video_frames(path: str | Path) -> np.ndarray:
    """Load an MP4 into a float32 array of shape (T, H, W, C) in [0, 1]."""
    import imageio.v3 as iio
    frames = iio.imread(str(path), plugin="pyav")  # (T, H, W, C) uint8
    return frames.astype(np.float32) / 255.0


def compute_ssim_sequence(ref: np.ndarray, gen: np.ndarray) -> list[float]:
    """Return per-frame SSIM between two (T, H, W, C) float32 arrays."""
    from skimage.metrics import structural_similarity as ssim
    assert ref.shape == gen.shape, (
        f"Shape mismatch: reference {ref.shape} vs generated {gen.shape}"
    )
    scores = []
    for t in range(ref.shape[0]):
        score = ssim(ref[t], gen[t], data_range=1.0, channel_axis=-1)
        scores.append(float(score))
    return scores


@pytest.mark.gpu
@pytest.mark.skipif(
    not REFERENCE_VIDEO.exists(),
    reason=f"Reference video not found: {REFERENCE_VIDEO}. "
           "Generate it once with the official repo and save to that path.",
)
def test_ssim_against_official():
    """FastVideo output must match official reference video within SSIM threshold.

    GPU requirement: TODO (e.g. "Requires ≥24 GB VRAM").
    """
    # TODO: import and instantiate the FastVideo pipeline
    # from fastvideo import VideoGenerator
    # from fastvideo.configs.sample import SamplingParam
    #
    # generator = VideoGenerator.from_pretrained(MODEL_PATH, num_gpus=1)
    # sampling = SamplingParam.from_pretrained(MODEL_PATH)
    # sampling.seed = SEED
    # TODO: set resolution / frame count to match reference video

    # generated_path = "outputs/<model_name>_ssim_test.mp4"
    # generator.generate_video(PROMPT, sampling_param=sampling,
    #                          output_path="outputs", save_video=True)

    # ref_frames = load_video_frames(REFERENCE_VIDEO)
    # gen_frames = load_video_frames(generated_path)
    # scores = compute_ssim_sequence(ref_frames, gen_frames)
    # mean_ssim = float(np.mean(scores))
    #
    # print(f"SSIM: mean={mean_ssim:.4f}  min={min(scores):.4f}  max={max(scores):.4f}")
    # assert mean_ssim >= SSIM_THRESHOLD, (
    #     f"SSIM {mean_ssim:.4f} < threshold {SSIM_THRESHOLD}"
    # )
    raise NotImplementedError(
        "Fill in the VideoGenerator call and uncomment assertions above."
    )


# ---------------------------------------------------------------------------
# Utility: generate the reference video from the official repo (run once).
# ---------------------------------------------------------------------------
def generate_reference_video():
    """Run this once to create assets/videos/<model_name>_reference.mp4.

    Replace the body with the official repo's inference call.
    """
    # TODO: call official inference script here
    # import subprocess
    # subprocess.run([
    #     "python", "<model_name>/inference.py",
    #     "--prompt", PROMPT,
    #     "--seed", str(SEED),
    #     "--output", str(REFERENCE_VIDEO),
    # ], check=True)
    raise NotImplementedError("Fill in official inference call.")


if __name__ == "__main__":
    generate_reference_video()
