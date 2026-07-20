"""E2E overfit tests for LTX-2 fine-tuning on the modular trainer (fastvideo/train).

Parametrized over dense LTX-2.0 and LTX-2.3 distilled checkpoints plus
an LTX-2.0 NVFP4 linear-QAT case. Each case downloads the single-sample
cats dataset, preprocesses it with the matching LTX-2 VAE + Gemma text
encoder into the t2v parquet format, trains LTX2Model with FineTuneMethod
for 300 steps on 4 GPUs, and checks that the final validation video
reproduces the training clip (MS-SSIM against the preprocessed ground-truth
clip) better than the step-0 baseline.

GPU assumptions: 4 x large-memory GPUs (developed on 4x GB200 192GB;
the 18.9B-param DiT trains FSDP-sharded with fp32 master weights).
Requires the FastVideo LTX-2 distilled checkpoints (set HF_HOME to a
cache that contains them, or allow ~60GB of downloads per version).
The NVFP4-QAT case additionally requires FlashInfer FP4 support and SM100+.

Guarded by FASTVIDEO_NIGHTLY=1 so the default test suite stays fast.
"""

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from huggingface_hub import snapshot_download

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from fastvideo.tests.utils import compute_video_ssim_torchvision  # noqa: E402

NUM_GPUS_TRAINING = "4"

DATA_DIR = "data"
LOCAL_RAW_DATA_DIR = Path(DATA_DIR) / "cats"

PREPROCESSING_SCRIPT = "fastvideo/pipelines/preprocess/preprocess_ltx2_overfit.py"

_CASES = {
    "ltx2": {
        "model": "FastVideo/LTX2-Distilled-Diffusers",
        "config": "examples/train/configs/overfit_ltx2_t2v.yaml",
        "prep_dir": Path(DATA_DIR) / "ltx2_overfit_preprocessed",
        "out_dir": Path(DATA_DIR) / "outputs_ltx2_overfit",
    },
    "ltx2_3": {
        "model": "FastVideo/LTX-2.3-Distilled-Diffusers",
        "config": "examples/train/configs/overfit_ltx2_3_t2v.yaml",
        "prep_dir": Path(DATA_DIR) / "ltx2_3_overfit_preprocessed",
        "out_dir": Path(DATA_DIR) / "outputs_ltx2_3_overfit",
    },
    # NVFP4 quantization-aware training: same LTX-2.0 data, attention/FFN
    # linears forward through real FP4 GEMMs with an STE backward.
    # Requires flashinfer FP4 kernels (sm_100-class GPU).
    "ltx2_nvfp4_qat": {
        "model": "FastVideo/LTX2-Distilled-Diffusers",
        "config": "examples/train/configs/overfit_ltx2_t2v_nvfp4_qat.yaml",
        "prep_dir": Path(DATA_DIR) / "ltx2_overfit_preprocessed",
        "out_dir": Path(DATA_DIR) / "outputs_ltx2_nvfp4_qat_overfit",
        "requires_nvfp4": True,
    },
}


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Downloading raw dataset to {LOCAL_RAW_DATA_DIR}...")
    snapshot_download(
        repo_id="wlsaidhi/cats-overfit-merged",
        local_dir=str(LOCAL_RAW_DATA_DIR),
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    assert LOCAL_RAW_DATA_DIR.exists(), (
        f"Download appeared to succeed but {LOCAL_RAW_DATA_DIR} does not exist")


def run_preprocessing(case: dict):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["LTX2_OVERFIT_DATA_DIR"] = str(LOCAL_RAW_DATA_DIR)
    env["LTX2_OVERFIT_OUTPUT_DIR"] = str(case["prep_dir"])
    env["LTX2_OVERFIT_MODEL"] = case["model"]
    cmd = [sys.executable, PREPROCESSING_SCRIPT]
    print(f"Running preprocessing: {cmd} (model={case['model']})")
    subprocess.run(cmd, check=True, env=env)


def run_training(case: dict):
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    cmd = [
        "torchrun",
        "--nnodes", "1",
        "--nproc_per_node", NUM_GPUS_TRAINING,
        "-m", "fastvideo.train.entrypoint.train",
        "--config", case["config"],
        "--training.checkpoint.output_dir", str(case["out_dir"]),
        "--training.data.data_path", str(case["prep_dir"]),
        "--callbacks.validation.dataset_file",
        str(case["prep_dir"] / "validation_prompts.json"),
    ]
    print(f"Running training: {cmd}")
    subprocess.run(cmd, check=True, env=env)


def _validation_videos_by_step(out_dir: Path) -> dict[int, str]:
    pattern = os.path.join(str(out_dir), "validation_step_*_video_0.mp4")
    videos: dict[int, str] = {}
    for path in glob.glob(pattern):
        match = re.search(r"validation_step_(\d+)_", os.path.basename(path))
        if match:
            videos[int(match.group(1))] = path
    return videos


@pytest.mark.skipif(
    os.environ.get("FASTVIDEO_NIGHTLY") != "1",
    reason="nightly e2e overfit test; set FASTVIDEO_NIGHTLY=1 to run",
)
@pytest.mark.parametrize("case_id", _CASES.keys())
def test_e2e_ltx2_overfit_new_stack(case_id: str):
    case = _CASES[case_id]
    if case.get("requires_nvfp4"):
        if (not torch.cuda.is_available() or
                torch.cuda.get_device_capability(0) < (10, 0)):
            pytest.skip("NVFP4 QAT requires an SM100+ GPU")
        try:
            from flashinfer import (  # noqa: F401
                SfLayout,
                mm_fp4,
                nvfp4_quantize,
            )
        except ImportError:
            pytest.skip("NVFP4 QAT requires flashinfer with FP4 kernels")
    download_data()
    run_preprocessing(case)
    run_training(case)

    reference_video = case["prep_dir"] / "training_sample_0.mp4"
    assert reference_video.exists(), (
        f"Reference (preprocessed training clip) not found at {reference_video}")

    videos = _validation_videos_by_step(case["out_dir"])
    assert videos, f"No validation videos found under {case['out_dir']}"
    final_step = max(videos)
    final_video = videos[final_step]
    print(f"Final validation video (step {final_step}): {final_video}")

    final_mean, final_min, final_max = compute_video_ssim_torchvision(
        str(reference_video), final_video, use_ms_ssim=True)
    print(f"\n===== MS-SSIM vs training clip at step {final_step} ({case_id}) =====")
    print(f"Mean: {final_mean:.4f}  Min: {final_min:.4f}  Max: {final_max:.4f}")

    results = {"case": case_id, "final_step": final_step, "final_mean_ssim": final_mean}
    baseline_step = min(videos)
    if baseline_step != final_step:
        base_mean, _, _ = compute_video_ssim_torchvision(
            str(reference_video), videos[baseline_step], use_ms_ssim=True)
        print(f"Baseline (step {baseline_step}) mean MS-SSIM: {base_mean:.4f}")
        results["baseline_mean_ssim"] = base_mean
        assert final_mean > base_mean, (
            f"Overfitting did not improve similarity to the training clip: "
            f"step {final_step} mean SSIM {final_mean:.4f} <= "
            f"step {baseline_step} mean SSIM {base_mean:.4f}")
    print(json.dumps(results))

    assert final_mean > 0.5, (
        f"Mean MS-SSIM vs the training clip is below 0.5: {final_mean:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("FASTVIDEO_NIGHTLY", "1")
    for _case_id in _CASES:
        test_e2e_ltx2_overfit_new_stack(_case_id)
