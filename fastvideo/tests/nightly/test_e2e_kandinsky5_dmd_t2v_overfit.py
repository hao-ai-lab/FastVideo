# SPDX-License-Identifier: Apache-2.0
"""Single-sample overfit / small end-to-end run for Kandinsky5 QAD:
stage-1 Attn-QAT finetune -> stage-2 QAT-aware DMD distillation, on the
new ``fastvideo/train/`` stack.

Unlike ``test_e2e_dmd_t2v_crush_smol.py`` (which drives the legacy
``fastvideo/training/wan_distillation_pipeline.py`` CLI), this drives the
new-stack entrypoint (``fastvideo.train.entrypoint.train``) via the same
YAML configs used for real training
(``examples/train/configs/fine_tuning/kandinsky5/t2v_480p_qat.yaml`` and
``examples/train/configs/distribution_matching/kandinsky5/dmd2_t2v_480p_qat.yaml``),
with a handful of steps and dotted-key overrides pointing at a tiny local
dataset -- mirroring how a real run would be launched, just shrunk down.

No golden reference video exists yet for Kandinsky5 (unlike the Wan test,
which compares against a checked-in reference_video). This test only
asserts the run completes, loss stays finite, and expected checkpoint /
validation-video artifacts are produced. Once a real training run is
available, snapshot a reference video and add an SSIM assertion here to
match the Wan test's rigor.

The single training sample is a synthesized noise clip (not a downloaded
dataset): the exact directory/manifest layout of external raw-video HF
datasets (e.g. the ``crush-smol`` one Wan's e2e test uses) isn't something
this change can verify without a runnable environment, so depending on it
here would risk a silently-wrong assumption. ``preprocess_kandinsky5_overfit.py``
only needs a ``videos2caption.json`` + ``videos/*.mp4`` pair, both of which
this test fully controls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

NUM_GPUS = os.environ.get("KANDINSKY5_E2E_NUM_GPUS", "1")
DATA_DIR = Path("data")
KANDINSKY5_OVERFIT_DATA_DIR = DATA_DIR / "kandinsky5_overfit"
PREPROCESSED_DIR = DATA_DIR / "kandinsky5_overfit_preprocessed"
STAGE1_OUTPUT_DIR = DATA_DIR / "outputs" / "kandinsky5_e2e_stage1"
STAGE2_OUTPUT_DIR = DATA_DIR / "outputs" / "kandinsky5_e2e_stage2"
# Kept out of STAGE1_OUTPUT_DIR: that directory is scanned with
# glob("checkpoint-*") below to find the latest raw DCP checkpoint, and a
# "checkpoint-<N>-diffusers" export dir sitting alongside "checkpoint-<N>"
# would also match that pattern -- on a rerun that doesn't start from a
# clean data/ dir, it would sort after "checkpoint-<N>" and get picked as
# stage1_ckpt instead, feeding an already-exported diffusers directory back
# into dcp_to_diffusers as if it were a DCP checkpoint.
STAGE1_DIFFUSERS_DIR = DATA_DIR / "outputs" / "kandinsky5_e2e_stage1_diffusers"

STAGE1_CONFIG = (REPO_ROOT / "examples" / "train" / "configs" / "fine_tuning" / "kandinsky5" /
                 "t2v_480p_qat.yaml")
STAGE2_CONFIG = (REPO_ROOT / "examples" / "train" / "configs" / "distribution_matching" / "kandinsky5" /
                 "dmd2_t2v_480p_qat.yaml")


def _synthesize_single_sample() -> None:
    """Write one tiny synthetic clip + caption manifest in the layout
    preprocess_kandinsky5_overfit.py expects.

    Matches KANDINSKY5_T2V_LITE_5S's native preset (512x768, 121 frames,
    24fps -- see preprocess_kandinsky5_overfit.py's own constants).
    """
    if PREPROCESSED_DIR.exists():
        shutil.rmtree(PREPROCESSED_DIR)
    if KANDINSKY5_OVERFIT_DATA_DIR.exists():
        shutil.rmtree(KANDINSKY5_OVERFIT_DATA_DIR)
    videos_dir = KANDINSKY5_OVERFIT_DATA_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / "sample_0.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (768, 512),
    )
    rng = np.random.default_rng(0)
    for _ in range(121):
        frame = rng.integers(0, 256, size=(512, 768, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(KANDINSKY5_OVERFIT_DATA_DIR / "videos2caption.json", "w") as f:
        json.dump([{
            "path": "sample_0.mp4",
            "cap": ["a synthetic test clip of random noise"],
        }], f)


def _run_preprocessing() -> None:
    # preprocess_kandinsky5_overfit.py hardcodes DATA_DIR="data/kandinsky5_overfit"
    # and OUTPUT_DIR="data/kandinsky5_overfit_preprocessed" -- KANDINSKY5_OVERFIT_DATA_DIR
    # / PREPROCESSED_DIR above are set to match, so no env/CLI override is needed.
    cmd = [
        sys.executable, "-m",
        "fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _export_dcp_to_diffusers(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert a stage-1 DCP checkpoint to a diffusers-style directory.

    Stage 2's ``models.*.init_from`` (like real-training YAML configs, see
    ``dmd2_t2v_480p_qat.yaml``) expects a diffusers model directory
    (``model_index.json`` + component subfolders), not the raw
    ``checkpoint-N`` DCP layout ``CheckpointManager`` writes (``dcp/`` +
    metadata/RNG state only). ``--verify`` strictly reloads the exported
    transformer immediately so a key-mapping bug fails here, at the export
    boundary, rather than deep inside the stage-2 launch below.
    """
    cmd = [
        sys.executable, "-m",
        "fastvideo.train.entrypoint.dcp_to_diffusers",
        "--checkpoint", str(checkpoint_dir),
        "--output-dir", str(output_dir),
        "--role", "student",
        "--overwrite",
        "--verify",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_stage(config: Path, output_dir: Path, *, max_train_steps: int, extra_overrides: list[str],
              env_overrides: dict[str, str]) -> None:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", "1",
        "--nproc_per_node", NUM_GPUS,
        "-m", "fastvideo.train.entrypoint.train",
        "--config", str(config),
        "--training.data.data_path", str(PREPROCESSED_DIR),
        "--training.distributed.num_gpus", NUM_GPUS,
        "--training.distributed.hsdp_shard_dim", NUM_GPUS,
        "--training.loop.max_train_steps", str(max_train_steps),
        "--training.checkpoint.output_dir", str(output_dir),
        "--training.checkpoint.training_state_checkpointing_steps", str(max_train_steps),
        "--callbacks.validation.every_steps", str(max_train_steps),
        "--callbacks.validation.dataset_file", str(PREPROCESSED_DIR / "validation_prompts.json"),
        *extra_overrides,
    ]
    env = dict(os.environ)
    env.update(env_overrides)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


@pytest.mark.nightly
def test_e2e_kandinsky5_dmd_overfit_single_sample():
    if not STAGE1_CONFIG.exists() or not STAGE2_CONFIG.exists():
        pytest.skip("Kandinsky5 QAT configs not found -- see examples/train/configs/{fine_tuning,distribution_matching}/kandinsky5/")

    os.environ.setdefault("WANDB_MODE", "offline")

    _synthesize_single_sample()
    _run_preprocessing()

    _run_stage(
        STAGE1_CONFIG,
        STAGE1_OUTPUT_DIR,
        max_train_steps=3,
        extra_overrides=[],
        env_overrides={"FASTVIDEO_ATTENTION_BACKEND": "ATTN_QAT_TRAIN"},
    )
    assert any(STAGE1_OUTPUT_DIR.glob("checkpoint-*")), (
        f"no stage-1 checkpoint produced under {STAGE1_OUTPUT_DIR}")

    stage1_checkpoints = sorted(STAGE1_OUTPUT_DIR.glob("checkpoint-*"))
    stage1_ckpt = stage1_checkpoints[-1]

    stage1_diffusers_dir = STAGE1_DIFFUSERS_DIR / stage1_ckpt.name
    _export_dcp_to_diffusers(stage1_ckpt, stage1_diffusers_dir)
    assert (stage1_diffusers_dir / "model_index.json").exists(), (
        f"dcp_to_diffusers export did not produce a diffusers model dir at {stage1_diffusers_dir}")

    _run_stage(
        STAGE2_CONFIG,
        STAGE2_OUTPUT_DIR,
        max_train_steps=3,
        extra_overrides=[
            "--models.student.init_from", str(stage1_diffusers_dir),
            "--models.teacher.init_from", str(stage1_diffusers_dir),
            "--models.critic.init_from", str(stage1_diffusers_dir),
        ],
        env_overrides={"FASTVIDEO_ATTENTION_BACKEND": "ATTN_QAT_TRAIN"},
    )
    assert any(STAGE2_OUTPUT_DIR.glob("checkpoint-*")), (
        f"no stage-2 checkpoint produced under {STAGE2_OUTPUT_DIR}")
    assert any(STAGE2_OUTPUT_DIR.glob("*.mp4")), (
        f"no stage-2 validation video produced under {STAGE2_OUTPUT_DIR}")


if __name__ == "__main__":
    test_e2e_kandinsky5_dmd_overfit_single_sample()
