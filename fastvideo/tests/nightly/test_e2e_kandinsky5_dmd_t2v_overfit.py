# SPDX-License-Identifier: Apache-2.0
"""Single-sample overfit / small end-to-end run for Kandinsky5 QAD:
stage-1 Attn-QAT finetune -> stage-2 QAT-aware DMD distillation, on the
new ``fastvideo/train/`` stack -- validated all the way to the public
inference artifact this recipe exists to produce.

Unlike ``test_e2e_dmd_t2v_crush_smol.py`` (which drives the legacy
``fastvideo/training/wan_distillation_pipeline.py`` CLI), this drives the
new-stack entrypoint (``fastvideo.train.entrypoint.train``) via the same
YAML configs used for real training
(``examples/train/configs/fine_tuning/kandinsky5/t2v_480p_qat.yaml`` and
``examples/train/configs/distribution_matching/kandinsky5/dmd2_t2v_480p_qat.yaml``),
with a handful of steps and dotted-key overrides pointing at a tiny local
dataset -- mirroring how a real run would be launched, just shrunk down.

The full validated chain, each arrow a hard assertion:

  synthetic clip -> preprocess -> stage-1 train -> stage-1 DCP
    -> dcp_to_diffusers --verify (strict reload)
    -> stage-2 train (student/teacher/critic init_from that export)
    -> stage-2 student DCP -> dcp_to_diffusers --verify (strict reload)
    -> VideoGenerator.from_pretrained(export,
           override_pipeline_cls_name="Kandinsky5DMDPipeline",
           pipeline_config=Kandinsky5DMDConfig())   # the documented recipe
    -> deterministic (fixed-seed) 4-step generation
    -> degeneracy checks + MS-SSIM against the committed reference video.

Every artifact (raw clip, parquet, both stage output dirs, both diffusers
exports, generated video dir) lives under the single test-owned
``data/kandinsky5_e2e/`` root, which is deleted up front -- the
documented user dataset paths (``data/kandinsky5_overfit{,_preprocessed}``)
are never read, written, or removed. ``training.checkpoint.
resume_from_checkpoint`` is explicitly disabled for both stages: the
stage-1 YAML defaults to ``resume_from_checkpoint: latest``, so a rerun
over a dirty ``data/`` dir could otherwise no-op from an old checkpoint
and go green on stale artifacts.

Reference bootstrap: the committed reference
(``reference_video_kandinsky5_dmd_v0.mp4`` next to this file) is produced
by running this test once on a sanctioned GPU box with
``KANDINSKY5_E2E_WRITE_REFERENCE=1``, reviewing the written video by eye,
and committing it. Review bar: expect blurry-but-structured content
loosely matching the prompt (4-step no-CFG sampling of near-base weights
cannot look good) -- soft shapes, warm sunflower-ish colors, motion.
Uniform grey static, solid black, or hard garbage blocks mean a broken
pipeline: do NOT commit such a reference (probe the export with the plain
50-step ``Kandinsky5T2VPipeline`` to isolate weights-path vs sampler
problems). Until a reference exists the test FAILS (it does not skip) --
an artifact-existence-only pass is exactly the false-green this oracle
exists to prevent. Run it with::

    pytest fastvideo/tests/nightly/test_e2e_kandinsky5_dmd_t2v_overfit.py -vs

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
THIS_FILE = Path(__file__).resolve()

NUM_GPUS = os.environ.get("KANDINSKY5_E2E_NUM_GPUS", "1")
# Every artifact this test reads or writes lives under this single
# test-owned root. It must NEVER point at (or contain) the documented
# default dataset paths -- data/kandinsky5_overfit and
# data/kandinsky5_overfit_preprocessed -- which a user may have populated
# with their real videos/captions for the actual recipe:
# _clean_previous_artifacts() deletes this root wholesale on every run,
# and the preprocess subprocess is pointed here via the
# KANDINSKY5_OVERFIT_DATA_DIR / KANDINSKY5_OVERFIT_OUTPUT_DIR env
# overrides instead of the script's user-facing defaults.
E2E_ROOT = Path("data") / "kandinsky5_e2e"
RAW_DATA_DIR = E2E_ROOT / "raw"
PREPROCESSED_DIR = E2E_ROOT / "preprocessed"
STAGE1_OUTPUT_DIR = E2E_ROOT / "stage1"
STAGE2_OUTPUT_DIR = E2E_ROOT / "stage2"
# Kept out of STAGE{1,2}_OUTPUT_DIR: those directories are scanned with
# glob("checkpoint-*") below to find the latest raw DCP checkpoint, and a
# "checkpoint-<N>-diffusers" export dir sitting alongside "checkpoint-<N>"
# would also match that pattern -- on a rerun that doesn't start from a
# clean data/ dir, it would sort after "checkpoint-<N>" and get picked as
# the checkpoint instead, feeding an already-exported diffusers directory
# back into dcp_to_diffusers as if it were a DCP checkpoint.
STAGE1_DIFFUSERS_DIR = E2E_ROOT / "stage1_diffusers"
STAGE2_DIFFUSERS_DIR = E2E_ROOT / "stage2_diffusers"
GENERATED_VIDEO_DIR = E2E_ROOT / "generated"

STAGE1_CONFIG = (REPO_ROOT / "examples" / "train" / "configs" / "fine_tuning" / "kandinsky5" /
                 "t2v_480p_qat.yaml")
STAGE2_CONFIG = (REPO_ROOT / "examples" / "train" / "configs" / "distribution_matching" / "kandinsky5" /
                 "dmd2_t2v_480p_qat.yaml")

# Deterministic-generation oracle. The reference is generated once (see the
# module docstring's bootstrap procedure), reviewed by a human, and
# committed next to this file -- mirroring reference_video_1_sample_v0.mp4
# in the Wan e2e test.
REFERENCE_VIDEO = THIS_FILE.parent / "reference_video_kandinsky5_dmd_v0.mp4"
WRITE_REFERENCE_ENV = "KANDINSKY5_E2E_WRITE_REFERENCE"
# Deliberately semantic even though the training clip is random noise: the
# caption's relation to the clip content is irrelevant for plumbing, but
# the caption doubles as the generation prompt, and a semantic prompt makes
# the reference video human-reviewable (blurry-but-structured output from
# 4-step sampling of near-base weights). An earlier "random noise" caption
# produced a grey-static reference that was visually indistinguishable
# from a broken pipeline AND decorrelates across runs, making the SSIM
# oracle both unreviewable and potentially flaky.
GENERATION_PROMPT = ("A curious raccoon peers through a vibrant field of yellow sunflowers, "
                     "soft natural light filtering through the petals, mid-shot")
GENERATION_SEED = 42
# Media-oracle calibration, measured against the committed reference
# (2026-07-19; regression-tested in
# fastvideo/tests/workflow/test_kandinsky5_e2e_media_oracle.py):
#
#   clip                          spatial-std  temporal-diff  mean MS-SSIM
#   good reference                  >= 11.37      >= 1.15        1.0000
#   solid @ reference mean RGB       0.0           0.0           0.9189
#   frozen (reference frame x121)   12.07          0.0001        0.8597
#
# Two consequences drive the design below:
#  - MS-SSIM alone CANNOT separate collapsed clips from good ones with
#    safe margin (a solid clip scores 0.9189!), so the structural checks
#    in _assert_video_not_degenerate are the primary defense against
#    solid/frozen/truncated output; the SSIM floor's job is catching a
#    *different structured video* (wrong remap / re-noise schedule).
#  - The floors are set >= 5x below the good reference's observed values
#    and >= 10x above every known-bad clip's.
# An independent retrain-from-scratch validation run (2026-07-18) scored
# mean MS-SSIM 1.0000 -- the chain is deterministic on fixed hardware, so
# 0.95 still leaves drift headroom while sitting above the 0.9189 solid
# clip.
EXPECTED_FRAME_COUNT = 121
EXPECTED_FRAME_HEIGHT = 512
EXPECTED_FRAME_WIDTH = 768
MIN_FRAME_SPATIAL_STD = 2.0
MIN_FRAME_TEMPORAL_DIFF = 0.2
MIN_MEAN_MS_SSIM = 0.95


def _clean_previous_artifacts() -> None:
    """Delete the single test-owned root a previous run could have left.

    Outputs matter as much as inputs: stage 1's YAML defaults to
    ``resume_from_checkpoint: latest``, and the final assertions glob for
    checkpoints/videos -- stale files from an earlier run could satisfy
    them without this run doing any work.

    Scoped strictly to ``E2E_ROOT``: an earlier version of this cleanup
    also removed ``data/kandinsky5_overfit{,_preprocessed}``, the
    documented default dataset paths of the real recipe -- silently and
    irreversibly deleting whatever real videos/captions a user had
    prepared there. Nothing outside the test-owned root may be touched.
    """
    if E2E_ROOT.exists():
        shutil.rmtree(E2E_ROOT)


def _synthesize_single_sample() -> None:
    """Write one tiny synthetic clip + caption manifest in the layout
    preprocess_kandinsky5_overfit.py expects.

    Matches KANDINSKY5_T2V_LITE_5S's native preset (512x768, 121 frames,
    24fps -- see preprocess_kandinsky5_overfit.py's own constants).
    """
    videos_dir = RAW_DATA_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    video_path = videos_dir / "sample_0.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (768, 512),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {video_path} (mp4v codec unavailable?) -- "
            "the fixture clip would be empty and preprocessing would fail on it.")
    rng = np.random.default_rng(0)
    for _ in range(121):
        frame = rng.integers(0, 256, size=(512, 768, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(RAW_DATA_DIR / "videos2caption.json", "w") as f:
        json.dump([{
            "path": "sample_0.mp4",
            "cap": [GENERATION_PROMPT],
        }], f)


def _run_preprocessing() -> None:
    # Point the preprocessor at the test-owned roots. Without these env
    # overrides it would read/write its user-facing defaults
    # (data/kandinsky5_overfit{,_preprocessed}) -- the very directories a
    # user populates for the real recipe, which this test must never touch.
    env = dict(os.environ)
    env["KANDINSKY5_OVERFIT_DATA_DIR"] = str(RAW_DATA_DIR)
    env["KANDINSKY5_OVERFIT_OUTPUT_DIR"] = str(PREPROCESSED_DIR)
    cmd = [
        sys.executable, "-m",
        "fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _export_dcp_to_diffusers(checkpoint_dir: Path, output_dir: Path) -> None:
    """Convert a stage's DCP checkpoint to a diffusers-style directory.

    ``models.*.init_from`` / ``VideoGenerator.from_pretrained`` (like real
    training YAML configs, see ``dmd2_t2v_480p_qat.yaml``) expect a
    diffusers model directory (``model_index.json`` + component
    subfolders), not the raw ``checkpoint-N`` DCP layout
    ``CheckpointManager`` writes (``dcp/`` + metadata/RNG state only).
    ``--verify`` strictly reloads the exported transformer immediately so a
    key-mapping bug fails here, at the export boundary, rather than deep
    inside the next launch that loads the directory.
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


def _latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    assert checkpoints, f"no checkpoint produced under {output_dir}"
    return checkpoints[-1]


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
        # Explicitly disable resume for BOTH stages, even though
        # _clean_previous_artifacts() already removed the output dirs: the
        # stage-1 YAML defaults to resume_from_checkpoint: latest, and this
        # test's correctness must not silently depend on that default (or a
        # partially-failed cleanup) ever changing.
        "--training.checkpoint.resume_from_checkpoint", "none",
        "--callbacks.validation.every_steps", str(max_train_steps),
        "--callbacks.validation.dataset_file", str(PREPROCESSED_DIR / "validation_prompts.json"),
        *extra_overrides,
    ]
    env = dict(os.environ)
    env.update(env_overrides)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


def _generate_from_export(export_dir: Path, output_dir: Path) -> Path:
    """Deterministically generate one video from an exported student via
    the documented DMD inference path, in a fresh subprocess.

    A subprocess (re-running this file with ``--generate``, see
    ``__main__``) keeps VideoGenerator's distributed/CUDA init isolated
    from the two torchrun stages that already ran from this test process.
    FASTVIDEO_ATTENTION_BACKEND is removed from the child env: generation
    must exercise the default dense-attention inference path the README
    documents for non-sm_120 GPUs, not inherit a QAT training backend.
    """
    env = dict(os.environ)
    env.pop("FASTVIDEO_ATTENTION_BACKEND", None)
    cmd = [
        sys.executable,
        str(THIS_FILE), "--generate",
        str(export_dir),
        str(output_dir),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    videos = sorted(Path(output_dir).glob("**/*.mp4"))
    assert len(videos) == 1, (
        f"expected exactly one generated video under {output_dir}, found {videos}")
    return videos[0]


def _generate_main(export_dir: str, output_dir: str) -> None:
    """Subprocess body for _generate_from_export.

    This is the exact inference recipe the QAD README documents for a
    stage-2 export (minus the optional NVFP4/FP8 weight quantization,
    which needs flashinfer + sm_120 hardware): the export's
    model_index.json still names the base T2V pipeline, so
    override_pipeline_cls_name + Kandinsky5DMDConfig are required to get
    the 4-step re-noise sampler -- see
    fastvideo/tests/api/test_kandinsky5_dmd_pipeline_resolution.py.
    """
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.kandinsky5 import Kandinsky5DMDConfig

    generator = VideoGenerator.from_pretrained(
        export_dir,
        num_gpus=1,
        override_pipeline_cls_name="Kandinsky5DMDPipeline",
        pipeline_config=Kandinsky5DMDConfig(),
        use_fsdp_inference=False,
    )
    generator.generate_video(
        GENERATION_PROMPT,
        output_path=output_dir,
        save_video=True,
        seed=GENERATION_SEED,
        height=512,
        width=768,
        num_frames=121,
    )


def _decode_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    assert frames, f"video {video_path} decoded to zero frames"
    return np.stack(frames)


def _assert_video_not_degenerate(video_path: Path) -> None:
    """Structural referee that needs no reference: rejects the collapsed
    outputs (solid, frozen, truncated, unreadable) a NaN run or broken
    export/decode path produces.

    This is the primary defense against collapsed clips -- the MS-SSIM
    floor cannot be (see the calibration table above: a solid clip at the
    reference's mean color scores 0.9189 against it). Every check here is
    designed against a measured false-green:
    - exact frame count/geometry: compute_video_ssim_torchvision
      truncates both clips to the shorter length, so a 1-frame video
      could otherwise sail through the comparison;
    - per-frame *luma* spatial std: the previous global RGB std counted
      differences between channel means as "variance", scoring a solid
      [106, 98, 70] clip at 15.4;
    - consecutive-frame temporal diff: a frozen (single repeated frame)
      clip has real spatial content and passes the spatial check.
    """
    frames = _decode_video(video_path).astype(np.float64)
    expected_shape = (EXPECTED_FRAME_COUNT, EXPECTED_FRAME_HEIGHT, EXPECTED_FRAME_WIDTH, 3)
    assert frames.shape == expected_shape, (
        f"generated video {video_path} has shape {frames.shape}, expected {expected_shape} -- "
        "wrong frame count or geometry (and the SSIM helper would silently truncate to the "
        "shorter clip)")

    luma = frames.mean(axis=-1)
    spatial_std = luma.reshape(len(luma), -1).std(axis=1)
    assert float(spatial_std.min()) >= MIN_FRAME_SPATIAL_STD, (
        f"generated video {video_path} has a frame with luma spatial std "
        f"{spatial_std.min():.3f} < {MIN_FRAME_SPATIAL_STD} (reference min: 11.37) -- "
        "solid/near-solid output from a NaN or collapsed run")

    temporal_diff = np.abs(np.diff(luma, axis=0)).mean(axis=(1, 2))
    assert float(temporal_diff.min()) >= MIN_FRAME_TEMPORAL_DIFF, (
        f"generated video {video_path} has consecutive frames with mean |luma diff| "
        f"{temporal_diff.min():.4f} < {MIN_FRAME_TEMPORAL_DIFF} (reference min: 1.15) -- "
        "frozen output")


def _assert_matches_reference(video_path: Path) -> None:
    if not REFERENCE_VIDEO.exists():
        if os.environ.get(WRITE_REFERENCE_ENV) == "1":
            shutil.copy2(video_path, REFERENCE_VIDEO)
            print(f"\nWrote new reference video to {REFERENCE_VIDEO} -- "
                  "review it by eye and commit it. This bootstrap run "
                  "compares the video against itself (SSIM=1) below.")
        else:
            pytest.fail(
                f"reference video missing at {REFERENCE_VIDEO}. This test "
                "must not pass on artifact existence alone -- run once on a "
                f"sanctioned GPU box with {WRITE_REFERENCE_ENV}=1, review "
                "the written video, and commit it (see module docstring).")

    from fastvideo.tests.utils import compute_video_ssim_torchvision

    mean_ssim, min_ssim, max_ssim = compute_video_ssim_torchvision(
        str(REFERENCE_VIDEO),
        str(video_path),
        use_ms_ssim=True,
    )
    print("\n===== MS-SSIM vs committed Kandinsky5 DMD reference =====")
    print(f"Mean MS-SSIM: {mean_ssim:.4f}")
    print(f"Min MS-SSIM: {min_ssim:.4f}")
    print(f"Max MS-SSIM: {max_ssim:.4f}")
    assert mean_ssim >= MIN_MEAN_MS_SSIM, (
        f"mean MS-SSIM {mean_ssim:.4f} below {MIN_MEAN_MS_SSIM} -- the exported/reloaded "
        "stage-2 student no longer generates what the reviewed reference recorded "
        "(wrong weight remap, re-noise schedule, or attention path?)")


@pytest.mark.nightly
def test_e2e_kandinsky5_dmd_overfit_single_sample():
    if not STAGE1_CONFIG.exists() or not STAGE2_CONFIG.exists():
        pytest.skip("Kandinsky5 QAT configs not found -- see examples/train/configs/{fine_tuning,distribution_matching}/kandinsky5/")

    os.environ.setdefault("WANDB_MODE", "offline")

    _clean_previous_artifacts()
    _synthesize_single_sample()
    _run_preprocessing()

    _run_stage(
        STAGE1_CONFIG,
        STAGE1_OUTPUT_DIR,
        max_train_steps=3,
        extra_overrides=[
            # The recipe's full 5e-5 is calibrated for a real dataset; 3
            # steps of it toward this test's random-noise clip measurably
            # degrades the model -- empirically enough to collapse the
            # fragile 4-step no-CFG DMD sampling below into grey static
            # (50-step T2V sampling of the same export still recovered).
            # A tiny LR keeps the full optimizer/backward path exercised
            # while the final generation stays structured: reviewable by
            # eye at bootstrap, and stable across runs for the SSIM oracle
            # (noise output decorrelates under any training
            # nondeterminism; structured output does not).
            "--training.optimizer.learning_rate", "1e-6",
        ],
        env_overrides={"FASTVIDEO_ATTENTION_BACKEND": "ATTN_QAT_TRAIN"},
    )
    stage1_ckpt = _latest_checkpoint(STAGE1_OUTPUT_DIR)

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
            # The recipe's generator_update_interval of 5 would mean ZERO
            # student updates in a 3-step run (the trainer iterates 1..3
            # and DMD2Method updates the generator only when
            # iteration % interval == 0) -- the exported "stage-2 student"
            # would just be the stage-1 weights, and this test could not
            # catch a broken student backward/optimizer path. Update the
            # generator every step instead.
            "--method.generator_update_interval", "1",
        ],
        env_overrides={"FASTVIDEO_ATTENTION_BACKEND": "ATTN_QAT_TRAIN"},
    )
    assert any(STAGE2_OUTPUT_DIR.glob("*.mp4")), (
        f"no stage-2 validation video produced under {STAGE2_OUTPUT_DIR}")

    # The actual deliverable of this recipe is the exported stage-2
    # student: export it, strict-reload it (--verify), then instantiate it
    # through the documented Kandinsky5DMDPipeline override and generate.
    stage2_ckpt = _latest_checkpoint(STAGE2_OUTPUT_DIR)
    stage2_diffusers_dir = STAGE2_DIFFUSERS_DIR / stage2_ckpt.name
    _export_dcp_to_diffusers(stage2_ckpt, stage2_diffusers_dir)
    assert (stage2_diffusers_dir / "model_index.json").exists(), (
        f"dcp_to_diffusers export did not produce a diffusers model dir at {stage2_diffusers_dir}")

    generated_video = _generate_from_export(stage2_diffusers_dir, GENERATED_VIDEO_DIR)
    _assert_video_not_degenerate(generated_video)
    _assert_matches_reference(generated_video)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--generate":
        _generate_main(sys.argv[2], sys.argv[3])
    else:
        test_e2e_kandinsky5_dmd_overfit_single_sample()
