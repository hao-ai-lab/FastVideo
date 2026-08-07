# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax H3 joint audio-video fine-tuning."""

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3Config
from fastvideo.dataset.parquet_dataset_map_style import (
    DP_SP_BatchSampler,
    get_parquet_files_and_length,
)
from fastvideo.models.dits.minimax_h3 import MiniMaxH3RotaryPosEmbed, MiniMaxH3Transformer3DModel
from fastvideo.pipelines import TrainingBatch
from fastvideo.train.methods.fine_tuning.finetune import _compute_finetune_loss_map
from fastvideo.train.models.minimax_h3 import MiniMaxH3Model
from fastvideo.train.models.minimax_h3.minimax_h3 import shift_noise_amount
from fastvideo.train.utils.config import load_run_config

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "minimax_h3_t2va_min.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXPERIMENT_CONFIG = _REPO_ROOT / "examples/train/configs/overfit_minimax_h3_t2va.yaml"
_VALIDATION_PROMPTS = (
    _REPO_ROOT / "examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json"
)
_EXPECTED_VALIDATION_CAPTIONS = [
    (
        "A large metal cylinder is seen pressing down on a pile of Oreo cookies, "
        "flattening them as if they were under a hydraulic press."
    ),
    (
        "A large metal cylinder is seen compressing colorful clay into a compact shape, "
        "demonstrating the power of a hydraulic press."
    ),
    (
        "A large metal cylinder is seen pressing down on a pile of colorful candies, "
        "flattening them as if they were under a hydraulic press. The candies are crushed "
        "and broken into small pieces, creating a mess on the table."
    ),
]


class _IdentityJointTransformer:
    """Return packed H3 inputs as deterministic velocity predictions."""

    patch_size = (1, 2, 2)

    def __call__(self, **kwargs):
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


@pytest.mark.parametrize(
    ("shift", "expected_midpoint"),
    [(12.0, 12.0 / 13.0), (3.0, 0.75)],
)
def test_shift_noise_amount_matches_h3_scheduler(shift: float, expected_midpoint: float) -> None:
    """Verify that training samples use the scheduler's rational noise shift."""
    base = torch.tensor([0.0, 0.5, 1.0])

    shifted = shift_noise_amount(base, shift)

    torch.testing.assert_close(shifted, torch.tensor([0.0, expected_midpoint, 1.0]))


def test_multimodal_finetune_loss_weights_modalities_equally() -> None:
    """Verify that each modality contributes its independently averaged MSE."""
    clean_video = torch.zeros(1, 1, 1, 1, 1)
    video_noise = torch.ones_like(clean_video)
    video_prediction = torch.full_like(clean_video, 2.0)
    clean_audio = torch.zeros(1, 2, 1, 1)
    audio_noise = torch.ones_like(clean_audio)
    audio_prediction = torch.full_like(clean_audio, 3.0)
    batch = TrainingBatch(
        audio_latents=clean_audio,
        audio_noisy_model_input=torch.zeros_like(clean_audio),
        audio_noise=audio_noise,
        audio_sigmas=torch.ones(1, 1, 1, 1),
    )

    losses = _compute_finetune_loss_map(
        (video_prediction, audio_prediction),
        clean_video,
        torch.zeros_like(clean_video),
        video_noise,
        torch.ones(1, 1, 1, 1, 1),
        batch,
        precondition_outputs=False,
    )

    torch.testing.assert_close(losses["video_finetune_loss"], torch.tensor(1.0))
    torch.testing.assert_close(losses["audio_finetune_loss"], torch.tensor(4.0))
    torch.testing.assert_close(losses["total_loss"], torch.tensor(5.0))
    assert losses["finetune_loss"] is losses["total_loss"]


def test_single_tensor_finetune_loss_preserves_video_contract() -> None:
    """Verify that video-only model plugins retain their loss keys and target."""
    clean_video = torch.zeros(1, 1, 1, 1, 1)
    video_noise = torch.ones_like(clean_video)

    losses = _compute_finetune_loss_map(
        video_noise,
        clean_video,
        torch.zeros_like(clean_video),
        video_noise,
        torch.ones(1, 1, 1, 1, 1),
        TrainingBatch(),
        precondition_outputs=False,
    )

    assert set(losses) == {"total_loss", "finetune_loss"}
    torch.testing.assert_close(losses["total_loss"], torch.tensor(0.0))


def test_h3_training_fixture_resolves_joint_contract() -> None:
    """Verify that YAML parsing selects the H3 T2VA data and model contract."""
    config = load_run_config(str(_FIXTURE))

    assert config.models["student"]["_target_"] == ("fastvideo.train.models.minimax_h3.MiniMaxH3Model")
    assert config.training.data.preprocessed_data_type == "t2va"
    assert config.training.pipeline_config is not None
    assert config.training.pipeline_config.text_encoder_configs[0].arch_config.text_len == 1024
    assert config.training.pipeline_config.dit_config.uniform_parameter_dtype is False
    assert MiniMaxH3Model.__name__ == "MiniMaxH3Model"


def test_h3_uniform_parameter_dtype_uses_fsdp_dtype() -> None:
    """Verify that training can select one FSDP-compatible parameter dtype."""
    model = cast(
        MiniMaxH3Transformer3DModel,
        SimpleNamespace(
            config=MiniMaxH3Config(uniform_parameter_dtype=True),
            _keep_in_fp32_modules=MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
        ),
    )

    parameter_dtype = MiniMaxH3Transformer3DModel._get_parameter_dtype(
        model,
        "proj_in.weight",
        torch.bfloat16,
    )

    assert parameter_dtype == torch.bfloat16


def test_h3_materializes_rotary_frequencies_on_loader_device() -> None:
    """Verify that checkpoint loading moves analytic rotary state to the model device."""
    model = cast(
        MiniMaxH3Transformer3DModel,
        SimpleNamespace(
            config=MiniMaxH3Config(),
            rope=SimpleNamespace(
                inv_freq=torch.ones(1),
                _buffers={"inv_freq": torch.ones(1)},
            ),
        ),
    )

    MiniMaxH3Transformer3DModel.materialize_non_persistent_buffers(
        model,
        device=torch.device("meta"),
    )

    assert model.rope._buffers["inv_freq"].is_meta


def test_h3_rotary_frequencies_follow_position_device_after_offload() -> None:
    """Verify that rotary state follows position IDs after training-state offload."""
    rotary_embedding = MiniMaxH3RotaryPosEmbed(rope_freq_dim=4, rope_theta=10_000.0)

    cosine, sine = rotary_embedding(torch.zeros(2, 3, device="meta"))

    assert cosine.is_meta
    assert sine.is_meta


def test_h3_plugin_prepares_and_restores_joint_latent_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify H3 batch preparation, packing, and output restoration together."""
    monkeypatch.setattr(MiniMaxH3Model, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.training_config = SimpleNamespace(
        data=SimpleNamespace(
            num_latent_t=2,
            num_frames=5,
            num_height=64,
            num_width=64,
        ),
        distributed=SimpleNamespace(sp_size=1),
    )
    model.transformer = _IdentityJointTransformer()
    model.sp_group = None
    raw_batch = {
        "vae_latent": torch.zeros(1, 24, 2, 4, 4),
        "audio_latent": torch.zeros(1, 2, 32, 8),
        "text_embedding": torch.zeros(1, 4, 5120),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
    }

    batch = model.prepare_batch(
        raw_batch,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    prediction = model.predict_noise(
        batch.noisy_model_input.permute(0, 2, 1, 3, 4),
        batch.timesteps,
        batch,
        conditional=True,
    )

    assert isinstance(prediction, tuple)
    assert prediction[0].shape == batch.latents.shape
    assert prediction[1].shape == batch.audio_latents.shape
    torch.testing.assert_close(
        prediction[0],
        -batch.noisy_model_input.permute(0, 2, 1, 3, 4),
    )
    torch.testing.assert_close(prediction[1], -batch.audio_noisy_model_input)
    assert batch.minimax_h3_layout.sequence_length == 26


def test_h3_validation_prompts_match_crush_smol_recipe() -> None:
    """Verify that H3 validation uses every Crush-Smol validation caption."""
    prompt_document = json.loads(_VALIDATION_PROMPTS.read_text())
    captions = [record["caption"] for record in prompt_document["data"]]

    assert captions == _EXPECTED_VALIDATION_CAPTIONS
    assert len(captions) == len(set(captions)) == 3
    assert all(caption.strip() for caption in captions)


def test_h3_experiment_config_uses_modular_validation_callback() -> None:
    """Verify that the H3 experiment uses modular validation and W&B logging."""
    config = yaml.safe_load(_EXPERIMENT_CONFIG.read_text())
    validation = config["callbacks"]["validation"]
    tracker = config["training"]["tracker"]

    assert config["method"]["_target_"] == "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod"
    assert validation["_target_"] == "fastvideo.train.callbacks.validation.ValidationCallback"
    assert validation["pipeline_target"] == (
        "fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline.MiniMaxH3Pipeline"
    )
    assert validation["dataset_file"] == (
        "examples/training/finetune/Wan2.1-Fun-1.3B-InP/crush_smol/validation.json"
    )
    assert validation["every_steps"] == 20
    assert validation["run_at_start"] is True
    assert validation["sampling_steps"] == [50]
    assert validation["num_frames"] == 124
    assert validation["num_videos_per_prompt"] == 1
    assert validation["use_validation_media_conditioning"] is False
    assert validation["offload_training_state"] is True
    assert validation["text_encoder_cpu_offload"] is True
    assert validation["vae_cpu_offload"] is True
    assert tracker["trackers"] == ["wandb"]
    assert tracker["project_name"] == "fastvideo_minimax_h3"


def test_h3_experiment_config_resolves_64_gpu_mesh_and_global_batch() -> None:
    """Verify the experiment uses eight data-parallel replicas and batch eight."""
    config = yaml.safe_load(_EXPERIMENT_CONFIG.read_text())
    training = config["training"]
    distributed = training["distributed"]

    assert distributed == {
        "num_gpus": 64,
        "sp_size": 8,
        "tp_size": 1,
        "hsdp_replicate_dim": 8,
        "hsdp_shard_dim": 8,
        "pin_cpu_memory": True,
    }
    data_parallel_degree = distributed["num_gpus"] // distributed["sp_size"]
    effective_global_batch = (
        training["data"]["train_batch_size"]
        * data_parallel_degree
        * training["loop"]["gradient_accumulation_steps"]
    )
    assert distributed["hsdp_replicate_dim"] * distributed["hsdp_shard_dim"] == 64
    assert data_parallel_degree == 8
    assert effective_global_batch == 8
    assert training["data"]["data_path"] == {
        "data/crush-smol_h3_t2va_single_sample_preprocessed": 8
    }
    assert training["loop"]["max_train_steps"] == 400
    assert training["checkpoint"]["training_state_checkpointing_steps"] == 20
    assert training["checkpoint"]["checkpoints_total_limit"] == 2
    assert training["tracker"]["run_name"] == "minimax_h3_t2va_crush_smol_single_sample_overfit"


def test_h3_experiment_data_repeat_supplies_every_sp_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify one parquet row repeats into one complete batch per SP group."""
    dataset_root = tmp_path / "one-row-parquet"
    dataset_root.mkdir()
    pq.write_table(pa.table({"sample_id": [0]}), dataset_root / "data_00000.parquet")
    monkeypatch.setattr(
        "fastvideo.dataset.parquet_dataset_map_style.get_world_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "fastvideo.dataset.parquet_dataset_map_style.get_world_group",
        lambda: SimpleNamespace(barrier=lambda: None),
    )

    parquet_files, parquet_lengths = get_parquet_files_and_length({str(dataset_root): 8})
    assert len(parquet_files) == 8
    assert sum(parquet_lengths) == 8

    assigned_indices = []
    for sp_group_index in range(8):
        sampler = DP_SP_BatchSampler(
            batch_size=1,
            dataset_size=8,
            num_sp_groups=8,
            sp_world_size=8,
            global_rank=sp_group_index * 8,
            drop_last=True,
            seed=42,
        )
        batches = list(sampler)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assigned_indices.extend(batches[0])
    assert sorted(assigned_indices) == list(range(8))


def test_h3_slurm_scripts_preserve_inherited_wandb_key(
    tmp_path: Path,
) -> None:
    """Verify shell syntax, H200 allocation, and secret-free job-script text."""
    run_slurm = _REPO_ROOT / "examples/train/run_slurm.sh"
    launcher = _REPO_ROOT / "examples/train/launch_minimax_h3_t2va_crush_smol_validation.sh"
    setup = _REPO_ROOT / "examples/train/setup_minimax_h3_t2va_crush_smol_single_sample.sh"
    for script in (run_slurm, launcher, setup):
        subprocess.run(["bash", "-n", str(script)], check=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    generated_job_script = tmp_path / "generated-job.sh"
    sbatch_arguments = tmp_path / "sbatch-arguments.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > \"$SBATCH_ARGUMENTS\"\n"
        "dd status=none of=\"$SBATCH_JOB_SCRIPT\"\n"
        "printf 'Submitted batch job 4242\\n'\n"
    )
    fake_sbatch.chmod(0o755)
    secret_value = "wandb-test-secret-must-not-appear"
    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "WANDB_API_KEY": secret_value,
        "WANDB_MODE": "online",
        "OUTPUT_DIR": str(tmp_path / "slurm"),
        "SBATCH_ARGUMENTS": str(sbatch_arguments),
        "SBATCH_JOB_SCRIPT": str(generated_job_script),
    })

    completed = subprocess.run(
        ["bash", str(run_slurm), str(_EXPERIMENT_CONFIG), "8"],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    generated_text = generated_job_script.read_text()
    submission_text = completed.stdout + completed.stderr
    assert secret_value not in generated_text
    assert secret_value not in submission_text
    assert "--export=ALL,WANDB_MODE=online" in sbatch_arguments.read_text()
    launcher_text = launcher.read_text()
    assert "SBATCH_CONSTRAINT=nvidia_h200" in launcher_text
    assert "SBATCH_TIMELIMIT=72:00:00" in launcher_text
    assert "data/crush-smol_h3_t2va_single_sample_preprocessed" in launcher_text
    assert "runs/minimax_h3_t2va_crush_smol_single_sample_overfit" in launcher_text
    assert "--validate-only" in launcher_text
    assert "git ls-files --modified --others --exclude-standard" in launcher_text
    assert "source-files.tar.gz" in launcher_text
    assert 'bash examples/train/run_slurm.sh "${CONFIG_FILE}" 8' in launcher_text


def test_h3_slurm_job_resolves_one_node_rank_per_srun_task(tmp_path: Path) -> None:
    """Execute the rendered job script with eight simulated Slurm task ranks."""
    run_slurm = _REPO_ROOT / "examples/train/run_slurm.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    generated_job_script = tmp_path / "generated-job.sh"
    torchrun_arguments_dir = tmp_path / "torchrun-arguments"
    torchrun_arguments_dir.mkdir()

    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "dd status=none of=\"$SBATCH_JOB_SCRIPT\"\n"
        "printf 'Submitted batch job 4242\\n'\n"
    )
    fake_sbatch.chmod(0o755)
    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text("#!/bin/bash\nprintf 'node%s\\n' {0..7}\n")
    fake_scontrol.chmod(0o755)
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "for rank in {0..7}; do\n"
        "    SLURM_PROCID=\"$rank\" \"$@\"\n"
        "done\n"
    )
    fake_srun.chmod(0o755)
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > "
        "\"$TORCHRUN_ARGUMENTS_DIR/rank_${SLURM_PROCID}.txt\"\n"
    )
    fake_torchrun.chmod(0o755)

    environment = os.environ.copy()
    environment.update({
        "PATH": f"{fake_bin}:{environment['PATH']}",
        "OUTPUT_DIR": str(tmp_path / "slurm"),
        "SBATCH_JOB_SCRIPT": str(generated_job_script),
        "TORCHRUN_ARGUMENTS_DIR": str(torchrun_arguments_dir),
    })
    subprocess.run(
        ["bash", str(run_slurm), str(_EXPERIMENT_CONFIG), "8"],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    environment.update({
        "SLURM_JOB_NODELIST": "node[0-7]",
        "SLURM_JOB_NUM_NODES": "8",
        "SLURM_PROCID": "0",
    })
    subprocess.run(
        ["bash", str(generated_job_script)],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    for rank in range(8):
        torchrun_arguments = (torchrun_arguments_dir / f"rank_{rank}.txt").read_text().splitlines()
        node_rank_index = torchrun_arguments.index("--node_rank")
        assert torchrun_arguments[node_rank_index + 1] == str(rank)


def test_h3_model_uses_modular_activation_checkpointing() -> None:
    """Verify that H3 uses activation checkpointing owned by the modular trainer."""
    source = (_REPO_ROOT / "fastvideo/train/models/minimax_h3/minimax_h3.py").read_text()

    assert "fastvideo.training." not in source
