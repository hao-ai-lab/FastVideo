import os
from pathlib import Path
import shutil
import subprocess
import sys

from huggingface_hub import snapshot_download
from fastvideo.tests.ssim.test_inference_similarity import compute_video_ssim_torchvision

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

NUM_NODES = "1"
MODEL_PATH = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

DATA_DIR = "data"
LOCAL_RAW_DATA_DIR = Path(DATA_DIR) / "cats15"
LOCAL_PREPROCESSED_DATA_DIR = Path(DATA_DIR) / "cats_processed_t2v_hunyuan15"
LOCAL_OUTPUT_DIR = Path(DATA_DIR) / "outputs_hunyuan15"

NUM_GPUS_PER_NODE_PREPROCESSING = "1"
NUM_GPUS_PER_NODE_TRAINING = "4"

# entrypoints (adjust to what hunyuan15 scripts use)
PREPROCESS_ENTRY = ["-m", "fastvideo.pipelines.preprocess.v1_preprocessing_new"]
TRAIN_ENTRY_FILE_PATH = "fastvideo/training/hunyuan15_training_pipeline.py"  # change if hunyuan15 uses a different file

LOCAL_TRAINING_DATA_DIR = os.path.join(LOCAL_PREPROCESSED_DATA_DIR, "training_dataset", "worker_0", "worker_0")
LOCAL_VALIDATION_DATASET_FILE = os.path.join(LOCAL_RAW_DATA_DIR, "validation_prompt.json")


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_download(
        repo_id="wlsaidhi/cats-overfit-merged",
        local_dir=str(LOCAL_RAW_DATA_DIR),
        repo_type="dataset",
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # normalize dataset layout like your hunyuan test
    video_dir = LOCAL_RAW_DATA_DIR / "video"
    videos_dir = LOCAL_RAW_DATA_DIR / "videos"
    if video_dir.exists() and not videos_dir.exists():
        video_dir.rename(videos_dir)

    src_val = LOCAL_RAW_DATA_DIR / "validation_prompt_1_sample.json"
    shutil.copy2(src_val, LOCAL_VALIDATION_DATASET_FILE)

    src_v2c = LOCAL_RAW_DATA_DIR / "videos2caption_1_sample.json"
    shutil.copy2(src_v2c, LOCAL_RAW_DATA_DIR / "videos2caption.json")


def run_preprocessing():
    if LOCAL_PREPROCESSED_DATA_DIR.exists():
        shutil.rmtree(LOCAL_PREPROCESSED_DATA_DIR)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE_PREPROCESSING,
        *PREPROCESS_ENTRY,
        "--model-path", MODEL_PATH,
        "--mode", "preprocess",
        "--workload-type", "t2v",
        "--preprocess.video_loader_type", "torchvision",
        "--preprocess.dataset_type", "merged",
        "--preprocess.dataset_path", str(LOCAL_RAW_DATA_DIR),
        "--preprocess.dataset_output_dir", str(LOCAL_PREPROCESSED_DATA_DIR),
        "--preprocess.preprocess_video_batch_size", "1",
        "--preprocess.dataloader_num_workers", "0",
        "--preprocess.max_height", "480",
        "--preprocess.max_width", "832",
        "--preprocess.num_frames", "77",
        "--preprocess.train_fps", "16",
        "--preprocess.samples_per_file", "1",
        "--preprocess.flush_frequency", "1",
        "--preprocess.video_length_tolerance_range", "5",

        # IMPORTANT: add a dtype/precision flag here if available in your codebase,
        # so saved arrays are fp16/fp32 instead of bf16.
        # Example (ONLY if supported by args):
        # "--preprocess.save_dtype", "fp16",
        # "--preprocess.embedding_dtype", "fp16",
    ]

    subprocess.run(cmd, check=True, env=env)


def run_training():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE_TRAINING,
        TRAIN_ENTRY_FILE_PATH,
        "--model_path", MODEL_PATH,
        "--pretrained_model_name_or_path", MODEL_PATH,
        "--data_path", LOCAL_TRAINING_DATA_DIR,
        "--validation_dataset_file", LOCAL_VALIDATION_DATASET_FILE,

        "--train_batch_size", "1",
        "--max_train_steps", "901",
        "--learning_rate", "1e-5",

        # If your pipeline produces bf16 artifacts that later go to numpy/parquet,
        # switch mixed_precision to fp16 to avoid bf16 -> numpy issues.
        "--mixed_precision", "fp16",

        "--validation_steps", "100",
        "--validation_sampling_steps", "50",
        "--output_dir", str(LOCAL_OUTPUT_DIR),
        "--tracker_project_name", "hunyuan15_finetune_overfit_ci",
        "--num_height", "480",
        "--num_width", "832",
        "--num_frames", "81",
        "--validation_guidance_scale", "1.0",
        "--embedded_cfg_scale", "6.0",
        "--num_euler_timesteps", "50",
    ]

    subprocess.run(cmd, check=True, env=env)


def test_e2e_hunyuan15_overfit_single_sample():
    os.environ["WANDB_MODE"] = "offline"

    # download_data()
    run_preprocessing()
    run_training()

    reference_video_file = os.path.join(os.path.dirname(__file__), "reference_video_1_sample_hy15.mp4")
    final_validation_video_file = os.path.join(LOCAL_OUTPUT_DIR, "validation_step_900_inference_steps_50_video_0.mp4")

    assert os.path.exists(reference_video_file)
    assert os.path.exists(final_validation_video_file)

    mean_ssim, min_ssim, max_ssim = compute_video_ssim_torchvision(
        reference_video_file,
        final_validation_video_file,
        use_ms_ssim=True,
    )
    assert max_ssim > 0.5, f"Max SSIM is below threshold: {max_ssim}"


if __name__ == "__main__":
    test_e2e_hunyuan15_overfit_single_sample()
