import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
import subprocess
import sys
from fastvideo.tests.ssim.test_inference_similarity import compute_video_ssim_torchvision

# Import the training pipeline
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

NUM_NODES = "1"
MODEL_PATH = "hunyuanvideo-community/HunyuanVideo"

# preprocessing
DATA_DIR = "data"
LOCAL_RAW_DATA_DIR = Path(os.path.join(DATA_DIR, "cats"))
NUM_GPUS_PER_NODE_PREPROCESSING = "1"
PREPROCESSING_ENTRY_FILE_PATH = "fastvideo/pipelines/preprocess/v1_preprocessing_new.py"

LOCAL_PREPROCESSED_DATA_DIR = Path(os.path.join(DATA_DIR, "cats_processed_t2v_hunyuan"))


# training
NUM_GPUS_PER_NODE_TRAINING = "1"
TRAINING_ENTRY_FILE_PATH = "fastvideo/training/hunyuan_training_pipeline.py"
# New preprocessing pipeline creates files in training_dataset/worker_0/worker_0/
LOCAL_TRAINING_DATA_DIR = os.path.join(LOCAL_PREPROCESSED_DATA_DIR, "training_dataset", "worker_0", "worker_0")
LOCAL_VALIDATION_DATASET_FILE = os.path.join(LOCAL_RAW_DATA_DIR, "validation_prompt.json")
LOCAL_OUTPUT_DIR = Path(os.path.join(DATA_DIR, "outputs_hunyuan"))

def download_data():
    # create the data dir if it doesn't exist
    data_dir = Path(DATA_DIR)

    print(f"Creating data directory at {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading raw dataset to {LOCAL_RAW_DATA_DIR}...")
    try:
        result = snapshot_download(
            repo_id="wlsaidhi/cats-overfit-merged",
            local_dir=str(LOCAL_RAW_DATA_DIR),
            repo_type="dataset",
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),  # In case authentication is needed
        )
        print(f"Download completed successfully. Files downloaded to: {result}")
        
        # Verify the download
        if not LOCAL_RAW_DATA_DIR.exists():
            raise RuntimeError(f"Download appeared to succeed but {LOCAL_RAW_DATA_DIR} does not exist")
            
        # List downloaded files
        print("Downloaded files:")
        for file in LOCAL_RAW_DATA_DIR.rglob("*"):
            if file.is_file():
                print(f"  - {file.relative_to(LOCAL_RAW_DATA_DIR)}")
        
        # Rename video directory if needed (dataset has 'video' but preprocessing expects 'videos')
        video_dir = os.path.join(LOCAL_RAW_DATA_DIR, "video")
        videos_dir = os.path.join(LOCAL_RAW_DATA_DIR, "videos")
        if os.path.exists(video_dir) and not os.path.exists(videos_dir):
            print(f"Renaming video directory to videos...")
            os.rename(video_dir, videos_dir)
                
    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise


def run_preprocessing():
    # remove the local_preprocessed_data_dir if it exists
    if LOCAL_PREPROCESSED_DATA_DIR.exists():
        print(f"Removing local_preprocessed_data_dir: {LOCAL_PREPROCESSED_DATA_DIR}")
        shutil.rmtree(LOCAL_PREPROCESSED_DATA_DIR)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    # Run torchrun command using the new preprocessing pipeline
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE_PREPROCESSING,
        "-m", "fastvideo.pipelines.preprocess.v1_preprocessing_new",
        "--model-path", MODEL_PATH,
        "--mode", "preprocess",
        "--workload-type", "t2v",
        "--preprocess.video_loader_type", "torchvision",
        "--preprocess.dataset_type", "merged",
        "--preprocess.dataset_path", str(LOCAL_RAW_DATA_DIR),
        "--preprocess.dataset_output_dir", str(LOCAL_PREPROCESSED_DATA_DIR),
        "--preprocess.preprocess_video_batch_size", "1",
        "--preprocess.dataloader_num_workers", "0",
        "--preprocess.max_height", "720",
        "--preprocess.max_width", "1280",
        "--preprocess.num_frames", "125",
        "--preprocess.train_fps", "24",
        "--preprocess.samples_per_file", "1",
        "--preprocess.flush_frequency", "1",
        "--preprocess.video_length_tolerance_range", "5",
    ]

    process = subprocess.run(cmd, check=True, env=env)


def run_training():
    cmd = [
        "torchrun",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE_TRAINING,
        TRAINING_ENTRY_FILE_PATH,
        "--model_path", MODEL_PATH,
        "--inference_mode", "False",
        "--pretrained_model_name_or_path", MODEL_PATH,
        "--data_path", LOCAL_TRAINING_DATA_DIR,
        "--validation_dataset_file", LOCAL_VALIDATION_DATASET_FILE,
        "--train_batch_size", "1",
        "--num_latent_t", "4",
        "--num_gpus", NUM_GPUS_PER_NODE_TRAINING,
        "--sp_size", NUM_GPUS_PER_NODE_TRAINING,
        "--tp_size", "1",
        "--hsdp_replicate_dim", "1",
        "--hsdp_shard_dim", NUM_GPUS_PER_NODE_TRAINING,
        "--train_sp_batch_size", "1",
        "--dataloader_num_workers", "10",
        "--gradient_accumulation_steps", "1",
        "--max_train_steps", "901",
        "--learning_rate", "1e-5",
        "--mixed_precision", "bf16",
        "--weight_only_checkpointing_steps", "6000",
        "--training_state_checkpointing_steps", "6000",
        "--validation_steps", "100",
        "--validation_sampling_steps", "50",
        "--log_validation",
        "--checkpoints_total_limit", "3",
        "--ema_start_step", "0",
        "--training_cfg_rate", "0.0",
        "--output_dir", str(LOCAL_OUTPUT_DIR),
        "--tracker_project_name", "hunyuan_finetune_overfit_ci",
        "--num_height", "720",
        "--num_width", "1280",
        "--num_frames", "125",
        "--validation_guidance_scale", "1.0",
        "--embedded_cfg_scale", "6.0",
        "--num_euler_timesteps", "50",
        "--multi_phased_distill_schedule", "4000-1",
        "--weight_decay", "0.01",
        "--not_apply_cfg_solver",
        "--dit_precision", "fp32",
        "--max_grad_norm", "1.0",
        "--flow_shift", "7",
    ]

    print(f"Running training with command: {cmd}")
    process = subprocess.run(cmd, check=True)


def test_e2e_hunyuan_overfit_single_sample():
    os.environ["WANDB_MODE"] = "online"

    #download_data()
    #run_preprocessing()
    run_training()

    reference_video_file = os.path.join(os.path.dirname(__file__), "reference_video_hunyuan_1_sample_v0.mp4")
    print(f"reference_video_file: {reference_video_file}")
    final_validation_video_file = os.path.join(LOCAL_OUTPUT_DIR, "validation_step_900_inference_steps_50_video_0.mp4")
    print(f"final_validation_video_file: {final_validation_video_file}")


    # Ensure both files exist
    assert os.path.exists(reference_video_file), f"Reference video not found at {reference_video_file}"
    assert os.path.exists(final_validation_video_file), f"Validation video not found at {final_validation_video_file}"

    # Compute SSIM
    mean_ssim, min_ssim, max_ssim = compute_video_ssim_torchvision(
        reference_video_file,
        final_validation_video_file,
        use_ms_ssim=True  # Using MS-SSIM for better quality assessment
    )

    print("\n===== SSIM Results for Step 900 Validation (Hunyuan) =====")
    print(f"Mean MS-SSIM: {mean_ssim:.4f}")
    print(f"Min MS-SSIM: {min_ssim:.4f}")
    print(f"Max MS-SSIM: {max_ssim:.4f}")

    assert max_ssim > 0.5, f"Max SSIM is below 0.5: {max_ssim}"


if __name__ == "__main__":
    test_e2e_hunyuan_overfit_single_sample()
