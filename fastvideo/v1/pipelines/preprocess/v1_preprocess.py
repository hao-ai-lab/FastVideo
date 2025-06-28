import argparse
import os

from fastvideo import PipelineConfig
from fastvideo.v1.configs.models.vaes import WanVAEConfig
from fastvideo.v1.distributed import (
    get_world_size, maybe_init_distributed_environment_and_model_parallel)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.preprocess.preprocess_pipeline_i2v import (
    PreprocessPipeline_I2V)
from fastvideo.v1.pipelines.preprocess.preprocess_pipeline_t2v import (
    PreprocessPipeline_T2V)
from fastvideo.v1.utils import maybe_download_model

logger = init_logger(__name__)


def main(args) -> None:
    args.model_path = maybe_download_model(args.model_path)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    num_gpus = int(os.environ["WORLD_SIZE"])
    assert num_gpus == 1, "Only support 1 GPU"
    pipeline_config = PipelineConfig.from_pretrained(args.model_path)
    kwargs = {
        "use_cpu_offload": False,
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=True, load_decoder=False),
    }
    pipeline_config.update_config_from_dict(kwargs)
    fastvideo_args = FastVideoArgs(
        model_path=args.model_path,
        num_gpus=get_world_size(),
        pipeline_config=pipeline_config,
    )
    PreprocessPipeline = PreprocessPipeline_I2V if args.preprocess_task == "i2v" else PreprocessPipeline_T2V
    pipeline = PreprocessPipeline(args.model_path, fastvideo_args)
    pipeline.forward(batch=None, fastvideo_args=fastvideo_args, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--validation_dataset_file", type=str)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help=
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--preprocess_video_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--samples_per_file", type=int, default=64)
    parser.add_argument("--flush_frequency",
                        type=int,
                        default=256,
                        help="how often to save to parquet files")
    parser.add_argument("--num_latent_t",
                        type=int,
                        default=28,
                        help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range", type=int, default=2.0)
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--preprocess_task", type=str, default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=256)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name",
                        type=str,
                        default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--training_cfg_rate", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=
        "The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    main(args)
