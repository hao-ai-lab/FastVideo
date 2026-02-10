import argparse
import os
import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.utils import maybe_download_model, save_decoded_latents_as_video


def _torch_dtype_from_precision(precision: str) -> torch.dtype:
    precision = precision.lower()
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {precision}")


def _denormalize_latents_for_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    if bool(getattr(vae, "handles_latent_denorm", False)):
        return latents

    cfg = getattr(vae, "config", None)

    if cfg is not None and hasattr(cfg, "latents_mean") and hasattr(
            cfg, "latents_std"):
        latents_mean = torch.tensor(cfg.latents_mean,
                                    device=latents.device,
                                    dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(cfg.latents_std,
                                   device=latents.device,
                                   dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return latents * latents_std + latents_mean

    if hasattr(vae, "scaling_factor"):
        if isinstance(vae.scaling_factor, torch.Tensor):
            latents = latents / vae.scaling_factor.to(latents.device,
                                                      latents.dtype)
        else:
            latents = latents / vae.scaling_factor

        if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
            if isinstance(vae.shift_factor, torch.Tensor):
                latents = latents + vae.shift_factor.to(latents.device,
                                                        latents.dtype)
            else:
                latents = latents + vae.shift_factor

    return latents


@torch.inference_mode()
def _decode_with_vae(vae, latents: torch.Tensor, *, device: torch.device,
                     precision: str) -> torch.Tensor:
    latents = latents.to(device=device)
    target_dtype = _torch_dtype_from_precision(precision)
    latents = latents.to(dtype=target_dtype)

    latents = _denormalize_latents_for_vae(vae, latents)

    use_autocast = (device.type == "cuda" and target_dtype != torch.float32)
    with torch.autocast(device_type=device.type,
                        dtype=target_dtype,
                        enabled=use_autocast):
        decoded = vae.decode(latents)

    return (decoded / 2 + 0.5).clamp(0, 1)

def main():
    parser = argparse.ArgumentParser(description="Visualize Trajectory from Parquet file")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the input parquet file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save output videos")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae_precision",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision for VAE decoding")
    parser.add_argument("--vae_subfolder",
                        type=str,
                        default="vae",
                        help="Subfolder name containing VAE weights/config")
    parser.add_argument("--fps", type=int, default=25, help="Output video FPS")
    parser.add_argument(
        "--decode_steps",
        type=str,
        default="last",
        help=
        "Which trajectory steps to decode: 'last', 'all', or comma-separated indices (e.g. '0,10,20')",
    )
    
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}, vae_precision: {args.vae_precision}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load VAE (must load weights; creating AutoencoderKLWan(config) alone leaves random weights)
    print(f"Loading model from {args.model_path}...")
    model_root = maybe_download_model(args.model_path)
    pipeline_config = PipelineConfig.from_pretrained(model_root)
    pipeline_config.update_config_from_dict({
        "vae_precision": args.vae_precision,
        "vae_config": WanVAEConfig(load_encoder=False, load_decoder=True),
    })
    fastvideo_args = FastVideoArgs(
        model_path=model_root,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pipeline_config=pipeline_config,
    )

    vae_path = os.path.join(model_root, args.vae_subfolder)
    vae = VAELoader().load(vae_path, fastvideo_args)
    vae.to(device)
    
    # Read Parquet
    print(f"Reading parquet file: {args.parquet_path}")
    table = pq.read_table(args.parquet_path)
    
    # Iterate over rows
    num_visualized = 0
    
    pbar = tqdm(total=min(args.num_samples, table.num_rows))
    
    for i in range(table.num_rows):
        if num_visualized >= args.num_samples:
            break
            
        row = table.slice(i, length=1)
        record = row.to_pydict()
        
        video_id = record["id"][0]
        
        # Parse Latents
        shape = record["trajectory_latents_shape"][0]
        dtype = record["trajectory_latents_dtype"][0]
        dtype = np.dtype(dtype)
        
        latents_bytes = record["trajectory_latents_bytes"][0]
        # Copy to avoid read-only warning
        latents_np = np.copy(np.frombuffer(latents_bytes, dtype=dtype).reshape(shape))
        
        latents_tensor = torch.from_numpy(latents_np)
        if latents_tensor.ndim == 6 and latents_tensor.shape[0] == 1:
            latents_tensor = latents_tensor.squeeze(0)
        
        print(f"Decoding video {video_id} with shape {latents_tensor.shape}...")
        
        # create subfolder
        vid_output_dir = os.path.join(args.output_dir, str(video_id))
        os.makedirs(vid_output_dir, exist_ok=True)
        
        # Pick steps to decode
        steps = latents_tensor.shape[0]
        if args.decode_steps == "last":
            indices_to_decode = [steps - 1]
        elif args.decode_steps == "all":
            indices_to_decode = list(range(steps))
        else:
            indices_to_decode = [int(x) for x in args.decode_steps.split(",") if x.strip() != ""]
        indices_to_decode = [i for i in indices_to_decode if 0 <= i < steps]
        if not indices_to_decode:
            raise ValueError(f"No valid indices selected for decode_steps='{args.decode_steps}' with steps={steps}")
        
        for step in tqdm(indices_to_decode, desc=f"Decoding {video_id}", leave=False):
            latent_step = latents_tensor[step].unsqueeze(0)  # [1, C, T, H, W]

            decoded_video = _decode_with_vae(vae,
                                            latent_step,
                                            device=device,
                                            precision=args.vae_precision)

            save_path = os.path.join(vid_output_dir, f"step_{step:03d}.mp4")
            save_decoded_latents_as_video(decoded_video.float(),
                                          save_path,
                                          fps=args.fps)

        print(f"Saved {len(indices_to_decode)} step(s) to {vid_output_dir}")
        num_visualized += 1
        pbar.update(1)
        
    pbar.close()

if __name__ == "__main__":
    main()
