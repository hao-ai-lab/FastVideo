import os
import glob
import argparse
from collections import defaultdict
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from fastvideo import PipelineConfig
from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.distributed import get_local_torch_device, maybe_init_distributed_environment_and_model_parallel
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines import ComposedPipelineBase
from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame
from fastvideo.utils import maybe_download_model


class FootsiesPreprocessor(ComposedPipelineBase):
    _required_config_modules = ["vae", "image_encoder", "image_processor"]

    def create_pipeline_stages(self, fastvideo_args):
        pass

    def encode_video(self, frames):
        video = torch.stack(frames, dim=2)
        video = video.unsqueeze(0).to(get_local_torch_device(), dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
            latent = self.get_module("vae").encode(video).mean
        vae = self.get_module("vae")
        if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
            latent = latent - vae.shift_factor.to(latent.device)
        latent = latent * vae.scaling_factor.to(latent.device)
        return latent.cpu().numpy()[0]

    def encode_clip(self, pil_image):
        processed = self.get_module("image_processor")(images=pil_image, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(get_local_torch_device())
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            clip_out = self.get_module("image_encoder")(pixel_values=pixel_values)
        return clip_out.last_hidden_state.cpu().numpy()[0]

    def encode_first_frame_latent(self, frame, num_frames, height, width):
        frame = frame.unsqueeze(0).unsqueeze(2)
        zeros = torch.zeros(1, 3, num_frames - 1, height, width)
        video = torch.cat([frame, zeros], dim=2).to(get_local_torch_device(), dtype=torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
            latent = self.get_module("vae").encode(video).mean
        vae = self.get_module("vae")
        if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
            latent = latent - vae.shift_factor.to(latent.device)
        latent = latent * vae.scaling_factor.to(latent.device)
        return latent.cpu().numpy()[0]


def parse_filename(fname):
    parts = fname.replace(".png", "").split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def action_to_keyboard(action, num_frames):
    keyboard = np.zeros((num_frames, 3), dtype=np.float32)
    for i, a in enumerate(action):
        if a == 1:
            keyboard[i, 0] = 1
        elif a == 2:
            keyboard[i, 1] = 1
    return keyboard


def main(args):
    args.model_path = maybe_download_model(args.model_path)
    maybe_init_distributed_environment_and_model_parallel(1, 1)

    pipeline_config = PipelineConfig.from_pretrained(args.model_path)
    pipeline_config.update_config_from_dict({
        "vae_precision": "fp32",
        "vae_config": WanVAEConfig(load_encoder=True, load_decoder=True),
    })

    fastvideo_args = FastVideoArgs(
        model_path=args.model_path,
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pipeline_config=pipeline_config,
    )

    preprocessor = FootsiesPreprocessor(args.model_path, fastvideo_args)
    preprocessor.get_module("vae").to(get_local_torch_device())
    preprocessor.get_module("image_encoder").to(get_local_torch_device())

    image_files = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    episodes = defaultdict(list)
    for f in image_files:
        episode, step, action = parse_filename(os.path.basename(f))
        episodes[episode].append((step, action, f))

    for ep in episodes:
        episodes[ep].sort(key=lambda x: x[0])

    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    sample_idx = 0
    file_idx = 0

    for episode_id in tqdm(sorted(episodes.keys()), desc="Episodes"):
        frames_data = episodes[episode_id]
        if len(frames_data) < args.num_frames:
            continue

        for start in range(0, len(frames_data) - args.num_frames + 1, args.stride):
            seq = frames_data[start:start + args.num_frames]
            frames = []
            actions = []

            for step, action, fpath in seq:
                img = Image.open(fpath).convert("RGB")
                if args.target_height and args.target_width:
                    img = img.resize((args.target_width, args.target_height), Image.BILINEAR)
                arr = np.array(img).astype(np.float32) / 127.5 - 1.0
                frames.append(torch.from_numpy(arr).permute(2, 0, 1))
                actions.append(action)

            first_pil = Image.open(seq[0][2]).convert("RGB")
            if args.target_height and args.target_width:
                first_pil = first_pil.resize((args.target_width, args.target_height), Image.BILINEAR)

            height, width = frames[0].shape[1], frames[0].shape[2]
            vae_latent = preprocessor.encode_video(frames)
            clip_feature = preprocessor.encode_clip(first_pil)
            first_frame_latent = preprocessor.encode_first_frame_latent(frames[0], args.num_frames, height, width)
            keyboard_cond = action_to_keyboard(actions, args.num_frames)
            pil_arr = np.array(first_pil)

            record = {
                "id": f"ep{episode_id}_s{start}",
                "vae_latent_bytes": vae_latent.tobytes(),
                "vae_latent_shape": list(vae_latent.shape),
                "vae_latent_dtype": str(vae_latent.dtype),
                "clip_feature_bytes": clip_feature.tobytes(),
                "clip_feature_shape": list(clip_feature.shape),
                "clip_feature_dtype": str(clip_feature.dtype),
                "first_frame_latent_bytes": first_frame_latent.tobytes(),
                "first_frame_latent_shape": list(first_frame_latent.shape),
                "first_frame_latent_dtype": str(first_frame_latent.dtype),
                "mouse_cond_bytes": b"",
                "mouse_cond_shape": [],
                "mouse_cond_dtype": "",
                "keyboard_cond_bytes": keyboard_cond.tobytes(),
                "keyboard_cond_shape": list(keyboard_cond.shape),
                "keyboard_cond_dtype": str(keyboard_cond.dtype),
                "pil_image_bytes": pil_arr.tobytes(),
                "pil_image_shape": list(pil_arr.shape),
                "pil_image_dtype": str(pil_arr.dtype),
                "file_name": f"ep{episode_id}_s{start}",
                "caption": "",
                "media_type": "video",
                "width": width,
                "height": height,
                "num_frames": args.num_frames,
                "duration_sec": args.num_frames / args.fps,
                "fps": float(args.fps),
            }
            records.append(record)
            sample_idx += 1

            if len(records) >= args.samples_per_file:
                table = pa.Table.from_pylist(records, schema=pyarrow_schema_matrixgame)
                pq.write_table(table, os.path.join(args.output_dir, f"data_{file_idx:05d}.parquet"))
                file_idx += 1
                records = []

    if records:
        table = pa.Table.from_pylist(records, schema=pyarrow_schema_matrixgame)
        pq.write_table(table, os.path.join(args.output_dir, f"data_{file_idx:05d}.parquet"))

    print(f"Done. Total samples: {sample_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=29)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--samples_per_file", type=int, default=32)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--target_width", type=int, default=None)
    args = parser.parse_args()
    main(args)
