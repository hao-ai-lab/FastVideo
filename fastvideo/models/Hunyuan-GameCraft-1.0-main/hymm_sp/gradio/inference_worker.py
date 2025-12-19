import os
import sys
import torch
import torch.distributed as dist
from loguru import logger
import time
import signal
import json
import traceback
import random
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import torchvision.transforms as transforms
from hymm_sp.sample_inference import HunyuanVideoSampler
from hymm_sp.data_kits.data_tools import save_videos_grid
from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)
from decord import VideoReader

# ========== Configuration and Global Variables ==========
TRIGGER_FILE = "./gradio_results/trigger.txt"
RESULT_DIR = "./gradio_results"
INPUT_IMAGES_DIR = os.path.join(RESULT_DIR, "input")
SHUTDOWN_FILE = "./gradio_results/shutdown_worker.signal"
CKPT_PATH = os.getenv("CKPT_PATH")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)

# Global variables
sampler = None
args = None
dataset = None
current_index = 0
shutdown_flag = False

class CropResize:
    """Crop-style resize transformation that maintains aspect ratio"""
    def __init__(self, size=(704, 1216)):
        self.target_h, self.target_w = size 
    def __call__(self, img):
        w, h = img.size
        scale = max(
            self.target_w / w,
            self.target_h / h
        )
        new_size = (int(h * scale), int(w * scale))
        resize_transform = transforms.Resize(new_size, 
                                           interpolation=transforms.InterpolationMode.BILINEAR)
        resized_img = resize_transform(img)
        crop_transform = transforms.CenterCrop((self.target_h, self.target_w))
        return crop_transform(resized_img)

def signal_handler(sig, frame):
    global shutdown_flag
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"Rank {rank} received shutdown signal.")
    shutdown_flag = True

# ========== Utility Function: Save Base64 Image ==========
def save_base64_image(base64_string, save_dir, prefix="input_image"):
    """Decode Base64 string and save as PNG image file"""
    if not base64_string:
        return None
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}_{timestamp}.png"
        file_path = os.path.join(save_dir, filename)
        image.save(file_path, "PNG")
        logger.info(f"Saved uploaded image to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save Base64 image: {e}")
        logger.error(traceback.format_exc())
        return None

# ========== Model Loading ==========
def load_model():
    """Load model with DDP and attribute mapping"""
    global sampler, args
    from hymm_sp.config import parse_args
    args = parse_args()

    args.ckpt = CKPT_PATH
    args.add_neg_prompt = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border."
    args.input_prompt = "Realistic, High-quality."
    args.cfg_scale = 2.0
    args.image_start = True
    args.action_list = ['w', 'w']
    args.action_speed_list = [0.2]
    args.seed = 250160
    args.sample_n_frames = 33
    args.infer_steps = 50
    args.flow_shift_eval_video = 5.0
    args.save_path = "./gradio_results"

    rank = torch.distributed.get_rank()
    if rank == 0:
        logger.info("🔧 Constructing model arguments...")
        logger.info(f"🔧 Key settings - ckpt: {args.ckpt}")
        logger.info("🚀 Loading model on GPUs...")
    try:
        # Load models
        device = torch.device("cuda")
        if nccl_info.sp_size > 1:
            device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise e

    if rank == 0:
        logger.info("✅ Model loaded. Ready for inference.")

# ========== Core Work Loop ==========
def worker_loop(rank, world_size, device):
    """Worker's main listening and processing loop"""
    global current_index, shutdown_flag, sampler, args
    logger.info(f"🔥 Worker rank {rank} (on {device}) entering listening loop...")
    
    closest_size = (704, 1216)
    ref_image_transform = transforms.Compose([
        CropResize(closest_size),
        transforms.CenterCrop(closest_size),
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])

    while not shutdown_flag:
        should_generate = False
        custom_params = None
        if rank == 0:
            if os.path.exists(TRIGGER_FILE):
                try:
                    with open(TRIGGER_FILE, 'r') as f:
                        trigger_data = json.load(f)
                    
                    if isinstance(trigger_data, dict):
                        requested_index = trigger_data.get("index")
                        custom_params = trigger_data.get("custom_params")
                        
                        if custom_params is not None:
                            should_generate = True
                            logger.info(f"Rank {rank}: Custom generation trigger found")
                            if not isinstance(custom_params, dict):
                                logger.error("Invalid custom_params format")
                                custom_params = None
                                should_generate = False
                        elif isinstance(requested_index, int) and requested_index == current_index:
                            should_generate = True
                            logger.info(f"Rank {rank}: Trigger for index {requested_index}")
                        else:
                            logger.warning("Trigger content invalid or index mismatch")
                    else:
                        logger.warning("Invalid trigger file content")
                except (json.JSONDecodeError, IOError) as e:
                    logger.error(f"Error reading trigger file: {e}")
                finally:
                    try:
                        os.remove(TRIGGER_FILE)
                        logger.debug("Removed trigger file")
                    except OSError as e:
                        logger.warning(f"Could not remove trigger file: {e}")

            if os.path.exists(SHUTDOWN_FILE):
                logger.info("Rank 0: Shutdown signal detected")
                shutdown_flag = True
                try:
                    os.remove(SHUTDOWN_FILE)
                except OSError:
                    pass

        # broadcast generate signal
        should_generate_tensor = torch.tensor([int(should_generate)], dtype=torch.long, device=device)
        dist.broadcast(should_generate_tensor, src=0)
        should_generate = bool(should_generate_tensor.item())

        # broadcast params
        if should_generate:
            if rank == 0 and custom_params is not None:
                custom_params_str = json.dumps(custom_params)
                custom_params_bytes = custom_params_str.encode('utf-8')
                custom_params_tensor = torch.ByteTensor(list(custom_params_bytes)).to(device)
                length_tensor = torch.tensor([len(custom_params_bytes)], dtype=torch.long, device=device)
            else:
                length_tensor = torch.tensor([0], dtype=torch.long, device=device)
                custom_params_tensor = torch.ByteTensor([0]).to(device)
            
            dist.broadcast(length_tensor, src=0)
            custom_params_length = length_tensor.item()
            
            if custom_params_length > 0:
                if rank != 0:
                    custom_params_tensor = torch.ByteTensor(custom_params_length).to(device)
                dist.broadcast(custom_params_tensor, src=0)
                if rank != 0:
                    custom_params_bytes = bytes(custom_params_tensor.cpu().tolist())
                    custom_params_str = custom_params_bytes.decode('utf-8')
                    custom_params = json.loads(custom_params_str)

        # broadcast index and shutdown
        index_tensor = torch.tensor([current_index], dtype=torch.long, device=device)
        dist.broadcast(index_tensor, src=0)
        current_index = index_tensor.item()

        shutdown_tensor = torch.tensor([int(shutdown_flag)], dtype=torch.long, device=device)
        dist.broadcast(shutdown_tensor, src=0)
        if shutdown_tensor.item() == 1:
            shutdown_flag = True

        if shutdown_flag:
            logger.info(f"Rank {rank}: Shutdown signal received. Stopping.")
            break

        if should_generate:
            result_data = {"index": current_index}
            temp_args = None
            
            try:
                logger.info(f"Rank {rank}: Starting generation...")
                
                if custom_params is not None:
                    temp_args = type('TempArgs', (object,), vars(args))()
                    
                    # cover
                    temp_args.add_pos_prompt = custom_params.get("add_pos_prompt", args.add_pos_prompt)
                    temp_args.add_neg_prompt = custom_params.get("add_neg_prompt", args.add_neg_prompt)
                    temp_args.cfg_scale = float(custom_params.get("cfg_scale", args.cfg_scale))
                    temp_args.image_start = bool(custom_params.get("image_start", args.image_start))
                    temp_args.video_path = custom_params.get("video_path", None)
                    temp_args.seed = int(custom_params.get("seed", args.seed))
                    temp_args.sample_n_frames = int(custom_params.get("sample_n_frames", args.sample_n_frames))
                    temp_args.infer_steps = int(custom_params.get("infer_steps", args.infer_steps))
                    temp_args.flow_shift_eval_video = float(custom_params.get("flow_shift_eval_video", args.flow_shift_eval_video))
                    temp_args.save_path = custom_params.get("save_path", args.save_path)
                    temp_args.input_prompt = custom_params.get("input_prompt", args.input_prompt)
                    
                    action_list_raw = custom_params.get("action_list", args.action_list)
                    temp_args.action_list = action_list_raw if isinstance(action_list_raw, list) else args.action_list
                    action_speed_list_raw = custom_params.get("action_speed_list", args.action_speed_list)
                    temp_args.action_speed_list = [float(a) for a in action_speed_list_raw] if isinstance(action_speed_list_raw, list) else args.action_speed_list
                    
                    video_size_raw = custom_params.get("video_size", [704, 1216])
                    temp_args.video_size = [int(video_size_raw[0]), int(video_size_raw[1])] if isinstance(video_size_raw, list) and len(video_size_raw) == 2 else [704, 1216]
                    
                    base64_image_str = custom_params.get("input_image", "")
                    if base64_image_str:
                        image_path = save_base64_image(base64_image_str, INPUT_IMAGES_DIR, prefix=f"custom_{int(time.time())}")
                        if image_path:
                            prompt = temp_args.add_pos_prompt
                            save_name = f"custom_{int(time.time())}"
                            batch = {
                                'prompt': temp_args.input_prompt,
                                'index': save_name,
                                'ref_image': image_path
                            }
                            logger.info(f"Custom generation setup: Prompt={prompt}, Image Path={image_path}")
                        else:
                            raise ValueError("Failed to process uploaded image")
                    else:
                        raise ValueError("No input image provided")
                else:
                    raise ValueError("No params provided")
                

                prompt = batch.get('prompt')
                save_name = batch.get('index')
                image_path = batch.get('ref_image')
                time.sleep(2)
                
                if image_path is None or not os.path.exists(image_path):
                        raise ValueError(f"Reference image not found: {image_path}")
                        
                logger.info(f"Rank {rank}: Processing: Prompt={prompt}, Image Path={image_path}")

                # preprocess image
                raw_ref_image = Image.open(image_path).convert('RGB')
                ref_image_pixel_value = ref_image_transform(raw_ref_image).unsqueeze(0).unsqueeze(2).to(device)
                ref_images = [raw_ref_image]

                # seed set
                seed = temp_args.seed if hasattr(temp_args, 'seed') else random.randint(0, 1_000_000)

                # code image to latent space
                if temp_args.video_path is None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        sampler.pipeline.vae.enable_tiling()
                        raw_last_latents = sampler.vae.encode(ref_image_pixel_value).latent_dist.sample().to(dtype=torch.float16)
                        raw_last_latents.mul_(sampler.vae.config.scaling_factor)
                        raw_ref_latents = raw_last_latents.clone()
                        sampler.pipeline.vae.disable_tiling()
                else:
                    ref_video = VideoReader(temp_args.video_path)
                    transformed_images = []
                    for index in range(len(ref_video)):
                        video_image = ref_video[index].asnumpy()
                        transformed_image = ref_image_transform(Image.fromarray(video_image))
                        transformed_images.append(transformed_image)
                    transformed_images = torch.stack(transformed_images,dim=1).unsqueeze(0).to(device=device, dtype=torch.float16)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        sampler.pipeline.vae.enable_tiling()
                        raw_last_latents = sampler.vae.encode(transformed_images[:, :, -33:, ...]).latent_dist.sample().to(dtype=torch.float16)
                        raw_last_latents.mul_(sampler.vae.config.scaling_factor)
                        raw_ref_latents = sampler.vae.encode(transformed_images[:, :, -33:-32, ...]).latent_dist.sample().to(dtype=torch.float16)
                        raw_ref_latents.mul_(sampler.vae.config.scaling_factor)
                        sampler.pipeline.vae.disable_tiling()

                last_latents = raw_last_latents
                ref_latents = raw_ref_latents
                out_cat = None

                # pipeline
                for idx, action_id in enumerate(temp_args.action_list):
                    is_image = idx == 0 and temp_args.image_start

                    outputs = sampler.predict(
                        prompt=prompt,
                        action_id=action_id,
                        action_speed=temp_args.action_speed_list[idx],                    
                        is_image=is_image,
                        size=tuple(temp_args.video_size) if hasattr(temp_args, 'video_size') else (704, 1216),
                        seed=seed,
                        last_latents=last_latents,
                        ref_latents=ref_latents,
                        video_length=temp_args.sample_n_frames,
                        guidance_scale=temp_args.cfg_scale,
                        num_images_per_prompt=1,
                        negative_prompt=temp_args.add_neg_prompt,
                        infer_steps=temp_args.infer_steps,
                        flow_shift=temp_args.flow_shift_eval_video,
                        use_linear_quadratic_schedule=getattr(temp_args, 'use_linear_quadratic_schedule', False),
                        linear_schedule_end=getattr(temp_args, 'linear_schedule_end', 0),
                        use_deepcache=getattr(temp_args, 'use_deepcache', False),
                        ref_images=ref_images,
                        output_dir=temp_args.save_path,
                        return_latents=True,
                    )
                    ref_latents = outputs["ref_latents"]
                    last_latents = outputs["last_latents"]
                    
                    if rank == 0:
                        sub_samples = outputs['samples'][0]
                        if idx == 0:
                            if temp_args.image_start or temp_args.video_path is not None:
                                out_cat = sub_samples
                            else:
                                out_cat = torch.cat([(ref_image_pixel_value + 1) / 2.0, sub_samples], dim=2)
                        else:
                            out_cat = torch.cat([out_cat, sub_samples], dim=2)

                if rank == 0:
                    # result save
                    if out_cat is not None:
                        save_path = os.path.join(temp_args.save_path, f"{save_name}.mp4")
                        os.makedirs(temp_args.save_path, exist_ok=True)
                        save_videos_grid(out_cat, save_path, n_rows=1, fps=24)
                        result_data["video_path"] = save_path
                        logger.info(f"Rank {rank}: Video saved to {save_path}")

                        # frame extract
                        frame_paths = []
                        video_tensor = out_cat.squeeze(0)
                        num_frames = video_tensor.shape[1]
                        frames_dir = os.path.join(temp_args.save_path, f"{save_name}_frames")
                        os.makedirs(frames_dir, exist_ok=True)
                        
                        for i in range(num_frames):
                            frame_tensor = video_tensor[:, i, :, :]
                            frame_array = (frame_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            frame_image = Image.fromarray(frame_array)
                            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
                            frame_image.save(frame_path)
                            frame_paths.append(frame_path)
                        
                        result_data["frame_paths"] = frame_paths
                        logger.info(f"Rank {rank}: Saved {len(frame_paths)} frames")
                    else:
                        error_msg = "No video samples generated"
                        logger.error(f"Rank {rank}: {error_msg}")
                        result_data["error"] = error_msg

            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                logger.error(f"Rank {rank}: {error_msg}")
                logger.error(traceback.format_exc())
                result_data["error"] = error_msg

            if rank == 0:
                result_file_path = os.path.join(RESULT_DIR, f"result_custom.json")
                result_tmp_path = result_file_path + ".tmp"
                try:
                    with open(result_tmp_path, 'w') as f:
                        json.dump(result_data, f, indent=4)
                    os.replace(result_tmp_path, result_file_path)
                    logger.info(f"Rank {rank}: Result written to {result_file_path}")
                except Exception as e:
                    logger.error(f"Failed to write result file: {e}")
                

        time.sleep(0.1)
    logger.info(f"Rank {rank}: Worker loop finished")

# ========== Main Function ==========
def main():
    global sampler, args, dataset, current_index, shutdown_flag
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    
    try:
        initialize_distributed(seed=42)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if rank == 0:
            logger.info(f"Process group initialized. Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
            logger.info(f"Using device: {device}")
        
        load_model()
        worker_loop(rank, world_size, device)
        
    except Exception as e:
        logger.error(f"Fatal error in worker (Rank {rank}): {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"Rank {rank}: Shutting down...")
        if dist.is_initialized():
            dist.destroy_process_group()
        logger.info(f"Rank {rank}: Shutdown complete")
        os.remove(SHUTDOWN_FILE)

if __name__ == "__main__":
    main()