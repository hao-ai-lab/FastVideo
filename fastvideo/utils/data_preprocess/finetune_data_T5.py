import argparse
import torch
from accelerate.logging import get_logger
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
import json
import os
import torch.distributed as dist
logger = get_logger(__name__)

def main(args): 
    local_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    print('world_size', world_size, 'local rank', local_rank)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)
    
    pipe = MochiPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device)
    pipe.vae.enable_tiling()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_attention_mask"), exist_ok=True)
    
    latents_json_path = os.path.join(args.output_dir, "videos2caption_temp.json")
    with open(latents_json_path, "r") as f:
        train_dataset = json.load(f)
        train_dataset = sorted(train_dataset, key=lambda x: x['latent_path'])
    
        json_data = []
        for _, data in enumerate(train_dataset):
            video_name =data['latent_path'].split(".")[0]
            if int(video_name) % world_size != local_rank:
                continue
            try:
                with torch.inference_mode():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        latent = torch.load(os.path.join(args.output_dir, 'latent', data['latent_path']))
                        prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
                            prompt=data['caption'],
                        )
                        prompt_embed_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                        video_path = os.path.join(args.output_dir, "video", video_name + ".mp4")
                        prompt_attention_mask_path = os.path.join(args.output_dir, "prompt_attention_mask", video_name + ".pt")
                        # save latent
                        torch.save(prompt_embeds[0], prompt_embed_path)
                        torch.save(prompt_attention_mask[0], prompt_attention_mask_path)
                        print(f"sample {video_name} saved")
                        video = pipe.vae.decode(latent.unsqueeze(0).to(device), return_dict=False)[0]
                        video = pipe.video_processor.postprocess_video(video)
                        export_to_video(video[0], video_path, fps=30)
                        item = {}
                        item["latent_path"] = video_name + ".pt"
                        item["prompt_embed_path"] = video_name + ".pt"
                        item["prompt_attention_mask"] = video_name + ".pt"
                        item["caption"] = data['caption']
                        json_data.append(item)
            except:
                print("video out of memory")
                continue
        dist.barrier()
        local_data = json_data
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        # os.remove(latents_json_path)
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), 'w') as f:
            json.dump(all_json_data, f, indent=4)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default='google/t5-v1_1-xxl')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--output_dir", type=str, default=None, help="The output directory where the model predictions and checkpoints will be written.")

    args = parser.parse_args()
    main(args)
