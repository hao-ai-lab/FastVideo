from fastvideo.v1.models.vaes.stepvideovae import AutoencoderKLStepvideo
import os
from fastvideo.v1.models.loader.component_loader import VAELoader
from fastvideo.v1.fastvideo_args import FastVideoArgs
import torch

from fastvideo.models.stepvideo.utils import VideoProcessor

VAE_PATH = "data/stepvideo-t2v/vae"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
precision = torch.bfloat16
precision_str = "bf16"
args = FastVideoArgs(model_path=VAE_PATH, vae_precision=precision_str)
def test_local_decode():

        
    # model_name = "vae_v2.safetensors"
    z_channels = 64

    vae = VAELoader().load(
            model_path      = "data/stepvideo-t2v/vae",
            architecture    = "",                       # ignored
            fastvideo_args  = args).to(device)


    print("VAE loaded dtype:", next(vae.parameters()).dtype, "device:", device)


    latents = torch.load("latents_rank0.pt", map_location="cpu").to(device)          # (B,C,F,H,W)

    print("latents  raw   shape:", latents.shape,
        "min:", latents.min().item(), "max:", latents.max().item())

    with torch.no_grad():
        video_bfctw = vae.decode(latents)   # still [-1,1], B F C H W
    print("decoded  range:", video_bfctw.min().item(), video_bfctw.max().item())

    # os.makedirs(os.path.dirname(fastvideo_args.output_path), exist_ok=True)
    video = video_bfctw
    video_processor = VideoProcessor(args.output_path, '')
    print("storing to ", args.output_path)
    video_processor.postprocess_video(
        video_tensor=video,
        output_file_name="testing",
)

def test_local_decode():
    outputs = torch.load("decoded_latents.pt", map_location="cpu").to(device)
    video_processor = VideoProcessor(args.output_path, '')
    print("storing to ", args.output_path)
    video_processor.postprocess_video(
        video_tensor=outputs,
        output_file_name="testing_2",
    )
    

def compare_encoders():
    # Load remote outputs
    y_remote    = torch.load("prompt_embeds_remote.pth")
    mask_remote = torch.load("prompt_attention_mask_remote.pth")
    clip_remote = torch.load("clip_remote.pth")

    # Load local outputs
    y_local     = torch.load("prompt_embeds.pth")
    mask_local  = torch.load("prompt_attention_mask.pth")
    clip_local  = torch.load("clip.pth")

    # Compare prompt embeddings
    print("=== Prompt Embeds ===")
    print("Shapes remote vs local:", y_remote.shape, y_local.shape)
    diff_emb = (y_remote - y_local).abs()
    print("Max absolute diff:", diff_emb.max().item())
    print("First element remote:", y_remote.flatten()[0].item())
    print("First element local :", y_local.flatten()[0].item(), "\n")

    # Compare attention masks
    print("=== Attention Masks ===")
    print("Equal:", torch.equal(mask_remote, mask_local))
    print("First mask element remote:", mask_remote.flatten()[0].item())
    print("First mask element local :", mask_local.flatten()[0].item(), "\n")

    # Compare CLIP embeddings
    print("=== CLIP Embeds ===")
    print("Shapes remote vs local:", clip_remote.shape, clip_local.shape)
    diff_clip = (clip_remote - clip_local).abs()
    print("Max absolute diff:", diff_clip.max().item())
    print("First element remote:", clip_remote.flatten()[0].item())
    print("First element local :", clip_local.flatten()[0].item(), "\n")

if __name__ == "__main__":
    compare_encoders()
