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
args = FastVideoArgs(model_path="data/stepvideo-t2v", vae_precision=precision_str)
STEP_LLM_DIR = "data/stepvideo-t2v/step_llm"
HUNYUAN_CLIP_DIR = "data/stepvideo-t2v/hunyuan_clip"
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


def build_llm(self, model_dir, device):
    from fastvideo.v1.models.encoders.stepllm import STEP1TextEncoder
    # from fastvideo.models.stepvideo.text_encoder.stepllm import STEP1TextEncoder
    text_encoder = STEP1TextEncoder(model_dir, max_length=320).to(device).to(torch.bfloat16).eval()
    print("Initialized text encoder...")
    return text_encoder

def build_clip(self, model_dir, device):
    from fastvideo.v1.models.encoders.bert import HunyuanClip
    # from fastvideo.models.stepvideo.text_encoder.clip import HunyuanClip
    clip = HunyuanClip(model_dir, max_length=77).to(device).eval()
    print("Initialized clip encoder...")
    return clip

def load_pairs(llm_dir: str, clip_dir: str, device: torch.device):
    """
    Return four modules:
        text_v1, text_orig, clip_v1, clip_orig
    """
    # ---- STEP-1 text encoder (v1 copy vs original) ----
    from fastvideo.v1.models.encoders.stepllm                     import STEP1TextEncoder as STEP1_v1
    from fastvideo.models.stepvideo.text_encoder.stepllm          import STEP1TextEncoder as STEP1_orig

    text_v1   = STEP1_v1  (llm_dir, 320).to(device).to(torch.bfloat16).eval()
    text_orig = STEP1_orig(llm_dir, 320).to(device).to(torch.bfloat16).eval()

    # ---- CLIP encoder (v1 copy vs original) ----
    from fastvideo.v1.models.encoders.bert                         import HunyuanClip as Clip_v1
    from fastvideo.models.stepvideo.text_encoder.clip              import HunyuanClip as Clip_orig

    clip_v1   = Clip_v1  (clip_dir, 77).to(device).eval()
    clip_orig = Clip_orig(clip_dir, 77).to(device).eval()

    return text_v1, text_orig, clip_v1, clip_orig

@torch.no_grad()
def encode(text_enc, clip_enc, prompts, device):
    y, y_mask = text_enc(prompts)
    clip, _   = clip_enc(prompts)
    # pad mask like remote API
    y_mask = torch.nn.functional.pad(y_mask, (clip.shape[1], 0), value=1)
    return (
        y.to(device=torch.device("cpu")).bfloat16(),
        y_mask.to(device="cpu"),
        clip.to(device="cpu").bfloat16()
    )
   
def diff(a, b):
    return (a - b).abs().max().item(), (a - b).abs().mean().item()


def compare_encoders_local():
    lib_path=os.path.join(args.model_path, 'lib/liboptimus_ths-torch2.5-cu124.cpython-310-x86_64-linux-gnu.so')
    torch.ops.load_library(lib_path)
    prompts="Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_v1, text_orig, clip_v1, clip_orig = load_pairs(
        STEP_LLM_DIR, HUNYUAN_CLIP_DIR, device
    )

    y1, m1, c1 = encode(text_v1,   clip_v1,   prompts, device)
    y0, m0, c0 = encode(text_orig, clip_orig, prompts, device)

    print("=== prompt_embeds ===")
    print("shape:", y0.shape)
    print("max / mean abs diff:", *diff(y0, y1))

    print("\n=== attention_mask ===")
    print("identical:", torch.equal(m0, m1))

    print("\n=== clip_embeds ===")
    print("shape:", c0.shape)
    print("max / mean abs diff:", *diff(c0, c1))

    print("\nfirst few values (prompt_embeds):")
    print("orig :", y0.flatten()[:5].tolist())
    print("v1   :", y1.flatten()[:5].tolist())
     
if __name__ == "__main__":
    compare_encoders_local()
