import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(
  pipe.scheduler.config,
  flow_shift=3.0
)
pipe.to("cuda")
pipe.load_lora_weights("benjamin-paine/steamboat-willie-1.3b")
pipe.enable_model_cpu_offload() # for low-vram environments
breakpoint()

prompt = "steamboat willie style, golden era animation, an anthropomorphic cat character wearing a hat removes it and performs a courteous bow"
output = pipe(
    prompt=prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=32
).frames[0]
export_to_video(output, "output.mp4", fps=16)