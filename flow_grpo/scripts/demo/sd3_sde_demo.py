import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "stabilityai/stable-diffusion-3.5-medium"
device = "cuda"

pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
prompt = 'A steaming cup of coffee'
generator = torch.Generator()
generator.manual_seed(42) 
noise_level_list = [0,0.5,0.6,0.7,0.8]
for noise_level in noise_level_list:
    images, _, _, _ = pipeline_with_logprob(pipe,prompt,num_inference_steps=10,guidance_scale=4.5,output_type="pt",height=512,width=512,generator=generator,noise_level=noise_level)
    pil = Image.fromarray((images[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    pil.save(f'scripts/demo/forward_sde-noise_level{noise_level}.png') 
