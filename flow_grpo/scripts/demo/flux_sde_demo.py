import torch
from PIL import Image
import numpy as np
from diffusers import FluxPipeline
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "black-forest-labs/FLUX.1-dev"
device = "cuda"

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
prompt = 'a cat holding a sign that says hello world'
generator = torch.Generator()
generator.manual_seed(42)
noise_level_list = [0,0.7,0.8,0.9,1.0]
for noise_level in noise_level_list:
    images, _, _, _, _ = pipeline_with_logprob(pipe,prompt,num_inference_steps=6,guidance_scale=3.5,output_type="pt",height=512,width=512,generator=generator,noise_level=noise_level)
    pil = Image.fromarray((images[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    pil.save(f'scripts/script/forward_sde-flux-noise_level{noise_level}.png') 