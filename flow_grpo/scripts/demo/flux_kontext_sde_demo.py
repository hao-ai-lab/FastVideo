import torch
from PIL import Image
import numpy as np
from diffusers import FluxKontextPipeline
from flow_grpo.diffusers_patch.flux_kontext_pipeline_with_logprob import pipeline_with_logprob
import importlib
from diffusers.utils import load_image

model_id = "black-forest-labs/FLUX.1-Kontext-dev"
device = "cuda"
height = 512
width = 512

pipe = FluxKontextPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
prompt = 'change the cat to a dog'
image = load_image('scripts/script/cat-hat.png')
image = image.resize((height, width))
generator = torch.Generator()
generator.manual_seed(42) 
noise_level_list = [0.5,0.6,0.7,0.8,0.9,1]
for noise_level in noise_level_list:
    images, _, _, _, _, _ = pipeline_with_logprob(pipe,image=image,prompt=prompt,num_inference_steps=6,guidance_scale=2.5,output_type="pt",height=height,width=width,max_area=height*width,generator=generator,noise_level=noise_level,_auto_resize=False)
    pil = Image.fromarray((images[0].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    pil.save(f'scripts/script/forward_sde-flux-kontext-noise_level{noise_level}.png') 