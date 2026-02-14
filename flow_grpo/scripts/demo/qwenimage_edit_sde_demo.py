import torch
from PIL import Image
import numpy as np
from diffusers import QwenImageEditPipeline
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "Qwen/Qwen-Image-Edit"
device = "cuda"

pipe = QwenImageEditPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

prompt = ["Change the cat's color to purple, with a flash light background."]
negative_prompt = [" "]*len(prompt)
width=512
height=512
image = Image.open("scripts/demo/cat.png").convert("RGB")
image = image.resize((width, height), Image.BICUBIC)
image = [image]
# image = pipe(
#     image=image[0],
#     prompt=prompt[0],
#     negative_prompt=negative_prompt[0],
#     width=width,
#     height=height,
#     num_inference_steps=50,
#     true_cfg_scale=4.0,
#     generator=torch.Generator(device="cuda").manual_seed(42)
# ).images
# for i, img in enumerate(image):
#     img.save(f"scripts/demo/example_{i}.png")
generator = torch.Generator()
generator.manual_seed(42)
noise_level_list = [0.9,1.0,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4]
for noise_level in noise_level_list:
    images = pipeline_with_logprob(pipe,image,prompt,negative_prompt=negative_prompt, num_inference_steps=10,true_cfg_scale=4,sde_window_size=2,height=height,width=width,generator=generator,noise_level=noise_level)
    for i, img in enumerate(images['images']):
        img.save(f'scripts/demo/forward_sdewindow2-qwenimage-noise_level{noise_level}_{i}.png') 
