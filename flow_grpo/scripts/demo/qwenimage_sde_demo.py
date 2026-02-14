import torch
from PIL import Image
import numpy as np
from diffusers import DiffusionPipeline
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob
import importlib

model_id = "Qwen/Qwen-Image"
device = "cuda"

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to(device)
prompt = ["a photo of a cat",
    '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197".''',
"a cat"
]
prompt = [p + positive_magic["en"] for p in prompt]
negative_prompt = [" "]*len(prompt)
width=1664
height=928
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
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
noise_level_list = [0.9,1.0,1.1,1.2,1.3]
for noise_level in noise_level_list:
    images = pipeline_with_logprob(pipe,prompt,negative_prompt=negative_prompt, num_inference_steps=10,true_cfg_scale=4,height=height,width=width,generator=generator,noise_level=noise_level)
    for i, img in enumerate(images['images']):
        img.save(f'scripts/demo/forward_sde-qwenimage-noise_level{noise_level}_{i}.png') 
