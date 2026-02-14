import torch
from PIL import Image
from pathlib import Path

from diffusers import QwenImageEditPipeline
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob

base_model_path = "Qwen/Qwen-Image-Edit" 
lora_checkpoint_path = "logs/pickscore/qwenimage_edit/checkpoint-84/model.safetensors" 
input_image_path = "scripts/demo/cat.png" 
output_dir = "scripts/demo/"
device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = ["Change the cat's color to purple, with a flash light background."]
negative_prompt = [" "] * len(prompt)
width = 512
height = 512
num_inference_steps = 40
true_cfg_scale = 4.0
seed = 42

pipe = QwenImageEditPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16
)

target_modules = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2",
]
lora_config = LoraConfig(
    r=64, lora_alpha=128, init_lora_weights="gaussian", target_modules=target_modules
)
pipe.transformer = get_peft_model(pipe.transformer, lora_config)

lora_state_dict = load_file(lora_checkpoint_path, device="cpu")
filtered_lora_state_dict = {}
for key, value in lora_state_dict.items():
    if "lora_" in key:
        new_key = key.replace("base_model.model.", "")
        filtered_lora_state_dict[new_key] = value

if not filtered_lora_state_dict:
    raise ValueError(f"在 {lora_checkpoint_path} 中未找到任何 LoRA 权重 (键中包含 'lora_')。")

set_peft_model_state_dict(pipe.transformer, filtered_lora_state_dict)

pipe.transformer = pipe.transformer.merge_and_unload()
pipe = pipe.to(device)

image = Image.open(input_image_path).convert("RGB")
image = image.resize((width, height), Image.BICUBIC)
image = [image] 
generator = torch.Generator(device=device).manual_seed(seed)

with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device=="cuda")):
    results = pipeline_with_logprob(
        pipe,
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        height=height,
        width=width,
        generator=generator,
        noise_level=0.0, 
    )
generated_images = results['images']

output_path.mkdir(parents=True, exist_ok=True)
output_path = Path(output_dir)
for i, img in enumerate(generated_images):
    save_path = output_path / f"qwenimage_edit_lora_inference_example_{i}.png"
    img.save(save_path)