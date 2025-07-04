import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler)
from diffusers.utils import export_to_video

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id,
                                       subfolder="vae",
                                       torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_id,
                                   vae=vae,
                                   torch_dtype=torch.bfloat16)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
                                                     flow_shift=3.0)
pipe.text_encoder = pipe.text_encoder.to(torch.float32)
pipe.to("cuda")
lora_path = "benjamin-paine/steamboat-willie-1.3b"
pipe.load_lora_weights(lora_path)
prompt = "steamboat willie style, golden era animation, close-up of a short fluffy monster  kneeling beside a melting red candle. the mood is one of wonder and curiosity,  as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression  convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time.  The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
output = pipe(prompt=prompt,
              negative_prompt=negative_prompt,
              height=480,
              width=832,
              num_frames=45,
              guidance_scale=5.0,
              num_inference_steps=32,
              generator=torch.Generator(device="cpu").manual_seed(42)).frames[0]

export_to_video(output, f"{lora_path.split('/')[-1]}_{prompt[:50]}.mp4", fps=24)

lora_path = "motimalu/wan-flat-color-1.3b-v2"
pipe.load_lora_weights(lora_path)
prompt = "flat color, no lineart, blending, negative space, artist:[john kafka|ponsuke kaikai|hara id 21|yoneyama mai|fuzichoco],  1girl, sakura miko, pink hair, cowboy shot, white shirt, floral print, off shoulder, outdoors, cherry blossom, tree shade, wariza, looking up, falling petals, half-closed eyes, white sky, clouds,  live2d animation, upper body, high quality cinematic video of a woman sitting under a sakura tree. Dreamy and lonely, the camera close-ups on the face of the woman as she turns towards the viewer. The Camera is steady, This is a cowboy shot. The animation is smooth and fluid."
negative_prompt = "bad quality video,色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

output = pipe(prompt=prompt,
              negative_prompt=negative_prompt,
              height=480,
              width=832,
              num_frames=45,
              guidance_scale=5.0,
              num_inference_steps=32,
              generator=torch.Generator(device="cpu").manual_seed(42)).frames[0]

export_to_video(output, f"{lora_path.split('/')[-1]}_{prompt[:50]}.mp4", fps=24)
