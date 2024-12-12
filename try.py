import torch
from fastvideo.models.hunyuan.diffusion.pipelines.pipeline_hunyuan_video import (
    HunyuanVideoPipeline,
)
from fastvideo.models.hunyuan.modules.models import HYVideoDiffusionTransformer
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D


transformer = HYVideoDiffusionTransformer.from_pretrained(
    "data/hyvideo-diffusers", torch_dtype=torch.bfloat16, subfolder="transformer"
)
vae = AutoencoderKLCausal3D.from_pretrained(
    "data/hyvideo-diffusers", torch_dtype=torch.float16, subfolder="vae"
)

pipe = HunyuanVideoPipeline.from_pretrained(
    "data/hyvideo-diffusers", transformer=transformer, vae=vae
)
pipe = pipe.to("cuda")
pipe.vae.enable_tiling()

prompt = "Close-up, A little girl wearing a red hoodie in winter strikes a match. The sky is dark, there is a layer of snow on the ground, and it is still snowing lightly. The flame of the match flickers, illuminating the girl's face intermittently."

result = pipe(
    prompt,
    height=512,
    width=512,
    video_length=29,
)

import PIL.Image
from diffusers.utils import export_to_video

output = result.videos[0].permute(1, 2, 3, 0).detach().cpu().numpy()
output = (output * 255).clip(0, 255).astype("uint8")
output = [PIL.Image.fromarray(x) for x in output]

export_to_video(output, "output.mp4", fps=24)
