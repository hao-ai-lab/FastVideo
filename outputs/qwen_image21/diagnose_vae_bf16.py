"""Compare real BF16 VAE at the reference image size without changing either model."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21 as OfficialVAE
from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.pipelines.basic.qwen_image21.inputs import reference_pixels, resize_reference


ROOT = Path('/root/fv/outputs/qwen_image21')
MODEL = Path((ROOT / 'model-path.txt').read_text().strip())
report = {}


def compare(name, actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    error = (actual - expected).abs()
    row = dict(shape=list(actual.shape), mean_abs=error.mean().item(), max_abs=error.max().item(),
               rmse=error.square().mean().sqrt().item(), mismatched=int(torch.count_nonzero(error)))
    report[name] = row
    print(name, json.dumps(row), flush=True)


args = SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(), model_paths={}, vae_cpu_offload=False)
args.pipeline_config.vae_config.use_tiling = True
native = VAELoader().load(str(MODEL / 'vae'), args).eval()
official = OfficialVAE.from_pretrained(MODEL / 'vae', local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
official.enable_tiling()
print('Native/reference tiling', [(m.tile_sample_min_height, m.tile_sample_min_width,
      m.tile_sample_stride_height, m.tile_sample_stride_width) for m in (native, official)], flush=True)

image = Image.open(ROOT / 'official_assets/1413730.png').convert('RGBA')
resized = resize_reference(image, 1024)
native_pixels = reference_pixels(resized)
processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False)
official_pixels = processor.preprocess(image, height=resized.height, width=resized.width).unsqueeze(2)
compare('preprocessing', native_pixels, official_pixels)

with torch.inference_mode():
    for name, pixels in [('small', torch.rand((1, 4, 1, 64, 80), generator=torch.Generator().manual_seed(23)).mul_(2).sub_(1)),
                         ('full_reference', native_pixels)]:
        pixels = pixels.to(device='cuda', dtype=torch.bfloat16)
        actual = native.encode(pixels).latent_dist
        expected = official.encode(pixels).latent_dist
        compare(name + '_posterior_mean', actual.mean, expected.mean)
        compare(name + '_posterior_logvar', actual.logvar, expected.logvar)
        actual_decode = native.decode(expected.mode()).sample
        expected_decode = official.decode(expected.mode()).sample
        compare(name + '_decode_identical_latents', actual_decode, expected_decode)
        compare(name + '_postprocess', actual_decode.div(2).add(.5).clamp(0, 1),
                processor.postprocess(expected_decode[:, :, 0], output_type='pt').unsqueeze(2))

(ROOT / 'vae-bf16-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
