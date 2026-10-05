"""Reproduce normalized image latents from exact pipeline inputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21 as OfficialVAE
from fastvideo.configs.pipelines.qwen_image21 import QwenImage21PipelineConfig
from fastvideo.models.loader.component_loader import VAELoader
from fastvideo.pipelines.basic.qwen_image21.inputs import normalize_latents, pack_latents

ROOT = Path('/root/fv/outputs/qwen_image21')
MODEL = Path((ROOT / 'model-path.txt').read_text().strip())
ARTIFACTS = ROOT / 'parity4-diagnostics/test_qwen_image21_modes_1024_p0'
data = {s: torch.load(ARTIFACTS / (s + '.pt'), map_location='cpu', weights_only=True)['diagnostics']
        for s in ('native', 'reference')}
args = SimpleNamespace(pipeline_config=QwenImage21PipelineConfig(), model_paths={}, vae_cpu_offload=False)
args.pipeline_config.vae_config.use_tiling = True
native = VAELoader().load(str(MODEL / 'vae'), args).eval()
official = OfficialVAE.from_pretrained(MODEL / 'vae', local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
official.enable_tiling()
report = {}

def compare(name, actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    e = (a - b).abs()
    row = dict(mean_abs=e.mean().item(), max_abs=e.max().item(), rmse=e.square().mean().sqrt().item(),
               mismatched=int(torch.count_nonzero(e)))
    report[name] = row
    print(name, json.dumps(row), flush=True)

def official_normalize(z):
    mean = torch.tensor(official.config.latents_mean).view(1,64,1,1,1).to(z.device,z.dtype)
    std = torch.tensor(official.config.latents_std).view(1,64,1,1,1).to(z.device,z.dtype)
    return (z-mean)/std

with torch.no_grad():
    native_pixels = data['native']['reference_pixels'][0].to(device='cuda', dtype=torch.bfloat16)
    reference_pixels = data['reference']['reference_pixels'][0].to(device='cuda', dtype=torch.bfloat16)
    print('GPU input strides', native_pixels.stride(), reference_pixels.stride(), flush=True)
    native_z = native.encode(native_pixels).latent_dist.mode()
    ref_z = official.encode(reference_pixels).latent_dist.mode()
    compare('posterior_different_input_strides', native_z, ref_z)
    native_norm = pack_latents(normalize_latents(native_z, native.config.latents_mean, native.config.latents_std))
    ref_norm = pack_latents(official_normalize(ref_z))
    compare('normalized_native_vs_reference', native_norm, ref_norm)
    for s, value in [('native', native_norm), ('reference', ref_norm)]:
        for t in ('native', 'reference'):
            compare(s + '_recomputed_vs_pipeline_' + t, value, data[t]['image_latents'])
    compare('normalization_same_posterior',
            normalize_latents(ref_z, native.config.latents_mean, native.config.latents_std), official_normalize(ref_z))

(ROOT / 'reference-latent-diagnostics.json').write_text(json.dumps(report, indent=2) + '\n')
