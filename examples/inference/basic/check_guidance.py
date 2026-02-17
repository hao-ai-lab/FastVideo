"""
Quick check to see what guidance parameters FastVideo and SGLang use.
"""
import sys
sys.path.insert(0, '/FastVideo')

from fastvideo import VideoGenerator
from fastvideo.configs.sample import FluxSamplingParam

# Check FastVideo defaults
gen = VideoGenerator.from_pretrained(
    model_path="black-forest-labs/FLUX.1-dev",
    num_gpus=1,
)

print("="*60)
print("FASTVIDEO CONFIGURATION")
print("="*60)
config = gen.fastvideo_args.pipeline_config
print(f"embedded_cfg_scale: {config.embedded_cfg_scale}")
print(f"flow_shift: {config.flow_shift}")
print(f"flux_shift: {config.flux_shift}")
print(f"flux_base_shift: {config.flux_base_shift}")
print(f"flux_max_shift: {config.flux_max_shift}")

sampling_param = FluxSamplingParam.from_pretrained("black-forest-labs/FLUX.1-dev")
print(f"\nDefault guidance_scale: {sampling_param.guidance_scale}")
print(f"Default seed: {sampling_param.seed}")
print(f"Default num_inference_steps: {sampling_param.num_inference_steps}")

print("\n" + "="*60)
print("SGLANG PARAMETERS (from flux_sgl.py)")
print("="*60)
print("guidance_scale: 1.0")
print("embedded_guidance_scale: 3.5")
print("seed: 42")
print("steps: 50")

print("\n" + "="*60)
print("KEY DIFFERENCE")
print("="*60)
print(f"FastVideo embedded_cfg_scale: {config.embedded_cfg_scale} (0.0035)")
print(f"SGLang embedded_guidance_scale: 3.5")
print(f"Ratio: {3.5 / config.embedded_cfg_scale:.1f}x difference!")
