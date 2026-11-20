# SPDX-License-Identifier: Apache-2.0
"""Load fine-tuned transformer weights and render a Wan video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--trained-transformer', required=True)
    parser.add_argument('--training-sample', required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    sample = torch.load(args.training_sample, map_location='cpu', weights_only=True)
    transformer = WanTransformer3DModel.from_pretrained(args.trained_transformer,
                                                        torch_dtype=torch.bfloat16,
                                                        local_files_only=True)
    transformer.set_attention_backend('_native_npu')
    vae = AutoencoderKLWan.from_pretrained(args.model,
                                           subfolder='vae',
                                           torch_dtype=torch.float32,
                                           local_files_only=True)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction',
                                        use_flow_sigmas=True,
                                        num_train_timesteps=1000,
                                        flow_shift=5.0)
    pipeline = WanPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=vae,
                           scheduler=scheduler).to('npu')
    frames = pipeline(prompt_embeds=sample['prompt_embeds'].to('npu', torch.bfloat16),
                      negative_prompt_embeds=sample['negative_prompt_embeds'].to('npu', torch.bfloat16),
                      height=sample['height'],
                      width=sample['width'],
                      num_frames=(sample['latents'].shape[2] - 1) * 4 + 1,
                      num_inference_steps=args.steps,
                      guidance_scale=5.0,
                      generator=torch.Generator(device='cpu').manual_seed(args.seed),
                      output_type='np').frames[0]
    if not np.isfinite(frames).all():
        raise RuntimeError('Generated video contains non-finite pixels')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 16, (sample['width'], sample['height']))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video output {args.output}')
    for frame in frames:
        writer.write(cv2.cvtColor((frame * 255).round().clip(0, 255).astype('uint8'), cv2.COLOR_RGB2BGR))
    writer.release()
    metadata = {
        'transformer': args.trained_transformer,
        'prompt': sample['prompt'],
        'base_model': args.model,
        'training_sample': args.training_sample,
        'scheduler': 'UniPCMultistepScheduler',
        'flow_shift': 5.0,
        'guidance_scale': 5.0,
        'steps': args.steps,
        'seed': args.seed,
        'frames': len(frames),
        'output': args.output,
        'height': sample['height'],
        'width': sample['width'],
        'pixel_std': float(frames.std()),
        'torch_npu': torch_npu.__version__
    }
    Path(args.output).with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
