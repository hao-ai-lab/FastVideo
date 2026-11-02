# SPDX-License-Identifier: Apache-2.0
"""Encode a public video using the released Wan VAE and text encoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu
from diffusers import AutoencoderKLWan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=448)
    parser.add_argument('--prompt', default='A robot arm pours liquid into a cup on a table.')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.npu.set_device(0)
    if args.text:
        from transformers import AutoTokenizer, UMT5EncoderModel

        sample = torch.load(args.output, map_location='cpu', weights_only=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder='tokenizer', local_files_only=True)
        encoder = UMT5EncoderModel.from_pretrained(args.model, subfolder='text_encoder',
                                                   torch_dtype=torch.bfloat16, local_files_only=True).to('npu').eval()
        tokens = tokenizer([args.prompt, ''], padding='max_length', max_length=512,
                           truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = encoder(input_ids=tokens.input_ids.to('npu'),
                                 attention_mask=tokens.attention_mask.to('npu')).last_hidden_state
        lengths = tokens.attention_mask.sum(dim=1).tolist()
        for index, length in enumerate(lengths):
            embeddings[index, length:] = 0
        sample['prompt_embeds'] = embeddings[:1].cpu()
        sample['negative_prompt_embeds'] = embeddings[1:].cpu()
        sample['prompt'] = args.prompt
        sample['text_token_count'] = lengths[0]
        sample['conditioning'] = 'public Wan UMT5 encoder'
        torch.save(sample, args.output)
        print(json.dumps({'output': args.output, 'text_tokens': lengths}))
        return
    if args.frames % 4 != 1 or args.height % 16 or args.width % 16:
        raise ValueError('Wan inputs require 4n+1 frames and spatial dimensions divisible by 16')
    capture = cv2.VideoCapture(args.video)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected = np.linspace(0, total_frames - 1, args.frames, dtype=int)
    frames = []
    for index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {index} from {args.video}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
    vae = AutoencoderKLWan.from_pretrained(args.model, subfolder='vae', torch_dtype=torch.float32,
                                         local_files_only=True).to('npu').eval()
    with torch.no_grad():
        latents = vae.encode(video.to('npu')).latent_dist.mode()
        mean = torch.tensor(vae.config.latents_mean, device='npu').view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device='npu').view(1, -1, 1, 1, 1)
        latents = (latents - mean) / std
    sample = {'latents': latents.cpu(), 'video': args.video, 'frame_indices': selected.tolist(),
              'height': args.height, 'width': args.width, 'prompt': args.prompt}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, args.output)
    print(json.dumps({'output': args.output, 'latent_shape': list(latents.shape),
                      'frame_indices': selected.tolist(), 'source_video': args.video}))


if __name__ == '__main__':
    main()
