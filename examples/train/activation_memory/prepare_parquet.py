# SPDX-License-Identifier: Apache-2.0
"""Write public video latents and text embeddings in FastVideo's T2V format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--rows', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sample = torch.load(args.sample, map_location='cpu', weights_only=True)
    vae = json.loads((Path(args.model) / 'vae/config.json').read_text())
    mean = torch.tensor(vae['latents_mean']).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae['latents_std']).view(1, -1, 1, 1, 1)
    latents = (sample['latents'] * std + mean)[0].float().contiguous().numpy()
    text = sample['prompt_embeds'][0, :sample['text_token_count']].float().contiguous().numpy()
    row = {
        'vae_latent_bytes': latents.tobytes(),
        'vae_latent_shape': list(latents.shape),
        'vae_latent_dtype': 'float32',
        'text_embedding_bytes': text.tobytes(),
        'text_embedding_shape': list(text.shape),
        'text_embedding_dtype': 'float32',
        'file_name': sample['video'],
        'caption': sample['prompt'],
        'media_type': 'video',
        'width': sample['width'],
        'height': sample['height'],
        'num_frames': len(sample['frame_indices']),
        'duration_sec': len(sample['frame_indices']) / 16,
        'fps': 16.0,
    }
    rows = [{'id': f'{Path(args.sample).stem}-{index}', **row} for index in range(args.rows)]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    print(
        json.dumps({
            'output': args.output,
            'rows': args.rows,
            'unique_videos': 1,
            'raw_latent_shape': list(latents.shape),
            'text_tokens': text.shape[0]
        }))


if __name__ == '__main__':
    main()
