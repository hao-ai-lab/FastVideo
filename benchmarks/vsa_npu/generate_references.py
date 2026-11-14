"""Generate complete local primary references into a NEW output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from reference import full_vsa

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
output = Path(a.output)
output.mkdir(parents=True, exist_ok=False)
torch.npu.set_device(0)
torch.set_num_threads(4)
os.environ['VSA_NATIVE_GATHER'] = '1'
os.environ['VSA_GATHER_BATCH'] = '32'
records = []
with torch.inference_mode():
    for seed in [1024, 1025, 1026]:
        gen = torch.Generator().manual_seed(seed)
        cpu = [(torch.randn(1, 32760, 12, 128, generator=gen) * .3).to(torch.bfloat16) for _ in range(4)]
        digest = hashlib.sha256(b''.join(x.view(torch.uint16).numpy().tobytes() for x in cpu)).hexdigest()
        xs = [x.to('npu') for x in cpu]
        free, _ = torch.npu.mem_get_info()
        assert free * .8 > 32 * 12 * 624 * 64 * 128 * 8
        out, mask = full_vsa(*xs, (21, 30, 52), .8)
        out = out.cpu()
        mask = mask.cpu()
        path = output / f'f81-d128-s80-bfloat16-seed{seed}.pt'
        torch.save({'output': out, 'mask': mask, 'input_hash': digest, 'seed': seed}, path)
        records.append({
            'seed':
            seed,
            'input_hash':
            digest,
            'path':
            str(path),
            'sha256':
            hashlib.sha256(path.read_bytes()).hexdigest(),
            'output_tensor_sha256':
            hashlib.sha256(out.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
            'mask_tensor_sha256':
            hashlib.sha256(mask.numpy().tobytes()).hexdigest()
        })
(output / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
print('Generated', len(records), 'complete references locally')
