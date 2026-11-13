# SPDX-License-Identifier: Apache-2.0
"""Run the pinned Linum portrait reference on Apple Metal, isolated from FastVideo.

Invoked by test_pipeline_parity.py. Only the reference's device-specific
operations are adapted. All caption, model, guidance and solver code comes
from the Linum checkout passed by the caller.
"""
import argparse
import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.reference.resolve()))
    from generate import DEFAULT_PROMPT, save_png
    from pjit_21k import PyramidJiT, QwenTextEncoder, SamplerConfig, model, sampler, set_attention_backend
    from pjit_21k.noise import StackedRandomGenerator

    device = torch.device("mps")
    set_attention_backend("sdpa")
    encoder = QwenTextEncoder(args.qwen, device=device)
    captions = {prompt: encoder([prompt])[0] for prompt in (DEFAULT_PROMPT, "")}
    del encoder
    gc.collect()
    torch.mps.empty_cache()

    # MPS and CUDA use different RNG streams. Use the upstream generator on
    # CPU to reproduce the port's documented seed convention on Metal.
    class CPUReferenceNoise(StackedRandomGenerator):
        def __init__(self, device, seeds, sm_count):
            super().__init__("cpu", seeds, sm_count=None)

        def randn(self, size, dtype, device):
            return super().randn(size, dtype=dtype, device="cpu").to(device)

    sampler.StackedRandomGenerator = CPUReferenceNoise
    sampler.amp = SimpleNamespace(autocast=lambda device_type, dtype: torch.autocast("mps", dtype=dtype))

    # Metal cannot execute fp64 operations. Keep Linum's original functions
    # and run those operations on CPU, returning the rounded values to Metal.
    cpu_project = sampler.project
    def project(v0, v1):
        return tuple(part.to(v0.device) for part in cpu_project(v0.cpu(), v1.cpu()))

    cpu_embedding = model.sinusoidal_embedding_1d
    def embed(dim, position, dtype):
        return cpu_embedding(dim=dim, position=position.cpu(), dtype=dtype).to(position.device)

    sampler.project = project
    model.sinusoidal_embedding_1d = embed
    denoiser = PyramidJiT.from_pretrained(args.weights, device="cpu")
    # Upstream uses fp32 real/imaginary components when multiplying rotations.
    denoiser.freqs = denoiser.freqs.to(torch.complex64)
    denoiser = denoiser.to(device)
    image, = sampler.generate(model=denoiser, text_encoder=lambda prompts: [captions[p] for p in prompts],
                              prompt=DEFAULT_PROMPT, seeds=[2], sampler=SamplerConfig(height=1216, width=864,
                                                                                     noise_sm_count=None),
                              device="mps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_png(image, str(args.output_dir / "seed_2.png"))
    (args.output_dir / "prompt.json").write_text(json.dumps({"caption": DEFAULT_PROMPT, "seeds": [2], "upsampled": False}))


if __name__ == "__main__":
    main()
