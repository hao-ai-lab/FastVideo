# SPDX-License-Identifier: Apache-2.0
"""Shared driver of the latent-upscaler parity test: runs one implementation on the real weights.

Used in two processes, because the new and the previous implementation both live in a package named ``fastvideo``:

    python lu_parity_common.py --impl previous --lu-dir <bundle>/latent_upscaler --out ref.pt   # PYTHONPATH=<old root>

and imported by ``test_kandinsky6_sr_latent_upscaler_parity.py`` for the current implementation.  Both sides build
the bank on the meta device, load the checkpoint with ``strict=True, assign=True`` and run the same inputs, so any
difference comes from the module code.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections.abc import Callable

import torch

LATENT_SHAPE = (1, 64, 5, 26, 48)  # one 5-frame KVAE latent tile of a ~416x768 clip
SEED = 0
SCALES = (2, 4)
DTYPES = ("fp32", "bf16")


def load_checkpoint(lu_dir: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(lu_dir, "*.safetensors"))):
        state.update(load_file(path))
    return state


def backend_flags() -> dict[str, bool]:
    return {
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def run_outputs(bank: torch.nn.Module, forward: Callable[[torch.nn.Module, torch.Tensor, int], torch.Tensor],
                device: str = "cuda") -> dict[str, torch.Tensor]:
    """``{"<dtype>_x<scale>": output}``: fp32 weights without autocast, then bf16 weights under bf16 autocast (the
    production LU call: input cast to the module dtype, ``torch.autocast("cuda", bfloat16)``)."""
    torch.backends.cudnn.benchmark = False
    z = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(SEED))
    outputs = {}
    with torch.no_grad():
        bank.to(device=device, dtype=torch.float32)
        for scale in SCALES:
            outputs[f"fp32_x{scale}"] = forward(bank, z.to(device), scale).float().cpu()
        bank.to(dtype=torch.bfloat16)
        for scale in SCALES:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs[f"bf16_x{scale}"] = forward(bank, z.to(device, torch.bfloat16), scale).float().cpu()
    return outputs


def _previous(lu_dir: str) -> dict:
    import fastvideo
    from fastvideo.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerBank
    from fastvideo.models.upsamplers.kandinsky6_sr_lu import latent_upscaler_for_scale, run_latent_upscaler

    config = json.load(open(os.path.join(lu_dir, "config.json")))
    with torch.device("meta"):
        bank = Kandinsky6SRLatentUpscalerBank(config)
    bank.load_state_dict(load_checkpoint(lu_dir), strict=True, assign=True)
    bank.eval()
    outputs = run_outputs(bank, lambda b, z, s: run_latent_upscaler(latent_upscaler_for_scale(b, s), z))
    return {
        "fastvideo_file": fastvideo.__file__,
        "outputs": outputs,
        "num_parameters": sum(p.numel() for p in bank.parameters()),
        "state_keys": list(bank.state_dict()),
        "backend_flags": backend_flags(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["previous"], required=True)
    parser.add_argument("--lu-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.save(_previous(args.lu_dir), args.out)


if __name__ == "__main__":
    main()
