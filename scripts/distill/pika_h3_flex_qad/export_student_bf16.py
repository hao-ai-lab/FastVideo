# SPDX-License-Identifier: Apache-2.0
"""Write a QAD student's trained weights as a bf16 diffusers-layout transformer directory (CPU only).

Reads the student's fp32 masters straight from a QAD DCP checkpoint (no model build, no GPU, no distributed init) and
rewrites the merged Flex trunk's bf16 shards with every trained tensor replaced by its bf16-rounded master. This is the
bf16 source the packed-NVFP4 converter (``convert_minimax_h3_modelopt_nvfp4_dit.py --quantize-*``) expects, and the
rounding the QAD linears apply (``weight.to(bfloat16)``) before RTN quantization. Untrained tensors are copied from
the base shards unchanged.

    python scripts/distill/pika_h3_flex_qad/export_student_bf16.py \\
        --dcp outputs/pika_flex_ref2va_nvfp4_qad/best/dcp --base data/models/pika-flex-ref2va-merged \\
        --trunk transformer_ref --out exports/ref2va/transformer_bf16
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

PREFIX = "roles.student.transformer."
# Training-time module name -> diffusers checkpoint name, for the trained linear families.
_RENAMES = ((re.compile(r"\.ff\.fc_in\."), ".ff.net.0.proj."), (re.compile(r"\.ff\.fc_out\."), ".ff.net.2."),
            (re.compile(r"\.attn\.to_out\."), ".attn.to_out.0."))
_WRAPPER = "_checkpoint_wrapped_module."


def diffusers_name(internal: str) -> str:
    name = internal.replace(_WRAPPER, "")
    for pattern, replacement in _RENAMES:
        name = pattern.sub(replacement, name)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dcp", required=True, help="checkpoint dir's dcp/ folder")
    parser.add_argument("--base", required=True, help="merged Flex model dir the QAD run started from")
    parser.add_argument("--trunk", default="transformer", choices=("transformer", "transformer_ref"),
                        help="trunk the run trained (transformer_ref for Ref2VA, transformer for FL2VA)")
    parser.add_argument("--out", required=True, help="output transformer dir")
    parser.add_argument("--list", action="store_true", help="print DCP student keys and exit")
    args = parser.parse_args()
    import torch.distributed.checkpoint as dcp
    from safetensors import safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import FileSystemReader

    reader = FileSystemReader(args.dcp)
    metadata = reader.read_metadata().state_dict_metadata
    student = {k[len(PREFIX):]: v for k, v in metadata.items() if k.startswith(PREFIX)}
    if args.list:
        for key in sorted(student)[:40]:
            print(key, getattr(student[key], "size", None))
        print(len(student), "student tensors")
        return
    if not student:
        raise SystemExit(f"{args.dcp} has no {PREFIX}* tensors; is it a student QAD checkpoint?")
    by_diffusers = {diffusers_name(k): k for k in student}
    base = Path(args.base) / args.trunk
    index = json.loads((base / "diffusion_pytorch_model.safetensors.index.json").read_text())
    unmatched = sorted(set(by_diffusers) - set(index["weight_map"]))
    if unmatched:
        raise SystemExit(f"{len(unmatched)} student tensors have no {args.trunk} weight in {base}, e.g. "
                         f"{unmatched[:3]}; check --trunk and --base")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(set(index["weight_map"].values()))
    replaced = same = missing = 0
    max_untrained_diff = 0.0
    for shard in shards:
        tensors = {}
        with safe_open(str(base / shard), framework="pt") as handle:
            keys = list(handle.keys())
            wanted = {key: by_diffusers[key] for key in keys if key in by_diffusers}
            if wanted:
                state = {PREFIX + internal: torch.empty(student[internal].size, dtype=torch.float32)
                         for internal in wanted.values()}
                dcp.load(state, storage_reader=reader)
            for key in keys:
                base_tensor = handle.get_tensor(key)
                if key in wanted:
                    master = state[PREFIX + wanted[key]]
                    tensors[key] = master.to(base_tensor.dtype).contiguous()
                    if torch.equal(tensors[key], base_tensor):
                        same += 1
                    else:
                        replaced += 1
                else:
                    tensors[key] = base_tensor
                    missing += 1
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(json.dumps({"shard": shard, "tensors": len(tensors), "trained_in_shard": len(wanted)}), flush=True)
    for name in ("diffusion_pytorch_model.safetensors.index.json", "config.json"):
        shutil.copyfile(base / name, out / name)
    print(json.dumps({"changed_vs_base": replaced, "identical_to_base": same, "not_in_dcp": missing}))


if __name__ == "__main__":
    main()
