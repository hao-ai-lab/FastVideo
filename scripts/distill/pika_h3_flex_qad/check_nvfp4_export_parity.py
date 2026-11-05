# SPDX-License-Identifier: Apache-2.0
"""Parity of a packed NVFP4 MiniMax-H3 DiT export against the QAD numerics (1 GPU, no pipeline).

For sampled blocks and every packed linear (q/k/v/out/fc_in/fc_out, plus the VSA gate when present):
  * the export's packed weight, block scales and alpha are byte-identical to what the QAD harness's ``nvfp4_gemm``
    quantizes from the bf16 master (``--bf16``);
  * the deployed ``NVFP4QuantizeMethod.apply`` output equals ``nvfp4_gemm`` bit for bit on the same input;
  * FFN static input scales equal ``448 * 6 / amax`` from the pinned table (attention linears use the unit scale).
When the export has VSA gates, also checks every block's gate master is byte-identical to ``--original`` (frozen
gates). Prints one JSON line per check and a final summary; exits non-zero on any mismatch.

    python scripts/distill/pika_h3_flex_qad/check_nvfp4_export_parity.py --export exports/ref2va/transformer_ref \\
        --bf16 exports/ref2va/transformer_bf16 --amax examples/train/calibration/pika_h3_flex/amax-ref2va-balanced.json
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

SUBS = {
    "attn.to_q": "attn.to_q.weight",
    "attn.to_k": "attn.to_k.weight",
    "attn.to_v": "attn.to_v.weight",
    "attn.to_out": "attn.to_out.0.weight",
    "attn.to_gate_compress": "attn.to_gate_compress.weight",
    "ff.fc_in": "ff.net.0.proj.weight",
    "ff.fc_out": "ff.net.2.weight",
}
FFN_UNIT = 448.0 * 6.0


def tensor_loader(directory: Path):
    from safetensors import safe_open
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    def load(key: str) -> torch.Tensor:
        with safe_open(str(directory / index[key]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    return load


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", required=True, type=Path, help="exported transformer dir (nvfp4_weights.safetensors)")
    parser.add_argument("--bf16", required=True, type=Path, help="bf16 QAD transformer dir the export was packed from")
    parser.add_argument("--original", type=Path, help="pre-QAD trunk dir (only needed for exports with VSA gates)")
    parser.add_argument("--amax", required=True, type=Path, help="pinned activation amax table")
    parser.add_argument("--blocks", type=int, nargs="+", default=[0, 25, 49])
    parser.add_argument("--num-blocks", type=int, default=50, help="transformer blocks the export must cover")
    args = parser.parse_args()
    from safetensors import safe_open

    from fastvideo.layers.quantization.nvfp4_config import NVFP4QuantizeMethod, _nvfp4_quantize, _require_flashinfer
    from fastvideo.layers.quantization.nvfp4_qad import nvfp4_gemm, weight_global_sf

    amax = json.loads(args.amax.read_text())
    # Calibration tables key FFN linears as b<block>.<sub> (Flex) or by full prefix (FastH3 OmniRef).
    amax = {("transformer_blocks." + k[1:].partition(".")[0] + "." + k.partition(".")[2] if k.startswith("b") else k): v
            for k, v in amax.items()}
    export = safe_open(str(args.export / "nvfp4_weights.safetensors"), framework="pt")
    names = set(export.keys())
    metadata = export.metadata() or {}
    failures = 0
    skip = json.loads(metadata.get("skip_blocks", "[]"))
    print(json.dumps({"check": "skip_blocks_metadata", "skip_blocks": skip, "ok": skip == []}), flush=True)
    failures += int(skip != [])
    packed_blocks = {int(n.split(".")[1]) for n in names if n.startswith("transformer_blocks.")}
    print(json.dumps({"check": "packed_blocks", "count": len(packed_blocks), "ok": len(packed_blocks) == args.num_blocks}), flush=True)
    failures += int(len(packed_blocks) != args.num_blocks)

    load_bf16 = tensor_loader(args.bf16)
    has_gate = any(n.split("::")[0].endswith("attn.to_gate_compress") for n in names)
    subs = {sub: key for sub, key in SUBS.items() if has_gate or sub != "attn.to_gate_compress"}
    if has_gate:
        if args.original is None:
            parser.error("this export has VSA gates; pass --original to check they stayed frozen")
        load_orig = tensor_loader(args.original)
        gate_key = "transformer_blocks.{}.attn.to_gate_compress.weight"
        gate_bad = [
            b for b in range(args.num_blocks)
            if not torch.equal(load_bf16(gate_key.format(b)), load_orig(gate_key.format(b)))
        ]
        print(json.dumps({"check": "gate_bf16_identical_to_original", "blocks": args.num_blocks, "differing": gate_bad,
                          "ok": not gate_bad}), flush=True)
        failures += len(gate_bad)

    sf_layout = _require_flashinfer()[0]
    torch.manual_seed(0)
    for block in args.blocks:
        for sub, wkey in subs.items():
            name = f"transformer_blocks.{block}.{sub}"
            fields = {k: export.get_tensor(f"{name}::{k}").cuda() for k in
                      ("_nvfp4_weight", "_nvfp4_weight_scale", "_weight_global_sf", "_nvfp4_alpha")}
            layer = types.SimpleNamespace(**fields)
            sf_key = f"{name}::_nvfp4_input_global_sf"
            static = sf_key in names
            if static:
                layer._nvfp4_input_global_sf = export.get_tensor(sf_key).float().cuda()
            weight = load_bf16(f"transformer_blocks.{block}.{wkey}").to("cuda", torch.bfloat16).contiguous()
            input_sf = layer._nvfp4_input_global_sf if static else torch.tensor(1.0, device="cuda")
            expected_static = name in amax
            scale_ok = static == expected_static and (not static or torch.equal(
                input_sf.reshape(()), torch.tensor(FFN_UNIT / amax[name], device="cuda", dtype=torch.float32)) or
                bool(torch.isclose(input_sf.reshape(()), torch.tensor(FFN_UNIT / amax[name], device="cuda"), rtol=1e-6)))
            w_sf = weight_global_sf(weight)
            w_fp4, w_scale = _nvfp4_quantize(weight, w_sf, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
            bytes_ok = (torch.equal(w_fp4.view(torch.uint8), fields["_nvfp4_weight"].view(torch.uint8))
                        and torch.equal(w_scale.view(torch.uint8), fields["_nvfp4_weight_scale"].view(torch.uint8))
                        and torch.equal((1.0 / w_sf).float().reshape(()), fields["_nvfp4_alpha"].float().reshape(())))
            x = (torch.randn(1024, weight.shape[1], device="cuda") * 2.0).to(torch.bfloat16)
            with torch.no_grad():
                deployed = NVFP4QuantizeMethod(name).apply(layer, x, None)
                ours = nvfp4_gemm(x, weight, None, input_sf)
            max_abs = float((deployed.float() - ours.float()).abs().max())
            ok = bytes_ok and scale_ok and max_abs == 0.0 and torch.equal(deployed, ours)
            failures += int(not ok)
            print(json.dumps({"check": f"b{block}.{sub}", "static_input_scale": static, "scale_ok": bool(scale_ok),
                              "packed_bytes_identical": bool(bytes_ok), "gemm_max_abs_diff": max_abs, "ok": bool(ok)}),
                  flush=True)
    print(json.dumps({"summary": "PASS" if not failures else "FAIL", "failures": failures}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
