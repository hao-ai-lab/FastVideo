"""Validate and time candidate kernels on captured real H3 Q/K/V.

Run on an idle sm89 GPU, separately from clip benchmarks. Captures come from
FASTVIDEO_H3_CAPTURE_QKV on the tile-first path; they retain two full heads.
"""
import argparse
import json
import pathlib

import torch
import triton

from fastvideo.attention.backends.minimax_h3_sparse_int8 import sparse_sm89_attention
from fastvideo_kernel.block_sparse_attn import block_sparse_attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    records = []
    with torch.inference_mode():
        for capture in sorted(args.captures.glob("layer-*.pt")):
            state = torch.load(capture, map_location="cuda", weights_only=True)
            q, k, v, mask, vbs = (state[key] for key in ("q", "k", "v", "mask", "vbs"))
            def baseline():
                return block_sparse_attn(q, k, v, mask, vbs)[0]
            expected = baseline()
            for int8_qk, fp8_pv in ((True, True), (True, False), (False, True), (False, False)):
                def candidate():
                    return sparse_sm89_attention(q, k, v, mask, vbs, int8_qk=int8_qk, fp8_pv=fp8_pv)
                output = candidate()
                valid_rows = state["untile"]
                ref = expected.index_select(2, valid_rows).float()
                actual = output.index_select(2, valid_rows).float()
                delta = actual - ref
                reference_ms = triton.testing.do_bench(baseline)
                candidate_ms = triton.testing.do_bench(candidate)
                record = {"capture": capture.name, "shape": list(q.shape), "int8_qk": int8_qk, "fp8_pv": fp8_pv,
                          "mask_density": float(mask.float().mean()),
                          "finite": bool(torch.isfinite(actual).all()),
                          "relative_l2": float(delta.norm() / ref.norm()),
                          "max_abs": float(delta.abs().max()),
                          "cosine": float(torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)),
                          "bf16_ms": reference_ms, "int8_fp8_ms": candidate_ms,
                          "speedup": reference_ms / candidate_ms}
                print(json.dumps(record), flush=True)
                records.append(record)
    if not records:
        raise RuntimeError("No real Q/K/V captures found")
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                       "triton": triton.__version__, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
