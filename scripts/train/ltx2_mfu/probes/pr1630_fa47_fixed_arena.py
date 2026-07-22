#!/usr/bin/env python3
"""Current-head packed LTX-2 fixed-arena gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist


EXPECTED_HEAD = "fa47ce1ab570d33bb245a49f4cd63267282b2a54"
BASE_PATH = Path(__file__).with_name("zero2_ltx2_input_probe.py")

spec = importlib.util.spec_from_file_location("zero2_ltx2_input_probe", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load fixed-arena base harness: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.EXPECTED_SHA = EXPECTED_HEAD

_original_init = base.LTX2FixedArenaZero2.__init__


def _checked_init(self: Any, transformer: torch.nn.Module, **kwargs: Any) -> None:
    names = [name for name, parameter in transformer.named_parameters() if parameter.requires_grad]
    packed_qkv = [name for name in names if ".attn1.to_qkv." in name]
    packed_kv = [name for name in names if ".attn2.to_kv." in name]
    split = [
        name for name in names
        if any(token in name for token in (".attn1.to_q.", ".attn1.to_k.", ".attn1.to_v.",
                                           ".attn2.to_k.", ".attn2.to_v."))
    ]
    if len(names) != 927 or len(packed_qkv) != 96 or len(packed_kv) != 96 or split:
        raise RuntimeError(
            "packed LTX-2 projection layout was not active: "
            f"params={len(names)} qkv={len(packed_qkv)} kv={len(packed_kv)} split={len(split)}")

    _original_init(self, transformer, **kwargs)
    fp32_fields = ("master", "master_grad", "exp_avg", "exp_avg_sq", "step")
    if self.param_arena.dtype != torch.bfloat16 or self.grad_arena.dtype != torch.bfloat16:
        raise RuntimeError("fixed working parameter and gradient arenas must be BF16")
    if any(getattr(bucket, field).dtype != torch.float32 for bucket in self.buckets for field in fp32_fields):
        raise RuntimeError("fixed-arena masters, gradients, moments, and steps must be FP32")
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            "FIXED_ARENA_LAYOUT " + json.dumps({
                "trainable_parameter_objects": len(names),
                "packed_qkv_parameters": len(packed_qkv),
                "packed_kv_parameters": len(packed_kv),
                "split_projection_parameters": len(split),
                "arena_numel": self.param_arena.numel(),
                "working_dtype": str(self.param_arena.dtype),
                "master_dtype": str(self.buckets[0].master.dtype),
                "master_precision_delta_max": self.master_precision_delta_max,
                "moment_dtype": str(self.buckets[0].exp_avg.dtype),
            }, sort_keys=True),
            flush=True,
        )


base.LTX2FixedArenaZero2.__init__ = _checked_init


if __name__ == "__main__":
    base.main()
