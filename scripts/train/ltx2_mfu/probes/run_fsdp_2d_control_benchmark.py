#!/usr/bin/env python3
"""Restore the historical 2-D mesh, then execute a benchmark driver."""

from __future__ import annotations

import json
import os
import runpy
import sys

from fastvideo.models.loader import fsdp_load


def _init_2d_mesh(device_type: str, replicate_dim: int, shard_dim: int):
    mesh = fsdp_load.init_device_mesh(
        device_type,
        mesh_shape=(replicate_dim, shard_dim),
        mesh_dim_names=("replicate", "shard"),
    )
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(
            "BF16_FSDP_MESH " + json.dumps({
                "control": "historical_2d",
                "device_type": device_type,
                "mesh_shape": list(mesh.mesh.shape),
                "mesh_dim_names": list(mesh.mesh_dim_names or ()),
                "replicate_dim": replicate_dim,
                "shard_dim": shard_dim,
            }, sort_keys=True),
            flush=True,
        )
    return mesh


fsdp_load._init_fsdp_device_mesh = _init_2d_mesh
target = sys.argv.pop(1)
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
