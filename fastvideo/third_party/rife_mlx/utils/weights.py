"""Model build + weight load (HF auto-download / local override).

Resolution: weights_dir arg | $RIFE_MLX_WEIGHTS_DIR | dist/<hf_name> | HF
mlx-community/<hf_name>. Each dir: model.safetensors + config.json.
"""

from __future__ import annotations

import os

import mlx.core as mx

from ..config import VERSIONS, VersionConfig
from ..model.RIFE_HDv3 import Model

WEIGHTS_DIR_ENV = "RIFE_MLX_WEIGHTS_DIR"
HF_ORG = "mlx-community"


def resolve_repo_id(version: str) -> str:
    return f"{HF_ORG}/{VERSIONS[version].hf_name}"


def _resolve_dir(cfg: VersionConfig, weights_dir: str | None) -> str:
    if weights_dir:
        return weights_dir
    env = os.environ.get(WEIGHTS_DIR_ENV)
    if env and os.path.isdir(env):
        return env
    local = os.path.join("dist", cfg.hf_name)
    if os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=f"{HF_ORG}/{cfg.hf_name}")


def build_model(version: str = "4.25", weights_dir: str | None = None) -> Model:
    cfg = VERSIONS[version]
    wdir = _resolve_dir(cfg, weights_dir)
    weights = dict(mx.load(os.path.join(wdir, "model.safetensors")).items())
    model = Model(scale_list=cfg.scale_list, pad_to=cfg.pad_to)
    model.flownet.load_weights(list(weights.items()), strict=True)
    mx.eval(model.flownet.parameters())
    return model
