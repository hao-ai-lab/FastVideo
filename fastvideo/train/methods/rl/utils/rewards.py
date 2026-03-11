# SPDX-License-Identifier: Apache-2.0
"""Reward function loading and composition for RL training."""

from __future__ import annotations

import importlib
import inspect
import time
from collections.abc import Callable
from contextlib import contextmanager

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.reward import (
    hpsv3_general_score,
    hpsv3_percentile_score,
    video_ocr_score,
    videoalign_mq_score,
    videoalign_ta_score,
)
from fastvideo.train.methods.rl.reward.hpsv3 import (
    set_hpsv3_device,
)
from fastvideo.train.methods.rl.reward.videoalign import (
    set_videoalign_device,
)

logger = init_logger(__name__)

_BUILTIN_REWARDS: dict[str, Callable] = {
    "video_ocr": video_ocr_score,
    "hpsv3_general": hpsv3_general_score,
    "hpsv3_percentile": hpsv3_percentile_score,
    "videoalign_mq": videoalign_mq_score,
    "videoalign_ta": videoalign_ta_score,
}

_GPU_REWARD_NAMES = {
    "hpsv3_general",
    "hpsv3_percentile",
    "videoalign_mq",
    "videoalign_ta",
}


def load_reward_fn(
    name: str,
    device,
    module_path: str | None = None,
):
    """Load a reward function by name."""
    if module_path:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, f"{name}_score", None)
        if fn is None:
            msg = (
                f"Reward {name}_score not found "
                f"in {module_path}"
            )
            raise ValueError(msg)
        return fn(device) if callable(fn) else fn

    if name in _BUILTIN_REWARDS:
        fn = _BUILTIN_REWARDS[name]
        sig = inspect.signature(fn)
        accepts_device = any(
            p.name in {"device", "dev"}
            for p in sig.parameters.values()
        )
        return fn(device) if accepts_device else fn()

    def _zero_fn(images, prompts, metadata, only_strict=False):
        batch = (
            len(prompts) if prompts is not None else 1
        )
        zeros = torch.zeros(batch, device=device)
        return {"avg": zeros}, {}

    return _zero_fn


def multi_score(
    device,
    reward_cfg: dict[str, float],
    module_path: str | None = None,
    return_raw_scores: bool = False,
):
    """Compose multiple reward heads.

    Args:
        device: Device for reward computation.
        reward_cfg: Dict mapping reward name to weight.
        module_path: Optional custom module path.
        return_raw_scores: If True, include raw scores.

    Returns:
        A callable (images, prompts, metadata, only_strict)
        -> (scores_dict, metadata_dict).
    """
    reward_fns = {}
    weights = {}
    for name, weight in reward_cfg.items():
        reward_fns[name] = load_reward_fn(
            name, device, module_path
        )
        weights[name] = weight

    def _fn(images, prompts, metadata, only_strict=True):
        scores = {}
        for name, fn in reward_fns.items():
            out, _meta = fn(images, prompts, metadata)
            if isinstance(out, dict):
                val = out.get(
                    "avg", out.get("reward", out)
                )
            else:
                val = out
            if return_raw_scores:
                scores[f"{name}_raw"] = val
            scores[name] = val * weights[name]
        stacked = torch.stack(
            [scores[name] for name in reward_cfg], dim=0
        )
        scores["avg"] = stacked.mean(0)
        return scores, {}

    return _fn


def _has_reward(reward_cfg, names) -> bool:
    if not reward_cfg:
        return False
    return any(name in reward_cfg for name in names)


def _device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return torch.device(device).type


def move_reward_models(reward_cfg, device) -> None:
    """Move GPU-backed reward models to device."""
    if _has_reward(
        reward_cfg,
        {"hpsv3_general", "hpsv3_percentile"},
    ):
        set_hpsv3_device(device)
    if _has_reward(
        reward_cfg,
        {"videoalign_mq", "videoalign_ta"},
    ):
        set_videoalign_device(device)


@contextmanager
def reward_models_on_device(reward_cfg, device):
    """Temporarily move reward models to device."""
    if _has_reward(reward_cfg, _GPU_REWARD_NAMES):
        use_cuda = _device_type(device) == "cuda"
        _t0 = time.perf_counter()
        move_reward_models(reward_cfg, device)
        if use_cuda:
            torch.cuda.synchronize()
        _t1 = time.perf_counter()
        logger.info(
            "[rewards] move_to_device=%.1fs", _t1 - _t0
        )
        try:
            yield
        finally:
            _t2 = time.perf_counter()
            move_reward_models(reward_cfg, "cpu")
            if use_cuda:
                import gc

                gc.collect()
                torch.cuda.empty_cache()
            _t3 = time.perf_counter()
            logger.info(
                "[rewards] move_to_cpu+gc=%.1fs",
                _t3 - _t2,
            )
    else:
        yield
