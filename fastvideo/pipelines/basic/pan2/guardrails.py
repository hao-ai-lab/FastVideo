# SPDX-License-Identifier: Apache-2.0
"""PAN2 guardrails: the prompt and every generated video are checked with the ``pan2_guardrail`` package's
``PAN2SafetyChecker``, as in the vLLM-Omni and Diffusers integrations.

They are on by default (``PAN2PipelineConfig.enable_guardrails``). Either check blocks the request with a
``PAN2GuardrailViolationError``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import torch

from fastvideo.distributed import get_local_torch_device, get_sp_group, model_parallel_is_initialized
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.utils import pixels_to_uint8

logger = init_logger(__name__)

_DISABLE_HINT = "or turn the guardrails off with `enable_guardrails=False` at your own risk."


class PAN2GuardrailViolationError(ValueError):
    """A prompt or a generated video was blocked by the PAN 2 guardrails."""


def _sp_rank_and_size() -> tuple[int, int]:
    if not model_parallel_is_initialized():
        return 0, 1
    group = get_sp_group()
    return group.rank_in_group, group.world_size


class PAN2Guardrails:
    """The checker, loaded on the first rank of each sequence-parallel group only.

    That rank runs every check and broadcasts its verdict, so the ranks of a group raise together and the guard models
    take memory on one GPU, not on all of them.
    """

    def __init__(self, offload_to_cpu: bool = False) -> None:
        if importlib.util.find_spec("pan2_guardrail") is None:
            raise ImportError("PAN 2 runs with guardrails by default, but the pan2-guardrail package is not installed. "
                              "Install it with `pip install git+https://github.com/MBZUAI-IFM/PAN2-guardrail`, " +
                              _DISABLE_HINT)
        self.offload_to_cpu = offload_to_cpu
        self.checker = None
        rank, _ = _sp_rank_and_size()
        if rank == 0:
            from pan2_guardrail import PAN2SafetyChecker

            logger.info("Loading the PAN 2 guardrails (offload_to_cpu=%s)", offload_to_cpu)
            # The checker loads on CPU. With offloading, each guard moves to the GPU only for its check.
            self.checker = PAN2SafetyChecker.from_pretrained()
            if not offload_to_cpu:
                self.checker.to(get_local_torch_device())

    @contextmanager
    def _on_device(self, module: torch.nn.Module) -> Iterator[None]:
        if not self.offload_to_cpu:
            yield
            return
        module.to(get_local_torch_device())
        try:
            yield
        finally:
            module.to("cpu")
            # Return the guard's cached blocks, so the offloaded guard does not keep GPU memory between checks.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _passes(self, check: Callable[[], bool]) -> bool:
        rank, world_size = _sp_rank_and_size()
        safe = bool(check()) if rank == 0 else None
        if world_size > 1:
            safe = get_sp_group().broadcast_object(safe, src=0)
        return bool(safe)

    def check_prompt(self, prompt: str) -> None:

        def check() -> bool:
            with self._on_device(self.checker.text_guard):
                return self.checker.check_text_safety(prompt)

        if not self._passes(check):
            # PAN2SafetyChecker logs the reason at CRITICAL.
            raise PAN2GuardrailViolationError("The prompt was blocked by the PAN 2 guardrails.")

    def check_video(self, frames: Callable[[], np.ndarray]) -> None:
        """``frames`` builds the uint8 ``[T, H, W, 3]`` frames, on the checking rank only."""

        def check() -> bool:
            with self._on_device(self.checker.video_guard):
                # `check_video_safety` returns the unchanged frames when they pass and None when the video is blocked.
                return self.checker.check_video_safety(frames()) is not None

        if not self._passes(check):
            raise PAN2GuardrailViolationError("The generated video was blocked by the PAN 2 guardrails.")


class PAN2PromptGuardrailStage(PipelineStage):
    """Checks the prompt before any compute. Latent output, which the video check cannot see, is refused here too."""

    def __init__(self, guardrails: PAN2Guardrails) -> None:
        super().__init__()
        self.guardrails = guardrails

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.output_type == "latent":
            raise ValueError("The PAN 2 guardrails check the decoded video, so latent output is not available with "
                             "them; " + _DISABLE_HINT)
        prompts = [batch.prompt] if isinstance(batch.prompt, str) else list(batch.prompt or [])
        for prompt in prompts:
            self.guardrails.check_prompt(prompt)
        return batch


class PAN2VideoGuardrailStage(PipelineStage):
    """Checks every decoded video, as the uint8 frames the generator delivers."""

    def __init__(self, guardrails: PAN2Guardrails) -> None:
        super().__init__()
        self.guardrails = guardrails

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # `batch.output` is `[B, 3, T, H, W]` in [0, 1]; `pixels_to_uint8` quantizes as the generator does.
        for video in batch.output:
            self.guardrails.check_video(lambda video=video: pixels_to_uint8(video).permute(1, 2, 3, 0).cpu().numpy())
        return batch
