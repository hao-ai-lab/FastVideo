# SPDX-License-Identifier: Apache-2.0
"""OmniRef PDD: fixed ordered reference conditions, target-only trajectories.

Only the student's output projections widen. Reference inputs, teacher weights,
packing, augmentation seeds and P2 multi-region VSA use the Ref2VA adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastvideo.configs.pipelines.minimax_h3 import (FASTH3_INFERENCE_FILE, build_pdd_inference_file,
                                                    parse_base_model_revision)
from fastvideo.layers.pdd import PDD_GRID_MAX_T
from fastvideo.logger import init_logger
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.train.models.minimax_h3.minimax_h3_pdd import MiniMaxH3PDDMixin
from fastvideo.train.models.minimax_h3.minimax_h3_ref2v_dmd import MiniMaxH3Ref2VDMDModel
from fastvideo.utils import maybe_download_model, read_optional_model_json

logger = init_logger(__name__)

# Hugging Face cache snapshots live at .../models--<organization>--<name>/snapshots/<commit>.
_HF_CACHE_REPO_PREFIX = "models--"


class MiniMaxH3Ref2VPDDModel(MiniMaxH3PDDMixin, MiniMaxH3Ref2VDMDModel):
    """Reference-conditioned PDD student on the top-level transformer_ref."""

    def inference_contract(self, *, pdd_step_indices: Sequence[int]) -> dict[str, Any] | None:
        """The fastvideo_inference.json that lets the FastH3 Ref2VA pipeline sample this student's export.

        None when that pipeline cannot sample the student at all: it supports
        only sparse, reference-aware VIDEO_SPARSE_ATTN_H3 students. An export
        reuses ``init_from``'s scheduler configs, so their shifts must equal
        the clocks the student trains with.
        """
        sparsity = float(self.training_config.vsa_sparsity)
        keep_rate = self._resolved_vsa_ref_keep_rate()
        if (self.attention_backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3
                or not (0.0 <= sparsity < 1.0 and 0.0 < keep_rate < 1.0)):
            logger.warning(
                "The FastH3 Ref2VA pipeline samples only sparse VIDEO_SPARSE_ATTN_H3 PDD students; this %s student "
                "(VSA sparsity %s, reference keep rate %s) exports without %s and will not load there.",
                self.attention_backend_name, sparsity, keep_rate, FASTH3_INFERENCE_FILE)
            return None
        local_init = maybe_download_model(self._init_from)
        shifts: dict[str, Any] = {}
        for key, owner, trained in (
            ("video_scheduler_shift", "scheduler/scheduler_config.json", self.video_scheduler_shift),
            ("audio_scheduler_shift", "audio_scheduler/scheduler_config.json", self.audio_scheduler_shift),
        ):
            scheduler_config = read_optional_model_json(local_init, owner)
            exported = None if scheduler_config is None else scheduler_config.get("shift")
            if exported is None or float(exported) != float(trained):
                raise ValueError(f"The PDD student trains with {key}={trained}, but its export reuses {owner} from "
                                 f"{self._init_from} with shift {exported!r}, which FastH3 inference samples with. "
                                 "Train with the checkpoint's shift.")
            shifts[key] = exported
        indices = [int(index) for index in pdd_step_indices]
        return build_pdd_inference_file(
            base_model_revision=self._inference_base_model_revision(local_init),
            pdd_steps=self.pdd_steps,
            pdd_step_indices=indices,
            num_inference_steps=len(indices) - 1,
            transformer_forwards=len(indices) - 1,
            grid_max_t=PDD_GRID_MAX_T,
            vsa_sparsity=sparsity,
            vsa_tile_size=int(self.training_config.vsa_tile_size),
            vsa_ref_keep_rate=keep_rate,
            **shifts,
        )

    def _inference_base_model_revision(self, local_init: str) -> str:
        """The base snapshot whose text encoder and VAEs the export reuses, as ``hf://<repo id>@<revision>``."""
        checkpoint_config = getattr(self.training_config, "checkpoint", None)
        configured = getattr(checkpoint_config, "inference_base_model_revision", None)
        if configured:
            return str(configured)
        # A PDD warm start already names the base it was distilled against.
        init_contract = read_optional_model_json(local_init, FASTH3_INFERENCE_FILE)
        if init_contract is not None and init_contract.get("base_model_revision") is not None:
            revision = init_contract["base_model_revision"]
            parse_base_model_revision(revision)
            return str(revision)
        snapshot = Path(local_init).resolve()
        repo_dir = snapshot.parent.parent.name
        if snapshot.parent.name == "snapshots" and repo_dir.startswith(_HF_CACHE_REPO_PREFIX):
            organization, separator, name = repo_dir[len(_HF_CACHE_REPO_PREFIX):].partition("--")
            if separator and organization and name:
                return f"hf://{organization}/{name}@{snapshot.name}"
        raise ValueError(f"{self._init_from} is neither a FastH3 PDD checkpoint nor a Hugging Face cache snapshot; set "
                         "training.checkpoint.inference_base_model_revision to hf://<repo id>@<revision> of the base "
                         f"MiniMax-H3 snapshot so the export's {FASTH3_INFERENCE_FILE} can name it.")


__all__ = ["MiniMaxH3Ref2VPDDModel"]
