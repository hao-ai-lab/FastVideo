# SPDX-License-Identifier: Apache-2.0
"""Source-aligned MJ-VIDEO aspect rewards for video post-training."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from fastvideo.train.methods.rl.rewards.media import (
    media_to_uint8_array,
)


MJ_VIDEO_SOURCE_REVISION = (
    "cc1d2c9587a620e9ebd3599ae4cdd21b5fd7c87a"
)
MJ_VIDEO_MODEL_REVISION = (
    "5d32c2416bf5ffb9331a175890744e73defb54c4"
)
MJ_VIDEO_BASE_MODEL_REVISION = "e4f6747"
MJ_VIDEO_MODEL_ID = "MJ-Bench/MJ-VIDEO-2B"
MJ_VIDEO_BASE_MODEL_ID = "OpenGVLab/InternVL2-2B"

MJ_VIDEO_ASPECT_INDICES = {
    "alignment": 0,
    "safety": 1,
    "fineness": 2,
    "cc": 3,
    "coherence_consistency": 3,
    "bias_fairness": 4,
}
MJ_VIDEO_ASPECT_TO_CRITERIA = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9, 10],
    2: [11, 12, 13, 14, 15],
    3: [16, 17, 18, 19, 20, 21, 22],
    4: [23, 24, 25, 26, 27],
}

_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)
_RUNTIME_CACHE: dict[tuple[Any, ...], "MJVideoRuntime"] = {}


def mj_video_frame_indices(
    num_frames: int,
    *,
    num_segments: int = 8,
) -> list[int]:
    """Match MJ-VIDEO's endpoint-exclusive uniform frame sampler."""
    frames = int(num_frames)
    segments = int(num_segments)
    if frames <= 0:
        raise ValueError("num_frames must be positive")
    if segments <= 0:
        raise ValueError("num_segments must be positive")
    return np.linspace(
        0,
        frames - 1,
        segments,
        endpoint=False,
        dtype=int,
    ).tolist()


@dataclass(frozen=True, slots=True)
class MJVideoRuntimeConfig:
    """Exact paths and source-derived inference settings."""

    runtime_path: str
    model_path: str
    base_model_path: str
    device: str
    source_revision: str = MJ_VIDEO_SOURCE_REVISION
    model_revision: str = MJ_VIDEO_MODEL_REVISION
    base_model_revision: str = MJ_VIDEO_BASE_MODEL_REVISION
    verify_revision: bool = True
    num_segments: int = 8
    input_size: int = 448
    max_num: int = 1
    batch_size: int = 1
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.num_segments != 8:
            raise ValueError(
                "MJ-VIDEO source-aligned inference requires num_segments=8"
            )
        if self.input_size != 448:
            raise ValueError(
                "MJ-VIDEO source-aligned inference requires input_size=448"
            )
        if self.max_num != 1:
            raise ValueError(
                "MJ-VIDEO source-aligned inference requires max_num=1"
            )
        if self.batch_size <= 0:
            raise ValueError("MJ-VIDEO batch_size must be positive")
        if self.dtype != "bfloat16":
            raise ValueError(
                "MJ-VIDEO source-aligned inference requires BF16"
            )


@contextmanager
def _temporary_sys_path(paths: Sequence[Path]):
    original = list(sys.path)
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path[:] = original


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not verify MJ-VIDEO source revision under {path}"
        ) from exc


def _verify_revision_marker(
    path: Path,
    expected: str,
    *,
    label: str,
) -> None:
    marker = path / ".fastvideo_revision"
    if not marker.is_file():
        raise RuntimeError(
            f"{label} is missing revision marker {marker}; "
            "use examples/train/rvm_h3/01_download_models.sh or set "
            "verify_revision=false for an explicitly audited local checkout"
        )
    observed = marker.read_text(encoding="utf-8").strip()
    if observed != expected:
        raise RuntimeError(
            f"{label} revision mismatch: expected {expected}, got {observed}"
        )


def _import_official_mj_video(
    runtime_path: Path,
) -> tuple[Any, Any, Any]:
    scripts = runtime_path / "scripts"
    model_root = scripts / "model"
    module_path = model_root / "moe_reward.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"MJ-VIDEO model source is missing: {module_path}"
        )
    importlib.invalidate_caches()
    with _temporary_sys_path((model_root, scripts)):
        internvl2 = importlib.import_module("internvl2")
        module_name = (
            "_fastvideo_mj_video_"
            + MJ_VIDEO_SOURCE_REVISION[:12]
        )
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                module_name,
                module_path,
            )
            if spec is None or spec.loader is None:
                raise ImportError(
                    f"Could not create an import spec for {module_path}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
    return (
        module.InternVLChatRewardModeling,
        module.InternVLChatRewardModelingConfig,
        internvl2.prepare_chat_input,
    )


def _ensure_distributed_rank_available() -> None:
    """Official InternVL2 calls torch.distributed.get_rank in forward."""
    if not dist.is_available() or dist.is_initialized():
        return
    rendezvous = tempfile.NamedTemporaryFile(
        prefix="fastvideo-mj-video-dist-",
        delete=False,
    )
    rendezvous.close()
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous.name}",
        rank=0,
        world_size=1,
    )


class MJVideoRuntime:
    """One shared official MJ-VIDEO model and batched aspect forward."""

    def __init__(
        self,
        config: MJVideoRuntimeConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        prepare_chat_input: Any | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16
        self._last_media: torch.Tensor | None = None
        self._last_prompts: tuple[str, ...] | None = None
        self._last_aspects: torch.Tensor | None = None
        self.forward_calls = 0

        if model is None:
            (
                model,
                tokenizer,
                prepare_chat_input,
            ) = self._load_official_runtime()
        if tokenizer is None or prepare_chat_input is None:
            raise ValueError(
                "MJVideoRuntime requires model, tokenizer, and "
                "prepare_chat_input together"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.prepare_chat_input = prepare_chat_input
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda image: (
                        image.convert("RGB")
                        if image.mode != "RGB"
                        else image
                    )
                ),
                transforms.Resize(
                    (config.input_size, config.input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=_IMAGE_MEAN,
                    std=_IMAGE_STD,
                ),
            ]
        )

    def _load_official_runtime(self) -> tuple[Any, Any, Any]:
        runtime_path = Path(
            self.config.runtime_path
        ).expanduser().resolve()
        model_path = Path(
            self.config.model_path
        ).expanduser().resolve()
        base_model_path = Path(
            self.config.base_model_path
        ).expanduser().resolve()
        for label, path in (
            ("MJ-VIDEO source", runtime_path),
            ("MJ-VIDEO checkpoint", model_path),
            ("InternVL2 base model", base_model_path),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} path does not exist: {path}"
                )

        if self.config.verify_revision:
            observed_source = _git_head(runtime_path)
            if observed_source != self.config.source_revision:
                raise RuntimeError(
                    "MJ-VIDEO source revision mismatch: expected "
                    f"{self.config.source_revision}, got {observed_source}"
                )
            _verify_revision_marker(
                model_path,
                self.config.model_revision,
                label="MJ-VIDEO checkpoint",
            )
            _verify_revision_marker(
                base_model_path,
                self.config.base_model_revision,
                label="InternVL2 base model",
            )

        try:
            (
                model_cls,
                config_cls,
                prepare_chat_input,
            ) = _import_official_mj_video(runtime_path)
            from safetensors.torch import load_file
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the pinned official MJ-VIDEO runtime. "
                "Its InternVL2 implementation targets an older Transformers "
                "API; do not silently substitute another reward model. "
                "Run the dedicated MJ-VIDEO preflight and commit any "
                "compatibility fix."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            use_fast=False,
        )
        config = config_cls.from_pretrained(
            str(base_model_path),
            pad_token_id=tokenizer.pad_token_id,
            num_objectives=28,
            num_aspects=5,
            aspect2criteria=MJ_VIDEO_ASPECT_TO_CRITERIA,
            gating_temperature=1.0,
            gating_hidden_dim=1024,
            gating_n_hidden=3,
        )

        try:
            model = model_cls(
                name=str(base_model_path),
                config=config,
            )
            checkpoint_files = sorted(
                model_path.glob("*.safetensors")
            )
            if len(checkpoint_files) != 1:
                raise RuntimeError(
                    "Expected exactly one MJ-VIDEO safetensors checkpoint "
                    f"under {model_path}, found {checkpoint_files}"
                )
            state = load_file(
                str(checkpoint_files[0]),
                device="cpu",
            )
            model.load_state_dict(
                state,
                strict=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct MJ-VIDEO-2B or strictly load its "
                "official checkpoint"
            ) from exc

        model.config.pad_token_id = tokenizer.pad_token_id
        model = (
            model.to(dtype=self.dtype)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        image_context_id = tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
        model.model.img_context_token_id = image_context_id
        _ensure_distributed_rank_available()
        return model, tokenizer, prepare_chat_input

    @torch.no_grad()
    def score_aspects(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        if (
            media is self._last_media
            and prompt_tuple == self._last_prompts
            and self._last_aspects is not None
        ):
            return self._last_aspects

        videos = media_to_uint8_array(media)
        if videos.ndim != 5:
            raise ValueError(
                "MJ-VIDEO requires video media with a temporal axis"
            )
        if videos.shape[0] != len(prompt_tuple):
            raise ValueError(
                "MJ-VIDEO media batch size must match prompts"
            )

        outputs: list[torch.Tensor] = []
        for start in range(
            0,
            len(prompt_tuple),
            self.config.batch_size,
        ):
            stop = min(
                len(prompt_tuple),
                start + self.config.batch_size,
            )
            outputs.append(
                self._score_chunk(
                    videos[start:stop],
                    prompt_tuple[start:stop],
                )
            )
        aspects = torch.cat(outputs, dim=0).float()
        if aspects.ndim != 2 or aspects.shape[1] != 5:
            raise RuntimeError(
                "MJ-VIDEO must return aspect_scores with shape [B, 5], "
                f"got {tuple(aspects.shape)}"
            )
        if not bool(torch.isfinite(aspects).all()):
            raise RuntimeError(
                "MJ-VIDEO returned NaN or Inf aspect scores"
            )
        self._last_media = media
        self._last_prompts = prompt_tuple
        self._last_aspects = aspects
        return aspects

    def _score_chunk(
        self,
        videos: np.ndarray,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        input_ids_list: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        pixel_values_list: list[torch.Tensor] = []
        generation_config: dict[str, Any] = {
            "max_new_tokens": 1024,
            "do_sample": True,
        }

        for video, prompt in zip(
            videos,
            prompts,
            strict=True,
        ):
            indices = mj_video_frame_indices(
                video.shape[0],
                num_segments=self.config.num_segments,
            )
            pixels = torch.stack(
                [
                    self.transform(
                        Image.fromarray(video[index]).convert("RGB")
                    )
                    for index in indices
                ]
            ).to(
                device=self.device,
                dtype=self.dtype,
            )
            prefix = "".join(
                f"Frame{index + 1}: <image>\n"
                for index in range(len(indices))
            )
            input_ids, attention_mask = self.prepare_chat_input(
                self.model.config,
                self.tokenizer,
                pixels,
                prefix + str(prompt),
                generation_config,
                num_patches_list=[1] * len(indices),
                device=self.device,
            )
            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            pixel_values_list.append(pixels)

        input_ids, attention_mask = self._pad_text_batch(
            input_ids_list,
            attention_masks,
        )
        pixel_values = torch.cat(
            pixel_values_list,
            dim=0,
        )
        self.forward_calls += 1
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(
            self.device.type,
            dtype=self.dtype,
            enabled=autocast_enabled,
        ):
            output = self.model.forward(
                pixel_values,
                input_ids,
                attention_mask,
            )
        aspects = getattr(output, "aspect_scores", None)
        if not isinstance(aspects, torch.Tensor):
            raise RuntimeError(
                "Official MJ-VIDEO output has no aspect_scores tensor"
            )
        return aspects.detach()

    def _pad_text_batch(
        self,
        input_ids: Sequence[torch.Tensor],
        attention_masks: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max(
            int(value.shape[-1])
            for value in input_ids
        )
        padded_ids: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        pad_id = int(self.tokenizer.pad_token_id)
        for ids, mask in zip(
            input_ids,
            attention_masks,
            strict=True,
        ):
            padding = max_length - int(ids.shape[-1])
            if padding:
                ids = torch.cat(
                    [
                        ids,
                        torch.full(
                            (ids.shape[0], padding),
                            pad_id,
                            dtype=ids.dtype,
                            device=ids.device,
                        ),
                    ],
                    dim=-1,
                )
                mask = torch.cat(
                    [
                        mask,
                        torch.zeros(
                            (mask.shape[0], padding),
                            dtype=mask.dtype,
                            device=mask.device,
                        ),
                    ],
                    dim=-1,
                )
            padded_ids.append(ids)
            padded_masks.append(mask)
        return (
            torch.cat(padded_ids, dim=0),
            torch.cat(padded_masks, dim=0),
        )


class MJVideoAspectScorer:
    """Expose one official MJ-VIDEO aspect as a FastVideo reward scorer."""

    def __init__(
        self,
        *,
        aspect: Literal["cc", "fineness"],
        device: torch.device | str = "cuda",
        runtime_path: str | None = None,
        model_path: str | None = None,
        base_model_path: str | None = None,
        source_revision: str = MJ_VIDEO_SOURCE_REVISION,
        model_revision: str = MJ_VIDEO_MODEL_REVISION,
        base_model_revision: str = MJ_VIDEO_BASE_MODEL_REVISION,
        verify_revision: bool = True,
        num_segments: int = 8,
        input_size: int = 448,
        max_num: int = 1,
        batch_size: int = 1,
        runtime: MJVideoRuntime | None = None,
    ) -> None:
        normalized = str(aspect).strip().lower()
        if normalized not in {"cc", "fineness"}:
            raise ValueError(
                "MJVideoAspectScorer aspect must be cc or fineness"
            )
        self.aspect = normalized
        self.aspect_index = MJ_VIDEO_ASPECT_INDICES[normalized]
        if runtime is None:
            config = MJVideoRuntimeConfig(
                runtime_path=runtime_path
                or os.environ.get("MJ_VIDEO_RUNTIME_PATH", ""),
                model_path=model_path
                or os.environ.get("MJ_VIDEO_MODEL_PATH", ""),
                base_model_path=base_model_path
                or os.environ.get("MJ_VIDEO_BASE_MODEL_PATH", ""),
                device=str(device),
                source_revision=str(source_revision),
                model_revision=str(model_revision),
                base_model_revision=str(base_model_revision),
                verify_revision=bool(verify_revision),
                num_segments=int(num_segments),
                input_size=int(input_size),
                max_num=int(max_num),
                batch_size=int(batch_size),
            )
            key = (
                config.runtime_path,
                config.model_path,
                config.base_model_path,
                config.device,
                config.source_revision,
                config.model_revision,
                config.base_model_revision,
                config.verify_revision,
                config.num_segments,
                config.input_size,
                config.max_num,
                config.batch_size,
                config.dtype,
            )
            runtime = _RUNTIME_CACHE.get(key)
            if runtime is None:
                runtime = MJVideoRuntime(config)
                _RUNTIME_CACHE[key] = runtime
        self.runtime = runtime

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        aspects = self.runtime.score_aspects(
            media,
            prompts,
        )
        return aspects[:, self.aspect_index]


__all__ = [
    "MJ_VIDEO_ASPECT_INDICES",
    "MJ_VIDEO_ASPECT_TO_CRITERIA",
    "MJ_VIDEO_BASE_MODEL_ID",
    "MJ_VIDEO_BASE_MODEL_REVISION",
    "MJ_VIDEO_MODEL_ID",
    "MJ_VIDEO_MODEL_REVISION",
    "MJ_VIDEO_SOURCE_REVISION",
    "MJVideoAspectScorer",
    "MJVideoRuntime",
    "MJVideoRuntimeConfig",
    "mj_video_frame_indices",
]
