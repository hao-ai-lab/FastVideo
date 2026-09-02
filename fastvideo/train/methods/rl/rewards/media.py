# SPDX-License-Identifier: Apache-2.0
"""Generic media reward composition utilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

RewardScorer = Callable[
    [torch.Tensor, Sequence[str]],
    torch.Tensor,
]


def select_first_frame(media: torch.Tensor) -> torch.Tensor:
    """Return first-frame media as ``[B, C, H, W]``."""
    if not torch.is_tensor(media):
        raise TypeError(
            "media must be a torch.Tensor, "
            f"got {type(media).__name__}"
        )
    if media.ndim == 5:
        return media[:, :, 0]
    if media.ndim == 4:
        return media
    raise ValueError(
        "media must have shape [B, C, H, W] or "
        "[B, C, T, H, W], "
        f"got {tuple(media.shape)}"
    )


def media_to_float_tensor(
    media: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert uint8 or floating media to float ``[0, 1]``."""
    value = media.detach()
    if value.dtype == torch.uint8:
        value = value.float().div_(255.0)
    else:
        value = value.float().clamp_(0.0, 1.0)
    return (
        value.to(device)
        if device is not None
        else value
    )


def media_to_uint8_array(
    media: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Convert image/video media to uint8 NHWC or NFHWC arrays."""
    if isinstance(media, torch.Tensor):
        tensor = media.detach().cpu()
        if tensor.dtype == torch.uint8:
            media = tensor.numpy()
        else:
            media = (
                tensor.float()
                .clamp(0, 1)
                .numpy()
            )
    media = np.asarray(media)
    if media.ndim == 4:
        if media.shape[-1] in (1, 3):
            pass
        elif media.shape[1] in (1, 3):
            media = media.transpose(0, 2, 3, 1)
    elif media.ndim == 5:
        if media.shape[-1] in (1, 3):
            pass
        elif media.shape[2] in (1, 3):
            media = media.transpose(
                0,
                1,
                3,
                4,
                2,
            )
        elif media.shape[1] in (1, 3):
            media = media.transpose(
                0,
                2,
                3,
                4,
                1,
            )
    else:
        raise ValueError(
            "media must have shape [B, C, H, W], "
            "[B, H, W, C], [B, C, T, H, W], "
            "[B, T, C, H, W], or [B, T, H, W, C], "
            f"got {tuple(media.shape)}"
        )
    if np.issubdtype(media.dtype, np.floating):
        media = (
            np.clip(media * 255.0, 0, 255)
            .round()
            .astype(np.uint8)
        )
    elif media.dtype != np.uint8:
        media = np.clip(
            media,
            0,
            255,
        ).astype(np.uint8)
    return media


class MultiRewardScorer:
    """Weighted sum of reusable media reward scorers."""

    def __init__(
        self,
        reward_weights: Mapping[str, float],
        *,
        scorers: Mapping[str, RewardScorer],
    ) -> None:
        self.reward_weights = {
            str(key): float(value)
            for key, value in reward_weights.items()
        }
        if not self.reward_weights:
            raise ValueError(
                "reward_weights must contain at least one reward"
            )
        self.scorers = dict(scorers)
        unsupported = sorted(
            set(self.reward_weights)
            - set(self.scorers)
        )
        if unsupported:
            raise ValueError(
                f"Unsupported reward(s): {unsupported}. "
                f"Available rewards: {sorted(self.scorers)}"
            )
        self.output_keys = self._build_output_keys()

    def _build_output_keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        for name in self.reward_weights:
            keys.append(name)
            scorer = self.scorers[name]
            declared: set[str] = set()
            for raw_diagnostic in getattr(
                scorer,
                "diagnostic_names",
                (),
            ):
                diagnostic = (
                    str(raw_diagnostic)
                    .strip()
                    .lower()
                )
                if not diagnostic:
                    raise ValueError(
                        f"Reward {name!r} declares an empty diagnostic name"
                    )
                if diagnostic in declared:
                    raise ValueError(
                        f"Reward {name!r} declares duplicate diagnostic "
                        f"{diagnostic!r}"
                    )
                declared.add(diagnostic)
                keys.append(f"{name}_{diagnostic}")
        keys.append("avg")
        if len(keys) != len(set(keys)):
            raise ValueError(
                f"Reward output keys collide: {keys}"
            )
        return tuple(keys)

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        prompts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        prompt_count = len(prompts)
        if media.shape[0] != prompt_count:
            raise ValueError(
                f"media batch size ({media.shape[0]}) "
                f"must match prompt count ({prompt_count})"
            )
        total: torch.Tensor | None = None
        details: dict[str, torch.Tensor] = {}
        for name, weight in self.reward_weights.items():
            scorer = self.scorers[name]
            scores = scorer(
                media,
                prompts,
            ).detach().float()
            self._validate_scores(
                name,
                scores,
                prompt_count,
            )
            details[name] = scores
            weighted = scores * float(weight)
            total = (
                weighted
                if total is None
                else total.to(weighted.device) + weighted
            )

            diagnostics = getattr(
                scorer,
                "last_diagnostics",
                None,
            )
            declared = {
                str(value).strip().lower()
                for value in getattr(
                    scorer,
                    "diagnostic_names",
                    (),
                )
            }
            if isinstance(diagnostics, Mapping):
                observed = {
                    str(value).strip().lower()
                    for value in diagnostics
                }
                undeclared = sorted(observed - declared)
                missing = sorted(declared - observed)
                if undeclared or missing:
                    raise RuntimeError(
                        f"Reward {name!r} diagnostic contract mismatch: "
                        f"undeclared={undeclared}, missing={missing}"
                    )
                for diagnostic_name, values in diagnostics.items():
                    key = (
                        f"{name}_"
                        f"{str(diagnostic_name).strip().lower()}"
                    )
                    diagnostic = torch.as_tensor(
                        values,
                        device=scores.device,
                        dtype=torch.float32,
                    ).detach()
                    self._validate_scores(
                        key,
                        diagnostic,
                        prompt_count,
                    )
                    if key in details:
                        raise ValueError(
                            f"Duplicate reward diagnostic key: {key}"
                        )
                    details[key] = diagnostic
            elif declared:
                raise RuntimeError(
                    f"Reward {name!r} declares diagnostics "
                    "but did not populate last_diagnostics"
                )

        assert total is not None
        details["avg"] = total
        if tuple(details) != self.output_keys:
            raise RuntimeError(
                "Reward output order changed at runtime: "
                f"expected={self.output_keys}, observed={tuple(details)}"
            )
        return details

    @staticmethod
    def _validate_scores(
        name: str,
        scores: torch.Tensor,
        prompt_count: int,
    ) -> None:
        if (
            scores.ndim != 1
            or int(scores.shape[0]) != prompt_count
        ):
            raise ValueError(
                f"Reward {name!r} must return shape "
                f"[{prompt_count}], got {tuple(scores.shape)}"
            )
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError(
                f"Reward {name!r} returned NaN or Inf"
            )


__all__ = [
    "MultiRewardScorer",
    "RewardScorer",
    "media_to_float_tensor",
    "media_to_uint8_array",
    "select_first_frame",
]
