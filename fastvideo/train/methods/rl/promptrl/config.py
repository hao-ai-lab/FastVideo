# SPDX-License-Identifier: Apache-2.0
"""PromptRL method configuration.

Parsed from the ``method:`` mapping of a run YAML.  Defaults follow the
PromptRL-Wan recipe:

* ``group_size: 8`` ranks per original prompt, ``retained_originals: 2``
  of which generate from the original prompt.
* Composite reward = ``format_reward_weight * format`` +
  ``video_reward_weight * VideoScore2``.
* KL to the frozen bases: ``refiner_kl_beta`` / ``student_kl_beta``.
* PPO clip ``1e-4`` for the Wan flow-policy loss (and refiner surrogate).
* Rollout: 20 flow-matching steps, the last 8 stochastic (SDE) with
  noise scale 0.8, transition losses recomputed with microbatch 1.
* Per-role optimizers: LoRA lr 1e-5, AdamW betas (0.9, 0.999), weight
  decay 0.01, gradient clipping 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PromptRLMode = Literal["prompt_only", "joint"]

_SUPPORTED_MODE = ("prompt_only", "joint")


@dataclass(slots=True)
class RoleOptimizerConfig:
    """Per-role optimizer + gradient clipping settings."""

    learning_rate: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, *, where: str) -> RoleOptimizerConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a mapping, got {type(raw).__name__}")
        betas_raw = raw.get("betas", (0.9, 0.999))
        if not isinstance(betas_raw, list | tuple) or len(betas_raw) != 2:
            raise ValueError(f"{where}.betas must be a 2-element list, got {betas_raw!r}")
        return cls(
            learning_rate=float(raw.get("learning_rate", 1e-5)),
            betas=(float(betas_raw[0]), float(betas_raw[1])),
            weight_decay=float(raw.get("weight_decay", 0.01)),
            max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
            lr_scheduler=str(raw.get("lr_scheduler", "constant") or "constant"),
            lr_warmup_steps=int(raw.get("lr_warmup_steps", 0) or 0),
        )


@dataclass(slots=True)
class RolloutConfig:
    """Wan rollout knobs for PromptRL."""

    num_steps: int = 20
    sde_steps: int = 8
    noise_scale: float = 0.8
    loss_microbatch_size: int = 1
    flow_shift: float | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> RolloutConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.rollout must be a mapping, got {type(raw).__name__}")
        flow_shift = raw.get("flow_shift", None)
        cfg = cls(
            num_steps=int(raw.get("num_steps", 20) or 20),
            sde_steps=int(raw.get("sde_steps", 8) or 0),
            noise_scale=float(raw.get("noise_scale", 0.8)),
            loss_microbatch_size=int(raw.get("loss_microbatch_size", 1) or 1),
            flow_shift=(None if flow_shift in (None, "inherit") else float(flow_shift)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError(f"method.rollout.num_steps must be positive, got {self.num_steps}")
        if not 0 <= self.sde_steps <= self.num_steps:
            raise ValueError("method.rollout.sde_steps must be within "
                             f"[0, num_steps={self.num_steps}], got {self.sde_steps}")
        if self.noise_scale <= 0.0:
            raise ValueError(f"method.rollout.noise_scale must be positive, got {self.noise_scale}")
        if self.loss_microbatch_size <= 0:
            raise ValueError("method.rollout.loss_microbatch_size must be positive, "
                             f"got {self.loss_microbatch_size}")



@dataclass(slots=True)
class PromptDataConfig:
    """Raw prompt dataset settings (JSONL or Parquet).

    Required column/key: ``prompt``.  Optional: ``id``, ``reward_tag``.
    """

    data_path: str = ""
    prompt_key: str = "prompt"
    id_key: str = "id"
    reward_tag_key: str = "reward_tag"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PromptDataConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.data must be a mapping, got {type(raw).__name__}")
        return cls(
            data_path=str(raw.get("data_path", "") or ""),
            prompt_key=str(raw.get("prompt_key", "prompt") or "prompt"),
            id_key=str(raw.get("id_key", "id") or "id"),
            reward_tag_key=str(raw.get("reward_tag_key", "reward_tag") or "reward_tag"),
        )


@dataclass(slots=True)
class RewardServiceConfig:
    """External reward service client settings."""

    endpoint_url: str = "http://127.0.0.1:8100"
    timeout_sec: float = 300.0
    retries: int = 2
    fps: int = 16
    score_path: str = "/v1/rewards:score"
    health_path: str = "/healthz"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> RewardServiceConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.reward must be a mapping, got {type(raw).__name__}")
        cfg = cls(
            endpoint_url=str(raw.get("endpoint_url", "http://127.0.0.1:8100") or "").rstrip("/"),
            timeout_sec=float(raw.get("timeout_sec", 300.0)),
            retries=int(raw.get("retries", 2) or 0),
            fps=int(raw.get("fps", 16) or 16),
            score_path=str(raw.get("score_path", "/v1/rewards:score") or "/v1/rewards:score"),
            health_path=str(raw.get("health_path", "/healthz") or "/healthz"),
        )
        if not cfg.endpoint_url:
            raise ValueError("method.reward.endpoint_url must be a non-empty URL")
        if cfg.retries < 0:
            raise ValueError(f"method.reward.retries must be >= 0, got {cfg.retries}")
        if cfg.timeout_sec <= 0:
            raise ValueError(f"method.reward.timeout_sec must be positive, got {cfg.timeout_sec}")
        return cfg


@dataclass(slots=True)
class RefinerSamplingConfig:
    """Refiner completion sampling settings."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    template_version: str = "v1"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> RefinerSamplingConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"method.refiner must be a mapping, got {type(raw).__name__}")
        cfg = cls(
            max_new_tokens=int(raw.get("max_new_tokens", 256) or 256),
            temperature=float(raw.get("temperature", 1.0)),
            top_p=float(raw.get("top_p", 1.0)),
            template_version=str(raw.get("template_version", "v1") or "v1"),
        )
        if cfg.max_new_tokens <= 0:
            raise ValueError("method.refiner.max_new_tokens must be positive, "
                             f"got {cfg.max_new_tokens}")
        if cfg.temperature <= 0:
            raise ValueError(f"method.refiner.temperature must be positive, got {cfg.temperature}")
        return cfg



@dataclass(slots=True)
class PromptRLMethodConfig:
    """Top-level ``method:`` block for PromptRL."""

    mode: PromptRLMode = "prompt_only"
    group_size: int = 8
    retained_originals: int = 2
    format_reward_weight: float = 1.0
    video_reward_weight: float = 1.0
    refiner_kl_beta: float = 0.01
    student_kl_beta: float = 0.01
    ppo_clip: float = 1e-4
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    data: PromptDataConfig = field(default_factory=PromptDataConfig)
    reward: RewardServiceConfig = field(default_factory=RewardServiceConfig)
    refiner: RefinerSamplingConfig = field(default_factory=RefinerSamplingConfig)
    refiner_optimizer: RoleOptimizerConfig = field(default_factory=RoleOptimizerConfig)
    generator_optimizer: RoleOptimizerConfig = field(default_factory=RoleOptimizerConfig)
    log_samples: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PromptRLMethodConfig:
        if not isinstance(raw, dict):
            raise ValueError(f"method must be a mapping, got {type(raw).__name__}")
        cfg = cls(
            mode=str(raw.get("mode", "prompt_only") or "prompt_only"),  # type: ignore[arg-type]
            group_size=int(raw.get("group_size", 8)),
            retained_originals=int(raw.get("retained_originals", 2)),
            format_reward_weight=float(raw.get("format_reward_weight", 1.0)),
            video_reward_weight=float(raw.get("video_reward_weight", 1.0)),
            refiner_kl_beta=float(raw.get("refiner_kl_beta", 0.01)),
            student_kl_beta=float(raw.get("student_kl_beta", 0.01)),
            ppo_clip=float(raw.get("ppo_clip", 1e-4)),
            rollout=RolloutConfig.from_mapping(raw.get("rollout")),
            data=PromptDataConfig.from_mapping(raw.get("data")),
            reward=RewardServiceConfig.from_mapping(raw.get("reward")),
            refiner=RefinerSamplingConfig.from_mapping(raw.get("refiner")),
            refiner_optimizer=RoleOptimizerConfig.from_mapping(
                raw.get("refiner_optimizer"), where="method.refiner_optimizer"),
            generator_optimizer=RoleOptimizerConfig.from_mapping(
                raw.get("generator_optimizer"), where="method.generator_optimizer"),
            log_samples=bool(raw.get("log_samples", True)),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.mode not in _SUPPORTED_MODE:
            raise ValueError(f"method.mode must be one of {_SUPPORTED_MODE}, got {self.mode!r}")
        if self.group_size <= 0:
            raise ValueError(f"method.group_size must be positive, got {self.group_size}")
        if not 0 < self.retained_originals < self.group_size:
            raise ValueError("method.retained_originals must be within "
                             f"(0, group_size={self.group_size}), got {self.retained_originals}")
        if self.ppo_clip <= 0:
            raise ValueError(f"method.ppo_clip must be positive, got {self.ppo_clip}")
        if self.refiner_kl_beta < 0 or self.student_kl_beta < 0:
            raise ValueError("KL betas must be non-negative")
