# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 conditioner.

T5-base text encoder + two NumberConditioners (`seconds_start`,
`seconds_total`), wrapped by `StableAudioMultiConditioner` which
produces the cross-attention and global-conditioning tensors the DiT
expects.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from fastvideo.configs.models.encoders import StableAudioConditionerConfig


class _LearnedPositionalEmbedding(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


def _time_positional_embedding(dim: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(_LearnedPositionalEmbedding(dim),
                         nn.Linear(in_features=dim + 1, out_features=out_features))


class NumberEmbedder(nn.Module):

    def __init__(self, features: int, dim: int = 256) -> None:
        super().__init__()
        self.features = features
        self.embedding = _time_positional_embedding(dim=dim, out_features=features)

    def forward(self, x: torch.Tensor | list[float]) -> torch.Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        out = self.embedding(x)
        return out.view(*shape, self.features)


class _Conditioner(nn.Module):

    def __init__(self, dim: int, output_dim: int, project_out: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = (nn.Linear(dim, output_dim) if dim != output_dim or project_out
                         else nn.Identity())


class T5Conditioner(_Conditioner):
    """T5 text conditioner. Pads to `model_max_length` (=128 for the SA
    repo's tokenizer, NOT the standard 512) and emits a masked
    last-hidden-state.
    """

    T5_MODEL_DIMS = {"t5-base": 768}

    def __init__(self, output_dim: int, t5_model_name: str = "t5-base",
                 max_length: int = 128, dtype: str = "float16") -> None:
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=False)
        from transformers import AutoTokenizer, T5EncoderModel
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # T5 loaded directly in fp16 (config-driven) to match official
        # `stable_audio_tools/models/conditioners.py:334`. Registered as
        # a normal submodule so `.to(device)` / `torch.compile` track it;
        # `from_official_state_dict` filters `conditioners.prompt.*` from
        # the missing-key check (T5 weights are absent from the SA
        # checkpoint by design).
        # Explicit lookup so a typo (e.g. "fp16" instead of "float16") errors
        # at load time rather than silently falling back to a wrong dtype.
        torch_dtype = getattr(torch, dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise ValueError(f"T5Conditioner dtype={dtype!r} is not a torch.dtype.")
        self._t5_dtype = torch_dtype
        self.model = (T5EncoderModel.from_pretrained(t5_model_name).eval().requires_grad_(False).to(torch_dtype))

    def forward(self, texts: list[str], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        # Mirror official's `autocast(fp16)` wrap on T5 forward.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self._t5_dtype):
            embeddings = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask)["last_hidden_state"]
        embeddings = self.proj_out(embeddings) * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class NumberConditioner(_Conditioner):
    """Float-valued conditioner with min/max clamping + NumberEmbedder."""

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1) -> None:
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: list[float], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        floats = [float(x) for x in floats]
        floats_t = torch.tensor(floats, device=device).clamp(self.min_val, self.max_val)
        normalized = (floats_t - self.min_val) / (self.max_val - self.min_val)
        emb_dtype = next(self.embedder.parameters()).dtype
        normalized = normalized.to(emb_dtype)
        float_embeds = self.embedder(normalized).unsqueeze(1)
        return float_embeds, torch.ones(float_embeds.shape[0], 1, device=device)


class StableAudioMultiConditioner(nn.Module):
    """SA-Open-1.0 conditioner: T5 prompt + duration NumberConditioners.

    All hardcoded constants (cond_dim, sub-conditioner ids, T5 model
    name + max_length, NumberConditioner ranges) live on
    `StableAudioConditionerConfig` — see
    `fastvideo/configs/models/encoders/stable_audio_conditioner.py`.
    """

    def __init__(self, config: StableAudioConditionerConfig | None = None) -> None:
        super().__init__()
        self.config = config or StableAudioConditionerConfig()
        arch = self.config.arch_config
        # Build sub-conditioners from the `configs` list (mirrors
        # upstream's `MultiConditioner` factory). SA-1.0 has 3 entries
        # (prompt + seconds_start + seconds_total); SA-small has 2
        # (prompt + seconds_total only).
        sub: dict[str, nn.Module] = {}
        for spec in arch.configs:
            sid = spec["id"]
            stype = spec["type"]
            scfg = spec["config"]
            if stype == "t5":
                sub[sid] = T5Conditioner(output_dim=arch.cond_dim,
                                         t5_model_name=scfg["t5_model_name"],
                                         max_length=scfg["max_length"],
                                         dtype=arch.t5_dtype)
            elif stype == "number":
                sub[sid] = NumberConditioner(output_dim=arch.cond_dim,
                                             min_val=scfg["min_val"], max_val=scfg["max_val"])
            else:
                raise ValueError(f"Unknown sub-conditioner type {stype!r} for id {sid!r}.")
        self.conditioners = nn.ModuleDict(sub)
        self.cross_attention_cond_ids = tuple(arch.cross_attention_cond_ids)
        self.global_cond_ids = tuple(arch.global_cond_ids)

    def forward(self, batch_metadata: list[dict],
                device: torch.device | str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for key, conditioner in self.conditioners.items():
            inputs = [x[key] for x in batch_metadata]
            out[key] = conditioner(inputs, device)
        return out

    def get_conditioning_inputs(
        self, cond: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack conditioner outputs into the (cross_attn_cond,
        cross_attn_mask, global_embed) triple the DiT consumes. Order
        is driven by `cross_attention_cond_ids` / `global_cond_ids`
        from the config — SA-1.0 uses three sub-conditioners
        (prompt + seconds_start + seconds_total); SA-small uses two
        (prompt + seconds_total).
        """
        x_embs = [cond[i][0] for i in self.cross_attention_cond_ids]
        x_masks = [cond[i][1] for i in self.cross_attention_cond_ids]
        cross_attn_cond = torch.cat(x_embs, dim=1)
        cross_attn_mask = torch.cat(x_masks, dim=1)
        global_embed = torch.cat([cond[i][0][:, 0] for i in self.global_cond_ids], dim=-1)
        return cross_attn_cond, cross_attn_mask, global_embed

    @classmethod
    def from_pretrained(cls, model_path: str) -> "StableAudioMultiConditioner":
        """Load NumberConditioner weights from a converted Diffusers-format
        `conditioner/` subfolder. T5 is fetched fresh from HF in
        `__init__` (T5 weights aren't part of the SA checkpoint by
        design — see `from_official_state_dict` for the rationale).
        """
        import os
        from safetensors.torch import load_file
        weights = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        state = load_file(weights)
        mc = cls()
        own = set(mc.state_dict().keys())
        # T5 weights are fetched in `__init__`; ignore them in the
        # missing-key check, only NumberConditioner keys must be present.
        missing = [k for k in own if k not in state and not k.startswith("conditioners.prompt.")]
        unexpected = [k for k in state if k not in own]
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioMultiConditioner load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}")
        mc.load_state_dict(state, strict=False)
        return mc

    @classmethod
    def from_official_state_dict(cls, state_dict: dict[str, torch.Tensor],
                                 prefix: str = "conditioner.") -> "StableAudioMultiConditioner":
        """Load NumberConditioner weights from a raw `stable_audio_tools`
        monolithic state dict. Kept for tests / older checkpoints;
        production loads go through `from_pretrained` against the
        Diffusers-format converted repo.
        """
        mc = cls()
        own_state = mc.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            stripped = k[len(prefix):]
            if stripped in own_state:
                loaded[stripped] = v
        # T5 keys are intentionally absent from the checkpoint.
        missing = [k for k in own_state.keys() if k not in loaded
                   and not k.startswith("conditioners.prompt.")]
        unexpected = [k for k in loaded.keys() if k not in own_state]
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioMultiConditioner load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        mc.load_state_dict(loaded, strict=False)
        return mc


EntryClass = StableAudioMultiConditioner
