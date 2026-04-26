# SPDX-License-Identifier: Apache-2.0
"""Stable Audio Open 1.0 conditioner — first-class FastVideo port of the
official `stable_audio_tools.models.conditioners` MultiConditioner +
T5Conditioner + NumberConditioner subset.

Vendored from `Stability-AI/stable-audio-tools` under Apache-2.0,
stripped to the conditioners the published 1.0 model actually uses
(prompt -> T5; seconds_start / seconds_total -> NumberConditioner).

Replaces the previous `diffusers.StableAudioProjectionModel` reuse so
the pipeline owns the conditioning pathway end-to-end (no
`from diffusers import ...` at runtime; see REVIEW item 30).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Number embedder: LearnedPositionalEmbedding + Linear head
# ---------------------------------------------------------------------------


class _LearnedPositionalEmbedding(nn.Module):
    """Upstream `adp.LearnedPositionalEmbedding`."""

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
    """Upstream `adp.TimePositionalEmbedding`."""
    return nn.Sequential(_LearnedPositionalEmbedding(dim),
                         nn.Linear(in_features=dim + 1, out_features=out_features))


class NumberEmbedder(nn.Module):
    """Upstream `adp.NumberEmbedder`."""

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


# ---------------------------------------------------------------------------
# Conditioner base + T5 + Number
# ---------------------------------------------------------------------------


class _Conditioner(nn.Module):

    def __init__(self, dim: int, output_dim: int, project_out: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = (nn.Linear(dim, output_dim) if dim != output_dim or project_out
                         else nn.Identity())


class T5Conditioner(_Conditioner):
    """T5 text conditioner.

    Loads `t5-base` via HuggingFace `transformers` (the upstream code
    does the same — `transformers` is a *tokenizer + pure-PyTorch model*
    library, not a model-class shortcut like `from diffusers`). Pads to
    `model_max_length` (=128 for the SA repo's tokenizer) and produces a
    masked last-hidden-state.
    """

    T5_MODEL_DIMS = {"t5-base": 768}

    def __init__(self, output_dim: int, t5_model_name: str = "t5-base",
                 max_length: int = 128) -> None:
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=False)
        from transformers import AutoTokenizer, T5EncoderModel
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        # Match upstream: keep the T5 weights non-trainable and outside the
        # parent module's `parameters()` so the SA checkpoint loader doesn't
        # see them.
        model = T5EncoderModel.from_pretrained(t5_model_name).eval().requires_grad_(False)
        self.__dict__["model"] = model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if hasattr(self, "model"):
            self.model.to(*args, **kwargs)
        return self

    def forward(self, texts: list[str], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        self.model.eval()
        with torch.no_grad():
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


# ---------------------------------------------------------------------------
# MultiConditioner with the SA conditioning config baked in
# ---------------------------------------------------------------------------


class StableAudioMultiConditioner(nn.Module):
    """Hardcoded for SA-Open-1.0:
        prompt        -> T5Conditioner(t5-base, max_length=128)
        seconds_start -> NumberConditioner(min=0, max=512)
        seconds_total -> NumberConditioner(min=0, max=512)
    cond_dim = 768
    """

    cross_attention_cond_ids = ("prompt", "seconds_start", "seconds_total")
    global_cond_ids = ("seconds_start", "seconds_total")
    cond_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.conditioners = nn.ModuleDict({
            "prompt": T5Conditioner(output_dim=self.cond_dim, t5_model_name="t5-base",
                                    max_length=128),
            "seconds_start": NumberConditioner(output_dim=self.cond_dim, min_val=0, max_val=512),
            "seconds_total": NumberConditioner(output_dim=self.cond_dim, min_val=0, max_val=512),
        })

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
        """Replicates `ConditionedDiffusionModelWrapper.get_conditioning_inputs`:
        cross-attn = cat([prompt; seconds_start; seconds_total], dim=1)
        global    = cat([seconds_start[:, 0]; seconds_total[:, 0]], dim=-1)
        Returns (cross_attn_cond, cross_attn_mask, global_embed).
        """
        prompt_emb, prompt_mask = cond["prompt"]
        ss_emb, ss_mask = cond["seconds_start"]
        st_emb, st_mask = cond["seconds_total"]
        cross_attn_cond = torch.cat([prompt_emb, ss_emb, st_emb], dim=1)
        cross_attn_mask = torch.cat([prompt_mask, ss_mask, st_mask], dim=1)
        global_embed = torch.cat([ss_emb[:, 0], st_emb[:, 0]], dim=-1)
        return cross_attn_cond, cross_attn_mask, global_embed

    @classmethod
    def from_official_state_dict(cls, state_dict: dict[str, torch.Tensor],
                                 prefix: str = "conditioner.") -> "StableAudioMultiConditioner":
        """Load number-conditioner weights from the official `model.safetensors`.
        T5 weights are NOT in that checkpoint — the upstream checkpoint
        intentionally omits them (`__dict__["model"] = ...` keeps T5 out
        of `parameters()`); T5-base is fetched fresh from HF in __init__.
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
        # T5 weights legitimately missing — they're loaded by HF in __init__.
        missing = [k for k in own_state.keys() if k not in loaded
                   and not k.startswith("conditioners.prompt.")]
        unexpected = [k for k in loaded.keys() if k not in own_state]
        if missing or unexpected:
            raise RuntimeError(
                f"StableAudioMultiConditioner load mismatch — missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        mc.load_state_dict(loaded, strict=False)
        return mc
