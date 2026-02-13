# SPDX-License-Identifier: Apache-2.0
"""
Minimal Stable Audio conditioners inlined for t5 + number (seconds) only.

No dependency on stable-audio-tools. Supports model_config conditioning with
type "t5" and "number" (stable-audio-open-1.0).
"""
from __future__ import annotations

import logging
import typing as tp
import warnings

import torch
from einops import rearrange
from torch import nn

# NumberEmbedder dependency: LearnedPositionalEmbedding -> TimePositionalEmbedding


class LearnedPositionalEmbedding(nn.Module):
    """Continuous time embedding (from stable-audio-tools adp)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * 3.141592653589793
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


def _time_positional_embedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class NumberEmbedder(nn.Module):
    """Embed floats for conditioning (e.g. seconds_start, seconds_total)."""

    def __init__(self, features: int, dim: int = 256) -> None:
        super().__init__()
        self.features = features
        self.embedding = _time_positional_embedding(dim=dim, out_features=features)

    def forward(self, x: tp.Union[tp.List[float], torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=next(self.embedding.parameters()).device)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        embedding = self.embedding(x)
        return embedding.view(*shape, self.features)


class Conditioner(nn.Module):
    """Base conditioner."""

    def __init__(
        self, dim: int, output_dim: int, project_out: bool = False
    ) -> None:
        super().__init__()
        self.proj_out = (
            nn.Linear(dim, output_dim)
            if (dim != output_dim or project_out)
            else nn.Identity()
        )

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class NumberConditioner(Conditioner):
    """Conditioner for float lists (e.g. seconds_start, seconds_total)."""

    def __init__(
        self,
        output_dim: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(
        self, floats: tp.List[float], device: tp.Optional[torch.device] = None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        floats = [float(x) for x in floats]
        t = torch.tensor(floats).to(device)
        t = t.clamp(self.min_val, self.max_val)
        normalized = (t - self.min_val) / (self.max_val - self.min_val)
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized = normalized.to(embedder_dtype)
        embeds = self.embedder(normalized).unsqueeze(1)
        ones = torch.ones(embeds.shape[0], 1, device=embeds.device)
        return [embeds, ones]


class T5Conditioner(Conditioner):
    """T5 text conditioner (stable-audio-open uses t5-base)."""

    T5_MODELS = [
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
        "google/flan-t5-xl", "google/flan-t5-xxl",
        "google/t5-v1_1-xl", "google/t5-v1_1-xxl",
    ]
    T5_MODEL_DIMS = {
        "t5-small": 512, "t5-base": 768, "t5-large": 1024,
        "t5-3b": 1024, "t5-11b": 1024,
        "google/t5-v1_1-xl": 2048, "google/t5-v1_1-xxl": 4096,
        "google/flan-t5-small": 512, "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024, "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024, "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        max_length: int = 128,
        enable_grad: bool = False,
        project_out: bool = False,
    ) -> None:
        assert t5_model_name in self.T5_MODELS
        super().__init__(
            self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out
        )
        from transformers import AutoTokenizer, T5EncoderModel

        self.max_length = max_length
        self.enable_grad = enable_grad
        prev = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name)
                model = model.train(enable_grad).requires_grad_(enable_grad)
                model = model.to(torch.float16)
        finally:
            logging.getLogger().setLevel(prev)
        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(
        self, texts: tp.List[str], device: tp.Union[torch.device, str]
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(device)
        self.proj_out.to(device)
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        self.model.eval()
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.set_grad_enabled(
            self.enable_grad
        ):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]
        if not isinstance(self.proj_out, nn.Identity):
            embeddings = embeddings.to(
                next(self.proj_out.parameters()).dtype
            )
        embeddings = self.proj_out(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """Applies multiple conditioners keyed by config."""

    def __init__(
        self,
        conditioners: tp.Dict[str, Conditioner],
        default_keys: tp.Dict[str, str] | None = None,
        pre_encoded_keys: tp.List[str] | None = None,
    ) -> None:
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}
        self.pre_encoded_keys = list(pre_encoded_keys or [])

    def forward(
        self,
        batch_metadata: tp.List[tp.Dict[str, tp.Any]],
        device: tp.Union[torch.device, str],
    ) -> tp.Dict[str, tp.Any]:
        output: tp.Dict[str, tp.Any] = {}
        for key, conditioner in self.conditioners.items():
            condition_key = key
            inputs = []
            for x in batch_metadata:
                if condition_key not in x:
                    condition_key = self.default_keys.get(condition_key, key)
                if condition_key not in x:
                    raise ValueError(
                        f"Conditioner key {condition_key} not in metadata"
                    )
                val = x[condition_key]
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                inputs.append(val)
            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(inputs).to(device), None]
            else:
                output[key] = conditioner(inputs, device)
        return output


def create_multi_conditioner_from_conditioning_config(
    config: tp.Dict[str, tp.Any],
) -> MultiConditioner:
    """
    Build MultiConditioner from model_config conditioning section.
    Only supports conditioner types: "t5", "number" (stable-audio-open-1.0).
    """
    conditioners: tp.Dict[str, Conditioner] = {}
    cond_dim = config["cond_dim"]
    default_keys = config.get("default_keys", {})
    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for info in config["configs"]:
        cid = info["id"]
        ctype = info["type"]
        cfg = {"output_dim": cond_dim, **info.get("config", {})}

        if ctype == "t5":
            conditioners[cid] = T5Conditioner(**cfg)
        elif ctype == "number":
            conditioners[cid] = NumberConditioner(**cfg)
        else:
            raise ValueError(
                f"Only t5 and number conditioners are supported inline; "
                f"got type={ctype}. Use stable-audio-tools clone for others."
            )

    return MultiConditioner(
        conditioners,
        default_keys=default_keys,
        pre_encoded_keys=pre_encoded_keys,
    )
