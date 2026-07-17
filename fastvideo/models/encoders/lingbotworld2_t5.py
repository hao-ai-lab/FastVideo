# SPDX-License-Identifier: Apache-2.0
"""LingBot World 2 UMT5 encoder with the released checkpoint's module names."""

from collections.abc import Iterable
import html
import math
import string

import ftfy
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.lingbotworld2_t5 import LingBotWorld2UMT5Config
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader


def basic_clean(text: str) -> str:
    """Apply LingBot World 2's ftfy/html cleanup before tokenization."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


def canonicalize(text: str, keep_punctuation_exact_string: str | None = None) -> str:
    """Normalize prompts with LingBot World 2's optional punctuation handling."""
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.lower().split())


def lingbotworld2_whitespace_preprocess(prompt: str) -> str:
    """Match the LingBot World 2 source tokenizer's `clean='whitespace'` behavior."""
    return whitespace_clean(basic_clean(prompt))


class GELU(nn.Module):
    """T5 gated GELU approximation used by the source checkpoint."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tanh GELU approximation."""
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    """T5 RMS-style layer norm with source-compatible parameter name."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize in fp32 and apply the learned scale."""
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    """LingBot World 2 source T5 attention block."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project QKV, add relative bias/mask, and return attended states."""
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in (2, 3)
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        return self.dropout(self.o(x.reshape(b, -1, n * c)))


class T5FeedForward(nn.Module):
    """LingBot World 2 source T5 gated feed-forward block."""

    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated feed-forward projection."""
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class T5RelativeEmbedding(nn.Module):
    """Per-block relative position embedding used by LingBot World 2 UMT5."""

    def __init__(self, num_buckets: int, num_heads: int, bidirectional: bool, max_dist: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        """Build a relative-position bias tensor for attention logits."""
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        """Map token offsets to T5 relative-position buckets."""
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
        ).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttention(nn.Module):
    """One source-compatible UMT5 encoder block."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, pos_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Run self-attention and feed-forward residual updates."""
        e = pos_bias if self.pos_embedding is None else self.pos_embedding(x.size(1), x.size(1))
        x = self._fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        return self._fp16_clamp(x + self.ffn(self.norm2(x)))

    @staticmethod
    def _fp16_clamp(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp, max=clamp)
        return x


class LingBotWorld2T5EncoderModel(TextEncoder):
    """FastVideo-native LingBot World 2 UMT5 encoder for the released `.pth` weights."""

    fall_back_to_pt_during_load = True
    allow_patterns_overrides = ["*.pt"]

    def __init__(self, config: LingBotWorld2UMT5Config, prefix: str = ""):
        super().__init__(config)
        del prefix
        arch = config.arch_config
        self.token_embedding = nn.Embedding(arch.vocab_size, arch.dim)
        self.dropout = nn.Dropout(arch.dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    arch.dim,
                    arch.dim_attn,
                    arch.dim_ffn,
                    arch.num_heads,
                    arch.num_buckets,
                    shared_pos=False,
                    dropout=arch.dropout,
                )
                for _ in range(arch.num_layers)
            ]
        )
        self.norm = T5LayerNorm(arch.dim)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        """Encode token ids and return source-compatible hidden states."""
        del position_ids, inputs_embeds, output_hidden_states, kwargs
        assert input_ids is not None
        x = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.dropout(self.norm(x))
        return BaseEncoderOutput(last_hidden_state=x, attention_mask=attention_mask)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load source `.pth` weights whose names already match this module."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = LingBotWorld2T5EncoderModel
