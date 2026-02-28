# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch


@dataclass
class KVCache:

    k: torch.Tensor
    v: torch.Tensor
    length: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        cache_size: int,
        num_heads: int,
        head_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "KVCache":
        return cls(
            k=torch.zeros(
                [batch_size, cache_size, num_heads, head_dim],
                dtype=dtype,
                device=device,
            ),
            v=torch.zeros(
                [batch_size, cache_size, num_heads, head_dim],
                dtype=dtype,
                device=device,
            ),
            length=torch.tensor([0], dtype=torch.long, device=device),
        )

    @classmethod
    def from_dict(cls, kv_cache: dict[str, torch.Tensor | int]) -> "KVCache":
        length = kv_cache["length"]
        if not isinstance(length, torch.Tensor):
            length = torch.tensor([int(length)], dtype=torch.long, device=kv_cache["k"].device)
        return cls(k=kv_cache["k"], v=kv_cache["v"], length=length)

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {"k": self.k, "v": self.v, "length": self.length}

    @property
    def capacity(self) -> int:
        return self.k.shape[1]

    @property
    def batch_size(self) -> int:
        return self.k.shape[0]

    def get_len(self) -> int:
        return int(self.length.item())

    def _preprocess_kv(self, new_k: torch.Tensor, new_v: torch.Tensor, num_new_tokens: int, store_first_only: bool):
        target_batch = self.batch_size
        if store_first_only:
            # Expand first token to match required sequence length
            write_k = new_k[:target_batch, :1].expand(-1, num_new_tokens, -1, -1)
            write_v = new_v[:target_batch, :1].expand(-1, num_new_tokens, -1, -1)
        else:
            write_k = new_k[:target_batch]
            write_v = new_v[:target_batch]
        return write_k, write_v

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        max_attn_size: int,
        store_first_only: bool = False,
        repeat_factor: int | None = None,
        num_new_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = num_new_tokens if num_new_tokens is not None else new_k.shape[1]
        write_k, write_v = self._preprocess_kv(new_k, new_v, num_tokens, store_first_only)
        
        cur_len = self.get_len()
        
        # Combine existing valid tokens with new ones
        combined_k = torch.cat([self.k[:, :cur_len], write_k], dim=1)
        combined_v = torch.cat([self.v[:, :cur_len], write_v], dim=1)
        
        # Keep only up to capacity
        new_len = min(combined_k.shape[1], self.capacity)
        self.k[:, :new_len] = combined_k[:, -new_len:]
        self.v[:, :new_len] = combined_v[:, -new_len:]
        self.length.fill_(new_len)
        
        return self._fetch_kv_slice(self.k, self.v, new_len, max_attn_size, repeat_factor)

    def get_view(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        max_attn_size: int,
        store_first_only: bool = False,
        repeat_factor: int | None = None,
        num_new_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = num_new_tokens if num_new_tokens is not None else new_k.shape[1]
        write_k, write_v = self._preprocess_kv(new_k, new_v, num_tokens, store_first_only)
        
        cur_len = self.get_len()
        combined_k = torch.cat([self.k[:, :cur_len], write_k], dim=1)
        combined_v = torch.cat([self.v[:, :cur_len], write_v], dim=1)
        
        return self._fetch_kv_slice(combined_k, combined_v, combined_k.shape[1], max_attn_size, repeat_factor)

    def _fetch_kv_slice(self, k, v, current_len, max_attn_size, repeat_factor):
        start = max(0, current_len - max_attn_size)
        out_k = k[:, start:current_len]
        out_v = v[:, start:current_len]
        
        if repeat_factor is not None:
            out_k = out_k.repeat(repeat_factor, 1, 1, 1)
            out_v = out_v.repeat(repeat_factor, 1, 1, 1)
        return out_k, out_v


@dataclass
class KVCacheDict:
    kv_cache: KVCache
    kv_cache_mouse: KVCache
    kv_cache_keyboard: KVCache

    def to_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "kv_cache": self.kv_cache.to_dict(),
            "kv_cache_mouse": self.kv_cache_mouse.to_dict(),
            "kv_cache_keyboard": self.kv_cache_keyboard.to_dict(),
        }


def attend_with_kv_cache(
    q: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    kv_cache: KVCache,
    *,
    max_attn_size: int,
    attend_fn,
    update_kv_cache: bool = True,
    store_first_only: bool = False,
    repeat_factor: int | None = None,
    num_new_tokens: int | None = None,
) -> torch.Tensor:
    """Unified KV-cache attention path"""

    if update_kv_cache:
        cached_k, cached_v = kv_cache.update(
            new_k, new_v, max_attn_size, store_first_only, repeat_factor, num_new_tokens
        )
    else:
        cached_k, cached_v = kv_cache.get_view(
            new_k, new_v, max_attn_size, store_first_only, repeat_factor, num_new_tokens
        )
    return attend_fn(q, cached_k, cached_v)
