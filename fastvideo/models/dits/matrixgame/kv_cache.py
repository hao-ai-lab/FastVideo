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

    def length_int(self) -> int:
        return int(self.length.item())

    def set_length(self, value: int) -> None:
        self.length.fill_(int(value))


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


def _store_first_tokens(
    new_tensor: torch.Tensor,
    num_new_tokens: int,
    target_batch: int,
) -> torch.Tensor:
    first = new_tensor[:1]
    first = first.expand(target_batch, num_new_tokens, -1, -1)
    return first


def get_attended_kv_without_update(
    kv_cache: KVCache,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    num_new_tokens: int,
    max_attn_size: int,
    store_first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_capacity = kv_cache.capacity
    target_batch = kv_cache.batch_size
    cache_length = kv_cache.length_int()

    if store_first_only:
        write_k = _store_first_tokens(new_k, num_new_tokens, target_batch)
        write_v = _store_first_tokens(new_v, num_new_tokens, target_batch)
    else:
        write_k = new_k
        write_v = new_v

    cache_k = kv_cache.k[:, :cache_length]
    cache_v = kv_cache.v[:, :cache_length]

    if num_new_tokens + cache_length > kv_capacity:
        num_evicted_tokens = num_new_tokens + cache_length - kv_capacity
        kept_k = cache_k[:, num_evicted_tokens:]
        kept_v = cache_v[:, num_evicted_tokens:]
        attend_k = torch.cat([kept_k, write_k], dim=1)
        attend_v = torch.cat([kept_v, write_v], dim=1)
    else:
        attend_k = torch.cat([cache_k, write_k], dim=1)
        attend_v = torch.cat([cache_v, write_v], dim=1)

    attend_length = attend_k.shape[1]
    attn_start = max(0, attend_length - max_attn_size)
    cached_k = attend_k[:, attn_start:attend_length]
    cached_v = attend_v[:, attn_start:attend_length]
    return cached_k, cached_v


def update_kv_cache_and_get_attended_kv(
    kv_cache: KVCache,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    num_new_tokens: int,
    max_attn_size: int,
    store_first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_capacity = kv_cache.capacity
    cache_length = kv_cache.length_int()

    cache_k = kv_cache.k[:, :cache_length]
    cache_v = kv_cache.v[:, :cache_length]

    if store_first_only:
        write_k = _store_first_tokens(new_k, num_new_tokens, kv_cache.batch_size)
        write_v = _store_first_tokens(new_v, num_new_tokens, kv_cache.batch_size)
    else:
        write_k = new_k
        write_v = new_v

    if num_new_tokens + cache_length > kv_capacity:
        num_evicted_tokens = num_new_tokens + cache_length - kv_capacity
        kept_k = cache_k[:, num_evicted_tokens:]
        kept_v = cache_v[:, num_evicted_tokens:]
        updated_k = torch.cat([kept_k, write_k], dim=1)
        updated_v = torch.cat([kept_v, write_v], dim=1)
    else:
        updated_k = torch.cat([cache_k, write_k], dim=1)
        updated_v = torch.cat([cache_v, write_v], dim=1)

    updated_length = min(updated_k.shape[1], kv_capacity)
    kv_cache.k[:, :updated_length] = updated_k[:, :updated_length]
    kv_cache.v[:, :updated_length] = updated_v[:, :updated_length]

    attn_start = max(0, updated_length - max_attn_size)
    cached_k = kv_cache.k[:, attn_start:updated_length]
    cached_v = kv_cache.v[:, attn_start:updated_length]

    kv_cache.set_length(updated_length)

    return cached_k, cached_v


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
    tokens_to_add = q.shape[1] if num_new_tokens is None else num_new_tokens

    if update_kv_cache:
        cached_k, cached_v = update_kv_cache_and_get_attended_kv(
            kv_cache=kv_cache,
            new_k=new_k,
            new_v=new_v,
            num_new_tokens=tokens_to_add,
            max_attn_size=max_attn_size,
            store_first_only=store_first_only,
        )
    else:
        cached_k, cached_v = get_attended_kv_without_update(
            kv_cache=kv_cache,
            new_k=new_k,
            new_v=new_v,
            num_new_tokens=tokens_to_add,
            max_attn_size=max_attn_size,
            store_first_only=store_first_only,
        )

    if repeat_factor is not None:
        cached_k = cached_k.repeat(repeat_factor, 1, 1, 1)
        cached_v = cached_v.repeat(repeat_factor, 1, 1, 1)

    return attend_fn(q, cached_k, cached_v)
