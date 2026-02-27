# SPDX-License-Identifier: Apache-2.0

import torch


def _store_first_tokens(
    new_tensor: torch.Tensor,
    num_new_tokens: int,
    target_batch: int,
) -> torch.Tensor:
    first = new_tensor[:1]
    first = first.expand(target_batch, num_new_tokens, -1, -1)
    return first


def get_attended_kv_without_update(
    kv_cache: dict[str, torch.Tensor | int],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    num_new_tokens: int,
    max_attn_size: int,
    sink_tokens: int = 0,
    store_first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_capacity = kv_cache["k"].shape[1]
    target_batch = kv_cache["k"].shape[0]
    cache_length = (
        int(kv_cache["length"].item())
        if isinstance(kv_cache["length"], torch.Tensor)
        else int(kv_cache["length"])
    )

    if store_first_only:
        write_k = _store_first_tokens(new_k, num_new_tokens, target_batch)
        write_v = _store_first_tokens(new_v, num_new_tokens, target_batch)
    else:
        write_k = new_k
        write_v = new_v

    cache_k = kv_cache["k"][:, :cache_length]
    cache_v = kv_cache["v"][:, :cache_length]

    if num_new_tokens + cache_length > kv_capacity:
        num_evicted_tokens = num_new_tokens + cache_length - kv_capacity
        if sink_tokens > 0:
            kept_k = torch.cat(
                [
                    cache_k[:, :sink_tokens],
                    cache_k[:, sink_tokens + num_evicted_tokens :],
                ],
                dim=1,
            )
            kept_v = torch.cat(
                [
                    cache_v[:, :sink_tokens],
                    cache_v[:, sink_tokens + num_evicted_tokens :],
                ],
                dim=1,
            )
        else:
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
    kv_cache: dict[str, torch.Tensor | int],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    num_new_tokens: int,
    max_attn_size: int,
    sink_tokens: int = 0,
    store_first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_capacity = kv_cache["k"].shape[1]
    cache_length = (
        int(kv_cache["length"].item())
        if isinstance(kv_cache["length"], torch.Tensor)
        else int(kv_cache["length"])
    )

    cache_k = kv_cache["k"][:, :cache_length]
    cache_v = kv_cache["v"][:, :cache_length]

    if store_first_only:
        write_k = _store_first_tokens(new_k, num_new_tokens, kv_cache["k"].shape[0])
        write_v = _store_first_tokens(new_v, num_new_tokens, kv_cache["v"].shape[0])
    else:
        write_k = new_k
        write_v = new_v

    if num_new_tokens + cache_length > kv_capacity:
        num_evicted_tokens = num_new_tokens + cache_length - kv_capacity
        if sink_tokens > 0:
            kept_k = torch.cat(
                [
                    cache_k[:, :sink_tokens],
                    cache_k[:, sink_tokens + num_evicted_tokens :],
                ],
                dim=1,
            )
            kept_v = torch.cat(
                [
                    cache_v[:, :sink_tokens],
                    cache_v[:, sink_tokens + num_evicted_tokens :],
                ],
                dim=1,
            )
        else:
            kept_k = cache_k[:, num_evicted_tokens:]
            kept_v = cache_v[:, num_evicted_tokens:]
        updated_k = torch.cat([kept_k, write_k], dim=1)
        updated_v = torch.cat([kept_v, write_v], dim=1)
    else:
        updated_k = torch.cat([cache_k, write_k], dim=1)
        updated_v = torch.cat([cache_v, write_v], dim=1)

    updated_length = min(updated_k.shape[1], kv_capacity)
    kv_cache["k"][:, :updated_length] = updated_k[:, :updated_length]
    kv_cache["v"][:, :updated_length] = updated_v[:, :updated_length]

    attn_start = max(0, updated_length - max_attn_size)
    cached_k = kv_cache["k"][:, attn_start:updated_length]
    cached_v = kv_cache["v"][:, attn_start:updated_length]

    if isinstance(kv_cache["length"], torch.Tensor):
        kv_cache["length"].fill_(updated_length)
    else:
        kv_cache["length"] = updated_length

    return cached_k, cached_v
