# SPDX-License-Identifier: Apache-2.0

import torch


def update_kv_cache_and_get_attended_kv(
    kv_cache: dict[str, torch.Tensor | int],
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    current_start: int,
    num_new_tokens: int,
    max_attn_size: int,
    sink_tokens: int = 0,
    store_first_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_end = current_start + num_new_tokens
    kv_cache_size = kv_cache["k"].shape[1]

    original_global_end_index = (
        int(kv_cache["global_end_index"].item())
        if isinstance(kv_cache["global_end_index"], torch.Tensor)
        else int(kv_cache["global_end_index"])
    )
    original_local_end_index = (
        int(kv_cache["local_end_index"].item())
        if isinstance(kv_cache["local_end_index"], torch.Tensor)
        else int(kv_cache["local_end_index"])
    )

    if (current_end > original_global_end_index) and (
        num_new_tokens + original_local_end_index > kv_cache_size
    ):
        num_evicted_tokens = (
            num_new_tokens + original_local_end_index - kv_cache_size
        )
        num_rolled_tokens = (
            original_local_end_index - num_evicted_tokens - sink_tokens
        )
        kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
            kv_cache["k"][
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        )
        kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = (
            kv_cache["v"][
                :,
                sink_tokens + num_evicted_tokens : sink_tokens
                + num_evicted_tokens
                + num_rolled_tokens,
            ].clone()
        )
        local_end_index = (
            original_local_end_index
            + current_end
            - original_global_end_index
            - num_evicted_tokens
        )
    else:
        local_end_index = (
            original_local_end_index
            + current_end
            - original_global_end_index
        )
    local_start_index = local_end_index - num_new_tokens

    if store_first_only:
        kv_cache["k"][:, local_start_index:local_end_index] = new_k[:1]
        kv_cache["v"][:, local_start_index:local_end_index] = new_v[:1]
    else:
        kv_cache["k"][:, local_start_index:local_end_index] = new_k
        kv_cache["v"][:, local_start_index:local_end_index] = new_v

    cache_start = max(0, local_end_index - max_attn_size)
    cached_k = kv_cache["k"][:, cache_start:local_end_index]
    cached_v = kv_cache["v"][:, cache_start:local_end_index]

    if isinstance(kv_cache["global_end_index"], torch.Tensor):
        kv_cache["global_end_index"].fill_(current_end)
    else:
        kv_cache["global_end_index"] = current_end
    if isinstance(kv_cache["local_end_index"], torch.Tensor):
        kv_cache["local_end_index"].fill_(local_end_index)
    else:
        kv_cache["local_end_index"] = local_end_index

    return cached_k, cached_v
