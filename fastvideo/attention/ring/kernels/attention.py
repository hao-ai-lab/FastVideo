import re
from typing import Any

import torch

from ..capabilities import HAS_FLASH_ATTN

if HAS_FLASH_ATTN:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

    def _parse_version(version: str) -> tuple[int, ...]:
        # A plain string compare (e.g. "2.10.0" <= "2.6.3") is lexicographic
        # and misclassifies once a two-digit minor version ships, so parse
        # to an int tuple instead. Non-numeric suffixes (e.g. "2.6.3.post1")
        # are tolerated by defaulting an unparsable segment to 0.
        parts = []
        for part in version.split(".")[:3]:
            match = re.match(r"\d+", part)
            parts.append(int(match.group()) if match else 0)
        return tuple(parts)

    _FLASH_ATTN_VERSION = _parse_version(flash_attn.__version__)


def flash_attn_forward(q: torch.Tensor,
                       k: torch.Tensor,
                       v: torch.Tensor,
                       dropout_p: float = 0.0,
                       softmax_scale: float | None = None,
                       causal: bool = False,
                       window_size: tuple[int, int] = (-1, -1),
                       softcap: float | None = None,
                       alibi_slopes: torch.Tensor | None = None,
                       return_softmax: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert HAS_FLASH_ATTN, "FlashAttention is not available"
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    if _FLASH_ATTN_VERSION <= (2, 6, 3):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    else:
        block_out, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax,
        )
    return block_out, block_lse


def flash_attn_backward(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor,
                        softmax_lse: torch.Tensor, block_dq_buffer: torch.Tensor, block_dk_buffer: torch.Tensor,
                        block_dv_buffer: torch.Tensor, dropout_p: float, softmax_scale: float | None, bwd_causal: bool,
                        window_size: tuple[int, int], softcap: float | None, alibi_slopes: torch.Tensor | None,
                        deterministic: bool, rng_state: Any) -> None:
    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    assert HAS_FLASH_ATTN
    if _FLASH_ATTN_VERSION <= (2, 6, 3):
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )
    else:
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            block_dq_buffer,
            block_dk_buffer,
            block_dv_buffer,
            dropout_p,
            softmax_scale,
            bwd_causal,
            window_size[0],  # Pass window_size_left
            window_size[1],  # Pass window_size_right
            softcap,
            alibi_slopes,
            deterministic,
            rng_state,
        )
