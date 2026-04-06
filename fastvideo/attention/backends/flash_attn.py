# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass

try:
    from fastvideo.attention.utils.flash_attn_cute import flash_attn_func

    fa_version = "4"
except ImportError:
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func

        # flash_attn 3 no longer have a different API, see following commit:
        # https://github.com/Dao-AILab/flash-attention/commit/ed209409acedbb2379f870bbd03abce31a7a51b7
        flash_attn_func = flash_attn_3_func
        fa_version = "3"
    except ImportError:
        from flash_attn import flash_attn_func as flash_attn_2_func
        flash_attn_func = flash_attn_2_func
        fa_version = "2"

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)
logger.info("Using FlashAttention-%s backend", fa_version)

# FP4 FA4 support: quantize Q/K to NVFP4 E2M1 for block-scaled MMA on Blackwell.
# Requires: flash-attention-fp4, flashinfer, cutlass-dsl. Enable via FASTVIDEO_NVFP4_FA4=1.
# The FP4 path calls flash_attn.cute.interface directly (not the custom_op wrapper)
# because it needs mSFQ/mSFK kwargs that the wrapper doesn't support.
try:
    from flash_attn.cute.interface import flash_attn_func as _flash_attn_func_fp4
    _FA4_FP4_AVAILABLE = True
except ImportError:
    _flash_attn_func_fp4 = None
    _FA4_FP4_AVAILABLE = False


def _is_nvfp4_fa4_enabled():
    """Check if NVFP4 FA4 is enabled via environment variable."""
    return os.environ.get("FASTVIDEO_NVFP4_FA4", "0") == "1"


def _nvfp4_quantize_for_fa4(tensor_4d):
    """Quantize a (batch, seqlen, nheads, headdim) BF16 tensor to FP4.

    Returns:
        fp4_tensor: torch.float4_e2m1fn_x2, shape (batch, seqlen, nheads, headdim//2)
        sf_tensor:  torch.uint8, shape (32, 4, rest_m, 4, rest_k, nheads, batch) with stride[3]=1
    """
    from flashinfer.quantization import nvfp4_quantize, SfLayout

    batch, seqlen, nheads, headdim = tensor_4d.shape
    sf_vec_size = 16

    # Pad seqlen to multiple of 128 for SF layout atom
    tile_m = 128
    seqlen_padded = (seqlen + tile_m - 1) // tile_m * tile_m
    if seqlen_padded != seqlen:
        tensor_4d = F.pad(tensor_4d, (0, 0, 0, 0, 0, seqlen_padded - seqlen))

    # Quantize with nheads squashed into K dimension so M=batch*seqlen (divisible by 128)
    # and K=nheads*headdim. This ensures 128-row SF tiles align with seqlen boundaries.
    t2d = tensor_4d.reshape(batch * seqlen_padded, nheads * headdim)
    one = torch.ones(1, device=t2d.device, dtype=torch.float32)
    fp4_data, sf_data = nvfp4_quantize(t2d, one, sfLayout=SfLayout.layout_128x4, do_shuffle=False)

    # FP4 data: (batch*seqlen, nheads*headdim/2) → (batch, seqlen, nheads, headdim/2)
    fp4_tensor = (fp4_data
        .reshape(batch, seqlen_padded, nheads, headdim // 2)
        .view(torch.int8)
        .view(torch.float4_e2m1fn_x2)
    )

    # SF layout conversion: nvfp4_quantize layout_128x4 → FA4 MMA layout
    # layout_128x4 buffer: [mTile, kTile, 32, 4, 4]
    # FA4 expects: (32, 4, rest_m, 4, rest_k, nheads, batch) with stride[3]=1
    atom_m0, atom_m1, atom_k = 32, 4, 4
    rest_m = seqlen_padded // tile_m
    sf_k_per_head = headdim // sf_vec_size  # 8 for headdim=128
    rest_k = sf_k_per_head // atom_k  # 2

    total_m_tiles = batch * rest_m
    total_k_tiles = (nheads * sf_k_per_head) // atom_k

    sf_swizzled = sf_data.reshape(total_m_tiles, total_k_tiles, atom_m0, atom_m1, atom_k)
    sf_decomposed = sf_swizzled.reshape(batch, rest_m, nheads, rest_k, atom_m0, atom_m1, atom_k)
    sf_canonical = sf_decomposed.permute(0, 2, 1, 3, 4, 5, 6).contiguous()
    sf_mma = sf_canonical.permute(4, 5, 2, 6, 3, 1, 0)

    return fp4_tensor, sf_mma


class FlashAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


@dataclass
class FlashAttnMetadata(AttentionMetadata):
    current_timestep: int
    attn_mask: torch.Tensor | None = None


class FlashAttnMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
            self,
            current_timestep: int,
            attn_mask: torch.Tensor,
    ) -> FlashAttnMetadata:
        return FlashAttnMetadata(current_timestep=current_timestep, attn_mask=attn_mask)


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.nvfp4_fa4 = extra_impl_args.get("nvfp4_fa4", False) or _is_nvfp4_fa4_enabled()
        if self.nvfp4_fa4:
            cap = torch.cuda.get_device_capability()
            assert cap in [(10, 0), (10, 3)], (
                f"NVFP4 FA4 requires Blackwell (sm100a/sm103a), got sm{cap[0]}{cap[1]}"
            )
            assert _FA4_FP4_AVAILABLE, (
                "NVFP4 FA4 requires flash-attention-fp4 (flash_attn.cute). "
                "Install via instructions in docs/inference/optimizations.md"
            )
            logger.info("NVFP4 FA4 enabled for FlashAttentionImpl (quant_qk only)")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: FlashAttnMetadata,
    ):

        def _key_padding_mask_from_attn_mask(attn_mask: torch.Tensor, key_len: int) -> torch.Tensor:
            # Normalize attn_mask to [B, key_len] where True means valid token.
            if attn_mask.dim() == 4:
                attn_mask = attn_mask[:, 0, 0, :]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask[:, 0, :]
            elif attn_mask.dim() != 2:
                raise ValueError(f"Unsupported attn_mask shape for FLASH_ATTN: {attn_mask.shape}")

            # SDPA additive mask convention: valid=0, masked=-inf/large negative.
            key_padding_mask = attn_mask if attn_mask.dtype == torch.bool else attn_mask >= 0

            if key_padding_mask.shape[-1] != key_len:
                raise ValueError("Invalid key padding mask length for FLASH_ATTN: "
                                 f"expected {key_len}, got {key_padding_mask.shape[-1]}")
            return key_padding_mask

        if (attn_metadata is not None and hasattr(attn_metadata, "attn_mask") and attn_metadata.attn_mask is not None):
            from fastvideo.attention.utils.flash_attn_no_pad import (
                flash_attn_no_pad,
                flash_attn_varlen_qk_no_pad,
            )

            attn_mask = attn_metadata.attn_mask

            # flash_attn_no_pad packs q/k/v as one tensor and assumes equal q/k
            # sequence lengths. Cross-attention can violate this.
            if query.shape[1] != key.shape[1]:
                query_padding_mask = torch.ones(
                    (query.shape[0], query.shape[1]),
                    dtype=torch.bool,
                    device=query.device,
                )
                key_padding_mask = _key_padding_mask_from_attn_mask(attn_mask, key.shape[1]).to(device=key.device)
                return flash_attn_varlen_qk_no_pad(
                    query,
                    key,
                    value,
                    query_padding_mask=query_padding_mask,
                    key_padding_mask=key_padding_mask,
                    causal=self.causal,
                    dropout_p=0.0,
                    softmax_scale=self.softmax_scale,
                )

            qkv = torch.stack([query, key, value], dim=2)

            attn_mask = F.pad(attn_mask, (qkv.shape[1] - attn_mask.shape[1], 0), value=True)
            output = flash_attn_no_pad(qkv, attn_mask, causal=False, dropout_p=0, softmax_scale=None)
        elif self.nvfp4_fa4:
            output = self._forward_nvfp4(query, key, value)
        else:
            output = flash_attn_func(
                query,  # type: ignore[no-untyped-call]
                key,
                value,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )
        return output

    def _forward_nvfp4(self, query, key, value):
        """FP4 flash attention with quantized Q and K, BF16 V."""
        orig_seqlen_q = query.shape[1]
        orig_seqlen_k = key.shape[1]
        tile_m = 128

        q_fp4, q_sf = _nvfp4_quantize_for_fa4(query)
        k_fp4, k_sf = _nvfp4_quantize_for_fa4(key)

        # Pad V to match K's padded seqlen
        seqlen_k_padded = (orig_seqlen_k + tile_m - 1) // tile_m * tile_m
        if seqlen_k_padded != orig_seqlen_k:
            value = F.pad(value, (0, 0, 0, 0, 0, seqlen_k_padded - orig_seqlen_k))

        result = _flash_attn_func_fp4(
            q_fp4, k_fp4, value,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
            mSFQ=q_sf,
            mSFK=k_sf,
        )
        output = result[0] if isinstance(result, tuple) else result
        # Trim back to original seqlen
        if output.shape[1] != orig_seqlen_q:
            output = output[:, :orig_seqlen_q]
        return output
