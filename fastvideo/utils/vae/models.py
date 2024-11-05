from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import fastvideo.utils.dit.joint_model.context_parallel as cp
from fastvideo.utils.vae.cp_conv import cp_pass_frames, gather_all_frames
from fastvideo.utils.vae.latent_dist import LatentDistribution


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class GroupNormSpatial(nn.GroupNorm):
    """
    GroupNorm applied per-frame.
    """

    def forward(self, x: torch.Tensor, *, chunk_size: int = 8):
        B, C, T, H, W = x.shape
        x = rearrange(x, "B C T H W -> (B T) C H W")
        # Run group norm in chunks.
        output = torch.empty_like(x)
        for b in range(0, B * T, chunk_size):
            output[b : b + chunk_size] = super().forward(x[b : b + chunk_size])
        return rearrange(output, "(B T) C H W -> B C T H W", B=B, T=T)


class SafeConv3d(torch.nn.Conv3d):
    """
    NOTE: No support for padding along time dimension.
          Input must already be padded along time.
    """

    def forward(self, input):
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        if memory_count > 2 and self.stride[0] == 1:
            part_num = int(memory_count / 2) + 1
            k = self.kernel_size[0]
            input_idx = torch.arange(k - 1, input.size(2))
            input_chunks_idx = torch.chunk(input_idx, part_num, dim=0)

            # assert self.stride[0] == 1, f"stride {self.stride}"
            assert self.dilation[0] == 1, f"dilation {self.dilation}"
            assert self.padding[0] == 0, f"padding {self.padding}"

            # Comptue output size
            assert not input.requires_grad
            B, _, T_in, H_in, W_in = input.shape
            output_size = (
                B,
                self.out_channels,
                T_in - k + 1,
                H_in // self.stride[1],
                W_in // self.stride[2],
            )
            output = torch.empty(output_size, dtype=input.dtype, device=input.device)
            for input_chunk_idx in input_chunks_idx:
                input_s = input_chunk_idx[0] - k + 1
                input_e = input_chunk_idx[-1] + 1
                input_chunk = input[:, :, input_s:input_e, :, :]
                output_chunk = super(SafeConv3d, self).forward(input_chunk)

                output_s = input_s
                output_e = output_s + output_chunk.size(2)
                output[:, :, output_s:output_e, :, :] = output_chunk

            return output
        else:
            return super(SafeConv3d, self).forward(input)


class StridedSafeConv3d(torch.nn.Conv3d):
    def forward(self, input, local_shard: bool = False):
        assert self.stride[0] == self.kernel_size[0]
        assert self.dilation[0] == 1
        assert self.padding[0] == 0

        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        T_in = input.size(2)
        T_out = T_in // kernel_size

        # Parallel implementation.
        if local_shard:
            idx = torch.arange(T_out)
            idx = cp.local_shard(idx, dim=0)
            start = idx.min() * stride
            end = idx.max() * stride + kernel_size
            local_input = input[:, :, start:end, :, :]
            return torch.nn.Conv3d.forward(self, local_input)

        raise NotImplementedError


class ContextParallelConv3d(SafeConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        causal: bool = True,
        context_parallel: bool = True,
        **kwargs,
    ):
        self.causal = causal
        self.context_parallel = context_parallel
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, height_pad, width_pad),
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        cp_rank, cp_world_size = cp.get_cp_rank_size()

        # Compute padding amounts.
        context_size = self.kernel_size[0] - 1
        if self.causal:
            pad_front = context_size
            pad_back = 0
        else:
            pad_front = context_size // 2
            pad_back = context_size - pad_front

        # Apply padding.
        mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
        if self.context_parallel and cp_world_size == 1:
            x = F.pad(x, (0, 0, 0, 0, pad_front, pad_back), mode=mode)
        else:
            if cp_rank == 0:
                x = F.pad(x, (0, 0, 0, 0, pad_front, 0), mode=mode)
            elif cp_rank == cp_world_size - 1 and pad_back:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_back), mode=mode)

        if self.context_parallel and cp_world_size == 1:
            return super().forward(x)

        if self.stride[0] == 1:
            # Receive some frames from previous rank.
            x = cp_pass_frames(x, context_size)
            return super().forward(x)

        # Less efficient implementation for strided convs.
        # All gather x, infer and chunk.
        assert x.dtype == torch.bfloat16, f"Expected x to be of type torch.bfloat16, got {x.dtype}"

        x = gather_all_frames(x)  # [B, C, k - 1 + global_T, H, W]
        return StridedSafeConv3d.forward(self, x, local_shard=True)


class Conv1x1(nn.Linear):
    """*1x1 Conv implemented with a linear layer."""

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, *] or [B, *, C].

        Returns:
            x: Output tensor. Shape: [B, C', *] or [B, *, C'].
        """
        x = x.movedim(1, -1)
        x = super().forward(x)
        x = x.movedim(-1, 1)
        return x

def norm_fn(
    in_channels: int,
    affine: bool = True,
):
    return GroupNormSpatial(affine=affine, num_groups=32, num_channels=in_channels)


class ResBlock(nn.Module):
    """Residual block that preserves the spatial dimensions."""

    def __init__(
        self,
        channels: int,
        *,
        affine: bool = True,
        attn_block: Optional[nn.Module] = None,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels

        assert causal
        self.stack = nn.Sequential(
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            ContextParallelConv3d(
                in_channels=channels,
                out_channels=channels // 2 if prune_bottleneck else channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=bias,
                causal=causal,
            ),
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            ContextParallelConv3d(
                in_channels=channels // 2 if prune_bottleneck else channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=bias,
                causal=causal,
            ),
        )

        self.attn_block = attn_block if attn_block else nn.Identity()

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
        """
        residual = x
        x = self.stack(x)
        x = x + residual
        del residual

        return self.attn_block(x)


def prepare_for_attention(qkv: torch.Tensor, head_dim: int, qk_norm: bool = True):
    """Prepare qkv tensor for attention and normalize qk.

    Args:
        qkv: Input tensor. Shape: [B, L, 3 * num_heads * head_dim].

    Returns:
        q, k, v: qkv tensor split into q, k, v. Shape: [B, num_heads, L, head_dim].
    """
    assert qkv.ndim == 3  # [B, L, 3 * num_heads * head_dim]
    assert qkv.size(2) % (3 * head_dim) == 0
    num_heads = qkv.size(2) // (3 * head_dim)
    qkv = qkv.unflatten(2, (3, num_heads, head_dim))

    q, k, v = qkv.unbind(2)  # [B, L, num_heads, head_dim]
    q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
    k = k.transpose(1, 2)  # [B, num_heads, L, head_dim]
    v = v.transpose(1, 2)  # [B, num_heads, L, head_dim]

    if qk_norm:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Mixed precision can change the dtype of normed q/k to float32.
        q = q.to(dtype=qkv.dtype)
        k = k.to(dtype=qkv.dtype)

    return q, k, v


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        out_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out = nn.Linear(dim, dim, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        chunk_size=2**15,
    ) -> torch.Tensor:
        """Compute temporal self-attention.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
            chunk_size: Chunk size for large tensors.

        Returns:
            x: Output tensor. Shape: [B, C, T, H, W].
        """
        B, _, T, H, W = x.shape

        if T == 1:
            # No attention for single frame.
            x = x.movedim(1, -1)  # [B, C, T, H, W] -> [B, T, H, W, C]
            qkv = self.qkv(x)
            _, _, x = qkv.chunk(3, dim=-1)  # Throw away queries and keys.
            x = self.out(x)
            return x.movedim(-1, 1)  # [B, T, H, W, C] -> [B, C, T, H, W]

        # 1D temporal attention.
        x = rearrange(x, "B C t h w -> (B h w) t C")
        qkv = self.qkv(x)

        # Input: qkv with shape [B, t, 3 * num_heads * head_dim]
        # Output: x with shape [B, num_heads, t, head_dim]
        q, k, v = prepare_for_attention(qkv, self.head_dim, qk_norm=self.qk_norm)

        attn_kwargs = dict(
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.head_dim**-0.5,
        )

        if q.size(0) <= chunk_size:
            x = F.scaled_dot_product_attention(q, k, v, **attn_kwargs)  # [B, num_heads, t, head_dim]
        else:
            # Evaluate in chunks to avoid `RuntimeError: CUDA error: invalid configuration argument.`
            # Chunks of 2**16 and up cause an error.
            x = torch.empty_like(q)
            for i in range(0, q.size(0), chunk_size):
                qc = q[i : i + chunk_size]
                kc = k[i : i + chunk_size]
                vc = v[i : i + chunk_size]
                chunk = F.scaled_dot_product_attention(qc, kc, vc, **attn_kwargs)
                x[i : i + chunk_size].copy_(chunk)

        assert x.size(0) == q.size(0)
        x = x.transpose(1, 2)  # [B, t, num_heads, head_dim]
        x = x.flatten(2)  # [B, t, num_heads * head_dim]

        x = self.out(x)
        x = rearrange(x, "(B h w) t C -> B C t h w", B=B, h=H, w=W)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        **attn_kwargs,
    ) -> None:
        super().__init__()
        self.norm = norm_fn(dim)
        self.attn = Attention(dim, **attn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))

def block_fn(channels, *, affine: bool = True, has_attention: bool = False, **block_kwargs):
    attn_block = AttentionBlock(channels) if has_attention else None
    return ResBlock(channels, affine=affine, attn_block=attn_block, **block_kwargs)


def add_fourier_features(inputs: torch.Tensor, start=6, stop=8, step=1):
    num_freqs = (stop - start) // step
    assert inputs.ndim == 5
    C = inputs.size(1)

    # Create Base 2 Fourier features.
    freqs = torch.arange(start, stop, step, dtype=inputs.dtype, device=inputs.device)
    assert num_freqs == len(freqs)
    w = torch.pow(2.0, freqs) * (2 * torch.pi)  # [num_freqs]
    C = inputs.shape[1]
    w = w.repeat(C)[None, :, None, None, None]  # [1, C * num_freqs, 1, 1, 1]

    # Interleaved repeat of input channels to match w.
    h = inputs.repeat_interleave(num_freqs, dim=1)  # [B, C * num_freqs, T, H, W]
    # Scale channels by frequency.
    h = w * h

    return torch.cat(
        [
            inputs,
            torch.sin(h),
            torch.cos(h),
        ],
        dim=1,
    )

class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks,
        *,
        temporal_reduction=2,
        spatial_reduction=2,
        **block_kwargs,
    ):
        """
        Downsample block for the VAE encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks.
            temporal_reduction: Temporal reduction factor.
            spatial_reduction: Spatial reduction factor.
        """
        super().__init__()
        layers = []

        assert in_channels != out_channels
        layers.append(
            ContextParallelConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(temporal_reduction, spatial_reduction, spatial_reduction),
                stride=(temporal_reduction, spatial_reduction, spatial_reduction),
                # First layer in each block always uses replicate padding
                padding_mode="replicate",
                bias=block_kwargs["bias"],
            )
        )

        for _ in range(num_res_blocks):
            layers.append(block_fn(out_channels, **block_kwargs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: List[int],
        latent_dim: int,
        temporal_reductions: List[int],
        spatial_reductions: List[int],
        prune_bottlenecks: List[bool],
        has_attentions: List[bool],
        affine: bool = True,
        bias: bool = True,
        input_is_conv_1x1: bool = False,
        padding_mode: str,
    ):
        super().__init__()
        self.temporal_reductions = temporal_reductions
        self.spatial_reductions = spatial_reductions
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.latent_dim = latent_dim

        ch = [mult * base_channels for mult in channel_multipliers]
        num_down_blocks = len(ch) - 1
        assert len(num_res_blocks) == num_down_blocks + 2

        layers = (
            [nn.Conv3d(in_channels, ch[0], kernel_size=(1, 1, 1), bias=True)]
            if not input_is_conv_1x1
            else [Conv1x1(in_channels, ch[0])]
        )

        assert len(prune_bottlenecks) == num_down_blocks + 2
        assert len(has_attentions) == num_down_blocks + 2
        block = partial(block_fn, padding_mode=padding_mode, affine=affine, bias=bias)

        for _ in range(num_res_blocks[0]):
            layers.append(block(ch[0], has_attention=has_attentions[0], prune_bottleneck=prune_bottlenecks[0]))
        prune_bottlenecks = prune_bottlenecks[1:]
        has_attentions = has_attentions[1:]

        assert len(temporal_reductions) == len(spatial_reductions) == len(ch) - 1
        for i in range(num_down_blocks):
            layer = DownsampleBlock(
                ch[i],
                ch[i + 1],
                num_res_blocks=num_res_blocks[i + 1],
                temporal_reduction=temporal_reductions[i],
                spatial_reduction=spatial_reductions[i],
                prune_bottleneck=prune_bottlenecks[i],
                has_attention=has_attentions[i],
                affine=affine,
                bias=bias,
                padding_mode=padding_mode,
            )

            layers.append(layer)

        # Additional blocks.
        for _ in range(num_res_blocks[-1]):
            layers.append(block(ch[-1], has_attention=has_attentions[-1], prune_bottleneck=prune_bottlenecks[-1]))

        self.layers = nn.Sequential(*layers)

        # Output layers.
        self.output_norm = norm_fn(ch[-1])
        self.output_proj = Conv1x1(ch[-1], 2 * latent_dim, bias=False)

    @property
    def temporal_downsample(self):
        return math.prod(self.temporal_reductions)

    @property
    def spatial_downsample(self):
        return math.prod(self.spatial_reductions)

    def forward(self, x) -> LatentDistribution:
        """Forward pass.

        Args:
            x: Input video tensor. Shape: [B, C, T, H, W]. Scaled to [-1, 1]

        Returns:
            means: Latent tensor. Shape: [B, latent_dim, t, h, w]. Scaled [-1, 1].
                   h = H // 8, w = W // 8, t - 1 = (T - 1) // 6
            logvar: Shape: [B, latent_dim, t, h, w].
        """
        assert x.ndim == 5, f"Expected 5D input, got {x.shape}"

        x = self.layers(x)

        x = self.output_norm(x)
        x = F.silu(x, inplace=True)
        x = self.output_proj(x)

        means, logvar = torch.chunk(x, 2, dim=1)

        assert means.ndim == 5
        assert logvar.shape == means.shape
        assert means.size(1) == self.latent_dim

        return LatentDistribution(means, logvar)