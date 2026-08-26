# SPDX-License-Identifier: Apache-2.0
"""Face-driving modules for Wan2.2-Animate.

These are the parts of the Animate checkpoint that plain Wan-I2V does not
have: the LIA-style motion encoder that turns 512x512 face crops into
identity-free motion vectors, the causal-conv face encoder that funnels those
vectors down to the latent frame rate, and the face-adapter cross-attention
blocks bolted onto the 40-block tower (one after every
``inject_face_latents_blocks``-th block).

Ported against BOTH references -- the official ``Wan-Video/Wan2.2``
(``wan/modules/animate/{motion_encoder,face_blocks}.py``) and diffusers'
``transformer_wan_animate.py`` (merged; the official
``Wan-AI/Wan2.2-Animate-14B-Diffusers`` checkpoint follows the diffusers
naming, which this module reproduces exactly so every tensor loads verbatim).

**The StyleGAN2 weight-scaling contract.** The motion encoder is LIA's
appearance/motion network, built from StyleGAN2 ``EqualConv2d``/``EqualLinear``
layers: the checkpoint stores weights **unit-scale** and the layer multiplies
by ``1 / sqrt(fan_in)`` at *forward time* (plus a fused leaky-ReLU that adds a
channel bias and scales by ``sqrt(2)``). Loading these tensors into vanilla
``nn.Conv2d``/``nn.Linear`` forwards succeeds and is silently wrong by a
per-layer constant -- the classic "loads cleanly, generates garbage" failure.
``MotionConv2d``/``MotionLinear`` below reproduce the runtime scaling exactly.

Verified checkpoint shapes (Wan-AI/Wan2.2-Animate-14B-Diffusers):
  motion_encoder.conv_in.weight                (32, 3, 1, 1)
  motion_encoder.res_blocks.0.conv1.weight     (32, 32, 3, 3)
  motion_encoder.conv_out.weight               (512, 512, 4, 4)
  motion_encoder.motion_network.4.weight       (20, 512)
  motion_encoder.motion_synthesis_weight       (512, 20)
  face_encoder.conv1_local.weight              (4096, 512, 3)
  face_encoder.padding_tokens                  (1, 1, 1, 5120)
  face_adapter.0.to_q.weight                   (5120, 5120)
  face_adapter.0.norm_q.weight                 (128,)   # per-head RMSNorm
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from fastvideo.attention import LocalAttention
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.platforms import AttentionBackendEnum

# Shared with the DiT in wan_animate.py, which imports this module.
ANIMATE_ATTENTION_BACKENDS = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)

# LIA appearance-encoder channel width per feature-map size. config.json ships
# motion_encoder_channel_sizes: null, which means "use this table" (it is the
# table LIA and the official code hardcode; diffusers names it the same).
WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


class FusedLeakyReLU(nn.Module):
    """StyleGAN2's activation: add a channel bias, leaky-ReLU, scale by sqrt(2).

    The sqrt(2) keeps activation variance roughly constant through the leaky
    ReLU; the bias lives here (``act_fn.bias`` in the checkpoint) rather than
    on the conv because StyleGAN2 fuses bias-add into the activation kernel.
    """

    def __init__(self, negative_slope: float = 0.2, scale: float = 2**0.5,
                 bias_channels: int | None = None) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(bias_channels)) if bias_channels is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            expanded_shape = [1] * x.ndim
            expanded_shape[1] = self.bias.shape[0]
            x = x + self.bias.reshape(*expanded_shape)
        return F.leaky_relu(x, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    """StyleGAN2 EqualConv2d: unit-scale stored weight, ``1/sqrt(fan_in)`` at forward.

    ``blur_kernel`` implements the anti-aliasing FIR filter StyleGAN2 applies
    before strided (downsampling) convs. It is derived from a python tuple and
    registered non-persistent -- absent from the checkpoint, so it must be
    rebuilt after meta-device construction (see the model's
    ``materialize_non_persistent_buffers``).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True, blur_kernel: tuple[int, ...] | None = None,
                 use_activation: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.use_activation = use_activation

        self.blur = blur_kernel is not None
        self._blur_kernel_taps = tuple(blur_kernel) if blur_kernel is not None else None
        if self.blur:
            p = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((p + 1) // 2, p // 2)
            self.register_buffer("blur_kernel", self._build_blur_kernel(), persistent=False)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # The load-bearing constant: applied to the *stored* weight every forward.
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        # With an activation the bias is fused into it (act_fn.bias); without,
        # it lives on the conv. Mutually exclusive, exactly like the checkpoint.
        self.bias = nn.Parameter(torch.zeros(out_channels)) if (bias and not use_activation) else None
        self.act_fn = FusedLeakyReLU(bias_channels=out_channels) if use_activation else None

    def _build_blur_kernel(self) -> torch.Tensor:
        kernel = torch.tensor(self._blur_kernel_taps, dtype=torch.float32)
        kernel = kernel[None, :] * kernel[:, None]
        return kernel / kernel.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blur:
            expanded = self.blur_kernel[None, None, :, :].expand(self.in_channels, 1, -1, -1)
            x = F.conv2d(x, expanded.to(x.dtype), padding=self.blur_padding, groups=self.in_channels)
        # Cast activations to the weight dtype: face crops arrive fp32 from the
        # pipeline while the loaded weights are bf16, and a raw (no-autocast)
        # forward must not mix them -- the S2V dtype lesson, pre-applied.
        x = x.to(self.weight.dtype)
        x = F.conv2d(x, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class MotionLinear(nn.Module):
    """StyleGAN2 EqualLinear: unit-scale stored weight, ``1/sqrt(fan_in)`` at forward."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, use_activation: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if (bias and not use_activation) else None
        self.act_fn = FusedLeakyReLU(bias_channels=out_dim) if use_activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x.to(self.weight.dtype), self.weight * self.scale, bias=self.bias)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x


class MotionEncoderResBlock(nn.Module):
    """One 2x-downsampling step of the LIA appearance encoder.

    Main path: 3x3 conv + 3x3 strided conv; skip path: 1x1 strided conv (no
    bias); both blurred before the stride. The sum is divided by sqrt(2) to
    keep variance flat -- the same convention as the fused activation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 kernel_size_skip: int = 1, blur_kernel: tuple[int, ...] = (1, 3, 3, 1),
                 downsample_factor: int = 2) -> None:
        super().__init__()
        self.conv1 = MotionConv2d(in_channels, in_channels, kernel_size, stride=1,
                                  padding=kernel_size // 2, use_activation=True)
        self.conv2 = MotionConv2d(in_channels, out_channels, kernel_size, stride=downsample_factor,
                                  padding=0, blur_kernel=blur_kernel, use_activation=True)
        self.conv_skip = MotionConv2d(in_channels, out_channels, kernel_size_skip, stride=downsample_factor,
                                      padding=0, bias=False, blur_kernel=blur_kernel, use_activation=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.conv2(self.conv1(x)) + self.conv_skip(x)) / math.sqrt(2)


class WanAnimateMotionEncoder(nn.Module):
    """Face crop -> identity-free 512-d motion vector (LIA linear motion decomposition).

    A ``size``x``size`` RGB crop is squeezed to a ``style_dim`` appearance
    vector, bottlenecked to ``motion_dim`` (=20) coefficients -- too narrow to
    carry identity, wide enough to carry expression/pose -- and re-expanded as
    a linear combination of ``motion_dim`` learned direction vectors that are
    QR-orthonormalised at forward time (in fp32; upstream and diffusers both
    upcast, because the orthogonalisation is precision-sensitive).
    """

    def __init__(self, size: int = 512, style_dim: int = 512, motion_dim: int = 20,
                 out_dim: int = 512, motion_blocks: int = 5,
                 channels: dict[str, int] | None = None) -> None:
        super().__init__()
        self.size = size
        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)
        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        for i in range(int(math.log(size, 2)), 2, -1):
            out_channels = channels[str(2**(i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels
        self.conv_out = MotionConv2d(in_channels, style_dim, 4, padding=0, bias=False, use_activation=False)

        # No activations between these linears -- LIA's design, kept verbatim.
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        if face_image.shape[-2] != self.size or face_image.shape[-1] != self.size:
            raise ValueError(f"face crops are {tuple(face_image.shape[-2:])} but the motion encoder "
                             f"was trained on ({self.size}, {self.size}); resize the face video "
                             "before the pipeline, do not let it silently rescale")

        x = self.conv_in(face_image)
        for block in self.res_blocks:
            x = block(x)
        motion_feat = self.conv_out(x).squeeze(-1).squeeze(-1)
        for linear in self.motion_network:
            motion_feat = linear(motion_feat)

        # Linear motion decomposition: m = sum_i alpha_i * d_i with D orthonormal.
        weight = (self.motion_synthesis_weight + 1e-8).to(torch.float32)
        original_dtype = motion_feat.dtype
        motion_feat = motion_feat.to(torch.float32)
        q = torch.linalg.qr(weight)[0].to(motion_feat.device)
        motion_vec = torch.matmul(torch.diag_embed(motion_feat), q.T).sum(dim=1)
        return motion_vec.to(original_dtype)


class WanAnimateFaceEncoder(nn.Module):
    """Per-frame motion vectors -> per-latent-frame face tokens.

    The two stride-2 causal convs downsample time 4x -- exactly the VAE's
    temporal compression, so one token group lines up with one latent frame
    (the same clock-matching move as S2V's audio funnel). Emits
    ``num_heads`` content tokens + 1 learned padding token per frame.

    Convs are plain ``nn.Conv1d`` with the causal left-pad applied inline
    (checkpoint keys ``face_encoder.conv2.weight``, no ``.conv.`` wrapper --
    this differs from S2V's CausalConv1d module and is pinned by a test).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 1024, num_heads: int = 4,
                 kernel_size: int = 3, eps: float = 1e-6, pad_mode: str = "replicate") -> None:
        super().__init__()
        self.num_heads = num_heads
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode

        self.act = nn.SiLU()
        self.conv1_local = nn.Conv1d(in_dim, hidden_dim * num_heads, kernel_size, stride=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def _causal(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        """Channels-last in/out; left-pad so frame t depends only on frames <= t."""
        x = F.pad(x.permute(0, 2, 1), self.time_causal_padding, mode=self.pad_mode)
        return conv(x).permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._causal(self.conv1_local, x)  # [B, T, hidden * N]
        x = x.unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3).flatten(0, 1)  # [B*N, T, hidden]
        x = self.act(self.norm1(x))
        x = self.act(self.norm2(self._causal(self.conv2, x)))
        x = self.act(self.norm3(self._causal(self.conv3, x)))
        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)  # [B, T/4, N, out]
        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1).to(x.device, x.dtype)
        return torch.cat([x, padding], dim=-2)  # [B, T/4, N + 1, out]


class WanAnimateFaceCrossAttention(nn.Module):
    """One face hatch: each latent frame's video tokens query that frame's 5 face tokens.

    The per-frame confinement (reshape ``[B, T*S_f] -> [(B*T), S_f]``) is the
    entire expression-sync mechanism -- frame i can only see frame i's face.
    Applied residually *outside* this module, after every
    ``inject_face_latents_blocks``-th transformer block.

    qk-norm here is per-head RMSNorm over ``head_dim`` (checkpoint:
    ``norm_q.weight`` of shape (128,)) -- NOT the across-heads norm the main
    tower uses. Pre-norms are affine-free and contribute no tensors.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.attn = LocalAttention(num_heads=num_heads, head_size=self.head_dim, dropout_rate=0,
                                   softmax_scale=None, causal=False,
                                   supported_attention_backends=ANIMATE_ATTENTION_BACKENDS)

    def forward(self, hidden_states: torch.Tensor, motion_vec: torch.Tensor) -> torch.Tensor:
        """hidden_states [B, S, C]; motion_vec [B, T, N, C]. Returns the residual (not added)."""
        b, s, _ = hidden_states.shape
        t, n = motion_vec.shape[1], motion_vec.shape[2]
        if s % t:
            raise ValueError(
                f"face/video misalignment: {s} video tokens do not divide into {t} latent frames. "
                "The face video must produce exactly one motion-vector group per latent frame "
                "(pixel frames = 4 * video latent frames - 3, e.g. 77 for 20).")

        q_in = self.pre_norm_q(hidden_states)
        kv_in = self.pre_norm_kv(motion_vec).flatten(1, 2)  # [B, T*N, C]

        q = self.norm_q(self.to_q(q_in)[0].view(b, s, self.num_heads, self.head_dim))
        k = self.norm_k(self.to_k(kv_in)[0].view(b, t, n, self.num_heads, self.head_dim))
        v = self.to_v(kv_in)[0].view(b, t, n, self.num_heads, self.head_dim)

        # Per-frame confinement: [B, S] -> [(B T), S/T] queries against that
        # frame's [(B T), N] keys/values.
        q = q.unflatten(1, (t, s // t)).flatten(0, 1)
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)

        out = self.attn(q, k, v).flatten(2)  # [(B T), S/T, C]
        out = out.unflatten(0, (b, t)).flatten(1, 2)  # [B, S, C]
        return self.to_out(out)[0]
