# SPDX-License-Identifier: Apache-2.0
"""Audio conditioning modules for Wan2.2-S2V.

These are the parts of the S2V checkpoint that plain Wan does not have:
the audio bridge (``casual_audio_encoder``, upstream's spelling) that turns
wav2vec2 features into DiT-width tokens, and the audio injector
(``audio_injector``) that re-introduces those tokens into 12 of the 40
transformer blocks.

Shapes below are verified against ``Wan-AI/Wan2.2-S2V-14B``:
  casual_audio_encoder.weights                        (1, 25, 1, 1)
  casual_audio_encoder.encoder.conv1_local.conv.w     (5120, 1024, 3)
  casual_audio_encoder.encoder.conv1_global.conv.w    (1280, 1024, 3)
  casual_audio_encoder.encoder.conv2.conv.w           (2560, 1280, 3)
  casual_audio_encoder.encoder.conv3.conv.w           (5120, 2560, 3)
  casual_audio_encoder.encoder.final_linear.w         (5120, 5120)
  casual_audio_encoder.encoder.padding_tokens         (1, 1, 1, 5120)
  audio_injector.injector_adain_layers.N.linear.w     (10240, 5120)

The ``(1, 25, 1, 1)`` layer-weight tensor is load-bearing: 25 == the number of
hidden states returned by wav2vec2-**large** (24 layers + embeddings). The
bundled ``wav2vec2-large-xlsr-53-english`` is required; wav2vec2-base returns
13 and will not fit this tensor.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fastvideo.attention import LocalAttention
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.platforms import AttentionBackendEnum

# Shared with the DiT blocks in wan_s2v.py, which imports this module.
S2V_ATTENTION_BACKENDS = (AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA)


class CausalConv1d(nn.Module):
    """Conv1d that only ever looks backwards in time.

    Left-pads by ``kernel_size - 1`` so output frame t depends on input frames
    <= t. Audio conditioning must not leak future sound into earlier video
    frames, which is what makes the whole stack streamable.
    """

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1,
                 pad_mode: str = "replicate") -> None:
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.time_causal_padding, mode=self.pad_mode))


class MotionEncoder(nn.Module):
    """wav2vec2 features -> per-frame audio tokens (and a global summary).

    Two parallel paths over the same input:
      * local  -- ``num_heads`` (== num_audio_token) tokens per frame, the
                  content the video tokens cross-attend to.
      * global -- one summary vector per frame, used by the AdaIN path.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, need_global: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.need_global = need_global

        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        """Strided conv stack shared by the local and global paths: [B, C, T] -> [B, T, C].

        The convs want channels-first and the norms channels-last, hence the
        transpose around every step.
        """
        for norm, conv in ((self.norm1, self.conv2), (self.norm2, self.conv3)):
            x = conv(self.act(norm(x.transpose(1, 2))).transpose(1, 2))
        return self.act(self.norm3(x.transpose(1, 2)))

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        b = x.shape[0]

        local = self._trunk(rearrange(self.conv1_local(x), "b (n c) t -> (b n) c t", n=self.num_heads))
        local = rearrange(local, "(b n) t c -> b t n c", b=b)
        # One learned padding token per frame keeps the token count stable when
        # a frame has no valid audio window.
        local = torch.cat([local, self.padding_tokens.repeat(b, local.shape[1], 1, 1)], dim=-2)
        if not self.need_global:
            return local

        glob = self.final_linear(self._trunk(self.conv1_global(x)))
        return rearrange(glob, "(b n) t c -> b t n c", b=b), local


class CausalAudioEncoder(nn.Module):
    """The audio bridge: [B, 25, C_a, T] wav2vec2 stack -> DiT-width tokens.

    ``weights`` is a learned softmax over wav2vec2's 25 hidden states -- the
    model decides for itself which depth of the audio encoder matters, rather
    than hardcoding "use the last layer".
    """

    def __init__(self, dim: int = 1024, num_layers: int = 25, out_dim: int = 5120,
                 num_token: int = 4, need_global: bool = False) -> None:
        super().__init__()
        self.encoder = MotionEncoder(in_dim=dim, hidden_dim=out_dim, num_heads=num_token,
                                     need_global=need_global)
        # num_layers is the wav2vec2 hidden-state count (25 for wav2vec2-large),
        # NOT a function of num_token. Deriving it is a silent-failure bug.
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # The layer mixdown runs in fp32 (upstream wraps it in an fp32 autocast):
        # it divides by a small denominator (~0.125) while accumulating 25 terms,
        # and its output is the signal that drives lip-sync. This is elementwise,
        # so upcasting costs nothing and needs no autocast juggling.
        weights = self.act(self.weights.float())
        # [B, num_layers, C_a, T] -> [B, C_a, T] -> [B, T, C_a]
        weighted = ((features.float() * weights) / weights.sum(dim=1, keepdims=True)).sum(dim=1)
        return self.encoder(weighted.permute(0, 2, 1).to(self.encoder.conv1_local.conv.weight.dtype))


class AdaLayerNorm(nn.Module):
    """Scale/shift a hidden state from a conditioning vector.

    ``linear`` emits 2*dim (verified: 10240 for dim 5120) -- one shift and one
    scale. The norm itself is affine-free; all the affine behaviour comes from
    the audio, which is the point.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.act = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(self.act(temb)).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AudioCrossAttention(nn.Module):
    """One audio hatch: video tokens query, audio tokens answer.

    Structurally identical to Wan's own cross-attention (q/k/v/o + qk RMSNorm),
    which is why the checkpoint's injector tensors have the same shapes as
    ``blocks.N.cross_attn.*``.
    """

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = LocalAttention(num_heads=num_heads, head_size=self.head_dim, dropout_rate=0,
                                   softmax_scale=None, causal=False,
                                   supported_attention_backends=S2V_ATTENTION_BACKENDS)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        return self.to_out(self.attn(q, k, v).flatten(2))[0]


class AudioInjector(nn.Module):
    """The 12 audio hatches bolted onto the 40-block tower.

    ``injected_block_id`` maps *block index* -> *dense injector index*. The
    checkpoint stores injectors densely (injector.0 .. injector.11) with no
    record of which block each one serves; that correspondence exists only in
    ``audio_inject_layers``. Getting it wrong loads cleanly and steers the
    wrong layers, so the mapping is built once here and asserted.

    ``injector_pre_norm_feat`` / ``injector_pre_norm_vec`` are affine-free
    LayerNorms and therefore contribute no checkpoint tensors -- they are
    present so the module tree matches upstream.
    """

    def __init__(self, dim: int, num_heads: int, inject_layers: tuple[int, ...],
                 enable_adain: bool = True, adain_dim: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.injected_block_id = {block_id: i for i, block_id in enumerate(sorted(inject_layers))}
        n_inject = len(self.injected_block_id)
        assert n_inject == len(inject_layers), "duplicate entries in audio_inject_layers"

        self.injector = nn.ModuleList(
            [AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True, eps=eps) for _ in range(n_inject)])
        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(n_inject)])
        self.injector_pre_norm_vec = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(n_inject)])
        self.enable_adain = enable_adain
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                # eps is AdaLayerNorm's own default (1e-5, from diffusers), not the
                # model-wide 1e-6 -- upstream constructs these without passing eps.
                [AdaLayerNorm(adain_dim or dim) for _ in range(n_inject)])

    def forward(self, hidden_states: torch.Tensor, block_idx: int, audio_emb: torch.Tensor,
                audio_emb_global: torch.Tensor | None, original_seq_len: int,
                adain_mode: str = "attn_norm") -> torch.Tensor:
        """Residually inject audio into the video-token span of one block's output.

        hidden_states     [B, L, C] -- L covers video + ref + motion tokens.
        audio_emb         [B, F, N, C] -- per-frame audio tokens.
        original_seq_len  number of leading tokens that are video (the rest are
                          ref/motion and must not be touched).
        """
        if block_idx not in self.injected_block_id:
            return hidden_states
        idx = self.injected_block_id[block_idx]
        num_frames = audio_emb.shape[1]

        # Each video frame attends to its own audio window, so the video span must
        # split evenly into frames -- that split is the whole lip-sync alignment
        # mechanism. A mismatch means audio and video are misaligned in time, so
        # fail loudly here rather than let a reshape error surface deeper down.
        if original_seq_len % num_frames:
            raise ValueError(
                f"audio/video misalignment: {original_seq_len} video tokens do not divide into "
                f"{num_frames} audio frames. Check that the audio stage resampled to the video "
                f"latent frame count.")

        video_tokens = rearrange(hidden_states[:, :original_seq_len], "b (t n) c -> (b t) n c", t=num_frames)
        if self.enable_adain and adain_mode == "attn_norm":
            glob = rearrange(audio_emb_global, "b t n c -> (b t) n c")
            normed = self.injector_adain_layers[idx](video_tokens, temb=glob[:, 0])
        else:
            normed = self.injector_pre_norm_feat[idx](video_tokens)

        residual = self.injector[idx](normed, rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames))
        residual = rearrange(residual, "(b t) n c -> b (t n) c", t=num_frames)
        return torch.cat([hidden_states[:, :original_seq_len] + residual, hidden_states[:, original_seq_len:]], dim=1)
