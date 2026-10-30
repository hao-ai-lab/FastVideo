"""Wan2.2 VAE decoder: a standalone PyTorch reference of diffusers' AutoencoderKLWan._decode (Wan2.2-TI2V-5B VAE).

decode(z, weights, dtype) maps latents z [B, 48, T, H/16, W/16] to a video [B, 3, 4T-3, H, W] with values in [-1, 1],
as AutoencoderKLWan.decode(z).sample does with tiling and slicing off (its defaults). It needs torch and nothing else:
`weights` is a plain dict name -> tensor of the decoder's and post_quant_conv's parameters under diffusers' names
(the 110 of wan22_vae_decode.weight_names()), such as the VAE's state dict:

    weights = {k: v.cuda().float() for k, v in vae.state_dict().items()}
    video = decode(latents.cuda(), weights)                    # fp32
    video = decode(latents.cuda(), weights, torch.bfloat16)    # a pipeline's bf16 decode

What is computed
----------------
The decoder of the TI2V-5B VAE (z_dim 48, decoder_base_dim 256, dim_mult [1, 2, 4, 4], num_res_blocks 2,
is_residual, patch_size 2, temporal upsampling in the first two up blocks, as the VAE's config.json says):

1. post_quant_conv: a 1x1x1 conv 48 -> 48, over all T latent frames at once.
2. The decoder, run on one latent frame at a time, in order (chunk i is latent frame i). A chunk sees the earlier
   ones only through the causal caches (below). With h x w = H/16 x W/16:
     conv_in     causal 3x3x3 conv 48 -> 1024                                                    at h x w
     mid block   residual block 1024, attention 1024, residual block 1024                        at h x w
     up block 0  3 residual blocks 1024; upsample3d 1024 (t x2, h w x2); + DupUp3D 1024 -> 1024     to 2h x 2w
     up block 1  3 residual blocks 1024; upsample3d 1024 (t x2, h w x2); + DupUp3D 1024 -> 1024     to 4h x 4w
     up block 2  residual blocks 1024 -> 512, 512, 512; upsample2d 512 (h w x2); + DupUp3D 1024 -> 512  to 8h x 8w
     up block 3  residual blocks 512 -> 256, 256, 256; no resampler, no shortcut                  at 8h x 8w
     norm_out    RMS norm 256, then SiLU
     conv_out    causal 3x3x3 conv 256 -> 12                                                     at 8h x 8w
   Up blocks 0 to 2 add their shortcut, DupUp3D of the block's input, to the output of their resampler.
   The chunks' outputs are concatenated along time.
3. unpatchify, [B, 12, F, 8h, 8w] -> [B, 3, F, H, W]: the 12 channels are (colour k, column offset j, row offset i)
   of 2 x 2 pixel patches, video[b, k, f, 2y + i, 2x + j] = out[b, 4k + 2j + i, f, y, x].
4. clamp to [-1, 1].

The parts, with diffusers' class names:
- causal conv (WanCausalConv3d), kernel kt x kh x kw, stride 1: zero padding of kh//2 and kw//2 on both sides in
  space; in time, 2 * (kt//2) frames before the chunk and none after: the frames of the conv's cache, and zeros for
  those the cache lacks.
- RMS norm (WanRMS_norm) over channels: F.normalize, x / max(||x||_2, 1e-12), times sqrt(C), times gamma.
- residual block (WanResidualBlock), in -> out channels: x1 = conv1(SiLU(norm1(x))), x2 = conv2(SiLU(norm2(x1))),
  causal 3x3x3 convs; it returns x2 + shortcut, the shortcut being x, or a 1x1x1 conv in -> out of x when in != out.
  (Dropout is 0, and diffusers decodes in eval mode.)
- attention (WanAttentionBlock), per frame: n = RMS norm of x; q, k and v are the channels [0, C), [C, 2C) and
  [2C, 3C) of a 1x1 conv C -> 3C of n; single-head attention softmax(q k^T / sqrt(C)) v over the h*w positions of the
  frame (head dim 1024, no mask: diffusers' docstring calls it causal, it is not); a 1x1 conv C -> C; plus x.
- upsample3d (WanResample "upsample3d"): the time conv, a causal 3x1x1 conv C -> 2C whose output channel c becomes
  frame 2t and channel C + c frame 2t + 1, then upsample2d. upsample2d (WanResample "upsample2d"): per frame, nearest
  x2 in h and w (mode nearest-exact, WanUpsample) and a 3x3 conv2d C -> C with zero padding 1.
- DupUp3D, in -> out channels, factor ft in time (2 in up blocks 0 and 1, 1 in up block 2) and 2 in h and w: each
  input channel repeated r = out * ft * 4 / in times, the copies laid out over the (ft, 2, 2) sub-positions:
  dup[c, ft*t + a, 2y + i, 2x + j] = x[(4*ft*c + 4a + 2i + j) // r, t, y, x]. In up blocks 0 and 1 (r = 8) that is
  nearest x2 in t, h and w; in up block 2 (r = 2) it is dup[c, t, 2y + i, 2x + j] = x[2c + i, t, y, x]: channel pairs
  become row pairs, each value repeated in two columns.

The first latent frame yields one frame
---------------------------------------
On the first chunk (diffusers' first_chunk):
- upsample3d skips the time conv: the chunk's frame passes through as one frame (diffusers marks the conv's cache
  "Rep"). The time conv's input sequence therefore begins at the second chunk, after zeros: it never sees latent
  frame 0. (Diffusers runs the second chunk's time conv without history and caches a zero frame in front of that
  chunk's frame; giving it no history, as here, and zero padding yields the same frames.)
- DupUp3D keeps only the last of its ft copies in time (x[:, :, ft - 1:]): one frame, as on the main path.
So chunk 0 gives one video frame and every later chunk four (each upsample3d doubles them): 1 + 4(T - 1) = 4T - 3.

Causal caches
-------------
Each causal conv with a 3-frame kernel (conv_in, conv1 and conv2 of every residual block, the two time convs,
conv_out) keeps the last two frames of its input, the activations it convolves; after a one-frame chunk, the previous
chunk's last frame and that one. The next chunk's conv sees them before its own frames, zero padding standing in for
what is missing (the first chunk has no history, a one-frame history lacks a frame). The loop thus computes, chunk by
chunk, what one causal conv over the whole sequence would, with zeros before the first frame. The 1x1x1 convs
(post_quant_conv, the residual shortcuts) need no cache.

Over the whole sequence at once, then, the decoder runs every causal conv over all the frames of its level, two zero
frames before the first; upsample3d's time conv over the frames after the first only, the first passing through; and
DupUp3D doubling every frame but the first. That form equals the frame loop up to rounding, not bit for bit, its convs
having other shapes (fp32: max |diff| 1.7e-5, mean 6e-7 on scripts/check_reference.py's latents).

Precision
---------
dtype=torch.float32 computes everything in fp32, as diffusers does in fp32 (both under the same torch.backends
settings; scripts/check_reference.py turns TF32 off).
dtype=torch.bfloat16 computes what diffusers does with the VAE in bf16 under torch.autocast("cuda", dtype=bfloat16),
which is how a video pipeline decodes with this VAE in bf16: the weights and the latents are rounded to bf16 (autocast
casts the latents at post_quant_conv); every conv and the attention take bf16 operands and return bf16 (autocast's
lower-precision ops; the kernels accumulate in fp32; for head dim 1024 PyTorch runs SDPA's memory-efficient kernel);
the RMS norm normalizes in fp32 and rounds to bf16 before the products with sqrt(C) and gamma, each of which rounds
again (diffusers' upcast for 16-bit inputs); the nearest upsampling runs in fp32 (WanUpsample's upcast; it only copies
values); SiLU, the sums, DupUp3D, unpatchify and clamp run in bf16. The output is bf16. decode turns autocast off
itself: the casts are explicit.

The operations and their order are diffusers', so that the same kernels see the same operands: on a GB200 with torch
2.12 and cuDNN 9.20, scripts/check_reference.py finds the reference bit-exact against diffusers in both modes. (The
bf16 decode itself is max 0.07, mean 0.003 from the fp32 one there.)

Credits
-------
Derived from the decoder of diffusers' AutoencoderKLWan (src/diffusers/models/autoencoders/autoencoder_kl_wan.py,
diffusers 0.39.0), Copyright 2025 The Wan Team and The HuggingFace Team, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0); diffusers' port of the Wan2.2 VAE of Wan-AI. Its structure, parameter
names and caching rules are theirs. Changes: the modules are rewritten as functions of a weight dict, the cache is
keyed by parameter name instead of a running index, the "Rep" marker is expressed as the time conv starting at the
second chunk, and the encoder, tiling and slicing are left out.
"""

import torch
import torch.nn.functional as F

CACHE_T = 2  # frames of history kept by a causal conv with a 3-frame kernel
PATCH_SIZE = 2
RESNETS_PER_UP_BLOCK = 3  # num_res_blocks + 1
# decoder.up_blocks.{i}: (in channels, out channels, temporal upsampling, spatial upsampling)
UP_BLOCKS = ((1024, 1024, True, True), (1024, 1024, True, True), (1024, 512, False, True), (512, 256, False, False))
PREFIXES = ("decoder.", "post_quant_conv.")


@torch.no_grad()
def decode(z, weights, dtype=torch.float32):
    """Decodes latents z [B, 48, T, H/16, W/16] into a video [B, 3, 4T-3, H, W] in [-1, 1], of `dtype`.

    `weights` maps diffusers' parameter names to tensors; those of the decoder and post_quant_conv are cast to `dtype`
    on z's device (no copy when they already are). dtype is torch.float32 or torch.bfloat16 (see the module docstring).
    """
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"dtype must be torch.float32 or torch.bfloat16, not {dtype}")
    weights = {name: t.to(z.device, dtype) for name, t in weights.items() if name.startswith(PREFIXES)}

    with torch.autocast(z.device.type, enabled=False):
        x = causal_conv(z.to(dtype), weights, "post_quant_conv")
        cache = {}
        chunks = [decode_chunk(x[:, :, i : i + 1], weights, cache, first=i == 0) for i in range(x.shape[2])]
        return unpatchify(torch.cat(chunks, dim=2), PATCH_SIZE).clamp(-1.0, 1.0)


def decode_chunk(x, weights, cache, first):
    """The decoder on one latent frame x [B, 48, 1, h, w]: [B, 12, 1, 8h, 8w] on the first chunk, [B, 12, 4, 8h, 8w]
    on the others. `cache` holds the causal convs' history, by conv name, and is updated."""
    x = cached_conv(x, weights, "decoder.conv_in", cache)
    x = residual_block(x, weights, "decoder.mid_block.resnets.0", cache)
    x = attention(x, weights, "decoder.mid_block.attentions.0")
    x = residual_block(x, weights, "decoder.mid_block.resnets.1", cache)
    for i in range(len(UP_BLOCKS)):
        x = up_block(x, weights, i, cache, first)
    x = F.silu(rms_norm(x, weights["decoder.norm_out.gamma"]))
    return cached_conv(x, weights, "decoder.conv_out", cache)


def up_block(x, weights, i, cache, first):
    """WanResidualUpBlock i: the residual blocks, then (up blocks 0 to 2) the resampler plus DupUp3D of x."""
    _, out_channels, temporal, spatial = UP_BLOCKS[i]
    name = f"decoder.up_blocks.{i}"
    block_input = x
    for j in range(RESNETS_PER_UP_BLOCK):
        x = residual_block(x, weights, f"{name}.resnets.{j}", cache)
    if not spatial:
        return x

    # The first chunk skips the time conv, whose history therefore starts empty at the second chunk.
    if temporal and not first:
        x = time_upsample(x, weights, f"{name}.upsampler.time_conv", cache)
    x = spatial_upsample(x, weights, f"{name}.upsampler.resample.1")
    return x + dup_up(block_input, out_channels, 2 if temporal else 1, 2, first)


def residual_block(x, weights, name, cache):
    """WanResidualBlock: conv2(SiLU(norm2(conv1(SiLU(norm1(x)))))) plus x, or plus a 1x1x1 conv of x."""
    shortcut = causal_conv(x, weights, f"{name}.conv_shortcut") if f"{name}.conv_shortcut.weight" in weights else x
    x = cached_conv(F.silu(rms_norm(x, weights[f"{name}.norm1.gamma"])), weights, f"{name}.conv1", cache)
    x = cached_conv(F.silu(rms_norm(x, weights[f"{name}.norm2.gamma"])), weights, f"{name}.conv2", cache)
    return x + shortcut


def attention(x, weights, name):
    """WanAttentionBlock: single-head attention over the h*w positions of each frame, plus x."""
    identity = x
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = rms_norm(x, weights[f"{name}.norm.gamma"])
    qkv = F.conv2d(x, weights[f"{name}.to_qkv.weight"], weights[f"{name}.to_qkv.bias"])
    qkv = qkv.reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous()  # [b*t, 1 head, h*w, 3C]
    q, k, v = qkv.chunk(3, dim=-1)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
    x = F.conv2d(x, weights[f"{name}.proj.weight"], weights[f"{name}.proj.bias"])
    return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4) + identity


def time_upsample(x, weights, name, cache):
    """upsample3d's time conv: a causal 3x1x1 conv C -> 2C; channel c becomes frame 2t, channel C + c frame 2t + 1."""
    b, c, t, h, w = x.shape
    x = cached_conv(x, weights, name, cache)
    x = x.reshape(b, 2, c, t, h, w)
    x = torch.stack((x[:, 0], x[:, 1]), 3)
    return x.reshape(b, c, t * 2, h, w)


def spatial_upsample(x, weights, name):
    """Per frame: nearest x2 in h and w (in fp32, as WanUpsample does; it only copies values), then a 3x3 conv2d."""
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = F.interpolate(x.float(), scale_factor=(2.0, 2.0), mode="nearest-exact").type_as(x)
    x = F.conv2d(x, weights[f"{name}.weight"], weights[f"{name}.bias"], padding=1)
    return x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)


def dup_up(x, out_channels, factor_t, factor_s, first):
    """DupUp3D: each channel repeated r = out * ft * fs * fs / in times, the copies spread over the (ft, fs, fs)
    sub-positions of time, rows and columns; on the first chunk only the last of the ft copies in time is kept."""
    b, c, t, h, w = x.shape
    x = x.repeat_interleave(out_channels * factor_t * factor_s * factor_s // c, dim=1)
    x = x.view(b, out_channels, factor_t, factor_s, factor_s, t, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(b, out_channels, t * factor_t, h * factor_s, w * factor_s)
    return x[:, :, factor_t - 1 :] if first else x


def rms_norm(x, gamma):
    """WanRMS_norm over channels (dim 1): F.normalize, in fp32 for a 16-bit x and rounded back, times sqrt(C), times
    gamma."""
    x_hat = F.normalize(x.float() if x.dtype in (torch.float16, torch.bfloat16) else x, dim=1).to(x.dtype)
    return x_hat * x.shape[1] ** 0.5 * gamma


def cached_conv(x, weights, name, cache):
    """causal_conv of one chunk with the history in cache[name], which it replaces by the last CACHE_T frames of its
    input: after a one-frame chunk, the previous chunk's last frame and this one."""
    history = cache.get(name)
    keep = x[:, :, -CACHE_T:].clone()
    if keep.shape[2] < CACHE_T and history is not None:
        keep = torch.cat([history[:, :, -1:], keep], dim=2)
    cache[name] = keep
    return causal_conv(x, weights, name, history)


def causal_conv(x, weights, name, history=None):
    """WanCausalConv3d `name` on x [B, C, t, h, w]: zero padding of kh//2 and kw//2 on both sides in space and, in
    time, 2 * (kt//2) frames before x: `history` (the frames before x, at most that many), then zeros."""
    weight, bias = weights[f"{name}.weight"], weights[f"{name}.bias"]
    kt, kh, kw = weight.shape[2:]
    pad = [kw // 2, kw // 2, kh // 2, kh // 2, 2 * (kt // 2), 0]
    if history is not None and pad[4] > 0:
        x = torch.cat([history, x], dim=2)
        pad[4] -= history.shape[2]
    return F.conv3d(F.pad(x, pad), weight, bias)


def unpatchify(x, p):
    """[B, 3*p*p, F, h, w] -> [B, 3, F, p*h, p*w]: channel (k*p + j)*p + i is colour k at row offset i, column j."""
    b, cp, f, h, w = x.shape
    x = x.view(b, cp // (p * p), p, p, f, h, w)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.view(b, cp // (p * p), f, h * p, w * p)
