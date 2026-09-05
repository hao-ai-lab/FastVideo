# SPDX-License-Identifier: Apache-2.0
"""Tests for the Wan2.2-Animate port.

Most of this file needs neither GPU memory nor the 34GB checkpoint, but it
does need a CUDA (or MPS) host: attention-backend selection has no CPU
implementation, so on a GPU-less machine only the config, mask/pad-helper,
motion-encoder and face-encoder tests run.

The tests pin the invariants that silently corrupt output when broken --
above all the StyleGAN2 runtime weight scaling in the motion encoder (loads
cleanly into vanilla layers and generates garbage), the pose add that must
skip the reference frame, and the per-frame confinement of the face adapter.

The weight-loading tests at the bottom need the real checkpoint::

    export WAN_ANIMATE_MODEL_PATH=/path/to/Wan2.2-Animate-14B-Diffusers
"""
import glob
import json
import math
import os
import re
import struct

import pytest
import torch
import torch.nn.functional as F

# The distributed_setup fixture rendezvouses via env:// even for a single
# GPU. setdefault (not assignment) keeps a launcher-assigned rendezvous
# intact, which the repo's port-inventory contract test requires.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29521")

from fastvideo.configs.models.dits.wan_animate import WanAnimateArchConfig, WanAnimateConfig
from fastvideo.configs.pipelines.wan import WanAnimate14BConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.wan_animate import WanAnimateTransformer3DModel
from fastvideo.models.dits.wan_animate_face import (FusedLeakyReLU, MotionConv2d, MotionLinear,
                                                    WanAnimateFaceCrossAttention, WanAnimateFaceEncoder,
                                                    WanAnimateMotionEncoder)
from fastvideo.pipelines.basic.wan.animate_stages import _fold_i2v_mask, _pad_frames

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_ANIMATE_PATH = os.path.join(_REPO_ROOT, "official_weights", "Wan2.2-Animate-14B-Diffusers")


def _animate_model_path() -> str:
    return os.environ.get("WAN_ANIMATE_MODEL_PATH", _DEFAULT_ANIMATE_PATH)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="Wan-Animate weight-loading tests require CUDA")

requires_weights = pytest.mark.skipif(
    not glob.glob(os.path.join(_animate_model_path(), "transformer", "*.safetensors")),
    reason=(f"No transformer safetensors under {_animate_model_path()} -- download "
            "Wan-AI/Wan2.2-Animate-14B-Diffusers or set WAN_ANIMATE_MODEL_PATH"))


# --------------------------------------------------------------------------
# Config: transcription fidelity and the asserts that guard silent failures
# --------------------------------------------------------------------------


def test_arch_config_matches_official_checkpoint_config() -> None:
    """The values WanAnimateArchConfig overrides or introduces, transcribed
    from Wan-AI/Wan2.2-Animate-14B-Diffusers transformer/config.json. The
    inherited tower geometry (40 layers, dim 5120, ...) is the base
    WanVideoArchConfig's and is exercised by every other Wan model."""
    a = WanAnimateArchConfig()
    assert a.in_channels == 36  # 16 noise + 4 mask + 16 conditional latent
    assert a.latent_channels == 16
    assert a.out_channels == 16
    assert a.image_dim == 1280  # CLIP ViT-H feature width
    assert a.added_kv_proj_dim == 5120  # I2V image cross-attention on
    assert a.motion_encoder_channel_sizes is None  # null -> LIA default table
    assert a.motion_encoder_size == 512
    assert a.motion_style_dim == 512
    assert a.motion_dim == 20
    assert a.motion_encoder_dim == 512
    assert a.face_encoder_hidden_dim == 1024
    assert a.face_encoder_num_heads == 4
    assert a.inject_face_latents_blocks == 5
    assert a.motion_encoder_batch_size == 8


def test_pipeline_config_wires_the_animate_dit() -> None:
    cfg = WanAnimate14BConfig()
    assert isinstance(cfg.dit_config, WanAnimateConfig)
    assert cfg.flow_shift == 5.0  # official wan_animate_14B.py: sample_shift
    # The VAE encoder is needed for the reference image, the pose video and
    # the previous segment's guidance frames -- not just decoding.
    assert cfg.vae_config.load_encoder is True
    assert cfg.vae_config.load_decoder is True


def test_in_channels_must_decompose_into_noise_mask_cond() -> None:
    with pytest.raises(AssertionError, match=r"2 \* latent_channels"):
        WanAnimateArchConfig(latent_channels=8)  # 36 != 2*8 + 4


def test_face_adapter_stride_must_divide_num_layers() -> None:
    """The checkpoint stores adapters densely (face_adapter.0..7) with no
    record of which block each serves; adapter i serves block i * stride.
    A stride that does not divide num_layers loads cleanly and steers the
    wrong blocks."""
    with pytest.raises(AssertionError, match="exact multiple"):
        WanAnimateArchConfig(inject_face_latents_blocks=7)  # 40 % 7 != 0


def _apply_mapping(name: str) -> str:
    for pattern, repl in WanAnimateArchConfig().param_names_mapping.items():
        new, count = re.subn(pattern, repl, name)
        if count:
            return new
    return name


def test_param_names_mapping_wraps_both_patchifiers_and_nothing_else() -> None:
    """pose_patch_embedding mirrors patch_embedding's .proj wrapping; the
    animate-specific modules keep checkpoint-identical names and must pass
    through verbatim."""
    assert _apply_mapping("patch_embedding.weight") == "patch_embedding.proj.weight"
    assert _apply_mapping("pose_patch_embedding.weight") == "pose_patch_embedding.proj.weight"
    assert _apply_mapping("pose_patch_embedding.bias") == "pose_patch_embedding.proj.bias"
    for verbatim in ("motion_encoder.conv_in.weight", "motion_encoder.res_blocks.0.conv1.act_fn.bias",
                     "motion_encoder.motion_synthesis_weight", "face_encoder.conv1_local.weight",
                     "face_encoder.padding_tokens", "face_adapter.0.to_q.weight", "face_adapter.7.norm_k.weight"):
        assert _apply_mapping(verbatim) == verbatim


# --------------------------------------------------------------------------
# Stage helpers: pure functions whose only failure mode is silently wrong video
# --------------------------------------------------------------------------


def test_pad_frames_reflects_like_the_reference() -> None:
    # The example from diffusers' pad_video_frames docstring, pinned exactly.
    assert _pad_frames(list("12345"), 10) == list("1234543212")


def test_pad_frames_truncates_and_handles_degenerate_inputs() -> None:
    assert _pad_frames(list("123"), 2) == list("12")
    assert _pad_frames(["x"], 4) == ["x"] * 4  # reflecting one frame is repetition
    with pytest.raises(ValueError, match="empty driving video"):
        _pad_frames([], 3)


def test_fold_i2v_mask_reference_slot_is_all_ones() -> None:
    mask = _fold_i2v_mask(None, 1, 4, 4, mask_len=1, temporal_ratio=4,
                          device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 4, 1, 4, 4)
    assert torch.all(mask == 1)


def test_fold_i2v_mask_animation_target_is_all_zeros() -> None:
    mask = _fold_i2v_mask(None, 5, 4, 4, mask_len=0, temporal_ratio=4,
                          device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 4, 5, 4, 4)
    assert torch.all(mask == 0)


def test_fold_i2v_mask_folds_pixel_frames_into_channels() -> None:
    """5 pixel frames -> 2 latent frames: frame 0 is repeated 4x and becomes
    all four channels of latent frame 0; frames 1..4 fold into latent frame 1."""
    pixel = torch.zeros(1, 1, 5, 2, 2)
    pixel[:, :, 0] = 1
    mask = _fold_i2v_mask(pixel, 2, 2, 2, mask_len=0, temporal_ratio=4,
                          device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1, 4, 2, 2, 2)
    assert torch.all(mask[:, :, 0] == 1)
    assert torch.all(mask[:, :, 1] == 0)


# --------------------------------------------------------------------------
# The StyleGAN2 runtime-scaling contract (the "loads cleanly, generates
# garbage" trap): checkpoint weights are stored unit-scale and every forward
# must multiply by 1/sqrt(fan_in).
# --------------------------------------------------------------------------


def test_motion_linear_applies_runtime_weight_scale() -> None:
    layer = MotionLinear(4, 3)
    x = torch.randn(2, 4)
    expected = F.linear(x, layer.weight * (1 / math.sqrt(4)), bias=layer.bias)
    torch.testing.assert_close(layer(x), expected)


def test_motion_conv_applies_runtime_weight_scale() -> None:
    conv = MotionConv2d(2, 3, kernel_size=1, bias=False, use_activation=False)
    x = torch.randn(1, 2, 4, 4)
    expected = F.conv2d(x, conv.weight * (1 / math.sqrt(2 * 1**2)))
    torch.testing.assert_close(conv(x), expected)


def test_motion_layers_cast_activations_to_the_weight_dtype() -> None:
    """Face crops arrive fp32 while loaded weights are bf16; a raw
    (no-autocast) forward must cast rather than raise on the mixed matmul."""
    conv = MotionConv2d(2, 3, kernel_size=1, use_activation=False).to(torch.bfloat16)
    assert conv(torch.randn(1, 2, 4, 4)).dtype == torch.bfloat16
    linear = MotionLinear(4, 3).to(torch.bfloat16)
    assert linear(torch.randn(2, 4)).dtype == torch.bfloat16


def test_fused_leaky_relu_adds_bias_then_scales() -> None:
    act = FusedLeakyReLU(bias_channels=3)
    with torch.no_grad():
        act.bias.copy_(torch.tensor([0.5, -0.5, 0.0]))
    x = torch.randn(2, 3, 4, 4)
    expected = F.leaky_relu(x + act.bias.view(1, 3, 1, 1), 0.2) * math.sqrt(2)
    torch.testing.assert_close(act(x), expected)


_TINY_MOTION_CHANNELS = {"16": 8, "8": 8, "4": 8}


def test_motion_encoder_bottlenecks_to_motion_dim_then_resynthesises() -> None:
    """The 20-d bottleneck (here 4-d) is the identity squeeze: the output is a
    linear combination of motion_dim orthonormal directions, so it cannot
    carry more than motion_dim degrees of freedom."""
    enc = WanAnimateMotionEncoder(size=16, style_dim=8, motion_dim=4, out_dim=8,
                                  channels=_TINY_MOTION_CHANNELS)
    assert len(enc.res_blocks) == 2  # feature map 16 -> 8 -> 4
    assert tuple(enc.motion_network[-1].weight.shape) == (4, 8)  # style -> motion code
    assert tuple(enc.motion_synthesis_weight.shape) == (8, 4)  # out_dim x motion_dim
    out = enc(torch.randn(3, 3, 16, 16))
    assert out.shape == (3, 8)


def test_motion_synthesis_ignores_the_stored_basis_scale() -> None:
    """QR orthonormalises the basis at forward time, so rescaling the stored
    weight cannot change the motion vector. Skipping the QR would scale the
    output with the weight."""
    enc = WanAnimateMotionEncoder(size=16, style_dim=8, motion_dim=4, out_dim=8,
                                  channels=_TINY_MOTION_CHANNELS)
    x = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        base = enc(x)
        enc.motion_synthesis_weight.mul_(4.0)
        rescaled = enc(x)
    # Loose tolerance: the +1e-8 anti-degeneracy offset does not scale with the weight.
    torch.testing.assert_close(base, rescaled, atol=1e-4, rtol=1e-4)


def test_default_motion_encoder_walks_the_full_lia_channel_table() -> None:
    """512 -> 4 halvings with the LIA widths; a hole in the table is a
    KeyError at construction and a wrong width is a shape mismatch at load.
    Meta device: structure only, no memory."""
    with torch.device("meta"):
        enc = WanAnimateMotionEncoder()
    assert len(enc.res_blocks) == 7  # 512 -> 256 -> ... -> 4
    assert tuple(enc.conv_in.weight.shape) == (32, 3, 1, 1)  # channels["512"] = 32
    assert tuple(enc.conv_out.weight.shape) == (512, 512, 4, 4)


def test_motion_encoder_rejects_wrong_crop_size() -> None:
    enc = WanAnimateMotionEncoder(size=16, style_dim=8, motion_dim=4, out_dim=8,
                                  channels=_TINY_MOTION_CHANNELS)
    with pytest.raises(ValueError, match="face crops"):
        enc(torch.randn(1, 3, 32, 32))


# --------------------------------------------------------------------------
# Face encoder and face adapter
# --------------------------------------------------------------------------


def test_face_encoder_downsamples_time_4x_and_appends_padding_token() -> None:
    """Two stride-2 causal convs match the VAE's temporal compression: 13
    pixel frames -> ceil(13/2)=7 -> ceil(7/2)=4 latent-frame groups, each
    num_heads content tokens + 1 learned padding token."""
    enc = WanAnimateFaceEncoder(in_dim=8, out_dim=16, hidden_dim=8, num_heads=2)
    with torch.no_grad():
        enc.padding_tokens.fill_(0.5)  # a marker: zeros would match a dropped weight
    out = enc(torch.randn(1, 13, 8))
    assert out.shape == (1, 4, 3, 16)  # [B, T/4, num_heads + 1, out_dim]
    # The last row is the learned padding token, broadcast to every frame.
    torch.testing.assert_close(out[:, :, -1], torch.full((1, 4, 16), 0.5))


def test_face_encoder_state_dict_matches_checkpoint_naming() -> None:
    """The Animate checkpoint stores plain Conv1d keys (face_encoder.conv2.weight)
    with the causal pad applied inline -- no wrapper sublevel like a
    `.conv.weight`. Renaming any of these breaks verbatim loading."""
    enc = WanAnimateFaceEncoder(in_dim=8, out_dim=16, hidden_dim=8, num_heads=2)
    assert set(enc.state_dict().keys()) == {
        "conv1_local.weight", "conv1_local.bias", "conv2.weight", "conv2.bias", "conv3.weight",
        "conv3.bias", "out_proj.weight", "out_proj.bias", "padding_tokens"
    }


def _init_params(module: torch.nn.Module) -> None:
    # FastVideo's linear layers allocate weights with torch.empty (values
    # normally arrive from a checkpoint), so stand in for loaded weights or
    # forwards read uninitialised memory: nondeterministic outputs, maybe NaN.
    # Scaling matters: a flat std=0.02 attenuates the face signal into
    # numerical dust over its ~15-layer path (verified on GPU: face influence
    # became bitwise invisible at the output). Weights get unit gain per layer;
    # the StyleGAN2-style motion-encoder layers divide by sqrt(fan_in) at
    # runtime and therefore expect unit-scale weights.
    for name, p in module.named_parameters():
        if p.dim() <= 1:
            torch.nn.init.normal_(p, std=0.02)
        elif "motion_encoder" in name:
            torch.nn.init.normal_(p, std=1.0)
        else:
            fan_in = p.shape[1] * (p[0][0].numel() if p.dim() > 2 else 1)
            torch.nn.init.normal_(p, std=fan_in**-0.5)


def test_face_attention_confines_each_frame_to_its_own_tokens() -> None:
    """Per-frame confinement IS the expression-sync mechanism: frame i's video
    tokens may only see frame i's face tokens. A leak here trains nothing and
    crashes nothing -- expressions just smear across time."""
    torch.manual_seed(0)
    attn = WanAnimateFaceCrossAttention(dim=16, num_heads=2)
    _init_params(attn)
    frames, tokens_per_frame, n_face = 3, 4, 3
    hs = torch.randn(1, frames * tokens_per_frame, 16)
    motion = torch.randn(1, frames, n_face, 16)
    perturbed = motion.clone()
    # Perturb per-channel: the affine-free pre-norm LayerNorm exactly cancels
    # a channel-constant offset (LN(x + c) == LN(x)), which would make this
    # test pass even with confinement broken.
    perturbed[:, 1] += torch.randn_like(perturbed[:, 1])
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        base = attn(hs, motion)
        moved = attn(hs, perturbed)
    delta = (base - moved).abs().reshape(1, frames, tokens_per_frame, 16)
    assert delta[:, 1].max() > 1e-4  # frame 1 genuinely moved
    torch.testing.assert_close(delta[:, 0], torch.zeros_like(delta[:, 0]))
    torch.testing.assert_close(delta[:, 2], torch.zeros_like(delta[:, 2]))


# --------------------------------------------------------------------------
# End-to-end behaviour on a tiny model with the real structure
# --------------------------------------------------------------------------


def _tiny_arch(**overrides) -> WanAnimateArchConfig:
    kwargs = dict(
        num_attention_heads=2,
        # head_dim 8 is not in FlashAttention's supported sizes, so backend
        # selection falls back to SDPA even if another test left the global
        # compute dtype at bf16 -- keeps the fp32 CPU-side forward stable.
        attention_head_dim=8,  # dim = 16
        ffn_dim=32,
        num_layers=2,
        inject_face_latents_blocks=1,
        latent_channels=4,
        in_channels=12,  # 2*4 + 4
        out_channels=4,
        text_dim=16,
        image_dim=16,
        added_kv_proj_dim=16,
        motion_encoder_size=16,
        motion_encoder_channel_sizes=dict(_TINY_MOTION_CHANNELS),
        motion_style_dim=8,
        motion_dim=4,
        motion_encoder_dim=8,
        face_encoder_hidden_dim=8,
        face_encoder_num_heads=2,
        motion_encoder_batch_size=4,
    )
    kwargs.update(overrides)
    return WanAnimateArchConfig(**kwargs)


def _tiny_model(arch: WanAnimateArchConfig | None = None) -> WanAnimateTransformer3DModel:
    arch = arch or _tiny_arch()
    model = WanAnimateTransformer3DModel(config=WanAnimateConfig(arch_config=arch), hf_config={})
    _init_params(model)
    return model.eval()


def _tiny_inputs(arch: WanAnimateArchConfig, video_frames: int = 2, hw: int = 8):
    """video_frames latent frames + 1 reference slot; face pixel frames = 4T - 3."""
    total_frames = video_frames + 1
    return dict(
        hidden_states=torch.randn(1, arch.in_channels, total_frames, hw, hw),
        encoder_hidden_states=torch.randn(1, 7, arch.text_dim),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states_image=torch.randn(1, 257, arch.image_dim),
        pose_latents=torch.randn(1, arch.latent_channels, video_frames, hw, hw),
        face_pixel_values=torch.randn(1, 3, 4 * video_frames - 3, arch.motion_encoder_size,
                                      arch.motion_encoder_size),
    )


@pytest.mark.usefixtures("distributed_setup")
def test_forward_returns_full_sequence_shaped_latents() -> None:
    """The denoiser emits the whole sequence including the reference slot;
    dropping that slot is the decode stage's job, not the model's."""
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(**inputs)
    assert out.shape == (1, arch.out_channels, 3, 8, 8)


@pytest.mark.usefixtures("distributed_setup")
def test_forward_is_deterministic() -> None:
    # Also the only guard against input mutation: _patchify_with_pose writes
    # in place, legal only because it writes a fresh conv output.
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        torch.testing.assert_close(model(**inputs), model(**inputs))


@pytest.mark.usefixtures("distributed_setup")
def test_forward_responds_to_pose_and_face() -> None:
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        base = model(**inputs)
        out_pose = model(**{**inputs, "pose_latents": inputs["pose_latents"] + 1.0})
        out_face = model(**{**inputs, "face_pixel_values": inputs["face_pixel_values"] + 1.0})
    assert not torch.allclose(base, out_pose)
    assert not torch.allclose(base, out_face)


@pytest.mark.usefixtures("distributed_setup")
def test_pose_add_skips_reference_frame() -> None:
    """Frame 0 is the reference latent and carries no pose; adding pose there
    corrupts identity conditioning without any error."""
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with torch.no_grad():
        with_pose = model._patchify_with_pose(inputs["hidden_states"], inputs["pose_latents"])
        no_pose = model._patchify_with_pose(inputs["hidden_states"],
                                            torch.zeros_like(inputs["pose_latents"]))
    torch.testing.assert_close(with_pose[:, :, 0], no_pose[:, :, 0])
    assert not torch.allclose(with_pose[:, :, 1:], no_pose[:, :, 1:])


@pytest.mark.usefixtures("distributed_setup")
def test_batched_clips_do_not_interleave_their_faces() -> None:
    """_encode_face flattens (batch, frame) to one axis and back; a transposed
    reshape would silently interleave two clips' faces in batched inference,
    which batch-size-1 tests can never catch."""
    arch = _tiny_arch()
    model = _tiny_model(arch)
    clip_a = torch.randn(1, 3, 5, 16, 16)
    clip_b = torch.randn(1, 3, 5, 16, 16)
    with torch.no_grad():
        single_a = model._encode_face(clip_a)
        single_b = model._encode_face(clip_b)
        batched = model._encode_face(torch.cat([clip_a, clip_b]))
    torch.testing.assert_close(batched[0], single_a[0])
    torch.testing.assert_close(batched[1], single_b[0])


@pytest.mark.usefixtures("distributed_setup")
def test_face_adapter_fires_after_configured_blocks() -> None:
    # Stride 2 over 4 layers: with stride 1, both idx % stride and idx // stride
    # are degenerate and any index-mangling would pass.
    arch = _tiny_arch(num_layers=4, inject_face_latents_blocks=2)
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    assert len(model.face_adapter) == 2
    calls: list[tuple[str, int]] = []
    for i, block in enumerate(model.blocks):
        block.register_forward_hook(lambda m, a, o, i=i: calls.append(("block", i)))
    for i, adapter in enumerate(model.face_adapter):
        adapter.register_forward_hook(lambda m, a, o, i=i: calls.append(("adapter", i)))
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        model(**inputs)
    assert calls == [("block", 0), ("adapter", 0), ("block", 1), ("block", 2), ("adapter", 1), ("block", 3)]


@pytest.mark.usefixtures("distributed_setup")
def test_missing_pose_or_face_fails_loudly() -> None:
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with pytest.raises(ValueError, match="pose video and a face video"):
        model(**{**inputs, "pose_latents": None})
    with pytest.raises(ValueError, match="pose video and a face video"):
        model(**{**inputs, "face_pixel_values": None})


@pytest.mark.usefixtures("distributed_setup")
def test_missing_clip_image_embeds_fails_loudly() -> None:
    """The I2V cross-attention splits the first 257 context tokens off as
    image tokens positionally; the guard turns what would be a bare TypeError
    at the concat into an error that names the missing input."""
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    with pytest.raises(ValueError, match="CLIP features"):
        model(**{**inputs, "encoder_hidden_states_image": None})


@pytest.mark.usefixtures("distributed_setup")
def test_pose_frame_count_mismatch_raises() -> None:
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    bad_pose = torch.randn(1, arch.latent_channels, 3, 8, 8)  # must be T (=2), not T+1
    with pytest.raises(ValueError, match="exactly one fewer"):
        model(**{**inputs, "pose_latents": bad_pose})


@pytest.mark.usefixtures("distributed_setup")
def test_face_frame_count_mismatch_raises() -> None:
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(arch)
    bad_face = torch.randn(1, 3, 9, 16, 16)  # 9 -> 5 -> 3 groups + pad = 4 != 3 frames
    with pytest.raises(ValueError, match="face/video misalignment"):
        with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
            model(**{**inputs, "face_pixel_values": bad_face})


def test_model_is_discoverable_by_class_name() -> None:
    """EntryClass auto-discovery must register the class under the exact name
    the checkpoint's _class_name uses -- there is no alias table entry, so
    this is the only wiring to break."""
    from fastvideo.models.registry import _FAST_VIDEO_MODELS
    assert "WanAnimateTransformer3DModel" in _FAST_VIDEO_MODELS
    component, module, cls = _FAST_VIDEO_MODELS["WanAnimateTransformer3DModel"]
    assert (component, module, cls) == ("dits", "wan_animate", "WanAnimateTransformer3DModel")


@pytest.mark.usefixtures("distributed_setup")
def test_blur_kernels_survive_meta_device_construction() -> None:
    """TransformerLoader builds the model on the meta device; the blur kernels
    are non-persistent (absent from the checkpoint) and must be rebuilt by the
    materialize hook or the motion encoder's first forward crashes on meta."""
    arch = _tiny_arch()
    with torch.device("meta"):
        model = WanAnimateTransformer3DModel(config=WanAnimateConfig(arch_config=arch), hf_config={})
    blurred_convs = [m for m in model.modules() if isinstance(m, MotionConv2d) and m.blur]
    assert blurred_convs and all(m.blur_kernel.is_meta for m in blurred_convs)
    model.materialize_non_persistent_buffers(device=torch.device("cpu"))
    for m in blurred_convs:
        assert not m.blur_kernel.is_meta
        torch.testing.assert_close(m.blur_kernel.sum(), torch.tensor(1.0))  # normalised FIR taps


@pytest.mark.usefixtures("distributed_setup")
def test_motion_synthesis_weight_is_pinned_to_fp32() -> None:
    """Diffusers keeps this tensor fp32 (_keep_in_fp32_modules): the QR basis
    is re-derived from it every forward, so loading it in bf16 changes the
    basis itself. The loader consults _get_parameter_dtype per parameter."""
    arch = _tiny_arch()
    model = _tiny_model(arch)
    assert model._get_parameter_dtype("motion_encoder.motion_synthesis_weight",
                                      torch.bfloat16) == torch.float32
    assert model._get_parameter_dtype("blocks.0.to_q.weight", torch.bfloat16) == torch.bfloat16


# --------------------------------------------------------------------------
# Weight-gated: the real checkpoint
# --------------------------------------------------------------------------


def _checkpoint_tensor_shapes(transformer_dir: str) -> dict[str, tuple]:
    shapes: dict[str, tuple] = {}
    for path in sorted(glob.glob(os.path.join(transformer_dir, "*.safetensors"))):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for tensor_name, meta in header.items():
            if tensor_name != "__metadata__":
                shapes[tensor_name] = tuple(meta["shape"])
    return shapes


@requires_weights
@pytest.mark.usefixtures("distributed_setup")
def test_every_checkpoint_tensor_has_a_home_in_the_model() -> None:
    """Meta-device strict mapping check: every transformer tensor in the real
    checkpoint must land on a model parameter with the right shape, and every
    model parameter must be fed. Costs no GPU and no memory.

    Settled on the real weights: the checkpoint ships no attn2.norm_added_q
    (the dead weight base Wan-I2V checkpoints carry), so the Animate model
    drops that parameter at construction -- both directions of this check
    must therefore come out empty.
    """
    shapes = _checkpoint_tensor_shapes(os.path.join(_animate_model_path(), "transformer"))
    assert shapes, "no tensors found in checkpoint"

    with torch.device("meta"):
        model = WanAnimateTransformer3DModel(config=WanAnimateConfig(), hf_config={})
    model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    mapped = {_apply_mapping(k): v for k, v in shapes.items()}

    missing = sorted(set(model_shapes) - set(mapped))
    unexpected = sorted(set(mapped) - set(model_shapes))
    mismatched = sorted(k for k in set(mapped) & set(model_shapes) if mapped[k] != model_shapes[k])
    assert not missing and not unexpected and not mismatched, (
        f"missing={missing[:5]} unexpected={unexpected[:5]} mismatched={mismatched[:5]} "
        f"(counts: {len(missing)}/{len(unexpected)}/{len(mismatched)})")


@requires_cuda
@requires_weights
@pytest.mark.usefixtures("distributed_setup")
def test_transformer_loads_and_runs_a_forward_pass() -> None:
    """Load every weight through the real loader and run one raw (no-autocast)
    bf16 forward -- the pass that exposes the dtype seams autocast otherwise
    hides."""
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader

    transformer_dir = os.path.join(_animate_model_path(), "transformer")
    args = FastVideoArgs(model_path=transformer_dir,
                         pipeline_config=WanAnimate14BConfig(dit_precision="bf16"))
    args.device = torch.device("cuda:0")
    model = TransformerLoader().load(transformer_dir, args).eval()

    video_frames, hw = 2, 16  # T=2 latent frames + ref slot; face F = 4*2-3 = 5
    device, dtype = torch.device("cuda:0"), torch.bfloat16
    inputs = dict(
        hidden_states=torch.randn(1, 36, video_frames + 1, hw, hw, device=device, dtype=dtype),
        encoder_hidden_states=torch.randn(1, 20, 4096, device=device, dtype=dtype),
        timestep=torch.tensor([500.0], device=device),
        encoder_hidden_states_image=torch.randn(1, 257, 1280, device=device, dtype=dtype),
        pose_latents=torch.randn(1, 16, video_frames, hw, hw, device=device, dtype=dtype),
        face_pixel_values=torch.randn(1, 3, 4 * video_frames - 3, 512, 512, device=device, dtype=dtype),
    )
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(**inputs)
    assert out.shape == (1, 16, video_frames + 1, hw, hw)
    assert torch.isfinite(out.float()).all()
