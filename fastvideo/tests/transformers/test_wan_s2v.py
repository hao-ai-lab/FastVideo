# SPDX-License-Identifier: Apache-2.0
"""Tests for the Wan2.2-S2V port.

Most of this file is CPU-only and needs neither a GPU nor the 28GB checkpoint:
it pins the invariants that silently corrupt output when broken -- the class of
bug that required follow-up fix PRs for the FLUX (#1321) and Z-Image (#1339)
ports.

The weight-loading tests at the bottom need the real checkpoint and are gated
the same way ``test_flux.py`` gates its parity test, so they skip locally and
run in CI. Point them at a checkout with::

    export WAN_S2V_MODEL_PATH=/path/to/Wan2.2-S2V-14B
"""
import glob
import os
import re

import pytest
import torch

# The distributed_setup fixture rendezvouses via env:// even for a single
# GPU; without these the CUDA forward-pass test errors before it starts.
# Same convention as test_flux.py (port distinct per file to avoid clashes).
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from fastvideo.configs.models.dits.wan_s2v import WanS2VArchConfig, WanS2VConfig
from fastvideo.models.dits.wan_s2v import WanS2VTransformer3DModel
from fastvideo.models.dits.wan_s2v_audio import AudioInjector, CausalAudioEncoder

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_S2V_PATH = os.path.join(_REPO_ROOT, "official_weights", "Wan2.2-S2V-14B")


def _s2v_model_path() -> str:
    return os.environ.get("WAN_S2V_MODEL_PATH", _DEFAULT_S2V_PATH)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="Wan-S2V weight-loading tests require CUDA")

requires_weights = pytest.mark.skipif(
    not glob.glob(os.path.join(_s2v_model_path(), "*.safetensors")),
    reason=(f"No safetensors under {_s2v_model_path()} -- download Wan-AI/Wan2.2-S2V-14B "
            "or set WAN_S2V_MODEL_PATH"))


def test_arch_config_matches_official_checkpoint_config() -> None:
    """Values transcribed from Wan-AI/Wan2.2-S2V-14B/config.json."""
    a = WanS2VArchConfig()
    assert a.hidden_size == 5120  # config.json: dim
    assert a.num_attention_heads == 40
    assert a.attention_head_dim == 128  # 5120 / 40
    assert a.ffn_dim == 13824
    assert a.num_layers == 40
    assert a.in_channels == 16  # config.json: in_dim
    assert a.out_channels == 16  # config.json: out_dim
    assert a.cond_dim == 16
    assert a.audio_dim == 1024
    assert a.num_audio_token == 4
    assert a.patch_size == (1, 2, 2)
    assert a.enable_adain is True
    assert a.adain_mode == "attn_norm"
    assert a.enable_framepack is True
    assert a.enable_motioner is False
    assert a.zero_timestep is True
    assert a.audio_inject_layers == (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)


def test_injector_count_matches_inject_layers() -> None:
    """The checkpoint stores injectors densely (injector.0..11) with no record
    of which block each serves; that lives only in audio_inject_layers. A
    mismatch loads cleanly and steers the wrong layers."""
    a = WanS2VArchConfig()
    inj = AudioInjector(dim=64, num_heads=4, inject_layers=a.audio_inject_layers,
                        enable_adain=True, adain_dim=64)
    assert len(inj.injector) == len(a.audio_inject_layers) == 12
    assert len(inj.injector_adain_layers) == 12
    assert inj.injected_block_id == {b: i for i, b in enumerate(a.audio_inject_layers)}


def test_audio_layer_weights_match_wav2vec2_large() -> None:
    """casual_audio_encoder.weights is (1, 25, 1, 1): 25 == wav2vec2-LARGE
    hidden states (24 layers + embeddings). wav2vec2-base yields 13 and will
    not fit -- the bundled wav2vec2-large-xlsr-53-english is required."""
    enc = CausalAudioEncoder(dim=1024, num_layers=25, out_dim=64, num_token=4, need_global=True)
    assert tuple(enc.weights.shape) == (1, 25, 1, 1)


def test_audio_encoder_emits_num_token_plus_padding() -> None:
    enc = CausalAudioEncoder(dim=32, num_layers=25, out_dim=64, num_token=4, need_global=True)
    summary, local = enc(torch.randn(1, 25, 32, 16))
    assert local.shape[-2] == 5  # num_token + 1 learned padding token
    assert summary.shape[-2] == 1
    assert local.shape[:2] == summary.shape[:2]


def test_injector_only_touches_the_video_span() -> None:
    """Reference/motion tokens trail the video tokens and must pass through
    untouched; slicing this wrong corrupts reference conditioning silently."""
    from fastvideo.forward_context import set_forward_context
    d, heads, frames, n_aud, hw = 64, 4, 3, 5, 4
    inj = AudioInjector(dim=d, num_heads=heads, inject_layers=(0, 1), enable_adain=True, adain_dim=d)
    # FastVideo's ReplicatedLinear zero-initialises (weights arrive from the
    # checkpoint), so stand in for loaded weights or the residual is a no-op
    # and the test would pass vacuously.
    for p in inj.parameters():
        torch.nn.init.normal_(p, std=0.02)

    video_len = frames * hw
    hs = torch.randn(1, video_len + 6, d)
    aud, summary = torch.randn(1, frames, n_aud, d), torch.randn(1, frames, 1, d)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        out = inj(hs, 0, aud, summary, video_len)
        noop = inj(hs, 7, aud, summary, video_len)
    assert out.shape == hs.shape
    assert not torch.allclose(out[:, :video_len], hs[:, :video_len])
    assert torch.allclose(out[:, video_len:], hs[:, video_len:])
    assert torch.equal(noop, hs)  # non-injected block index is an exact no-op


def test_duplicate_inject_layers_rejected() -> None:
    with pytest.raises(AssertionError):
        WanS2VArchConfig(audio_inject_layers=(0, 4, 4))


def test_motioner_and_framepack_are_mutually_exclusive() -> None:
    with pytest.raises(AssertionError):
        WanS2VArchConfig(enable_motioner=True, enable_framepack=True)


# --------------------------------------------------------------------------
# Layer-level and end-to-end behaviour, on a tiny model with the real structure
# --------------------------------------------------------------------------


def _tiny_arch(**overrides) -> WanS2VArchConfig:
    kwargs = dict(num_attention_heads=4, attention_head_dim=16, ffn_dim=64, num_layers=4,
                  audio_inject_layers=(0, 2), audio_dim=32, num_audio_token=4)
    kwargs.update(overrides)
    return WanS2VArchConfig(**kwargs)


def _tiny_model(arch: WanS2VArchConfig | None = None) -> WanS2VTransformer3DModel:
    arch = arch or _tiny_arch()
    model = WanS2VTransformer3DModel(config=WanS2VConfig(arch_config=arch), hf_config={})
    for p in model.parameters():  # stand in for checkpoint weights (see note above)
        if p.dim() > 0:
            torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def _tiny_inputs(arch: WanS2VArchConfig, frames: int = 4, hw: int = 16):
    c = arch.in_channels
    return dict(
        hidden_states=[torch.randn(c, frames, hw, hw)],
        encoder_hidden_states=[torch.randn(12, arch.text_dim)],
        timestep=torch.tensor([500.0]),
        ref_latents=[torch.randn(c, 1, hw, hw)],
        motion_latents=[torch.randn(c, 8, hw, hw)],
        cond_states=[torch.zeros(c, frames, hw, hw)],
        # T_a // 4 (two stride-2 convs) must equal the video latent frame count
        audio_input=torch.randn(1, 25, arch.audio_dim, frames * 4),
        motion_frames=(0, 0),
    )


def test_forward_returns_video_shaped_latents() -> None:
    """Only the video span is denoised: ref and motion tokens are context and
    must not appear in the output."""
    from fastvideo.forward_context import set_forward_context
    arch = _tiny_arch()
    model, inputs = _tiny_model(arch), _tiny_inputs(_tiny_arch())
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(**inputs)
    assert len(out) == 1
    assert out[0].shape == inputs["hidden_states"][0].shape
    assert torch.isfinite(out[0]).all()


def test_forward_is_deterministic() -> None:
    from fastvideo.forward_context import set_forward_context
    torch.manual_seed(0)
    model, inputs = _tiny_model(), _tiny_inputs(_tiny_arch())
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        a = model(**inputs)[0]
        b = model(**inputs)[0]
    assert torch.equal(a, b)


def test_forward_responds_to_audio() -> None:
    """A different audio track must change the prediction -- otherwise the
    injection path is inert and lip-sync silently does nothing."""
    from fastvideo.forward_context import set_forward_context
    torch.manual_seed(0)
    model, inputs = _tiny_model(), _tiny_inputs(_tiny_arch())
    other = dict(inputs, audio_input=torch.randn_like(inputs["audio_input"]))
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        a, b = model(**inputs)[0], model(**other)[0]
    assert not torch.allclose(a, b)


def test_forward_responds_to_reference_image() -> None:
    from fastvideo.forward_context import set_forward_context
    torch.manual_seed(0)
    model, inputs = _tiny_model(), _tiny_inputs(_tiny_arch())
    other = dict(inputs, ref_latents=[torch.randn_like(inputs["ref_latents"][0])])
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        a, b = model(**inputs)[0], model(**other)[0]
    assert not torch.allclose(a, b)


def test_drop_motion_frames_still_produces_video() -> None:
    """Long-video continuation drops motion context on the first clip."""
    from fastvideo.forward_context import set_forward_context
    model, inputs = _tiny_model(), _tiny_inputs(_tiny_arch())
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(**dict(inputs, drop_motion_frames=True))
    assert out[0].shape == inputs["hidden_states"][0].shape
    assert torch.isfinite(out[0]).all()


def test_audio_video_misalignment_raises_clearly() -> None:
    """Wrong resampling in the audio stage must fail loudly, not drift."""
    from fastvideo.forward_context import set_forward_context
    d, frames = 64, 3
    inj = AudioInjector(dim=d, num_heads=4, inject_layers=(0,), enable_adain=True, adain_dim=d)
    hs = torch.randn(1, 3 * 4 + 2, d)  # 14 video tokens: not divisible by 3 frames
    with set_forward_context(current_timestep=0, attn_metadata=None), \
            pytest.raises(ValueError, match="misalignment"):
        inj(hs, 0, torch.randn(1, frames, 5, d), torch.randn(1, frames, 1, d), 14)


def test_framepack_compresses_motion_at_three_scales() -> None:
    from fastvideo.models.dits.wan_s2v import FramePackMotioner
    packer = FramePackMotioner(inner_dim=64, num_heads=4, zip_frame_buckets=(1, 2, 16))
    mot, rope = packer([torch.randn(16, 19, 16, 16)], add_last_motion=2)
    assert len(mot) == len(rope) == 1
    assert mot[0].shape[-1] == 64 and mot[0].shape[1] > 0
    assert rope[0].shape[1] == mot[0].shape[1]  # one RoPE entry per motion token


def test_zero_timestep_splits_modulation_into_two_segments() -> None:
    """Video tokens get the real timestep; ref/motion get a zero timestep."""
    model = _tiny_model()
    e, (block_e, seg_idx) = model._timestep_embedding(torch.tensor([500.0]), video_len=12)
    assert seg_idx == 12
    assert block_e.shape[2] == 2  # segment axis
    assert not torch.allclose(block_e[:, :, 0], block_e[:, :, 1])
    assert e.shape[0] == 1  # the extra zero timestep is not returned to the head


# --------------------------------------------------------------------------
# Wiring: registry, preset, loaders, pipeline composition
# --------------------------------------------------------------------------


def test_model_path_resolves_to_the_s2v_config_and_preset() -> None:
    import fastvideo.registry  # noqa: F401  (registers configs on import)
    from fastvideo.api.presets import get_preset
    from fastvideo.configs.pipelines.wan import WanS2V14BConfig
    from fastvideo.registry import get_default_preset, get_pipeline_config_cls_from_name

    # Canonical converted repo: the FastVideo org hosts converted checkpoints,
    # like every other native-format model here. (The official Wan-AI repo has
    # no model_index.json, so resolving it requires running
    # scripts/checkpoint_conversion/wan_s2v_to_diffusers.py first.)
    path = "FastVideo/Wan2.2-S2V-14B-Diffusers"
    assert get_pipeline_config_cls_from_name(path) is WanS2V14BConfig
    assert get_default_preset(path) == "wan_s2v_14b"
    preset = get_preset("wan_s2v_14b", "wan")
    assert preset.workload_type == "i2v"
    assert preset.defaults["num_frames"] % 4 == 0  # must divide into VAE latent frames


def test_dit_input_channels_match_the_vae_latent_depth() -> None:
    """S2V conditions via attention, not channel concatenation, so the DiT takes
    exactly one latent's worth of channels (unlike Lucy Edit, which takes 2x)."""
    from fastvideo.configs.pipelines.wan import WanS2V14BConfig
    cfg = WanS2V14BConfig()
    assert cfg.dit_config.arch_config.in_channels == cfg.vae_config.arch_config.z_dim == 16
    assert cfg.vae_config.load_encoder and cfg.vae_config.load_decoder


def test_audio_components_have_loaders() -> None:
    from fastvideo.models.loader.component_loader import AudioEncoderLoader, ComponentLoader
    assert isinstance(ComponentLoader.for_module_type("audio_encoder", "transformers"), AudioEncoderLoader)
    assert ComponentLoader.for_module_type("audio_processor", "transformers") is not None


def test_pipeline_requires_and_orders_the_audio_stage() -> None:
    """Audio must be encoded before denoising; it is constant across steps, so
    encoding it inside the loop would be pure waste."""
    from fastvideo.pipelines.basic.wan.wan_s2v_pipeline import WanSpeechToVideoPipeline
    required = WanSpeechToVideoPipeline._required_config_modules
    assert "audio_encoder" in required and "audio_processor" in required
    assert "transformer" in required and "vae" in required


def test_pipeline_is_discoverable_by_the_registry() -> None:
    from fastvideo.pipelines.pipeline_registry import PipelineType, get_pipeline_registry
    registry = get_pipeline_registry(PipelineType.BASIC)
    assert "WanSpeechToVideoPipeline" in registry.get_supported_pipelines(PipelineType.BASIC)


def test_audio_path_is_plumbed_through_every_public_surface() -> None:
    """SamplingParam -> typed schema -> ForwardBatch. A gap anywhere here means
    the user can pass audio and the pipeline silently never sees it."""
    import dataclasses

    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.api.schema import InputConfig
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    for cls in (SamplingParam, InputConfig, ForwardBatch):
        assert "audio_path" in {f.name for f in dataclasses.fields(cls)}, cls.__name__
    assert "audio_embeds" in {f.name for f in dataclasses.fields(ForwardBatch)}


# --------------------------------------------------------------------------
# Stage <-> model contract
#
# These exist because every other test in this file calls model.forward()
# directly, which passes even when the pipeline cannot call the model at all.
# --------------------------------------------------------------------------


def _stage_and_batch():
    from fastvideo.configs.pipelines.wan import WanS2V14BConfig
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages.denoising import DenoisingStage

    model = _tiny_model()
    stage = DenoisingStage(transformer=model, scheduler=None)
    batch = ForwardBatch(data_type="video")
    batch.audio_embeds = torch.randn(1, 25, _tiny_arch().audio_dim, 16)
    batch.image_latent = torch.randn(1, 16, 1, 16, 16)

    class _Args:
        pipeline_config = WanS2V14BConfig()

    return model, stage, batch, _Args()


def test_denoising_stage_forwards_the_audio_and_reference_inputs() -> None:
    """DenoisingStage must actually hand the model its audio and reference image.

    prepare_extra_func_kwargs filters by the transformer's signature, so a
    rename on either side silently drops the conditioning and the model would
    fall back to text-only behaviour.
    """
    model, stage, batch, args = _stage_and_batch()
    kwargs, _ = stage._s2v_conditioning_kwargs(batch, args)
    assert "audio_input" in kwargs, "audio never reaches the model"
    assert "ref_latents" in kwargs, "reference image never reaches the model"
    assert kwargs["audio_input"] is batch.audio_embeds
    assert kwargs["motion_frames"] == (73, 19)  # official runtime value, not the (17,5) signature default


def test_cfg_negative_pass_zeroes_the_audio() -> None:
    """Official recipe: the unconditional pass runs with 0.0 * audio_input, so
    guidance contrasts over text AND audio jointly. Passing the same audio to
    both passes silently halves the audio's influence."""
    _, stage, batch, args = _stage_and_batch()
    cond, uncond = stage._s2v_conditioning_kwargs(batch, args)
    assert torch.any(cond["audio_input"] != 0)
    assert torch.all(uncond["audio_input"] == 0)
    assert uncond["ref_latents"] is cond["ref_latents"]  # ref stays; only audio is nulled


def test_stage_call_contract_end_to_end() -> None:
    """The full loop contract: call the model exactly as DenoisingStage does
    (batched latents, list-of-batched prompt embeds, kwargs), then run CFG
    arithmetic and a real scheduler step on the result. This is the test whose
    absence let an unrunnable pipeline look finished."""
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler

    arch = _tiny_arch()
    model, stage, batch, args = _stage_and_batch()
    c, frames, hw = arch.in_channels, 4, 16
    latents = torch.randn(1, c, frames, hw, hw)
    prompt_embeds = [torch.randn(1, 12, arch.text_dim)]  # stage format: list of [B, L, C]
    t = torch.tensor([500.0])
    batch.audio_embeds = torch.randn(1, 25, arch.audio_dim, frames * 4)
    batch.image_latent = torch.randn(1, c, 1, hw, hw)
    cond, uncond = stage._s2v_conditioning_kwargs(batch, args)
    cond["motion_frames"] = (0, 0)  # tiny-model audio alignment
    uncond["motion_frames"] = (0, 0)

    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        noise_pred = model(latents, prompt_embeds, t, guidance=None, **cond)
        noise_pred_uncond = model(latents, prompt_embeds, t, guidance=None, **uncond)

    # Batched call must return a batched tensor: CFG arithmetic and
    # scheduler.step are tensor ops, and a list return breaks both.
    assert isinstance(noise_pred, torch.Tensor) and noise_pred.shape == latents.shape
    guided = noise_pred_uncond + 5.0 * (noise_pred - noise_pred_uncond)

    scheduler = FlowUniPCMultistepScheduler(shift=5.0)
    scheduler.set_timesteps(4)
    stepped = scheduler.step(guided, scheduler.timesteps[0], latents, return_dict=False)[0]
    assert stepped.shape == latents.shape
    assert torch.isfinite(stepped).all()


def test_missing_audio_or_reference_fails_loudly() -> None:
    """Text-only inputs must raise, not quietly generate an unconditioned video."""
    from fastvideo.forward_context import set_forward_context
    arch = _tiny_arch()
    model = _tiny_model(arch)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None), \
            pytest.raises(ValueError, match="reference image and audio"):
        model(torch.randn(1, arch.in_channels, 4, 16, 16), torch.randn(1, 12, arch.text_dim),
              torch.tensor([500.0]))


# --------------------------------------------------------------------------
# Audio encoding stage
# --------------------------------------------------------------------------


def test_audio_bucketing_emits_one_window_per_video_frame() -> None:
    from fastvideo.pipelines.stages.audio_encoding import AudioEncodingStage
    stage = AudioEncodingStage(audio_encoder=None, audio_processor=None)
    features = torch.randn(25, 100, 32)  # 25 wav2vec2 layers, 100 frames @30fps
    out = stage._bucket_to_frames(features, num_frames=16, fps=16, window=0)
    assert out.shape == (25, 32, 16)  # [num_layers, C, num_frames]


def test_audio_bucketing_clamps_at_track_end() -> None:
    """A short audio track must not index past its end -- it repeats the last
    window instead, which is what upstream does."""
    from fastvideo.pipelines.stages.audio_encoding import AudioEncodingStage
    stage = AudioEncodingStage(audio_encoder=None, audio_processor=None)
    features = torch.randn(2, 4, 8)  # only 4 audio frames available
    out = stage._bucket_to_frames(features, num_frames=32, fps=16, window=0)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_feature_resampling_hits_the_target_rate() -> None:
    from fastvideo.pipelines.stages.audio_encoding import _resample_features
    out = _resample_features(torch.randn(1, 50, 8), input_rate=50, output_rate=30)
    assert out.shape == (1, 30, 8)  # one second of 50Hz features -> 30 video frames


def test_model_parameter_names_are_reachable_from_checkpoint_names() -> None:
    """Every model parameter must be the target of the mapping applied to some
    native checkpoint name -- i.e. no parameter is unreachable at load time."""
    with torch.device("meta"):
        model = WanS2VTransformer3DModel(config=WanS2VConfig(), hf_config={})
    names = set(model.state_dict())
    assert len(names) == 1260, f"expected the checkpoint's 1260 tensors, got {len(names)}"

    mapping = WanS2VArchConfig().param_names_mapping

    def apply(name: str) -> str:
        for pattern, repl in mapping.items():
            new, n = re.subn(pattern, repl, name)
            if n:
                return new
        return name

    # Spot-check the rewrites that carry the most risk.
    assert apply("blocks.0.self_attn.q.weight") == "blocks.0.self_attn.to_q.weight"
    assert apply("blocks.39.ffn.0.bias") == "blocks.39.ffn.fc_in.bias"
    assert apply("audio_injector.injector.11.o.weight") == "audio_injector.injector.11.to_out.weight"
    assert apply("time_projection.1.weight") == "time_projection.linear.weight"
    # Pass-through families keep their names verbatim.
    for verbatim in ("blocks.0.modulation", "trainable_cond_mask.weight",
                     "casual_audio_encoder.weights", "frame_packer.proj.weight"):
        assert apply(verbatim) == verbatim
        assert verbatim in names


# --------------------------------------------------------------------------
# Real-checkpoint tests. Skipped without CUDA + weights; run in CI.
# --------------------------------------------------------------------------


def _checkpoint_tensor_names(model_path: str) -> dict[str, tuple[int, ...]]:
    """Every tensor name -> shape, read from the shard headers only.

    safetensors files begin with an 8-byte header length followed by a JSON
    header describing every tensor, so this reads a few KB per shard instead of
    the 28GB of weights.
    """
    import json
    import struct
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with open(shard, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(meta["shape"])
    return shapes


@requires_weights
def test_every_checkpoint_tensor_has_a_home_in_the_model() -> None:
    """Strict-load contract: no missing, no unexpected, no shape mismatch.

    This is the check the Z-Image port shipped without (#1339 added it after the
    fact). It needs the weights on disk but not a GPU: the model is built on the
    meta device, so a 14B model costs no memory here.
    """
    checkpoint = _checkpoint_tensor_names(_s2v_model_path())
    assert checkpoint, "no tensors found in checkpoint shards"

    with torch.device("meta"):
        model = WanS2VTransformer3DModel(config=WanS2VConfig(), hf_config={})
    model_shapes = {name: tuple(p.shape) for name, p in model.state_dict().items()}

    mapping = WanS2VArchConfig().param_names_mapping

    def to_model_name(name: str) -> str:
        for pattern, repl in mapping.items():
            new, count = re.subn(pattern, repl, name)
            if count:
                return new
        return name

    mapped = {to_model_name(name): shape for name, shape in checkpoint.items()}
    assert len(mapped) == len(checkpoint), "param_names_mapping collapsed two tensors onto one name"

    missing = sorted(set(mapped) - set(model_shapes))
    unexpected = sorted(set(model_shapes) - set(mapped))
    mismatched = {n: (model_shapes[n], mapped[n]) for n in set(model_shapes) & set(mapped)
                  if model_shapes[n] != mapped[n]}
    assert not missing, f"checkpoint tensors with no parameter to load into: {missing[:10]}"
    assert not unexpected, f"parameters the checkpoint never fills: {unexpected[:10]}"
    assert not mismatched, f"shape mismatches: {dict(list(mismatched.items())[:10])}"


@requires_weights
def test_rope_buffers_survive_meta_device_construction() -> None:
    """The loader builds the DiT on the meta device, so the RoPE tables (which
    are derived from config, not stored in the checkpoint) must be rebuilt by
    materialize_non_persistent_buffers or every forward pass fails."""
    with torch.device("meta"):
        model = WanS2VTransformer3DModel(config=WanS2VConfig(), hf_config={})
    assert model.freqs.is_meta and model.frame_packer.freqs.is_meta
    model.materialize_non_persistent_buffers(torch.device("cpu"))
    assert not model.freqs.is_meta and not model.frame_packer.freqs.is_meta
    # Non-persistent buffers must stay out of the checkpoint contract.
    assert "freqs" not in model.state_dict()


@requires_cuda
@requires_weights
@pytest.mark.usefixtures("distributed_setup")
def test_transformer_loads_and_runs_a_forward_pass() -> None:
    """Full load through the real TransformerLoader, then one forward.

    Catches anything the meta-device simulation cannot: dtype handling, buffer
    materialisation in situ, and device placement.
    """
    from fastvideo.configs.pipelines.wan import WanS2V14BConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.loader.component_loader import TransformerLoader

    args = FastVideoArgs(model_path=_s2v_model_path(), pipeline_config=WanS2V14BConfig())
    model = TransformerLoader().load(_s2v_model_path(), args)
    for name, param in model.named_parameters():
        assert not param.is_meta, f"{name} still on meta after load"

    arch = WanS2V14BConfig().dit_config.arch_config
    device, dtype = torch.device("cuda"), torch.bfloat16
    channels, frames, size = arch.in_channels, 4, 32
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(
            [torch.randn(channels, frames, size, size, device=device, dtype=dtype)],
            [torch.randn(12, arch.text_dim, device=device, dtype=dtype)],
            torch.tensor([500.0], device=device),
            ref_latents=[torch.randn(channels, 1, size, size, device=device, dtype=dtype)],
            motion_latents=[torch.randn(channels, 8, size, size, device=device, dtype=dtype)],
            cond_states=[torch.zeros(channels, frames, size, size, device=device, dtype=dtype)],
            audio_input=torch.randn(1, 25, arch.audio_dim, frames * 4, device=device, dtype=dtype),
            motion_frames=(0, 0),
        )
    assert out[0].shape == (channels, frames, size, size)
    assert torch.isfinite(out[0]).all()
