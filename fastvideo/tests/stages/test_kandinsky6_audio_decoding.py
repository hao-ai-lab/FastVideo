# SPDX-License-Identifier: Apache-2.0
"""Tests for Kandinsky6 audio decoding with separate audio_vae (mel-VAE decoder) and vocoder (BigVGAN-v2)
components: model_index.json declares both, and audio_vae/diffusion_pytorch_model.safetensors also carries a
bundled copy of the vocoder weights under a `vocoder.*` prefix, which the loader drops.

Pure CPU: Kandinsky6AudioVAE's "44k" mode has fixed (not configurably tiny) dimensions, so those tests
build the real-sized module (random-initialized, no real weights) -- still fast, no GPU needed. The
BigVGAN-v2 side does accept a tiny config, so the vocoder-remap tests use one.
"""
from __future__ import annotations

import types

import pytest
import torch

B, A_LEN = 1, 4  # audio latent sequence length


def _audio_vae():
    from fastvideo.models.audio.kandinsky6_audio_vae import Kandinsky6AudioVAE

    vae = Kandinsky6AudioVAE({"mode": "44k", "scaling_factor": 0.5302, "need_vae_encoder": False})
    vae.remove_weight_norm()
    return vae.eval()


def test_audio_vae_has_no_vocoder_submodule_or_state_dict_keys():
    # The checkpoint's audio_vae/*.safetensors still bundles a redundant `vocoder.*` copy for
    # backward compat (see AudioDecoderLoader), but this class itself no longer has anywhere to load
    # it -- the vocoder is now a fully separate pipeline component.
    vae = _audio_vae()
    assert not hasattr(vae, "vocoder")
    assert not hasattr(vae, "vocode")
    assert not hasattr(vae, "wrapped_decode")
    keys = set(vae.state_dict())
    assert not any(k.startswith("vocoder.") for k in keys)
    assert any(k.startswith("vae.") for k in keys)
    assert any(k.startswith("mel_converter.") for k in keys)


def test_audio_vae_decode_returns_a_mel_spectrogram():
    vae = _audio_vae()
    latents = torch.randn(B, 40, A_LEN)  # [B, embed_dim, A], matching the stage's channel-first convention
    with torch.no_grad():
        mel = vae.decode(latents)
    assert mel.shape[0] == B
    assert mel.shape[1] == 128  # num_mels
    assert torch.isfinite(mel).all()


def test_mmaudio_vocoder_config_remaps_to_bigvgan_with_the_hardcoded_fixed_fields():
    from fastvideo.models.loader.component_loader import _mmaudio_vocoder_to_bigvgan

    # A flat config shaped like the real checkpoint's vocoder/config.json (diffusers' MMAudioVocoder
    # schema): no resblock/activation/snake_logscale/use_bias_at_final/use_tanh_at_final keys.
    raw = dict(num_mels=128, upsample_initial_channel=1536, upsample_rates=[8, 4, 2, 2, 2, 2],
              upsample_kernel_sizes=[16, 8, 4, 4, 4, 4], resblock_kernel_sizes=[3, 7, 11],
              resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    class_name, merged = _mmaudio_vocoder_to_bigvgan("MMAudioVocoder", raw)
    assert class_name == "BigVGANV2"
    assert merged["resblock"] == "1"
    assert merged["activation"] == "snakebeta"
    assert merged["snake_logscale"] is True
    assert merged["use_bias_at_final"] is False
    assert merged["use_tanh_at_final"] is False
    # Without this, BigVGANV2's default weight_norm-parametrized construction leaves the fresh
    # module's state_dict keys as conv_pre.parametrizations.weight.original0/1 etc., but the
    # checkpoint's vocoder/*.safetensors is saved with weight_norm already removed (plain conv_pre.weight) --
    # VocoderLoader's load_state_dict(strict=True) then fails with "Missing key(s)".
    assert merged["weight_norm_removed"] is True
    # Original fields pass through untouched.
    assert merged["num_mels"] == 128
    assert merged["upsample_rates"] == [8, 4, 2, 2, 2, 2]


def test_mmaudio_vocoder_remap_is_a_no_op_for_other_class_names():
    from fastvideo.models.loader.component_loader import _mmaudio_vocoder_to_bigvgan

    class_name, cfg = _mmaudio_vocoder_to_bigvgan("BigVGANV2", {"num_mels": 128})
    assert class_name == "BigVGANV2"
    assert cfg == {"num_mels": 128}


def test_remapped_bigvgan_builds_with_plain_non_parametrized_keys_matching_the_real_checkpoint():
    # A freshly-built BigVGANV2 defaults to weight_norm-parametrized keys
    # (conv_pre.parametrizations.weight.original0/1), but the checkpoint's vocoder/*.safetensors has plain
    # keys (conv_pre.weight) -- VocoderLoader.load() calls load_state_dict(strict=True) BEFORE
    # remove_weight_norm(), so without weight_norm_removed=True set at construction time (asserted in
    # the test above), that strict load fails with "Missing key(s)". Reproduces VocoderLoader's exact
    # load order: construct -> load_state_dict(strict=True) -> remove_weight_norm() (idempotent).
    from fastvideo.models.loader.component_loader import _mmaudio_vocoder_to_bigvgan
    from fastvideo.models.registry import ModelRegistry

    tiny = dict(num_mels=8, upsample_initial_channel=16, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
               resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]])
    class_name, cfg = _mmaudio_vocoder_to_bigvgan("MMAudioVocoder", tiny)
    model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
    vocoder = model_cls(cfg)
    keys = set(vocoder.state_dict())
    assert not any("parametrizations" in key for key in keys), keys
    assert "conv_pre.weight" in keys

    # Simulate loading a "real checkpoint" (here, the module's own freshly-initialized weights) the
    # same way VocoderLoader does: strict load first, remove_weight_norm second.
    own_state = {k: v.clone() for k, v in vocoder.state_dict().items()}
    vocoder.load_state_dict(own_state, strict=True)
    vocoder.remove_weight_norm()  # idempotent no-op here; must not raise


def test_remapped_bigvgan_decodes_without_tanh_saturation_or_a_final_bias():
    # Exercises the two fields that would otherwise default wrong (use_bias_at_final /
    # use_tanh_at_final default to True in BigVGANV2 itself).
    from fastvideo.models.loader.component_loader import _mmaudio_vocoder_to_bigvgan
    from fastvideo.models.registry import ModelRegistry

    tiny = dict(num_mels=8, upsample_initial_channel=16, upsample_rates=[2, 2], upsample_kernel_sizes=[4, 4],
               resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1, 3]])
    class_name, cfg = _mmaudio_vocoder_to_bigvgan("MMAudioVocoder", tiny)
    model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
    vocoder = model_cls(cfg)
    assert vocoder.conv_post.bias is None
    assert vocoder.use_tanh_at_final is False
    vocoder.remove_weight_norm()

    mel = torch.randn(1, 8, 5)
    with torch.no_grad():
        waveform = vocoder(mel)
    assert waveform.shape[0] == 1 and waveform.shape[1] == 1
    assert torch.isfinite(waveform).all()


class _StubAudioVAE(torch.nn.Module):
    """Records its decode() call's input and returns a deterministic fake mel."""

    def __init__(self) -> None:
        super().__init__()
        self.scaling_factor = 2.0
        self.linear = torch.nn.Linear(1, 1)  # so next(self.parameters()) works
        self.seen_latents: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.seen_latents = latents
        batch, _, length = latents.shape
        return torch.full((batch, 8, length), 0.5)


class _StubVocoder(torch.nn.Module):
    """Records its forward() call's input (the VAE's mel output) and returns a fake waveform."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.seen_mel: torch.Tensor | None = None

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        self.seen_mel = mel
        batch, _, length = mel.shape
        return torch.full((batch, 1, length * 4), 2.0)  # out of [-1, 1] to exercise the stage's clamp


@pytest.fixture()
def cpu_device(monkeypatch):
    import fastvideo.pipelines.stages.kandinsky6 as k6_stages
    cpu = torch.device("cpu")
    monkeypatch.setattr(k6_stages, "get_local_torch_device", lambda: cpu)
    return cpu


def test_audio_decoding_stage_calls_vae_decode_then_vocoder_in_sequence(cpu_device):
    from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6AudioDecodingStage

    audio_vae = _StubAudioVAE()
    vocoder = _StubVocoder()
    stage = Kandinsky6AudioDecodingStage(audio_vae=audio_vae, vocoder=vocoder)

    audio_latents = torch.ones(1, A_LEN, 40) * 4.0  # [B, A, D]; scaling_factor=2.0 -> latents=2.0
    batch = types.SimpleNamespace(audio_latents=audio_latents, extra={})
    fastvideo_args = types.SimpleNamespace(vae_cpu_offload=False,
                                           pipeline_config=types.SimpleNamespace())

    out = stage.forward(batch, fastvideo_args)

    # audio_vae.decode saw latents already divided by scaling_factor (no + mean_value term: the new
    # diffusers reference's postprocess_audio dropped that term) and transposed to [B, D, A].
    assert audio_vae.seen_latents is not None
    assert audio_vae.seen_latents.shape == (1, 40, A_LEN)
    torch.testing.assert_close(audio_vae.seen_latents, torch.full((1, 40, A_LEN), 2.0))

    # vocoder saw exactly the VAE's mel output (the two-step decode -> vocode chain).
    assert vocoder.seen_mel is not None
    torch.testing.assert_close(vocoder.seen_mel, torch.full((1, 8, A_LEN), 0.5))

    # Final waveform is clamped to [-1, 1] even though the stub vocoder returned 2.0.
    assert out.extra["audio"].max().item() == pytest.approx(1.0)
    assert out.extra["audio"].shape == (A_LEN * 4,)


def test_audio_decoding_stage_is_a_noop_without_audio_latents(cpu_device):
    from fastvideo.pipelines.stages.kandinsky6 import Kandinsky6AudioDecodingStage

    audio_vae, vocoder = _StubAudioVAE(), _StubVocoder()
    stage = Kandinsky6AudioDecodingStage(audio_vae=audio_vae, vocoder=vocoder)
    batch = types.SimpleNamespace(audio_latents=None, extra={})
    out = stage.forward(batch, types.SimpleNamespace(vae_cpu_offload=False))
    assert out.extra == {}
    assert audio_vae.seen_latents is None
    assert vocoder.seen_mel is None
