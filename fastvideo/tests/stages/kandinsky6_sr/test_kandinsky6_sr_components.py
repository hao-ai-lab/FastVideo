# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR component loading: KVAE wrapper, latent-upscaler bank config, strict loaders, dtype policy (CPU)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import k6_sr_tiny
import pytest
import torch
from safetensors.torch import load_file, save_file

from fastvideo.configs.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerConfig
from fastvideo.models.loader.component_loader import PipelineComponentLoader


# --------------------------------------------------------------------------- KVAE wrapper


def test_vae_wrapper_contract():
    vae = k6_sr_tiny.build_vae()
    assert vae.scaling_factor == k6_sr_tiny.TINY_SCALING_FACTOR and vae.config.scaling_factor == 0.5
    assert vae.spatial_factor == 16 and vae.temporal_factor == 4
    assert all(key.startswith(("encoder.", "decoder.")) for key in vae.state_dict())  # bundle keys load unchanged
    pixels = torch.arange(0, 256, dtype=torch.float32)
    torch.testing.assert_close(vae.denormalize_data(vae.normalize_data(pixels)), pixels)
    assert vae.normalize_data(torch.tensor(128.0)).item() == 0.0  # x / 128 - 1, the KVAE convention (not / 127.5)


def test_vae_encode_decode_geometry():
    vae = k6_sr_tiny.build_vae()
    video = torch.rand(1, 3, 9, 32, 48) * 2 - 1
    with torch.no_grad():
        latent, split_list = vae.encode(video)
        decoded = vae.decode(latent).sample
    assert split_list == [9]
    assert latent.shape == (1, 4, (9 - 1) // 4 + 1, 2, 3)  # temporal x4 (causal), spatial x16
    assert decoded.shape == video.shape


def test_vae_without_architecture_fails_loudly():
    from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEConfig
    from fastvideo.models.vaes.kandinsky6_sr import Kandinsky6SRVAE

    with pytest.raises(ValueError, match="encoder_config"):
        Kandinsky6SRVAE(Kandinsky6SRVAEConfig())
    config = Kandinsky6SRVAEConfig()
    with pytest.raises(ValueError, match="video-kvae"):
        config.update_model_arch({"vae_type": "other-vae"})


# --------------------------------------------------------------------------- LU bank


def test_lu_bank_state_keys_and_dispatch():
    bank = k6_sr_tiny.build_lu()
    assert bank.scales == (2, 4)
    assert all(key.startswith("_models.") for key in bank.state_dict())
    z = torch.randn(1, 4, 5, 4, 6)
    with torch.no_grad():
        assert bank(z * 0.5, 2).shape == (1, 4, 5, 8, 12)
        assert bank(z * 0.5, 4).shape == (1, 4, 5, 16, 24)
        with pytest.raises(ValueError, match="no x3 entry"):
            bank(z, 3)
    only_x4 = k6_sr_tiny.build_lu(scales=("4x", ))
    assert only_x4.scales == (4, )
    with pytest.raises(ValueError, match="no x2 entry"):
        only_x4(z, 2)


def _lu_config(**model_overrides):
    config = k6_sr_tiny.lu_config_dict(("4x", ))
    config["models"][0]["model"].update(model_overrides)
    return config


@pytest.mark.parametrize("config,match", [
    ({"models": [{"target_scale": "2x"}]}, "models\\[0\\]"),
    ({"models": [{"target_scale": "8x", "model": k6_sr_tiny.TINY_LU_MODEL}]}, "target_scale"),
    (_lu_config(bogus=1), "bogus"),
    (_lu_config(architecture="flat"), "architecture"),
    (_lu_config(input_skip=True), "input_skip"),
    (_lu_config(upsample_mode="pixel_shuffle"), "upsample_mode"),
    (_lu_config(motion_attention={"num_heads": 2}), "motion_attention"),
    (_lu_config(x2_tail_mode="shared"), "x2_tail_mode"),
    (_lu_config(enable_x2_entry=False), "enable_x2_entry"),
    (_lu_config(hidden_channels=16), "stage_channels"),
])
def test_lu_config_errors_name_the_offending_entry_or_key(config, match):
    with pytest.raises(ValueError, match=match):
        Kandinsky6SRLatentUpscalerConfig(**config)


def test_lu_config_omitted_key_means_the_training_default():
    config = _lu_config()
    del config["models"][0]["model"]["input_skip"]  # training default: input_skip=True, not implemented
    with pytest.raises(ValueError, match="input_skip.*omitted"):
        Kandinsky6SRLatentUpscalerConfig(**config)


def test_lu_bank_duplicate_scales_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        Kandinsky6SRLatentUpscalerConfig(**k6_sr_tiny.lu_config_dict(("2x", "2x")))


def test_lu_config_survives_the_pipeline_config_json_round_trip():
    import dataclasses

    config = Kandinsky6SRLatentUpscalerConfig(**k6_sr_tiny.lu_config_dict())
    restored = Kandinsky6SRLatentUpscalerConfig()
    dumped = json.loads(json.dumps(dataclasses.asdict(config)))
    dumped.pop("arch_config")
    restored.update_model_config(dumped)
    assert restored.models == config.models and restored.scaling_factor == config.scaling_factor


# --------------------------------------------------------------------------- loaders


def _load(name, root, fv_args, subdir):
    # Official component configs (like the Hub repo's) carry no ``_class_name``; ``ComposedPipelineBase.load_modules``
    # hands the model_index classes to the loaders through ``fastvideo_args._model_index_class_names``, mirrored here.
    index = json.loads((root / "model_index.json").read_text())
    fv_args._model_index_class_names = {key: spec[1] for key, spec in index.items() if isinstance(spec, list)}
    return PipelineComponentLoader.load_module(name, str(root / subdir), "diffusers", fv_args)


def test_loaders_build_the_components_with_the_intended_dtypes(distilled_bundle, fv_args):
    fv_args.pipeline_config.dit_precision = "bf16"
    fv_args.pipeline_config.vae_precision = "bf16"
    fv_args.pipeline_config.upsampler_precision = "bf16"
    dit = _load("transformer", distilled_bundle, fv_args, "transformer")
    vae = _load("vae", distilled_bundle, fv_args, "vae")
    lu = _load("latent_upscaler", distilled_bundle, fv_args, "latent_upscaler")

    assert dit.compute_dtype == torch.bfloat16 and next(vae.parameters()).dtype == torch.bfloat16
    assert next(lu.parameters()).dtype == torch.bfloat16 and not lu.training
    assert all(not p.requires_grad for p in lu.parameters())
    params = dict(dit.named_parameters())
    # fp32 where the reference computes in fp32 (autocast(float32) regions / .float() norms), bf16 for the linears
    assert params["pooled_bias"].dtype == torch.float32
    assert params["time_embeddings.in_layer.weight"].dtype == torch.float32
    assert params["visual_transformer_blocks.0.visual_modulation.out_layer.weight"].dtype == torch.float32
    assert params["visual_transformer_blocks.0.self_attention.query_norm.weight"].dtype == torch.float32
    assert params["visual_transformer_blocks.0.self_attention.to_query.weight"].dtype == torch.bfloat16
    assert dit.out_layer.out_layer.weight.shape[0] == 4 * 3 * 4  # C * n_grid * prod(patch)
    assert dit.config.arch_config.sr_params == k6_sr_tiny.TINY_SR_PARAMS
    assert all(not p.requires_grad for p in dit.parameters())
    assert vae.scaling_factor == 0.5 and lu.scales == (2, 4)


@pytest.mark.parametrize("bundle, class_name", [("distilled_bundle", "PiflowScheduler"),
                                                ("flow_matching_bundle", "FlowMatchEulerDiscreteScheduler")])
def test_each_bundle_loads_its_own_scheduler(request, cpu_args, bundle, class_name):
    root = request.getfixturevalue(bundle)
    scheduler = _load("scheduler", root, cpu_args(root), "scheduler")
    assert type(scheduler).__name__ == class_name


def test_latent_upscaler_may_be_declared_with_the_kandinsky6_library():
    # the official save_pretrained writes ["kandinsky6", ...], the Hub repo ["diffusers", ...]
    from fastvideo.models.loader.component_loader import ComponentLoader

    for library in ("diffusers", "kandinsky6"):
        assert type(ComponentLoader.for_module_type("latent_upscaler", library)).__name__ == "UpsamplerLoader"
    with pytest.raises(AssertionError, match="latent_upscaler must be loaded from"):
        ComponentLoader.for_module_type("latent_upscaler", "transformers")


def test_bf16_dit_stays_close_to_fp32(distilled_bundle, fv_args):
    # dtype boundary: the fp32 residual stream / bf16 linear inputs must not drift (a bf16 residual would).
    fp32 = _load("transformer", distilled_bundle, fv_args, "transformer")
    fv_args.pipeline_config.dit_precision = "bf16"
    bf16 = _load("transformer", distilled_bundle, fv_args, "transformer")
    x = torch.randn(2, 3, 4, 6, 2 * 4 + 1, generator=torch.Generator().manual_seed(0))
    t = torch.tensor([300.0, 800.0])
    pos = [torch.arange(3), torch.arange(2), torch.arange(3)]
    with torch.no_grad():
        ref = fp32(x, t, pos)
        out = bf16(x.to(torch.bfloat16), t, pos)
    assert torch.isfinite(out).all()
    assert (out.float() - ref).abs().max() < 0.05 * ref.abs().max()


def _corrupt(bundle: Path, tmp_path: Path, component: str, mutate) -> Path:
    copy_root = tmp_path / f"corrupt_{component}"
    shutil.copytree(bundle, copy_root)
    path = copy_root / component / "diffusion_pytorch_model.safetensors"
    tensors = load_file(str(path))
    mutate(tensors)
    save_file(tensors, str(path))
    return copy_root


def _drop_first(fragment):
    def mutate(tensors):
        tensors.pop(next(key for key in tensors if fragment in key))

    return mutate


def _add_unexpected(tensors):
    tensors["model.unexpected_extra_weight"] = torch.zeros(3)


@pytest.mark.parametrize("component,name,mutate", [
    ("transformer", "transformer", _drop_first("visual_transformer_blocks.0.self_attention.to_query.weight")),
    ("transformer", "transformer", _add_unexpected),
    ("vae", "vae", _drop_first("decoder")),
    ("vae", "vae", lambda t: t.__setitem__("model.unexpected_extra_weight", torch.zeros(3))),
    ("latent_upscaler", "latent_upscaler", _drop_first("_models.0.")),
    ("latent_upscaler", "latent_upscaler", lambda t: t.__setitem__("_models.0.unexpected", torch.zeros(3))),
])
def test_partial_or_extra_weights_never_load_silently(distilled_bundle, fv_args, tmp_path, component, name, mutate):
    # A silently partial load runs plausible-looking garbage: every SR component must refuse it.
    corrupted = _corrupt(distilled_bundle, tmp_path, component, mutate)
    with pytest.raises((ValueError, RuntimeError)):
        _load(name, corrupted, fv_args, component)


def test_unknown_component_class_is_reported(distilled_bundle, fv_args, tmp_path):
    import json

    copy_root = tmp_path / "bad_class"
    shutil.copytree(distilled_bundle, copy_root)
    config_path = copy_root / "latent_upscaler" / "config.json"
    config = json.loads(config_path.read_text())
    config["_class_name"] = "NoSuchBank"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not supported"):
        _load("latent_upscaler", copy_root, fv_args, "latent_upscaler")


# --------------------------------------------------------------------------- _ChunkedConv3d


@pytest.mark.parametrize("stride", [1, 2])
def test_chunked_conv3d_matches_a_plain_conv3d(monkeypatch, stride):
    # No existing test exercises the actual chunking branch (real inputs never hit _MAX_CONV_NUMEL on
    # CPU-test-sized tensors), so force it by lowering the threshold. Covers both the stride-1 path
    # (chunk-with-carried-tail) and the strided path (one kernel-width window at a time).
    from fastvideo.models.vaes import kandinsky6_sr as k6_vae

    # Small enough that the full 3072-element input needs chunking, but big enough that one
    # kernel-width window (1 x 6 x 3 x 8 x 8 = 1152 elements) still fits in a single window.
    monkeypatch.setattr(k6_vae, "_MAX_CONV_NUMEL", 1200)
    torch.manual_seed(0)
    x = torch.randn(1, 6, 8, 8, 8)  # numel = 3072 > 1200

    plain = torch.nn.Conv3d(6, 4, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=0)
    chunked = k6_vae._ChunkedConv3d(6, 4, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=0)
    chunked.load_state_dict(plain.state_dict())

    with torch.no_grad():
        expected = plain(x)
        actual = chunked(x)
    torch.testing.assert_close(actual, expected)


def test_chunked_conv3d_strided_raises_only_when_a_single_window_still_overflows(monkeypatch):
    from fastvideo.models.vaes import kandinsky6_sr as k6_vae

    monkeypatch.setattr(k6_vae, "_MAX_CONV_NUMEL", 10)  # even one kernel-width window overflows this
    x = torch.randn(1, 6, 8, 8, 8)
    chunked = k6_vae._ChunkedConv3d(6, 4, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=0)
    with pytest.raises(ValueError, match="too big"):
        chunked(x)


def test_chunked_conv3d_rejects_temporal_padding(monkeypatch):
    from fastvideo.models.vaes import kandinsky6_sr as k6_vae

    monkeypatch.setattr(k6_vae, "_MAX_CONV_NUMEL", 400)  # force the chunking branch to be reached at all
    conv = k6_vae._ChunkedConv3d(6, 4, kernel_size=(3, 1, 1), padding=(1, 0, 0))
    x = torch.randn(1, 6, 8, 8, 8)
    with pytest.raises(ValueError, match="unpadded"):
        conv(x)
