# SPDX-License-Identifier: Apache-2.0
"""Real-import compatibility for the family-local Wan package.

Use an installed FastVideo environment with its runtime dependencies; these
checks do not download checkpoints or instantiate the full transformer.
"""

import importlib
import pickle
import subprocess
import sys

import pytest

from fastvideo.models.wan import causal_transformer, config, transformer, vae, vae_config

TRANSFORMER_EXPORTS = (
    "EntryClass",
    "LayerNormScaleShift",
    "PatchEmbed",
    "WanI2VCrossAttention",
    "WanImageEmbedding",
    "WanSelfAttention",
    "WanT2VCrossAttention",
    "WanTimeTextImageEmbedding",
    "WanTransformer3DModel",
    "WanTransformerBlock",
    "WanTransformerBlock_VSA",
)
CONFIG_EXPORTS = ("WanVideoArchConfig", "WanVideoConfig", "is_blocks")
VAE_EXPORTS = (
    "CACHE_T", "AutoencoderKLWan", "AvgDown3D", "DiagonalGaussianDistribution",
    "DupUp3D", "EntryClass", "ParallelTiledVAE", "WanAttentionBlock", "WanCausalConv3d",
    "WanDecoder3d", "WanEncoder3d", "WanMidBlock", "WanRMS_norm", "WanResample",
    "WanResidualBlock", "WanResidualDownBlock", "WanResidualUpBlock", "WanUpBlock",
    "WanUpsample", "WanVAEConfig", "_is_wan_vae_codec", "feat_cache", "feat_idx",
    "first_chunk", "forward_context", "is_first_frame", "patchify", "unpatchify",
    "use_light_vae",
)
VAE_CONFIG_EXPORTS = ("WanVAEArchConfig", "WanVAEConfig")


def test_legacy_transformer_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.models.dits.wanvideo")
    assert set(legacy.__all__) == set(TRANSFORMER_EXPORTS)
    for name in TRANSFORMER_EXPORTS:
        assert getattr(legacy, name) is getattr(transformer, name)
    assert legacy.EntryClass is transformer.WanTransformer3DModel


def test_legacy_causal_transformer_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.models.dits.causal_wanvideo")
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(causal_transformer, name)
    assert legacy.EntryClass is causal_transformer.CausalWanTransformer3DModel


def test_legacy_and_aggregate_config_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.configs.models.dits.wanvideo")
    aggregate = importlib.import_module("fastvideo.configs.models.dits")
    assert set(legacy.__all__) == set(CONFIG_EXPORTS)
    for name in CONFIG_EXPORTS:
        assert getattr(legacy, name) is getattr(config, name)
    assert aggregate.WanVideoConfig is config.WanVideoConfig


def test_legacy_vae_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.models.vaes.wanvae")
    assert set(legacy.__all__) == set(VAE_EXPORTS)
    for name in VAE_EXPORTS:
        assert getattr(legacy, name) is getattr(vae, name)
    assert legacy.EntryClass is vae.AutoencoderKLWan


def test_legacy_and_aggregate_vae_config_exports_are_the_canonical_objects():
    legacy = importlib.import_module("fastvideo.configs.models.vaes.wanvae")
    aggregate = importlib.import_module("fastvideo.configs.models.vaes")
    assert set(legacy.__all__) == set(VAE_CONFIG_EXPORTS)
    for name in VAE_CONFIG_EXPORTS:
        assert getattr(legacy, name) is getattr(vae_config, name)
    assert aggregate.WanVAEConfig is vae_config.WanVAEConfig


@pytest.mark.parametrize("config_name, load_encoder", [
    ("WanT2V480PConfig", False),
    ("WanI2V480PConfig", True),
    ("LucyEditDevConfig", True),
])
def test_wan_pipeline_configs_use_canonical_vae_types(config_name, load_encoder):
    pipeline = importlib.import_module("fastvideo.configs.pipelines.wan")
    assert pipeline.WanVAEConfig is vae_config.WanVAEConfig
    assert pipeline.WanVAEArchConfig is vae_config.WanVAEArchConfig
    value = getattr(pipeline, config_name)().vae_config
    assert type(value) is vae_config.WanVAEConfig
    assert type(value.arch_config) is vae_config.WanVAEArchConfig
    assert value.load_encoder is load_encoder
    assert value.load_decoder


def test_legacy_vae_cache_context_is_shared_and_restored_after_errors():
    legacy = importlib.import_module("fastvideo.models.vaes.wanvae")
    contexts = (vae.is_first_frame, vae.feat_cache, vae.feat_idx, vae.first_chunk, vae.use_light_vae)
    original = tuple(context.get() for context in contexts)
    cache = [None]
    with legacy.forward_context(True, cache, 3, True, True):
        assert tuple(context.get() for context in contexts) == (True, cache, 3, True, True)
        with pytest.raises(RuntimeError, match="nested decode"):
            with vae.forward_context(False, [], 0, False, False):
                assert legacy.feat_cache.get() == []
                legacy.feat_idx.set(2)
                assert vae.feat_idx.get() == 2
                raise RuntimeError("nested decode")
        assert vae.feat_cache.get() is cache
        assert tuple(context.get() for context in contexts) == (True, cache, 3, True, True)
    assert tuple(context.get() for context in contexts) == original


@pytest.mark.parametrize("module_name", [
    "fastvideo.models.dits.causal_wanvideo",
    "fastvideo.models.dits.dreamx_world",
    "fastvideo.models.dits.matrixgame2.model",
    "fastvideo.models.dits.matrixgame2.causal_model",
    "fastvideo.models.dits.matrixgame3.model",
    "fastvideo.models.dits.lingbotworld.model",
    "fastvideo.configs.models.dits.dreamx_world",
    "fastvideo.configs.models.dits.matrixgame2",
    "fastvideo.configs.models.dits.matrixgame3",
    "fastvideo.configs.pipelines.wan",
    "fastvideo.configs.pipelines.dreamx_world",
    "fastvideo.configs.pipelines.longcat",
    "fastvideo.configs.pipelines.lingbot_video",
    "fastvideo.configs.pipelines.turbodiffusion",
    "fastvideo.configs.pipelines.lingbotworld2",
])
def test_downstream_wan_consumers_import(module_name):
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name, architecture", [
    ("transformer", "WanTransformer3DModel"),
    ("causal_transformer", "CausalWanTransformer3DModel"),
    ("vae", "AutoencoderKLWan"),
])
def test_registry_discovers_and_loads_the_canonical_wan_class(module_name, architecture, caplog):
    from fastvideo.models import registry

    expected = ("wan", module_name, architecture)
    discovered = registry._discover_and_register_models()
    assert discovered[architecture] == expected
    assert registry._LEGACY_FAST_VIDEO_MODELS[architecture] == expected
    assert registry._FAST_VIDEO_MODELS[architecture] == expected
    assert f"Duplicate architecture found: {architecture}." not in caplog.text
    model_cls, resolved_architecture = registry.ModelRegistry.resolve_model_cls(architecture)
    module = importlib.import_module(f"fastvideo.models.wan.{module_name}")
    assert model_cls is getattr(module, architecture)
    assert resolved_architecture == architecture


@pytest.mark.parametrize("first_module", [
    "fastvideo.models.wan.config",
    "fastvideo.models.wan.transformer",
    "fastvideo.models.wan.causal_transformer",
    "fastvideo.models.dits.causal_wanvideo",
    "fastvideo.configs.models.dits.wanvideo",
    "fastvideo.models.dits.wanvideo",
    "fastvideo.models.wan.vae_config",
    "fastvideo.models.wan.vae",
    "fastvideo.configs.models.vaes.wanvae",
    "fastvideo.models.vaes.wanvae",
])
def test_import_order_and_config_transport_in_fresh_process(first_module):
    original = config.WanVideoConfig()
    # Protocol 0 GLOBAL references reproduce the old qualified names without
    # changing __module__ or keeping a second config implementation alive.
    serialized = pickle.dumps(original, protocol=0)
    legacy_serialized = serialized.replace(
        b"fastvideo.models.wan.config\n",
        b"fastvideo.configs.models.dits.wanvideo\n",
    )
    assert legacy_serialized != serialized
    vae_original = vae_config.WanVAEConfig(use_feature_cache=False, use_tiling=True, load_encoder=False)
    vae_serialized = pickle.dumps(vae_original, protocol=0)
    legacy_vae_serialized = vae_serialized.replace(
        b"fastvideo.models.wan.vae_config\n",
        b"fastvideo.configs.models.vaes.wanvae\n",
    )
    assert legacy_vae_serialized != vae_serialized
    script = """
import importlib
import pickle
import sys
import torch

importlib.import_module(sys.argv[1])
from fastvideo.models.wan.config import WanVideoArchConfig, WanVideoConfig, is_blocks
from fastvideo.configs.models.dits import WanVideoConfig as AggregateConfig
from fastvideo.models.wan.transformer import WanTransformer3DModel
from fastvideo.models.dits.wanvideo import WanTransformer3DModel as LegacyTransformer
from fastvideo.models.wan.vae_config import WanVAEArchConfig, WanVAEConfig
from fastvideo.configs.models.vaes import WanVAEConfig as AggregateVAEConfig
from fastvideo.models.wan.vae import AutoencoderKLWan
from fastvideo.models.vaes.wanvae import AutoencoderKLWan as LegacyVAE

assert AggregateConfig is WanVideoConfig
assert LegacyTransformer is WanTransformer3DModel
assert AggregateVAEConfig is WanVAEConfig
assert LegacyVAE is AutoencoderKLWan
assert is_blocks('blocks.0', None)
assert not is_blocks('blocks.0.attn1', None)
transformer_payloads, vae_payloads = pickle.load(sys.stdin.buffer)
for payload in transformer_payloads:
    value = pickle.loads(payload)
    assert type(value) is WanVideoConfig
    assert type(value.arch_config) is WanVideoArchConfig
    assert value == WanVideoConfig()
    assert value.arch_config.hidden_size == 40 * 128
    assert value.arch_config._fsdp_shard_conditions == [is_blocks]
    assert pickle.loads(pickle.dumps(value)) == value
for payload in vae_payloads:
    value = pickle.loads(payload)
    assert type(value) is WanVAEConfig
    assert type(value.arch_config) is WanVAEArchConfig
    assert not value.use_feature_cache
    assert value.use_tiling
    assert not value.load_encoder
    assert value.load_decoder
    expected = WanVAEConfig()
    assert value.blend_num_frames == expected.blend_num_frames
    for field in ('scaling_factor', 'shift_factor'):
        assert torch.equal(getattr(value, field), getattr(expected, field))
        assert torch.equal(getattr(pickle.loads(pickle.dumps(value)), field), getattr(value, field))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, first_module],
        input=pickle.dumps(((serialized, legacy_serialized), (vae_serialized, legacy_vae_serialized))),
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
