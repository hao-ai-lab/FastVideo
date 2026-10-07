# SPDX-License-Identifier: Apache-2.0
"""The real component configs (no weights) of both official ``Kandinsky6SRPipeline`` repos against the SR modules.

Each component is built on the meta device from the release config exactly as the loaders would, and the tensor counts /
key groups are compared with the safetensors headers of the Hub repos (459 DiT, 378 KVAE, 489 latent-upscaler tensors).
The flow-matching repo (``Kandinsky-6.0-VSR-5s-Diffusers``) and the distilled repo
(``Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``) share the VAE and the latent upscaler and differ in the DiT head,
the scheduler and ``sr_params``.  This guards the port against drifting away from the official release without needing
a GPU, the weights or the Hub.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any, NamedTuple

import pytest
import torch

from fastvideo.configs.pipelines.kandinsky6_sr_options import Kandinsky6SROptions
from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRConfig
from fastvideo.configs.models.vaes.kandinsky6_sr import Kandinsky6SRVAEConfig
from fastvideo.pipelines.basic.kandinsky6_sr.presets import KANDINSKY6_SR
from k6_sr_release import (DISTILLED_MODEL_INDEX, DISTILLED_SCHEDULER_CONFIG, DISTILLED_TRANSFORMER_CONFIG,
                           FLOW_MATCHING_MODEL_INDEX, FLOW_MATCHING_SCHEDULER_CONFIG, FLOW_MATCHING_SR_CONFIG,
                           FLOW_MATCHING_TRANSFORMER_CONFIG, OLD_RELEASE_VAE_CONFIG,
                           RELEASE_LATENT_UPSCALER_CONFIG, RELEASE_VAE_CONFIG)


class _Bundle(NamedTuple):
    transformer: dict[str, Any]
    scheduler: dict[str, Any]
    model_index: dict[str, Any]
    tiny_bundle: str  # name of the conftest fixture with the tiny repo of the same kind
    head_width: int  # ``out_visual_dim``: the DiT output channels
    scheduler_scale: float  # ``sr_params.scheduler_scale``: the timestep shift the model was trained with


_BUNDLES = {
    "flow_matching": _Bundle(FLOW_MATCHING_TRANSFORMER_CONFIG, FLOW_MATCHING_SCHEDULER_CONFIG,
                             FLOW_MATCHING_MODEL_INDEX, "flow_matching_bundle", 64, 5.0),
    "distilled": _Bundle(DISTILLED_TRANSFORMER_CONFIG, DISTILLED_SCHEDULER_CONFIG, DISTILLED_MODEL_INDEX,
                         "distilled_bundle", 640, 3.5),
}
both_bundles = pytest.mark.parametrize("bundle", list(_BUNDLES.values()), ids=list(_BUNDLES))


def _prefixes(keys, depth=2):
    return Counter(".".join(key.split(".")[:depth]) for key in keys)


def _meta_dit(transformer_config: dict[str, Any]):
    from fastvideo.models.dits.kandinsky6_sr import Kandinsky6SRTransformer3DModel

    config = Kandinsky6SRConfig()
    config.update_model_arch(transformer_config)
    with torch.device("meta"):
        return Kandinsky6SRTransformer3DModel(config, transformer_config)  # rejects any undeclared config key


def test_the_release_transformer_configs_differ_only_in_head_timestep_warp_and_nabla_warmup():
    flow, distilled = copy.deepcopy(FLOW_MATCHING_TRANSFORMER_CONFIG), copy.deepcopy(DISTILLED_TRANSFORMER_CONFIG)
    assert (flow.pop("out_visual_dim"), distilled.pop("out_visual_dim")) == (64, 640)
    assert (flow["sr_params"].pop("scheduler_scale"), distilled["sr_params"].pop("scheduler_scale")) == (5.0, 3.5)
    assert (flow["attention_params"]["512"].pop("P_warmup_steps"),
            distilled["attention_params"]["512"].pop("P_warmup_steps")) == (200, 0)
    assert flow == distilled


@both_bundles
def test_real_transformer_config_builds_the_release_dit(distributed_setup, bundle):
    dit = _meta_dit(bundle.transformer)
    state = dit.state_dict()
    per_block = Counter(key.split(".")[1] for key in state if key.startswith("visual_transformer_blocks."))
    assert len(state) == 459 and len(per_block) == 32 and set(per_block.values()) == {14}
    assert len([key for key in state if not key.startswith("visual_transformer_blocks.")]) == 11
    # ``out_visual_dim`` is the full head width (the distilled head is 64 * n_grid 10); only
    # ``out_layer.out_layer.{weight,bias}`` differ between the two repos
    assert dit.out_layer.out_layer.weight.shape == (bundle.head_width, 1792)
    assert dit.out_layer.out_layer.bias.shape == (bundle.head_width, )
    assert dit.input_channels == 2 * 64 + 1  # noised latent | anchor | anchor mask
    arch = dit.config.arch_config
    assert arch.rope_scale_factor == (1.0, 2.0, 2.0) and arch.visual_size == 512
    assert arch.sr_params["scheduler_scale"] == bundle.scheduler_scale


@both_bundles
def test_real_scheduler_config_loads_natively_with_the_training_shift(distributed_setup, fv_args, tmp_path, bundle):
    from fastvideo.models.loader.component_loader import SchedulerLoader

    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(bundle.scheduler))
    scheduler = SchedulerLoader().load(str(tmp_path / "scheduler"), fv_args)
    assert type(scheduler).__name__ == bundle.scheduler["_class_name"]
    # The scheduler's shift is the timestep warp the SR DiT was trained with.
    assert scheduler.config.shift == bundle.scheduler_scale
    if getattr(scheduler, "is_piflow", False):
        assert scheduler.n_grid * 64 == bundle.head_width  # the head holds n_grid predictions per channel


@both_bundles
def test_real_model_index_passes_through_load_modules(distributed_setup, request, cpu_args, bundle):
    from fastvideo.pipelines.basic.kandinsky6_sr.kandinsky6_sr_pipeline import Kandinsky6SRPipeline

    repo = request.getfixturevalue(bundle.tiny_bundle)
    index = json.loads((repo / "model_index.json").read_text())
    assert index == bundle.model_index  # the tiny repo carries the release index verbatim
    pipeline = Kandinsky6SRPipeline(str(repo), cpu_args(repo))
    pipeline.post_init()
    # components are the [library, class] entries; the dict-valued ``_kandinsky6_sr`` metadata is skipped, not loaded
    assert set(pipeline.modules) == {"transformer", "vae", "scheduler", "latent_upscaler"}
    assert type(pipeline.get_module("scheduler")).__name__ == bundle.scheduler["_class_name"]


def test_the_flow_matching_release_carries_kandinsky6_sr_metadata_and_the_preset_default_scale():
    assert isinstance(FLOW_MATCHING_MODEL_INDEX["_kandinsky6_sr"], dict)
    assert "_kandinsky6_sr" not in DISTILLED_MODEL_INDEX
    # root sr_config.json: the default scale of the release is the preset default (and the request default)
    default_scale = FLOW_MATCHING_SR_CONFIG["default_resolution_scale"]
    assert default_scale == KANDINSKY6_SR.defaults["sr_resolution_scale"] == Kandinsky6SROptions.sr_resolution_scale == 2.25


@pytest.mark.parametrize("vae_config", [RELEASE_VAE_CONFIG, OLD_RELEASE_VAE_CONFIG],
                        ids=["flat_2026_09_30", "nested_pre_2026_09_30"])
def test_real_vae_config_builds_the_release_kvae_and_maps_official_keys(vae_config):
    # The Hub re-exported vae/config.json's layout (flat vs. nested encoder_config/decoder_config,
    # see k6_sr_release.py) between when these two fixtures were captured; both must build the
    # identical KVAE since the underlying architecture never changed.
    from fastvideo.models.vaes.kandinsky6_sr import Kandinsky6SRVAE

    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(dict(vae_config))
    with torch.device("meta"):
        vae = Kandinsky6SRVAE(config)
    official = vae.state_dict()
    assert len(official) == 378
    assert _prefixes(official) == Counter({
        "decoder.up": 238, "encoder.down": 86, "decoder.mid": 28, "encoder.mid": 12, "decoder.norm_out": 5,
        "decoder.conv_in": 2, "decoder.conv_out": 2, "encoder.conv_in": 2, "encoder.conv_out": 2, "encoder.norm_out": 1})
    assert vae.scaling_factor == pytest.approx(0.910344004631042) and vae.spatial_factor == 16
    # the official keys load strictly as they are
    vae.load_state_dict(official, strict=True, assign=True)


def test_real_latent_upscaler_config_builds_the_release_bank():
    from fastvideo.configs.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerConfig
    from fastvideo.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerBank

    with torch.device("meta"):
        bank = Kandinsky6SRLatentUpscalerBank(Kandinsky6SRLatentUpscalerConfig(**RELEASE_LATENT_UPSCALER_CONFIG))
    state = bank.state_dict()
    # Entries are indexed in ``config.scales`` order (default (2, 4)), as in the release safetensors.
    assert len(state) == 489 and _prefixes(state) == Counter({"_models.0": 311, "_models.1": 178})
    assert not any("motion_attention" in entry["model"] for entry in RELEASE_LATENT_UPSCALER_CONFIG["models"])
    x2 = [key for key in state if key.startswith("_models.0.")]
    assert _prefixes((key.split(".", 2)[2] for key in x2), 2)["x2_branch.blocks"] == 44
    assert _prefixes((key.split(".", 2)[2] for key in x2), 2)["x2_branch.adapter"] == 28
    assert bank.scales == (2, 4)
