# SPDX-License-Identifier: Apache-2.0
"""CPU-only registry, preset, and model-index checks for MAGI-2 Preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fastvideo.api.presets import get_preset
from fastvideo.configs.pipelines import Magi2PreviewPipelineConfig
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.registry import (
    _get_config_info,
    get_default_preset,
    get_model_family,
    get_model_info,
    get_pipeline_config_cls_from_name,
    get_registered_model_paths,
)
from scripts.checkpoint_conversion.convert_magi2_to_fastvideo import (
    MODEL_INDEX,
    REQUIRED_COMPONENT_FILES,
    SOURCE_REPOSITORY,
    convert_checkpoint_layout,
)


def _create_indexed_source(source: Path) -> Path:
    """Create the smallest valid source layout with one shard per index."""
    indexed_shard = source / "preview" / "preview.safetensors"
    for component, relative_paths in REQUIRED_COMPONENT_FILES.items():
        for relative_path in relative_paths:
            source_path = source / component / relative_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if relative_path.endswith(".index.json"):
                shard_name = f"{component}.safetensors"
                source_path.write_text(json.dumps({"weight_map": {"weight": shard_name}}), encoding="utf-8")
                shard_path = source_path.parent / shard_name
                shard_path.write_bytes(b"weights")
                if component == "preview":
                    indexed_shard = shard_path
            else:
                source_path.write_bytes(b"component")
    return indexed_shard


@pytest.mark.parametrize("workload_type", [WorkloadType.T2V, WorkloadType.I2V])
def test_get_model_info_magi2_model_index_for_each_workload(
    tmp_path: Path,
    workload_type: WorkloadType,
) -> None:
    """Resolve the MAGI-2 pipeline class from local model-index metadata."""
    model_path = tmp_path / "checkpoint"
    (model_path / "transformer").mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        json.dumps({
            "_class_name": "Magi2Pipeline",
            "_diffusers_version": "0.37.0",
            "transformer": [None, None],
        }),
        encoding="utf-8",
    )

    model_info = get_model_info(str(model_path), workload_type=workload_type)
    config_info = _get_config_info(str(model_path))

    assert model_info.pipeline_cls.__name__ == "Magi2Pipeline"
    assert model_info.pipeline_config_cls is Magi2PreviewPipelineConfig
    assert config_info is not None
    assert config_info.workload_types == (WorkloadType.T2V, WorkloadType.I2V)
    assert get_pipeline_config_cls_from_name(str(model_path)) is Magi2PreviewPipelineConfig
    assert get_model_family(str(model_path)) == "magi2"
    assert get_default_preset(str(model_path)) == "magi2_preview_1080p"


def test_magi2_preview_pipeline_config_published_geometry() -> None:
    """Expose the published preview and refiner latent geometry."""
    pipeline_config = Magi2PreviewPipelineConfig()

    pipeline_config.check_pipeline_config()
    assert (pipeline_config.preview_width, pipeline_config.preview_height) == (896, 512)
    assert (pipeline_config.output_width, pipeline_config.output_height) == (1920, 1088)
    assert pipeline_config.output_frames == 249
    assert pipeline_config.output_fps == 25
    assert (pipeline_config.refiner_latent_width, pipeline_config.refiner_latent_height) == (120, 68)
    assert pipeline_config.text_encoder_configs == ()
    assert pipeline_config.text_encoder_precisions == ()


def test_get_preset_magi2_matches_published_profile() -> None:
    """Keep the registered preset aligned with the published inference profile."""
    preset = get_preset("magi2_preview_1080p", "magi2")

    assert preset.defaults["seed"] == 42
    assert preset.defaults["height"] == 1088
    assert preset.defaults["width"] == 1920
    assert preset.defaults["num_frames"] == 249
    assert preset.defaults["fps"] == 25
    assert preset.defaults["num_inference_steps"] == 100
    assert preset.defaults["num_inference_steps_sr"] == 5
    negative_prompt = preset.defaults["negative_prompt"]
    assert len(negative_prompt) == 1721
    assert hashlib.sha256(negative_prompt.encode()).hexdigest() == (
        "5ac0746de6c7e0388a16122d9c7e751b1cb067242218bf6f41ec1e88271cdcb9"
    )


def test_model_index_fastvideo_components_use_defining_module_paths() -> None:
    """Record each FastVideo component with its defining Python module."""
    expected_libraries = {
        "audio_vae": "fastvideo.models.vaes.magi2_audio_vae",
        "image_encoder": "fastvideo.models.vaes.magi2_wan_loader",
        "scheduler": "fastvideo.models.schedulers.scheduling_flow_unipc_multistep",
        "text_encoder": "fastvideo.models.encoders.qwen3_5",
        "transformer": "fastvideo.models.dits.magi2",
        "transformer_2": "fastvideo.models.dits.magi2_refiner",
        "vae": "fastvideo.models.vaes.magi2_turbo_vae",
    }

    assert {
        component: MODEL_INDEX[component][0]
        for component in expected_libraries
    } == expected_libraries


def test_source_repository_remains_converter_provenance() -> None:
    """Keep the official checkpoint ID as converter provenance metadata."""
    assert SOURCE_REPOSITORY == "sand-ai/MAGI-2-preview"
    assert SOURCE_REPOSITORY not in get_registered_model_paths()


def test_convert_checkpoint_layout_missing_indexed_shard(tmp_path: Path) -> None:
    """Reject a source snapshot when a weight index references a missing shard."""
    source = tmp_path / "official"
    indexed_shard = _create_indexed_source(source)
    indexed_shard.unlink()
    output = tmp_path / "fastvideo"

    with pytest.raises(FileNotFoundError, match="missing indexed checkpoint shards"):
        convert_checkpoint_layout(source, output)
    assert not output.exists()
