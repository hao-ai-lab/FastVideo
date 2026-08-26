# SPDX-License-Identifier: Apache-2.0
"""Public API contracts for the Helios-Distilled pipeline."""

from argparse import ArgumentParser
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.api.schema import SamplingConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = REPO_ROOT / "official_weights" / "helios"


HELIOS_SAMPLING_VALUES = {
    "pyramid_num_inference_steps_list": [2, 2, 2],
    "history_sizes": [16, 2, 1],
    "num_latent_frames_per_chunk": 9,
    "keep_first_frame": True,
    "is_skip_first_chunk": False,
    "use_zero_init": True,
    "zero_steps": 1,
    "is_amplify_first_chunk": True,
}


def _pipeline_symbols():
    try:
        from fastvideo.configs.pipelines.helios import (
            HeliosPipelineConfig,
            helios_postprocess_text,
            helios_preprocess_text,
        )
        from fastvideo.pipelines.basic.helios.helios_pipeline import HeliosPyramidPipeline
        from fastvideo.pipelines.basic.helios.presets import HELIOS_DISTILLED_T2V
    except ImportError as exc:
        raise AssertionError("Helios pipeline/config/preset have not been implemented") from exc
    return (
        HeliosPipelineConfig,
        helios_preprocess_text,
        helios_postprocess_text,
        HeliosPyramidPipeline,
        HELIOS_DISTILLED_T2V,
    )


def test_helios_public_sampling_fields_reach_forward_batch() -> None:
    sampling_fields = {item.name for item in fields(SamplingParam)}
    typed_fields = {item.name for item in fields(SamplingConfig)}
    batch_fields = {item.name for item in fields(ForwardBatch)}
    expected = set(HELIOS_SAMPLING_VALUES)

    assert expected <= sampling_fields
    assert expected <= typed_fields
    assert expected <= batch_fields


def test_helios_internal_chunk_state_is_declared_on_forward_batch() -> None:
    batch_fields = {item.name for item in fields(ForwardBatch)}

    assert "helios_latent_chunks" in batch_fields


def test_helios_typed_request_maps_sampling_fields() -> None:
    request = normalize_generation_request({"sampling": HELIOS_SAMPLING_VALUES})
    sampling = request_to_sampling_param(
        request,
        model_path="BestWishYsh/Helios-Distilled",
    )

    for name, expected in HELIOS_SAMPLING_VALUES.items():
        assert getattr(sampling, name) == expected


def test_helios_sampling_fields_are_available_from_cli() -> None:
    parser = ArgumentParser()
    SamplingParam.add_cli_args(parser)

    args = parser.parse_args([
        "--pyramid-num-inference-steps-list",
        "3",
        "2",
        "1",
        "--history-sizes",
        "8",
        "2",
        "1",
        "--num-latent-frames-per-chunk",
        "9",
        "--keep-first-frame",
        "true",
        "--is-skip-first-chunk",
        "false",
        "--use-zero-init",
        "true",
        "--zero-steps",
        "2",
        "--is-amplify-first-chunk",
        "true",
    ])

    assert args.pyramid_num_inference_steps_list == [3, 2, 1]
    assert args.history_sizes == [8, 2, 1]
    assert args.num_latent_frames_per_chunk == 9
    assert args.keep_first_frame is True
    assert args.is_skip_first_chunk is False
    assert args.use_zero_init is True
    assert args.zero_steps == 2
    assert args.is_amplify_first_chunk is True


def _valid_helios_batch(**overrides) -> ForwardBatch:
    values = {
        "data_type": "video",
        "prompt": "A paper boat floats across a rain puddle.",
        "seed": 42,
        "height": 128,
        "width": 192,
        "num_frames": 33,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "pyramid_num_inference_steps_list": [2, 2, 2],
        "history_sizes": [16, 2, 1],
        "num_latent_frames_per_chunk": 9,
        "keep_first_frame": True,
        "is_skip_first_chunk": False,
        "zero_steps": 1,
    }
    values.update(overrides)
    return ForwardBatch(**values)


def _validation_args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            ti2v_task=False,
            is_causal=False,
        ), )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"height": 120}, "divisible by 64"),
        ({"pyramid_num_inference_steps_list": [2, 2]}, "three positive pyramid step counts"),
        ({"pyramid_num_inference_steps_list": [2, 0, 2]}, "three positive pyramid step counts"),
        ({"history_sizes": [16, 2]}, "three positive history sizes"),
        ({"history_sizes": [16, 0, 1]}, "three positive history sizes"),
        ({"num_latent_frames_per_chunk": 8}, "requires num_latent_frames_per_chunk=9"),
        ({"keep_first_frame": False}, "requires keep_first_frame=True"),
        ({"is_skip_first_chunk": True}, "only meaningful for conditioned Helios modes"),
        ({"zero_steps": -1}, "zero_steps must be non-negative"),
    ],
)
def test_helios_input_validation_rejects_unverified_contracts(overrides, message) -> None:
    try:
        from fastvideo.pipelines.basic.helios.stages import HeliosInputValidationStage
    except ImportError as exc:
        raise AssertionError("HeliosInputValidationStage has not been implemented") from exc

    with pytest.raises(ValueError, match=message):
        HeliosInputValidationStage().forward(_valid_helios_batch(**overrides), _validation_args())


def test_helios_pipeline_config_and_text_contract() -> None:
    from fastvideo.configs.models.encoders import BaseEncoderOutput

    HeliosPipelineConfig, preprocess, postprocess, _, _ = _pipeline_symbols()
    config = HeliosPipelineConfig()
    assert config.dit_config.__class__.__name__ == "HeliosConfig"
    assert config.vae_config.__class__.__name__ == "WanVAEConfig"
    assert config.vae_config.load_encoder is False
    assert config.vae_config.load_decoder is True
    assert config.text_encoder_configs[0].architectures == ["UMT5EncoderModel"]
    assert config.text_encoder_configs[0].text_len == 512
    assert config.dit_precision == "bf16"
    assert config.text_encoder_precisions == ("bf16", )
    assert config.vae_precision == "fp32"
    assert config.flow_shift is None
    assert preprocess("  A\n fish &amp; reef  ") == "A fish & reef"

    hidden = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    output = postprocess(BaseEncoderOutput(last_hidden_state=hidden, attention_mask=mask))
    assert output.shape == (2, 512, 4)
    assert torch.equal(output[0, :3], hidden[0, :3])
    assert torch.count_nonzero(output[0, 3:]) == 0
    assert torch.equal(output[1, :5], hidden[1, :5])


def test_helios_preset_matches_official_distilled_defaults() -> None:
    _, _, _, _, preset = _pipeline_symbols()
    assert preset.model_family == "helios"
    assert preset.workload_type == "t2v"
    assert preset.defaults["height"] == 384
    assert preset.defaults["width"] == 640
    assert preset.defaults["num_frames"] == 240
    assert preset.defaults["fps"] == 24
    assert preset.defaults["guidance_scale"] == 1.0
    assert preset.defaults["pyramid_num_inference_steps_list"] == [2, 2, 2]
    assert preset.defaults["history_sizes"] == [16, 2, 1]
    assert preset.defaults["num_latent_frames_per_chunk"] == 9
    assert preset.defaults["is_amplify_first_chunk"] is True


def test_helios_pipeline_entry_class_is_discoverable() -> None:
    from fastvideo.pipelines.pipeline_registry import PipelineType, import_pipeline_classes

    _, _, _, pipeline_class, _ = _pipeline_symbols()
    assert pipeline_class.__name__ == "HeliosPyramidPipeline"
    assert pipeline_class._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]
    discovered = import_pipeline_classes(PipelineType.BASIC)["basic"]
    assert discovered["HeliosPyramidPipeline"] is pipeline_class


def test_helios_local_model_registry_selection() -> None:
    from fastvideo.registry import get_model_family, get_pipeline_config_cls_from_name

    if not MODEL_DIR.exists():
        pytest.skip(f"Pinned Helios checkpoint not found at {MODEL_DIR}")
    assert get_model_family(str(MODEL_DIR)) == "helios"
    assert get_pipeline_config_cls_from_name(str(MODEL_DIR)).__name__ == "HeliosPipelineConfig"


def test_helios_registry_requires_distilled_metadata() -> None:
    script = r'''
import json
import tempfile
from pathlib import Path

from fastvideo.registry import get_model_family

def write_model_index(model_dir, is_distilled):
    model_dir.mkdir()
    for component in ("transformer", "scheduler", "text_encoder", "vae"):
        (model_dir / component).mkdir()
    (model_dir / "model_index.json").write_text(json.dumps({
        "_class_name": "HeliosPyramidPipeline",
        "_diffusers_version": "0.39.0",
        "is_distilled": is_distilled,
        "scheduler": ["diffusers", "HeliosDMDScheduler"],
        "transformer": ["diffusers", "HeliosTransformer3DModel"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }))

with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    distilled = root / "renamed_helios_checkpoint"
    write_model_index(distilled, True)
    assert get_model_family(str(distilled)) == "helios"
    for name in ("Helios-Base", "Helios-Mid", "unrelated-helios-experiment"):
        model_dir = root / name
        write_model_index(model_dir, False)
        assert get_model_family(str(model_dir)) is None, name
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
