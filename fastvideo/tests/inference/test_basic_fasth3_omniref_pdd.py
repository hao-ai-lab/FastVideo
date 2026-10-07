# SPDX-License-Identifier: Apache-2.0
"""CPU checks of examples/inference/basic/basic_fasth3_omniref_pdd.py (no weights, no GPU)."""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "inference" / "basic" / "basic_fasth3_omniref_pdd.py"
BASE_PIN = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
CONTRACT = {
    "schema_version": "fasth3-inference-contract-v1",
    "base_model_revision": f"hf://MiniMaxAI/MiniMax-H3@{BASE_PIN}",
    "pdd_steps": 32,
    "pdd_step_indices": list(range(0, 33, 4)),
    "num_inference_steps": 8,
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "audio_scheduler_shift": 3.0,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_ref_keep_rate": 0.1,
}


def _load_example():
    spec = importlib.util.spec_from_file_location("basic_fasth3_omniref_pdd", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example()


def _args(*overrides: str):
    return example.parse_args(["--model-path", "export", "--image", "a.png", "--prompt", "p", *overrides])


@pytest.mark.parametrize("overrides,expected", [
    ((), ("MiniMaxAI/MiniMax-H3", BASE_PIN)),
    (("--base-revision", "abc"), ("MiniMaxAI/MiniMax-H3", "abc")),
    # The export's pin names a revision of its own base repo; it never applies to another one.
    (("--base-model-path", "someone/mirror"), ("someone/mirror", None)),
    (("--base-model-path", "someone/mirror", "--base-revision", "abc"), ("someone/mirror", "abc")),
])
def test_base_model_source_command_line_overrides(overrides, expected):
    assert example.base_model_source(_args(*overrides), CONTRACT) == expected


@pytest.mark.parametrize("pin", [
    BASE_PIN, f"MiniMaxAI/MiniMax-H3@{BASE_PIN}", "hf://MiniMaxAI/MiniMax-H3", "hf://@abc", "", None,
])
@pytest.mark.parametrize("overrides", [(), ("--base-model-path", "someone/mirror")])
def test_base_model_source_unparsable_base_pin(pin, overrides):
    """Never a silent fall back to the base repo's latest revision; the pipeline rejects the same values."""
    with pytest.raises(ValueError, match="must be hf://<repo id>@<revision>"):
        example.base_model_source(_args(*overrides), {**CONTRACT, "base_model_revision": pin})


_COMPONENT = ["diffusers", "Component"]
_FULL_MANIFEST = {
    "_class_name": "MiniMaxH3ModularPipeline",
    **{name: _COMPONENT for name in ("transformer", "transformer_ref", "scheduler", "audio_scheduler",
                                     "text_encoder", "tokenizer", "processor", "vae", "audio_vae")},
}
_EXPORT_ONLY_MANIFEST = {
    "_class_name": "MiniMaxH3ModularPipeline",
    **{name: _COMPONENT for name in ("transformer_ref", "scheduler", "audio_scheduler")},
}


def _snapshot_dir(root: Path, components, manifest):
    root.mkdir()
    for name in components:
        (root / name).mkdir()
    if manifest is not None:
        (root / "modular_model_index.json").write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize("export_manifest,chosen", [
    (_FULL_MANIFEST, "export"),
    (_EXPORT_ONLY_MANIFEST, "base"),
    (None, "base"),
])
def test_compose_model_dir_selects_complete_manifest(tmp_path, export_manifest, chosen):
    export = _snapshot_dir(tmp_path / "export", example.EXPORT_COMPONENTS, export_manifest)
    (export / example.CONTRACT).write_text(json.dumps(CONTRACT))
    base = _snapshot_dir(tmp_path / "base", example.BASE_COMPONENTS, _FULL_MANIFEST)
    composed = example.compose_model_dir(export, base, tmp_path / "composed")
    manifest = composed / "modular_model_index.json"
    assert manifest.resolve() == ((export if chosen == "export" else base) / "modular_model_index.json").resolve()
    for name in (*example.EXPORT_COMPONENTS, *example.BASE_COMPONENTS, example.CONTRACT):
        assert (composed / name).exists()


def test_compose_model_dir_no_complete_manifest(tmp_path):
    export = _snapshot_dir(tmp_path / "export", example.EXPORT_COMPONENTS, _EXPORT_ONLY_MANIFEST)
    (export / example.CONTRACT).write_text(json.dumps(CONTRACT))
    base = _snapshot_dir(tmp_path / "base", example.BASE_COMPONENTS, _EXPORT_ONLY_MANIFEST)
    with pytest.raises(FileNotFoundError, match="manifest that declares"):
        example.compose_model_dir(export, base, tmp_path / "composed")


def test_build_generator_config_leaves_trained_settings_to_checkpoint(tmp_path):
    # FastVideo reads the trained attention settings from the composed directory's fastvideo_inference.json.
    assert example.build_generator_config(tmp_path, 1).pipeline.experimental == {}


@pytest.mark.parametrize("num_gpus,sharded", [(1, False), (4, True)])
def test_build_generator_config_multi_gpu_shards_dit(tmp_path, num_gpus, sharded):
    config = example.build_generator_config(tmp_path, num_gpus)
    assert config.engine.use_fsdp_inference is sharded
    assert config.engine.parallelism.sp_size == num_gpus
    assert config.pipeline.components.override_pipeline_cls_name == "MiniMaxH3Ref2VAModularPipeline"


def test_default_config_offloads_the_encoders(tmp_path):
    offload = example.build_generator_config(tmp_path, 4).engine.offload
    assert (offload.text_encoder, offload.vae, offload.pin_cpu_memory) == (True, True, False)


@pytest.mark.parametrize("num_gpus", [1, 4])
def test_lossless_accel_config(tmp_path, num_gpus):
    config = example.build_generator_config(tmp_path, num_gpus, lossless_accel=True)
    offload = config.engine.offload
    assert (offload.text_encoder, offload.vae, offload.pin_cpu_memory) == (False, False, True)
    assert config.pipeline.experimental == {
        "vae_parallel_decode": num_gpus > 1,
        "vae_parallel_encode": num_gpus > 1,
    }


def test_lossless_accel_env_names_registered_flags_and_keeps_user_values() -> None:
    from fastvideo import envs

    with contextlib.ExitStack() as stack:
        for name in example.LOSSLESS_ACCEL_ENV:
            assert name in envs.environment_variables
            stack.enter_context(envs.environment_variables[name].override(None))
        stack.enter_context(envs.FASTVIDEO_ULYSSES_A2A.override("off"))
        example.apply_lossless_accel_env()
        assert envs.FASTVIDEO_ULYSSES_A2A.get() == "off"
        assert envs.FASTVIDEO_MINIMAX_H3_EXACT_KERNELS.get() == "all"
        assert envs.FASTVIDEO_H3_VSA_HEADS_FIRST_TILE.get() is True
        assert envs.FASTVIDEO_H3_VAE_TILE_PARALLEL.get() is True
        assert envs.FASTVIDEO_H3_REF2VA_MEMO_ENTRIES.get() == 16


def test_lossless_accel_flag_parses():
    argv = ["--model-path", "m", "--prompt", "p", "--image", "a.png"]
    assert example.parse_args(argv).lossless_accel is False
    assert example.parse_args([*argv, "--lossless-accel"]).lossless_accel is True
