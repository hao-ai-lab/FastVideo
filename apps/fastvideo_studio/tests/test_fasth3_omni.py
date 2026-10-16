# SPDX-License-Identifier: Apache-2.0
"""FastH3-Omni in the Studio: recipe, composition, catalogue, persistence and the run path (no GPU, no weights)."""
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from fastvideo_studio import fasth3_omni
from fastvideo_studio.database import Database
from fastvideo_studio.fasth3_omni import (FastH3OmniRegistry, OmniHardwareError, OmniWeightsError, apply_recipe,
                                          classify_error, validate_omni_job)
from fastvideo_studio.job_runner import JobRunner, JobStatus
from fastvideo_studio.model_catalog import build_catalog

BASE_PIN = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
# The OmniRef PDD-8 export's fastvideo_inference.json.
CONTRACT = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "audio_scheduler_shift": 3.0,
    "base_model_revision": f"hf://MiniMaxAI/MiniMax-H3@{BASE_PIN}",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "grid_max_t": 0.999,
    "guidance_scale": 1.0,
    "model_type": "ref2va",
    "num_inference_steps": 8,
    "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "pdd_steps": 32,
    "schema": "fasth3-inference-contract-v1",
    "schema_version": "fasth3-inference-contract-v1",
    "transformer_component": "transformer_ref",
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "vsa_ref_keep_rate": 0.1,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
}
_COMPONENT = ["diffusers", "Component"]
MANIFEST = {
    "_class_name": "MiniMaxH3ModularPipeline",
    **{name: _COMPONENT for name in ("transformer", "transformer_ref", "scheduler", "audio_scheduler",
                                     "text_encoder", "tokenizer", "processor", "vae", "audio_vae")},
}
REFS = [
    {"source": "/up/dance.mp4", "media_type": "video"},
    {"source": "/up/outfit.png", "media_type": "image"},
    {"source": "/up/voice.wav", "media_type": "audio"},
]


def make_export(root: Path, contract=None, pdd_steps=32) -> Path:
    contract = dict(CONTRACT if contract is None else contract)
    for name, config in (("transformer_ref", {"pdd_steps": pdd_steps}),
                         ("scheduler", {"shift": contract.get("video_scheduler_shift", 12.0)}),
                         ("audio_scheduler", {"shift": contract.get("audio_scheduler_shift", 3.0)})):
        (root / name).mkdir(parents=True)
        filename = "config.json" if name == "transformer_ref" else "scheduler_config.json"
        (root / name / filename).write_text(json.dumps(config))
    (root / "fastvideo_inference.json").write_text(json.dumps(contract))
    return root


def make_base(root: Path, skip=()) -> Path:
    for name in ("text_encoder", "tokenizer", "processor", "vae", "audio_vae"):
        if name not in skip:
            (root / name).mkdir(parents=True)
    (root / "modular_model_index.json").write_text(json.dumps(MANIFEST))
    return root


@pytest.fixture
def no_gpu_check(monkeypatch):
    monkeypatch.setattr(fasth3_omni, "hardware_status", lambda tile: {"supported": True, "reason": ""})


@pytest.fixture
def registry(tmp_path, no_gpu_check):
    export = make_export(tmp_path / "omni_export")
    base = make_base(tmp_path / "h3_base")
    return FastH3OmniRegistry([str(export)], str(base), str(tmp_path / "cache"))


def omni_fields(**over):
    fields = dict(job_type="inference", model_id="x", prompt="p", references=REFS, num_gpus=1, num_frames=124,
                  height=480, width=832, image_path="", last_image_path="", num_inference_steps=50,
                  guidance_scale=5.0, guidance_rescale=0.3, negative_prompt="blurry", vsa_sparsity=0.0, fps=16,
                  enable_torch_compile=True, tp_size=-1, sp_size=-1, use_fsdp_inference=False)
    fields.update(over)
    return fields


# --- Recipe ----------------------------------------------------------------


def test_registry_reads_recipe_from_contract(registry, tmp_path):
    export = registry.get(str(tmp_path / "omni_export"))
    assert export.error is None
    recipe = export.recipe()
    assert recipe["source"] == "fastvideo_inference.json"
    assert recipe["contract"]["num_inference_steps"] == 8
    assert recipe["contract"]["pdd_steps"] == 32
    # --fasth3-omni-base overrides the export's pin, and then carries no revision of its own.
    assert (recipe["base"], recipe["base_revision"]) == (str(tmp_path / "h3_base"), None)


def test_recipe_defaults_to_the_pinned_base(tmp_path, no_gpu_check):
    export = make_export(tmp_path / "e")
    recipe = FastH3OmniRegistry([str(export)], None, str(tmp_path / "c")).get(str(export)).recipe()
    assert (recipe["base"], recipe["base_revision"]) == ("MiniMaxAI/MiniMax-H3", BASE_PIN)


@pytest.mark.parametrize("contract,pdd_steps,match", [
    ({**CONTRACT, "num_inference_steps": 9}, 32, "disagrees"),
    ({**CONTRACT, "attention_backend": "FLASH_ATTN"}, 32, "unsupported"),
    (CONTRACT, 16, "pdd_steps"),
    ({k: v for k, v in CONTRACT.items() if k != "vsa_tile_size"}, 32, "missing"),
    ({**CONTRACT, "base_model_revision": "MiniMaxAI/MiniMax-H3"}, 32, "hf://"),
])
def test_malformed_contract_marks_export_unusable(tmp_path, no_gpu_check, contract, pdd_steps, match):
    export = make_export(tmp_path / "e", contract, pdd_steps=pdd_steps)
    entry = FastH3OmniRegistry([str(export)], None, str(tmp_path / "c")).get(str(export))
    assert entry.contract is None and entry.recipe() is None
    assert match in entry.error


def test_missing_local_export_is_reported_as_missing(tmp_path, no_gpu_check):
    entry = FastH3OmniRegistry([str(tmp_path / "nope")], None, str(tmp_path / "c")).get(str(tmp_path / "nope"))
    assert entry.error.endswith("local export directory not found.")


def test_v2_style_contract_is_not_an_omni_export(tmp_path, no_gpu_check):
    """A T2AV DMD export (FastH3 V2) has no PDD fields; renaming it does not make it Omni."""
    dmd = {"schema_version": "fasth3-inference-contract-v1", "num_inference_steps": 9, "transformer_forwards": 8,
           "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125], "video_scheduler_shift": 10.0,
           "audio_scheduler_shift": 3.0}
    export = make_export(tmp_path / "fasth3_omni_renamed", dmd)
    entry = FastH3OmniRegistry([str(export)], None, str(tmp_path / "c")).get(str(export))
    assert "not a FastH3-Omni export" in entry.error


def test_apply_recipe_overrides_generic_form_defaults(registry, tmp_path):
    recipe = registry.get(str(tmp_path / "omni_export")).recipe()
    locked = apply_recipe(omni_fields(num_gpus=4), recipe)
    assert locked["num_inference_steps"] == 8           # eight fused blocks, eight forwards
    assert locked["guidance_scale"] == 1.0
    assert locked["guidance_rescale"] == 0.0
    assert locked["negative_prompt"] == ""
    assert locked["vsa_sparsity"] == 0.9
    assert locked["fps"] == 24
    assert locked["enable_torch_compile"] is False
    assert (locked["use_fsdp_inference"], locked["tp_size"], locked["sp_size"]) == (True, 1, 4)
    assert locked["recipe"] is recipe
    # User-owned fields pass through.
    assert (locked["num_frames"], locked["height"], locked["width"]) == (124, 480, 832)
    assert locked["references"] == REFS


@pytest.mark.parametrize("over,match", [
    ({"references": []}, "at least one ordered reference"),
    ({"references": [REFS[2]]}, "audio alone"),
    ({"references": [{"source": "/a.png", "media_type": "image"}] * 10}, "At most 9 image"),
    ({"references": [{"source": "", "media_type": "image"}]}, "no source"),
    ({"references": [{"source": "/a.txt", "media_type": "text"}]}, "media_type"),
    ({"image_path": "/start.png"}, "not a start or end frame"),
    ({"num_gpus": 3}, "56 attention heads"),
    ({"num_frames": 400}, "1 to 361 frames"),
    ({"job_type": "finetuning"}, "inference model"),
])
def test_validate_omni_job_rejects(over, match):
    with pytest.raises(ValueError, match=match):
        validate_omni_job(omni_fields(**over))


def test_validate_omni_job_accepts_mixed_references():
    validate_omni_job(omni_fields())


# --- Composition and caching ------------------------------------------------


def test_prepare_composes_once_per_snapshot_pair(registry, tmp_path):
    model_id = str(tmp_path / "omni_export")
    model_dir, recipe = registry.prepare(model_id)
    for name in ("transformer_ref", "scheduler", "audio_scheduler", "fastvideo_inference.json", "text_encoder",
                 "tokenizer", "processor", "vae", "audio_vae", "modular_model_index.json"):
        assert (model_dir / name).is_symlink(), name
    assert (model_dir / "transformer_ref").resolve() == (tmp_path / "omni_export" / "transformer_ref").resolve()
    assert (model_dir / "vae").resolve() == (tmp_path / "h3_base" / "vae").resolve()
    provenance = json.loads((model_dir.parent / f"{model_dir.name}.json").read_text())
    assert provenance["export_dir"] == str(tmp_path / "omni_export")
    assert provenance["base_dir"] == str(tmp_path / "h3_base")
    assert provenance["contract_sha256"] == recipe["contract_sha256"]
    # Reuse: the same pair maps to the same directory.
    assert registry.prepare(model_id)[0] == model_dir


def test_different_base_snapshots_get_different_directories(tmp_path, no_gpu_check):
    export = make_export(tmp_path / "export")
    dir_a = FastH3OmniRegistry([str(export)], str(make_base(tmp_path / "base_rev_a")),
                               str(tmp_path / "cache")).prepare(str(export))[0]
    dir_b = FastH3OmniRegistry([str(export)], str(make_base(tmp_path / "base_rev_b")),
                               str(tmp_path / "cache")).prepare(str(export))[0]
    assert dir_a != dir_b
    assert (dir_a / "vae").resolve() != (dir_b / "vae").resolve()


def test_missing_base_component_fails_clearly(tmp_path, no_gpu_check):
    export = make_export(tmp_path / "export")
    base = make_base(tmp_path / "base", skip=("audio_vae", ))
    registry = FastH3OmniRegistry([str(export)], str(base), str(tmp_path / "cache"))
    with pytest.raises(OmniWeightsError, match="Missing checkpoint component.*audio_vae"):
        registry.prepare(str(export))


def test_prepare_rejects_unconfigured_model(registry):
    with pytest.raises(OmniWeightsError, match="not a FastH3-Omni export configured"):
        registry.prepare("/somewhere/else")


# --- Hardware ----------------------------------------------------------------


def test_tile_128_needs_data_center_blackwell():
    verdict = fasth3_omni._hardware_verdict(128, [("NVIDIA RTX 6000 Ada Generation", (8, 9))], True)
    assert verdict["supported"] is False
    assert "sm_89" in verdict["reason"] and "no Triton fallback" in verdict["reason"]


def test_tile_128_needs_the_kernel_extension():
    verdict = fasth3_omni._hardware_verdict(128, [("NVIDIA B200", (10, 0))], False)
    assert verdict["supported"] is False and "block_sparse_attn_sm100a" in verdict["reason"]


def test_tile_128_on_b200_with_kernel():
    assert fasth3_omni._hardware_verdict(128, [("NVIDIA B200", (10, 0))] * 2, True)["supported"] is True


def test_smaller_tiles_have_a_triton_route():
    assert fasth3_omni.hardware_status(64)["supported"] is True


# --- Errors -------------------------------------------------------------------


@pytest.mark.parametrize("exc,category", [
    (OmniHardwareError("x"), "kernel"),
    (OmniWeightsError("x"), "weights"),
    (FileNotFoundError("Missing checkpoint component: /b/vae"), "weights"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "memory"),
    (RuntimeError("Worker failed: block_sparse_attn_sm100a.is_supported returned False"), "kernel"),
    (ValueError("This FastH3 PDD checkpoint runs exactly 8 transformer forwards"), "parameters"),
    (RuntimeError("worker raised ValueError: bad request"), "parameters"),
    (RuntimeError("connection reset"), "runtime"),
])
def test_classify_error(exc, category):
    assert classify_error(exc) == category


# --- Catalogue ----------------------------------------------------------------


def test_catalog_capabilities_replace_name_matching(registry, tmp_path):
    registered = [{"id": "MiniMaxAI/MiniMax-H3", "workload_types": ["t2v", "i2v"]},
                  {"id": "FastVideo/FastVideo-FastH3-8-Step-V2", "workload_types": ["t2v", "i2v"]},
                  {"id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "workload_types": ["t2v"]}]
    i2v = {e["id"]: e for e in build_catalog(registered, registry, "i2v")}
    assert "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" not in i2v
    assert i2v["MiniMaxAI/MiniMax-H3"]["capabilities"]["references"]["required"] is False
    assert i2v["MiniMaxAI/MiniMax-H3"]["capabilities"]["end_frame"] is True
    # V2 is T2AV-only: no references, whatever its name says.
    assert i2v["FastVideo/FastVideo-FastH3-8-Step-V2"]["capabilities"]["references"] is None
    omni = i2v[str(tmp_path / "omni_export")]
    assert omni["config_source"] == "server"
    assert omni["capabilities"]["references"]["required"] is True
    assert omni["capabilities"]["start_frame"] is False
    assert omni["capabilities"]["num_gpus"] == [1, 2, 4, 7, 8]
    assert omni["recipe"]["contract"]["vsa_tile_size"] == 128
    assert omni["recipe_defaults"]["num_frames"] == 124
    # Ref2VA only: Omni is not offered for text-to-video.
    assert str(tmp_path / "omni_export") not in {e["id"] for e in build_catalog(registered, registry, "t2v")}


# --- Persistence, edit, duplicate ----------------------------------------------


@pytest.fixture
def runner_factory(tmp_path, registry):

    def make():
        return JobRunner(output_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"),
                         database=Database(tmp_path / "studio.db"), fasth3_omni=registry)

    return make


def _create_omni(runner, registry, tmp_path, **over):
    model_id = str(tmp_path / "omni_export")
    fields = apply_recipe(omni_fields(model_id=model_id, **over), registry.get(model_id).recipe())
    return runner.create_job(job_id=str(uuid.uuid4()),
                             **{k: fields[k]
                                for k in JobRunner.CONFIG_FIELDS if k in fields})


def test_recipe_and_reference_order_survive_restart(runner_factory, registry, tmp_path):
    job = _create_omni(runner_factory(), registry, tmp_path)
    restored = runner_factory().get_job(job.id)
    assert restored.references == REFS
    assert restored.recipe["contract_sha256"] == job.recipe["contract_sha256"]
    assert restored.num_inference_steps == 8


def test_edited_reference_order_survives_restart(runner_factory, registry, tmp_path):
    runner = runner_factory()
    job = _create_omni(runner, registry, tmp_path)
    reordered = [REFS[1], REFS[0], REFS[2]]
    runner.update_job_config(job.id, {"references": reordered, "seed": 7, "dit_layerwise_offload": False})
    restored = runner_factory().get_job(job.id)
    assert restored.references == reordered
    assert restored.seed == 7


def test_duplicate_keeps_order_and_recipe(runner_factory, registry, tmp_path):
    runner = runner_factory()
    job = _create_omni(runner, registry, tmp_path)
    dup = runner.duplicate_job(job.id, str(uuid.uuid4()))
    assert dup.references == REFS and dup.references is not job.references
    assert dup.recipe == job.recipe and dup.recipe is not job.recipe
    assert runner_factory().get_job(dup.id).recipe == job.recipe


def test_old_database_is_migrated(tmp_path):
    """A database from before this change gains the new columns and keeps its jobs."""
    db_path = tmp_path / "old.db"
    Database(db_path)
    conn = sqlite3.connect(db_path)
    for col in ("recipe_json", "error_category", "dit_layerwise_offload"):
        conn.execute(f"ALTER TABLE jobs DROP COLUMN {col}")
    conn.execute("INSERT INTO jobs (id, model_id, prompt, created_at) VALUES ('old', 'm', 'p', 0)")
    conn.commit()
    conn.close()
    row = Database(db_path).get_job("old")
    assert row["recipe"] == "" and row["error_category"] is None
    columns = {r[1] for r in sqlite3.connect(db_path).execute("PRAGMA table_info(jobs)")}
    assert {"recipe_json", "error_category", "dit_layerwise_offload"} <= columns


# --- Run path ---------------------------------------------------------------------


class FakeGenerator:

    def __init__(self):
        self.calls = []

    def generate_video(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["output_path"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["output_path"]) / "out.mp4").write_bytes(b"mp4")


def test_omni_run_uses_composed_model_and_recipe(runner_factory, registry, tmp_path, monkeypatch):
    runner = runner_factory()
    job = _create_omni(runner, registry, tmp_path, num_gpus=2)
    fake = FakeGenerator()
    loads = []

    def fake_get(model_id, workload_type, num_gpus, **kwargs):
        loads.append((model_id, workload_type, num_gpus, kwargs))
        return fake

    monkeypatch.setattr(runner, "_get_or_create_generator", fake_get)
    monkeypatch.setattr("fastvideo_studio.job_runner.hardware_status", lambda tile: {"supported": True, "reason": ""})
    monkeypatch.setattr("fastvideo_studio.job_runner._build_h3_references", lambda refs: [r["source"] for r in refs])
    runner._run_inference_job(job)

    assert job.status is JobStatus.COMPLETED, job.error
    model_dir, workload, num_gpus, kwargs = loads[0]
    assert Path(model_dir).parent == tmp_path / "cache"
    assert (workload, num_gpus) == ("i2v", 2)
    assert kwargs["override_pipeline_cls_name"] == "MiniMaxH3Ref2VAModularPipeline"
    assert kwargs["vsa_sparsity"] is None          # FastVideo takes the trained value from the contract
    assert kwargs["recipe_key"] == job.recipe["contract_sha256"]
    assert (kwargs["use_fsdp_inference"], kwargs["tp_size"], kwargs["sp_size"]) == (True, 1, 2)
    call = fake.calls[0]
    assert call["num_inference_steps"] == 8 and call["guidance_scale"] == 1.0 and call["fps"] == 24
    assert call["references"] == [r["source"] for r in REFS]   # order preserved end to end
    assert job.recipe["model_dir"] == model_dir
    assert runner_factory().get_job(job.id).recipe["model_dir"] == model_dir


def test_omni_run_fails_fast_on_unsupported_hardware(runner_factory, registry, tmp_path, monkeypatch):
    runner = runner_factory()
    job = _create_omni(runner, registry, tmp_path)
    monkeypatch.setattr("fastvideo_studio.job_runner.hardware_status",
                        lambda tile: {"supported": False, "reason": "found RTX 6000 Ada (sm_89)"})
    monkeypatch.setattr(runner, "_get_or_create_generator",
                        lambda *a, **k: pytest.fail("weights must not load on unsupported hardware"))
    runner._run_inference_job(job)
    assert job.status is JobStatus.FAILED
    assert job.error_category == "kernel"
    assert "sm_89" in job.error
    restored = runner_factory().get_job(job.id)
    assert restored.error_category == "kernel"
    # Nothing was composed either.
    assert not (tmp_path / "cache").exists()


def test_omni_run_reports_missing_weights(runner_factory, registry, tmp_path, monkeypatch):
    runner = runner_factory()
    job = _create_omni(runner, registry, tmp_path)
    monkeypatch.setattr("fastvideo_studio.job_runner.hardware_status", lambda tile: {"supported": True, "reason": ""})
    (tmp_path / "h3_base" / "vae").rmdir()
    runner._run_inference_job(job)
    assert job.status is JobStatus.FAILED
    assert job.error_category == "weights"
    assert "vae" in job.error
