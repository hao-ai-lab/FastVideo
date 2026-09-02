# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fastvideo.dataset.h3_rest_cache import (
    H3RESTCacheDataset,
    H3_REST_CACHE_SCHEMA_VERSION,
    sha256_file,
    validate_h3_rest_cache,
)
from fastvideo.train.methods.knowledge_distillation.h3_rest_utils import canonical_json_hash


def _write_cache(root: Path) -> None:
    (root / "prompts").mkdir(parents=True)
    (root / "trajectories").mkdir()
    prompt_path = root / "prompts" / "p0.pt"
    torch.save(
        {
            "text_embedding": torch.zeros(1, 4, 8, dtype=torch.bfloat16),
            "text_attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        prompt_path,
    )
    manifest = []
    for candidate in range(2):
        trajectory_path = root / "trajectories" / f"p0_c{candidate}.pt"
        torch.save(
            {
                "anchor_states": torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
                + candidate,
                "anchor_timesteps": torch.tensor(
                    [1000.0, 750.0, 500.0, 250.0, 0.0], dtype=torch.float32
                ),
            },
            trajectory_path,
        )
        manifest.append(
            {
                "trajectory_id": f"p0-c{candidate}",
                "prompt_id": "p0",
                "prompt": "a test prompt",
                "candidate_index": candidate,
                "seed": 10 + candidate,
                "prompt_file": str(prompt_path.relative_to(root)),
                "prompt_sha256": sha256_file(prompt_path),
                "prompt_bytes": prompt_path.stat().st_size,
                "trajectory_file": str(trajectory_path.relative_to(root)),
                "trajectory_sha256": sha256_file(trajectory_path),
                "trajectory_bytes": trajectory_path.stat().st_size,
                "reward_scores": {"quality": float(candidate)},
                "reward_advantages": {"quality": -1.0 if candidate == 0 else 1.0},
                "mixed_advantage": -1.0 if candidate == 0 else 1.0,
            }
        )
    manifest_path = root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in manifest:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    metadata = {
        "schema_version": H3_REST_CACHE_SCHEMA_VERSION,
        "num_prompts": 1,
        "samples_per_prompt": 2,
        "num_trajectories": 2,
        "num_segments": 4,
        "student_timesteps": [1000, 750, 500, 250, 0],
        "reward_names": ["quality"],
        "manifest_sha256": sha256_file(manifest_path),
        "provenance": {"unit_test": True},
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (root / "COMPLETE").write_text(metadata["fingerprint"] + "\n", encoding="utf-8")


def test_validate_cache_and_expand_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_cache(tmp_path)
    summary = validate_h3_rest_cache(tmp_path, verify_file_hashes=True)
    assert summary.num_examples == 8
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_sp_world_size", lambda: 1)
    monkeypatch.setattr("fastvideo.dataset.h3_rest_cache.get_world_rank", lambda: 0)
    dataset = H3RESTCacheDataset(tmp_path, seed=4)
    batch = dataset.__getitems__([5])
    assert batch["rest_segment_index"].item() == 1
    assert tuple(batch["trajectory_current"].shape) == (1, 3)
    assert tuple(batch["text_embedding"].shape) == (1, 4, 8)
    assert batch["info_list"][0]["candidate_index"] == 1


def test_cache_rejects_partial_or_mutated_artifacts(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    (tmp_path / "COMPLETE").unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        validate_h3_rest_cache(tmp_path)
    _write_cache(tmp_path)
    with (tmp_path / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_h3_rest_cache(tmp_path)


def test_cache_rejects_wrong_student_grid(tmp_path: Path) -> None:
    _write_cache(tmp_path)
    with pytest.raises(ValueError, match="timestep mismatch"):
        validate_h3_rest_cache(
            tmp_path,
            expected_student_timesteps=[1000, 800, 600, 400, 0],
        )
