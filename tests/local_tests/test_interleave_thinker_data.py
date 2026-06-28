# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastvideo.train.models.interleave_thinker import (
    load_critic_rl_records,
    load_critic_sft_records,
    load_interleave_dataset,
    load_planner_rl_records,
    load_planner_sft_records,
    normalize_critic_rl_record,
    resolve_interleave_image_path,
)


def test_load_planner_sft_records_normalizes_sharegpt_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "planner_sft.json").write_text(
        json.dumps([{
            "messages": [
                {
                    "role": "user",
                    "content": "draw a cat step by step"
                },
                {
                    "role": "assistant",
                    "content": "<answer>{\"execution_plan\": []}</answer>"
                },
            ],
            "images": ["planner/cat_step.png"],
        }]),
        encoding="utf-8",
    )

    records = load_planner_sft_records(data_dir, image_dir=image_dir)

    assert len(records) == 1
    assert records[0]["instruction"] == "draw a cat step by step"
    assert records[0]["response"] == '<answer>{"execution_plan": []}</answer>'
    assert records[0]["images"] == [str(image_dir / "planner/cat_step.png")]
    assert records[0]["input_image_paths"] == [str(image_dir / "planner/cat_step.png")]


def test_load_planner_rl_records_accepts_prompt_only_rows(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    data_path = tmp_path / "planner_rl.jsonl"
    data_path.write_text(
        json.dumps({
            "text_input": "draw a cat in three clear steps",
            "images": ["planner/start.png"],
            "plan_score": 0.75,
        }) + "\n",
        encoding="utf-8",
    )

    records = load_planner_rl_records(data_path, image_dir=image_dir)

    assert records[0]["instruction"] == "draw a cat in three clear steps"
    assert records[0]["images"] == [str(image_dir / "planner/start.png")]
    assert records[0]["input_image_paths"] == [str(image_dir / "planner/start.png")]
    assert records[0]["plan_score"] == 0.75


def test_load_critic_sft_records_adds_image_pair_aliases(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    data_path = tmp_path / "critic_sft.json"
    data_path.write_text(
        json.dumps({
            "records": [{
                "messages": [
                    {
                        "role": "user",
                        "content": "evaluate this edit"
                    },
                    {
                        "role": "assistant",
                        "content": "<answer>{\"previous_step_success\": true, \"refine_prompt\": \"keep it\"}</answer>"
                    },
                ],
                "images": ["before.png", "after.webp"],
                "rewritten_prompt": "make the cat orange",
            }]
        }),
        encoding="utf-8",
    )

    records = load_critic_sft_records(data_path, image_dir=image_dir)

    assert records[0]["origin_image_path"] == str(image_dir / "before.png")
    assert records[0]["edited_image_path"] == str(image_dir / "after.webp")
    assert records[0]["previous_image_path"] == str(image_dir / "before.png")
    assert records[0]["generated_image_path"] == str(image_dir / "after.webp")
    assert records[0]["previous_prompt"] == "make the cat orange"
    assert records[0]["response"].startswith("<answer>")


def test_load_critic_rl_records_normalizes_reward_fields_and_aliases(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    data_path = tmp_path / "critic_rl.jsonl"
    data_path.write_text(
        json.dumps({
            "origin_prompt": "draw a chair",
            "rewritten_prompt": "a wooden chair by a window",
            "origin_image_path": "chairs/original.jpg",
            "edited_image_path": "chairs/edited.png",
            "evaluation": {
                "success": False,
                "semantic_score": "6.5",
                "quality_score": 7,
            },
            "responses": ["<answer>{}</answer>"],
        }) + "\n",
        encoding="utf-8",
    )

    records = load_critic_rl_records(data_path, image_dir=image_dir)

    assert records[0]["origin_prompt"] == "draw a chair"
    assert records[0]["previous_prompt"] == "a wooden chair by a window"
    assert records[0]["rewritten_prompt"] == "a wooden chair by a window"
    assert records[0]["origin_image_path"] == str(image_dir / "chairs/original.jpg")
    assert records[0]["edited_image_path"] == str(image_dir / "chairs/edited.png")
    assert records[0]["previous_image_path"] == str(image_dir / "chairs/original.jpg")
    assert records[0]["generated_image_path"] == str(image_dir / "chairs/edited.png")
    assert records[0]["ground_truth"] == {
        "success": False,
        "semantics": 6.5,
        "quality": 7.0,
    }
    assert records[0]["responses"] == ["<answer>{}</answer>"]


def test_load_interleave_dataset_rejects_empty_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "critic_rl.jsonl"
    data_path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No critic_rl records"):
        load_interleave_dataset(data_path, kind="critic_rl")


def test_normalize_critic_rl_record_requires_ground_truth_success() -> None:
    with pytest.raises(ValueError, match="ground_truth requires boolean success"):
        normalize_critic_rl_record({
            "origin_prompt": "draw a chair",
            "previous_prompt": "a wooden chair",
            "origin_image_path": "before.png",
            "edited_image_path": "after.png",
            "ground_truth": {
                "semantics": 5
            },
        })


def test_resolve_interleave_image_path_validates_extension(tmp_path: Path) -> None:
    assert resolve_interleave_image_path("image.png", image_dir=tmp_path) == str(tmp_path / "image.png")

    with pytest.raises(ValueError, match="Unsupported image file extension"):
        resolve_interleave_image_path("not-an-image.txt", image_dir=tmp_path)
