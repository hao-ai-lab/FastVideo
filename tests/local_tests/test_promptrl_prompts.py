# SPDX-License-Identifier: Apache-2.0
"""Prompt dataset + refiner output parsing tests for PromptRL."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fastvideo.train.methods.rl.promptrl.prompts import (
    PromptDataset,
    group_assignments,
    parse_answer_tag,
    render_refinement_prompt,
)


class TestParseAnswerTag:
    def test_valid_completion(self):
        parsed = parse_answer_tag("Some text <answer> a cinematic cat </answer> trailing")
        assert parsed.format_valid
        assert parsed.refined_prompt == "a cinematic cat"

    def test_multiline_answer(self):
        parsed = parse_answer_tag("<answer>line one\nline two</answer>")
        assert parsed.format_valid
        assert parsed.refined_prompt == "line one\nline two"

    def test_missing_tags_is_malformed(self):
        parsed = parse_answer_tag("no tags at all")
        assert not parsed.format_valid
        assert parsed.refined_prompt == ""
        assert parsed.raw == "no tags at all"

    def test_empty_answer_is_malformed(self):
        parsed = parse_answer_tag("<answer>   </answer>")
        assert not parsed.format_valid

    def test_multiple_answer_blocks_are_malformed(self):
        parsed = parse_answer_tag("<answer>a</answer><answer>b</answer>")
        assert not parsed.format_valid


class TestGroupAssignments:
    def test_retention_ordering_originals_first(self):
        assignments = group_assignments(group_size=8, retained_originals=2)
        kinds = [a.kind for a in assignments]
        assert kinds == (["original"] * 2 + ["refined"] * 6)
        participation = [a.refiner_participation for a in assignments]
        assert participation == [False, False] + [True] * 6

    def test_invalid_layout_rejected(self):
        with pytest.raises(ValueError):
            group_assignments(group_size=8, retained_originals=8)
        with pytest.raises(ValueError):
            group_assignments(group_size=8, retained_originals=0)
        with pytest.raises(ValueError):
            group_assignments(group_size=0, retained_originals=0)


class TestPromptDataset:
    def test_jsonl_round_trip(self, tmp_path):
        rows = [
            {"prompt": "a cat", "id": "c1", "reward_tag": "animal"},
            {"prompt": "a dog", "id": "d1"},
            {"prompt": "  spaced  "},
        ]
        path = tmp_path / "prompts.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        dataset = PromptDataset.load(str(path))
        assert len(dataset) == 3
        assert dataset[0].prompt == "a cat"
        assert dataset[0].sample_id == "c1"
        assert dataset[0].reward_tag == "animal"
        assert dataset[1].reward_tag == ""  # optional field defaults
        assert dataset[2].sample_id == "row-2"  # positional fallback id
        assert dataset[2].prompt == "spaced"

    def test_parquet_round_trip(self, tmp_path):
        table = pa.table({
            "prompt": ["p1", "p2"],
            "id": ["1", "2"],
            "reward_tag": ["t", "t"],
        })
        path = tmp_path / "prompts.parquet"
        pq.write_table(table, path)
        dataset = PromptDataset.load(str(path))
        assert len(dataset) == 2
        assert dataset[1].prompt == "p2"
        assert dataset[1].reward_tag == "t"

    def test_missing_prompt_field_rejected(self):
        with pytest.raises(ValueError, match="prompt"):
            PromptDataset.from_rows([{"text": "wrong key"}])

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError):
            PromptDataset.from_rows([{"prompt": "   "}])

    def test_custom_keys(self):
        dataset = PromptDataset.from_rows(
            [{"caption": "hello", "uid": 7, "tag": "x"}],
            prompt_key="caption",
            id_key="uid",
            reward_tag_key="tag",
        )
        record = dataset[0]
        assert record.prompt == "hello"
        assert record.sample_id == "7"
        assert record.reward_tag == "x"

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptDataset.load(str(tmp_path / "nope.jsonl"))

    def test_render_template_contains_prompt_and_tags(self):
        rendered = render_refinement_prompt("my prompt", template_version="v1")
        assert "my prompt" in rendered
        assert "<answer>" in rendered
        with pytest.raises(ValueError):
            render_refinement_prompt("x", template_version="v99")
