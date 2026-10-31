# SPDX-License-Identifier: Apache-2.0
"""Prompt dataset + refiner output parsing for PromptRL.

Datasets are raw prompt files (JSONL or Parquet) with a required
``prompt`` field and optional ``id`` / ``reward_tag`` fields.  Generated
videos are always scored against the *original* prompt, never the
refined one.

Refiner completions must contain ``<answer>...</answer>``.  Valid
refinements and retained originals receive format reward 1; malformed
refinements receive format reward 0 and fall back to the original
prompt for video generation (the completion itself is preserved so it
still receives its negative language-model signal).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

SampleKind = Literal["original", "refined"]

REFINEMENT_PROMPT_TEMPLATE_V1 = (
    "You are an expert prompt engineer for text-to-video generation. "
    "Rewrite the user's prompt to be more detailed, cinematic, and "
    "visually grounded while preserving its meaning. Respond with the "
    "rewritten prompt inside <answer>...</answer> tags only.\n"
    "User prompt: {prompt}")

SUPPORTED_TEMPLATE_VERSIONS = ("v1",)


def render_refinement_prompt(prompt: str, *, template_version: str = "v1") -> str:
    """Render the instruction template used to query the refiner."""
    if template_version == "v1":
        return REFINEMENT_PROMPT_TEMPLATE_V1.format(prompt=prompt)
    raise ValueError(f"Unsupported refiner template_version: {template_version!r}")


@dataclass(frozen=True, slots=True)
class ParsedCompletion:
    """Result of parsing one raw refiner completion."""

    raw: str
    refined_prompt: str
    format_valid: bool


def parse_answer_tag(completion: str) -> ParsedCompletion:
    """Extract ``<answer>...</answer>`` from a raw refiner completion.

    A completion is format-valid when it contains exactly one non-empty
    answer block.  The refined prompt is the stripped answer content.
    """
    matches = _ANSWER_RE.findall(completion)
    if len(matches) != 1:
        return ParsedCompletion(raw=completion, refined_prompt="", format_valid=False)
    refined = matches[0].strip()
    if not refined:
        return ParsedCompletion(raw=completion, refined_prompt="", format_valid=False)
    return ParsedCompletion(raw=completion, refined_prompt=refined, format_valid=True)


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One original prompt row."""

    prompt: str
    sample_id: str
    reward_tag: str


@dataclass(frozen=True, slots=True)
class GroupAssignment:
    """Per-rank role assignment inside one PromptRL group."""

    rank: int
    kind: SampleKind
    # Whether this rank's completion counts toward the refiner loss.
    refiner_participation: bool


def group_assignments(*, group_size: int, retained_originals: int) -> list[GroupAssignment]:
    """Rank -> role layout for one replicated-prompt group.

    Ranks ``[0, retained_originals)`` are retained originals: they
    generate video from the original prompt and receive format reward 1,
    but contribute zero refiner advantage.  Remaining ranks generate
    from sampled refiner outputs.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if not 0 < retained_originals < group_size:
        raise ValueError(f"retained_originals must be within (0, {group_size}), "
                         f"got {retained_originals}")
    assignments: list[GroupAssignment] = []
    for rank in range(group_size):
        retained = rank < retained_originals
        assignments.append(
            GroupAssignment(
                rank=rank,
                kind="original" if retained else "refined",
                refiner_participation=not retained,
            ))
    return assignments



class PromptDataset:
    """In-memory raw prompt dataset loaded from JSONL or Parquet."""

    def __init__(
        self,
        records: list[PromptRecord],
        *,
        prompt_key: str = "prompt",
        id_key: str = "id",
        reward_tag_key: str = "reward_tag",
    ) -> None:
        if not records:
            raise ValueError("PromptDataset requires at least one record")
        self.records = list(records)
        self.prompt_key = prompt_key
        self.id_key = id_key
        self.reward_tag_key = reward_tag_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PromptRecord:
        return self.records[int(index) % len(self.records)]

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        prompt_key: str = "prompt",
        id_key: str = "id",
        reward_tag_key: str = "reward_tag",
    ) -> PromptDataset:
        records: list[PromptRecord] = []
        for position, row in enumerate(rows):
            prompt = row.get(prompt_key)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Prompt row {position} is missing a non-empty "
                                 f"{prompt_key!r} field")
            raw_id = row.get(id_key)
            sample_id = str(raw_id) if raw_id is not None else f"row-{position}"
            raw_tag = row.get(reward_tag_key)
            reward_tag = str(raw_tag) if raw_tag is not None else ""
            records.append(PromptRecord(prompt=prompt.strip(),
                                        sample_id=sample_id,
                                        reward_tag=reward_tag))
        return cls(records, prompt_key=prompt_key, id_key=id_key, reward_tag_key=reward_tag_key)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        prompt_key: str = "prompt",
        id_key: str = "id",
        reward_tag_key: str = "reward_tag",
    ) -> PromptDataset:
        """Load a raw prompt dataset from ``.jsonl``/``.json``/``.parquet``.

        Parquet inputs may be a single file, a directory of parquet
        files, or a glob pattern.
        """
        resolved = Path(path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"Prompt dataset not found: {resolved}")
        suffix = resolved.suffix.lower()
        if suffix in (".jsonl", ".json"):
            rows = _load_jsonl(resolved)
        elif suffix == ".parquet" or resolved.is_dir():
            rows = _load_parquet(resolved)
        else:
            raise ValueError(f"Unsupported prompt dataset format: {resolved} "
                             "(expected .jsonl/.json/.parquet or a parquet directory)")
        return cls.from_rows(rows,
                             prompt_key=prompt_key,
                             id_key=id_key,
                             reward_tag_key=reward_tag_key)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON list in {path}")
            return [dict(row) for row in payload]
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files under {path}")
    else:
        files = [path]
    rows: list[dict[str, Any]] = []
    for file in files:
        table = pq.read_table(file)
        rows.extend(table.to_pylist())
    return rows
