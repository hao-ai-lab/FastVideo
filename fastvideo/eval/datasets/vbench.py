"""VBench prompt corpus.

Single source of truth: upstream's ``VBench_full_info.json`` (946 entries,
each with ``prompt_en``, a ``dimension`` list, optional ``auxiliary_info``
keyed by dimension).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastvideo.eval.datasets.base import PromptDataset
from fastvideo.eval.datasets.registry import register_dataset

# VBench's official sampling protocol: 5 generations per prompt, except
# temporal_flickering which requires 25 (averaging over 5 is too noisy
# for a high-frequency-noise metric). See upstream prompts/README.md.
TEMPORAL_FLICKERING_SAMPLES = 25
DEFAULT_SAMPLES = 5

_FULL_INFO_REL = "fastvideo/third_party/eval/vbench/vbench/VBench_full_info.json"


def _locate_full_info() -> Path:
    env = os.environ.get("VBENCH_FULL_INFO_JSON")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VBENCH_FULL_INFO_JSON={env} does not point at a file")
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / _FULL_INFO_REL
        if candidate.is_file():
            return candidate
        if (ancestor / ".git").exists():
            break
    raise FileNotFoundError("Could not locate VBench_full_info.json. Initialize the upstream "
                            "submodule (`git submodule update --init "
                            "fastvideo/third_party/eval/vbench`) or set VBENCH_FULL_INFO_JSON.")


@register_dataset("vbench")
class VBenchPromptDataset(PromptDataset):
    """VBench prompts filtered by evaluation dimension.

    Args:
        dimensions: List of dimension names, or ``"all"``. Unknown
            dimensions raise ``ValueError``.
        full_info_path: Optional override for ``VBench_full_info.json``;
            defaults to autodetection.

    A prompt that belongs to several requested dimensions is yielded once;
    its ``dimensions`` list carries all matches so the scorer can route.
    """

    description = ("VBench (Vchitect) prompt corpus, 946 prompts across 16 "
                   "evaluation dimensions.")
    supports_dimensions = True

    def __init__(
        self,
        dimensions: list[str] | str = "all",
        full_info_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        path = Path(full_info_path) if full_info_path else _locate_full_info()
        with path.open() as f:
            entries = json.load(f)

        all_dims = sorted({d for e in entries for d in e["dimension"]})
        if dimensions == "all":
            self.dimensions: list[str] = all_dims
        else:
            unknown = set(dimensions) - set(all_dims)
            if unknown:
                raise ValueError(f"Unknown VBench dimensions: {sorted(unknown)}. "
                                 f"Available: {all_dims}")
            self.dimensions = list(dimensions)

        wanted = set(self.dimensions)
        for entry in entries:
            relevant = [d for d in entry["dimension"] if d in wanted]
            if not relevant:
                continue
            n = (TEMPORAL_FLICKERING_SAMPLES if "temporal_flickering" in relevant else DEFAULT_SAMPLES)

            # Strip the outer {dim_name: ...} wrapper from upstream's aux
            # schema so every metric reads its inputs from a flat dict.
            #
            # This unwraps exactly one level — the dimension key. Whatever
            # shape lives inside is the metric's contract:
            #
            #   color:                {"color": {"color": "red"}}
            #     → flat: {"color": "red"}                  (scalar)
            #
            #   object_class:         {"object_class": {"object": "person"}}
            #     → flat: {"object": "person"}              (scalar)
            #
            #   multiple_objects:     {"multiple_objects": {"object": "a and b"}}
            #     → flat: {"object": "a and b"}             (scalar)
            #
            #   spatial_relationship: {"spatial_relationship":
            #                            {"spatial_relationship":
            #                                {"object_a": ..., "object_b": ...,
            #                                 "relationship": ...}}}
            #     → flat: {"spatial_relationship": {object_a,object_b,relationship}}
            #
            # Note the spatial_relationship case keeps a nested inner dict
            # by design — upstream double-wraps it, the SpatialRelationship
            # metric reads ``aux["spatial_relationship"]`` expecting that
            # inner dict. Don't "simplify" the wrapping away.
            raw_aux = entry.get("auxiliary_info") or {}
            flat_aux: dict = {}
            for v in raw_aux.values():
                if isinstance(v, dict):
                    flat_aux.update(v)

            self._rows.append({
                "prompt": entry["prompt_en"],
                "n_samples": n,
                "dimensions": relevant,
                "auxiliary_info": flat_aux,
            })
        self.full_info_path = path
