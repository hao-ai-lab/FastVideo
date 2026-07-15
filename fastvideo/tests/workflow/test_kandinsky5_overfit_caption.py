# SPDX-License-Identifier: Apache-2.0
"""Regression test: preprocess_kandinsky5_overfit.py must not silently
truncate a string caption to its first character.

The documented ``videos2caption.json`` schema
(``docs/training/data_preprocess.md``) stores ``cap`` as a plain string
(e.g. ``"Ocean waves at sunset..."``), while some producers (this repo's
own e2e test fixtures, mirroring ``preprocess_hunyuan_overfit.py``) store a
non-empty list of caption variants instead. Indexing unconditionally with
``item["cap"][0]`` silently takes the first *character* of a string caption
("O") instead of erroring -- ``get_caption`` accepts and validates both
forms once at ingestion.

Pure logic test -- no GPU, no model load.
"""
from __future__ import annotations

import pytest

from fastvideo.pipelines.preprocess.preprocess_kandinsky5_overfit import get_caption


def test_get_caption_accepts_plain_string():
    assert get_caption({"path": "a.mp4", "cap": "Ocean waves at sunset..."}) == "Ocean waves at sunset..."


def test_get_caption_accepts_nonempty_list():
    assert get_caption({"path": "a.mp4", "cap": ["a synthetic test clip"]}) == "a synthetic test clip"


def test_get_caption_uses_first_list_element():
    assert get_caption({"path": "a.mp4", "cap": ["first", "second"]}) == "first"


@pytest.mark.parametrize("bad_cap", [
    "",
    [],
    [""],
    None,
    123,
    {"nested": "dict"},
])
def test_get_caption_rejects_invalid_values(bad_cap):
    with pytest.raises(ValueError):
        get_caption({"path": "a.mp4", "cap": bad_cap})
