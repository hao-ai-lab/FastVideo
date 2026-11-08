# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the token reduction flags in examples/inference/basic/basic_fasth3.py (no weights, no GPU).

basic_fasth3_8step.py, the FastH3 8-Step V2 example, reuses this parser and generator config.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples" / "inference" / "basic"


def _load_example(name: str, path: Path):
    # basic_fasth3_8step.py imports basic_fasth3 from its own directory.
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


fasth3 = _load_example("basic_fasth3_token_reduction_contract", EXAMPLES / "basic_fasth3.py")
eight_step = _load_example("basic_fasth3_8step_token_reduction_contract", EXAMPLES / "basic_fasth3_8step.py")


def _args(*overrides: str):
    return fasth3.parse_args(["--prompt", "a test prompt", *overrides])


def test_token_reduction_is_opt_in_experimental():
    assert "h3_token_reduction" not in fasth3.build_generator_config(_args()).pipeline.experimental
    config = fasth3.build_generator_config(_args("--token-reduction", "--token-reduction-blocks", "4:20"))
    assert config.pipeline.experimental["h3_token_reduction"] == {"enabled": True, "blocks": "4:20"}


@pytest.mark.parametrize("flags", [
    ("--token-reduction", "--token-reduction-blocks", "4:60"),
    ("--token-reduction", "--token-reduction-early", "4:30"),
    ("--token-reduction", "--token-reduction-scale", "3"),
    ("--token-reduction-scale", "0.5"),
])
def test_token_reduction_flags_are_checked_before_download(flags):
    with pytest.raises(SystemExit):
        _args(*flags)


def test_eight_step_example_accepts_token_reduction():
    args = eight_step.parse_args(["--prompt", "p", "--token-reduction", "--token-reduction-early", "0:0"])
    config = fasth3.build_generator_config(args)
    assert config.model_path == "FastVideo/FastVideo-FastH3-8-Step-V2"
    assert config.pipeline.experimental["h3_token_reduction"] == {"enabled": True, "early": "0:0"}
