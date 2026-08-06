# SPDX-License-Identifier: Apache-2.0
"""Strict MAGI-2 preview and refiner state-structure parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(os.environ.get("MAGI2_OFFICIAL_ROOT", REPO_ROOT.parent / "MAGI-2-preview"))
WEIGHTS_ROOT = Path(os.environ.get("MAGI2_WEIGHTS_ROOT", REPO_ROOT / "official_weights" / "magi2"))

if not (OFFICIAL_ROOT / "inference" / "model" / "magi2_preview.py").is_file():
    pytest.skip(f"Official MAGI-2 checkout is missing: {OFFICIAL_ROOT}", allow_module_level=True)
if not (WEIGHTS_ROOT / "preview" / "model.safetensors.index.json").is_file():
    pytest.skip(f"Official MAGI-2 weights are missing: {WEIGHTS_ROOT}", allow_module_level=True)


def _structure_dump_script(component: str, implementation: str) -> str:
    """Create an isolated process script that instantiates one model on meta."""
    if implementation == "official":
        model_setup = {
            "preview": """
                from inference.common.magi2_config import load_config
                from inference.model.magi2_preview import Transformer
                config = load_config(str(official_root / "configs" / "magi2_preview.json"))
                with torch.device("meta"):
                    model = Transformer(config.arch_config, ep_size=1)
            """,
            "refiner": """
                from inference.common.magi2_config import load_config
                from inference.model.magi2_refiner import Transformer
                config = load_config(str(official_root / "configs" / "magi2_refiner.json"))
                with torch.device("meta"):
                    model = Transformer(config.magi2_refiner_arch_config)
            """,
        }[component]
        path_setup = "sys.path.insert(0, str(official_root))"
    else:
        model_setup = {
            "preview": """
                from fastvideo.configs.models.dits.magi2 import Magi2PreviewVideoConfig
                from fastvideo.models.dits.magi2 import Magi2PreviewDiT
                with torch.device("meta"):
                    model = Magi2PreviewDiT(config=Magi2PreviewVideoConfig())
            """,
            "refiner": """
                from fastvideo.configs.models.dits.magi2 import Magi2RefinerVideoConfig
                from fastvideo.models.dits.magi2_refiner import Magi2RefinerDiT
                with torch.device("meta"):
                    model = Magi2RefinerDiT(config=Magi2RefinerVideoConfig())
            """,
        }[component]
        path_setup = "sys.path.insert(0, str(repo_root))"

    setup_script = textwrap.dedent(
        f"""
        import json
        import os
        import pathlib
        import sys
        import torch

        repo_root = pathlib.Path({str(REPO_ROOT)!r})
        official_root = pathlib.Path({str(OFFICIAL_ROOT)!r})
        {path_setup}
        os.environ["SKIP_LOAD_MODEL"] = "1"
        os.environ["MAGI2_DISABLE_MAGI_COMPILE"] = "1"
        os.environ["MAGI_COMPILE_COMPILE_MODE"] = "NONE"
        """
    )
    dump_script = textwrap.dedent(
        """
        structure = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in model.state_dict().items()
        }
        print("MAGI2_STRUCTURE=" + json.dumps(structure, sort_keys=True))
        """
    )
    return setup_script + textwrap.dedent(model_setup) + dump_script


def _dump_structure(component: str, implementation: str) -> dict[str, dict]:
    """Run one model definition in isolation and parse its state metadata."""
    completed = subprocess.run(
        [sys.executable, "-c", _structure_dump_script(component, implementation)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{implementation} {component} structure dump failed.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("MAGI2_STRUCTURE=")]
    if len(lines) != 1:
        raise AssertionError(f"Expected one structure record, received stdout:\n{completed.stdout}")
    return json.loads(lines[0].removeprefix("MAGI2_STRUCTURE="))


@pytest.mark.parametrize("component", ["preview", "refiner"])
def test_magi2_transformer_state_structure_matches_official(component: str) -> None:
    """Match every state name, shape, and dtype to the official definition."""
    official_structure = _dump_structure(component, "official")
    fastvideo_structure = _dump_structure(component, "fastvideo")
    assert fastvideo_structure == official_structure

    index_path = WEIGHTS_ROOT / component / "model.safetensors.index.json"
    checkpoint_keys = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    assert set(fastvideo_structure) == checkpoint_keys
