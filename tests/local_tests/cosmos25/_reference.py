# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest

REFERENCE_COMMIT = "a2c298b0a3df3778b973fe65e9e58877b292d8a7"
SCALING_SHA256 = "04f7d06fb349e317dd4a3c1ea3f78978b85c66a693c31a854f032bfb1c469dd2"
DEFAULT_REFERENCE_ROOT = Path(__file__).resolve().parents[3] / "cosmos-predict2.5"


def reference_root() -> Path:
    return Path(os.environ.get("COSMOS25_OFFICIAL_REF_DIR", DEFAULT_REFERENCE_ROOT)).resolve()


def load_official_scaling_module() -> ModuleType:
    source = reference_root() / "cosmos_predict2/_src/predict2/modules/denoiser_scaling.py"
    if not source.is_file():
        pytest.skip("Cosmos Predict2.5 reference checkout is missing; set COSMOS25_OFFICIAL_REF_DIR")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SCALING_SHA256:
        pytest.fail(f"Official denoiser scaling source drifted from {REFERENCE_COMMIT}: {digest}")

    spec = importlib.util.spec_from_file_location("cosmos25_official_denoiser_scaling", source)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not load official scaling module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
