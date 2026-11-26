# SPDX-License-Identifier: Apache-2.0
"""Import the pinned official oracle without importing its optional MLLM stack.

All numerical modules are loaded unchanged from the official checkout. Only
the models package's eager __init__ is bypassed; attention and parallel ops
are the official implementations, including their single-GPU SDPA backend.
"""
import importlib
import importlib.util
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "reference_repos/bernini"
WEIGHTS = ROOT / "official_weights/bernini-r-1.3b"
REFERENCE_SHA = "e6c2cf11843470b08491745947abcba9c3c6b079"
ORACLE_DEPS = REFERENCE / ".oracle-deps"


def official_scheduler():
    """Load the unmodified scheduler from the reference's pinned dependency.

    Bernini pins diffusers==0.35.2. Newer versions changed flow timesteps;
    using the host's latest diffusers would silently change the oracle.
    """
    path = ORACLE_DEPS / "diffusers/schedulers/scheduling_unipc_multistep.py"
    if not path.is_file():
        raise FileNotFoundError("Install diffusers==0.35.2 into reference_repos/bernini/.oracle-deps")
    name = "diffusers.schedulers._bernini_unipc_0352"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].UniPCMultistepScheduler


@lru_cache(maxsize=1)
def official_modules():
    revision = subprocess.check_output(["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_SHA:
        raise ValueError(f"Bernini reference must be pinned to {REFERENCE_SHA}, got {revision}")
    subprocess.run(["git", "-C", str(REFERENCE), "diff", "--exit-code", "HEAD", "--", "bernini"],
                   check=True, capture_output=True, text=True)
    for name, folder in (("_bernini_oracle", REFERENCE / "bernini"),
                         ("_bernini_oracle.models", REFERENCE / "bernini/models")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(folder)]
            sys.modules[name] = package
    return (importlib.import_module("_bernini_oracle.models.transformer_wan"),
            importlib.import_module("_bernini_oracle.models.wan_diffusion"))
