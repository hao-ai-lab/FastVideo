# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
ltx_core_path = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
if ltx_core_path.exists() and str(ltx_core_path) not in sys.path:
    sys.path.insert(0, str(ltx_core_path))

from fastvideo.pipelines.stages.ltx2_denoising import _ltx2_sigmas


def test_ltx2_sigma_schedule_parity():
    try:
        from ltx_core.components.schedulers import LTX2Scheduler
    except ImportError as exc:
        pytest.skip(f"LTX-2 import failed: {exc}")

    scheduler = LTX2Scheduler()
    device = torch.device("cpu")

    for steps in (4, 8, 40):
        ref = scheduler.execute(steps=steps, latent=None).to(torch.float32)
        ours = _ltx2_sigmas(steps=steps, latent=None, device=device)
        assert_close(ref, ours, atol=1e-6, rtol=1e-6)

    latent = torch.randn(1, 128, 4, 8, 8)
    ref = scheduler.execute(steps=10, latent=latent).to(torch.float32)
    ours = _ltx2_sigmas(steps=10, latent=latent, device=device)
    assert_close(ref, ours, atol=1e-6, rtol=1e-6)
