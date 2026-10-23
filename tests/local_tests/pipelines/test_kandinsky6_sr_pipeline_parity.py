# SPDX-License-Identifier: Apache-2.0
"""Real-weights end-to-end parity of the Kandinsky6 SR pipeline against a previous FastVideo checkout.

The previous implementation ran its own Euler / pi-Flow loops; the current pipeline steps the bundle's scheduler. The
two differ only in scheduler arithmetic: the Euler update ``dt * velocity`` is fp32 (the previous loop multiplied in
bf16) and ``PiflowScheduler`` computes its schedule in float32 with the final segment ending at ``eps``. With those two
conventions patched back, the outputs were bitwise identical on GB200; as shipped, the observed PSNR was 50.9-56.5 dB.

Environment (the test skips without them):
    KANDINSKY6_SR_MODELS         directory holding ``Kandinsky-6.0-VSR-5s-Diffusers`` and
                                 ``Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``
    KANDINSKY6_SR_VIDEO          a short low-resolution clip (e.g. 17 frames at 384x208)
    KANDINSKY6_SR_PREVIOUS_ROOT  checkout of the previous implementation (PR head 12be8fdd)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_HELPER = Path(__file__).resolve().parents[1] / "kandinsky6_sr" / "sr_pipeline_capture.py"
_MIN_PSNR_DB = 48.0
_BUNDLES = {
    "flow_matching": ("Kandinsky-6.0-VSR-5s-Diffusers", [("x2", 2.0, None), ("x2_25", 2.25, None), ("x2_steps2", 2.0, 2),
                                                       ("x4", 4.0, None)]),
    "distilled": ("Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", [("x2", 2.0, None), ("x4", 4.0, None)]),
}


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _capture(root: Path, model: Path, cases: list[dict], out: Path, legacy: bool) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, str(_HELPER), "--model", str(model), "--cases", json.dumps(cases), "--out", str(out)]
    if legacy:
        command.append("--legacy-steps")
    subprocess.run(command, cwd=root, env=env, check=True)
    assert Path((out / "fastvideo_path.txt").read_text()) == (root / "fastvideo").resolve()


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


@pytest.mark.parametrize("bundle", list(_BUNDLES))
def test_pipeline_matches_the_previous_implementation(bundle, tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    models, video = Path(_env("KANDINSKY6_SR_MODELS")), _env("KANDINSKY6_SR_VIDEO")
    previous = Path(_env("KANDINSKY6_SR_PREVIOUS_ROOT"))
    name, case_specs = _BUNDLES[bundle]
    cases = [{"name": case, "video": video, "scale": scale, "steps": steps} for case, scale, steps in case_specs]
    current = Path(__file__).resolve().parents[3]
    _capture(current, models / name, cases, tmp_path / "current", legacy=False)
    _capture(previous, models / name, cases, tmp_path / "previous", legacy=True)
    for case in cases:
        got = np.load(tmp_path / "current" / f"{case['name']}.npy")
        expected = np.load(tmp_path / "previous" / f"{case['name']}.npy")
        assert got.shape == expected.shape, case
        psnr = _psnr(got, expected)
        print(f"{bundle}/{case['name']}: PSNR {psnr:.2f} dB, max diff {np.abs(got.astype(int) - expected).max()}")
        assert psnr >= _MIN_PSNR_DB, (case, psnr)
