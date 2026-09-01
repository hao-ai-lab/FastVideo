# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODAL_WRAPPER = ROOT / "examples/train/rvm_h3/modal_h3_rvm.py"
PORTABLE_RUNNER = ROOT / "examples/train/rvm_h3/12_run_portable_smoke.sh"


def test_modal_file_is_only_a_thin_transport_wrapper() -> None:
    source = MODAL_WRAPPER.read_text(encoding="utf-8")

    assert len(source.splitlines()) < 220
    assert "12_run_portable_smoke.sh" in source
    for forbidden in (
        "_apply_runtime_fixes",
        "videoalign_ta:",
        "num_latent_t",
        "training.optimizer",
        "prepare_prompts.py",
        "uv pip install",
    ):
        assert forbidden not in source


def test_portable_runner_owns_cloud_agnostic_orchestration() -> None:
    source = PORTABLE_RUNNER.read_text(encoding="utf-8")

    for required in (
        "00_install_current_env.sh",
        "01_download_models.sh",
        "02_prepare_dataset.sh",
        "03_public_inference_smoke.sh",
        "03_preflight_1gpu.sh",
        "run_rvm_training",
        "verify_clean_source.py",
    ):
        assert required in source
