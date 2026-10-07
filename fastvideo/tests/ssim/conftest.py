import contextlib
import os

import fastvideo.envs as envs
from fastvideo.tests.ssim.reference_utils import (
    get_output_quality_tier, )
from fastvideo.tests.ssim.reference_videos_cli import (
    HF_REPO_ENV_KEY,
    ensure_reference_videos_available,
)


def pytest_addoption(parser):
    parser.addoption(
        "--ssim-full-quality",
        action="store_true",
        default=False,
        help=("Use *_FULL_QUALITY_PARAMS for SSIM tests. "
              "Default keeps the original CI-friendly params."),
    )
    parser.addoption(
        "--ssim-reference-repo",
        default="",
        help=("HF repo id for SSIM reference videos "
              f"(overrides {HF_REPO_ENV_KEY})."),
    )
    parser.addoption(
        "--skip-ssim-reference-download",
        action="store_true",
        default=False,
        help="Skip auto-download of missing SSIM reference videos from HF.",
    )
    parser.addoption(
        "--ssim-bootstrap-mode",
        action="store_true",
        default=False,
        help="Treat missing SSIM references as draft-reference bootstrap cases.",
    )


def pytest_configure(config):
    if config.getoption("--ssim-full-quality"):
        envs.FASTVIDEO_TEST_SSIM_FULL_QUALITY.set(True)

    repo_id = config.getoption("--ssim-reference-repo")
    if repo_id:
        envs.FASTVIDEO_TEST_SSIM_REFERENCE_HF_REPO.set(repo_id)

    if config.getoption("--ssim-bootstrap-mode"):
        # Kept for the whole pytest session and restored when pytest exits.
        session_env = contextlib.ExitStack()
        config.add_cleanup(session_env.close)
        session_env.enter_context(envs.override_external("FASTVIDEO_SSIM_BOOTSTRAP_MODE", "1"))

    skip_download = config.getoption("--skip-ssim-reference-download")
    skip_download = skip_download or envs.FASTVIDEO_TEST_SSIM_SKIP_REFERENCE_DOWNLOAD.get()

    if not skip_download:
        ensure_reference_videos_available(
            repo_id=repo_id or None,
            quality_tier=get_output_quality_tier(),
        )


def pytest_collection_modifyitems(config, items):
    """Optionally keep only tests with a matching model_id parameter."""
    model_id = os.environ.get("FASTVIDEO_SSIM_MODEL_ID")
    if not model_id:
        return

    selected = []
    deselected = []
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            deselected.append(item)
            continue
        if callspec.params.get("model_id") == model_id:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected
