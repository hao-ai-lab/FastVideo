import os

from fastvideo.tests.ssim.reference_utils import (
    FULL_QUALITY_ENV_VAR,
    get_output_quality_tier,
)
from fastvideo.tests.ssim.reference_videos_cli import (
    HF_REPO_ENV_KEY,
    ensure_reference_videos_available,
)


def pytest_addoption(parser):
    parser.addoption(
        "--ssim-full-quality",
        action="store_true",
        default=False,
        help=(
            "Use *_FULL_QUALITY_PARAMS for SSIM tests. "
            "Default keeps the original CI-friendly params."
        ),
    )
    parser.addoption(
        "--ssim-reference-repo",
        default=os.environ.get(HF_REPO_ENV_KEY, ""),
        help=(
            "HF repo id for SSIM reference videos "
            f"(overrides {HF_REPO_ENV_KEY})."
        ),
    )
    parser.addoption(
        "--skip-ssim-reference-download",
        action="store_true",
        default=False,
        help="Skip auto-download of missing SSIM reference videos from HF.",
    )


def pytest_configure(config):
    if config.getoption("--ssim-full-quality"):
        os.environ[FULL_QUALITY_ENV_VAR] = "1"

    repo_id = config.getoption("--ssim-reference-repo")
    if repo_id:
        os.environ[HF_REPO_ENV_KEY] = repo_id

    skip_download = config.getoption("--skip-ssim-reference-download")
    skip_download = skip_download or os.environ.get(
        "FASTVIDEO_SSIM_SKIP_REFERENCE_DOWNLOAD", ""
    ).strip().lower() in {"1", "true", "yes", "on"}

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
