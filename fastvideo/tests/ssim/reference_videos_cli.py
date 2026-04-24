# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")
# Additional artefact types stored under the same reference folders. Latent
# tensors (`.pt`) back the latent-slice regression tests used by flaky
# pixel-space models (e.g. LTX-2 distilled). Keeping them in the same upload
# flow means seeding a new test only requires one HF round-trip.
LATENT_EXTENSIONS = (".pt",)
REFERENCE_EXTENSIONS = VIDEO_EXTENSIONS + LATENT_EXTENSIONS
HF_TOKEN_ENV_KEYS = ("HF_API_KEY", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")
HF_REPO_ENV_KEY = "FASTVIDEO_SSIM_REFERENCE_HF_REPO"
HF_REPO_TYPE_ENV_KEY = "FASTVIDEO_SSIM_REFERENCE_HF_REPO_TYPE"

DEFAULT_REPO_ID = "FastVideo/ssim-reference-videos"
DEFAULT_REPO_TYPE = "dataset"
DEFAULT_OUTPUT_QUALITY_TIER = "default"
FULL_OUTPUT_QUALITY_TIER = "full_quality"
ALL_OUTPUT_QUALITY_TIERS = "all"
REFERENCE_VIDEOS_DIRNAME = "reference_videos"
DEFAULT_DEVICE_REFERENCE_FOLDER = "L40S_reference_videos"
QUALITY_TIERS = (
    DEFAULT_OUTPUT_QUALITY_TIER,
    FULL_OUTPUT_QUALITY_TIER,
)


def _ssim_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_repo_id() -> str:
    return os.environ.get(HF_REPO_ENV_KEY, DEFAULT_REPO_ID)


def _default_repo_type() -> str:
    return os.environ.get(HF_REPO_TYPE_ENV_KEY, DEFAULT_REPO_TYPE)


def _iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _iter_reference_files(root: Path) -> Iterable[Path]:
    """Yield video files and latent `.pt` references under `root`.

    Used by copy-local and the "has local references" marker probe so that
    latent-only tests (no mp4) still satisfy readiness checks.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in REFERENCE_EXTENSIONS:
            yield path


def _discover_reference_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.endswith("_reference_videos")]
    return sorted(dirs)


@contextlib.contextmanager
def _exclusive_download_lock(lock_path: Path):
    """Best-effort cross-process lock for reference video download/copy."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_file:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _merge_downloaded_reference_content(
    staging_dir: Path,
    local_dir: Path,
) -> int:
    copied_roots = 0

    tier_root = staging_dir / REFERENCE_VIDEOS_DIRNAME
    if tier_root.exists():
        dst_tier_root = local_dir / REFERENCE_VIDEOS_DIRNAME
        shutil.copytree(tier_root, dst_tier_root, dirs_exist_ok=True)
        copied_roots += 1

    # Legacy default layout support: <staging>/<GPU>_reference_videos/...
    legacy_reference_dirs = _discover_reference_dirs(staging_dir)
    for legacy_dir in legacy_reference_dirs:
        dst_legacy_dir = local_dir / legacy_dir.name
        shutil.copytree(legacy_dir, dst_legacy_dir, dirs_exist_ok=True)
        copied_roots += 1

    return copied_roots


def _resolve_device_reference_folder(
    *,
    explicit_device_folder: str | None,
    reference_dir: Path | None,
) -> str:
    if explicit_device_folder and explicit_device_folder.strip():
        return explicit_device_folder.strip()
    if reference_dir is not None and reference_dir.name.endswith("_reference_videos"):
        return reference_dir.name
    return DEFAULT_DEVICE_REFERENCE_FOLDER


def _reference_tier_root(base_dir: Path, quality_tier: str) -> Path:
    return base_dir / REFERENCE_VIDEOS_DIRNAME / quality_tier


def _resolve_quality_tiers(quality_tier: str) -> list[str]:
    if quality_tier == ALL_OUTPUT_QUALITY_TIERS:
        return list(QUALITY_TIERS)
    if quality_tier not in QUALITY_TIERS:
        raise ValueError(f"Unsupported quality tier: {quality_tier}")
    return [quality_tier]


def _discover_reference_dirs_for_tier(
    base_dir: Path,
    quality_tier: str,
) -> list[Path]:
    tier_root = _reference_tier_root(base_dir, quality_tier)
    tier_dirs = _discover_reference_dirs(tier_root)
    if tier_dirs:
        return tier_dirs
    if quality_tier == DEFAULT_OUTPUT_QUALITY_TIER:
        # Keep supporting legacy default refs at <ssim_dir>/<GPU>_reference_videos.
        return _discover_reference_dirs(base_dir)
    return []


def _has_local_reference_videos(base_dir: Path, quality_tier: str) -> bool:
    marker_path = base_dir / REFERENCE_VIDEOS_DIRNAME / quality_tier / f".download_complete_{quality_tier}"
    if not marker_path.exists():
        return False
    tier_root = _reference_tier_root(base_dir, quality_tier)
    for ref_dir in _discover_reference_dirs(tier_root):
        for _ in _iter_reference_files(ref_dir):
            return True
    return False


def _get_hf_token() -> str | None:
    for key in HF_TOKEN_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    return None


def _load_hf_sdk():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for download/upload.\nInstall with: pip install huggingface_hub"
        ) from exc
    return HfApi, snapshot_download


def copy_generated_to_reference(
    generated_dir: Path,
    reference_dir: Path,
    *,
    dry_run: bool = False,
) -> int:
    if not generated_dir.exists():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    copied = 0
    for ref_file in _iter_reference_files(generated_dir):
        rel = ref_file.relative_to(generated_dir)
        dst = reference_dir / rel
        if dry_run:
            print(f"[dry-run] {ref_file} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_file, dst)
            print(f"Copied: {rel}")
        copied += 1
    return copied


def download_reference_videos(
    *,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    quality_tiers: Sequence[str] | None = None,
    device_folders: Sequence[str] | None = None,
    token: str | None = None,
) -> None:
    _, snapshot_download = _load_hf_sdk()

    local_dir.mkdir(parents=True, exist_ok=True)
    tiers = tuple(quality_tiers or QUALITY_TIERS)
    for quality_tier in tiers:
        if quality_tier not in QUALITY_TIERS:
            raise ValueError(f"Unsupported quality tier: {quality_tier}")
    allow_patterns: list[str] = []
    for quality_tier in tiers:
        tier_prefix = f"{REFERENCE_VIDEOS_DIRNAME}/{quality_tier}"
        if device_folders:
            allow_patterns.extend(f"{tier_prefix}/{folder}/**" for folder in device_folders)
        else:
            allow_patterns.append(f"{tier_prefix}/*_reference_videos/**")

        # Keep compatibility for older HF layout where default refs lived at
        # repo root as <GPU>_reference_videos.
        if quality_tier == DEFAULT_OUTPUT_QUALITY_TIER:
            if device_folders:
                allow_patterns.extend(f"{folder}/**" for folder in device_folders)
            else:
                allow_patterns.append("*_reference_videos/**")

    # Work around long-path issues in some huggingface_hub versions:
    # avoid `local_dir` mode and download into a short cache dir instead,
    # then merge files into local_dir.
    with tempfile.TemporaryDirectory(prefix="fv2-ssim-cache-") as tmp_cache_dir:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            cache_dir=tmp_cache_dir,
            token=token,
            allow_patterns=allow_patterns,
        )
        staging_dir = Path(snapshot_path)
        copied_roots = _merge_downloaded_reference_content(
            staging_dir=staging_dir,
            local_dir=local_dir,
        )
        if copied_roots == 0:
            raise RuntimeError(
                "HF download completed but no reference_videos content was found in downloaded artifacts."
            )
        for quality_tier in tiers:
            marker_path = local_dir / REFERENCE_VIDEOS_DIRNAME / quality_tier / f".download_complete_{quality_tier}"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.touch()


def upload_reference_videos(
    *,
    repo_id: str,
    repo_type: str,
    reference_dirs_by_tier: Sequence[tuple[str, Path]],
    token: str,
    private: bool,
    model_id: str | None = None,
    force: bool = False,
) -> None:
    HfApi, _ = _load_hf_sdk()
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    try:
        existing_repo_files = set(
            api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception:
        # Fresh repo or list failure — treat as empty so upload can proceed.
        existing_repo_files = set()

    for quality_tier, reference_dir in reference_dirs_by_tier:
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
        base_in_repo = f"{REFERENCE_VIDEOS_DIRNAME}/{quality_tier}/{reference_dir.name}"
        if model_id:
            folder_path = reference_dir / model_id
            if not folder_path.exists():
                raise FileNotFoundError(
                    f"Model subfolder not found for upload: {folder_path}")
            path_in_repo = f"{base_in_repo}/{model_id}"
        else:
            folder_path = reference_dir
            path_in_repo = base_in_repo

        conflicts = sorted(
            f for f in existing_repo_files
            if f.startswith(f"{path_in_repo}/") or f == path_in_repo)
        if conflicts and not force:
            preview = "\n".join(f"  - {c}" for c in conflicts[:10])
            more = f"\n  ... and {len(conflicts) - 10} more" if len(conflicts) > 10 else ""
            raise RuntimeError(
                f"Refusing to overwrite existing HF files under {path_in_repo} "
                f"({len(conflicts)} file(s) already present):\n{preview}{more}\n"
                f"Re-run with --force to overwrite.")

        target_desc = f"{reference_dir.name}/{model_id}" if model_id else reference_dir.name
        print(f"Uploading {target_desc} ({quality_tier}) to {repo_id}/{path_in_repo} ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            token=token,
        )


def _resolve_upload_reference_dirs(
    *,
    base_dir: Path,
    quality_tier: str,
    explicit_reference_dirs: Sequence[str],
    device_folders: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    tiers = _resolve_quality_tiers(quality_tier)
    selected_device_folders = set(device_folders or [])

    if explicit_reference_dirs:
        if len(tiers) != 1:
            raise RuntimeError("--reference-dir requires a single --quality-tier (default or full_quality).")
        resolved_tier = tiers[0]
        return [(resolved_tier, Path(p).resolve()) for p in explicit_reference_dirs]

    resolved: list[tuple[str, Path]] = []
    for tier in tiers:
        for reference_dir in _discover_reference_dirs_for_tier(base_dir, tier):
            if selected_device_folders and reference_dir.name not in selected_device_folders:
                continue
            resolved.append((tier, reference_dir.resolve()))
    return resolved


def ensure_reference_videos_available(
    *,
    local_dir: Path | None = None,
    repo_id: str | None = None,
    repo_type: str | None = None,
    quality_tier: str = DEFAULT_OUTPUT_QUALITY_TIER,
) -> bool:
    """Return True if downloaded from HF, False if already present locally."""
    if quality_tier not in QUALITY_TIERS:
        raise ValueError(f"Unsupported quality tier: {quality_tier}")
    target_dir = local_dir or _ssim_dir()
    lock_path = target_dir / ".reference_videos_download.lock"
    with _exclusive_download_lock(lock_path):
        if _has_local_reference_videos(target_dir, quality_tier):
            print(f"Reference videos ({quality_tier}) already available at {target_dir}")
            return False

        resolved_repo_id = repo_id or _default_repo_id()
        resolved_repo_type = repo_type or _default_repo_type()
        if not resolved_repo_id:
            raise RuntimeError(
                f"No local reference videos found and no HF repo configured.\nSet {HF_REPO_ENV_KEY} or pass --repo-id."
            )

        print(f"Repo ID: {resolved_repo_id}")
        print(f"Quality tier: {quality_tier}")
        print(f"No local {quality_tier} reference videos found under {target_dir}. Starting download...")
        try:
            download_reference_videos(
                repo_id=resolved_repo_id,
                repo_type=resolved_repo_type,
                local_dir=target_dir,
                quality_tiers=[quality_tier],
            )
            print(f"Download completed for {quality_tier} reference videos.")
        except Exception as exc:
            print(f"ERROR: Failed to download {quality_tier} reference videos from {resolved_repo_id}.")
            print(
                f"Suggested command to retry: "
                f"python fastvideo/tests/ssim/reference_videos_cli.py download "
                f"--quality-tier {quality_tier}"
            )
            raise

        if not _has_local_reference_videos(target_dir, quality_tier):
            raise RuntimeError(f"HF download completed but no {quality_tier} *_reference_videos content found.")
        return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reference_videos_cli",
        description=(
            "Manage SSIM reference videos locally and on Hugging Face.\n"
            "This tool can copy generated videos into reference folders,\n"
            "download references from a public HF repo, and upload updates."
        ),
        epilog=(
            "Examples:\n"
            "  # Copy default generated videos into local L40S refs\n"
            "  python fastvideo/tests/ssim/reference_videos_cli.py copy-local \\\n"
            "    --quality-tier default \\\n"
            "    --device-folder L40S_reference_videos \\\n"
            "    --reference-dir fastvideo/tests/ssim/reference_videos/default/L40S_reference_videos\n\n"
            "  # Download both default and full-quality references from HF\n"
            "  python fastvideo/tests/ssim/reference_videos_cli.py download \\\n"
            "    --repo-id FastVideo/ssim-reference-videos\n\n"
            "  # Upload both tiers and all GPU folders (fails if token is unset)\n"
            "  python fastvideo/tests/ssim/reference_videos_cli.py upload \\\n"
            "    --repo-id FastVideo/ssim-reference-videos"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    copy_parser = subparsers.add_parser(
        "copy-local",
        help="Copy generated videos into a local reference folder.",
    )
    copy_parser.add_argument(
        "--quality-tier",
        choices=(DEFAULT_OUTPUT_QUALITY_TIER, FULL_OUTPUT_QUALITY_TIER),
        default=DEFAULT_OUTPUT_QUALITY_TIER,
        help="Generated quality tier to copy from.",
    )
    copy_parser.add_argument(
        "--generated-dir",
        type=Path,
        default=None,
        help=("Source generated directory. Default: <ssim_dir>/generated_videos/<quality-tier>/<device-folder>"),
    )
    copy_parser.add_argument(
        "--device-folder",
        default=None,
        help=("GPU folder name (e.g., H200_reference_videos) used to build default generated/reference paths."),
    )
    copy_parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Destination reference directory.",
    )
    copy_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy operations without writing files.",
    )

    download_parser = subparsers.add_parser(
        "download",
        help="Download reference videos from Hugging Face.",
    )
    download_parser.add_argument(
        "--repo-id",
        default=_default_repo_id(),
        help=f"HF repo id. Default: env {HF_REPO_ENV_KEY} or {DEFAULT_REPO_ID}",
    )
    download_parser.add_argument(
        "--repo-type",
        default=_default_repo_type(),
        help=f"HF repo type. Default: env {HF_REPO_TYPE_ENV_KEY} or {DEFAULT_REPO_TYPE}",
    )
    download_parser.add_argument(
        "--local-dir",
        type=Path,
        default=_ssim_dir(),
        help=(
            "Local SSIM directory where references are stored under "
            "reference_videos/<quality-tier>/<GPU>_reference_videos."
        ),
    )
    download_parser.add_argument(
        "--quality-tier",
        choices=(
            DEFAULT_OUTPUT_QUALITY_TIER,
            FULL_OUTPUT_QUALITY_TIER,
            ALL_OUTPUT_QUALITY_TIERS,
        ),
        default=ALL_OUTPUT_QUALITY_TIERS,
        help="Quality tier to download (default: all).",
    )
    download_parser.add_argument(
        "--device-folder",
        action="append",
        default=[],
        help="Specific reference folder to fetch (repeatable), e.g. L40S_reference_videos",
    )

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload local reference folders to Hugging Face.",
    )
    upload_parser.add_argument(
        "--repo-id",
        default=_default_repo_id(),
        help=f"HF repo id. Default: env {HF_REPO_ENV_KEY} or {DEFAULT_REPO_ID}",
    )
    upload_parser.add_argument(
        "--repo-type",
        default=_default_repo_type(),
        help=f"HF repo type. Default: env {HF_REPO_TYPE_ENV_KEY} or {DEFAULT_REPO_TYPE}",
    )
    upload_parser.add_argument(
        "--reference-dir",
        action="append",
        default=[],
        help=(
            "Reference directory to upload (repeatable). "
            "Requires single --quality-tier. "
            "Default: discover folders under --base-dir for chosen tier(s)."
        ),
    )
    upload_parser.add_argument(
        "--quality-tier",
        choices=(
            DEFAULT_OUTPUT_QUALITY_TIER,
            FULL_OUTPUT_QUALITY_TIER,
            ALL_OUTPUT_QUALITY_TIERS,
        ),
        default=ALL_OUTPUT_QUALITY_TIERS,
        help="Quality tier to upload (default: all).",
    )
    upload_parser.add_argument(
        "--device-folder",
        action="append",
        default=[],
        help=("Specific GPU reference folder to upload (repeatable), e.g. H200_reference_videos."),
    )
    upload_parser.add_argument(
        "--base-dir",
        type=Path,
        default=_ssim_dir(),
        help=("Base SSIM directory that contains reference_videos/<quality-tier>/<GPU>_reference_videos."),
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create/use a private repo instead of public.",
    )
    upload_parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Restrict upload to a single model subfolder "
            "(reference_videos/<tier>/<device>/<model_id>). "
            "Use when seeding references for a single new test."),
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow overwriting files that already exist at the target path on "
            "Hugging Face. Off by default so seeding a new test cannot "
            "clobber existing references."),
    )

    ensure_parser = subparsers.add_parser(
        "ensure",
        help="Download refs only when the selected local quality tier is missing.",
    )
    ensure_parser.add_argument(
        "--repo-id",
        default=_default_repo_id(),
        help=f"HF repo id. Default: env {HF_REPO_ENV_KEY} or {DEFAULT_REPO_ID}",
    )
    ensure_parser.add_argument(
        "--repo-type",
        default=_default_repo_type(),
        help=f"HF repo type. Default: env {HF_REPO_TYPE_ENV_KEY} or {DEFAULT_REPO_TYPE}",
    )
    ensure_parser.add_argument(
        "--local-dir",
        type=Path,
        default=_ssim_dir(),
        help=("Local SSIM directory that should contain reference_videos/<quality-tier>/<GPU>_reference_videos."),
    )
    ensure_parser.add_argument(
        "--quality-tier",
        choices=(
            DEFAULT_OUTPUT_QUALITY_TIER,
            FULL_OUTPUT_QUALITY_TIER,
        ),
        default=DEFAULT_OUTPUT_QUALITY_TIER,
        help="Quality tier to ensure locally.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "copy-local":
        device_folder = _resolve_device_reference_folder(
            explicit_device_folder=args.device_folder,
            reference_dir=args.reference_dir,
        )
        generated_dir = args.generated_dir
        if generated_dir is None:
            tier_root = _ssim_dir() / "generated_videos" / args.quality_tier
            tier_device_dir = tier_root / device_folder
            if tier_device_dir.exists():
                generated_dir = tier_device_dir
            else:
                # Backward compatibility for pre-device-folder layout.
                generated_dir = tier_root
        reference_dir = (
            args.reference_dir
            if args.reference_dir is not None
            else (_ssim_dir() / REFERENCE_VIDEOS_DIRNAME / args.quality_tier / device_folder)
        )
        copied = copy_generated_to_reference(
            generated_dir=generated_dir,
            reference_dir=reference_dir,
            dry_run=args.dry_run,
        )
        print(f"Done. {'Would copy' if args.dry_run else 'Copied'} {copied} video files.")
        if not args.dry_run and copied > 0:
            marker = (
                _ssim_dir() / REFERENCE_VIDEOS_DIRNAME / args.quality_tier / f".download_complete_{args.quality_tier}"
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        return 0

    if args.command == "download":
        quality_tiers = _resolve_quality_tiers(args.quality_tier)
        download_reference_videos(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            quality_tiers=quality_tiers,
            device_folders=args.device_folder or None,
        )
        print("Download complete for quality tier(s): " + ", ".join(quality_tiers) + ".")
        return 0

    if args.command == "upload":
        token = _get_hf_token()
        if token is None:
            raise RuntimeError(
                "Hugging Face API key is required for upload. Set HF_API_KEY, HUGGINGFACE_HUB_TOKEN, or HF_TOKEN."
            )

        reference_dirs_by_tier = _resolve_upload_reference_dirs(
            base_dir=args.base_dir.resolve(),
            quality_tier=args.quality_tier,
            explicit_reference_dirs=args.reference_dir,
            device_folders=args.device_folder or None,
        )
        if not reference_dirs_by_tier:
            raise RuntimeError(
                f"No *_reference_videos directories found for the selected quality tier(s) under {args.base_dir}"
            )

        upload_reference_videos(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            reference_dirs_by_tier=reference_dirs_by_tier,
            token=token,
            private=args.private,
            model_id=args.model_id,
            force=args.force,
        )
        print("Upload complete.")
        return 0

    if args.command == "ensure":
        downloaded = ensure_reference_videos_available(
            local_dir=args.local_dir,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            quality_tier=args.quality_tier,
        )
        if downloaded:
            print(f"{args.quality_tier} reference videos were missing and have been downloaded.")
        else:
            print(f"{args.quality_tier} reference videos already exist locally.")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
