from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

LOCAL_DEMO_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    repo_id: str
    source_filename: str
    target_filename: str
    env_var: str


CLASSIFIERS = {
    "nsfw": ClassifierSpec(
        name="nsfw",
        repo_id="allenai/dolma-jigsaw-fasttext-bigrams-nsfw",
        source_filename="model.bin",
        target_filename="jigsaw_fasttext_bigrams_nsfw_final.bin",
        env_var="LTX2_NSFW_CLASSIFIER_PATH",
    ),
    "hatespeech": ClassifierSpec(
        name="hatespeech",
        repo_id="allenai/dolma-jigsaw-fasttext-bigrams-hatespeech",
        source_filename="model.bin",
        target_filename="jigsaw_fasttext_bigrams_hatespeech_final.bin",
        env_var="LTX2_HATESPEECH_CLASSIFIER_PATH",
    ),
}


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def default_output_dir() -> Path:
    raw_value = os.getenv("LTX2_CLASSIFIER_DIR", str(LOCAL_DEMO_DIR))
    return expand_path(raw_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the fastText safety classifiers used by "
            "gradio_local_demo_ltx2_3.py."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help=(
            "Directory where the demo looks for classifiers. Defaults to "
            "LTX2_CLASSIFIER_DIR or the local Gradio demo directory."
        ),
    )
    parser.add_argument(
        "--classifier",
        choices=sorted(CLASSIFIERS),
        nargs="+",
        default=sorted(CLASSIFIERS),
        help="Subset of classifiers to download. Defaults to both.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing classifier files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for authenticated downloads.",
    )
    return parser.parse_args()


def materialize_download(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    shutil.copy2(source, destination)


def download_classifier(
    spec: ClassifierSpec,
    output_dir: Path,
    cache_dir: Path | None,
    force: bool,
    token: str | None,
) -> Path:
    destination = output_dir / spec.target_filename
    if destination.is_file() and not force:
        print(f"[skip] {spec.name}: {destination}")
        return destination

    print(
        f"[download] {spec.name}: "
        f"{spec.repo_id}/{spec.source_filename}"
    )
    cached_path = Path(
        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.source_filename,
            cache_dir=cache_dir,
            force_download=force,
            token=token,
        )
    )
    materialize_download(cached_path, destination)
    print(f"[ready] {spec.name}: {destination}")
    return destination


def print_summary(
    selected_specs: list[ClassifierSpec],
    output_dir: Path,
    downloaded_paths: list[Path],
) -> None:
    print("\nDownloaded classifier paths:")
    for path in downloaded_paths:
        print(f"  - {path}")

    print("\nThese filenames are auto-discovered by gradio_local_demo_ltx2_3.py.")

    if output_dir != default_output_dir():
        print(
            "\nBecause you used a custom output directory, point the demo to it:"
        )
        print(f'  export LTX2_CLASSIFIER_DIR="{output_dir}"')

    for spec, path in zip(selected_specs, downloaded_paths, strict=True):
        print(f'  export {spec.env_var}="{path}"')


def main() -> None:
    args = parse_args()
    output_dir = expand_path(args.output_dir).resolve()
    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = expand_path(args.cache_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = [CLASSIFIERS[name] for name in args.classifier]
    downloaded_paths = [
        download_classifier(
            spec=spec,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force=args.force,
            token=args.token,
        )
        for spec in selected_specs
    ]
    print_summary(selected_specs, output_dir, downloaded_paths)


if __name__ == "__main__":
    main()
