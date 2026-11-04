"""Require prepared cookbook assets without importing the model runtime."""

import json
from pathlib import Path

PREPARE = "Run npm ci --prefix docs && npm run build:catalog --prefix docs from the repository root."


def check_assets(directory: Path) -> None:
    """Check the browser bundle, notices, and every catalog advertised by the index."""
    for name in ("cookbook-config.js", "cookbook-config.LICENSE.txt", "index.json"):
        if not (directory / name).is_file():
            raise RuntimeError(f"Missing cookbook asset: {directory / name}. {PREPARE}")
    try:
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        urls = [deployment["catalog_url"] for model in index["models"] for deployment in model["deployments"]]
        if not urls or any(not isinstance(url, str) for url in urls):
            raise ValueError("expected nonempty catalog URLs")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid cookbook index: {directory / 'index.json'}. {PREPARE}") from error
    for url in urls:
        path = (directory / url).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise RuntimeError(f"Missing cookbook catalog: {url}. {PREPARE}")


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    check_assets(root / "assets/cookbook-config")
    print(f"Cookbook assets verified in {root}.")
