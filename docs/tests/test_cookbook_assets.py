"""Lightweight docs-hook checks; no MkDocs or model imports are needed."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("cookbook_assets", Path(__file__).parents[1] / "check_cookbook_assets.py")
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class CookbookAssetTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "assets/cookbook-config"
        self.directory.mkdir(parents=True)

    def prepare(self) -> None:
        for name in ("cookbook-config.js", "cookbook-config.LICENSE.txt"):
            (self.directory / name).write_text("prepared", encoding="utf-8")
        (self.directory / "recipes/example").mkdir(parents=True)
        (self.directory / "recipes/example/rest.json").write_text("{}", encoding="utf-8")
        index = {"models": [{"deployments": [{"catalog_url": "recipes/example/rest.json"}]}]}
        (self.directory / "index.json").write_text(json.dumps(index), encoding="utf-8")

    def test_unprepared_build_reports_preparation_command(self):
        with self.assertRaisesRegex(RuntimeError, "npm run build:catalog --prefix docs"):
            assets.on_pre_build({"docs_dir": self.root})

    def test_prepared_source_and_published_assets_pass(self):
        self.prepare()
        assets.on_pre_build({"docs_dir": self.root})
        assets.on_post_build({"site_dir": self.root})

    def test_missing_catalog_and_notices_fail(self):
        self.prepare()
        catalog = self.directory / "recipes/example/rest.json"
        catalog.unlink()
        with self.assertRaisesRegex(RuntimeError, "recipes/example/rest.json"):
            assets.check_assets(self.directory)
        (self.directory / "cookbook-config.LICENSE.txt").unlink()
        with self.assertRaisesRegex(RuntimeError, "cookbook-config.LICENSE.txt"):
            assets.check_assets(self.directory)

    def test_malformed_index_is_actionable(self):
        self.prepare()
        (self.directory / "index.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Invalid cookbook index"):
            assets.check_assets(self.directory)


if __name__ == "__main__":
    unittest.main()
