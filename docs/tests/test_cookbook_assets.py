"""Lightweight docs-hook checks; no MkDocs or model imports are needed."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("cookbook_assets", Path(__file__).parents[1] / "check_cookbook_assets.py")
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)

GENERATE_SPEC = importlib.util.spec_from_file_location("generate_examples",
                                                       Path(__file__).parents[1] / "generate_examples.py")
assert GENERATE_SPEC is not None and GENERATE_SPEC.loader is not None
examples = importlib.util.module_from_spec(GENERATE_SPEC)
sys.modules[GENERATE_SPEC.name] = examples
GENERATE_SPEC.loader.exec_module(examples)


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


class CookbookCacheBustTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.page = self.directory / "page.md"

    def test_prose_references_do_not_require_cache_version(self):
        self.page.write_text("Family pages load `docs/assets/cookbook-recipes.json` through `cookbook.js`.",
                             encoding="utf-8")
        examples.validate_cookbook_cache_bust(self.directory, 15)

    def test_current_loader_urls_pass_with_either_quote_style(self):
        self.page.write_text('''<div data-recipes="../../assets/cookbook-recipes.json?v=15"></div>
<div data-recipes='../../assets/cookbook-recipes.json?v=15'></div>''',
                             encoding="utf-8")
        examples.validate_cookbook_cache_bust(self.directory, 15)

    def test_stale_loader_fails_even_when_current_version_is_also_present(self):
        self.page.write_text('''<div data-recipes="../../assets/cookbook-recipes.json?v=14"></div>
<div data-recipes="../../assets/cookbook-recipes.json?v=15"></div>''',
                             encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recipe JSON cache-bust must be cookbook-recipes.json\\?v=15"):
            examples.validate_cookbook_cache_bust(self.directory, 15)

    def test_unversioned_loader_fails(self):
        self.page.write_text('<div data-recipes="../../assets/cookbook-recipes.json"></div>', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recipe JSON cache-bust must be cookbook-recipes.json\\?v=15"):
            examples.validate_cookbook_cache_bust(self.directory, 15)


if __name__ == "__main__":
    unittest.main()
