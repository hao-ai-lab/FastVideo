"""Regression checks for source-only API rendering, without importing model code."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import griffe

SPEC = importlib.util.spec_from_file_location("python_reference", Path(__file__).parents[1] / "scripts/prepare-python-docs.py")
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class PythonReferenceTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        (root / "sample.py").write_text('''raise RuntimeError("This module must never be imported")

def generate(prompt: str, /, count: int = 0, *, audio: bool = False, **kwargs) -> str:
    """Generate an output.

    Args:
        prompt: Text describing the video.
        count: Number of outputs.
        audio: Include audio.
    """
    return {"outputs": []}["outputs"](prompt)
''')
        self.module = griffe.load("sample", search_paths=[root], allow_inspection=False, docstring_parser="google")

    def test_source_is_read_without_executing_module(self):
        self.assertIn("generate", self.module.members)
        rendered = renderer.symbol(self.module["generate"], "sample.generate")
        self.assertIn('id="sample.generate"', rendered)
        self.assertIn("Text describing the video", rendered)
        self.assertIn("<summary>Source</summary>", rendered)

    def test_signature_preserves_parameter_kinds_zero_and_false(self):
        self.assertEqual(renderer.signature(self.module["generate"]),
                         "generate(prompt: str, /, count: int = 0, *, audio: bool = False, **kwargs) -> str")

    def test_source_brackets_cannot_be_misread_as_markdown_links(self):
        rendered = renderer.symbol(self.module["generate"], "sample.generate")
        self.assertNotIn('[]', rendered)
        self.assertIn("&#91;", rendered)


if __name__ == "__main__":
    unittest.main()
