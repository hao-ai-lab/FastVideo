"""Regression checks for source-only API rendering, without importing model code."""
import ast
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

class Generator:
    """A configurable generator."""

    def __init__(self, count: int = 0):
        """Initialize the generator.

        Args:
            count: Number of videos to generate.
        """
        self.count = count

    def run(self):
        """Generate the videos."""

class Empty:
    """A class without an explicit constructor."""
''')
        (root / "public.py").write_text("from sample import Generator\n")
        loader = griffe.GriffeLoader(search_paths=[root], allow_inspection=False, docstring_parser="google")
        self.module = loader.load("sample")
        self.public = loader.load("public")
        loader.resolve_aliases(implicit=True, external=False)

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

    def test_example_controls_edit_literals_without_imports_and_use_javascript_offsets(self):
        source = '''raise RuntimeError("Never execute this example 🥣")
PROMPT = "A bowl"
generator = VideoGenerator.from_pretrained("Example/Model", num_gpus=1)
parser.add_argument("--num-frames", type=int, default=124)
parser.add_argument("--fps", type=float, default=24)
generator.generate_video(height=480, prompt=PROMPT)
'''
        controls = renderer.source_controls(source)
        self.assertEqual({item["name"] for item in controls}, {"prompt", "num_gpus", "num_frames", "height", "fps"})
        self.assertFalse(next(item for item in controls if item["name"] == "fps")["integer"])
        frames = next(item for item in controls if item["name"] == "num_frames")
        self.assertEqual(frames["flag"], "--num-frames")
        for item in controls:
            for start, end in item["spans"]:
                literal = source.encode("utf-16-le")[start * 2:end * 2].decode("utf-16-le")
                self.assertEqual(ast.literal_eval(literal), item["value"])

    def test_class_includes_constructor_documentation_without_a_separate_method(self):
        for obj, path in [(self.module["Generator"], "sample.Generator"),
                          (self.public["Generator"], "public.Generator")]:
            with self.subTest(path=path):
                rendered = renderer.symbol(obj, path, source=False)
                self.assertIn("A configurable generator.", rendered)
                self.assertEqual(rendered.count("Number of videos to generate."), 1)
                self.assertIn('<dt><code>count: int</code></dt>', rendered)
                self.assertNotIn('__init__', rendered)
                self.assertIn(f'id="{path}.run"', rendered)
        self.assertIn("A class without an explicit constructor.",
                      renderer.symbol(self.module["Empty"], "sample.Empty", source=False))

    def test_overview_resolves_module_symbol_and_public_alias_references(self):
        rendered = renderer.api_overview("[Package][sample]\n\n[`sample.Generator`][]", self.module)
        self.assertIn('href="/api/sample#sample"', rendered)
        self.assertIn('href="/api/sample#sample.Generator"', rendered)
        rendered = renderer.api_overview("[`public.Generator`][]", self.public)
        self.assertIn('href="/api/public#public.Generator"', rendered)

    def test_overview_converts_guide_paths_and_preserves_external_links(self):
        rendered = renderer.api_overview(
            "[Guide](../getting_started/v1_api.md?mode=full#configuration)\n\n"
            "[Overview](../training/index.md)\n\n[External](https://example.com/guide.md)", self.module)
        self.assertIn('href="/getting_started/v1_api?mode=full#configuration"', rendered)
        self.assertIn('href="/training"', rendered)
        self.assertIn('href="https://example.com/guide.md"', rendered)


if __name__ == "__main__":
    unittest.main()
