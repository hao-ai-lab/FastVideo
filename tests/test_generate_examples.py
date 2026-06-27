import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location("generate_examples", REPO_ROOT / "docs/generate_examples.py")
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
GENERATE_EXAMPLES = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(GENERATE_EXAMPLES)
rebase_relative_markdown_links = GENERATE_EXAMPLES.rebase_relative_markdown_links

CHECK_LINKS_SPEC = importlib.util.spec_from_file_location("check_docs_links", REPO_ROOT / "scripts/check_docs_links.py")
assert CHECK_LINKS_SPEC is not None and CHECK_LINKS_SPEC.loader is not None
CHECK_DOCS_LINKS = importlib.util.module_from_spec(CHECK_LINKS_SPEC)
CHECK_LINKS_SPEC.loader.exec_module(CHECK_DOCS_LINKS)
check_links = CHECK_DOCS_LINKS.check_links


def test_rebase_relative_markdown_links_for_generated_destination(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "examples/training/model/dataset/README.md"
    destination_path = tmp_path / "docs/training/examples/model_dataset.md"
    asset_path = source_path.parent / "plot.png"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"image")
    monkeypatch.setattr(GENERATE_EXAMPLES, "ROOT_DIR", tmp_path)
    content = (
        '[guide](../../../../docs/training/guide.md?mode=full#options "Guide")\n'
        "[single-title](../../../../docs/training/guide.md 'Guide')\n"
        "[parenthesized-title](../../../../docs/training/guide.md (Guide))\n"
        '![plot](plot.png "Training plot")\n'
        "[external](https://example.com/guide.md)\n"
        "[section](#options)\n"
        "```markdown\n"
        "[example](../../../../docs/example.md)\n"
        "```\n")

    result = rebase_relative_markdown_links(content, source_path, destination_path)

    assert '[guide](../guide.md?mode=full#options "Guide")' in result
    assert "[single-title](../guide.md 'Guide')" in result
    assert "[parenthesized-title](../guide.md (Guide))" in result
    assert ('![plot](https://raw.githubusercontent.com/hao-ai-lab/FastVideo/main/'
            'examples/training/model/dataset/plot.png "Training plot")') in result
    assert "[external](https://example.com/guide.md)" in result
    assert "[section](#options)" in result
    assert "```markdown\n[example](../../../../docs/example.md)\n```" in result


def test_check_docs_links_handles_optional_titles(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs_root / "page.md").write_text('[guide](guide.md "Guide")\n[missing](missing.md \'Missing\')\n',
                                         encoding="utf-8")

    assert check_links(docs_root) == ["docs/page.md:2 -> missing.md"]
