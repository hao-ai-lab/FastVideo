"""Render the existing Python API without importing FastVideo's runtime."""
from __future__ import annotations

import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin

import yaml
from selectolax.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "docs-site/.generated"
LEGACY = GENERATED / "mkdocs"


def prepare() -> None:
    GENERATED.mkdir(exist_ok=True)
    # Reuse only the general example generator; cookbook generation belongs to its own build.
    runpy.run_path(str(ROOT / "docs/generate_examples.py"))["generate_examples"](generate_main_index=True)
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
    config["docs_dir"] = str(ROOT / "docs")
    config["site_dir"] = str(LEGACY)
    config.pop("hooks", None)
    config["plugins"] = [
        plugin for plugin in config["plugins"]
        if not isinstance(plugin, dict) or not {"minify", "git-revision-date-localized"}.intersection(plugin)
    ]
    for plugin in config["plugins"]:
        if isinstance(plugin, dict) and "api-autonav" in plugin:
            plugin["api-autonav"]["modules"] = [str(ROOT / "fastvideo")]
        if isinstance(plugin, dict) and "mkdocstrings" in plugin:
            plugin["mkdocstrings"]["handlers"]["python"]["paths"] = [str(ROOT)]
    config_file = GENERATED / "mkdocs.yml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))

    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for folder in (ROOT / "docs", ROOT / "fastvideo", ROOT / "examples"):
        for path in sorted(folder.rglob("*")):
            if "node_modules" in path.parts or not path.is_file() or path.suffix not in {".md", ".mdx", ".py", ".sh", ".yaml"}:
                continue
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update((ROOT / "mkdocs.yml").read_bytes())
    digest.update((ROOT / "requirements-mkdocs.txt").read_bytes())
    stamp = GENERATED / "python-docs.sha256"
    fingerprint = digest.hexdigest()
    if not stamp.exists() or stamp.read_text() != fingerprint or not (LEGACY / "api/index.html").exists():
        log_file = GENERATED / "mkdocs-build.log"
        with log_file.open("w") as log:
            result = subprocess.run([sys.executable, "-m", "mkdocs", "build", "-f", str(config_file)], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Python reference build failed. See {log_file}\n{log_file.read_text()[-6000:]}")
        stamp.write_text(fingerprint)

    headings = {}
    reference = GENERATED / "python-reference"
    shutil.rmtree(reference, ignore_errors=True)
    reference.mkdir(exist_ok=True)
    pages = []
    for path in sorted(LEGACY.rglob("index.html")):
        route = path.parent.relative_to(LEGACY).as_posix()
        if route.startswith("cookbook"):
            continue
        tree = HTMLParser(path.read_text())
        article = tree.css_first("article.md-content__inner")
        if article is None:
            continue
        headings[route] = [{"text": node.text(), "id": node.attributes["id"], "level": int(node.tag[1])}
                           for node in article.css("h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]")]
        if not (route == "api" or route.startswith("api/") or route == "getting_started/v1_api"):
            continue
        for node in article.css(".md-content__button, .headerlink, script"):
            node.decompose()
        for node in article.css("[href], [src]"):
            for attr in ("href", "src"):
                value = node.attributes.get(attr)
                if not value or value.startswith(("#", "mailto:", "http:", "https:", "data:")):
                    continue
                resolved = urljoin(f"/{route}/", value)
                node.attrs[attr] = resolved
        title_node = article.css_first("h1")
        title = title_node.text().strip() if title_node else route.split("/")[-1]
        # Keep the original H1 anchor for incoming links, while Blume renders the visible title.
        if title_node:
            title_node.replace_with(HTMLParser(f'<a id="{title_node.attributes.get("id", "")}"></a>').css_first("a"))
        body = article.html
        body = body[body.index(">") + 1:body.rfind("</article>")]
        target = reference / f"{route}/index.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("---\n" + yaml.safe_dump({"title": title, "noindex": True}, sort_keys=False) +
                          '---\n\n<div class="python-reference">\n' + body + "\n</div>\n")
        pages.append(route)
    (GENERATED / "old-headings.json").write_text(json.dumps(headings))
    (GENERATED / "mkdocs-nav.json").write_text(json.dumps(config["nav"]))
    print(f"Prepared {len(pages)} Python reference pages and the original heading anchors.")


if __name__ == "__main__":
    prepare()
