"""Build source-only Python reference pages with Griffe, without importing FastVideo."""
from __future__ import annotations

import html
import json
import re
import runpy
import shutil
from pathlib import Path

import griffe
import markdown
import yaml

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs-site"
GENERATED = SITE / ".generated"


def signature(obj) -> str:
    """Preserve parameter kinds, annotations, defaults, and return annotations."""
    if obj.is_attribute:
        annotation = f": {obj.annotation}" if obj.annotation else ""
        value = f" = {obj.value}" if obj.value is not None else ""
        return f"{obj.name}{annotation}{value}"
    parameters = list(getattr(obj, "parameters", []))
    parts = []
    keyword_separator = False
    for i, param in enumerate(parameters):
        if param.name in {"self", "cls"}:
            continue
        kind = param.kind.value
        if kind == "keyword-only" and not keyword_separator:
            parts.append("*")
            keyword_separator = True
        prefix = "**" if kind == "variadic keyword" else "*" if kind == "variadic positional" else ""
        if prefix == "*":
            keyword_separator = True
        text = prefix + param.name
        if param.annotation is not None:
            text += f": {param.annotation}"
        if param.default is not None and not prefix:
            text += f" = {param.default}"
        parts.append(text)
        if kind == "positional-only" and (i + 1 == len(parameters) or parameters[i + 1].kind.value != kind):
            parts.append("/")
    result = f"{obj.name}({', '.join(parts)})"
    if obj.is_function and obj.returns is not None:
        result += f" -> {obj.returns}"
    return result


def code_html(text: str) -> str:
    return html.escape(text).replace("[", "&#91;").replace("]", "&#93;")


def prose(text: str) -> str:
    # Docstrings are trusted repository source, rendered as Markdown rather than executed.
    return markdown.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])


def docstring(obj, level: int) -> str:
    if not obj.docstring:
        return ""
    output = []
    for section in obj.docstring.parsed:
        kind = section.kind.value
        if kind == "text":
            output.append(prose(section.value))
            continue
        title = section.title or kind.replace("_", " ").capitalize()
        output.append(f"<h{level}>{html.escape(title)}</h{level}>")
        if kind == "examples":
            for example_kind, value in section.value:
                output.append(f"<pre><code>{html.escape(value)}</code></pre>" if example_kind.value == "examples" else prose(value))
        elif isinstance(section.value, list):
            output.append("<dl>")
            for item in section.value:
                name = getattr(item, "name", "") or ""
                annotation = getattr(item, "annotation", "") or ""
                label = f"{name}: {annotation}" if name and annotation else str(name or annotation)
                output.append(f"<dt><code>{html.escape(label)}</code></dt><dd>{prose(getattr(item, 'description', ''))}</dd>")
            output.append("</dl>")
        else:
            output.append(prose(getattr(section.value, "description", str(section.value))))
    return "\n".join(output)


def symbol(obj, path: str, level: int = 2, members: list[str] | None = None, source: bool = True) -> str:
    level = min(level, 6)
    output = [f'<h{level} id="{html.escape(path)}"><code>{html.escape(path)}</code></h{level}>']
    output.append(f'<pre class="doc-signature"><code>{code_html(signature(obj))}</code></pre>')
    output.append(docstring(obj, min(level + 1, 6)))
    if source and not obj.is_alias:
        try:
            code = obj.source
        except (ValueError, KeyError):
            code = ""
        if code:
            output.append(f'<details><summary>Source</summary><pre><code>{code_html(code)}</code></pre></details>')
    if obj.is_class:
        for name, member in obj.members.items():
            if members is not None and name not in members:
                continue
            if name.startswith("_") or member.is_module:
                continue
            if member.is_attribute and not member.docstring:
                continue
            try:
                output.append(symbol(member, f"{path}.{name}", min(level + 1, 6), source=source))
            except griffe.AliasResolutionError:
                continue
    return "\n".join(output)


def write_page(route: str, title: str, body: str, anchors: dict) -> None:
    # Preserve incoming fragments, including historical category labels whose layout changed.
    present = set(re.findall(r'id="([^"]+)"', body))
    aliases = "\n".join(f'<a id="{html.escape(h["id"])}"></a>' for h in anchors.get(route, []) if h["id"] not in present)
    target = GENERATED / "python-reference" / route / "index.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("---\n" + yaml.safe_dump({"title": title}, sort_keys=False) +
                      f'---\n\n<div class="python-reference">\n{aliases}\n{body}\n</div>\n')


def prepare() -> None:
    GENERATED.mkdir(exist_ok=True)
    examples = runpy.run_path(str(ROOT / "docs/generate_examples.py"))
    examples["validate_cookbook"]()
    examples["generate_cookbook_serving"]()
    examples["generate_examples"](generate_main_index=True)
    anchors = json.loads((SITE / "migration-anchors.json").read_text())
    shutil.rmtree(GENERATED / "python-reference", ignore_errors=True)
    package = griffe.load("fastvideo", search_paths=[ROOT], allow_inspection=False, resolve_aliases=True,
                          resolve_external=False, docstring_parser="google", docstring_options={"warnings": False})
    modules = []

    def visit(module):
        if module.is_alias or any(part.startswith("_") or part in {"tests", "third_party"} for part in module.path.split(".")):
            return
        modules.append(module)
        for child in module.modules.values():
            visit(child)

    visit(package)
    for module in sorted(modules, key=lambda item: item.path):
        body = [f'<a id="{module.path}"></a>', docstring(module, 2)]
        for name, member in module.members.items():
            if name.startswith("_"):
                continue
            try:
                # External imports belong to their upstream reference, not FastVideo's API.
                if member.is_alias and not member.target_path.startswith("fastvideo."):
                    continue
                if member.is_module or member.is_attribute and not member.docstring:
                    continue
                body.append(symbol(member, f"{module.path}.{name}"))
            except griffe.AliasResolutionError:
                continue
        write_page("api/" + module.path.replace(".", "/"), module.name, "\n".join(body), anchors)
    write_page("api", "Python API reference", prose((ROOT / "docs/api/index.md").read_text()), anchors)
    v1 = (ROOT / "docs/getting_started/v1_api.md").read_text()

    def directive(match):
        path, options = match.groups()
        settings = yaml.safe_load(options) or {}
        settings = settings.get("options", {})
        return "\n" + symbol(package[path.removeprefix("fastvideo.")], path,
                               settings.get("heading_level", 3), settings.get("members"), source=False) + "\n"

    v1 = re.sub(r"^::: (\S+)\n((?:[ \t]+.*\n|\n)*)", directive, v1, flags=re.MULTILINE)
    write_page("getting_started/v1_api", "V1 API", prose(v1.removeprefix("# V1 API\n")), anchors)
    (GENERATED / "old-headings.json").write_text(json.dumps(anchors))
    print(f"Rendered {len(modules) + 1} Python reference pages directly from source, without MkDocs or runtime imports.")


if __name__ == "__main__":
    prepare()
