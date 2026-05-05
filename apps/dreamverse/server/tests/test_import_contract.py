import ast
from pathlib import Path

ALLOWED_PREFIXES = (
    "fastvideo.api",
    "fastvideo.entrypoints.streaming",
    "fastvideo.entrypoints.video_generator",
    "fastvideo.configs",
)
ALLOWED_EXACT = ("fastvideo",)
FORBIDDEN_PREFIXES = (
    "fastvideo.pipelines",
    "fastvideo.models",
    "fastvideo.layers",
    "fastvideo.worker",
    "fastvideo.fastvideo_args",
)
ALLOWED_INTERNAL_IMPORTS = {
    (
        "video_generation.py",
        "fastvideo.models.audio.ltx2_audio_processing",
    ),
    (
        "video_generation.py",
        "fastvideo.models.loader.component_loader",
    ),
}


def test_dreamverse_server_imports_only_public_fastvideo_surfaces() -> None:
    root = Path(__file__).resolve().parents[1]
    bad: list[tuple[str, int, str]] = []
    for path in root.rglob("*.py"):
        if "/tests/" in path.as_posix():
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as task_exc:
            raise AssertionError(f"Failed to parse {path}") from task_exc
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            for name in names:
                if not name:
                    continue
                rel_path = str(path.relative_to(root))
                if (
                    name.startswith(FORBIDDEN_PREFIXES)
                    and (rel_path, name) not in ALLOWED_INTERNAL_IMPORTS
                ):
                    bad.append((str(path.relative_to(root)), getattr(node, "lineno", 0), name))

    assert bad == [], f"Forbidden internal imports: {bad}"
