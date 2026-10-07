import ast
import subprocess
import sys
from pathlib import Path

ALLOWED_PREFIXES = (
    "fastvideo.api",
    "fastvideo.entrypoints.streaming",
    "fastvideo.entrypoints.video_generator",
    "fastvideo.configs",
)
ALLOWED_EXACT = ("fastvideo", )
FORBIDDEN_PREFIXES = (
    "fastvideo.pipelines",
    "fastvideo.models",
    "fastvideo.layers",
    "fastvideo.worker",
    "fastvideo.fastvideo_args",
)
ALLOWED_INTERNAL_IMPORTS = {
    (
        "ltx2_generation.py",
        "fastvideo.models.audio.ltx2_audio_processing",
    ),
    (
        "ltx2_generation.py",
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
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                if not name:
                    continue
                rel_path = str(path.relative_to(root))
                if (name.startswith(FORBIDDEN_PREFIXES) and (rel_path, name) not in ALLOWED_INTERNAL_IMPORTS):
                    bad.append((str(path.relative_to(root)), getattr(node, "lineno", 0), name))

    assert bad == [], f"Forbidden internal imports: {bad}"


def test_h3_reference_public_export_is_lazy_and_preserves_type_identity() -> None:
    """Only explicit reference usage should load H3's optional GPU dependencies."""
    repo_root = Path(__file__).resolve().parents[4]
    # Isolate the import graph: keep the real public API implementation/schema,
    # substituting only the unrelated legacy sampling module and heavy H3 leaf.
    script = r'''
import importlib
import sys
from pathlib import Path
from types import ModuleType

root = Path(sys.argv[1])
fastvideo = ModuleType("fastvideo")
fastvideo.__path__ = [str(root / "fastvideo")]
sys.modules["fastvideo"] = fastvideo
sampling = ModuleType("fastvideo.api.sampling_param")
sampling.SamplingParam = type("SamplingParam", (), {})
sys.modules[sampling.__name__] = sampling

api = importlib.import_module("fastvideo.api")
assert "MiniMaxH3Reference" in api.__all__
assert "MiniMaxH3Reference" not in vars(api)
assert not any(name.startswith("fastvideo.pipelines") for name in sys.modules)

internal = ModuleType("fastvideo.pipelines.basic.minimax_h3.reference")
internal.MiniMaxH3Reference = type("MiniMaxH3Reference", (), {})
sys.modules[internal.__name__] = internal
from fastvideo.api import MiniMaxH3Reference
assert MiniMaxH3Reference is internal.MiniMaxH3Reference
assert api.MiniMaxH3Reference is internal.MiniMaxH3Reference
assert not hasattr(api, "UnknownReference")
'''
    result = subprocess.run([sys.executable, "-c", script, str(repo_root)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
