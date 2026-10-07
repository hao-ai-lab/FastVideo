import ast
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).resolve().parents[1] / "attn_qat_infer" / "quantization" / "bench"


@pytest.mark.parametrize("script_name", ["bench_quant_q.py", "bench_quant_k.py", "bench_quant_v.py"])
def test_quant_benchmark_imports_built_extension(script_name: str) -> None:
    tree = ast.parse((BENCH_DIR / script_name).read_text())
    imports = {
        (alias.name, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert ("fp4quant_cuda", "fp4quant") in imports
    assert ("fp4quant", None) not in imports
