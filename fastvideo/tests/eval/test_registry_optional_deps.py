# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys


def test_dynamic_degree_stays_registered_without_easydict():
    script = """
import sys
sys.modules['easydict'] = None
from fastvideo.eval.registry import get_metric, list_metrics
assert 'vbench.dynamic_degree' in list_metrics()
try:
    get_metric('vbench.dynamic_degree')
except ImportError as error:
    assert 'easydict' in str(error)
else:
    raise AssertionError('Missing optional dependency was not reported')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
