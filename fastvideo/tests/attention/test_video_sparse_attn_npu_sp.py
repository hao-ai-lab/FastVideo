# SPDX-License-Identifier: Apache-2.0
"""Opt-in real four-NPU regression: run under an exclusive four-device lease."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark=pytest.mark.skipif(os.environ.get('VSA_RUN_SP4_TESTS')!='1',reason='requires explicit four-NPU lease')
REQUIRED_GPUS=4


def launch(tmp_path,fail=False):
    devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
    assert len(devices)==4 and len(set(devices))==4
    repo=Path(__file__).resolve().parents[3]
    output=tmp_path/'ranks'
    command=[sys.executable,str(repo/'benchmarks/vsa_npu/launch_group.py'),'--output',str(output),
             '--suite','contract','--timeout','240']
    if fail:command+=['--fail-rank','2']
    result=subprocess.run(command,timeout=300)
    jobs=json.loads((output/'jobs.json').read_text())
    assert all(code is not None for code in jobs['exit_codes'])
    for child in jobs['children']:
        with pytest.raises(ProcessLookupError):os.kill(child['pid'],0)
    return result,jobs,output


def test_sp4_global_selection_padding_gates_and_restore(tmp_path):
    result,jobs,output=launch(tmp_path)
    assert result.returncode==0 and jobs['status']=='PASS'
    for rank in range(4):
        record=json.loads((output/f'rank-{rank}.json').read_text())
        assert record['status']=='PASS'
        assert record['head_divisibility_rejection_verified']
        assert all(r['mask_agreement'] and r['input_preserved'] for r in record['records'])


def test_sp4_rank_failure_reaps_all_peers(tmp_path):
    result,jobs,_=launch(tmp_path,fail=True)
    assert result.returncode!=0 and jobs['status']=='FAILED'
    assert 42 in jobs['exit_codes']
