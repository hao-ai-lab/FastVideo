# SPDX-License-Identifier: Apache-2.0
"""CPU regression for the ModelOpt activation-scale export contract."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def converter(monkeypatch):
    path = Path(__file__).resolve().parents[4] / 'scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py'
    spec = importlib.util.spec_from_file_location('h3_modelopt_converter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packed = torch.zeros((2, 8), dtype=torch.uint8)
    scales = torch.ones((2, 1), dtype=torch.float8_e4m3fn)
    def quantize(*args, **kwargs):
        return packed, scales
    monkeypatch.setattr(module, '_flashinfer', lambda: (
        SimpleNamespace(layout_128x4=0), None, quantize, lambda value: value))
    return module


def convert(converter, input_scale):
    return converter.convert_modelopt_linear(
        torch.zeros((2, 8), dtype=torch.uint8),
        torch.ones((2, 1), dtype=torch.float8_e4m3fn),
        torch.tensor(0.25), 'cpu', input_scale=input_scale)[0]


def test_modelopt_activation_scale_is_preserved_as_reciprocal(converter):
    buffers = convert(converter, torch.tensor(2.0))
    assert buffers['_nvfp4_input_global_sf'].dtype == torch.float32
    assert buffers['_nvfp4_input_global_sf'].shape == ()
    assert buffers['_nvfp4_input_global_sf'].item() == 0.5
    assert buffers['_nvfp4_alpha'].item() == 0.25


def test_missing_modelopt_activation_scale_keeps_legacy_export(converter):
    assert '_nvfp4_input_global_sf' not in convert(converter, None)


@pytest.mark.parametrize('value', [0.0, -1.0, float('nan'), float('inf')])
def test_invalid_modelopt_activation_scale_is_rejected(converter, value):
    with pytest.raises(ValueError, match='finite positive scalar'):
        convert(converter, torch.tensor(value))
