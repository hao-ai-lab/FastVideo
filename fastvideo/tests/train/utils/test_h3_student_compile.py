# SPDX-License-Identifier: Apache-2.0
"""CPU lifecycle checks for scoped H3 compilation; kernels are tested separately."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.utils import minimax_h3_compile as policy
from fastvideo.train.utils.training_config import ModelTrainingConfig, TrainingConfig


def test_default_policy_does_not_inspect_or_modify_models(monkeypatch):
    monkeypatch.setattr(policy, '_validate_policy', lambda *args: pytest.fail('unexpected validation'))
    monkeypatch.setattr(policy, 'NativeBackwardBlockSizeGuard', lambda: pytest.fail('unexpected kernel import'))
    with policy.minimax_h3_student_compile(SimpleNamespace(training=TrainingConfig()), object()):
        pass


@pytest.mark.parametrize('value', [None, False, True, 32, '', 'eager', 'full_block'])
def test_invalid_policy_rejected(value):
    with pytest.raises(ValueError, match='minimax_h3_student_compile'):
        ModelTrainingConfig(minimax_h3_student_compile=value)


def test_attention_flags_restored_when_installation_fails(monkeypatch):
    class Attention(torch.nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self._compile_forward_enabled = enabled

        def _set_compile_forward_enabled(self, enabled):
            self._compile_forward_enabled = enabled

    transformer = torch.nn.Sequential(Attention(False), Attention(True))
    method = SimpleNamespace(student=SimpleNamespace(transformer=transformer))
    cfg = SimpleNamespace(training=TrainingConfig(model=ModelTrainingConfig(
        minimax_h3_student_compile='grad_only_full_block',
        torch_compile_kwargs={'recompile_limit': 8},
    )))
    monkeypatch.setattr(policy, '_validate_policy', lambda *args: {})
    from fastvideo.attention import layer
    monkeypatch.setattr(layer, '_attention_compile_explicitly_disabled', lambda: False)

    def fail_install():
        assert all(module._compile_forward_enabled for module in transformer)
        raise RuntimeError('kernel installation failed')

    monkeypatch.setattr(policy, 'NativeBackwardBlockSizeGuard', fail_install)
    with pytest.raises(RuntimeError, match='kernel installation failed'):
        with policy.minimax_h3_student_compile(cfg, method):
            pytest.fail('body reached after installation error')
    assert not transformer[0]._compile_forward_enabled
    assert transformer[1]._compile_forward_enabled
