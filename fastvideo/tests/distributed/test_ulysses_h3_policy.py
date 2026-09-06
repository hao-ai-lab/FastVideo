# SPDX-License-Identifier: Apache-2.0
"""Policy, rank agreement and saved backward-plan regressions without a GPU."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from fastvideo.distributed.device_communicators import ulysses_a2a as ulysses


def _helper():
    return ulysses.UlyssesA2AHelper(object(), object(), 4, torch.device('cuda:0'), object())


def _operand(planes, global_sequence, mode, requires_grad):
    shape = (planes, global_sequence // 4, 56, 128) if mode == 0 else (planes, global_sequence, 14, 128)
    x = Mock(shape=shape, dtype=torch.bfloat16, device=torch.device('cuda:0'), is_cuda=True,
             requires_grad=requires_grad)
    x.dim.return_value = 4
    x.numel.return_value = planes * global_sequence * 14 * 128
    x.element_size.return_value = 2
    x.is_contiguous.return_value = True
    return x


@pytest.mark.parametrize('sequence,large_pack', [(32000, False), (64000, False), (128000, False),
                                               (149796, False), (149800, True), (250000, True)])
@pytest.mark.parametrize('planes,mode,training', [(3, 0, False), (4, 0, True), (1, 1, True), (4, 1, True)])
def test_h3_policy_caps_each_plane_and_retains_fast_gathers(sequence, large_pack, planes, mode, training):
    helper = _helper()
    x = _operand(planes, sequence, mode, training)
    with patch.object(helper, '_execution_plan', return_value=(True, 144)), \
            patch.object(ulysses, 'is_enabled', return_value=True), \
            patch('torch.cuda.is_current_stream_capturing', return_value=False), torch.set_grad_enabled(training):
        signature, _ = helper._call_signature(x, *((2, 1) if mode == 0 else (1, 2)))
    declined = training and planes > 1 and large_pack
    assert signature[0] == (0 if declined else 1)
    original_plan = training and large_pack
    assert signature[8] == sequence * 14 * 128 * 2 * (planes if original_plan else 1)
    assert signature[10:] == ((0, 36) if original_plan else (1, 144))


def test_long_training_chunk_opt_in_keeps_a_bounded_window():
    helper = _helper()
    x = _operand(4, 250000, 0, True)
    with patch.object(helper, '_execution_plan', return_value=(True, 144)), \
            patch.object(ulysses, 'is_enabled', return_value=True), \
            patch.object(ulysses.envs, 'FASTVIDEO_ULYSSES_A2A_LONG_TRAINING', 'chunked', create=True), \
            patch('torch.cuda.is_current_stream_capturing', return_value=False), torch.enable_grad():
        signature, _ = helper._call_signature(x, 2, 1)
    assert signature[0] == 1
    assert signature[8] == 896000000
    assert signature[8] <= ulysses.MAX_WINDOW_BYTES
    assert signature[10:] == (1, 144)


def test_older_wheel_keeps_original_launch_and_capacity_policy():
    helper = _helper()
    x = _operand(3, 250000, 0, False)
    from fastvideo_kernel import comm_ops
    props = SimpleNamespace(name='NVIDIA GB200', major=10, minor=0, multi_processor_count=152)
    with patch('torch.cuda.get_device_properties', return_value=props), \
            patch.object(comm_ops, 'supports_tuned_launch', return_value=False, create=True), \
            patch.object(ulysses, 'is_enabled', return_value=True), \
            patch('torch.cuda.is_current_stream_capturing', return_value=False):
        signature, _ = helper._call_signature(x, 2, 1)
    assert signature[0] == 0
    assert signature[8] == x.numel() * 2
    assert signature[10:] == (0, 36)


def test_rank_disagreement_on_launch_or_chunking_declines_and_recovers():
    helper = _helper()
    signature = (1, 0, 0, 2, 4, 32000, 56, 128, 458752000, 0, 1, 144)
    for changed_index, value in [(10, 0), (11, 36), (0, 0)]:
        peer = list(signature)
        peer[changed_index] = value

        def gather(output, local, **kwargs):
            output.copy_(torch.tensor([*signature, *peer, *signature, *signature]))

        with patch.object(ulysses.dist, 'all_gather_into_tensor', side_effect=gather):
            assert helper._agree_call(signature) == (False, False, True)

    def unanimous(output, local, **kwargs):
        output.copy_(local.repeat(4))

    with patch.object(ulysses.dist, 'all_gather_into_tensor', side_effect=unanimous):
        assert helper._agree_call(signature) == (True, False, True)


def test_backward_uses_its_own_forward_plan_after_another_call():
    calls = []

    class Helper:
        def run_armed(self, x, mode, chunked, blocks):
            calls.append((mode, chunked, blocks))
            return x.clone()

    helper = Helper()
    x = torch.randn(4, requires_grad=True)
    first = ulysses._FusedUlyssesA2A.apply(helper, x, 0, True, 144)
    second = ulysses._FusedUlyssesA2A.apply(helper, x, 1, False, 36)
    first.sum().backward()
    assert calls == [(0, True, 144), (1, False, 36), (1, True, 144)]
    assert torch.equal(first, second)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(x.grad, torch.ones_like(x))
