# SPDX-License-Identifier: Apache-2.0
"""Explicit torchrun GPU gate for the native H3 transport policy (4x GB200)."""
import os
from unittest.mock import patch

import torch
import torch.distributed as dist

from fastvideo import envs
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase
from fastvideo.distributed.parallel_state import get_sp_group
from fastvideo.tests.distributed.test_ulysses_a2a_parity import _check_shape


def _check_native_shape(helper, batch, local_sequence, dtype, device, expected_plans):
    # Numerical parity also passes when both sides use NCCL. Assert the actual
    # helper executions, including the two backward collectives, separately.
    with patch.object(helper, 'run_armed', wraps=helper.run_armed) as run:
        _check_shape(batch, local_sequence, 56, 128, dtype, 4, device)
    plans = [call.args[1:] for call in run.call_args_list]
    assert plans == expected_plans, (batch, local_sequence, dtype, plans, expected_plans)


def main():
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.manual_seed(20260906 + rank)
    envs.FASTVIDEO_ULYSSES_A2A_LONG_TRAINING = 'auto'
    maybe_init_distributed_environment_and_model_parallel(1, 4)
    comm = get_sp_group().device_communicator
    helper = comm.ulysses_a2a
    assert helper is not None
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        tuned = dtype == torch.bfloat16
        plans = [(mode, tuned, 144 if tuned else 36) for mode in [0, 1, 0, 1, 1, 0]]
        _check_native_shape(helper, 3, 64, dtype, device, plans)
    for sequence in [32000, 128000, 250000]:
        modes = [0, 1] if sequence == 250000 else [0, 1, 0, 1, 1, 0]
        _check_native_shape(helper, 4, sequence // 4, torch.bfloat16, device,
                            [(mode, True, 144) for mode in modes])
        assert helper._nbytes <= 1024**3
    envs.FASTVIDEO_ULYSSES_A2A_LONG_TRAINING = 'chunked'
    _check_native_shape(helper, 4, 250000 // 4, torch.bfloat16, device,
                        [(mode, True, 144) for mode in [0, 1, 0, 1, 1, 0]])
    envs.FASTVIDEO_ULYSSES_A2A_LONG_TRAINING = 'auto'
    x = torch.randn(3, 8000, 56, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    first = comm.all_to_all_4D(x, 2, 1)
    saved = first.detach().clone()
    # A later call and inverse cannot overwrite an earlier autograd output.
    second = comm.all_to_all_4D(x.detach() + 1, 2, 1)
    comm.all_to_all_4D(second, 1, 2)
    assert torch.equal(first, saved)
    grad = torch.randn_like(first)
    first.backward(grad)
    expected = DeviceCommunicatorBase.all_to_all_4D(comm, grad, 1, 2)
    assert torch.equal(x.grad, expected)

    # A rank with an older launch capability must make the whole group decline.
    helper._h3_tuning_available = rank != 0
    assert helper.try_all_to_all_4D(x.detach(), 2, 1) is None
    helper._h3_tuning_available = True
    assert helper.try_all_to_all_4D(x.detach(), 2, 1) is not None
    contract, _ = helper._call_signature(x.detach(), 2, 1)
    assert contract[10:] == (1, 144)
    dist.barrier()
    if rank == 0:
        print('NATIVE_H3_OK dtypes=3 lengths=32000,128000,250000 exact_gradients=True ownership=True recovery=True '
              'execution_plans=True',
              flush=True)
    cleanup_dist_env_and_memory()


if __name__ == '__main__':
    main()
