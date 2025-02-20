from functools import partial
from multiprocessing import Manager

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fastvideo.utils.communications import nccl_info, prepare_sequence_parallel_data


def _init_distributed_test_gpu(rank, world_size, backend, port, data, results):
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
    )

    device = torch.device(f"cuda:{rank}")

    nccl_info.sp_size = world_size
    nccl_info.rank_within_group = rank
    nccl_info.group_id = 0

    seq_group = dist.new_group(ranks=list(range(world_size)))
    nccl_info.group = seq_group

    hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask = data
    hidden_states = hidden_states[rank].unsqueeze(dim=0).to(device)
    encoder_hidden_states = encoder_hidden_states.to(device)
    attention_mask = attention_mask.to(device)
    encoder_attention_mask = encoder_attention_mask.to(device)
    print(f"Rank {rank} input hidden_states:\n", hidden_states)
    print(f"Rank {rank} input hidden_states shape:\n", hidden_states.shape)
    out_hidden, out_encoder, out_attn_mask, out_encoder_mask = prepare_sequence_parallel_data(
        hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask)
    print(f"Rank {rank} output out_hidden:\n", out_hidden)

    shapes = (
        out_hidden.shape,
        out_encoder.shape,
        out_attn_mask.shape,
        out_encoder_mask.shape,
    )
    shape_tensor = torch.tensor([*shapes[0], *shapes[1], *shapes[2], *shapes[3]], dtype=torch.int32, device=device)
    shape_list = [torch.zeros_like(shape_tensor) for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor, group=seq_group)
    gathered_shapes = [tuple(s.tolist()) for s in shape_list]
    out_hidden_cpu = out_hidden.to("cpu")

    results[rank] = {
        "shapes": gathered_shapes,
        "out_hidden": out_hidden_cpu,
    }

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="Requires at least 2 GPUs to run NCCL tests")
def test_prepare_sequence_parallel_data_gpu():
    world_size = 2
    backend = "nccl"
    port = 12355  # or use a random free port if collisions occur

    # Create test tensors on CPU; the dimension at index=2 should be divisible by world_size=2 (if applicable).
    hidden_states = torch.randn(2, 1, 2, 1, 1)
    encoder_hidden_states = torch.randn(2, 2)
    attention_mask = torch.randn(2, 2)
    encoder_attention_mask = torch.randn(2, 2)

    print("init hidden states", hidden_states)

    manager = Manager()
    results_dict = manager.dict()

    # Wrap our helper function with partial
    mp_func = partial(_init_distributed_test_gpu,
                      world_size=world_size,
                      backend=backend,
                      port=port,
                      data=(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask),
                      results=results_dict)

    # Spawn two GPU processes (rank=0, rank=1)
    mp.spawn(mp_func, nprocs=world_size)

    first_rank_shapes = None

    overall_hidden_out = []

    for rank in sorted(results_dict.keys()):
        rank_data = results_dict[rank]
        rank_shapes = rank_data["shapes"]
        if first_rank_shapes is None:
            first_rank_shapes = rank_shapes
        assert rank_shapes == first_rank_shapes, (
            f"Mismatch in shapes across ranks: {rank_shapes} != {first_rank_shapes}")
        overall_hidden_out.append(rank_data["out_hidden"])

    overall_hidden_out = torch.cat(overall_hidden_out, dim=2)
    print("overall_hidden_out", overall_hidden_out)
    print("overall_hidden_out_shape", overall_hidden_out.shape)

    assert torch.allclose(hidden_states, torch.tensor(overall_hidden_out), rtol=1e-7, atol=1e-6)


if __name__ == "__main__":
    test_prepare_sequence_parallel_data_gpu()
