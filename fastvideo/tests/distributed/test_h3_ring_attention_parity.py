# SPDX-License-Identifier: Apache-2.0
"""Two-GPU Ring / four-GPU USP parity for the real H3 packed forward.

Run with pytest. No checkpoint needed; tiny random H3 weights exercise both
text-refiner blocks and packed text/video/audio blocks with 96/128 RoPE.
"""
import argparse
import os
from pathlib import Path

import pytest
import torch

from fastvideo.tests.distributed.test_ring_attention import _run_torchrun


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_h3_ring_matches_full_flash_attention(tmp_path, world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"H3 Ring parity requires {world_size} CUDA GPUs")
    output = tmp_path / "h3_ring.pt"
    _run_torchrun(Path(__file__).resolve(), world_size, output, worker_flag="--h3-worker")
    results = torch.load(output, weights_only=True)
    for actual, expected in results:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def _worker(output):
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    from flash_attn import flash_attn_func
    import fastvideo.models.dits.minimax_h3 as h3
    from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
    from fastvideo.distributed import cleanup_dist_env_and_memory
    from fastvideo.distributed.parallel_state import maybe_init_distributed_environment_and_model_parallel

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=world, ring_size=min(2, world))
        from fastvideo.utils import set_mixed_precision_policy
        set_mixed_precision_policy(torch.bfloat16, torch.float32)
        torch.manual_seed(42)
        # Pure Ring: 3 heads / 2 ranks. Hybrid: 6 heads / 4 SP ranks,
        # divisible only by the Ulysses subgroup size (2).
        arch = MiniMaxH3ArchConfig(num_attention_heads=3 if world == 2 else 6,
                                  hidden_size=24, ffn_dim=48, num_layers=2, num_refiner_layers=1,
                                  text_dim=24, time_embed_hidden_dim=24, time_embed_dim=24)
        model = h3.MiniMaxH3Transformer3DModel(MiniMaxH3Config(arch_config=arch), {}).cuda().bfloat16().eval()
        # ReplicatedLinear intentionally allocates empty parameters for the
        # checkpoint loader; initialize every parameter explicitly for parity.
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.ndim == 1:
                    param.fill_(1 if "norm" in name and name.endswith("weight") else 0)
                else:
                    param.normal_(mean=0, std=0.05)
        modules = [m.distributed_attention for m in model.modules() if isinstance(m, h3.MiniMaxH3Attention)]
        assert all(m.backend.name == "FLASH_ATTN" for m in modules)
        forwards = [m.forward for m in modules]
        original_sp = h3.get_sp_world_size

        def dense(q, k, v, **kwargs):
            assert kwargs["freqs_cis"] is None  # H3 already applied partial RoPE.
            return flash_attn_func(q, k, v, causal=False), None

        results = []
        # Both packed lengths and text lengths have padding. text_len=1
        # additionally exercises ranks with no valid refiner Q/K/V tokens.
        for length, text_len in [(13, 1), (129, 5)]:
            text_indices = torch.arange(text_len, device="cuda")
            video_indices = torch.arange(text_len, length, 2, device="cuda")
            audio_indices = torch.arange(text_len + 1, length, 2, device="cuda")
            tags = torch.zeros(length, device="cuda", dtype=torch.long)
            tags[video_indices] = 1
            tags[audio_indices] = 2
            inputs = dict(
                hidden_states=torch.randn(1, video_indices.numel(), 96, device="cuda", dtype=torch.bfloat16),
                audio_hidden_states=torch.randn(1, audio_indices.numel(), 32, device="cuda", dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1, text_len, 24, device="cuda", dtype=torch.bfloat16),
                timestep=torch.tensor([0.5], device="cuda"),
                timestep_indices=torch.zeros(length, device="cuda", dtype=torch.long),
                token_tags=tags,
                position_ids=torch.arange(length * 3, device="cuda").reshape(length, 3),
                video_indices=video_indices, audio_indices=audio_indices, text_indices=text_indices)
            with torch.inference_mode():
                from fastvideo.forward_context import set_forward_context
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    actual = model(**inputs)
                # Same parameters, full unsharded H3 forward, ordinary FA as
                # reference. Clear the text cache to test refiner parity too.
                model._text_cache = None
                try:
                    h3.get_sp_world_size = lambda: 1
                    for module in modules:
                        module.forward = dense
                    expected = model(**inputs)
                finally:
                    h3.get_sp_world_size = original_sp
                    for module, forward in zip(modules, forwards):
                        module.forward = forward
                    model._text_cache = None
            for a, b in zip(actual, expected):
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2)
                results.append((a.cpu(), b.cpu()))
        if rank == 0:
            torch.save(results, output)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h3-worker", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _worker(args.output)
