"""AMD-only autotune configs for the block-sparse Triton forward.

On ROCm the forward's autotuner is offered four extra configs that set the
AMD launch keys (``matrix_instr_nonkdim``, ``waves_per_eu``). On
gfx950 (MI355X) they beat every generic config at FastH3-like sequence
lengths, so the autotuner has to keep picking one of them there; the CUDA
backend rejects these keys, so it must never be offered them.
"""

import pytest
import torch

from fastvideo_kernel.block_sparse_attn import _map_to_index
from fastvideo_kernel.triton_kernels import block_sparse_attn_triton as bsa_triton

from .utils import generate_block_sparse_mask_for_function

BLOCK = 64
HIP_KEYS = {"matrix_instr_nonkdim", "waves_per_eu"}
IS_HIP = bool(getattr(torch.version, "hip", None))


def _is_hip_config(config) -> bool:
    return HIP_KEYS <= set(config.kwargs)


def _is_gfx950() -> bool:
    if not (IS_HIP and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")


def test_hip_configs_set_the_amd_launch_keys() -> None:
    hip_configs = bsa_triton._hip_configs()
    assert {(c.num_warps, c.kwargs["waves_per_eu"]) for c in hip_configs} == {(w, wpe) for w in (4, 8) for wpe in (1, 2)}
    for config in hip_configs:
        # BLOCK_M / BLOCK_N are structural: they must match the 64-token index granularity.
        assert config.kwargs["BLOCK_M"] == BLOCK and config.kwargs["BLOCK_N"] == BLOCK
        assert config.kwargs["matrix_instr_nonkdim"] == 16
        # Triton deprecates kpack from gfx950 on and overrides kpack=2 to 1 there.
        assert "kpack" not in config.kwargs
        assert config.num_stages == 2


def test_autotuner_offers_the_hip_configs_only_on_rocm() -> None:
    offered = [c for c in bsa_triton._attn_fwd_sparse.configs if _is_hip_config(c)]
    assert len(offered) == (len(bsa_triton._hip_configs()) if IS_HIP else 0)


@pytest.mark.skipif(not _is_gfx950(), reason="the HIP configs are tuned for gfx950 (MI355X)")
def test_gfx950_autotuner_picks_a_hip_config() -> None:
    torch.manual_seed(0)
    heads, dim, num_blocks, topk = 8, 128, 256, 40  # 16k tokens, ~16% block density
    seq = num_blocks * BLOCK
    q, k, v = (torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    block_mask = generate_block_sparse_mask_for_function(heads, num_blocks, num_blocks, k=topk,
                                                         device="cuda").unsqueeze(0)
    q2k_index, q2k_num = _map_to_index(block_mask)
    variable_block_sizes = torch.full((num_blocks, ), BLOCK, dtype=torch.int32, device="cuda")

    tuner = bsa_triton._attn_fwd_sparse
    tuner.cache.clear()  # re-tune even if an earlier test already tuned this shape
    bsa_triton.triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, variable_block_sizes)
    torch.cuda.synchronize()
    assert _is_hip_config(tuner.best_config), f"autotuner picked a generic config: {tuner.best_config}"
