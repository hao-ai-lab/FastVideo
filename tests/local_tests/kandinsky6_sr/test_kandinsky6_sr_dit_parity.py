# SPDX-License-Identifier: Apache-2.0
"""FastVideo Kandinsky6 SR DiT vs the k6_video reference DiT (random tiny weights, batched vs packed layout).

The reference ``state_dict()`` (unprefixed, i.e. the official Diffusers layout) is loaded into the port through the arch config's
``param_names_mapping`` with ``strict=True``; outputs are compared in fp32.  On CPU the result is bit-identical;
on CUDA (flash-attn vs SDPA) a small tolerance applies.  Skips when the reference package is unavailable.
"""
from __future__ import annotations

import pytest
import torch

# FastVideo is imported before the reference is loaded (see reference.py).
from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

from . import reference as ref_env

k6_sr_tiny = ref_env.k6_sr_tiny

TOL = dict(rtol=0.0, atol=0.0) if not torch.cuda.is_available() else dict(rtol=1e-3, atol=1e-3)
B, T, H, W, C = 2, 3, 4, 6, 4


def _packed_call(ref_dit, x, t, extra=None):
    """Run the reference DiT on the packed ragged layout that the reference pipeline uses."""
    batch, frames = x.shape[:2]
    packed = x.reshape(batch * frames, *x.shape[2:])
    cu = frames * torch.arange(batch + 1, dtype=torch.int32)
    pos = [torch.cat([torch.arange(frames) for _ in range(batch)]), torch.arange(x.shape[2] // 2),
           torch.arange(x.shape[3] // 2)]
    kwargs = dict(scale_factor=(1.0, 1.0, 1.0))
    kwargs.update(extra or {})
    with torch.no_grad():
        return ref_dit(packed, torch.zeros(0, 1), torch.zeros(batch, 1), t, cu,
                       torch.zeros(batch + 1, dtype=torch.int32), pos, torch.zeros(0, dtype=torch.long), **kwargs)


def _port_call(dit, x, t, extra=None):
    pos = [torch.arange(x.shape[1]), torch.arange(x.shape[2] // 2), torch.arange(x.shape[3] // 2)]
    with torch.no_grad():
        return dit(x, t, pos, (1.0, 1.0, 1.0), **(extra or {}))


@pytest.mark.parametrize("piflow", [k6_sr_tiny.TINY_PIFLOW, None])
def test_dit_forward_matches_the_reference(piflow):
    ref = ref_env.load_reference()
    ref_dit = ref_env.build_reference_dit(ref, piflow)
    dit = k6_sr_tiny.build_dit(piflow)
    ref_env.load_into_port(ref_dit, dit)  # strict
    x = torch.randn(B, T, H, W, 2 * C + 1, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([300.0, 700.0])
    expected = _packed_call(ref_dit, x, t)
    got = _port_call(dit, x, t)
    if piflow is not None:  # DX wrapper already moved the grid axis: [B*T, n_grid, H, W, out]
        n_grid = piflow["n_grid"]
        got = got.reshape(B * T, H, W, n_grid, C).movedim(-2, 1)
    else:
        got = got.reshape(B * T, H, W, C)
    torch.testing.assert_close(got, expected, **TOL)


def test_reference_state_dict_keys_map_onto_the_port_exactly():
    ref = ref_env.load_reference()
    ref_dit = ref_env.build_reference_dit(ref)
    dit = k6_sr_tiny.build_dit()
    mapping = get_param_names_mapping(dit.param_names_mapping)
    custom, _ = hf_to_custom_state_dict(dict(ref_dit.state_dict()), mapping)  # unprefixed = the official layout
    assert set(custom) == set(dit.state_dict())
    assert all(custom[k].shape == v.shape for k, v in dit.state_dict().items())
