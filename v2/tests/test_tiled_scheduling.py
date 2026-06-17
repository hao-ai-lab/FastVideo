"""Heterogeneous WorkUnit co-scheduling (design_v3 §6.1, §9.3, §17 falsifier).

The `VAE_TILE` WorkUnit kind had zero coverage. These tests put tiled VAE decode (one `VAE_TILE` unit
per latent row-band) through the same admission/scheduler/interleave machinery as denoise steps, and
assert:

  * **tiling is exact** — tiled decode == one-shot decode (a C0 component parity);
  * **the kind is real WorkUnits** — the tiled run emits one `VAE_TILE` unit per row;
  * **mixed kinds interleave bit-identically** — a `VAE_TILE` pipeline and a `DIFFUSION_STEP` pipeline,
    interleaved, equal their serial run (the §9.3 gate over *heterogeneous* units);
  * **the scheduler co-runs both kinds** in one interleaved batch.

This validates the *mechanism* the §17 falsifier questions (non-step kinds flow through one budget). Whether
it *pays* on a real GPU duty-cycle is the empirical half of the falsifier — deferred to the port.
"""
from __future__ import annotations

import numpy as np

from v2.models import build_tiled_engine
from v2.parity import assert_interleave_parity
from v2.request import DiffusionParams, TaskType, make_request


def _tiled_req(prompt="a fox", seed=3):
    return make_request(TaskType.T2V, "wan-tiled", prompt,
                        diffusion=DiffusionParams(num_steps=4, num_frames=81, seed=seed))


def test_tiled_decode_equals_oneshot():
    """The VAE is spatially local, so decoding in row-tiles and stitching is *exactly* one-shot decode."""
    eng = build_tiled_engine()
    out = eng.run(_tiled_req())
    tiled = np.asarray(out.artifacts["video"].frames)
    latent = out.artifacts["latents"].latent                       # the denoise latent that was decoded
    oneshot = eng._registry["wan-tiled"][0].component("vae").decode(latent)
    assert np.array_equal(tiled, np.asarray(oneshot))


def test_tiled_decode_emits_one_workunit_per_row():
    eng = build_tiled_engine()
    out = eng.run(_tiled_req())
    latent_rows = np.asarray(out.artifacts["latents"].latent).shape[2]
    assert out.metrics["vae_tiles"] == float(latent_rows)          # tile_rows=1 ⇒ one VAE_TILE unit/row
    assert latent_rows >= 1


def test_heterogeneous_interleave_parity():
    """A VAE_TILE pipeline and a DIFFUSION_STEP-only pipeline, interleaved, are bit-identical to serial —
    the interleave gate holds across *mixed* WorkUnit kinds, not just homogeneous ones."""
    eng = build_tiled_engine()
    reqs = [
        _tiled_req("alpha", 1),
        make_request(TaskType.T2V, "wan2.1-1.3b", "beta", diffusion=DiffusionParams(num_steps=4, seed=2)),
        _tiled_req("alpha", 1),
    ]
    assert not assert_interleave_parity(eng, reqs)


def test_scheduler_co_runs_both_kinds_in_one_interleaved_batch():
    eng = build_tiled_engine()
    reqs = [_tiled_req("x", 5),
            make_request(TaskType.T2V, "wan2.1-1.3b", "y", diffusion=DiffusionParams(num_steps=8, seed=6))]
    outs = eng.run_interleaved(reqs)                               # both kinds scheduled in one run
    tiled_out = outs[reqs[0].request_id]
    wan_out = outs[reqs[1].request_id]
    assert tiled_out.metrics["vae_tiles"] > 0                      # VAE_TILE units ran
    assert tiled_out.artifacts["video"].frames is not None
    assert wan_out.artifacts["video"].frames is not None          # the DIFFUSION_STEP pipeline finished too
