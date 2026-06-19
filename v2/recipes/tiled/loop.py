"""VAETileLoop — tiled VAE decode (``LoopKind.VAE_TILE``).

Decodes a latent in spatial row-tiles, each tile a ``VAE_TILE`` WorkUnit flowing through the same
admission + step-scheduling + interleave machinery as a denoise step. So VAE_TILE units interleave with
DIFFUSION_STEP units across requests through one scheduler; this exercises whether that general scheduler
earns its generality on non-step kinds (whether it pays on a real GPU duty-cycle is deferred to the port).

The toy VAE is spatially local (channel-mix + repeat-upsample), so tiled decode is exactly equal to
one-shot decode — a C0 component parity the tests assert.
"""
from __future__ import annotations

import numpy as np

from v2._enums import WorkUnitKind
from v2.loop.contracts import (
    Done,
    LoopResult,
    LoopState,
    ResourceRequest,
    ShapeSignature,
    StepResult,
    WorkPlan,
)


class VAETileLoop:

    def __init__(self, *, loop_id, vae_id="vae", cost, tile_rows=1, latent_slot="denoise_out"):
        self.loop_id = loop_id
        self.vae_id = vae_id
        self.cost = cost
        self.tile_rows = tile_rows
        self.latent_slot = latent_slot

    def init(self, req, model, ctx) -> LoopState:
        out = ctx.slots.get(self.latent_slot)
        latent = out["latents"] if isinstance(out, dict) else out
        latent = np.asarray(latent, dtype="float32")
        st = LoopState(loop_id=self.loop_id,
                       instance_id=model.card.model_id,
                       request_id=req.request_id,
                       profile=ctx.profile)
        h = latent.shape[2]
        st.scratch["latent"] = latent
        st.scratch["tiles"] = [(r, min(r + self.tile_rows, h)) for r in range(0, h, self.tile_rows)] or [(0, h)]
        st.scratch["decoded"] = [None] * len(st.scratch["tiles"])
        return st

    def next(self, st: LoopState):
        i = st.step_idx
        tiles = st.scratch["tiles"]
        if i >= len(tiles):
            return Done()
        r0, r1 = tiles[i]
        latent, vid = st.scratch["latent"], self.vae_id

        def run(model, override=None):
            tile = latent[:, :, r0:r1, :]  # a spatial row-band of the latent
            dec = model.component(vid).decode(tile)
            return StepResult(output={"tile": np.asarray(dec, dtype="float32"), "i": i})

        rows = r1 - r0
        res = ResourceRequest(compute_seconds=self.cost.predict(int(rows * latent.shape[1] * latent.shape[3])),
                              resident_bytes=int(latent.nbytes),
                              peak_activation_bytes=int(latent[:, :, r0:r1, :].nbytes))
        return WorkPlan(loop_id=self.loop_id,
                        instance_id=st.instance_id,
                        kind=WorkUnitKind.VAE_TILE,
                        shape_sig=ShapeSignature(WorkUnitKind.VAE_TILE, dims=(rows, latent.shape[3])),
                        resources=res,
                        payload={"tile": i},
                        run=run,
                        label=f"vae.tile{i}")

    def advance(self, st: LoopState, result: StepResult) -> LoopState:
        st.scratch["decoded"][int(result.output["i"])] = result.output["tile"]
        st.step_idx += 1
        return st

    def finalize(self, st: LoopState) -> LoopResult:
        tiles = [t for t in st.scratch["decoded"] if t is not None]
        video = np.concatenate(tiles, axis=2) if tiles else None  # stitch row-bands back together
        return LoopResult(outputs={
            "video": video,
            "latents": st.scratch["latent"]
        },
                          metrics={"vae_tiles": float(st.step_idx)})
