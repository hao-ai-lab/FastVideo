"""Tiled-decode card (design_v3 §4, §17) — a diffusion model whose VAE decode is a scheduled
``vae_tile`` loop, so its tiles co-schedule with denoise steps under the one WorkUnit budget."""
from __future__ import annotations

from ..._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from ...card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    CostModel,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    PrecisionContract,
    RecipeSpec,
)
from ...loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from ...parallel import ParallelPlan
from ..backend import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from ..wan21.loop import WanDenoiseLoop
from .loop import VAETileLoop


def build_tiled_card(model_id: str = "wan-tiled", *, tile_rows: int = 1) -> ModelCard:
    seed = _seed_from(model_id)
    dn_cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    tile_cost = CostModel(kind=WorkUnitKind.VAE_TILE, base_seconds=2e-5, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def denoise_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=dn_cost)

    def tile_factory():
        return VAETileLoop(loop_id="vae_tile", vae_id="vae", cost=tile_cost,
                           tile_rows=tile_rows, latent_slot="denoise_out")

    components = {
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "transformer": ComponentSpec("transformer", kind="dit", factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["diffusion_denoise"], required_for={"t2v"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(),
                             resident_for=["vae_tile"], required_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise": LoopSpec("diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
                                      work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=dn_cost,
                                      shared_weight_components=["transformer"], cache_policy=["feature"],
                                      loop_factory=denoise_factory),
        "vae_tile": LoopSpec("vae_tile", kind=LoopKind.VAE_TILE, work_unit_kind=WorkUnitKind.VAE_TILE,
                             step_cost_model=tile_cost, shared_weight_components=["vae"],
                             cache_policy=[], loop_factory=tile_factory),
    }
    return ModelCard(
        model_id=model_id, family="wan", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C0, ConsistencyLevel.C1],
                          interleave_required=True),
        caches={"feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    ).validate()
