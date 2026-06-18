"""Two single-purpose generation cards for the T2I→I2V workflow (design_v3 §4, §13).

These are deliberately *separate models* — distinct weights, distinct cards — because the realistic
"text-to-image followed by image-to-video" pipeline is FLUX→Wan, not one model's two stages (that
same-card case is already covered by LTX-2's base→refine program). Chaining them is a ``Workflow``
(``program/workflow.py``), not a single-instance ``Program``.

Both reuse the *unchanged* ``WanDenoiseLoop`` — the image-conditioning for I2V lives in the I2V
*program* (it folds the conditioning image into ``text_embeds``), so no loop change is needed. That
is the point: a new pipeline is cards + a program + a workflow, never a new loop or runtime.
"""
from __future__ import annotations

from v2._enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.card import (
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
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.wan21.loop import WanDenoiseLoop


def _diffusion_card(model_id, family, loop_id, caps, tasks, *, shift):
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=shift)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def loop_factory():
        return WanDenoiseLoop(loop_id=loop_id, cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=cost)

    components = {
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      factory=lambda inst: ToyTextEncoder(), required_for=tasks),
        "transformer": ComponentSpec("transformer", kind="dit",
                                     factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=[loop_id], required_for=tasks),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(), required_for=tasks),
    }
    loops = {
        loop_id: LoopSpec(loop_id, kind=LoopKind.DIFFUSION_DENOISE,
                          work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
                          shared_weight_components=["transformer"], cache_policy=["feature"],
                          loop_factory=loop_factory),
    }
    return ModelCard(
        model_id=model_id, family=family, components=components, loops=loops,
        capabilities=CapabilityMatrix.of(*caps),
        recipe=RecipeSpec(method="base", assumes_loop=loop_id,
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    ).validate()


def build_flux_t2i_card(model_id: str = "flux-t2i") -> ModelCard:
    """Stage 1 — text → image (a single-frame diffusion). Stand-in for FLUX/SD."""
    return _diffusion_card(model_id, "flux", "t2i_denoise",
                           caps=(Capability.TEXT_TO_IMAGE, Capability.VAE_DECODE),
                           tasks={"t2i"}, shift=3.0)


def build_wan_i2v_card(model_id: str = "wan-i2v") -> ModelCard:
    """Stage 2 — text + conditioning image → video. Stand-in for Wan-I2V / SVD."""
    return _diffusion_card(model_id, "wan", "i2v_denoise",
                           caps=(Capability.IMAGE_TO_VIDEO, Capability.VAE_DECODE),
                           tasks={"i2v"}, shift=5.0)
