"""Wan2.1-1.3B ModelCard (design_v3 §4) — the (recipe, runtime) pair for T2V.

Mirrors the real Wan2.1-1.3B wiring discovered in the repo:
  * DiT  ``fastvideo.models.dits.wanvideo:WanTransformer3DModel`` (40 layers, dim 5120)
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` (z=16, 4×/8× compression)
  * T5   ``fastvideo.models.encoders.t5:T5EncoderModel``
  * sched ``FlowUniPCMultistepScheduler`` (flow_prediction; 480p shift 3.0, 720p 5.0)
``load_id`` records those for the GPU torch adapter; ``factory`` returns the CPU toy here.
"""
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
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
)
from ...loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from ...parallel import ParallelPlan
from ..backend import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from .loop import WanDenoiseLoop


def build_wan21_card(model_id: str = "wan2.1-1.3b", *, cfg_policy=None, flow_shift: float = 3.0) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = cfg_policy or ClassicCFG()
    flow = FlowShiftPolicy(shift=flow_shift, bucket_lookup={480 * 832: 3.0, 720 * 1280: 5.0})
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return WanDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=cost)

    components = {
        "text_encoder": ComponentSpec(
            component_id="text_encoder", kind="text_encoder",
            load_id="fastvideo.models.encoders.t5:T5EncoderModel",
            factory=lambda inst: ToyTextEncoder(), required_for={"t2v", "i2v", "ti2v"}),
        "vae": ComponentSpec(
            component_id="vae", kind="vae",
            load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
            factory=lambda inst: ToyVAE(), required_for={"t2v", "i2v"}),
        "transformer": ComponentSpec(
            component_id="transformer", kind="dit",
            load_id="fastvideo.models.dits.wanvideo:WanTransformer3DModel",
            factory=lambda inst: ToyDiT(seed=seed),
            resident_for=["diffusion_denoise"], required_for={"t2v", "i2v", "ti2v"}),
    }
    loops = {
        "diffusion_denoise": LoopSpec(
            loop_id="diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
            work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
            shared_weight_components=["transformer"], cache_policy=["feature"],
            graph_capture="breakable_cudagraph", loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id, family="wan",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO, Capability.IMAGE_TO_VIDEO,
            Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1,
                                                tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24,
                                         reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(
            valid_plans=[ParallelPlan.single(),
                         ParallelPlan(axes={"sp": 2, "cfgp": 2}, mesh_order=["cfgp", "sp"])],
            default_plan=ParallelPlan.single()),
    )
    return card.validate()
