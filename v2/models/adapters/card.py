"""Adapter-serving card — one base DiT + a set of swappable LoRA/ControlNet adapters (design_v3 §9.19).

Many adapters are declared as components over one resident ``transformer``; a request picks which to
apply (``DiffusionParams.adapters``). Adapters are versioned independently (the cache key's
``adapter_versions``) and hot-swappable.
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
    PrecisionContract,
    RecipeSpec,
)
from ...loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from ...parallel import ParallelPlan
from ..backend import ToyControlNet, ToyDiT, ToyLoRA, ToyTextEncoder, ToyVAE, _seed_from
from .loop import AdapterDenoiseLoop

ADAPTERS = ("lora_anime", "lora_realistic", "control_pose")   # the served adapter library


def build_adapter_card(model_id: str = "wan-adapters") -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def loop_factory():
        return AdapterDenoiseLoop(loop_id="diffusion_denoise", cfg=cfg, flow_shift=flow,
                                  precision=precision, expert=expert, cost=cost)

    components = {
        "text_encoder": ComponentSpec("text_encoder", kind="text_encoder",
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v"}),
        "transformer": ComponentSpec("transformer", kind="dit", factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["diffusion_denoise"], required_for={"t2v"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(), required_for={"t2v"}),
        # the adapter library — lightweight, optional, applied per request:
        "lora_anime": ComponentSpec("lora_anime", kind="adapter",
                                    factory=lambda inst: ToyLoRA("lora_anime", scale=0.6, seed=seed + 1),
                                    optional_for={"t2v"}),
        "lora_realistic": ComponentSpec("lora_realistic", kind="adapter",
                                        factory=lambda inst: ToyLoRA("lora_realistic", scale=0.6, seed=seed + 2),
                                        optional_for={"t2v"}),
        "control_pose": ComponentSpec("control_pose", kind="adapter",
                                      factory=lambda inst: ToyControlNet("control_pose", scale=0.8, seed=seed + 3),
                                      optional_for={"t2v"}),
    }
    loops = {
        "diffusion_denoise": LoopSpec("diffusion_denoise", kind=LoopKind.DIFFUSION_DENOISE,
                                      work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=cost,
                                      shared_weight_components=["transformer"], cache_policy=["feature"],
                                      loop_factory=loop_factory),
    }
    return ModelCard(
        model_id=model_id, family="wan", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base", assumes_loop="diffusion_denoise",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={"feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    ).validate()
