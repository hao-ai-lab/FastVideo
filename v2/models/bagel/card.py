"""BAGEL/lance-style MoT ModelCard — the canonical vllm-omni omni model (design_v3 §1, §16, §19).

vllm-omni's ``bagel_single_stage``/``lance`` prove one resident MoT instance can run both AR
``generate_text`` and diffusion ``generate_image`` on co-resident experts in a single request — BUT
they bury that interleaving inside one opaque ``DIFFUSION`` stage the scheduler never sees inside
(``max_num_running_reqs=1``). This card expresses the same shared-weight MoT, but makes both loops
**runtime-visible, step-scheduled, and batchable**: ``generate_text`` (ar_decode) and
``generate_image`` (diffusion_denoise) both bind the one resident ``transformer``, and every AR token
and denoise step is a WorkUnit the scheduler can interleave and price. That visibility is the
differentiation the §1 thesis names.
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
from v2.models.backend import ToyMoTDiT, ToyTokenizer, ToyVAE, _seed_from
from v2.models.omni import ARDecodeLoop
from v2.models.wan21.loop import WanDenoiseLoop


def build_bagel_card(model_id: str = "bagel-mot") -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)
    dn_cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg, flow = ClassicCFG(), FlowShiftPolicy(shift=3.0)
    precision, expert = PrecisionPolicy(), NoRouting("transformer")

    def text_factory():
        return ARDecodeLoop(loop_id="generate_text", transformer_id="transformer", cost=ar_cost, max_tokens=6)

    def image_factory():
        return WanDenoiseLoop(loop_id="generate_image", cfg=cfg, flow_shift=flow,
                              precision=precision, expert=expert, cost=dn_cost)

    components = {
        "tokenizer": ComponentSpec("tokenizer", kind="tokenizer", factory=lambda inst: ToyTokenizer(),
                                   required_for={"reason", "t2i"}),
        "transformer": ComponentSpec(
            "transformer", kind="dit",
            load_id="vllm_omni.diffusion.models.bagel:BagelTransformer",
            factory=lambda inst: ToyMoTDiT(seed=seed),
            resident_for=["generate_text", "generate_image"],   # one resident copy for BOTH loops
            required_for={"reason", "t2i"}),
        "vae": ComponentSpec("vae", kind="vae", factory=lambda inst: ToyVAE(), required_for={"t2i"}),
    }
    loops = {
        "generate_text": LoopSpec("generate_text", kind=LoopKind.AR_DECODE,
                                  work_unit_kind=WorkUnitKind.AR_TOKEN, step_cost_model=ar_cost,
                                  shared_weight_components=["transformer"], cache_policy=["paged_kv"],
                                  loop_factory=text_factory),
        "generate_image": LoopSpec("generate_image", kind=LoopKind.DIFFUSION_DENOISE,
                                   work_unit_kind=WorkUnitKind.DIFFUSION_STEP, step_cost_model=dn_cost,
                                   shared_weight_components=["transformer"], cache_policy=["feature"],
                                   loop_factory=image_factory),
    }
    card = ModelCard(
        model_id=model_id, family="bagel", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_IMAGE, Capability.REASONING_TEXT,
                                         Capability.VAE_DECODE),
        recipe=RecipeSpec(method="base", assumes_loop="generate_image",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={
            "feature": CacheContract("feature", max_bytes=1 << 24, reuse_across_requests=True),
            "paged_kv": CacheContract("paged_kv", max_bytes=1 << 24, block_bytes=1 << 12,
                                      reuse_across_requests=False),
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
