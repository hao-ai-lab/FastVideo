"""Speculative-decoding card — a draft + target AR pair on one instance (design_v3 §2.2, §9.16)."""
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
from v2.parallel import ParallelPlan
from v2.models.backend import ToyDraftModel, ToyTargetModel, ToyTokenizer, _seed_from
from v2.models.speculative.loop import SpeculativeARLoop


def build_speculative_card(model_id: str = "spec-decode", *, spec_len: int = 4,
                           draft_agree: float = 0.7) -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)

    def spec_factory():
        return SpeculativeARLoop(loop_id="spec_decode", draft_id="draft", target_id="target",
                                 cost=ar_cost, spec_len=spec_len, max_tokens=12)

    components = {
        "tokenizer": ComponentSpec("tokenizer", kind="tokenizer", factory=lambda inst: ToyTokenizer(),
                                   required_for={"reason"}),
        "draft": ComponentSpec("draft", kind="dit", load_id="fastvideo.models.llm:DraftModel",
                               factory=lambda inst: ToyDraftModel(agree=draft_agree),
                               resident_for=["spec_decode"], required_for={"reason"}),
        "target": ComponentSpec("target", kind="dit", load_id="fastvideo.models.llm:TargetModel",
                                factory=lambda inst: ToyTargetModel(seed=seed),   # the ground-truth AR
                                resident_for=["spec_decode"], required_for={"reason"}),
    }
    loops = {
        "spec_decode": LoopSpec("spec_decode", kind=LoopKind.AR_DECODE, work_unit_kind=WorkUnitKind.AR_TOKEN,
                                step_cost_model=ar_cost, shared_weight_components=["draft", "target"],
                                cache_policy=["paged_kv"], loop_factory=spec_factory),
    }
    return ModelCard(
        model_id=model_id, family="speculative", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(Capability.REASONING_TEXT),
        recipe=RecipeSpec(method="base", assumes_loop="spec_decode",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C0, ConsistencyLevel.C1],
                          interleave_required=True),
        caches={"paged_kv": CacheContract("paged_kv", max_bytes=1 << 24, block_bytes=1 << 12,
                                          reuse_across_requests=False)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    ).validate()
