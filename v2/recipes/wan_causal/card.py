"""Wan-causal (self-forcing) ModelCard — causal/streaming video (design_v3 §4, §10).

This is the causal student *created by* self-forcing distillation: ``recipe.method='self_forcing'``,
``assumes_loop='chunk_rollout'``. It declares a ``slab_kv`` cache class (chunk-KV) in addition to the
text feature cache. The chunk_rollout loop is causal/AR, so ``pp_patch`` parallelism is rejected for
this card by the validator (stale KV breaks causality).
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
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.wan_causal.loop import ChunkRolloutLoop


def build_wan_causal_card(model_id: str = "wan-causal-sf-1.3b", *,
                          num_chunks: int = 3, chunk_size: int = 2, steps_per_chunk: int = 2,
                          training_mode: bool = False) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.CHUNK_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=5.0)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)

    def loop_factory():
        return ChunkRolloutLoop(loop_id="chunk_rollout", num_chunks=num_chunks, chunk_size=chunk_size,
                                steps_per_chunk=steps_per_chunk, cfg=cfg, flow_shift=flow,
                                precision=precision, cost=cost)

    components = {
        "text_encoder": ComponentSpec(component_id="text_encoder", kind="text_encoder",
                                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                                      factory=lambda inst: ToyTextEncoder(), required_for={"t2v", "v2w"}),
        "vae": ComponentSpec(component_id="vae", kind="vae",
                             load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                             factory=lambda inst: ToyVAE(), required_for={"t2v", "v2w"}),
        "transformer": ComponentSpec(component_id="transformer", kind="dit",
                                     load_id="fastvideo.models.dits.causal_wanvideo:CausalWanTransformer3DModel",
                                     factory=lambda inst: ToyDiT(seed=seed),
                                     resident_for=["chunk_rollout"], required_for={"t2v", "v2w"}),
    }
    loops = {
        "chunk_rollout": LoopSpec(loop_id="chunk_rollout", kind=LoopKind.CHUNK_ROLLOUT,
                                  work_unit_kind=WorkUnitKind.CHUNK_STEP, step_cost_model=cost,
                                  shared_weight_components=["transformer"],
                                  cache_policy=["feature", "slab_kv"], loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id, family="wan",
        components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.TEXT_TO_VIDEO, Capability.STREAMING_VIDEO_CONTINUATION,
            Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="self_forcing", parents=["wan2.1-1.3b"], assumes_loop="chunk_rollout",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={
            "feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True),
            "slab_kv": CacheContract(cache_class="slab_kv", max_bytes=1 << 26, reuse_across_requests=False,
                                     per_component={"window": (num_chunks if training_mode else 2)},
                                     training_mode_disables_recycle=training_mode),
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
