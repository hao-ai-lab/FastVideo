"""Wan-causal (self-forcing) ModelCard — causal/streaming video.

The causal student *created by* self-forcing distillation: ``recipe.method='self_forcing'``,
``assumes_loop='chunk_rollout'``. Declares a ``slab_kv`` cache class (chunk-KV) alongside the text
feature cache. The chunk_rollout loop is causal/AR, so the validator rejects ``pp_patch`` parallelism
for this card (stale KV breaks causality).
"""
from __future__ import annotations

from v2.core.enums import Capability, ConsistencyLevel, LoopKind, WorkUnitKind
from v2.core.card import (
    CacheContract,
    CapabilityMatrix,
    ComponentSpec,
    LoopSpec,
    ModelCard,
    ParallelismContract,
    ParitySpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.core.loop.policies import ClassicCFG, FlowShiftPolicy, PrecisionPolicy
from v2.core.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes._prompts import WAN_NEG_CN
from v2.recipes.wan_causal.loop import ChunkRolloutLoop


def build_wan_causal_card(model_id: str = "wan-causal-sf-1.3b",
                          *,
                          num_chunks: int = 7,
                          chunk_size: int = 3,
                          steps_per_chunk: int = 4,
                          preserve_full_context: bool = False) -> ModelCard:
    # Self-forcing DMD reference (fastvideo SelfForcingWanT2V480PConfig + causal_denoising.py): the
    # distilled student is CFG-FREE (guidance 1.0, single forward/step) and denoises with the 4-step DMD
    # schedule (dmd_denoising_steps=[1000,750,500,250], warped — reproduced by FlowShiftPolicy(5.0) over 4
    # steps); the native causal block is chunk_size=3 latent frames (7 chunks x 3 = 21 latent -> 81 video
    # frames). Earlier defaults (CFG 6.0, 2 steps, block 2) overcooked + under-denoised the output.
    seed = _seed_from(model_id)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=5.0)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)

    def loop_factory():
        return ChunkRolloutLoop(
            loop_id="chunk_rollout",
            num_chunks=num_chunks,
            chunk_size=chunk_size,
            steps_per_chunk=steps_per_chunk,
            cfg=cfg,
            flow_shift=flow,
            precision=precision,
        )

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v", "v2w"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v", "v2w"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.causal_wanvideo:CausalWanTransformer3DModel",
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["chunk_rollout"],
                      required_for={"t2v", "v2w"}),
    }
    loops = {
        "chunk_rollout":
        LoopSpec(loop_id="chunk_rollout",
                 kind=LoopKind.CHUNK_ROLLOUT,
                 work_unit_kind=WorkUnitKind.CHUNK_STEP,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature", "slab_kv"],
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="wan",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.TEXT_TO_VIDEO, Capability.STREAMING_VIDEO_CONTINUATION,
                                         Capability.VAE_DECODE),
        recipe=RecipeSpec(method="self_forcing",
                          parents=["wan2.1-1.3b"],
                          assumes_loop="chunk_rollout",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1]),
        caches={
            "feature":
            CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True),
            "slab_kv":
            CacheContract(cache_class="slab_kv",
                          max_bytes=1 << 26,
                          reuse_across_requests=False,
                          per_component={"window": (num_chunks if preserve_full_context else 2)},
                          disable_recycle=preserve_full_context),
        },
        precision=PrecisionContract(default_dtype="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=SamplingDefaults(
            num_steps=4, guidance_scale=1.0, height=480, width=832, num_frames=81, fps=16,
            negative_prompt=WAN_NEG_CN),  # CFG-free distilled, 4 DMD steps (see build_wan_causal_card)
    )
    return card.validate()
