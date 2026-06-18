"""Self-Forcing Wan2.2-A14B ModelCard — CAUSAL + Wan2.2 MoE (two experts + boundary) + optional i2v.

This is the causal *student* of the Wan2.2 MoE produced by self-forcing distillation:
  * ``recipe.method='self_forcing'`` / ``assumes_loop='chunk_rollout'`` (causal/AR; ``pp_patch`` rejected);
  * DiT  ``fastvideo.models.dits.causal_wanvideo:CausalWanTransformer3DModel`` — TWO experts
    (``transformer`` high-noise + ``transformer_2`` low-noise) switched by ``boundary_ratio`` (design_v3
    §6.2.3); the GPU adapter is the existing CausalWan/Wan torch path (pure Wan arch -> ``load_id`` only,
    no ``adapter=``, exactly like the wan_causal + wan21 cards);
  * VAE  ``fastvideo.models.vaes.wanvae:AutoencoderKLWan`` (z=16, 8x/4x) and UMT5 text encoder — reused;
  * a ``slab_kv`` (chunk-KV) cache class in addition to the text feature cache.

Two HF ids served by this package (the orchestrator maps each to its builder):
  * ``FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers`` -> ``build_sfwan22_i2v_a14b_card`` (i2v: CLIP image
    encoder + first-frame VAE conditioning; boundary 0.900, flow_shift 5.0);
  * ``rand0nmr/SFWan2.2-T2V-A14B-Diffusers``          -> ``build_sfwan22_t2v_a14b_card`` (t2v; boundary
    0.875, flow_shift 12.0).

SamplingDefaults are the fastvideo ``SF_WAN_2_2_{T2V,I2V}_A14B`` presets (h=448, w=832, 81 frames, 16 fps,
guidance 4.0, 8 DMD steps); ``dmd_denoising_steps`` + ``num_frame_per_block`` come from the SF Wan2.2
configs. The boundary-routed DMD few-step causal math lives in ``SFWan22ChunkRolloutLoop``.

BRINGUP (written-not-run): GPU-verify (a) both ``CausalWanTransformer3DModel`` experts load via the Wan
loader with ``boundary_ratio`` set; (b) the i2v ``image_encoder`` subfolder/CLIP class; (c) the few-step
DMD trajectory + final-frame trim match the fastvideo ``CausalDMDDenosingStage`` on real weights. 2x14B
bf16 needs the DiT CPU-offload the example enables (``dit_cpu_offload=True``).
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
    SamplingDefaults,
)
from v2.loop.policies import BoundaryTimestepRouting, ClassicCFG, FlowShiftPolicy, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyImageEncoder, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.sfwan22.loop import SFWan22ChunkRolloutLoop
from v2.recipes.wan21.card import stamp_wan21_checkpoints

# Self-Forcing Wan2.2 negative prompt (Chinese), verbatim from the SF_WAN_2_2_*_A14B presets. Local copy
# (recipe DATA, not model code) — kept here so the package is self-contained per the hard rules.
SFWAN22_NEG_CN = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
                  "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                  "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                  "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
                  "背景人很多，倒着走")

# fastvideo SF Wan2.2 configs (SelfForcingWan2_2_T2V480PConfig / Wan2_2_I2V_A14B_Config):
_SF_DMD_STEPS = [1000, 850, 700, 550, 350, 275, 200, 125]  # 8-step DMD schedule
_SF_NUM_FRAMES_PER_BLOCK = 7  # num_frame_per_block from the t2v example
_CAUSAL_WAN_DIT = "fastvideo.models.dits.causal_wanvideo:CausalWanTransformer3DModel"


def _build_sfwan22_card(model_id: str, *, capability: Capability, required_for: set[str], boundary_ratio: float,
                        flow_shift: float, sampling_defaults: SamplingDefaults, with_image_encoder: bool,
                        checkpoint_root: str | None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.CHUNK_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    flow = FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    # Two CausalWan experts switched at boundary_ratio·1000 (timestep space; the loop routes faithfully).
    expert = BoundaryTimestepRouting(high_noise="transformer", low_noise="transformer_2", boundary=boundary_ratio)

    def loop_factory():
        return SFWan22ChunkRolloutLoop(loop_id="chunk_rollout",
                                       cfg=cfg,
                                       flow_shift=flow,
                                       precision=precision,
                                       expert=expert,
                                       cost=cost,
                                       dmd_denoising_steps=_SF_DMD_STEPS,
                                       boundary_ratio=boundary_ratio,
                                       num_frames_per_block=_SF_NUM_FRAMES_PER_BLOCK)

    def _dit(cid: str) -> ComponentSpec:
        return ComponentSpec(component_id=cid,
                             kind="dit",
                             load_id=_CAUSAL_WAN_DIT,
                             factory=lambda inst: ToyDiT(seed=seed),
                             resident_for=["chunk_rollout"],
                             required_for=required_for)

    components = {
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.t5:T5EncoderModel",
                      factory=lambda inst: ToyTextEncoder(),
                      required_for=required_for),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.wanvae:AutoencoderKLWan",
                      factory=lambda inst: ToyVAE(),
                      required_for=required_for),
        "transformer":
        _dit("transformer"),
        "transformer_2":
        _dit("transformer_2"),
    }
    if with_image_encoder:
        # i2v CLIP vision encoder (first-frame conditioning context), like the wan21 i2v card.
        components["image_encoder"] = ComponentSpec(
            component_id="image_encoder",
            kind="image_encoder",
            load_id="fastvideo.models.encoders.clip:CLIPVisionModel",  # BRINGUP: class/subfolder
            factory=lambda inst: ToyImageEncoder(),
            required_for=required_for)

    loops = {
        "chunk_rollout":
        LoopSpec(loop_id="chunk_rollout",
                 kind=LoopKind.CHUNK_ROLLOUT,
                 work_unit_kind=WorkUnitKind.CHUNK_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer", "transformer_2"],
                 cache_policy=["feature", "slab_kv"],
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="wan",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(capability, Capability.STREAMING_VIDEO_CONTINUATION, Capability.VAE_DECODE,
                                         Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="self_forcing",
                          parents=["wan2.2-t2v-a14b"],
                          assumes_loop="chunk_rollout",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={
            "feature":
            CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True),
            "slab_kv":
            CacheContract(cache_class="slab_kv",
                          max_bytes=1 << 26,
                          reuse_across_requests=False,
                          per_component={"window": 2}),
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults,
    )
    card.validate()
    if checkpoint_root:
        stamp_wan21_checkpoints(card, checkpoint_root)
    return card


def build_sfwan22_t2v_a14b_card(model_id: str = "sfwan2.2-t2v-a14b",
                                *,
                                checkpoint_root: str | None = None) -> ModelCard:
    """Self-Forcing Wan2.2-T2V-A14B (``rand0nmr/SFWan2.2-T2V-A14B-Diffusers``). boundary 0.875,
    flow_shift 12.0; SF_WAN_2_2_T2V_A14B preset defaults (h=448, w=832, 81 frames, 8 DMD steps)."""
    return _build_sfwan22_card(model_id,
                               capability=Capability.TEXT_TO_VIDEO,
                               required_for={"t2v", "v2w"},
                               boundary_ratio=0.875,
                               flow_shift=12.0,
                               with_image_encoder=False,
                               checkpoint_root=checkpoint_root,
                               sampling_defaults=SamplingDefaults(num_steps=8,
                                                                  guidance_scale=4.0,
                                                                  height=448,
                                                                  width=832,
                                                                  num_frames=81,
                                                                  fps=16,
                                                                  negative_prompt=SFWAN22_NEG_CN))


def build_sfwan22_i2v_a14b_card(model_id: str = "sfwan2.2-i2v-a14b",
                                *,
                                checkpoint_root: str | None = None) -> ModelCard:
    """Self-Forcing Wan2.2-I2V-A14B (``FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers``). boundary 0.900,
    flow_shift 5.0 + a CLIP image encoder & first-frame VAE conditioning; SF_WAN_2_2_I2V_A14B preset."""
    return _build_sfwan22_card(model_id,
                               capability=Capability.IMAGE_TO_VIDEO,
                               required_for={"i2v"},
                               boundary_ratio=0.900,
                               flow_shift=5.0,
                               with_image_encoder=True,
                               checkpoint_root=checkpoint_root,
                               sampling_defaults=SamplingDefaults(num_steps=8,
                                                                  guidance_scale=4.0,
                                                                  height=448,
                                                                  width=832,
                                                                  num_frames=81,
                                                                  fps=16,
                                                                  negative_prompt=SFWAN22_NEG_CN))


# The package's primary card builder (the I2V Preview id is the package's primary HF id).
def build_sfwan22_card(model_id: str = "sfwan2.2-i2v-a14b", *, checkpoint_root: str | None = None) -> ModelCard:
    return build_sfwan22_i2v_a14b_card(model_id, checkpoint_root=checkpoint_root)
