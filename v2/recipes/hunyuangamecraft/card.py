"""HunyuanGameCraft ModelCard — camera/action-conditioned interactive i2v. The registered preset is the
t2v/degenerate path; the action/camera conditioning is BRINGUP (see the recipe ``__init__`` docstring).

Architecture deltas vs Wan (all declared on the card so the recipe is self-contained):
  * DiT  ``HunyuanGameCraftTransformer3DModel`` — a flow-match DiT whose forward takes a 33-channel input
    ``cat([latent16|gt_latent16|mask1])`` (assembled by the adapter, not the DiT), a list of text states
    ``[LLaMA(4096d), CLIP-pooled(768d)]``, an optional ``camera_states`` (CameraNet Plücker), and
    ``guidance=None`` (plain ClassicCFG, no embedded guidance). Adapter ``GameCraftDiT``; the
    ``GameCraftDenoiseLoop`` does the 33ch concat (via the adapter), per-step clean-ref injection, and
    flow-match Euler.
  * VAE  ``GameCraftVAE`` — SCALAR ``scaling_factor=0.476986`` normalization (not Wan per-channel stats);
    cannot reuse ``WanVAE``.
  * Text  ``LlamaModel`` (LLaVA-LLaMA-3-8B, 4096d -> ``text_states``) and ``CLIPTextModel`` (CLIP ViT-L/14,
    768d pooled -> ``text_states_2``). Adapters ``GameCraftLlamaEncoder`` / ``GameCraftClipEncoder``.
The diffusers layout adds ``text_encoder_2/`` + ``tokenizer_2/`` subfolders the Wan stamp lacks, so this
file extends ``stamp_wan21_checkpoints`` with a GameCraft-specific stamp.
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
    ParityTestSpec,
    PrecisionContract,
    RecipeSpec,
    SamplingDefaults,
)
from v2.loop.policies import ClassicCFG, FlowShiftPolicy, NoRouting, PrecisionPolicy
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyDiT, ToyTextEncoder, ToyVAE, _seed_from
from v2.recipes.hunyuangamecraft.loop import GameCraftDenoiseLoop

# GameCraft uses an empty negative prompt by default (the preset's negative_prompt=""). Kept as a LOCAL
# module constant so this recipe does not depend on the shared ``v2/recipes/_prompts.py`` (per the
# self-contained-recipe rule). CFG still runs (uncond branch encodes the empty string).
GAMECRAFT_NEG = ""

_GAMECRAFT_DIT = "v2.recipes.hunyuangamecraft.adapter:GameCraftDiT"
_GAMECRAFT_VAE = "v2.recipes.hunyuangamecraft.adapter:GameCraftVAE"
_GAMECRAFT_LLAMA = "v2.recipes.hunyuangamecraft.adapter:GameCraftLlamaEncoder"
_GAMECRAFT_CLIP = "v2.recipes.hunyuangamecraft.adapter:GameCraftClipEncoder"


def build_hunyuangamecraft_card(model_id: str = "hunyuan-gamecraft",
                                *,
                                checkpoint_root: str | None = None,
                                flow_shift: float = 5.0,
                                sampling_defaults: SamplingDefaults | None = None) -> ModelCard:
    seed = _seed_from(model_id)
    cost = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-4, per_unit_seconds=1e-7)
    cfg = ClassicCFG()
    # Single (resolution-independent) flow-shift 5.0; no bucket lookup (GameCraft fixes 704x1280).
    flow = FlowShiftPolicy(shift=flow_shift)
    precision = PrecisionPolicy(compute_dtype="float32", scheduler_step_in_fp32=True)
    expert = NoRouting("transformer")

    def loop_factory():
        return GameCraftDenoiseLoop(loop_id="diffusion_denoise",
                                    cfg=cfg,
                                    flow_shift=flow,
                                    precision=precision,
                                    expert=expert,
                                    cost=cost)

    components = {
        # LLaVA-LLaMA-3 -> text_states (4096d). The toy CPU path uses ToyTextEncoder for both encoders.
        "text_encoder":
        ComponentSpec(component_id="text_encoder",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.llama:LlamaModel",
                      adapter=_GAMECRAFT_LLAMA,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v", "i2v"}),
        # CLIP ViT-L/14 -> text_states_2 (768d pooled).
        "text_encoder_2":
        ComponentSpec(component_id="text_encoder_2",
                      kind="text_encoder",
                      load_id="fastvideo.models.encoders.clip:CLIPTextModel",
                      adapter=_GAMECRAFT_CLIP,
                      factory=lambda inst: ToyTextEncoder(),
                      required_for={"t2v", "i2v"}),
        "vae":
        ComponentSpec(component_id="vae",
                      kind="vae",
                      load_id="fastvideo.models.vaes.gamecraftvae:GameCraftVAE",
                      adapter=_GAMECRAFT_VAE,
                      factory=lambda inst: ToyVAE(),
                      required_for={"t2v", "i2v"}),
        "transformer":
        ComponentSpec(component_id="transformer",
                      kind="dit",
                      load_id="fastvideo.models.dits.hunyuangamecraft:HunyuanGameCraftTransformer3DModel",
                      adapter=_GAMECRAFT_DIT,
                      factory=lambda inst: ToyDiT(seed=seed),
                      resident_for=["diffusion_denoise"],
                      required_for={"t2v", "i2v"}),
    }
    loops = {
        "diffusion_denoise":
        LoopSpec(loop_id="diffusion_denoise",
                 kind=LoopKind.DIFFUSION_DENOISE,
                 work_unit_kind=WorkUnitKind.DIFFUSION_STEP,
                 step_cost_model=cost,
                 shared_weight_components=["transformer"],
                 cache_policy=["feature"],
                 graph_capture="breakable_cudagraph",
                 loop_factory=loop_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="gamecraft",
        components=components,
        loops=loops,
        # IMAGE_TO_VIDEO is the real capability (interactive i2v); the registered path degenerates to a
        # standard denoise when no camera/action input is given (BRINGUP). VAE_DECODE + POLICY_ROLLOUT
        # mirror the other video cards.
        capabilities=CapabilityMatrix.of(Capability.IMAGE_TO_VIDEO, Capability.VAE_DECODE, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base",
                          assumes_loop="diffusion_denoise",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1],
                          interleave_required=True,
                          tests=[ParityTestSpec(name="denoise_trajectory", level=ConsistencyLevel.C1, tap="latents")]),
        caches={"feature": CacheContract(cache_class="feature", max_bytes=1 << 24, reuse_across_requests=True)},
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
        sampling_defaults=sampling_defaults or SamplingDefaults(num_steps=50,
                                                                guidance_scale=6.0,
                                                                height=704,
                                                                width=1280,
                                                                num_frames=33,
                                                                fps=24,
                                                                negative_prompt=GAMECRAFT_NEG),
    )
    card.validate()
    if checkpoint_root:
        stamp_gamecraft_checkpoints(card, checkpoint_root)
    return card


# Diffusers checkpoint layout for GameCraft: dual text encoders + dual tokenizers + scheduler, in addition
# to transformer/vae. ``stamp_wan21_checkpoints`` only knows a single tokenizer + lacks text_encoder_2, so
# GameCraft carries its own stamp map. (The torch adapters resolve tokenizers from ``<root>/tokenizer*``.)
_GAMECRAFT_SUBFOLDERS = {
    "transformer": "transformer",
    "vae": "vae",
    "text_encoder": "text_encoder",  # LLaVA-LLaMA-3
    "text_encoder_2": "text_encoder_2",  # CLIP ViT-L/14
}


def stamp_gamecraft_checkpoints(card: ModelCard, model_root: str) -> ModelCard:
    """Point each component's ``ComponentSpec.checkpoint`` at its weights subfolder under ``model_root`` (a
    local diffusers dir or an HF id resolved via ``snapshot_download``). Like ``stamp_wan21_checkpoints``
    but also covers GameCraft's ``text_encoder_2`` (the Wan stamp lacks it). The torch adapters load the
    sibling ``tokenizer/`` (LLaMA), ``tokenizer_2/`` (CLIP), and ``scheduler/`` subfolders from
    ``dirname(checkpoint)``. Mutates and returns the card. BRINGUP risk A."""
    import os
    if not os.path.isdir(model_root):
        from huggingface_hub import snapshot_download
        model_root = snapshot_download(model_root)
    for component_id, subfolder in _GAMECRAFT_SUBFOLDERS.items():
        if component_id in card.components:
            card.components[component_id].checkpoint = os.path.join(model_root, subfolder)
    return card
