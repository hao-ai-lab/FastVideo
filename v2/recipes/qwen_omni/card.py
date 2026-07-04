"""Qwen-Omni thinker→talker→vocoder ModelCard (vllm-omni ``qwen2_5_omni`` pipeline).

vllm-omni expresses this as three opaque stages (``thinker``/``talker``/``code2wav``) with
``custom_process_input_func`` hand-offs, request-scheduled, the scheduler never seeing inside. v2
expresses the same cascade as three driven loops on three components, where every thinker token,
talker token, and vocoder chunk is a runtime-visible WorkUnit.

A third weight-sharing topology, in the same Card/Loop/Program vocabulary:

  * Cosmos3/BAGEL — ONE shared MoT module, two loops (weight sharing within a request).
  * multi-expert inference — TWO disjoint experts, two loops, one request.
  * Qwen-Omni    — THREE disjoint experts (thinker, talker, vocoder), three loops, cascaded:
                   the talker conditions on the thinker's tokens+hidden state, the vocoder on the
                   talker's speech tokens; streaming codec→waveform.

Reuses existing loop kinds: ``LoopKind.AR_DECODE`` for the two language stages,
``LoopKind.AUDIO_DECODE`` for the vocoder.
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
)
from v2.core.parallel import ParallelPlan
from v2.platform.backends.toy import ToyMoTDiT, ToyTalker, ToyTokenizer, ToyVocoder, _seed_from
from v2.recipes.omni import ARDecodeLoop, VocoderLoop


def build_qwen_omni_card(model_id: str = "qwen-omni-tts") -> ModelCard:
    seed = _seed_from(model_id)

    def thinker_factory():
        return ARDecodeLoop(loop_id="thinker_decode",
                            transformer_id="thinker",
                            max_tokens=4,
                            prompt_slot="prompt_tokens")

    def talker_factory():
        return ARDecodeLoop(loop_id="talker_decode",
                            transformer_id="talker",
                            max_tokens=4,
                            prompt_slot="talker_prompt_tokens")

    def vocoder_factory():
        return VocoderLoop(loop_id="vocoder", vocoder_id="vocoder", chunk_tokens=2, speech_slot="speech_tokens")

    components = {
        "tokenizer":
        ComponentSpec("tokenizer",
                      kind="tokenizer",
                      factory=lambda inst: ToyTokenizer(),
                      required_for={"reason", "t2a"}),
        "thinker":
        ComponentSpec(  # stage 0: multimodal understanding + text
            "thinker",
            kind="dit",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Thinker",
            factory=lambda inst: ToyMoTDiT(seed=seed),
            resident_for=["thinker_decode"],
            required_for={"reason", "t2a"}),
        "talker":
        ComponentSpec(  # stage 1: text+hidden → speech tokens
            "talker",
            kind="dit",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Talker",
            factory=lambda inst: ToyTalker(seed=seed + 1),
            resident_for=["talker_decode"],
            required_for={"t2a"}),
        "vocoder":
        ComponentSpec(  # stage 2: speech tokens → waveform
            "vocoder",
            kind="audio_vae",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Code2Wav",
            factory=lambda inst: ToyVocoder(seed=seed + 2),
            resident_for=["vocoder"],
            required_for={"t2a"}),
    }
    loops = {
        "thinker_decode":
        LoopSpec("thinker_decode",
                 kind=LoopKind.AR_DECODE,
                 work_unit_kind=WorkUnitKind.AR_TOKEN,
                 shared_weight_components=["thinker"],
                 cache_policy=["paged_kv"],
                 loop_factory=thinker_factory),
        "talker_decode":
        LoopSpec("talker_decode",
                 kind=LoopKind.AR_DECODE,
                 work_unit_kind=WorkUnitKind.AR_TOKEN,
                 shared_weight_components=["talker"],
                 cache_policy=["paged_kv"],
                 loop_factory=talker_factory),
        "vocoder":
        LoopSpec("vocoder",
                 kind=LoopKind.AUDIO_DECODE,
                 work_unit_kind=WorkUnitKind.AUDIO_CHUNK,
                 shared_weight_components=["vocoder"],
                 cache_policy=[],
                 loop_factory=vocoder_factory),
    }
    card = ModelCard(
        model_id=model_id,
        family="qwen_omni",
        components=components,
        loops=loops,
        capabilities=CapabilityMatrix.of(Capability.REASONING_TEXT, Capability.TEXT_TO_SPEECH),
        recipe=RecipeSpec(method="base",
                          assumes_loop="vocoder",
                          assumes_precision="float32",
                          consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1]),
        caches={
            "paged_kv": CacheContract("paged_kv", max_bytes=1 << 24, block_bytes=1 << 12, reuse_across_requests=False),
        },
        precision=PrecisionContract(default_dtype="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()], default_plan=ParallelPlan.single()),
    )
    return card.validate()
