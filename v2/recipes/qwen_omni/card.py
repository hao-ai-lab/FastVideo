"""Qwen-Omni thinker→talker→vocoder ModelCard (design_v3 §4; vllm-omni ``qwen2_5_omni`` pipeline).

The canonical vllm-omni 3-stage omni-speech model, made model-native. vllm-omni expresses it as three
opaque stages (``thinker``/``talker``/``code2wav``) with ``custom_process_input_func`` hand-offs, each
request-scheduled, the scheduler never seeing inside the AR generation or the code2wav synthesis. v2
expresses the SAME cascade as three driven loops on three components — and every thinker token, talker
token, and vocoder chunk is a runtime-visible WorkUnit (the differentiation §1 names).

It is a *third* weight-sharing topology, in the same Card/Loop/Program vocabulary:

  * Cosmos3/BAGEL — ONE shared MoT module, two loops (weight sharing within a request).
  * unified/UniRL — TWO disjoint experts, two loops, jointly RL-trained.
  * Qwen-Omni    — THREE disjoint experts (thinker, talker, vocoder), three loops, **cascaded**:
                   the talker conditions on the thinker's tokens+hidden state, the vocoder on the
                   talker's speech tokens; streaming codec→waveform.

The loop kinds were anticipated: ``LoopKind.AR_DECODE`` (its docstring names "thinker/talker") for the
two language stages, ``LoopKind.AUDIO_DECODE`` for the vocoder — no new primitive.
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
from v2.parallel import ParallelPlan
from v2.platform.backends.toy import ToyMoTDiT, ToyTalker, ToyTokenizer, ToyVocoder, _seed_from
from v2.recipes.omni import ARDecodeLoop, VocoderLoop


def build_qwen_omni_card(model_id: str = "qwen-omni-tts") -> ModelCard:
    seed = _seed_from(model_id)
    ar_cost = CostModel(kind=WorkUnitKind.AR_TOKEN, base_seconds=5e-5, per_unit_seconds=1e-7)
    wav_cost = CostModel(kind=WorkUnitKind.AUDIO_CHUNK, base_seconds=3e-5, per_unit_seconds=2e-7)

    def thinker_factory():
        return ARDecodeLoop(loop_id="thinker_decode", transformer_id="thinker", cost=ar_cost,
                            max_tokens=4, prompt_slot="prompt_tokens")

    def talker_factory():
        return ARDecodeLoop(loop_id="talker_decode", transformer_id="talker", cost=ar_cost,
                            max_tokens=4, prompt_slot="talker_prompt_tokens")

    def vocoder_factory():
        return VocoderLoop(loop_id="vocoder", vocoder_id="vocoder", cost=wav_cost,
                           chunk_tokens=2, speech_slot="speech_tokens")

    components = {
        "tokenizer": ComponentSpec("tokenizer", kind="tokenizer", factory=lambda inst: ToyTokenizer(),
                                   required_for={"reason", "t2a"}),
        "thinker": ComponentSpec(                              # stage 0: multimodal understanding + text
            "thinker", kind="dit",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Thinker",
            factory=lambda inst: ToyMoTDiT(seed=seed),
            resident_for=["thinker_decode"], required_for={"reason", "t2a"}),
        "talker": ComponentSpec(                               # stage 1: text+hidden → speech tokens
            "talker", kind="dit",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Talker",
            factory=lambda inst: ToyTalker(seed=seed + 1),
            resident_for=["talker_decode"], required_for={"t2a"}),
        "vocoder": ComponentSpec(                              # stage 2: speech tokens → waveform
            "vocoder", kind="audio_vae",
            load_id="vllm_omni.model_executor.models.qwen2_5_omni:Code2Wav",
            factory=lambda inst: ToyVocoder(seed=seed + 2),
            resident_for=["vocoder"], required_for={"t2a"}),
    }
    loops = {
        "thinker_decode": LoopSpec("thinker_decode", kind=LoopKind.AR_DECODE,
                                   work_unit_kind=WorkUnitKind.AR_TOKEN, step_cost_model=ar_cost,
                                   shared_weight_components=["thinker"], cache_policy=["paged_kv"],
                                   loop_factory=thinker_factory),
        "talker_decode": LoopSpec("talker_decode", kind=LoopKind.AR_DECODE,
                                  work_unit_kind=WorkUnitKind.AR_TOKEN, step_cost_model=ar_cost,
                                  shared_weight_components=["talker"], cache_policy=["paged_kv"],
                                  loop_factory=talker_factory),
        "vocoder": LoopSpec("vocoder", kind=LoopKind.AUDIO_DECODE,
                            work_unit_kind=WorkUnitKind.AUDIO_CHUNK, step_cost_model=wav_cost,
                            shared_weight_components=["vocoder"], cache_policy=[],
                            loop_factory=vocoder_factory),
    }
    card = ModelCard(
        model_id=model_id, family="qwen_omni", components=components, loops=loops,
        capabilities=CapabilityMatrix.of(
            Capability.REASONING_TEXT, Capability.TEXT_TO_SPEECH, Capability.POLICY_ROLLOUT),
        recipe=RecipeSpec(method="base", assumes_loop="vocoder",
                          assumes_precision="float32", consistency_required=ConsistencyLevel.C1),
        parity=ParitySpec(consistency_levels=[ConsistencyLevel.C1], interleave_required=True),
        caches={
            "paged_kv": CacheContract("paged_kv", max_bytes=1 << 24, block_bytes=1 << 12,
                                      reuse_across_requests=False),
        },
        precision=PrecisionContract(default_dtype="float32", training_precision="float32"),
        parallelism=ParallelismContract(valid_plans=[ParallelPlan.single()],
                                        default_plan=ParallelPlan.single()),
    )
    return card.validate()
