"""Word Error Rate for generated audio.

Per-sample. Default backend is Whisper-base; ``glm_asr`` (vendored under
``third_party/eval/glmasr/``) and ``sensevoice`` (FunASR, install-on-
demand) are selectable via ``asr_backend=``. Text is normalized per the
MagiHuman convention and CJK inputs are scored at the character level.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Any

import torch

from fastvideo.eval.metrics.base import BaseMetric
from fastvideo.eval.registry import register
from fastvideo.eval.types import MetricResult

PUNCTUATION_SET = set(string.punctuation)
PUNCTUATION = string.punctuation + "，。！？、；：「」『』（）《》【】—…"

_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0xAC00, 0xD7AF),  # Hangul
)


def _contains_cjk(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in _CJK_RANGES)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    text = re.sub(r"\s+", " ", text)
    return text


def _to_char_level(text: str) -> str:
    return " ".join(ch for ch in text if not ch.isspace())


def _prepare_for_wer(reference: str, hypothesis: str, force_char_level: bool | None) -> tuple[str, str, bool]:
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    char_level = (force_char_level if force_char_level is not None else (_contains_cjk(ref) or _contains_cjk(hyp)))
    if char_level:
        ref, hyp = _to_char_level(ref), _to_char_level(hyp)
    return ref, hyp, char_level


def _resolve_char_level(language: str | None) -> bool | None:
    """Map a language hint to an explicit char-level decision.

    Returns ``None`` if the hint is missing or unknown — callers fall
    back to CJK auto-detection.
    """
    if not isinstance(language, str):
        return None
    lang = language.lower()
    if any(t in lang for t in ("zh", "cn", "ja", "jp", "ko", "cjk", "yue", "cantonese")):
        return True
    if any(t in lang for t in ("en", "de", "fr", "es", "it", "pt")):
        return False
    return None


@register("audio.wer")
class WERMetric(BaseMetric):
    """WER metric with selectable ASR backend.

    Sample fields: ``audio``, ``reference_text`` (or ``text_prompt``),
    optional ``language``. ``instruction`` is passed to GLM-ASR.
    """

    name = "audio.wer"
    requires_reference = False
    higher_is_better = False
    needs_gpu = True
    is_set_metric = False
    dependencies = ["jiwer", "librosa"]

    def __init__(
        self,
        asr_backend: str = "whisper",
        model_name: str | None = None,
        instruction: str = "Please transcribe this audio into text",
    ) -> None:
        super().__init__()
        if asr_backend not in ("whisper", "glm_asr", "sensevoice"):
            raise ValueError(f"Unknown ASR backend '{asr_backend}'. "
                             "Supported: ['whisper', 'glm_asr', 'sensevoice']")
        self._asr_backend = asr_backend
        self._model_name = model_name
        self._instruction = instruction
        self._model: Any = None
        self._processor: Any = None

    def to(self, device):
        super().to(device)
        return self

    def setup(self) -> None:
        if self._model is not None:
            return
        if self._asr_backend == "whisper":
            self._setup_whisper()
        elif self._asr_backend == "glm_asr":
            self._setup_glm_asr()
        else:
            self._setup_sensevoice()

    def _setup_whisper(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        repo_id = self._model_name or "openai/whisper-base"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._processor = WhisperProcessor.from_pretrained(repo_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(repo_id, torch_dtype=dtype).to(self.device)
        self._model.eval()

    def _setup_glm_asr(self) -> None:
        # transformers ≤ 4.57 doesn't register ``model_type=glmasr`` and
        # the HF repo ships no remote modeling code; use our vendored copy.
        from fastvideo.third_party.eval.glmasr import register_with_auto
        register_with_auto()

        from transformers import AutoModel, AutoProcessor

        repo_id = self._model_name or "zai-org/GLM-ASR-Nano-2512"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(repo_id)
        self._model = AutoModel.from_pretrained(repo_id, torch_dtype=dtype)
        self._model.to(self.device)
        self._model.eval()

    def _setup_sensevoice(self) -> None:
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError("SenseVoice backend requires `funasr`. Install it or "
                              "switch to asr_backend='glm_asr'.") from e

        model_name = self._model_name or "FunAudioLLM/SenseVoiceSmall"
        self._model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=str(self.device),
            hub="hf",
            disable_update=True,
        )

    def _move_to_device(self, inputs: Any) -> Any:
        """Move processor outputs to device, casting only floats to model dtype.

        ``BatchFeature.to(device)`` moves tensors but doesn't cast floats,
        so log-mel ``input_features`` (float32) would hit a bf16 model.
        """
        try:
            model_dtype = next(self._model.parameters()).dtype
        except StopIteration:
            model_dtype = None

        if hasattr(inputs, "items"):
            moved: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(self.device)
                    if model_dtype is not None and torch.is_floating_point(v):
                        v = v.to(model_dtype)
                moved[k] = v
            return moved
        return inputs.to(self.device) if hasattr(inputs, "to") else inputs

    def _transcribe_glm_asr(self, audio_path: str) -> str:
        # transformers ≤ 4.57's ``apply_chat_template`` doesn't auto-extract
        # audio features, and the repo's chat template emits a stray
        # ``<|user|>`` token that breaks generate. Build the prompt by hand
        # and call ``processor.__call__`` directly so it expands the
        # ``<|pad|>`` placeholder and extracts log-mel features in one pass.
        import librosa
        prompt = (f"<|user|>\n"
                  f"<|begin_of_audio|>{self._processor.audio_token}<|end_of_audio|>\n"
                  f"{self._instruction}\n"
                  f"<|assistant|>\n")
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self._processor(text=prompt, audio=audio, return_tensors="pt")
        inputs = self._move_to_device(inputs)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)[0].strip()

    def _transcribe_whisper(self, audio_path: str) -> str:
        import librosa
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        inputs = self._processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        if torch.is_floating_point(input_features):
            input_features = input_features.to(next(self._model.parameters()).dtype)
        with torch.no_grad():
            ids = self._model.generate(input_features=input_features, max_new_tokens=256)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _transcribe_sensevoice(self, audio_path: str) -> str:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        res = self._model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        return rich_transcription_postprocess(res[0]["text"]).strip()

    def _transcribe(self, audio_path: str) -> str:
        if self._asr_backend == "whisper":
            return self._transcribe_whisper(audio_path)
        if self._asr_backend == "glm_asr":
            return self._transcribe_glm_asr(audio_path)
        return self._transcribe_sensevoice(audio_path)

    @torch.no_grad()
    def compute(self, sample: dict) -> MetricResult:
        if self._model is None:
            self.setup()

        audio = sample.get("audio")
        if audio is None:
            return self._skip(sample, "missing 'audio'")
        text = sample.get("reference_text", sample.get("text_prompt"))
        if text is None:
            return self._skip(sample, "missing reference_text/text_prompt")
        language = sample.get("language")

        import jiwer

        asr = self._transcribe(audio)
        if set(asr).issubset(PUNCTUATION_SET):
            asr = ""

        force_char_level = _resolve_char_level(language)
        ref_for_wer, hyp_for_wer, char_level = _prepare_for_wer(text, asr, force_char_level)
        wer = jiwer.wer(ref_for_wer, hyp_for_wer)

        return MetricResult(
            name=self.name,
            score=float(wer),
            details={
                "transcription": _normalize_text(asr),
                "reference_text": _normalize_text(text),
                "asr_backend": self._asr_backend,
                "char_level": char_level,
            },
        )
