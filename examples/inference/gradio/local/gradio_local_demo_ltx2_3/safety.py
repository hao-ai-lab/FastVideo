import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fasttext

from .config import CLASSIFIER_DIR

def resolve_classifier_path(
    classifier_kind: str,
    env_var: str,
    filename: str,
    legacy_path: str,
    shared_filename: str,
) -> str:
    candidates: list[Path] = []
    env_path = os.getenv(env_var)
    if env_path:
        candidates.append(
            Path(os.path.expandvars(os.path.expanduser(env_path)))
        )
    candidates.extend(
        [
            CLASSIFIER_DIR / filename,
            Path(f"/home/shared/{shared_filename}"),
            Path(legacy_path),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find the {classifier_kind} classifier.\n"
        f"Checked:\n{checked}\n"
        "Run "
        "`python examples/inference/gradio/local/download_fasttext_classifiers.py` "
        "or set the appropriate classifier path environment variable."
    )

def fasttext_predict(
    model_path: str,
    text: str,
    classifier_name: str,
) -> tuple[str, float]:
    model = load_fasttext_model(model_path)
    text = text.replace('\n', ' ')
    start_time = time.perf_counter()
    try:
        labels, probs = model.predict(text)
    except ValueError as error:
        if "Unable to avoid copy while creating an array" not in str(error):
            raise
        predictions = model.f.predict(f"{text}\n", 1, 0.0, "strict")
        if not predictions:
            raise ValueError("fastText returned no predictions") from error
        probs, labels = zip(*predictions)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    identifier = labels[0].replace('__label__', '')
    confidence = probs[0]
    print(
        "[safety] "
        f"{classifier_name} fastText latency={latency_ms:.2f}ms "
        f"label={identifier} confidence={float(confidence):.4f}"
    )
    return identifier, confidence

@lru_cache(maxsize=None)
def load_fasttext_model(model_path: str):
    return fasttext.load_model(model_path)

def classify_nsfw(text: str) -> tuple[str, float]:
    return fasttext_predict(
        resolve_classifier_path(
            "NSFW",
            "LTX2_NSFW_CLASSIFIER_PATH",
            "jigsaw_fasttext_bigrams_nsfw_final.bin",
            "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
            "dolma-jigsaw-fasttext-bigrams-nsfw-final.bin",
        ),
        text,
        "nsfw",
    )

def classify_toxic_speech(text: str) -> tuple[str, float]:
    return fasttext_predict(
        resolve_classifier_path(
            "hate speech",
            "LTX2_HATESPEECH_CLASSIFIER_PATH",
            "jigsaw_fasttext_bigrams_hatespeech_final.bin",
            "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
            "dolma-jigsaw-fasttext-bigrams-hatespeech-final.bin",
        ),
        text,
        "hate_speech",
    )

def _normalize_classifier_label(identifier: str) -> str:
    return identifier.strip().lower().replace("-", "_").replace(" ", "_")

def _label_matches(
    identifier: str,
    blocked_markers: tuple[str, ...],
    safe_markers: tuple[str, ...],
) -> bool:
    normalized = _normalize_classifier_label(identifier)
    if any(marker in normalized for marker in safe_markers):
        return False
    return any(marker in normalized for marker in blocked_markers)

@dataclass(frozen=True)
class PromptSafetyCheck:
    blocked: bool
    category: str | None = None
    message: str | None = None

def get_prompt_safety_check(prompt: str) -> PromptSafetyCheck:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return PromptSafetyCheck(blocked=False)

    nsfw_label, _ = classify_nsfw(normalized_prompt)
    if _label_matches(
        nsfw_label,
        blocked_markers=("nsfw",),
        safe_markers=("sfw", "safe"),
    ):
        return PromptSafetyCheck(
            blocked=True,
            category="NSFW",
            message=(
                "This request was blocked by the safety filter because it "
                "appears to contain NSFW content. Please revise the prompt "
                "and try again."
            ),
        )

    hate_label, _ = classify_toxic_speech(normalized_prompt)
    if _label_matches(
        hate_label,
        blocked_markers=("hatespeech", "hate", "toxic", "offensive", "abusive"),
        safe_markers=(
            "non_hatespeech",
            "not_hatespeech",
            "non_toxic",
            "not_toxic",
            "safe",
            "clean",
        ),
    ):
        return PromptSafetyCheck(
            blocked=True,
            category="Hate Speech",
            message=(
                "This request was blocked by the safety filter because it "
                "appears to contain hate speech or abusive content. Please "
                "revise the prompt and try again."
            ),
        )

    return PromptSafetyCheck(blocked=False)
