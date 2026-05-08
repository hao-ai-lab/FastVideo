from __future__ import annotations
# mypy: ignore-errors

import importlib
import os
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVER_DIR.parent
_DEFAULT_CLASSIFIER_DIR = _REPO_ROOT / "classifiers"
CLASSIFIER_DIR = Path(
    os.path.expandvars(os.path.expanduser(os.getenv("LTX2_CLASSIFIER_DIR", str(_DEFAULT_CLASSIFIER_DIR)))))


@dataclass(frozen=True)
class BlockedPrompt:
    index: int
    prompt: str
    error: str


def _load_fasttext_module():
    try:
        return importlib.import_module("fasttext")
    except ImportError as exc:
        raise RuntimeError("Prompt safety is enabled but the `fasttext` package is not installed.") from exc


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
        candidates.append(Path(os.path.expandvars(os.path.expanduser(env_path))))
    candidates.extend([
        CLASSIFIER_DIR / filename,
        Path(f"/home/shared/{shared_filename}"),
        Path(legacy_path),
    ])

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find the {classifier_kind} classifier.\n"
                            f"Checked:\n{checked}\n"
                            "Set LTX2_CLASSIFIER_DIR or the appropriate classifier path environment "
                            "variable.")


@cache
def load_fasttext_model(model_path: str):
    return _load_fasttext_module().load_model(model_path)


def fasttext_predict(
    model_path: str,
    text: str,
    classifier_name: str,
) -> tuple[str, float]:
    model = load_fasttext_model(model_path)
    text = text.replace("\n", " ")
    start_time = time.perf_counter()
    try:
        labels, probs = model.predict(text)
    except ValueError as error:
        if "Unable to avoid copy while creating an array" not in str(error):
            raise
        predictions = model.f.predict(f"{text}\n", 1, 0.0, "strict")
        if not predictions:
            raise ValueError("fastText returned no predictions") from error
        probs, labels = zip(*predictions, strict=False)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    identifier = labels[0].replace("__label__", "")
    confidence = float(probs[0])
    print("[safety] "
          f"{classifier_name} fastText latency={latency_ms:.2f}ms "
          f"label={identifier} confidence={confidence:.4f}")
    return identifier, confidence


def _normalize_classifier_label(identifier: str) -> str:
    return identifier.strip().lower().replace("-", "_").replace(" ", "_")


def _label_matches(
    identifier: str,
    blocked_markers: tuple[str, ...],
    safe_markers: tuple[str, ...],
) -> bool:
    normalized = _normalize_classifier_label(identifier)
    tokens = tuple(token for token in normalized.split("_") if token)

    def has_marker(marker: str) -> bool:
        normalized_marker = _normalize_classifier_label(marker)
        return (normalized == normalized_marker or normalized_marker in tokens)

    if any(has_marker(marker) for marker in safe_markers):
        return False
    return any(has_marker(marker) for marker in blocked_markers)


class PromptSafetyFilter:

    def __init__(
        self,
        *,
        nsfw_classifier=None,
        hate_speech_classifier=None,
    ):
        if nsfw_classifier is None or hate_speech_classifier is None:
            _load_fasttext_module()

        if nsfw_classifier is None:
            nsfw_model_path = resolve_classifier_path(
                "NSFW",
                "LTX2_NSFW_CLASSIFIER_PATH",
                "jigsaw_fasttext_bigrams_nsfw_final.bin",
                "/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin",
                "dolma-jigsaw-fasttext-bigrams-nsfw-final.bin",
            )
            self._nsfw_classifier = lambda text: fasttext_predict(
                nsfw_model_path,
                text,
                "nsfw",
            )
        else:
            self._nsfw_classifier = nsfw_classifier

        if hate_speech_classifier is None:
            hate_speech_model_path = resolve_classifier_path(
                "hate speech",
                "LTX2_HATESPEECH_CLASSIFIER_PATH",
                "jigsaw_fasttext_bigrams_hatespeech_final.bin",
                "/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin",
                "dolma-jigsaw-fasttext-bigrams-hatespeech-final.bin",
            )
            self._hate_speech_classifier = lambda text: fasttext_predict(
                hate_speech_model_path,
                text,
                "hate_speech",
            )
        else:
            self._hate_speech_classifier = hate_speech_classifier

    def get_prompt_safety_error(self, prompt: str) -> str | None:
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            return None

        nsfw_label, _ = self._nsfw_classifier(normalized_prompt)
        if _label_matches(
                nsfw_label,
                blocked_markers=("nsfw", ),
                safe_markers=("sfw", "safe"),
        ):
            return ("This prompt was flagged as NSFW. "
                    "You can't generate a video with this specified prompt.")

        hate_label, _ = self._hate_speech_classifier(normalized_prompt)
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
            return ("This prompt was flagged as hate speech. "
                    "You can't generate a video with this specified prompt.")

        return None

    def get_first_blocked_prompt(
        self,
        prompts: list[str],
    ) -> BlockedPrompt | None:
        for index, prompt in enumerate(prompts):
            normalized_prompt = str(prompt or "").strip()
            if not normalized_prompt:
                continue
            error = self.get_prompt_safety_error(normalized_prompt)
            if error is not None:
                return BlockedPrompt(
                    index=index,
                    prompt=normalized_prompt,
                    error=error,
                )
        return None
