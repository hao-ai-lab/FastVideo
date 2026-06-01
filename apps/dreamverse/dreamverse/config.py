import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SERVER_ROOT = Path(__file__).resolve().parent
_FASTVIDEO_DREAMVERSE_HOME = os.environ.get("FASTVIDEO_DREAMVERSE_HOME")
_FASTVIDEO_DREAMVERSE_FRONTEND_ROOT = os.environ.get("FASTVIDEO_DREAMVERSE_FRONTEND_ROOT")
_XDG_STATE_HOME = os.environ.get("XDG_STATE_HOME")
_DEFAULT_STATE_ROOT = (Path(_FASTVIDEO_DREAMVERSE_HOME) if _FASTVIDEO_DREAMVERSE_HOME else
                       (Path(_XDG_STATE_HOME) if _XDG_STATE_HOME else Path.home() / ".local/state") /
                       "fastvideo/dreamverse")
_OUTPUTS_ROOT = _DEFAULT_STATE_ROOT / "outputs"
_PROMPTS_ROOT = _SERVER_ROOT / "prompts"
_PROMPTS_LOCAL_ROOT = _SERVER_ROOT / "prompts.local"
_APP_ROOT = _REPO_ROOT


def _resolve_frontend_root() -> Path:
    for candidate in (
            Path(_FASTVIDEO_DREAMVERSE_FRONTEND_ROOT) if _FASTVIDEO_DREAMVERSE_FRONTEND_ROOT else None,
            _APP_ROOT / "web",
            _APP_ROOT / "prod-ui",
    ):
        if candidate is not None and candidate.is_dir():
            return candidate
    if _FASTVIDEO_DREAMVERSE_FRONTEND_ROOT:
        return Path(_FASTVIDEO_DREAMVERSE_FRONTEND_ROOT)
    return _APP_ROOT / "web"


def _resolve_frontend_static_dir_candidates() -> tuple[str, ...]:
    roots = (
        FRONTEND_ROOT,
        Path.cwd() / "apps/dreamverse/web",
        Path.cwd() / "web",
        Path("/opt/FastVideo/apps/dreamverse/web"),
    )
    candidates: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        resolved_root = root.resolve(strict=False)
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        candidates.extend(str(resolved_root / dirname) for dirname in ("out", "dist"))
    return tuple(candidates)


FRONTEND_ROOT = _resolve_frontend_root()
_CLIENT_PROMPTS_ROOT = FRONTEND_ROOT / "prompts"
_CLIENT_PROMPTS_LOCAL_ROOT = FRONTEND_ROOT / "prompts.local"
FRONTEND_STATIC_DIR_CANDIDATES = _resolve_frontend_static_dir_candidates()

# Model registry
MODEL_REGISTRY = {
    "fast-ltx2": {
        "name": "FastLTX2",
        "model_path": "FastVideo/LTX2-Distilled-Diffusers",
        "config_model_path": "FastVideo/LTX2-Distilled-Diffusers",
        "lora_repo": "FastVideo/LTX2-OmniNFT-LoRA",
    },
    "fast-ltx23": {
        "name": "FastLTX23",
        "model_path": "FastVideo/LTX-2.3-Distilled-Diffusers",
        "config_model_path": "FastVideo/LTX-2.3-Distilled-Diffusers",
        "lora_repo": "FastVideo/LTX-2.3-OmniNFT-LoRA",
    },
}

DEFAULT_MODEL_ID = "fast-ltx2"

ACTIVE_MODEL_ID = (os.getenv("DREAMVERSE_MODEL_ID", "").strip() or DEFAULT_MODEL_ID)
if ACTIVE_MODEL_ID not in MODEL_REGISTRY:
    ACTIVE_MODEL_ID = DEFAULT_MODEL_ID

# Active model configuration
MODEL_CONFIG = MODEL_REGISTRY[ACTIVE_MODEL_ID]

# Generation limits
SESSION_TIMEOUT_SECONDS = 300

# Frame settings
NUM_FRAMES = 121
FRAME_HEIGHT = 1088
FRAME_WIDTH = 1920
NUM_INFERENCE_STEPS = 5
JPEG_QUALITY = 100
BATCH_SIZE = 3

# Streaming mode:
# - legacy_jpeg: send frame_batch JSON payloads with base64 JPEGs
# - av_fmp4: send muxed fMP4 binary chunks over WebSocket
STREAM_MODE = os.getenv("STREAM_MODE", "av_fmp4").strip().lower()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = os.getenv(name)
    normalized = default.strip().lower() if value is None else value.strip().lower()
    if normalized in allowed:
        return normalized
    allowed_values = ", ".join(allowed)
    raise RuntimeError(f"Invalid {name}: {normalized!r}. Expected one of {allowed_values}.")


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in raw.split(",")]
    unique_values: list[str] = []
    for value in values:
        if not value or value in unique_values:
            continue
        unique_values.append(value)
    return unique_values


def _required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    joined_names = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable: one of {joined_names}")


def _optional_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


DEVTOOLS_ENABLED = _env_bool("FASTVIDEO_ENABLE_DEVTOOLS", False)
PROMPT_SAFETY_ENABLED = _env_bool("FASTVIDEO_ENABLE_PROMPT_SAFETY", False)
DREAMVERSE_MAX_AUTOTUNE = _env_bool("DREAMVERSE_MAX_AUTOTUNE", True)

DREAMVERSE_MODEL_PATH = (os.getenv("DREAMVERSE_MODEL_PATH", "").strip() or None)
if DREAMVERSE_MODEL_PATH:
    MODEL_CONFIG = {
        **MODEL_CONFIG,
        "model_path": DREAMVERSE_MODEL_PATH,
        "config_model_path": DREAMVERSE_MODEL_PATH,
    }

AVAILABLE_LORAS = {
    "pixar": {
        "repo": "vrgamedevgirl84/LTX_2.3_Pixar_Toon_Style_LoRa",
        "trigger": "P1x4r",
        "model": "fast-ltx23",
        "position": "prepend",
        "label": "Pixar Toon",
    },
    "transition": {
        "repo": "valiantcat/LTX-2.3-Transition-LORA",
        "trigger": "zhuanchang",
        "model": "fast-ltx23",
        "position": "append",
        "label": "Transition",
    },
}


def _active_model_key() -> str:
    return ACTIVE_MODEL_ID


def _available_styles_for_active_model() -> list[str]:
    active = _active_model_key()
    return [k for k, v in AVAILABLE_LORAS.items() if v.get("model") == active]


def _resolve_lora_spec(spec: str) -> str | None:
    spec = (spec or "").strip()
    if not spec:
        return None
    if spec.lower() == "omninft":
        return MODEL_CONFIG.get("lora_repo")
    if spec.lower() in AVAILABLE_LORAS:
        return AVAILABLE_LORAS[spec.lower()]["repo"]
    return spec


def _parse_lora_stack(raw: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        spec, _, stren = item.partition("@")
        resolved = _resolve_lora_spec(spec)
        if resolved:
            try:
                strength = float(stren) if stren.strip() else 1.0
            except ValueError:
                strength = 1.0
            out.append((resolved, strength))
    return out


DREAMVERSE_LORA_PATH = _resolve_lora_spec(os.getenv("DREAMVERSE_LORA_PATH", ""))
DREAMVERSE_LORA_NICKNAME = (os.getenv("DREAMVERSE_LORA_NICKNAME", "omninft").strip() or "omninft")
DREAMVERSE_LORA_STRENGTH = _env_float("DREAMVERSE_LORA_STRENGTH", 1.0)
DREAMVERSE_LORA_STACK = _parse_lora_stack(os.getenv("DREAMVERSE_LORA_STACK", ""))


def _resolve_devtools_paths(
    default_path: Path,
    overlay_path: Path,
    env_name: str | None = None,
) -> tuple[str, str | None]:
    env_value = os.getenv(env_name) if env_name else None
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip(), None
    if DEVTOOLS_ENABLED:
        return str(overlay_path), str(default_path)
    return str(default_path), None


# Prompt LLM configuration.
PROMPT_SUPPORTED_PROVIDERS = (
    "cerebras",
    "groq",
)
if os.getenv("FASTVIDEO_PROMPT_PROVIDER") is not None:
    _env_choice(
        "FASTVIDEO_PROMPT_PROVIDER",
        "cerebras",
        PROMPT_SUPPORTED_PROVIDERS,
    )
PROMPT_PROVIDER = "cerebras"
PROMPT_PROVIDER_RUNTIME_STAGES = (("cerebras", "groq"), )
PROMPT_PROVIDER_PRIORITY = (
    "cerebras",
    "groq",
)
PROMPT_PROVIDER_API_KEY_NAMES = {
    "cerebras": ("CEREBRAS_API_KEY", ),
    "groq": ("GROQ_API_KEY", ),
}
PROMPT_API_KEYS = {
    provider: _optional_env(*PROMPT_PROVIDER_API_KEY_NAMES[provider])
    for provider in PROMPT_SUPPORTED_PROVIDERS
}
PROMPT_API_BASE_URLS = {
    "cerebras": (os.getenv("FASTVIDEO_PROMPT_CEREBRAS_API_BASE_URL", "").strip() or None),
    "groq": (os.getenv(
        "FASTVIDEO_PROMPT_GROQ_API_BASE_URL",
        "https://api.groq.com/openai/v1",
    ).strip() or None),
}
PROMPT_API_KEY = PROMPT_API_KEYS[PROMPT_PROVIDER]
PROMPT_API_BASE_URL = PROMPT_API_BASE_URLS[PROMPT_PROVIDER]
PROMPT_MODEL = (os.getenv("FASTVIDEO_PROMPT_MODEL", "gpt-oss-120b").strip() or "gpt-oss-120b")
_PROMPT_CEREBRAS_REQUEST_MODEL = (os.getenv("FASTVIDEO_PROMPT_CEREBRAS_MODEL", PROMPT_MODEL).strip() or PROMPT_MODEL)
PROMPT_PROVIDER_MODELS = {
    "cerebras": _PROMPT_CEREBRAS_REQUEST_MODEL,
    "groq": (os.getenv(
        "FASTVIDEO_PROMPT_GROQ_MODEL",
        f"openai/{PROMPT_MODEL}",
    ).strip() or f"openai/{PROMPT_MODEL}"),
}
PROMPT_REWRITE_MODEL = PROMPT_MODEL
PROMPT_REWRITE_MODEL_OPTIONS = [PROMPT_REWRITE_MODEL]
PROMPT_TIMEOUT_MS = 20000
PROMPT_HTTP_TIMEOUT_MS = 3000
PROMPT_INITIAL_STAGE_TIMEOUT_MS = 1500
PROMPT_TEMPERATURE = 1.0
PROMPT_MAX_COMPLETION_TOKENS = 3000
(
    PROMPT_ENHANCE_SYSTEM_PROMPT_PATH,
    PROMPT_ENHANCE_SYSTEM_PROMPT_FALLBACK_PATH,
) = _resolve_devtools_paths(
    _PROMPTS_ROOT / "next_segment_system_prompt.md",
    _PROMPTS_LOCAL_ROOT / "next_segment_system_prompt.md",
    "FASTVIDEO_PROMPT_ENHANCE_SYSTEM_PROMPT_PATH",
)
(
    PROMPT_AUTO_SYSTEM_PROMPT_PATH,
    PROMPT_AUTO_SYSTEM_PROMPT_FALLBACK_PATH,
) = _resolve_devtools_paths(
    _PROMPTS_ROOT / "auto_extension_system_prompt.md",
    _PROMPTS_LOCAL_ROOT / "auto_extension_system_prompt.md",
    "FASTVIDEO_PROMPT_AUTO_SYSTEM_PROMPT_PATH",
)
(
    PROMPT_REWRITE_ALL_SYSTEM_PROMPT_PATH,
    PROMPT_REWRITE_ALL_SYSTEM_PROMPT_FALLBACK_PATH,
) = _resolve_devtools_paths(
    _PROMPTS_ROOT / "rewrite_window_system_prompt.md",
    _PROMPTS_LOCAL_ROOT / "rewrite_window_system_prompt.md",
    "FASTVIDEO_PROMPT_REWRITE_ALL_SYSTEM_PROMPT_PATH",
)
(
    PROMPT_REWRITE_USER_SYSTEM_PROMPT_PATH,
    PROMPT_REWRITE_USER_SYSTEM_PROMPT_FALLBACK_PATH,
) = _resolve_devtools_paths(
    _PROMPTS_ROOT / "rewrite_user_system_prompt.md",
    _PROMPTS_LOCAL_ROOT / "rewrite_user_system_prompt.md",
    "FASTVIDEO_PROMPT_REWRITE_USER_SYSTEM_PROMPT_PATH",
)
(
    CURATED_PRESETS_FILE_PATH,
    CURATED_PRESETS_FALLBACK_FILE_PATH,
) = _resolve_devtools_paths(
    _CLIENT_PROMPTS_ROOT / "selected_ltx2_continuation_story_presets.json",
    _CLIENT_PROMPTS_LOCAL_ROOT / "selected_ltx2_continuation_story_presets.json",
    "FASTVIDEO_CURATED_PRESETS_FILE_PATH",
)
PROMPT_REWRITE_LOG_PATH = os.getenv(
    "FASTVIDEO_PROMPT_REWRITE_LOG_PATH",
    str(_OUTPUTS_ROOT / "prompt_rewrite.jsonl"),
).strip()
PROMPT_ENHANCE_LOG_PATH = os.getenv(
    "FASTVIDEO_PROMPT_ENHANCE_LOG_PATH",
    str(_OUTPUTS_ROOT / "prompt_enhance.jsonl"),
).strip()
PROMPT_AUTO_EXTENSION_LOG_PATH = os.getenv(
    "FASTVIDEO_PROMPT_AUTO_EXTENSION_LOG_PATH",
    str(_OUTPUTS_ROOT / "prompt_auto_extension.jsonl"),
).strip()
SESSION_LOG_ROOT = os.getenv(
    "FASTVIDEO_SESSION_LOG_ROOT",
    str(_OUTPUTS_ROOT / "session_logs"),
).strip()

# Auto extension behavior.
# Sleep is used as idle backoff when no prompt source is available.
PROMPT_AUTO_SLEEP_MS = _env_int("FASTVIDEO_PROMPT_AUTO_SLEEP_MS", 120)
PROMPT_AUTO_TIMEOUT_MS = _env_int("FASTVIDEO_PROMPT_AUTO_TIMEOUT_MS", 1800)
GENERATION_SEGMENT_CAP = max(0, _env_int("FASTVIDEO_GENERATION_SEGMENT_CAP", 6))

# Startup warmup behavior. Warmup compiles segment 1 and segment 2 inference
# paths before the worker is considered ready for serving.
STARTUP_WARMUP_ENABLED = _env_bool("FASTVIDEO_ENABLE_STARTUP_WARMUP", True)
STARTUP_WARMUP_PROMPT = os.getenv(
    "FASTVIDEO_STARTUP_WARMUP_PROMPT",
    ("A cinematic drone shot over coastal cliffs at sunrise, "
     "golden light, gentle ocean waves, ultra detailed"),
).strip()
STARTUP_WARMUP_TIMEOUT_SECONDS = max(1, _env_int("FASTVIDEO_STARTUP_WARMUP_TIMEOUT_SECONDS", 2400))
