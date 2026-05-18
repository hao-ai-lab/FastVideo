from functools import lru_cache

from .prompt_enhancer import PromptEnhancer

@lru_cache(maxsize=1)
def get_prompt_enhancer() -> PromptEnhancer:
    return PromptEnhancer()

def maybe_enhance_prompt(
    prompt: str,
    curated_prompts: set[str],
) -> str:
    normalized_prompt = prompt.strip()
    if not normalized_prompt:
        return normalized_prompt

    if normalized_prompt in curated_prompts:
        print(
            "[ENHANCE][INFO] Skipping prompt enhancement for curated prompt."
        )
        return normalized_prompt

    enhancer = get_prompt_enhancer()
    if not enhancer.api_key:
        raise RuntimeError(
            "Prompt enhancement is enabled for custom prompts, but "
            "FASTVIDEO_PROMPT_API_KEY or CEREBRAS_API_KEY is not set."
        )

    result = enhancer.enhance_prompt(normalized_prompt)
    if result.fallback_used or not result.prompt.strip():
        print(
            "[ENHANCE][WARN] Falling back to raw prompt "
            f"error={result.error}"
        )
        return normalized_prompt

    print(
        "[ENHANCE][INFO] Prompt enhanced "
        f"latency={result.latency_ms:.2f}ms model={result.model}"
    )
    return result.prompt.strip()
