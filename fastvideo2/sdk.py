"""The offline SDK — ``Model``, the loaded form of a card.

    import fastvideo2 as fv2
    model = fv2.load("wan2.1-t2v-1.3b")
    result = model.generate("a cat surfing a wave", seed=7)
    result.save("cat.mp4")

Named ``Model`` (not VideoGenerator) deliberately: the handle is the loaded
*artifact*, and one card may serve several tasks and modalities — what it can
do comes from ``model.capabilities``, not from the class name. ``generate`` is
a thin synchronous facade over the exact engine path the verifier gates
(``engine.run`` on the card's declared pipeline); there is no second
implementation of generation.

The runtime ladder this sits on (each stage consumed before the next exists):
one-shot ``run()`` (this) → async engine with admission + typed events
(``submit() -> handle.events()/result()/cancel()``) → sessions with forkable
server-side state (``branch()`` — the Dreamverse / RL-environment consumer).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from fastvideo2.card import ModelCard
from fastvideo2.engine import Instance, Output, Request
from fastvideo2.engine import load as _load_instance
from fastvideo2.engine import run as _run
from fastvideo2.pipeline import Pipeline


@dataclass
class Result:
    """Typed outcome of one request. Modality accessors return ``None`` when
    the card's pipeline didn't produce that artifact."""
    outputs: dict[str, Any]
    trace: list[dict]
    request: Request
    model_id: str
    card_digest: str
    fps: int

    @property
    def video(self) -> Any:                     # uint8 [T, H, W, C] or None
        return self.outputs.get("video")

    @property
    def image(self) -> Any:
        return self.outputs.get("image")

    @property
    def audio(self) -> Any:
        return self.outputs.get("audio")

    @property
    def latents(self) -> Any:
        out = self.outputs.get("latents")
        return out.get("latents") if isinstance(out, dict) else out

    @property
    def seconds(self) -> float:
        return sum(t["seconds"] for t in self.trace)

    def save(self, path: str, fps: int | None = None) -> str:
        """Save the primary artifact, dispatched by modality."""
        if self.video is not None:
            import imageio.v2 as imageio
            imageio.mimsave(path, list(self.video), fps=fps or self.fps, format="mp4")
            return path
        if self.image is not None:
            import imageio.v2 as imageio
            imageio.imwrite(path, self.image)
            return path
        raise ValueError(f"nothing saveable in outputs {sorted(self.outputs)}")


class Model:
    """A resident, runnable card. Construct via :func:`fastvideo2.load`; the
    explicit constructor exists for tests and custom wiring."""

    def __init__(self, card: ModelCard, pipeline: Pipeline, instance: Instance):
        self.card = card
        self.pipeline = pipeline
        self.instance = instance

    # --- discovery --------------------------------------------------------- #
    @property
    def model_id(self) -> str:
        return self.card.model_id

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self.card.capabilities

    def describe(self) -> dict:
        """The card as data — machine-readable discovery, same as the CLI."""
        return self.card.to_dict()

    # --- the one generation path ------------------------------------------- #
    def generate(self, prompt: str | None = None, *,
                 request: Request | None = None, **overrides: Any) -> Result:
        """Run one request. Either a prompt plus keyword overrides
        (seed / num_steps / guidance_scale / height / width / num_frames /
        shift / negative_prompt / capture_trajectory), or a full ``Request``.
        Unset fields resolve from the card's sampling defaults."""
        if (prompt is None) == (request is None):
            raise ValueError("pass exactly one of `prompt` or `request`")
        req = request if request is not None else Request(prompt=prompt, **overrides)
        out: Output = _run(self.instance, self.pipeline, req)
        return Result(outputs=out.outputs, trace=out.trace,
                      request=req.resolve(self.card), model_id=self.card.model_id,
                      card_digest=self.card.digest(),
                      fps=self.card.sampling_defaults.fps)

    def __repr__(self) -> str:
        return (f"Model({self.card.model_id!r}, capabilities={list(self.capabilities)}, "
                f"digest={self.card.digest()})")


def load(model_id: str, *, root: str | None = None, device: str | None = None) -> Model:
    """Resolve a model id from the catalog, load it resident, return the
    handle. HF repo strings are card ingredients, not load keys."""
    from fastvideo2.registry import resolve
    card, build_pipeline = resolve(model_id)
    return Model(card, build_pipeline(), _load_instance(card, root=root, device=device))
