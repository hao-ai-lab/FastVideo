"""fastvideo2 — a post-training-to-serving substrate for video models, MVP.

Four surfaces (see README.md at the repo root):
  contracts   fastvideo2.card / pipeline / loop  — frozen data cards, enforced
              stage edges, the driven-loop protocol
  reference   fastvideo2.<family>.reference       — the standalone eager oracle
  verifier    fastvideo2.verify                   — tiered gates + evidence ledger
  trace       engine identity chain               — request/stage/loop.step -> NVTX

``import fastvideo2`` is dependency-light: torch / diffusers / transformers
load lazily, only when weights are actually touched.
"""
from fastvideo2.card import ModelCard, derive
from fastvideo2.engine import Instance, Output, Request, load, run
from fastvideo2.registry import resolve

__version__ = "0.1.0"
__all__ = ["ModelCard", "derive", "Instance", "Output", "Request", "load", "run",
           "resolve", "generate", "__version__"]


def generate(model: str, prompt: str, *, root: str | None = None,
             device: str | None = None, **request_kwargs) -> Output:
    """One-call convenience: resolve, load, run.

    >>> out = fastvideo2.generate("wan2.1-t2v-1.3b", "a cat surfing", seed=7)
    >>> out.outputs["video"].shape   # [T, H, W, C] uint8
    """
    card, build_pipeline = resolve(model)
    instance = load(card, root=root, device=device)
    return run(instance, build_pipeline(), Request(prompt=prompt, **request_kwargs))
