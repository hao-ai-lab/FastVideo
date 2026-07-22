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
from fastvideo2.engine import Instance, Output, Request, run
from fastvideo2.registry import resolve
from fastvideo2.sdk import Model, Result, load

__version__ = "0.1.0"
__all__ = ["ModelCard", "derive", "Model", "Result", "load", "resolve",
           "Instance", "Output", "Request", "run", "generate", "__version__"]


def generate(model: str, prompt: str, *, root: str | None = None,
             device: str | None = None, **request_kwargs) -> Result:
    """One-call convenience over the SDK: load then generate (loads per call —
    hold a :class:`Model` via :func:`load` to amortize residency).

    >>> result = fastvideo2.generate("wan2.1-t2v-1.3b", "a cat surfing", seed=7)
    >>> result.video.shape   # [T, H, W, C] uint8
    """
    return load(model, root=root, device=device).generate(prompt, **request_kwargs)
