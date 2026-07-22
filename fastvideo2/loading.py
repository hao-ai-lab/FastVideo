"""Model loading — checkpoint to live modules, standalone.

Each component of a card loads independently: ``load_component(spec, root)``
works in a REPL with no engine, no distributed init, no framework state. The
module references on cards name upstream classes (diffusers / transformers),
so the checkpoint layout is the plain diffusers layout and there is no
conversion step.

torch / diffusers / transformers are imported lazily inside functions so that
importing ``fastvideo2`` stays dependency-light.
"""
from __future__ import annotations

import hashlib
from typing import Any

from fastvideo2.card import CardError, ComponentSpec, ModelCard

_DTYPES = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}


def resolve_weights(card: ModelCard, root: str | None = None) -> str:
    """Local checkpoint root for a card: an explicit path, or the card's
    canonical HF repo resolved through the local snapshot cache."""
    import os
    if root:
        if not os.path.isdir(root):
            raise CardError(f"weights root {root!r} is not a directory")
        return root
    from huggingface_hub import snapshot_download
    return snapshot_download(card.weights)


def load_component(spec: ComponentSpec, root: str, device: str = "cpu") -> Any:
    """Materialize one component from its checkpoint subfolder.

    Weight-bearing modules come back on ``device``, in the card-declared dtype,
    in eval mode with grads off. Tokenizers come back as-is.
    """
    import importlib
    lib, _, clsname = spec.module.partition(":")
    if not lib or not clsname:
        raise CardError(f"component {spec.component_id!r}: bad module ref {spec.module!r}")
    cls = getattr(importlib.import_module(lib), clsname)

    if spec.kind == "tokenizer":
        return cls.from_pretrained(root, subfolder=spec.subfolder)

    import torch
    dtype = getattr(torch, _DTYPES[spec.dtype])
    module = cls.from_pretrained(root, subfolder=spec.subfolder, torch_dtype=dtype)
    # torch_dtype was renamed to `dtype` upstream and newer releases may ignore
    # the old kwarg — force the card-declared dtype so the contract, not the
    # loader's default, decides. (T1 fingerprints hash fp32-normalized values,
    # so only a forward would have caught this otherwise.)
    return module.to(device=device, dtype=dtype).requires_grad_(False).eval()


def component_fingerprint(component: Any, spec: ComponentSpec, slices: int = 8) -> dict:
    """A cheap identity fingerprint for the T1 gate: parameter count plus
    content hashes of the first ``slices`` parameter tensors (first 4096 bytes
    each, fp32-normalized so the hash is dtype-stable across load policies)."""
    if spec.kind == "tokenizer":
        size = getattr(component, "vocab_size", None) or len(component)
        return {"kind": "tokenizer", "vocab_size": int(size)}
    import torch
    hashes: dict[str, str] = {}
    n_params = 0
    with torch.no_grad():
        for i, (name, p) in enumerate(component.named_parameters()):
            n_params += p.numel()
            if i < slices:
                blob = p.detach().flatten()[:1024].to(torch.float32).cpu().numpy().tobytes()
                hashes[name] = hashlib.sha256(blob).hexdigest()[:16]
    return {"kind": spec.kind, "param_count": int(n_params), "slices": hashes}
