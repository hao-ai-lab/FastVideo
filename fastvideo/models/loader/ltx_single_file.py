# SPDX-License-Identifier: Apache-2.0
"""Single-file LTX checkpoint support.

LTX ships one ``.safetensors`` bundle holding every component, so the two
assumptions the directory-based loaders make do not hold:

* there is no per-component ``config.json`` -- every component's config lives
  in the file's ``__metadata__`` (safetensors stores an 8-byte little-endian
  header length, then that many bytes of JSON whose ``__metadata__`` object is
  a flat ``str -> str`` map; ``safe_open(...).metadata()`` returns it);
* there is no per-component directory -- components are told apart by the
  top-level prefix on each tensor key.

This module reads that metadata and routes tensors by prefix. It does not
convert anything to a diffusers layout.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import torch
from safetensors import safe_open

from fastvideo.configs.models.dits.ltx2 import LTX2VideoConfig
from fastvideo.logger import init_logger

logger = init_logger(__name__)

# Top-level tensor key prefix per component. Verified against the LTX bundle
# header; every tensor in the file starts with one of these except
# ``duration_head.``, which has no FastVideo component today.
COMPONENT_PREFIXES: dict[str, str] = {
    "transformer": "model.diffusion_model.",
    "vae": "vae.",
    "audio_vae": "audio_vae.",
    "vocoder": "vocoder.",
    "text_encoder": "text_embedding_projection.",
}

# Both embeddings connectors are stored *under* the transformer prefix, but
# FastVideo builds and runs them in the text encoder
# (``LTX2GemmaTextEncoderModel``'s ``Embeddings1DConnector``), not in the DiT,
# so they are routed to the text encoder instead. Only the transformer prefix
# is stripped: the sub-tree name survives so the text encoder's own rename
# table can map it onto the module name, the same way the already-converted
# repo layout is remapped. Matches how ``convert_ltx2_weights.py`` splits them.
TEXT_STACK_SUBPREFIXES: tuple[str, ...] = (
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
)


def is_single_file_bundle(path: str) -> bool:
    """True when a model path names a bundle rather than a component directory."""
    return str(path).endswith(".safetensors")


@dataclass(frozen=True)
class LTXCheckpointMetadata:
    """Parsed ``__metadata__`` of a single-file LTX checkpoint.

    ``config`` holds the per-component config sections keyed by component
    (``transformer``, ``vae``, ``scheduler``, ``audio_vae``, ``vocoder``).
    ``model_version`` and ``gemma_source_checkpoint`` are carried through so
    callers can validate the text encoder against the checkpoint that trained
    it; ``gemma_source_checkpoint`` is absent on some variants.
    """

    config: dict[str, Any]
    model_version: str | None
    gemma_source_checkpoint: dict[str, Any] | None


def read_ltx_metadata(path: str) -> LTXCheckpointMetadata:
    """Read the ``__metadata__`` config out of a single-file LTX checkpoint.

    Only the header is parsed; no tensor data is touched. The remaining
    metadata entries are deliberately not returned or logged -- the bundle also
    carries a full license text and an ``encrypted_wandb_properties`` blob.
    """
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}

    config = json.loads(metadata["config"])

    transformer = config.get("transformer")
    if transformer is not None:
        # ``frequencies_precision`` is the checkpoint's name for what the arch
        # config calls ``double_precision_rope``; without this the field would
        # silently fall back to its dataclass default.
        precision = transformer.get("frequencies_precision")
        if precision is not None:
            transformer["double_precision_rope"] = precision == "float64"

    gemma_source = metadata.get("gemma_source_checkpoint")
    return LTXCheckpointMetadata(
        config=config,
        model_version=metadata.get("model_version"),
        gemma_source_checkpoint=(json.loads(gemma_source)
                                 if gemma_source is not None else None),
    )


def bundle_model_index(path: str) -> dict[str, Any]:
    """Build a ``model_index.json``-shaped dict out of a bundle's own metadata.

    The pipeline loader is written against a diffusers repo layout, which
    answers two questions: which components exist, and what class is each. A
    bundle already answers both in its ``__metadata__``, so this only reshapes
    the answer -- it mirrors the entries
    ``convert_ltx2_weights.py::_build_model_index`` writes for the converted
    directory layout, including the library each is declared under.

    A section that exists but declares no class is emitted as ``[None, None]``
    rather than dropped. ``ComposedPipelineBase.load_modules`` already treats a
    null library as "declared, but not something to build" and removes the
    component from the required set; dropping the key instead would fail its
    required-module check for a component the checkpoint does carry.

    ``text_encoder`` and ``tokenizer`` are always declared: they live outside
    the bundle, but the pipeline needs both.
    """
    model_index: dict[str, Any] = {
        # ponytail: the pipeline class is selected by the caller through
        # `override_pipeline_cls_name` (see `registry._bundle_config_info`),
        # and `load_modules` pops both of these without reading them. Nothing
        # here is entitled to name a pipeline, so these are placeholders --
        # they exist only because those pops have no default.
        "_class_name": None,
        "_diffusers_version": None,
    }
    for component, section in read_ltx_metadata(path).config.items():
        cls_name = (section.get("_class_name")
                    if isinstance(section, dict) else None)
        model_index[component] = (["diffusers", cls_name]
                                  if cls_name else [None, None])
    model_index["text_encoder"] = ["transformers", "LTX2GemmaTextEncoderModel"]
    model_index["tokenizer"] = ["transformers", "AutoTokenizer"]
    return model_index


def build_dit_config(metadata: LTXCheckpointMetadata) -> LTX2VideoConfig:
    """Build the LTX-2 DiT config from checkpoint metadata.

    Reuses ``update_model_arch``, so the metadata keys that name an arch field
    win and the rest of the section is ignored -- no hand-written constants.
    """
    config = LTX2VideoConfig()
    config.update_model_arch(metadata.config["transformer"])
    return config


def component_weights(
    path: str,
    component: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(key_with_prefix_stripped, tensor)`` for one component.

    The transformer prefix covers two owners -- see
    ``TEXT_STACK_SUBPREFIXES`` -- so keys under it are split before the
    component filter applies.

    ``safe_open`` mmaps the file and ``get_tensor`` materializes one tensor at
    a time, so iterating never holds more than a single tensor in RAM. Callers
    that need a state dict for a small component can do
    ``dict(component_weights(path, "vocoder"))``; do not do that for the
    transformer.

    ponytail: every rank reads the file itself, unlike
    ``safetensors_weights_iterator``, which has local rank 0 read and broadcast.
    Correct either way, but the ceiling is filesystem bandwidth -- N ranks read
    N copies. Add the broadcast if loading a bundle over a shared filesystem
    turns out to be the bottleneck.
    """
    prefix = COMPONENT_PREFIXES[component]
    transformer_prefix = COMPONENT_PREFIXES["transformer"]
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if key.startswith(transformer_prefix):
                name = key[len(transformer_prefix):]
                # str.startswith takes a tuple.
                owner = ("text_encoder"
                         if name.startswith(TEXT_STACK_SUBPREFIXES) else
                         "transformer")
                if owner != component:
                    continue
            elif key.startswith(prefix):
                name = key[len(prefix):]
            else:
                continue
            yield name, f.get_tensor(key)


def resolve_text_encoder_root(
    configured_path: str | None,
    override_path: str | None = None,
    metadata: LTXCheckpointMetadata | None = None,
) -> str:
    """Locate the text-encoder root that goes with a single-file bundle.

    A bundle carries every component's *weights* but no pointer to the text
    encoder, which lives outside it. The root therefore has to be declared,
    and there are exactly two ways to declare it:

    * ``override_path`` -- an explicit per-run argument, so a one-off run needs
      no config edit;
    * ``configured_path`` -- the pipeline config's encoder path, which is where
      model composition already lives and which survives files being moved.

    Deliberately absent: any search of the bundle's directory. Picking up an
    encoder nobody asked for changes what gets loaded without being requested,
    and silently picks the wrong one when two sit side by side. When neither
    source is set this raises and names both, rather than guessing.

    ``metadata`` is used only to *validate* a declared root, never to find one:
    a bundle may not declare an encoder pairing at all, so discovery cannot
    depend on it.
    """
    root = override_path or configured_path
    if not root:
        raise ValueError(
            "A single-file checkpoint does not carry its text encoder, so the "
            "encoder root must be declared. Set `gemma_model_path` in the "
            "pipeline config, or pass the text-encoder path explicitly for "
            "this run. It is not inferred from the checkpoint's directory: "
            "loading whichever encoder happens to sit beside the file would "
            "silently pick the wrong one when several are present.")

    expected = (metadata.gemma_source_checkpoint or {}).get(
        "gemma_version") if metadata is not None else None
    if expected:
        # ponytail: warn, don't raise -- the declared root is the user's
        # explicit instruction and the pairing is advisory. Promote to a hard
        # error if a mismatch ever turns out to produce silent garbage rather
        # than an obvious shape failure.
        actual = _read_encoder_version(root)
        if actual is not None and actual != expected:
            logger.warning(
                "Checkpoint expects text-encoder version %r but the encoder at "
                "%s reports %r; continuing with the declared root.", expected,
                root, actual)
    return root


def _read_encoder_version(root: str) -> str | None:
    """The encoder's declared version, or None if it declares none."""
    config_path = os.path.join(root, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("gemma_version")
    except (OSError, json.JSONDecodeError):
        return None
