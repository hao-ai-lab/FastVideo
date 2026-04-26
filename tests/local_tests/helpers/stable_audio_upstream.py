# SPDX-License-Identifier: Apache-2.0
"""Stubs to make `stable_audio_tools.models.autoencoders.OobleckEncoder` /
`OobleckDecoder` importable in-process for parity testing.

The upstream `autoencoders.py` does top-level `from ..inference.sampling
import sample` and similar imports that pull in heavy training-time
dependencies (k_diffusion, laion_clap, etc.) we don't need just to
exercise the Oobleck encoder/decoder forward path. Following the
magi_compiler-stub pattern (see REVIEW item 17), we register no-op
modules in `sys.modules` BEFORE the first `from stable_audio_tools...`
import so the chain resolves.

Only the Oobleck classes (and their building blocks: `EncoderBlock`,
`DecoderBlock`, `ResidualUnit`, `WNConv1d`, `WNConvTranspose1d`,
`get_activation`, plus `SnakeBeta` from `.blocks`) are actually
exercised after the stubs are in place.
"""
from __future__ import annotations

import sys
import types


_STUBBED_MODULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stable_audio_tools.inference.sampling", ("sample",)),
    ("stable_audio_tools.inference.utils", ("prepare_audio",)),
    ("stable_audio_tools.models.bottleneck", ("Bottleneck", "DiscreteBottleneck")),
    ("stable_audio_tools.models.diffusion",
        ("ConditionedDiffusionModel", "DAU1DCondWrapper",
         "UNet1DCondWrapper", "DiTWrapper")),
    ("stable_audio_tools.models.factory",
        ("create_pretransform_from_config", "create_bottleneck_from_config")),
    ("stable_audio_tools.models.pretransforms",
        ("Pretransform", "AutoencoderPretransform")),
    ("stable_audio_tools.models.transformer",
        ("ContinuousTransformer", "TransformerBlock", "RotaryEmbedding")),
)


def install_stubs() -> None:
    """Install no-op stubs for the unused chain. Idempotent.

    Order matters: the real `stable_audio_tools` package must be imported
    first so its `models/`, `inference/` subpackages exist as real
    packages (with `__path__`); we then overlay leaf-module stubs in
    `sys.modules` for the heavy ones we want to bypass.
    """
    # Import the real top-level package so the namespace exists. Lazy —
    # no submodules loaded yet beyond what __init__ pulls.
    import stable_audio_tools  # noqa: F401
    import stable_audio_tools.models  # noqa: F401  — needed as a package parent
    import stable_audio_tools.inference  # noqa: F401

    for mod_name, attrs in _STUBBED_MODULES:
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        for attr in attrs:
            # Placeholder class; never instantiated because OobleckEncoder
            # / OobleckDecoder don't reference these in their forward
            # path. Top-level `from x import Y` just needs Y to resolve.
            setattr(mod, attr, type(attr, (), {}))
        sys.modules[mod_name] = mod
