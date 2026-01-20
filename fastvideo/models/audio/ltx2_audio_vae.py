# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 audio VAE and vocoder wrappers using the official ltx-core implementation.
"""

from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn


def _require_ltx2():
    """Import LTX-2 audio VAE components from ltx_core.

    Attempts to import from an installed ltx_core package first.
    Falls back to local LTX-2 submodule if the package is not installed.
    """
    try:
        # Try importing from installed package first
        from ltx_core.model.audio_vae import (  # type: ignore
            AudioDecoder,
            AudioDecoderConfigurator,
            AudioEncoder,
            AudioEncoderConfigurator,
            Vocoder,
            VocoderConfigurator,
        )
    except ImportError:
        # Fall back to local LTX-2 submodule
        repo_root = Path(__file__).resolve().parents[3]
        local_core = repo_root / "LTX-2" / "packages" / "ltx-core" / "src"
        if local_core.exists() and str(local_core) not in sys.path:
            sys.path.insert(0, str(local_core))

        try:
            from ltx_core.model.audio_vae import (  # type: ignore
                AudioDecoder,
                AudioDecoderConfigurator,
                AudioEncoder,
                AudioEncoderConfigurator,
                Vocoder,
                VocoderConfigurator,
            )
        except ImportError as exc:
            raise ImportError(
                "ltx_core is required for LTX-2 audio components. "
                "Install it via pip or ensure FastVideo/LTX-2 submodule is present."
            ) from exc

    return (
        AudioDecoder,
        AudioDecoderConfigurator,
        AudioEncoder,
        AudioEncoderConfigurator,
        Vocoder,
        VocoderConfigurator,
    )


class LTX2AudioEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        (
            _AudioDecoder,
            _AudioDecoderConfigurator,
            _AudioEncoder,
            AudioEncoderConfigurator,
            _Vocoder,
            _VocoderConfigurator,
        ) = _require_ltx2()
        self.model: _AudioEncoder = AudioEncoderConfigurator.from_config(config)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)


class LTX2AudioDecoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        (
            _AudioDecoder,
            AudioDecoderConfigurator,
            _AudioEncoder,
            _AudioEncoderConfigurator,
            _Vocoder,
            _VocoderConfigurator,
        ) = _require_ltx2()
        self.model: _AudioDecoder = AudioDecoderConfigurator.from_config(config)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.model(sample)


class LTX2Vocoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        (
            _AudioDecoder,
            _AudioDecoderConfigurator,
            _AudioEncoder,
            _AudioEncoderConfigurator,
            _Vocoder,
            VocoderConfigurator,
        ) = _require_ltx2()
        self.model: _Vocoder = VocoderConfigurator.from_config(config)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.model(spectrogram)
