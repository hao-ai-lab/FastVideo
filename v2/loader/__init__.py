# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""``loader`` facade — the v2-owned component-loading seam. ``component_loader`` exposes the loader
classes (TransformerLoader / VAELoader / TextEncoderLoader / TokenizerLoader / UpsamplerLoader /
AudioDecoderLoader / VocoderLoader) that build real modules from a checkpoint. Re-exported so v2 code
imports ``v2.loader``; a vendored cutover replaces this with a slimmed v2-native loader (no caller change)."""
from fastvideo.models.loader import component_loader  # noqa: F401
