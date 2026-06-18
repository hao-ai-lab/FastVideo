# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""LTX-2 audio VAE facade. The backend uses ``AudioLatentShape`` (audio token-count helper); the
AudioDecoder / Vocoder classes are constructed by the loaders from the card's load_id, not imported here."""
from fastvideo.models.audio.ltx2_audio_vae import AudioLatentShape  # noqa: F401
