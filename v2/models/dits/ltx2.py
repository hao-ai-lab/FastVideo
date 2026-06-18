# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""LTX-2 DiT facade. The backend uses ``VideoLatentShape`` (token-count helper); the transformer class
itself is constructed by the loader from the card's load_id, not imported here."""
from fastvideo.models.dits.ltx2 import VideoLatentShape  # noqa: F401
