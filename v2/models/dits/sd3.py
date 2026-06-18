# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""SD3 (Stable Diffusion 3.5) DiT facade. The transformer class itself is constructed by the loader
from the card's ``load_id``; this stub only re-exports the output dataclass so the SD3 torch adapter can
type-check / unwrap the forward result symbolically (mirrors ``v2/models/dits/ltx2.py``)."""
from fastvideo.models.dits.sd3 import SD3Transformer2DModelOutput  # noqa: F401
