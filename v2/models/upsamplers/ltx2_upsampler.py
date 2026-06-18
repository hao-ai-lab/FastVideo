# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""LTX-2 latent upsampler facade. The backend uses ``upsample_video`` (un_normalize -> learned 2x ->
normalize); the LTX2LatentUpsampler module is constructed by the loader from the card's load_id."""
from fastvideo.models.upsamplers.ltx2_upsampler import upsample_video  # noqa: F401
