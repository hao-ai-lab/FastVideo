from fastvideo.models.dits.trackwan.model import (
    CausalTrackWanTransformer3DModel,
    TrackWanTransformer3DModel,
)
from fastvideo.models.dits.trackwan.track_encoder import TrackEncoder

__all__ = [
    "TrackEncoder",
    "TrackWanTransformer3DModel",
    "CausalTrackWanTransformer3DModel",
]

# Entry points for model registry discovery.
EntryClass = [
    TrackWanTransformer3DModel,
    CausalTrackWanTransformer3DModel,
]
