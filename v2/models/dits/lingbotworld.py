# STUB: re-exports fastvideo until vendored (see memory: v2-vendoring-approach).
"""LingBot-World DiT facade.

The transformer class itself is constructed by the loader from the card's ``load_id``
(``fastvideo.models.dits.lingbotworld.model:LingBotWorldTransformer3DModel``), so it is not imported
here. What the v2 recipe DOES need from the fastvideo source is the camera/Plucker embedding builder
``prepare_camera_embedding`` (poses.npy + intrinsics.npy -> ``c2ws_plucker_emb [B, 6*s^2, F, H, W]``),
re-exported so the program's camera node can build the tensor without reaching into the model package.
"""
from fastvideo.models.dits.lingbotworld.cam_utils import (  # noqa: F401
    prepare_camera_embedding, )
