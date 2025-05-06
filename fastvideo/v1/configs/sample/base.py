from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from fastvideo.v1.logger import init_logger

logger = init_logger(__name__)


@dataclass
class SamplingParam:
    """
    Sampling parameters for video generation.
    """
    # All fields below are copied from ForwardBatch
    data_type: str = "video"

    # Image inputs
    image_path: Optional[str] = None

    # Text inputs
    prompt: Optional[Union[str, List[str]]] = None
    negative_prompt: Optional[str] = None
    prompt_path: Optional[str] = None
    output_path: str = "outputs/"

    # Batch info
    num_videos_per_prompt: int = 1
    seed: int = 1024

    # Original dimensions (before VAE scaling)
    num_frames: int = 125
    height: int = 720
    width: int = 1280
    fps: int = 24

    # Denoising parameters
    num_inference_steps: int = 50
    guidance_scale: float = 1.0
    guidance_rescale: float = 0.0

    # TeaCache parameters
    enable_teacache: bool = False

    # Misc
    save_video: bool = True
    return_frames: bool = False

    def __post_init__(self) -> None:
        self.data_type = "video" if self.num_frames > 1 else "image"

    def check_sampling_param(self):
        if self.prompt_path and not self.prompt_path.endswith(".txt"):
            raise ValueError("prompt_path must be a txt file")

    def update(self, source_dict: Dict[str, Any]) -> None:
        for key, value in source_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.exception("%s has no attribute %s",
                                 type(self).__name__, key)

        self.__post_init__()

    @classmethod
    def from_pretrained(cls, model_path: str) -> "SamplingParam":
        from fastvideo.v1.configs.sample.registry import (
            get_sampling_param_cls_for_name)
        sampling_cls = get_sampling_param_cls_for_name(model_path)
        if sampling_cls is not None:
            sampling_param: SamplingParam = sampling_cls()
        else:
            logger.warning(
                "Couldn't find an optimal sampling param for %s. Using the default sampling param.",
                model_path)
            sampling_param = cls()

        return sampling_param


@dataclass
class CacheParams:
    cache_type: str = "none"
