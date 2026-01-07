import os

from fastvideo.version import __version__

_LIGHT_IMPORT = os.getenv("FASTVIDEO_LIGHT_IMPORT", "0") == "1"

if not _LIGHT_IMPORT:
    from fastvideo.configs.pipelines import PipelineConfig
    from fastvideo.configs.sample import SamplingParam
    from fastvideo.entrypoints.video_generator import VideoGenerator

    __all__ = [
        "VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"
    ]
else:
    __all__ = ["__version__"]
