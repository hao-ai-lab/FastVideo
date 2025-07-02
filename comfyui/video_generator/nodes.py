from .dit_config import DITConfig
from .inference_args import InferenceArgs
from .load_image import LoadImagePath
from .text_encoder_config import TextEncoderConfig
from .vae_config import VAEConfig
from .video_generator import VideoGenerator

NODE_CLASS_MAPPINGS = {
    "VideoGenerator": VideoGenerator,
    "InferenceArgs": InferenceArgs,
    "VAEConfig": VAEConfig,
    "TextEncoderConfig": TextEncoderConfig,
    "DITConfig": DITConfig,
    "LoadImagePath": LoadImagePath
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGenerator": "Video Generator",
    "InferenceArgs": "Inference Args",
    "VAEConfig": "VAE Config",
    "TextEncoderConfig": "Text Encoder Config",
    "DITConfig": "DIT Config",
    "LoadImagePath": "Load Image Path"
}