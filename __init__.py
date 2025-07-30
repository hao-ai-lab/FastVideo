try:
    from .comfyui.video_generator.nodes import (NODE_CLASS_MAPPINGS,
                                                NODE_DISPLAY_NAME_MAPPINGS)
    WEB_DIRECTORY = "./web"
    __all__ = [
        'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'
    ]
except ImportError:
    # ComfyUI environment not available, skip comfyui imports
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = "./web"
    __all__ = [
        'NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'
    ]
