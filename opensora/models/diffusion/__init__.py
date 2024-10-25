
from .opensora.modeling_opensora import OpenSora_models


Diffusion_models = {}
Diffusion_models.update(OpenSora_models)

from .opensora.modeling_opensora import OpenSora_models_class

Diffusion_models_class = {}
Diffusion_models_class.update(OpenSora_models_class)