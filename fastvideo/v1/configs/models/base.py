from dataclasses import dataclass, fields
from typing import Dict, Any

# 1. ArchConfig contains all fields from diffuser's/transformer's config.json (i.e. all fields related to the architecture of the model)
# 2. ArchConfig should be inherited & overriden by each model arch_config
# 3. Any field in ArchConfig is fixed upon initialization, and should be hidden away from users
@dataclass
class ArchConfig:
    pass

@dataclass
class ModelConfig:
    # Every model config parameter can be categorized into either ArchConfig or everything else
    # Diffuser/Transformer parameters
    arch_config: ArchConfig = ArchConfig()

    # FastVideo-specific parameters here
    # i.e. STA, quantization, teacache

    # This should be used only when loading from transformers/diffusers
    def update_model_arch(
        self, 
        source_model_dict: Dict[str, Any]
    ) -> None:
        arch_config = self.arch_config
        valid_fields = {f.name for f in fields(arch_config)}
        
        for key, value in source_model_dict.items():
            if key in valid_fields:
                setattr(arch_config, key, value)
            else:
                raise AttributeError(f"{type(arch_config).__name__} has no field '{key}'")
            
        if hasattr(arch_config, "__post_init__"):
            arch_config.__post_init__()

    def update_model_config(
        self,
        source_model_dict: Dict[str, Any]    
    ) -> None:
        assert "arch_config" not in source_model_dict.keys(), "Source model config shouldn't contain arch_config."
        
        valid_fields = {f.name for f in fields(self)}

        for key, value in source_model_dict.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                print(f"{type(self).__name__} does not contain field '{key}'!")
                raise AttributeError(f"Invalid field: {key}")
        
        if hasattr(self, "__post_init__"):
            self.__post_init__()