from dataclasses import dataclass

# ArchConfig contains all fields from diffuser's/transformer's config.json (i.e. all fields related to the architecture of the model)
# ArchConfig should be inherited & overriden by each model config
# Any field in ArchConfig is fixed upon initialization, and should be hidden away from users
@dataclass
class ArchConfig:
    pass

# For future use
@dataclass
class QuantizationConfig:
    pass