from .action_module import MatrixGame3ActionModule
from .model import MatrixGame3CrossAttention, MatrixGame3TransformerBlock, MatrixGame3WanModel

__all__ = [
    "MatrixGame3WanModel",
    "MatrixGame3TransformerBlock",
    "MatrixGame3CrossAttention",
    "MatrixGame3ActionModule",
]

EntryClass = [MatrixGame3WanModel]
