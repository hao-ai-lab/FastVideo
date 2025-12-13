# SPDX-License-Identifier: Apache-2.0
"""
RL Algorithms module.

This module provides different RL algorithms for video generation training:
- GRPO: Group Relative Policy Optimization (with GRPO-Guard safety)
- PPO: Proximal Policy Optimization
- DPO: Direct Preference Optimization

Each algorithm implements the BaseRLAlgorithm interface and can be used
interchangeably in the RLPipeline.
"""

from typing import Any

from fastvideo.logger import init_logger

from .base import AlgorithmOutput, BaseRLAlgorithm
from .grpo import GRPOAlgorithm

logger = init_logger(__name__)

# Registry of available algorithms
ALGORITHM_REGISTRY: dict[str, type[BaseRLAlgorithm]] = {
    "grpo": GRPOAlgorithm,
    # "ppo": PPOAlgorithm,
    # "dpo": DPOAlgorithm,
}


def create_algorithm(algorithm_name: str, config: Any) -> BaseRLAlgorithm:
    """
    Factory function to create an RL algorithm.

    Args:
        algorithm_name: Name of the algorithm ("grpo", "ppo", or "dpo")
        config: Configuration object (typically RLArgs)

    Returns:
        Instantiated algorithm

    Raises:
        ValueError: If algorithm_name is not recognized

    Example:
        >>> from fastvideo.fastvideo_args import RLArgs
        >>> config = RLArgs(rl_algorithm="grpo")
        >>> algorithm = create_algorithm("grpo", config)
    """
    algorithm_name = algorithm_name.lower()

    if algorithm_name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm: '{algorithm_name}'. "
            f"Available algorithms: {available}"
        )

    algorithm_cls = ALGORITHM_REGISTRY[algorithm_name]
    algorithm = algorithm_cls(config)

    logger.info("Created %s algorithm", algorithm.name)
    return algorithm


def register_algorithm(name: str, algorithm_cls: type[BaseRLAlgorithm]) -> None:
    """
    Register a custom algorithm.

    Args:
        name: Name to register the algorithm under
        algorithm_cls: Algorithm class (must inherit from BaseRLAlgorithm)

    Example:
        >>> class CustomAlgorithm(BaseRLAlgorithm):
        ...     pass
        >>> register_algorithm("custom", CustomAlgorithm)
    """
    if not issubclass(algorithm_cls, BaseRLAlgorithm):
        raise TypeError(
            f"Algorithm class must inherit from BaseRLAlgorithm, "
            f"got {algorithm_cls.__name__}"
        )

    ALGORITHM_REGISTRY[name.lower()] = algorithm_cls
    logger.info("Registered custom algorithm: %s", name)


def get_available_algorithms() -> list[str]:
    """
    Get list of available algorithm names.

    Returns:
        List of registered algorithm names
    """
    return list(ALGORITHM_REGISTRY.keys())


__all__ = [
    # Base classes
    "BaseRLAlgorithm",
    "AlgorithmOutput",
    # Factory functions
    "create_algorithm",
    "register_algorithm",
    "get_available_algorithms",
]
