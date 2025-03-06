"""
Registry for pipeline implementations.

This module provides a registry for pipeline implementations, allowing them to be
looked up by name.
"""

from typing import Dict, Type, Any, Optional, Callable, TypeVar, cast

# Define a type variable for the pipeline class
T = TypeVar('T')

class PipelineRegistry:
    """Registry for pipeline implementations."""
    
    _registry: Dict[str, Type[Any]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Register a pipeline implementation.
        
        Args:
            name: The name to register the pipeline under.
            
        Returns:
            A decorator that registers the pipeline.
        """
        def decorator(pipeline_cls: Type[T]) -> Type[T]:
            cls._registry[name] = pipeline_cls
            return pipeline_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[Any]]:
        """
        Get a pipeline implementation by name.
        
        Args:
            name: The name of the pipeline.
            
        Returns:
            The pipeline class, or None if not found.
        """
        return cls._registry.get(name)
    
    @classmethod
    def list(cls) -> Dict[str, Type[Any]]:
        """
        List all registered pipelines.
        
        Returns:
            A dictionary of pipeline names to pipeline classes.
        """
        return cls._registry.copy()


def register_pipeline(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Register a pipeline implementation.
    
    Args:
        name: The name to register the pipeline under.
        
    Returns:
        A decorator that registers the pipeline.
    """
    return PipelineRegistry.register(name) 