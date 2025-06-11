from typing import Any, Dict, Optional


def update_config_from_args(config: Any,
                            args_dict: Dict[str, Any],
                            prefix: Optional[str] = None,
                            pop_args: bool = False) -> None:
    """
    Update configuration object from arguments dictionary.
    
    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
        prefix: Prefix for the configuration parameters in the args_dict.
               If None, assumes direct attribute mapping without prefix.
    """
    # Handle top-level attributes (no prefix)
    if prefix is None:
        for key, value in args_dict.items():
            if hasattr(config, key) and value is not None:
                if key == "text_encoder_precisions" and isinstance(value, list):
                    setattr(config, key, tuple(value))
                else:
                    setattr(config, key, value)
        return

    # Handle nested attributes with prefix
    args_to_remove = []
    prefix_with_dot = f"{prefix}."
    for key, value in args_dict.items():
        if key.startswith(prefix_with_dot) and value is not None:
            attr_name = key[len(prefix_with_dot):]
            if hasattr(config, attr_name):
                setattr(config, attr_name, value)
            if pop_args:
                args_to_remove.append(key)

    if pop_args:
        for key in args_to_remove:
            args_dict.pop(key)
