# SPDX-License-Identifier: Apache-2.0
"""
Common validators for pipeline stage verification.

This module provides reusable validation functions that can be used across
all pipeline stages for input/output verification.
"""

from typing import Any

import torch


class StageValidators:
    """Common validators for pipeline stages."""

    @staticmethod
    def not_none(value: Any) -> bool:
        """Check if value is not None."""
        return value is not None

    @staticmethod
    def positive_int(value: Any) -> bool:
        """Check if value is a positive integer."""
        return isinstance(value, int) and value > 0

    @staticmethod
    def positive_float(value: Any) -> bool:
        """Check if value is a positive float."""
        return isinstance(value, (int, float)) and value > 0

    @staticmethod
    def non_negative_float(value: Any) -> bool:
        """Check if value is a non-negative float."""
        return isinstance(value, (int, float)) and value >= 0

    @staticmethod
    def divisible_by(value: Any, divisor: int) -> bool:
        """Check if value is divisible by divisor."""
        return value is not None and isinstance(value,
                                                int) and value % divisor == 0

    @staticmethod
    def is_tensor(value: Any) -> bool:
        """Check if value is a torch tensor."""
        return isinstance(value, torch.Tensor)

    @staticmethod
    def tensor_with_dims(value: Any, dims: int) -> bool:
        """Check if value is a tensor with specific dimensions."""
        return isinstance(value, torch.Tensor) and value.dim() == dims

    @staticmethod
    def tensor_min_dims(value: Any, min_dims: int) -> bool:
        """Check if value is a tensor with at least min_dims dimensions."""
        return isinstance(value, torch.Tensor) and value.dim() >= min_dims

    @staticmethod
    def tensor_shape_matches(value: Any, expected_shape: tuple) -> bool:
        """Check if tensor shape matches expected shape (None for any size)."""
        if not isinstance(value, torch.Tensor):
            return False
        if len(value.shape) != len(expected_shape):
            return False
        for actual, expected in zip(value.shape, expected_shape):
            if expected is not None and actual != expected:
                return False
        return True

    @staticmethod
    def list_not_empty(value: Any) -> bool:
        """Check if value is a non-empty list."""
        return isinstance(value, list) and len(value) > 0

    @staticmethod
    def list_length(value: Any, length: int) -> bool:
        """Check if list has specific length."""
        return isinstance(value, list) and len(value) == length

    @staticmethod
    def list_min_length(value: Any, min_length: int) -> bool:
        """Check if list has at least min_length items."""
        return isinstance(value, list) and len(value) >= min_length

    @staticmethod
    def string_not_empty(value: Any) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    @staticmethod
    def string_or_list_strings(value: Any) -> bool:
        """Check if value is a string or list of strings."""
        if isinstance(value, str):
            return True
        if isinstance(value, list):
            return all(isinstance(item, str) for item in value)
        return False

    @staticmethod
    def bool_value(value: Any) -> bool:
        """Check if value is a boolean."""
        return isinstance(value, bool)

    @staticmethod
    def generator_or_list_generators(value: Any) -> bool:
        """Check if value is a Generator or list of Generators."""
        if isinstance(value, torch.Generator):
            return True
        if isinstance(value, list):
            return all(isinstance(item, torch.Generator) for item in value)
        return False


# Alias for convenience
V = StageValidators
