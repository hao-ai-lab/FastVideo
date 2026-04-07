# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.api import ConfigValidationError, RunConfig, parse_config


def test_unknown_field_error_includes_nested_path() -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"generator\.engine\.bogus: unknown field",
    ):
        parse_config(
            RunConfig,
            {
                "generator": {
                    "model_path": "/models/base",
                    "engine": {"bogus": True},
                },
                "request": {},
            },
        )


def test_missing_required_field_error_includes_path() -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"generator\.model_path: missing required field",
    ):
        parse_config(
            RunConfig,
            {
                "generator": {},
                "request": {},
            },
        )


def test_invalid_literal_error_includes_path() -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"generator\.engine\.execution_backend: expected one of \['mp', 'ray'\]",
    ):
        parse_config(
            RunConfig,
            {
                "generator": {
                    "model_path": "/models/base",
                    "engine": {"execution_backend": "threaded"},
                },
                "request": {},
            },
        )


def test_invalid_nested_type_error_includes_list_path() -> None:
    with pytest.raises(
        ConfigValidationError,
        match=r"request\.plan\.stages\[0\]\.name: expected str",
    ):
        parse_config(
            RunConfig,
            {
                "generator": {"model_path": "/models/base"},
                "request": {
                    "plan": {
                        "stages": [
                            {
                                "name": 123,
                                "kind": "sample",
                            }
                        ]
                    }
                },
            },
        )
