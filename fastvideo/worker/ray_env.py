# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os

import fastvideo.envs as envs
from fastvideo.logger import init_logger

logger = init_logger(__name__)


def _load_non_carry_over_env_vars(non_carry_over_file: str) -> set[str]:
    """Read the env vars that must not be copied from the driver to the Ray workers."""
    try:
        if os.path.exists(non_carry_over_file):
            with open(non_carry_over_file) as f:
                return set(json.load(f))
        return set()
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse %s. Using an empty set for non-carry-over env vars.",
            non_carry_over_file,
        )
        return set()


def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,
    additional_vars: set[str] | None = None,
    destination: str | None = None,
) -> set[str]:
    """
    Get the environment variables to copy to downstream Ray actors.

    Example use cases:
    - Copy environment variables from RayDistributedExecutor to Ray workers.
    - Copy environment variables from RayDPClient to Ray DPEngineCoreActor.

    Args:
        exclude_vars: A set of FastVideo defined environment variables to exclude
            from copying.
        additional_vars: A set of additional environment variables to copy.
            If a variable is in both exclude_vars and additional_vars, it will
            be excluded.
        destination: The destination of the environment variables.
    Returns:
        A set of environment variables to copy.
    """
    exclude_vars = exclude_vars or set()
    additional_vars = additional_vars or set()
    # This file contains a list of env vars that should not be copied
    # from the driver to the Ray workers.
    non_carry_over_file = os.path.join(envs.FASTVIDEO_CONFIG_ROOT.get(), "ray_non_carry_over_env_vars.json")
    non_carry_over_vars = _load_non_carry_over_env_vars(non_carry_over_file)
    registered_vars = {
        name
        for field in envs.environment_variables.values()
        for name in (field.name, *field.deprecated_names)
    }

    env_vars_to_copy = {
        v
        for v in registered_vars.union(additional_vars) if v not in exclude_vars and v not in non_carry_over_vars
    }

    to_destination = " to " + destination if destination is not None else ""

    logger.info(
        "RAY_NON_CARRY_OVER_ENV_VARS from config: %s",
        non_carry_over_vars,
    )
    logger.info(
        "Copying the following environment variables%s: %s",
        to_destination,
        [v for v in env_vars_to_copy if v in os.environ],
    )
    logger.info(
        "If certain env vars should NOT be copied, add them to %s file",
        non_carry_over_file,
    )

    return env_vars_to_copy
