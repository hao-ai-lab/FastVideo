# SPDX-License-Identifier: Apache-2.0
"""Standalone Triton kernels used by specific model code paths.

Unlike ``fastvideo/attention/backends``, these are not registered with the
attention-backend selector; models import them directly.
"""
