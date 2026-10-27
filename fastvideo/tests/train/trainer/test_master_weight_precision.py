# SPDX-License-Identifier: Apache-2.0
"""PDD-family methods refuse low-precision trainable masters unless explicitly opted in."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.trainer import _verify_master_weight_precision


def _method(*, requires_fp32: bool, student_dtype: torch.dtype) -> SimpleNamespace:
    student = SimpleNamespace(_trainable=True, transformer=torch.nn.Linear(2, 2).to(student_dtype))
    # A frozen role's storage dtype never receives optimizer updates.
    teacher = SimpleNamespace(_trainable=False, transformer=torch.nn.Linear(2, 2).to(torch.bfloat16))
    return SimpleNamespace(requires_fp32_master_weights=requires_fp32,
                           _role_models={
                               "student": student,
                               "teacher": teacher
                           })


def _training_config(*, allow_low_precision: bool = False) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(allow_low_precision_master_weights=allow_low_precision))


def test_fp32_methods_reject_low_precision_trainable_masters() -> None:
    with pytest.raises(RuntimeError, match="allow_low_precision_master_weights"):
        _verify_master_weight_precision(_method(requires_fp32=True, student_dtype=torch.bfloat16), _training_config())


def test_fp32_masters_or_explicit_opt_in_start() -> None:
    _verify_master_weight_precision(_method(requires_fp32=True, student_dtype=torch.float32), _training_config())
    _verify_master_weight_precision(_method(requires_fp32=True, student_dtype=torch.bfloat16),
                                    _training_config(allow_low_precision=True))


def test_other_methods_keep_their_precision_policy() -> None:
    _verify_master_weight_precision(_method(requires_fp32=False, student_dtype=torch.bfloat16), _training_config())


def test_pdd_methods_declare_the_fp32_requirement() -> None:
    from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
    from fastvideo.train.methods.distribution_matching.dmpdd import DMPDDMethod
    from fastvideo.train.methods.trajectory_matching.pdd import PDDMethod

    assert PDDMethod.requires_fp32_master_weights
    assert DMPDDMethod.requires_fp32_master_weights
    assert not DMD2Method.requires_fp32_master_weights
