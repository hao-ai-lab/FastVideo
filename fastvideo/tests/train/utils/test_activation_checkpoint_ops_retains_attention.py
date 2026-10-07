# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._library.custom_ops import OPDEFS
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

import fastvideo.train.utils.activation_checkpoint as activation_checkpoint
from fastvideo.train.utils.activation_checkpoint import apply_activation_checkpointing
from fastvideo.training.training_utils import EMA_FSDP

_TEST_OP_NAME = "fastvideo_activation_checkpoint_test::expensive_op"
_TEST_OP_CALLS = 0

# The module that registers each retained non-aten op. An op exists only after
# its module imports, so the registration checks import these first.
_RETAINED_OP_MODULES = {
    "fastvideo::_flash_attn_default_forward": "fastvideo.attention.utils.flash_attn_default",
    "fastvideo::_flash_attn_cute_forward": "fastvideo.attention.utils.flash_attn_cute",
    "fastvideo::_flash_attn_cute_varlen_forward": "fastvideo.attention.utils.flash_attn_cute",
    "fastvideo::_flash_attn_cute_fp4_forward": "fastvideo.attention.utils.flash_attn_cute",
    "fastvideo::_flash_attn_no_pad_forward": "fastvideo.attention.utils.flash_attn_no_pad",
    "fastvideo::_flash_attn_varlen_qk_no_pad_forward": "fastvideo.attention.utils.flash_attn_no_pad",
    "fastvideo_kernel::block_sparse_attn_sm90": "fastvideo_kernel.block_sparse_attn",
    "fastvideo_kernel::block_sparse_attn_sm100a": "fastvideo_kernel.block_sparse_attn",
    "fastvideo_kernel::block_sparse_attn_triton": "fastvideo_kernel.block_sparse_attn",
}


@torch.library.custom_op(_TEST_OP_NAME, mutates_args=())
def _expensive_op(value: torch.Tensor) -> torch.Tensor:
    global _TEST_OP_CALLS
    _TEST_OP_CALLS += 1
    return torch.sin(value)


@_expensive_op.register_fake
def _expensive_op_fake(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value)


def _setup_expensive_op_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(inputs[0])


def _backward_expensive_op(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    (value,) = ctx.saved_tensors
    return grad_output * torch.cos(value)


_expensive_op.register_autograd(_backward_expensive_op, setup_context=_setup_expensive_op_context)


class _ToyBlock(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + _expensive_op(self.projection(value))


class _ToyTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ToyBlock() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return value


class _RecordingLogger:

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args) -> None:
        self.warnings.append(message % args if args else message)


def _run_toy_model(
    state_dict: dict[str, torch.Tensor],
    checkpointing_type: str | None,
    apply_checkpointing=apply_activation_checkpointing,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    global _TEST_OP_CALLS
    _TEST_OP_CALLS = 0

    model = _ToyTransformer()
    model.load_state_dict(copy.deepcopy(state_dict))
    if checkpointing_type is not None:
        apply_checkpointing(model, checkpointing_type)

    value = torch.linspace(-0.5, 0.5, 12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    model.to(dtype=torch.float64)
    output = model(value)
    loss = output.square().mean()
    loss.backward()
    parameter_grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    return _TEST_OP_CALLS, output.detach(), loss.detach(), value.grad.detach().clone(), parameter_grads


def _toy_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return _ToyTransformer().state_dict()


def _retain_test_op(monkeypatch: pytest.MonkeyPatch, module=activation_checkpoint) -> None:
    op_names = module._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES | {_TEST_OP_NAME}
    monkeypatch.setattr(module, "_SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES", op_names)


def _resolve_op(op_name: str):
    namespace, name = op_name.split("::")
    return getattr(getattr(torch.ops, namespace), name, None)


def _import_or_skip(module_name: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError, RuntimeError) as exc:
        # A missing package raises ImportError, a missing shared library
        # OSError, and Triton without a GPU driver RuntimeError.
        pytest.skip(f"{module_name} is unavailable on this platform: {exc}")


@pytest.mark.parametrize(
    ("checkpointing_type", "retain_test_op", "expected_calls"),
    [
        (None, True, 4),
        ("full", True, 8),
        ("ops", True, 4),
        ("ops", False, 8),
    ],
)
def test_checkpoint_policy_controls_expensive_op_recomputation(
    monkeypatch: pytest.MonkeyPatch,
    checkpointing_type: str | None,
    retain_test_op: bool,
    expected_calls: int,
) -> None:
    if retain_test_op:
        _retain_test_op(monkeypatch)

    calls, *_ = _run_toy_model(_toy_state_dict(), checkpointing_type)

    assert calls == expected_calls


def test_checkpoint_modes_preserve_outputs_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    _retain_test_op(monkeypatch)
    state_dict = _toy_state_dict()

    baseline = _run_toy_model(state_dict, None)
    for checkpointing_type in ("full", "ops"):
        result = _run_toy_model(state_dict, checkpointing_type)
        torch.testing.assert_close(result[1], baseline[1], rtol=0, atol=0)
        torch.testing.assert_close(result[2], baseline[2], rtol=0, atol=0)
        torch.testing.assert_close(result[3], baseline[3], rtol=0, atol=0)
        assert result[4].keys() == baseline[4].keys()
        for name, baseline_grad in baseline[4].items():
            torch.testing.assert_close(result[4][name], baseline_grad, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing_type", ["full", "ops"])
def test_checkpointing_wraps_each_block_not_transformer_root(checkpointing_type: str) -> None:
    model = _ToyTransformer()

    result = apply_activation_checkpointing(model, checkpointing_type)

    assert result is model
    assert not isinstance(model, CheckpointWrapper)
    assert all(isinstance(block, CheckpointWrapper) for block in model.blocks)


@pytest.mark.parametrize("checkpointing_type", ["full", "ops"])
def test_checkpointing_rejects_transformers_without_known_block_lists(checkpointing_type: str) -> None:
    with pytest.raises(ValueError, match="Activation checkpointing is not applied successfully"):
        apply_activation_checkpointing(nn.Linear(4, 4), checkpointing_type)


@pytest.mark.parametrize(("retain_test_op", "expected_warnings"), [(True, 0), (False, 1)])
def test_ops_warns_once_when_a_block_retains_nothing(
    monkeypatch: pytest.MonkeyPatch,
    retain_test_op: bool,
    expected_warnings: int,
) -> None:
    recording_logger = _RecordingLogger()
    monkeypatch.setattr(activation_checkpoint, "logger", recording_logger)
    monkeypatch.setattr(activation_checkpoint, "_warned_nothing_retained", False)
    if retain_test_op:
        _retain_test_op(monkeypatch)

    # Two steps over four blocks: the warning must not repeat per block or step.
    for _ in range(2):
        _run_toy_model(_toy_state_dict(), "ops")

    assert len(recording_logger.warnings) == expected_warnings


def test_every_retained_op_names_its_registering_module() -> None:
    retained_op_names = activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
    unmapped = {name for name in retained_op_names if not name.startswith("aten::")} - _RETAINED_OP_MODULES.keys()

    assert not unmapped, f"Add the registering module for {unmapped} to _RETAINED_OP_MODULES"
    assert _RETAINED_OP_MODULES.keys() <= retained_op_names


@pytest.mark.parametrize(
    "op_name",
    sorted(name for name in activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
           if name.startswith("aten::")),
)
def test_retained_aten_ops_exist(op_name: str) -> None:
    assert _resolve_op(op_name) is not None


@pytest.mark.parametrize("module_name", sorted(set(_RETAINED_OP_MODULES.values())))
def test_retained_ops_match_registered_training_attention_ops(module_name: str) -> None:
    module = _import_or_skip(module_name)
    if getattr(module, "fa_version", None) == "4":
        pytest.skip("FA4 serves the default path through flash_attn_cute's op")
    expected_op_names = {name for name, owner in _RETAINED_OP_MODULES.items() if owner == module_name}
    training_op_names = {
        qualname
        for qualname, opdef in list(OPDEFS.items())
        if opdef._init_fn.__module__ == module_name and opdef._backward_fn is not None
    }

    # A renamed op stops matching its retained name and silently turns ops into full.
    unregistered = {name for name in expected_op_names if _resolve_op(name) is None}
    assert not unregistered, f"Retained op names that {module_name} no longer registers: {unregistered}"
    missing = training_op_names - activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES
    assert not missing, f"Training attention ops missing from the checkpoint save policy: {missing}"


def test_legacy_ops_policy_matches_modular_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.training.activation_checkpoint as legacy_activation_checkpoint

    assert (legacy_activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES ==
            activation_checkpoint._SELECTIVE_ACTIVATION_CHECKPOINTING_OP_NAMES)
    assert legacy_activation_checkpoint.TRANSFORMER_BLOCK_NAMES == activation_checkpoint._TRANSFORMER_BLOCK_NAMES

    _retain_test_op(monkeypatch, legacy_activation_checkpoint)
    model = _ToyTransformer()
    legacy_activation_checkpoint.apply_activation_checkpointing(model, "ops")
    calls, *_ = _run_toy_model(_toy_state_dict(), "ops", legacy_activation_checkpoint.apply_activation_checkpointing)

    assert all(isinstance(block, CheckpointWrapper) for block in model.blocks)
    assert calls == 4


@pytest.mark.parametrize("checkpointing_type", ["full", "ops"])
def test_ema_state_survives_activation_checkpointing_changes(checkpointing_type: str) -> None:
    unwrapped = _ToyTransformer()
    ema = EMA_FSDP(unwrapped, decay=0.5)
    saved_state = ema.state_dict()
    wrapped = apply_activation_checkpointing(copy.deepcopy(unwrapped), checkpointing_type)
    with torch.no_grad():
        for parameter in wrapped.parameters():
            parameter.add_(1.0)

    resumed = EMA_FSDP(wrapped, decay=0.5)
    resumed.load_state_dict(saved_state)
    resumed.update(wrapped)

    assert resumed.shadow.keys() == saved_state.keys() == unwrapped.state_dict().keys()
    with resumed.apply_to_model(wrapped):
        for name, parameter in wrapped.state_dict().items():
            torch.testing.assert_close(parameter, unwrapped.state_dict()[name] + 0.5)
    # Prefixed keys from checkpoints written before EMA names were canonical
    # still load.
    prefixed_state = {f"blocks.0._checkpoint_wrapped_module.{name[len('blocks.0.'):]}": value
                      for name, value in saved_state.items() if name.startswith("blocks.0.")}
    resumed.load_state_dict(prefixed_state)
    assert resumed.shadow.keys() == {name for name in saved_state if name.startswith("blocks.0.")}


@pytest.mark.parametrize(
    "module_name",
    [
        "fastvideo.train.models.kandinsky5.kandinsky5",
        "fastvideo.train.models.ltx2.ltx2",
        "fastvideo.train.models.minimax_h3.minimax_h3",
        "fastvideo.train.models.wan.wan",
    ],
)
def test_modular_model_plugins_use_modular_activation_checkpointing(module_name: str) -> None:
    model_module = importlib.import_module(module_name)

    assert model_module.apply_activation_checkpointing is apply_activation_checkpointing


def _checkpointing_type_of(transformer: nn.Module) -> str | None:
    wrappers = [module for module in transformer.modules() if isinstance(module, CheckpointWrapper)]
    if not wrappers:
        return None
    context_fn = wrappers[0].checkpoint_fn.keywords.get("context_fn")
    return "ops" if context_fn is activation_checkpoint._selective_checkpointing_context_fn else "full"


def _load_toy_transformer(**kwargs) -> nn.Module:
    """Stand in for load_module_from_path, honoring its pre_fsdp_transform.

    Wan hands activation checkpointing to the loader, which wraps the blocks
    before FSDP. Kandinsky5 and MiniMax-H3 pass no transform and wrap the
    returned module themselves.
    """
    transformer = _ToyTransformer()
    pre_fsdp_transform = kwargs.get("pre_fsdp_transform")
    return transformer if pre_fsdp_transform is None else pre_fsdp_transform(transformer)


@pytest.mark.parametrize(
    ("role_checkpointing_type", "fallback_checkpointing_type", "trainable", "expected_type"),
    [
        ("ops", "full", True, "ops"),
        (None, "full", True, "full"),
        (None, None, True, None),
        ("ops", None, False, None),
    ],
)
def test_wan_causal_checkpoint_safe_cache_follows_applied_checkpointing(
    monkeypatch: pytest.MonkeyPatch,
    role_checkpointing_type: str | None,
    fallback_checkpointing_type: str | None,
    trainable: bool,
    expected_type: str | None,
) -> None:
    from fastvideo.train.models.wan.wan_causal import WanCausalModel
    from fastvideo.train.utils.training_config import TrainingConfig

    training_config = TrainingConfig()
    training_config.model.enable_gradient_checkpointing_type = fallback_checkpointing_type
    monkeypatch.setattr("fastvideo.train.models.wan.wan.load_module_from_path", _load_toy_transformer)

    model = WanCausalModel(
        init_from="unused-by-test",
        training_config=training_config,
        trainable=trainable,
        enable_gradient_checkpointing_type=role_checkpointing_type,
    )

    assert _checkpointing_type_of(model.transformer) == expected_type
    assert model._should_use_checkpoint_safe_kv_cache() is (expected_type is not None)


@pytest.mark.parametrize(("checkpointing_type", "wrap_nested_model"), [("full", False), (None, True)])
def test_wan_causal_checkpoint_safe_cache_follows_overridden_loader(
    monkeypatch: pytest.MonkeyPatch,
    checkpointing_type: str | None,
    wrap_nested_model: bool,
) -> None:
    from fastvideo.train.models.wan.wan_causal import WanCausalModel
    from fastvideo.train.utils.training_config import TrainingConfig

    class _NestedLoaderModel(WanCausalModel):

        def _load_transformer(self, **kwargs) -> nn.Module:
            transformer = nn.Module()
            transformer.model = _ToyTransformer()
            if wrap_nested_model:
                apply_activation_checkpointing(transformer.model, "full")
            return transformer

    model = _NestedLoaderModel(
        init_from="unused-by-test",
        training_config=TrainingConfig(),
        enable_gradient_checkpointing_type=checkpointing_type,
    )

    assert model._should_use_checkpoint_safe_kv_cache() is wrap_nested_model


def _h3_training_config(training_config):
    training_config.pipeline_config = SimpleNamespace(dit_config=SimpleNamespace())
    training_config.data.train_batch_size = 1
    training_config.data.training_cfg_rate = 0.0
    training_config.data.preprocessed_data_type = "t2va"
    return training_config


@pytest.mark.parametrize(
    ("module_name", "class_name", "configure"),
    [
        ("fastvideo.train.models.kandinsky5.kandinsky5", "Kandinsky5Model", None),
        ("fastvideo.train.models.minimax_h3.minimax_h3", "MiniMaxH3Model", _h3_training_config),
        ("fastvideo.train.models.wan.wan", "WanModel", None),
    ],
)
@pytest.mark.parametrize(
    ("role_checkpointing_type", "fallback_checkpointing_type", "expected_type"),
    [
        ("ops", "full", "ops"),
        (None, "ops", "ops"),
        (None, None, None),
    ],
)
def test_model_plugins_wrap_with_role_or_run_wide_checkpointing_type(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
    configure,
    role_checkpointing_type: str | None,
    fallback_checkpointing_type: str | None,
    expected_type: str | None,
) -> None:
    from fastvideo.train.utils.training_config import TrainingConfig

    model_module = importlib.import_module(module_name)
    training_config = TrainingConfig()
    training_config.model.enable_gradient_checkpointing_type = fallback_checkpointing_type
    if configure is not None:
        training_config = configure(training_config)
    monkeypatch.setattr(model_module, "load_module_from_path", _load_toy_transformer)

    model = getattr(model_module, class_name)(
        init_from="unused-by-test",
        training_config=training_config,
        enable_gradient_checkpointing_type=role_checkpointing_type,
    )

    assert _checkpointing_type_of(model.transformer) == expected_type
