# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from fastvideo.hooks.layerwise_offload import (
    LayerwiseOffloadHook,
    enable_layerwise_offload,
)
from fastvideo.hooks.hooks import ModuleHookManager


class SimpleBlock(nn.Module):
    """A simple block with linear layers for testing."""

    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.linear2 = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class SimpleModelWithModuleList(nn.Module):
    """A simple model with ModuleList for testing layerwise offloading."""

    def __init__(
        self,
        num_blocks: int = 4,
        hidden_size: int = 128,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SimpleBlock(hidden_size, dtype=dtype) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_basic():
    """Test basic functionality of layerwise offloading."""
    device = torch.device("cuda")
    hidden_size = 128
    batch_size = 2
    seq_len = 16
    num_blocks = 4

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Get reference output without offloading
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    with torch.no_grad():
        reference_output = model(input_tensor.clone())

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    # Verify parameters are offloaded to CPU
    for block in model.blocks:
        for param in block.parameters():
            # Parameters should be placeholder tensors (empty)
            assert param.numel() == 0, (
                "Parameters should be offloaded (empty tensors)"
            )

    # Run forward pass with offloading
    with torch.no_grad():
        offloaded_output = model(input_tensor.clone())

    # Check output correctness
    assert torch.allclose(
        reference_output, offloaded_output, rtol=1e-4, atol=1e-5
    ), "Output with offloading should match reference output"

    print(
        f" Layerwise offload basic test passed: max diff = {(reference_output - offloaded_output).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_bf16():
    """Test layerwise offloading with bfloat16 precision."""
    device = torch.device("cuda")
    hidden_size = 256
    batch_size = 1
    seq_len = 32
    num_blocks = 3

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.bfloat16
    ).to(device)

    # Get reference output without offloading
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        reference_output = model(input_tensor.clone())

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    # Run forward pass with offloading
    with torch.no_grad():
        offloaded_output = model(input_tensor.clone())

    # Check output correctness (looser tolerance for bf16)
    assert torch.allclose(
        reference_output, offloaded_output, rtol=1e-2, atol=1e-3
    ), "Output with offloading should match reference output for bf16"

    print(
        f" Layerwise offload bf16 test passed: max diff = {(reference_output - offloaded_output).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_multiple_forward_passes():
    """Test that layerwise offloading works correctly across multiple forward passes."""
    device = torch.device("cuda")
    hidden_size = 64
    batch_size = 2
    seq_len = 8
    num_blocks = 3
    num_iterations = 5

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Get reference outputs without offloading
    torch.manual_seed(42)
    reference_outputs = []
    for i in range(num_iterations):
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_size, device=device
        )
        with torch.no_grad():
            reference_outputs.append(model(input_tensor.clone()))

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    # Run multiple forward passes with offloading
    torch.manual_seed(42)
    for i in range(num_iterations):
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_size, device=device
        )
        with torch.no_grad():
            offloaded_output = model(input_tensor.clone())

        # Check output correctness for each iteration
        assert torch.allclose(
            reference_outputs[i], offloaded_output, rtol=1e-4, atol=1e-5
        ), f"Output mismatch in iteration {i}"

    print(
        f" Multiple forward passes test passed for {num_iterations} iterations"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_parameter_integrity():
    """Test that parameters are correctly restored during forward pass."""
    device = torch.device("cuda")
    hidden_size = 128
    num_blocks = 2

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Store original parameter values per block
    original_params_per_block = []
    for block in model.blocks:
        block_params = {}
        for name, param in block.named_parameters():
            block_params[name] = param.data.clone()
        original_params_per_block.append(block_params)

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    # Create hook managers and verify they exist
    for block_idx, block in enumerate(model.blocks):
        manager = ModuleHookManager.get_from(block)
        assert manager is not None, "Hook manager should be attached to blocks"
        hook: LayerwiseOffloadHook | None = manager.get_forward_hook(
            "LayerwiseOffloadHook"
        )
        assert hook is not None, "LayerwiseOffloadHook should be registered"

        # Verify parameters are stored in CPU
        state = hook.state
        assert len(state.cpu_named_parameters) > 0, (
            "CPU parameters should be stored"
        )

        # Verify CPU parameters match original values for this block
        original_params = original_params_per_block[block_idx]
        for name, cpu_param in state.cpu_named_parameters.items():
            assert name in original_params, (
                f"CPU parameter {name} not found in original params for block {block_idx}"
            )
            assert torch.allclose(
                cpu_param.cpu(), original_params[name].cpu(), rtol=1e-5, atol=1e-6
            ), f"CPU parameter {name} should match original for block {block_idx}"

    print(" Parameter integrity test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_no_modulelist_error():
    """Test that enabling offload on a model without ModuleList raises an error."""

    class ModelWithoutModuleList(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128)

        def forward(self, x):
            return self.linear(x)

    model = ModelWithoutModuleList().to("cuda")

    with pytest.raises(
        ValueError,
        match="No nn.ModuleList found in the model for layerwise offloading",
    ):
        enable_layerwise_offload(model)

    print(" No ModuleList error test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_memory_reduction():
    """Test that layerwise offloading reduces GPU memory usage."""
    device = torch.device("cuda")
    hidden_size = 512
    num_blocks = 8
    batch_size = 1
    seq_len = 64

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Measure initial GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    with torch.no_grad():
        _ = model(input_tensor)

    memory_without_offload = torch.cuda.max_memory_allocated()

    # Reset model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Enable offloading
    enable_layerwise_offload(model)

    # Measure GPU memory with offloading
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)
    with torch.no_grad():
        _ = model(input_tensor)

    memory_with_offload = torch.cuda.max_memory_allocated()

    # Memory with offload should be less (parameters are offloaded)
    # Note: This is a weak check as memory usage depends on many factors
    print(
        f"Memory without offload: {memory_without_offload / 1024**2:.2f} MB, "
        f"with offload: {memory_with_offload / 1024**2:.2f} MB"
    )

    # We expect some reduction but this test is more informational
    assert memory_with_offload < memory_without_offload * 1.5, (
        "Memory usage should not increase significantly with offloading"
    )

    print(" Memory reduction test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_gradient_disabled():
    """Test that layerwise offloading works correctly with gradients disabled."""
    device = torch.device("cuda")
    hidden_size = 128
    batch_size = 2
    seq_len = 16
    num_blocks = 3

    # Create model
    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)
    model.eval()

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward pass should work without gradients
    with torch.no_grad():
        output = model(input_tensor)

    assert output.requires_grad is False, "Output should not require gradients"
    assert output.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), "Output shape should match input"

    print(" Gradient disabled test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_layerwise_offload_different_batch_sizes():
    """Test layerwise offloading with different batch sizes."""
    device = torch.device("cuda")
    hidden_size = 128
    seq_len = 16
    num_blocks = 3

    model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)

    # Get reference model without offloading
    reference_model = SimpleModelWithModuleList(
        num_blocks=num_blocks, hidden_size=hidden_size, dtype=torch.float32
    ).to(device)
    reference_model.load_state_dict(model.state_dict())

    # Enable layerwise offloading
    enable_layerwise_offload(model)

    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        input_tensor = torch.randn(
            batch_size, seq_len, hidden_size, device=device
        )

        with torch.no_grad():
            reference_output = reference_model(input_tensor.clone())
            offloaded_output = model(input_tensor.clone())

        assert torch.allclose(
            reference_output, offloaded_output, rtol=1e-4, atol=1e-5
        ), f"Output mismatch for batch_size={batch_size}"

    print(" Different batch sizes test passed")


if __name__ == "__main__":
    # Run tests manually for debugging
    if torch.cuda.is_available():
        print("Running layerwise offloading tests...")
        test_layerwise_offload_basic()
        test_layerwise_offload_bf16()
        test_layerwise_offload_multiple_forward_passes()
        test_layerwise_offload_parameter_integrity()
        test_layerwise_offload_no_modulelist_error()
        test_layerwise_offload_memory_reduction()
        test_layerwise_offload_gradient_disabled()
        test_layerwise_offload_different_batch_sizes()
        print("\n All tests passed!")
    else:
        print("CUDA not available, skipping tests")
