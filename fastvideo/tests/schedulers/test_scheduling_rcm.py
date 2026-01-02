# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RCMScheduler."""

import math

import pytest
import torch

from fastvideo.models.schedulers.scheduling_rcm import RCMScheduler, RCMSchedulerOutput


class TestRCMSchedulerTimesteps:
    """Tests for timestep schedule generation."""

    def test_set_timesteps_1_step(self):
        """Test 1-step schedule."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=1)
        
        # Should have 2 timesteps: [t_initial, 0]
        assert len(scheduler.timesteps) == 2
        assert scheduler.timesteps[-1] == 0.0

    def test_set_timesteps_4_steps(self):
        """Test 4-step schedule (typical use case)."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        # Should have 5 timesteps: [t_initial, mid1, mid2, mid3, 0]
        assert len(scheduler.timesteps) == 5
        assert scheduler.timesteps[-1] == 0.0
        
        # Timesteps should be monotonically decreasing
        for i in range(len(scheduler.timesteps) - 1):
            assert scheduler.timesteps[i] > scheduler.timesteps[i + 1]

    def test_trigflow_to_rectifiedflow_conversion(self):
        """Verify TrigFlow → RectifiedFlow conversion is applied."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        # First timestep should be sin(atan(80)) / (cos(atan(80)) + sin(atan(80)))
        expected_t0 = math.sin(math.atan(80.0)) / (math.cos(math.atan(80.0)) + math.sin(math.atan(80.0)))
        assert abs(scheduler.timesteps[0].item() - expected_t0) < 1e-6

    def test_device_placement(self):
        """Test timesteps are placed on correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        scheduler = RCMScheduler()
        scheduler.set_timesteps(num_inference_steps=4, device="cuda")
        
        assert scheduler.timesteps.device.type == "cuda"


class TestRCMSchedulerStep:
    """Tests for the step function."""

    def test_step_output_shape(self):
        """Test step function returns correct shape."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        batch_size, channels, frames, height, width = 1, 16, 21, 48, 80
        sample = torch.randn(batch_size, channels, frames, height, width)
        model_output = torch.randn_like(sample)
        
        output = scheduler.step(model_output, 0, sample)
        
        assert isinstance(output, RCMSchedulerOutput)
        assert output.prev_sample.shape == sample.shape

    def test_step_with_generator(self):
        """Test reproducibility with generator."""
        scheduler1 = RCMScheduler(sigma_max=80.0)
        scheduler2 = RCMScheduler(sigma_max=80.0)
        
        scheduler1.set_timesteps(num_inference_steps=4)
        scheduler2.set_timesteps(num_inference_steps=4)
        
        sample = torch.randn(1, 16, 21, 48, 80)
        model_output = torch.randn_like(sample)
        
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        
        out1 = scheduler1.step(model_output, 0, sample, generator=gen1)
        out2 = scheduler2.step(model_output, 0, sample, generator=gen2)
        
        assert torch.allclose(out1.prev_sample, out2.prev_sample)

    def test_step_index_increments(self):
        """Test step index increments after each step."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        sample = torch.randn(1, 16, 21, 48, 80)
        model_output = torch.randn_like(sample)
        
        assert scheduler._step_index is None
        
        scheduler.step(model_output, 0, sample)
        assert scheduler._step_index == 1
        
        scheduler.step(model_output, 1, sample)
        assert scheduler._step_index == 2

    def test_full_sampling_loop(self):
        """Test complete 4-step sampling loop."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4, device="cpu")
        
        # Initialize with scaled noise
        noise = torch.randn(1, 16, 21, 48, 80)
        sample = scheduler.scale_noise(None, None, noise)
        
        generator = torch.Generator().manual_seed(0)
        
        # Run all 4 steps
        for i in range(4):
            model_output = torch.randn_like(sample)  # Simulated model output
            output = scheduler.step(model_output, i, sample, generator=generator)
            sample = output.prev_sample
        
        # Should complete without error
        assert sample.shape == noise.shape
        assert scheduler._step_index == 4


class TestRCMSchedulerScaleNoise:
    """Tests for noise scaling."""

    def test_scale_noise(self):
        """Test initial noise scaling."""
        scheduler = RCMScheduler(sigma_max=80.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        noise = torch.randn(1, 16, 21, 48, 80)
        scaled = scheduler.scale_noise(None, None, noise)
        
        # Should be scaled by first timestep
        expected = noise.to(torch.float64) * scheduler.timesteps[0]
        assert torch.allclose(scaled, expected)

    def test_scale_noise_requires_noise(self):
        """Test that scale_noise raises error without noise."""
        scheduler = RCMScheduler()
        scheduler.set_timesteps(num_inference_steps=4)
        
        with pytest.raises(ValueError, match="noise must be provided"):
            scheduler.scale_noise(None, None, None)


class TestRCMSchedulerConfig:
    """Tests for configuration."""

    def test_custom_sigma_max(self):
        """Test custom sigma_max value."""
        scheduler = RCMScheduler(sigma_max=160.0)
        scheduler.set_timesteps(num_inference_steps=4)
        
        # Higher sigma_max should result in higher initial timestep
        scheduler2 = RCMScheduler(sigma_max=80.0)
        scheduler2.set_timesteps(num_inference_steps=4)
        
        assert scheduler.timesteps[0] > scheduler2.timesteps[0]

    def test_custom_mid_timesteps(self):
        """Test custom intermediate timesteps."""
        custom_mid = [1.3, 1.0, 0.6]
        scheduler = RCMScheduler(mid_timesteps=custom_mid)
        scheduler.set_timesteps(num_inference_steps=4)
        
        # Should have 5 timesteps with custom mid values
        assert len(scheduler.timesteps) == 5
