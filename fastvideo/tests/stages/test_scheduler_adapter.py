# SPDX-License-Identifier: Apache-2.0
import torch

from fastvideo.models.schedulers.adapter import DefaultSchedulerAdapter


class DummyScheduler:
    def __init__(self) -> None:
        self.calls: dict[str, tuple] = {}

    def scale_model_input(self, latents: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        self.calls["scale_model_input"] = (latents, t)
        return latents + 1

    def step(self, noise_pred: torch.Tensor, t: torch.Tensor,
             latents: torch.Tensor, **kwargs):
        self.calls["step"] = (noise_pred, t, latents, kwargs)
        return "step-result"

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        self.calls["add_noise"] = (latents, noise, t)
        return latents + noise

    def set_timesteps(self, num_steps: int, device=None, **kwargs):
        self.calls["set_timesteps"] = (num_steps, device, kwargs)
        return ["ok"]


def test_default_scheduler_adapter_delegates():
    scheduler = DummyScheduler()
    adapter = DefaultSchedulerAdapter(scheduler)
    latents = torch.zeros(2, 2)
    noise = torch.ones(2, 2)
    t = torch.tensor(1)

    scaled = adapter.scale_model_input(latents, t)
    assert torch.allclose(scaled, latents + 1)
    assert scheduler.calls["scale_model_input"] == (latents, t)

    step_out = adapter.step(torch.ones(1), t, latents, foo="bar")
    assert step_out == "step-result"
    assert scheduler.calls["step"][3]["foo"] == "bar"

    noised = adapter.add_noise(latents, noise, t)
    assert torch.allclose(noised, latents + noise)
    assert scheduler.calls["add_noise"] == (latents, noise, t)

    set_out = adapter.set_timesteps(4, device=torch.device("cpu"), baz=3)
    assert set_out == ["ok"]
    assert scheduler.calls["set_timesteps"][0] == 4
