import torch

from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.train.models.wan.wan import WanModel


class _CPUWanModel(WanModel):

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class _AutocastProbe(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.autocast_enabled: bool | None = None
        self.autocast_dtype: torch.dtype | None = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep: torch.Tensor,
        return_dict: bool,
    ) -> torch.Tensor:
        del encoder_hidden_states, encoder_attention_mask, timestep, return_dict
        self.autocast_enabled = torch.is_autocast_enabled("cpu")
        self.autocast_dtype = torch.get_autocast_dtype("cpu")
        self.hidden_states_dtype = hidden_states.dtype
        return hidden_states


def test_wan_predict_noise_uses_training_dtype_autocast_for_fp32_inputs():
    model = object.__new__(_CPUWanModel)
    model.transformer = _AutocastProbe()

    batch = TrainingBatch()
    batch.timesteps = torch.tensor([1], dtype=torch.long)
    batch.conditional_dict = {
        "encoder_hidden_states": torch.randn(1, 4, 8, dtype=torch.float32),
        "encoder_attention_mask": torch.ones(1, 4, dtype=torch.float32),
    }

    noisy_latents = torch.randn(1, 1, 2, 4, 4, dtype=torch.float32)
    timestep = torch.tensor([1], dtype=torch.long)

    pred_noise = model.predict_noise(
        noisy_latents,
        timestep,
        batch,
        conditional=True,
    )

    assert pred_noise.shape == noisy_latents.shape
    assert pred_noise.dtype is torch.bfloat16
    assert model.transformer.hidden_states_dtype is torch.bfloat16
    assert model.transformer.autocast_enabled is True
    assert model.transformer.autocast_dtype is torch.bfloat16
