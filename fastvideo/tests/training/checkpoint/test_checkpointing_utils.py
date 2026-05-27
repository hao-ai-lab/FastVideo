import torch
from torch import nn

from fastvideo.training.checkpointing_utils import ModelWrapper


class DummyWrappedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._lora_a = nn.Parameter(torch.tensor([1.0, 2.0]))
        self._lora_b = nn.Parameter(torch.tensor([3.0, 4.0]))
        self._frozen = nn.Parameter(torch.tensor([5.0, 6.0]), requires_grad=False)

    def named_parameters(self, *args, **kwargs):
        # Simulate wrapped names returned by named_parameters()
        yield "layer._checkpoint_wrapped_module.lora_A", self._lora_a
        yield "layer._checkpoint_wrapped_module.lora_B", self._lora_b
        yield "layer._checkpoint_wrapped_module.frozen_weight", self._frozen


def test_model_wrapper_filters_wrapped_trainable_params(monkeypatch):
    """Regression test for wrapped parameter name mismatch during checkpoint filtering."""
    model = DummyWrappedModule()
    wrapper = ModelWrapper(model)

    mocked_state_dict = {
        "layer.lora_A": torch.tensor([10.0, 20.0]),
        "layer.lora_B": torch.tensor([30.0, 40.0]),
        "layer.frozen_weight": torch.tensor([50.0, 60.0]),
    }

    def mock_get_model_state_dict(_model):
        return mocked_state_dict

    monkeypatch.setattr(
        "fastvideo.training.checkpointing_utils.get_model_state_dict",
        mock_get_model_state_dict,
    )

    filtered_state_dict = wrapper.state_dict()

    assert set(filtered_state_dict.keys()) == {"layer.lora_A", "layer.lora_B"}
    assert torch.equal(filtered_state_dict["layer.lora_A"], mocked_state_dict["layer.lora_A"])
    assert torch.equal(filtered_state_dict["layer.lora_B"], mocked_state_dict["layer.lora_B"])
    assert "layer.frozen_weight" not in filtered_state_dict