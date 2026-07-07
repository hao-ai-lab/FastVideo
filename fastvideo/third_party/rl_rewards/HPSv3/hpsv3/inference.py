import os
from collections.abc import Mapping
from pathlib import Path

import huggingface_hub
import torch

from .dataset.utils import process_vision_info
from .runtime import create_model_and_processor
from .utils.parser import (
    DataConfig,
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    parse_args_with_yaml,
)

_MODEL_CONFIG_PATH = Path(__file__).parent / "config/"

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and
Text Alignment and give a overall score to estimate the human preference.
Please provide a rating from 0 to 10, with 0 being the worst and 10 being the
best.

Textual prompt - {text_prompt}


"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image:
"""


class HPSv3RewardInferencer:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
        differentiable=False,
    ):
        if differentiable:
            raise ValueError("The vendored HPSv3 runtime is inference-only and does not support differentiable mode.")
        if config_path is None:
            config_path = os.path.join(_MODEL_CONFIG_PATH, "HPSv3_7B.yaml")

        if checkpoint_path is None:
            checkpoint_path = huggingface_hub.hf_hub_download("MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model")

        (
            (
                data_config,
                training_args,
                model_config,
                peft_lora_config,
            ),
            config_path,
        ) = parse_args_with_yaml(
            (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig),
            config_path,
            is_train=False,
        )
        training_args.output_dir = os.path.join(training_args.output_dir, config_path.split("/")[-1].split(".")[0])
        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch

            state_dict = safetensors.torch.load_file(checkpoint_path, device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)
        self.data_config = data_config

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, "constant", 0)

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch(self, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": max_pixels,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": (
                                INSTRUCTION.format(text_prompt=text) + prompt_with_special_token
                                if self.use_special_tokens
                                else prompt_without_special_token
                            ),
                        },
                    ],
                }
            ]

            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)

        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    @torch.inference_mode()
    def reward(self, prompts, image_paths):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]

        return rewards
