import os
from collections.abc import Mapping
from pathlib import Path

import huggingface_hub
import numpy as np
import torch
from hpsv3.dataset.data_collator_qwen import (
    INSTRUCTION,
    prompt_with_special_token,
    prompt_without_special_token,
)
from hpsv3.dataset.utils import process_vision_info
from hpsv3.train import create_model_and_processor
from hpsv3.utils.parser import (
    DataConfig,
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    parse_args_with_yaml,
)
from PIL import Image

_MODEL_CONFIG_PATH = Path(__file__).parent / "config/"


class HPSv3RewardInferencer:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
        differentiable=False,
    ):
        if config_path is None:
            config_path = os.path.join(_MODEL_CONFIG_PATH, "HPSv3_7B.yaml")

        if checkpoint_path is None:
            checkpoint_path = os.path.join(huggingface_hub.hf_hub_download("xilanhua12138/HPSv3"), "model.pth")

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
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
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
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
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

    def reward(self, image_paths, prompts):
        batch = self.prepare_batch(image_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]

        return rewards


if __name__ == "__main__":
    config_path = "/preflab/shuiyunhao/tasks/HPSv3_official/hpsv3/config/HPSv3_7B.yaml"
    checkpoint_path = "/preflab/shuiyunhao/tasks/HPSv3_official/checkpoints/HPSv3_7B/model.pth"
    device = "cuda"
    dtype = torch.bfloat16
    inferencer = HPSv3RewardInferencer(config_path, checkpoint_path, differentiable=True, device=device)

    images = [
        torch.from_numpy(np.array(Image.open("assets/example1.png"))),
        torch.from_numpy(np.array(Image.open("assets/example2.png"))),
    ]
    prompts = [
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
        "cute chibi anime cartoon fox, smiling wagging tail with a small cartoon heart above sticker",
    ]
    rewards = inferencer.reward(images, prompts)
    print(rewards[0][0].item())  # miu and sigma. we select miu as the final output
    print(rewards[1][0].item())

    loss = rewards[0][0]
    print(f"Loss value: {loss.item()}")
    print(f"Loss requires_grad: {loss.requires_grad}")

    if loss.requires_grad:
        loss.backward(retain_graph=True)

        has_grad = False
        for name, param in inferencer.model.named_parameters():
            if param.grad is not None:
                has_grad = True
                print(f"参数 {name} 有梯度: {param.grad.norm().item():.6f}")
                break

        if has_grad:
            print("has grad")
        else:
            print("NO GRAD!!")
    else:
        print("Final loss does not require gradient computation.")

    # Compare
    img_pil = Image.open("assets/example1.png").convert("RGB")
    non_diff_result = inferencer.processor.preprocess(images=[img_pil], return_tensors="pt")

    img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
    diff_result = inferencer.processor.preprocess_tensor(img_tensor)

    if non_diff_result["pixel_values"].shape == diff_result["pixel_values"].shape:
        diff = torch.abs(non_diff_result["pixel_values"] - diff_result["pixel_values"]).mean()
        if diff.item() < 0.01:
            print("Right")
        else:
            print("Different outputs")
    else:
        print("Shape mismatch between non-differentiable and differentiable outputs.")
