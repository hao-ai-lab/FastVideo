import pdb
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import torch
from .utils import process_vision_info
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

INSTRUCTION_debug = """
{text_prompt}
"""

prompt_with_special_token = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

prompt_without_special_token = """
Please provide the overall ratings of this image: 
"""


class QWen2VLDataCollator:

    def __init__(
        self,
        processor,
        with_instruction=True,
        max_pixels=256 * 28 * 28,  # Default max pixels
        min_pixels=256 * 28 * 28,  # Default min pixels
        use_special_tokens=True,
    ):
        self.processor = processor
        self.with_instruction = with_instruction
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.use_special_tokens = use_special_tokens

    def _clean_message(
        self,
        texts,
        images,
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
        with_instruction=True,
        use_special_tokens=True,
    ):
        """
        remove unnecessary keys from message(very very necessary)
        """
        message_list = []
        for text, image in zip(texts, images):
            out_message = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {
                        "type":
                        "text",
                        "text": (INSTRUCTION.format(text_prompt=text) +
                                 prompt_with_special_token if use_special_tokens else prompt_without_special_token),
                    },
                ],
            }]

            message_list.append(out_message)

        return message_list

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(sequences, padding, "constant",
                                                   self.processor.tokenizer.pad_token_id)
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, "constant", 0)

        return sequences_padded, attention_mask_padded

    def __call__(self, inputs, with_instruction=True):
        """
        Preprocess inputs to token sequences and return a batch
        """
        images_1, images_2, texts_1, texts_2 = [], [], [], []

        for idx, batch in enumerate(inputs):
            texts_1.append(batch["text_1"])
            texts_2.append(batch["text_2"])
            images_1.append(batch["image_1"])
            images_2.append(batch["image_2"])

        messages_batch_1 = self._clean_message(
            texts_1,
            images_1,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )
        messages_batch_2 = self._clean_message(
            texts_2,
            images_2,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            with_instruction=self.with_instruction,
            use_special_tokens=self.use_special_tokens,
        )
        # import pdb; pdb.set_trace()
        image_inputs_1, _ = process_vision_info(messages_batch_1)
        image_inputs_2, _ = process_vision_info(messages_batch_2)
        image_inputs_1 = [np.array(image_inputs_1[i]) / 255.0 for i in range(len(image_inputs_1))]
        image_inputs_2 = [np.array(image_inputs_2[i]) / 255.0 for i in range(len(image_inputs_2))]
        do_rescale = False

        batch_1 = self.processor(
            text=self.processor.apply_chat_template(messages_batch_1, tokenize=False, add_generation_prompt=True),
            images=image_inputs_1,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": do_rescale},
        )
        batch_2 = self.processor(
            text=self.processor.apply_chat_template(messages_batch_2, tokenize=False, add_generation_prompt=True),
            images=image_inputs_2,
            videos=None,
            padding=True,
            return_tensors="pt",
            images_kwargs={"do_rescale": do_rescale},
        )

        # pdb.set_trace()
        max_len = max(batch_1["input_ids"].shape[1], batch_2["input_ids"].shape[1])
        batch_1["input_ids"], batch_1["attention_mask"] = self._pad_sequence(batch_1["input_ids"],
                                                                             batch_1["attention_mask"], max_len,
                                                                             "right")
        batch_2["input_ids"], batch_2["attention_mask"] = self._pad_sequence(batch_2["input_ids"],
                                                                             batch_2["attention_mask"], max_len,
                                                                             "right")

        batch = {
            "batch_1": batch_1,
            "batch_2": batch_2,
            "choice_dist": torch.stack([batch["choice_dist"] for batch in inputs]),
            # Store original text prompts for visualization
            "text_1": texts_1,
            "text_2": texts_2,
            "image_1": image_inputs_1,
            "image_2": image_inputs_2,
        }

        return batch
