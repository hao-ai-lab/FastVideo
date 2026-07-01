from dataclasses import dataclass

import torch
from prompt_template import build_prompt

# from qwen_vl_utils import process_vision_info
from vision_process import process_vision_info


@dataclass
class DataConfig:
    meta_data: str = "/path/to/dataset/meta_data.csv"
    data_dir: str = "/path/to/dataset"
    meta_data_test: str = None
    max_frame_pixels: int = 240 * 320
    num_frames: float = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: str | list[str] = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True


def convert_GSB_csv_to_reward_data(
    example,
    data_dir,
    eval_dims=["VQ"],
    max_pixels=448 * 448,
    fps=2.0,
    num_frames=None,
    prompt_template_type="none",
    sample_type="uniform",
):
    """
    Convert Good/Same/Bad csv data to reward data.

    Args:
        example (dict): A dataframe containing the GSB csv data.
        data_dir (str): The directory path to the video files.
        eval_dim (str): The dimension to evaluate ("VQ"/"MQ"/"TA").
        max_pixels (int): The maximum number of pixels allowed for videos.
        num_frames (float): Number of frames.
        prompt_template_type (str): The type of prompt template to use ("none"/"simple"/"video_score").

    Returns:
        dict: A dictionary containing the reward data.
    """

    A_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{data_dir}/{example['path_A']}",
                    "max_pixels": max_pixels,
                    "fps": fps if num_frames is None else None,
                    "nframes": (min(num_frames, example["num_frames_A"]) if num_frames is not None else None),
                    "sample_type": sample_type,
                },
                {
                    "type": "text",
                    "text": build_prompt(example["prompt"], eval_dims, prompt_template_type),
                },
            ],
        }
    ]
    B_data = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{data_dir}/{example['path_B']}",
                    "max_pixels": max_pixels,
                    "fps": fps if num_frames is None else None,
                    "nframes": (min(num_frames, example["num_frames_B"]) if num_frames is not None else None),
                    "sample_type": sample_type,
                },
                {
                    "type": "text",
                    "text": build_prompt(example["prompt"], eval_dims, prompt_template_type),
                },
            ],
        }
    ]

    chosen_labels = []
    A_scores = []
    B_scores = []

    for eval_dim in eval_dims:
        ### chosen_label: 1 if A is chosen, -1 if B is chosen, 0 if tied.
        ### 22 if invalid. ooaaeeaa o.O
        try:
            if example[f"{eval_dim}"] is not None:
                if example[f"{eval_dim}"] == "A":
                    chosen_label = 1
                elif example[f"{eval_dim}"] == "B":
                    chosen_label = -1
                elif example[f"{eval_dim}"] == "same":
                    chosen_label = 0
                elif example[f"{eval_dim}"] == "invalid":
                    chosen_label = 22
                else:
                    chosen_label = 22
            else:
                chosen_label = 22
        except Exception:
            chosen_label = 22

        chosen_labels.append(chosen_label)
        if f"MOS_A_{eval_dim}" in example and f"MOS_B_{eval_dim}" in example:
            try:
                A_score = example[f"MOS_A_{eval_dim}"] if example[f"MOS_A_{eval_dim}"] is not None else 0.0
                B_score = example[f"MOS_B_{eval_dim}"] if example[f"MOS_B_{eval_dim}"] is not None else 0.0
            except Exception:
                A_score = 0.0
                B_score = 0.0
            A_scores.append(A_score)
            B_scores.append(B_score)
        else:
            A_scores.append(0.0)
            B_scores.append(0.0)

    chosen_labels = torch.tensor(chosen_labels, dtype=torch.long)
    A_scores = torch.tensor(A_scores, dtype=torch.float)
    B_scores = torch.tensor(B_scores, dtype=torch.float)
    metainfo_idx = None
    if "metainfo_idx" in example:
        metainfo_idx = example["metainfo_idx"]

    return {
        "A_data": A_data,
        "B_data": B_data,
        "A_scores": A_scores,
        "B_scores": B_scores,
        "chosen_label": chosen_labels,
        "metainfo_idx": metainfo_idx,
    }


class QWen2VLDataCollator:
    def __init__(self, processor, add_noise=False, p_shuffle_frames=0.0, p_color_jitter=0.0):
        self.processor = processor
        self.add_noise = add_noise
        self.set_noise_step = None

        self.p_shuffle_frames = p_shuffle_frames
        self.p_color_jitter = p_color_jitter

        self.noise_adder = None

    def _clean_message(self, message):
        """
        remove unnecessary keys from message(very very necessary)
        """
        out_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": message[0]["content"][0]["video"],
                        "max_pixels": message[0]["content"][0]["max_pixels"],
                        "fps": (message[0]["content"][0]["fps"] if "fps" in message[0]["content"][0] else None),
                        "nframes": (
                            message[0]["content"][0]["nframes"] if "nframes" in message[0]["content"][0] else None
                        ),
                        "sample_type": (
                            message[0]["content"][0]["sample_type"]
                            if "sample_type" in message[0]["content"][0]
                            else "uniform"
                        ),
                    },
                    {"type": "text", "text": message[0]["content"][1]["text"]},
                ],
            }
        ]

        if out_message[0]["content"][0]["fps"] is None:
            out_message[0]["content"][0].pop("fps")
        if out_message[0]["content"][0]["nframes"] is None:
            out_message[0]["content"][0].pop("nframes")

        return out_message

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

    def __call__(self, features, enable_noise=True):
        """
        Preprocess inputs to token sequences and return a batch
        """
        # try:
        features_A = []
        features_B = []
        # check if we have a margin. If we do, we need to batch it as well
        # has_margin = "margin" in features[0]
        has_idx = "metainfo_idx" in features[0] and features[0]["metainfo_idx"] is not None

        for idx, feature in enumerate(features):
            features_A.append(self._clean_message(feature["A_data"]))
            features_B.append(self._clean_message(feature["B_data"]))

        # import pdb; pdb.set_trace()
        image_inputs_A, video_inputs_A = process_vision_info(features_A)
        image_inputs_B, video_inputs_B = process_vision_info(features_B)

        video_inputs_A = [video_inputs_A[i].float() / 255.0 for i in range(len(video_inputs_A))]
        video_inputs_B = [video_inputs_B[i].float() / 255.0 for i in range(len(video_inputs_B))]
        do_rescale = False
        # print(f"{video_inputs_A[0].shape}, {video_inputs_B[0].shape}")

        # if not enable_noise:
        #     print("Not training, no noise added.")
        batch_A = self.processor(
            text=self.processor.apply_chat_template(features_A, tokenize=False, add_generation_prompt=True),
            images=image_inputs_A,
            videos=video_inputs_A,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": do_rescale},
        )
        batch_B = self.processor(
            text=self.processor.apply_chat_template(features_B, tokenize=False, add_generation_prompt=True),
            images=image_inputs_B,
            videos=video_inputs_B,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": do_rescale},
        )

        # pdb.set_trace()
        max_len = max(batch_A["input_ids"].shape[1], batch_B["input_ids"].shape[1])
        batch_A["input_ids"], batch_A["attention_mask"] = self._pad_sequence(
            batch_A["input_ids"], batch_A["attention_mask"], max_len, "right"
        )
        batch_B["input_ids"], batch_B["attention_mask"] = self._pad_sequence(
            batch_B["input_ids"], batch_B["attention_mask"], max_len, "right"
        )
        # print(f"Batch A: {batch_A['input_ids'].shape}, Batch B: {batch_B['input_ids'].shape}")

        chosen_label = torch.stack([torch.tensor(feature["chosen_label"]) for feature in features])

        A_scores = torch.stack([torch.tensor(feature["A_scores"]) for feature in features])
        B_scores = torch.stack([torch.tensor(feature["B_scores"]) for feature in features])

        batch = {
            "A": batch_A,
            "B": batch_B,
            "return_loss": True,
            "chosen_label": chosen_label,
            "A_scores": A_scores,
            "B_scores": B_scores,
        }

        if has_idx:
            metainfo_idx = torch.stack([torch.tensor(feature["metainfo_idx"]) for feature in features])
            batch["metainfo_idx"] = metainfo_idx

        # pdb.set_trace()
        return batch

        # except Exception as e:
        #     print(f"Error processing batch: {e} in reading.")
        #     # get next batch
        #     return None
