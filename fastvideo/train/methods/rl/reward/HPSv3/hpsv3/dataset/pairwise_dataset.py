import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import os
from tqdm import tqdm

class PairwiseOriginalDataset(Dataset):
    def __init__(
        self,
        json_list,
        soft_label=False,
        confidence_threshold=None,
    ):
        self.samples = []
        for json_file in json_list:
            with open(json_file, "r") as f:
                data = json.load(f)
            self.samples.extend(data)

        self.soft_label = soft_label
        self.confidence_threshold = confidence_threshold

        if confidence_threshold is not None:
            new_samples = []
            for sample in tqdm(
                self.samples, desc="Filtering samples according to confidence threshold"
            ):
                if sample.get("confidence", float("inf")) >= confidence_threshold:
                    new_samples.append(sample)
            self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            index = idx
            try:
                return self.get_single_item(index)
            except Exception as e:
                print(f"Error processing sample at index {idx}: {e}")
                import traceback
                traceback.print_exc()
                index = random.randint(0, len(self.samples) - 1)
                if index == idx:
                    continue
                idx = index

    def get_single_item(self, idx):
        sample = self.samples[idx]
        # Load image paths
        image_1 = sample["path1"]
        image_2 = sample["path2"]
        assert os.path.exists(image_1) and os.path.exists(image_2), f'{image_1} or {image_2}'
        text_1 = sample["prompt"]
        text_2 = sample["prompt"]

        # Process Label
        if self.soft_label:
            choice_dist = sorted(sample["choice_dist"], reverse=True)
            assert (
                torch.sum(torch.tensor(choice_dist)) > 0
            ), "Choice distribution cannot be zero."
            label = torch.tensor(choice_dist[0]) / torch.sum(torch.tensor(choice_dist))
        else:
            label = torch.tensor(1).float()
        # breakpoint()
        return {
            "image_1": image_1,
            "image_2": image_2,
            "text_1": text_1,
            "text_2": text_2,
            "label": label,
            "confidence": sample.get("confidence", 1.0),
            "choice_dist": torch.tensor(sample.get("choice_dist", [1.0, 0.0])),
        }
