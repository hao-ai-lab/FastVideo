# SPDX-License-Identifier: Apache-2.0
"""Text prompt datasets and samplers for RL training."""

from __future__ import annotations

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class TextPromptDataset(Dataset):
    """Load plain text prompts from train.txt / test.txt."""

    def __init__(self, dataset: str, split: str = "train"):
        self.file_path = os.path.join(
            dataset, f"{split}.txt"
        )
        with open(self.file_path) as f:
            self.prompts = [
                line.strip() for line in f.readlines()
            ]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(
        self, idx: int | tuple[int, int]
    ) -> dict:
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {
            "epoch": epoch_tag,
            "prompt": self.prompts[idx],
            "metadata": {},
        }

    @staticmethod
    def collate_fn(
        examples: list[dict],
    ) -> tuple[int | None, list[str], list[dict]]:
        epoch_tags = [
            example.get("epoch") for example in examples
        ]
        epoch_tag = (
            epoch_tags[0]
            if all(tag == epoch_tags[0] for tag in epoch_tags)
            else None
        )
        prompts = [
            example["prompt"] for example in examples
        ]
        metadatas = [
            example["metadata"] for example in examples
        ]
        return epoch_tag, prompts, metadatas


class JsonPromptDataset(Dataset):
    """Load prompts from JSONL files."""

    def __init__(self, dataset: str, split: str = "train"):
        self.file_path = os.path.join(
            dataset, f"{split}.json"
        )
        self._prompts: list[str] = []
        self._metadatas: list[dict] = []
        self._load_all_prompts()

    def _load_all_prompts(self):
        with open(self.file_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", "")
                    if prompt:
                        self._prompts.append(prompt)
                        metadata = {
                            k: v
                            for k, v in item.items()
                            if k != "prompt"
                        }
                        self._metadatas.append(metadata)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Skipping invalid JSON line: %s",
                        e,
                    )

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(
        self, idx: int | tuple[int, int]
    ) -> dict:
        epoch_tag = None
        if isinstance(idx, tuple):
            epoch_tag, idx = idx
        return {
            "epoch": epoch_tag,
            "prompt": self._prompts[idx],
            "metadata": (
                self._metadatas[idx]
                if self._metadatas
                else {}
            ),
        }

    @staticmethod
    def collate_fn(
        examples: list[dict],
    ) -> tuple[int | None, list[str], list[dict]]:
        epoch_tags = [
            example.get("epoch") for example in examples
        ]
        epoch_tag = (
            epoch_tags[0]
            if all(tag == epoch_tags[0] for tag in epoch_tags)
            else None
        )
        prompts = [
            example["prompt"] for example in examples
        ]
        metadatas = [
            example["metadata"] for example in examples
        ]
        return epoch_tag, prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    """Repeat each prompt k times per global batch and
    shard across ranks."""
+
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = num_replicas * batch_size
        assert self.total_samples % self.k == 0, (
            f"k cannot divide n*b: k={k} "
            f"num_replicas={num_replicas} "
            f"batch_size={batch_size}"
        )
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(
                len(self.dataset), generator=g
            )[: self.m].tolist()
            repeated = [
                idx
                for idx in indices
                for _ in range(self.k)
            ]
            shuffled_idx = torch.randperm(
                len(repeated), generator=g
            ).tolist()
            shuffled = [
                repeated[i] for i in shuffled_idx
            ]
            per_card = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card.append(
                    [
                        (self.epoch, idx)
                        for idx in shuffled[start:end]
                    ]
                )
            yield per_card[self.rank]

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def build_prompt_dataloaders(
    prompt_dataset_path: str,
    prompt_fn: str,
    sample_batch_size: int,
    eval_batch_size: int,
    num_video_per_prompt: int,
    num_processes: int,
    process_index: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DistributedKRepeatSampler]:
    """Build train/eval prompt dataloaders.

    Returns:
        (train_dataloader, test_dataloader, train_sampler)
    """
    if prompt_fn == "general_ocr":
        train_ds = TextPromptDataset(
            prompt_dataset_path, "train"
        )
        test_ds = TextPromptDataset(
            prompt_dataset_path, "test"
        )
        collate = TextPromptDataset.collate_fn
    elif prompt_fn == "filtered_prompts":
        train_ds = JsonPromptDataset(
            prompt_dataset_path, "train"
        )
        test_ds = JsonPromptDataset(
            prompt_dataset_path, "test"
        )
        collate = JsonPromptDataset.collate_fn
    else:
        msg = (
            f"Unsupported prompt_fn: {prompt_fn}. "
            "Use 'general_ocr' or 'filtered_prompts'."
        )
        raise NotImplementedError(msg)

    train_sampler = DistributedKRepeatSampler(
        dataset=train_ds,
        batch_size=sample_batch_size,
        k=num_video_per_prompt,
        num_replicas=num_processes,
        rank=process_index,
        seed=seed,
    )

    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=1,
        collate_fn=collate,
        prefetch_factor=1,
        persistent_workers=False,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=8,
    )
    return train_dl, test_dl, train_sampler
