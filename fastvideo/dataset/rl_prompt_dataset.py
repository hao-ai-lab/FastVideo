# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json
import os


class TextPromptDataset(Dataset):
    """Dataset for loading text prompts from a simple text file (one prompt per line)."""
    
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    """Dataset for loading prompts with metadata from JSONL files (e.g., GenEval format)."""
    
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class KRepeatSampler(Sampler):
    """Sampler that repeats each sample k times, ensuring synchronized random selection. For single-node training, set num_replicas=1 and rank=0."""
    
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size        # Batch size per GPU/card
        self.k = k                          # Number of repetitions per sample
        self.num_replicas = num_replicas    # Total number of GPUs/cards
        self.rank = rank                    # Current GPU/card rank
        self.seed = seed                    # Random seed for synchronization
        
        # Calculate the number of unique samples needed for each iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # different number of samples
        self.step = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all cards are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.step)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate a total of n*b samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle the order to ensure even distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples among all cards
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return the sample indices for the current card
            yield per_card_samples[self.rank]
    
    def set_step(self, step):
        """Used to synchronize the random state for different epochs."""
        self.step = step


def build_rl_prompt_dataloader(
    dataset_path: str,
    dataset_type: str = "text",
    split: str = "train",
    train_batch_size: int = 8,
    test_batch_size: int = 8,
    k: int = 1,
    seed: int = 42,
    train_num_workers: int = 1,
    test_num_workers: int = 8,
    num_replicas: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Factory function to create train and test dataloaders for RL prompt datasets.
    
    Args:
        dataset_path: Path to dataset directory
        dataset_type: "text" for TextPromptDataset or "geneval" for GenevalPromptDataset
        split: Dataset split ("train" or "test")
        train_batch_size: Batch size per GPU for training
        test_batch_size: Batch size for testing
        k: Number of times to repeat each sample (num_image_per_prompt)
        seed: Random seed for sampler synchronization
        train_num_workers: Number of workers for training dataloader
        test_num_workers: Number of workers for test dataloader
        num_replicas: Number of replicas (default 1 for single-node)
        rank: Rank of current process (default 0 for single-node)
    
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets based on type
    if dataset_type == "text":
        train_dataset = TextPromptDataset(dataset_path, 'train')
        test_dataset = TextPromptDataset(dataset_path, 'test')
        collate_fn = TextPromptDataset.collate_fn
    elif dataset_type == "geneval":
        train_dataset = GenevalPromptDataset(dataset_path, 'train')
        test_dataset = GenevalPromptDataset(dataset_path, 'test')
        collate_fn = GenevalPromptDataset.collate_fn
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'text' or 'geneval'")
    
    # Create infinite-loop training sampler
    train_sampler = KRepeatSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        k=k,
        num_replicas=num_replicas,
        rank=rank,
        seed=seed
    )
    
    # Create training dataloader with batch_sampler (infinite loop)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=train_num_workers,
        collate_fn=collate_fn,
    )
    
    # Create standard test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=test_num_workers,
    )
    
    return train_dataloader, test_dataloader

