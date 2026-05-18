"""Data loading for MDLM Multi-Head Distillation.

Uses the pre-tokenized OpenWebText cache from the MDLM repo.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import datasets

from config import Config

# Pre-tokenized dataset path (from MDLM repo)
OWT_TRAIN_CACHE = '/data1/wulingdan/data/diffusion/mdlm/cache/openwebtext-train_train_bs1024_wrapped.dat'
OWT_VALID_CACHE = '/data1/wulingdan/data/diffusion/mdlm/cache/openwebtext-valid_validation_bs1024_wrapped.dat'


class ArrowDataset(Dataset):
    """Thin wrapper around HuggingFace Arrow dataset."""

    def __init__(self, hf_dataset, seq_len: int):
        self.ds = hf_dataset
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ids = self.ds[idx]['input_ids']
        # Truncate to seq_len (dataset is L=1024, should match)
        return {'input_ids': torch.tensor(ids[:self.seq_len], dtype=torch.long)}


def get_dataloader(
    config: Config,
    tokenizer=None,
    split: str = "train",
) -> DataLoader:
    """Load pre-tokenized OpenWebText from MDLM cache."""
    if split == "train":
        path = OWT_TRAIN_CACHE
    else:
        path = OWT_VALID_CACHE

    print(f"Loading cached dataset from {path}")
    hf_ds = datasets.load_from_disk(path)
    dataset = ArrowDataset(hf_ds, config.seq_len)
    print(f"Dataset: {len(dataset)} sequences of length {config.seq_len}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader
