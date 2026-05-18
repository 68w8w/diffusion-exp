"""Data loading for MDLM Multi-Head Distillation.

Uses OpenWebText dataset with GPT-2 BPE tokenizer, matching MDLM training.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import datasets
import transformers

from config import Config


class TokenizedDataset(Dataset):
    """Pre-tokenized and chunked dataset."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        """
        Args:
            token_ids: [N] flat tensor of all token IDs
            seq_len: sequence length per sample
        """
        self.seq_len = seq_len
        # Truncate to exact multiple of seq_len
        n_samples = len(token_ids) // seq_len
        self.data = token_ids[:n_samples * seq_len].reshape(n_samples, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': self.data[idx]}


def get_dataloader(
    config: Config,
    tokenizer: transformers.PreTrainedTokenizer,
    split: str = "train",
) -> DataLoader:
    """
    Load OpenWebText dataset, tokenize, chunk into seq_len blocks.

    Args:
        config: configuration
        tokenizer: GPT-2 tokenizer
        split: "train" or "validation"

    Returns:
        DataLoader yielding {'input_ids': [B, L]} batches
    """
    cache_dir = os.path.join(config.output_dir, "data_cache")

    # Try to load cached tokenized data
    cache_file = os.path.join(cache_dir, f"owt_{split}_tokens.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached tokenized data from {cache_file}")
        token_ids = torch.load(cache_file)
    else:
        print(f"Tokenizing OpenWebText ({split})...")
        os.makedirs(cache_dir, exist_ok=True)

        # Load dataset
        if split == "train":
            ds = datasets.load_dataset("openwebtext", split="train",
                                       trust_remote_code=True)
        else:
            # OpenWebText doesn't have a validation split;
            # use a small portion of train as validation
            ds = datasets.load_dataset("openwebtext", split="train[:1%]",
                                       trust_remote_code=True)

        # Tokenize all texts
        all_tokens = []
        for example in ds:
            tokens = tokenizer.encode(example['text'])
            all_tokens.extend(tokens)

            # For training, limit to reasonable size
            if split == "train" and len(all_tokens) > 100_000_000:
                break

        token_ids = torch.tensor(all_tokens, dtype=torch.long)
        torch.save(token_ids, cache_file)
        print(f"Saved tokenized data to {cache_file} ({len(token_ids):,} tokens)")

    # Create dataset
    dataset = TokenizedDataset(token_ids, config.seq_len)
    print(f"Dataset: {len(dataset)} sequences of length {config.seq_len}")

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    return loader
