"""Data loading helpers used by the training and inference scripts."""

from __future__ import annotations

import functools
import re
from typing import Iterable, List, Tuple

import numpy as np
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset

from .labels import LABELS

NUM_LABELS = len(LABELS)


def clean_text(text: str) -> str:
    """Removes newlines and duplicated escape sequences."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\\n\s*", ". ", text)
    return text.strip()


def _parse_label_field(label_field: str) -> List[str]:
    return [token.strip() for token in label_field.split(",") if token.strip()]


def read_custom_data(filepath: str, is_one_hot: bool = True) -> Iterable[dict]:
    """Yields dataset records compatible with ``paddlenlp.datasets``."""

    with open(filepath, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text, label_field = line.rstrip("\n").split("\t")
            label_tokens = _parse_label_field(label_field)
            if is_one_hot:
                labels = [
                    float(1) if str(i) in label_tokens else float(0)
                    for i in range(NUM_LABELS)
                ]
            else:
                labels = [int(token) for token in label_tokens]
            yield {"text": clean_text(text), "labels": labels}


def _preprocess_function(example: dict, tokenizer, max_seq_length: int) -> dict:
    encoded = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    encoded["labels"] = np.array(example["labels"], dtype="float32")
    return encoded


def build_dataset(filepath: str, tokenizer, max_seq_length: int):
    dataset = load_dataset(read_custom_data, filepath=filepath, lazy=False)
    trans_func = functools.partial(
        _preprocess_function, tokenizer=tokenizer, max_seq_length=max_seq_length
    )
    return dataset.map(trans_func)


def build_data_loader(
    filepath: str,
    tokenizer,
    batch_size: int,
    max_seq_length: int,
    shuffle: bool,
) -> Tuple[object, DataLoader]:
    """Creates a preprocessed dataset and matching ``DataLoader``."""

    dataset = build_dataset(filepath, tokenizer, max_seq_length)
    collate_fn = DataCollatorWithPadding(tokenizer)
    batch_sampler = BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    return dataset, data_loader
