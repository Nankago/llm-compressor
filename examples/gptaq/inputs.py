"""Fixed article windows from the public WikiText-2 raw train and test splits."""

import hashlib
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

WEIGHTS = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
TRAIN = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
TEST = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows(path, expected, tokenizer, count, length, seed):
    if digest(path) != expected:
        raise ValueError("WikiText parquet digest differs")
    rows = pq.read_table(path, columns=["text"])["text"].to_pylist()
    starts = [
        i
        for i, row in enumerate(rows)
        if re.fullmatch(r"= (?![=]).*? (?<![=] )=", row.strip())
    ]
    articles = list(zip(starts, starts[1:] + [len(rows)]))
    rng = random.Random(seed)
    rng.shuffle(articles)
    data, positions = [], []
    for first, last in articles:
        ids = tokenizer.encode("".join(rows[first:last]), add_special_tokens=False)
        if len(ids) < length:
            continue
        offset = rng.randrange(len(ids) - length + 1)
        data.append(ids[offset : offset + length])
        positions.append([first, offset])
        if len(data) == count:
            break
    arr = np.asarray(data, dtype="<i8")
    if arr.shape != (count, length):
        raise ValueError("Insufficient distinct articles")
    return torch.from_numpy(arr), dict(
        seed=seed,
        shape=list(arr.shape),
        sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
        article_and_offset=positions,
    )
