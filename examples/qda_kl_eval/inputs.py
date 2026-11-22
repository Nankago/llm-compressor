"""Create the fixed, held-out evaluation rows from the pinned WikiText-2 test shard."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

TEST_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("parquet", type=Path)
    p.add_argument("tokenizer", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    if hashlib.sha256(a.parquet.read_bytes()).hexdigest() != TEST_SHA:
        raise ValueError("Unexpected WikiText test shard")
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    texts = pq.read_table(a.parquet, columns=["text"]).column("text").to_pylist()
    indices = list(range(len(texts)))
    random.Random(42).shuffle(indices)
    rows = []
    for i in indices:
        ids = tok.encode(texts[i].strip(), add_special_tokens=False)
        if len(ids) >= 64:
            rows.append({"id": f"test:{i}", "input_ids": ids[:128], "score_start": 1})
        if len(rows) == 32:
            break
    a.out.write_text("".join(json.dumps(row) + "\n" for row in rows))
