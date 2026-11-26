"""Select disjoint WikiText paragraph rows and reproducible next-token anchors."""

import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

TEST_SHA = "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91"


def main(model, parquet, out):
    assert hashlib.sha256(Path(parquet).read_bytes()).hexdigest() == TEST_SHA
    tokenizer = AutoTokenizer.from_pretrained(model)
    texts = pq.read_table(parquet, columns=["text"]).column(0).to_pylist()
    rng = random.Random(727)
    order = list(range(len(texts)))
    rng.shuffle(order)
    rows, seen = [], set()
    for i in order:
        text = texts[i].strip()
        if text in seen:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[:129]
        if len(ids) < 65:
            continue
        n = rng.randrange(32, len(ids))
        rows.append({"row": i, "ids": ids[: n + 1]})
        seen.add(text)
        if len(rows) == 128:
            break
    Path(out).write_text(json.dumps(rows) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
