"""Prepare deterministic Granite calibration windows from WikiText-2 raw train."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DATASET = "Salesforce/wikitext"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SHARD = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
SHA256 = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
MODEL_REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
TOKEN_SHA256 = "9784cb0bc0b32cfa39a4714d5e5270f376993cc219cc37ebb258b7977f65cc6d"
SEED = 42


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="New directory outside git")
    args = p.parse_args()
    with args.parquet.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == SHA256
    rows = pq.read_table(args.parquet, columns=["text"])["text"].to_pylist()
    # WikiText represents top-level headings as '= Title =', subheads as '= = ...'.
    starts = [
        i
        for i, row in enumerate(rows)
        if re.fullmatch(r"= (?![=]).*? (?<![=] )=", row.strip())
    ]
    articles = list(zip(starts, starts[1:] + [len(rows)]))
    rng = random.Random(SEED)
    rng.shuffle(articles)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    windows, offsets = [], []
    for start, stop in articles:
        ids = tokenizer.encode("".join(rows[start:stop]), add_special_tokens=False)
        if len(ids) >= 1024:
            offset = rng.randrange(len(ids) - 1024 + 1)
            windows.append(ids[offset : offset + 1024])
            offsets.append([start, offset])
            if len(windows) == 64:
                break
    tokens = np.asarray(windows, dtype="<i8")
    assert tokens.shape == (64, 1024)
    meta = {
        "dataset": DATASET,
        "revision": REVISION,
        "shard": SHARD,
        "shard_sha256": SHA256,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "shape": list(tokens.shape),
        "article_row_token_offsets": offsets,
        "input_ids_sha256_le_i64": hashlib.sha256(tokens.tobytes()).hexdigest(),
    }
    if meta["input_ids_sha256_le_i64"] != TOKEN_SHA256:
        raise ValueError(
            "Token bank differs from the pinned Granite tokenizer and sample selection"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    np.save(args.out / "tokens.npy", tokens, allow_pickle=False)
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta["input_ids_sha256_le_i64"])


if __name__ == "__main__":
    main()
