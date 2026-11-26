"""ARC-Easy zero-shot answer-text likelihood for the packed Transformers exports."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from inputs import digest
from score import load
from transformers import AutoTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ("model", "tokenizer", "test", "out"):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--count", type=int, default=2376)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if (
        digest(args.test)
        != "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"
    ):
        raise ValueError("ARC-Easy split mismatch")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    config = json.loads((args.model / "config.json").read_text())
    model = load(args.model, args.device, "quantization_config" in config)
    raw = pq.read_table(args.test).to_pylist()
    selected = np.sort(
        np.random.default_rng(797).choice(len(raw), args.count, replace=False)
    )
    examples, answers, scores = [], [], []
    for i, row_id in enumerate(selected):
        row = raw[row_id]
        prefix = "Question: " + row["question"].strip() + "\nAnswer:"
        begin = tok.encode(prefix, add_special_tokens=False)
        answers.append(row["choices"]["label"].index(row["answerKey"]))
        scores.append([None] * len(row["choices"]["text"]))
        for j, choice in enumerate(row["choices"]["text"]):
            ids = tok.encode(prefix + " " + choice.strip(), add_special_tokens=False)
            if ids[: len(begin)] != begin or len(ids) <= len(begin):
                raise ValueError("Answer boundary differs")
            examples.append((i, j, len(begin), ids))
    examples.sort(key=lambda e: len(e[3]))
    start = time.perf_counter()
    for step in range(0, len(examples), args.batch):
        batch = examples[step : step + args.batch]
        width = max(len(e[3]) for e in batch)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        which, pos, gold, counts = [], [], [], []
        for r, (_, _, b, seq) in enumerate(batch):
            ids[r, : len(seq)] = torch.tensor(seq)
            mask[r, : len(seq)] = 1
            which.extend([r] * (len(seq) - b))
            pos.extend(range(b - 1, len(seq) - 1))
            gold.extend(seq[b:])
            counts.append(len(seq) - b)
        ids, mask = ids.to(args.device), mask.to(args.device)
        features = model.model(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).last_hidden_state
        logits = model.lm_head(features[which, pos]).float().log_softmax(-1)
        ll = logits.gather(-1, torch.tensor(gold, device=ids.device)[:, None]).flatten()
        if step == 0:
            full = (
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                .logits[which, pos]
                .float()
                .log_softmax(-1)
            )
            torch.testing.assert_close(logits, full, rtol=0, atol=0.13)
        for (i, j, _, _), piece in zip(batch, ll.split(counts), strict=True):
            scores[i][j] = piece.double().sum().item()
        if step % (args.batch * 100) == 0:
            print(f"{step}/{len(examples)} choices", flush=True)
    preds = [int(np.argmax(s)) for s in scores]
    correct = np.asarray(preds) == answers
    args.out.write_text(
        json.dumps(
            dict(
                total=len(raw),
                evaluated=len(selected),
                correct=int(correct.sum()),
                row_ids=selected.tolist(),
                correct_bits=np.packbits(correct, bitorder="little").tobytes().hex(),
                elapsed_s=time.perf_counter() - start,
                model_sha256=digest(args.model / "model.safetensors"),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
