"""Evaluate the two packed checkpoints using the pinned qda_kl_eval code on PYTHONPATH."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from backend import granite_packed, load_model, sha_file, tokenizer_info
from evaluate import batches, forward, read_rows
from paired import Projection, full_kl, make_identity, score
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

ARC_SHA = "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"


def changed_tensors(on, off):
    kinds = Counter()
    totals = Counter()
    with safe_open(on, framework="pt") as a, safe_open(off, framework="pt") as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError("Different checkpoint layouts")
        for key in a.keys():
            kind = key.rsplit(".", 1)[-1]
            totals[kind] += 1
            left, right = a.get_tensor(key), b.get_tensor(key)
            kinds[kind] += (
                left.dtype != right.dtype
                or left.shape != right.shape
                or not torch.equal(
                    left.reshape(-1).view(torch.uint8),
                    right.reshape(-1).view(torch.uint8),
                )
            )
    return {"changed": dict(kinds), "total": dict(totals)}


def arc_choices(path, tok):
    import pyarrow.parquet as pq

    if sha_file(path) != ARC_SHA:
        raise ValueError("ARC-Easy test shard differs")
    questions = pq.read_table(path).to_pylist()
    choices, gold = [], []
    for i, row in enumerate(questions):
        prefix = "Question: " + row["question"].strip() + "\nAnswer:"
        prompt_ids = tok.encode(prefix, add_special_tokens=False)
        gold.append(row["choices"]["label"].index(row["answerKey"]))
        for j, choice in enumerate(row["choices"]["text"]):
            ids = tok.encode(prefix + " " + choice.strip(), add_special_tokens=False)
            if ids[: len(prompt_ids)] != prompt_ids or len(ids) <= len(prompt_ids):
                raise ValueError("Ambiguous continuation boundary")
            choices.append((i, j, ids, len(prompt_ids)))
    return sorted(choices, key=lambda x: len(x[2])), gold


def arc_accuracy(model, choices, gold, pad):
    scores = [dict() for _ in gold]
    for batch in batches(choices, 16):
        n = max(len(ids) for _, _, ids, _ in batch)
        ids = torch.full((len(batch), n), pad, dtype=torch.long, device="cpu")
        mask = torch.zeros_like(ids)
        owners, positions, targets = [], [], []
        for i, (_, _, seq, start) in enumerate(batch):
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
            owners.extend([i] * (len(seq) - start))
            positions.extend(range(start - 1, len(seq) - 1))
            targets.extend(seq[start:])
        logits = model(
            input_ids=ids.to(model.device),
            attention_mask=mask.to(model.device),
            use_cache=False,
        ).logits
        lp = logits[owners, positions].float().log_softmax(-1)
        vals = (
            lp.gather(1, torch.tensor(targets, device=model.device)[:, None])
            .flatten()
            .cpu()
        )
        offset = 0
        for i, j, seq, start in batch:
            count = len(seq) - start
            scores[i][j] = vals[offset : offset + count].double().sum().item()
            offset += count
    # Raw log-likelihood of the entire answer text; original choice order breaks ties.
    return [
        max(sorted(opts), key=opts.get) == label
        for opts, label in zip(scores, gold, strict=True)
    ]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("teacher", "on", "off", "inputs", "out"):
        p.add_argument(key, type=Path)
    p.add_argument("--granite-cache", type=Path)
    p.add_argument("--arc", type=Path)
    a = p.parse_args()
    device = "cuda:0"
    rows = read_rows(a.inputs)
    models = {"BF16": load_model(a.teacher, device)[0]}
    tok = tokenizer_info(a.teacher)
    sources, signatures, loaders = {}, {}, {}
    for arm, path in (("on", a.on), ("off", a.off)):
        source = json.loads((path / "qda_source.json").read_text())
        if source["propagate_error"] != (arm == "on") or tokenizer_info(path) != tok:
            raise ValueError("Candidate provenance mismatch")
        if a.granite_cache:
            models[arm], loaders[arm] = load_model(path, device)
        else:
            # Both native architectures expose the same linearized experts checkpoint layout.
            model, loaders[arm] = granite_packed(
                path, AutoConfig.from_pretrained(path, local_files_only=True)
            )
            models[arm] = model.eval().to(device)
            loaders[arm][
                "decoder"
            ] = "compressed-tensors -> BF16; Transformers native Mixtral experts"
        sources[arm] = source
        signatures[arm] = sha_file(path / "model.safetensors")
    if sources["on"]["calibration_sha256"] != sources["off"]["calibration_sha256"]:
        raise ValueError("Calibration input mismatch")
    ident = make_identity(
        models["BF16"], rows, tok, source["model"], source["revision"]
    )
    result = {
        "identity": ident,
        "source": sources,
        "checkpoint_sha256": signatures,
        "backend": loaders,
        "bitwise_comparison": changed_tensors(
            a.on / "model.safetensors", a.off / "model.safetensors"
        ),
    }
    per_seq = {n: [] for n in models}
    nll = {n: 0.0 for n in models}
    counts = [len(r[2]) for r in rows]
    for batch in batches(rows, 2):
        teacher = forward(models["BF16"], batch, tok["pad_id"])
        target = torch.tensor([r[1][pos] for r in batch for pos in r[2]], device=device)
        for name, model in models.items():
            lp = teacher if name == "BF16" else forward(model, batch, tok["pad_id"])
            div = full_kl(teacher, lp)
            nll[name] -= lp.gather(1, target[:, None]).sum().item()
            per_seq[name].extend(
                t.sum().item() for t in div.split([len(r[2]) for r in batch])
            )
    total = sum(counts)
    result["exact_kl_nats"] = {n: sum(values) / total for n, values in per_seq.items()}
    result["perplexity"] = {n: math.exp(x / total) for n, x in nll.items()}
    result["position_counts"] = counts
    result["sequence_kl_sums"] = per_seq
    if a.granite_cache:
        result["cached_exact_kl_nats"] = {
            n: score(m, Projection(m), rows, ident, a.granite_cache, "hidden")
            for n, m in models.items()
        }
        if any(
            abs(value - result["exact_kl_nats"][n]) > 1e-6
            for n, value in result["cached_exact_kl_nats"].items()
        ):
            raise ValueError("Cached and direct exact KL disagree")
    # Paired paragraph bootstrap; resample paragraphs and retain token weighting.
    idx = np.random.default_rng(42).integers(len(rows), size=(10000, len(rows)))
    diff = np.array(per_seq["off"]) - np.array(per_seq["on"])
    result["off_minus_on_kl_bootstrap95"] = np.quantile(
        diff[idx].sum(1) / np.asarray(counts)[idx].sum(1), [0.025, 0.975]
    ).tolist()
    if a.arc:
        choices, gold = arc_choices(
            a.arc, AutoTokenizer.from_pretrained(a.teacher, local_files_only=True)
        )
        correct = {
            n: arc_accuracy(m, choices, gold, tok["pad_id"]) for n, m in models.items()
        }
        result["arc_easy"] = {
            "sha256": ARC_SHA,
            "questions": len(gold),
            "correct": {n: sum(v) for n, v in correct.items()},
            "paired": dict(
                Counter(
                    f"on{int(x)}_off{int(y)}"
                    for x, y in zip(correct["on"], correct["off"], strict=True)
                )
            ),
        }
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
