"""Teacher-forced KL in nats per scored next token. All paths are local."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from backend import checkpoint_hashes, load_model, sha_file, tokenizer_info

VERSION = 1


def digest(items):
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        item = json.loads(line)
        ids = item["input_ids"]
        if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError("input_ids must be nonnegative integer token IDs")
        if "score_positions" in item and "score_start" in item:
            raise ValueError("Choose either explicit positions or a suffix")
        start = item.get("score_start", 1)
        if type(start) is not int:
            raise ValueError("score_start must be an integer")
        pos = item.get("score_positions", list(range(start, len(ids))))
        if (
            not isinstance(pos, list)
            or not pos
            or any(type(i) is not int for i in pos)
            or pos != sorted(set(pos))
            or pos[0] < 1
            or pos[-1] >= len(ids)
        ):
            raise ValueError(
                "Expected increasing next-token target indices in [1, length)"
            )
        rows.append((item["id"], ids, pos))
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError("Expected non-empty rows with unique IDs")
    return rows


def identity(rows, tok, model_id, revision, vocab):
    if any(max(r[1]) >= min(tok["length"], vocab) for r in rows):
        raise ValueError("Token ID outside vocabulary")
    return {
        "format_version": VERSION,
        "reference_model": model_id,
        "reference_revision": revision,
        "tokenizer": tok,
        "vocab_size": vocab,
        "rows": len(rows),
        "tokens_scored": sum(len(r[2]) for r in rows),
        "inputs_sha256": digest([(r[0], r[1]) for r in rows]),
        "positions_sha256": digest([(r[0], r[2]) for r in rows]),
    }


def collate(rows, pad, device):
    length = max(len(r[1]) for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    row_idx, logit_idx = [], []
    for i, (_, tokens, targets) in enumerate(rows):
        ids[i, : len(tokens)] = torch.tensor(tokens)
        mask[i, : len(tokens)] = 1
        row_idx.extend([i] * len(targets))
        logit_idx.extend(t - 1 for t in targets)
    return (
        ids.to(device),
        mask.to(device),
        torch.tensor(row_idx, device=device),
        torch.tensor(logit_idx, device=device),
    )


def forward(model, rows, pad):
    ids, mask, ri, ti = collate(rows, pad, model.device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    return logits[ri, ti].double().log_softmax(dim=-1)


def select(logp, ids):
    keep = logp.gather(-1, ids)
    # Summing the complement directly retains tails too small for 1 - sum(top).
    rest = logp.scatter(-1, ids, -torch.inf).logsumexp(dim=-1)
    return keep, rest


def teacher_top(logp, k):
    if not 1 <= k <= logp.size(-1):
        raise ValueError("k out of bounds")
    ids = logp.topk(k, dim=-1).indices
    lp, tail = select(logp, ids)
    return {"ids": ids.int(), "logp": lp, "tail_logmass": tail}


def terms(logp, logq):
    weight = logp.exp()
    finite = weight * torch.where(weight == 0, 0.0, logp - logq)
    return torch.where(torch.isfinite(logp) & torch.isneginf(logq), torch.inf, finite)


def binned(ref, candidate_logp):
    lq, tail_q = select(candidate_logp, ref["ids"].long())
    return terms(ref["logp"], lq).sum(-1) + terms(ref["tail_logmass"], tail_q)


def batches(rows, size):
    if size < 1:
        raise ValueError("batch size must be positive")
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


@torch.inference_mode()
def prepare(model, rows, ident, cache, k, size):
    cache = Path(cache)
    if cache.exists():
        raise FileExistsError(cache)
    manifest = {**ident, "k": k, "batch_size": size, "shards": []}
    # Publish the manifest and shards together, on the same filesystem.
    with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
        stage = Path(tmp) / "cache"
        stage.mkdir()
        for i, batch in enumerate(batches(rows, size)):
            lp = forward(model, batch, ident["tokenizer"]["pad_id"])
            ref = teacher_top(lp, k)
            path = stage / f"{i:06}.safetensors"
            save_file({key: t.cpu() for key, t in ref.items()}, path)
            manifest["shards"].append(sha_file(path))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(cache)
    return manifest


def validate(manifest, expected):
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Incompatible reference cache: {key}")
    if not 1 <= manifest["k"] <= expected["vocab_size"]:
        raise ValueError("Invalid reference k")


@torch.inference_mode()
def score(model, rows, ident, cache):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    validate(manifest, ident)
    groups = list(batches(rows, manifest["batch_size"]))
    if len(groups) != len(manifest["shards"]):
        raise ValueError("Incomplete reference shards")
    total = torch.zeros((), device=model.device, dtype=torch.float64)
    for i, (batch, sha) in enumerate(zip(groups, manifest["shards"], strict=True)):
        path = cache / f"{i:06}.safetensors"
        if sha_file(path) != sha:
            raise ValueError(f"Reference shard checksum mismatch: {path.name}")
        ref = load_file(path, device=str(model.device))
        n = sum(len(r[2]) for r in batch)
        if (
            tuple(ref["ids"].shape) != (n, manifest["k"])
            or ref["logp"].shape != ref["ids"].shape
            or ref["tail_logmass"].shape != (n,)
        ):
            raise ValueError(f"Reference tensor shape mismatch: {path.name}")
        lp = forward(model, batch, ident["tokenizer"]["pad_id"])
        total += binned(ref, lp).sum()
    return total.item() / ident["tokens_scored"]


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    for name in ("prepare-reference", "score"):
        s = subs.add_parser(name)
        s.add_argument(
            "--model", type=Path, required=True, help="Local checkpoint directory"
        )
        s.add_argument("--inputs", type=Path, required=True)
        s.add_argument("--cache", type=Path, required=True)
        s.add_argument(
            "--reference-model", required=True, help="Expected teacher model ID"
        )
        s.add_argument(
            "--reference-revision",
            required=True,
            help="Expected immutable teacher revision",
        )
        s.add_argument("--device", default="cuda:0")
        if name == "prepare-reference":
            s.add_argument("--k", default=128, type=int)
            s.add_argument("--batch-size", default=2, type=int)
    a = p.parse_args()
    start = time.perf_counter()
    rows = read_rows(a.inputs)
    tok = tokenizer_info(a.model)
    if a.action == "prepare-reference":
        cfg = json.loads((a.model / "config.json").read_text())
        if cfg.get("quantization_config"):
            raise ValueError("Expected an unquantized teacher checkpoint")
    model, backend = load_model(a.model, a.device)
    if "source" in backend and (
        backend["source"]["model"],
        backend["source"]["revision"],
    ) != (a.reference_model, a.reference_revision):
        raise ValueError("Candidate was built from a different reference")
    ident = identity(
        rows, tok, a.reference_model, a.reference_revision, model.config.vocab_size
    )
    if a.action == "prepare-reference":
        ident["reference_weights_sha256"] = checkpoint_hashes(a.model)
    sync(a.device)
    loaded = time.perf_counter()
    result = {}
    if a.action == "score":
        result["mean_kl_nats"] = score(model, rows, ident, a.cache)
    else:
        prepare(model, rows, ident, a.cache, a.k, a.batch_size)
    sync(a.device)
    print(
        json.dumps(
            {
                **result,
                "backend": backend,
                "tokens": ident["tokens_scored"],
                "load_and_setup_s": loaded - start,
                "stage_s": time.perf_counter() - loaded,
            }
        )
    )


if __name__ == "__main__":
    main()
