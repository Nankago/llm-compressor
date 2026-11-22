"""Paired Granite reference formats, with a shared Transformers inference path."""

import argparse
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file

from backend import checkpoint_hashes, load_model, sha_file, tokenizer_info
from evaluate import batches, binned, collate, identity, read_rows, sync, teacher_top

DTYPES = {
    "hidden": {"hidden": "bfloat16"},
    "topk": {"ids": "int32", "logp": "float32", "tail_logmass": "float32"},
}
VERSION = 2


def tensor_hash(t):
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(raw).hexdigest()


class Projection:
    """Granite's native BF16 output head followed by the model's logit divisor."""

    def __init__(self, model):
        if model.config.model_type != "granitemoe":
            raise ValueError("This paired adapter implements GraniteMoEForCausalLM")
        self.head = model.get_output_embeddings()
        self.scale = model.config.logits_scaling
        self.signature = {
            "weight_sha256": tensor_hash(self.head.weight),
            "shape": list(self.head.weight.shape),
            "dtype": str(self.head.weight.dtype),
            "bias_sha256": (
                tensor_hash(self.head.bias) if self.head.bias is not None else None
            ),
            "logits_scaling": self.scale,
        }

    def logits(self, h):
        return self.head(h) / self.scale

    def __call__(self, h):
        return self.logits(h).float().log_softmax(-1)


def hidden(model, batch, pad):
    ids, mask, ri, ti = collate(batch, pad, model.device)
    # GraniteMoeModel applies its own final RMSNorm before last_hidden_state.
    h = model.model(
        input_ids=ids, attention_mask=mask, use_cache=False
    ).last_hidden_state
    return h[ri, ti]


def full_kl(p, q):
    # Same log_target direction as #2663. Restore the limits for impossible tokens.
    loss = F.kl_div(q, p, reduction="none", log_target=True)
    return (
        loss.nan_to_num_(nan=torch.inf, posinf=torch.inf)
        .masked_fill_(p.isneginf(), 0)
        .sum(-1)
    )


def make_identity(model, rows, tok, model_id, rev):
    return {
        **identity(rows, tok, model_id, rev, model.config.vocab_size),
        "format_version": VERSION,
    }


def manifest_check(meta, ident, kind, projection):
    for key, value in {**ident, "kind": kind, "storage_dtypes": DTYPES[kind]}.items():
        if meta.get(key) != value:
            raise ValueError(f"Cache mismatch: {key}")
    if meta["capture"] != "granitemoe.post_final_norm" or meta["chunk"] < 1:
        raise ValueError("Unsupported capture point or chunk size")
    if kind == "hidden" and meta["reference_projection"] != projection.signature:
        raise ValueError("Teacher head mismatch; supply --reference-head-model")
    if kind == "topk" and not 1 <= meta["k"] <= ident["vocab_size"]:
        raise ValueError("Invalid k")


class Breakdown:
    """Optional diagnostic only. Events are read after scoring has completed."""

    def __init__(self):
        self.events = defaultdict(list)
        self.read_host_s = 0.0

    @contextmanager
    def interval(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        yield
        end.record()
        self.events[name].append((start, end))

    def result(self):
        torch.cuda.synchronize()
        return {
            "gpu_event_s": {
                n: sum(a.elapsed_time(b) for a, b in spans) / 1000
                for n, spans in self.events.items()
            },
            "host_read_decode_s": self.read_host_s,
        }


@torch.inference_mode()
def prepare(model, proj, rows, ident, cache, kind, k=128, size=2, chunk=128):
    if chunk < 1 or size < 1 or (kind == "topk" and not 1 <= k <= ident["vocab_size"]):
        raise ValueError("Invalid chunk, batch size or k")
    if cache.exists():
        raise FileExistsError(cache)
    meta = {
        **ident,
        "kind": kind,
        "storage_dtypes": DTYPES[kind],
        "chunk": chunk,
        "batch_size": size,
        "capture": "granitemoe.post_final_norm",
        "reference_projection": proj.signature,
        "shards": [],
    }
    if kind == "topk":
        meta["k"] = k
    with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
        stage = Path(tmp) / "cache"
        stage.mkdir()
        for i, batch in enumerate(batches(rows, size)):
            h = hidden(model, batch, ident["tokenizer"]["pad_id"])
            if kind == "hidden":
                tensors = {"hidden": h}
            else:
                chunks = [teacher_top(proj(part), k) for part in h.split(chunk)]
                tensors = {
                    key: torch.cat([part[key] for part in chunks])
                    for key in DTYPES[kind]
                }
            if any(
                str(t.dtype) != "torch." + DTYPES[kind][key]
                for key, t in tensors.items()
            ):
                raise ValueError("Unexpected reference storage dtype")
            path = stage / f"{i:06}.safetensors"
            save_file({key: t.cpu() for key, t in tensors.items()}, path)
            meta["shards"].append(sha_file(path))
        (stage / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        stage.rename(cache)
    return meta


@torch.inference_mode()
def score(model, proj, rows, ident, cache, kind, ref_proj=None, diag=None):
    meta = json.loads((cache / "manifest.json").read_text())
    ref_proj = ref_proj or proj
    manifest_check(meta, ident, kind, ref_proj)
    groups = list(batches(rows, meta["batch_size"]))
    if len(groups) != len(meta["shards"]):
        raise ValueError("Incomplete reference shards")
    interval = diag.interval if diag else lambda _: nullcontext()
    total = torch.zeros((), dtype=torch.float64, device=model.device)
    for i, (batch, sha) in enumerate(zip(groups, meta["shards"], strict=True)):
        start = time.perf_counter() if diag else None
        data = (cache / f"{i:06}.safetensors").read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"Cache checksum mismatch: shard {i}")
        tensors = load(data)
        n = sum(len(r[2]) for r in batch)
        shapes = (
            {"hidden": (n, ref_proj.signature["shape"][1])}
            if kind == "hidden"
            else {"ids": (n, meta["k"]), "logp": (n, meta["k"]), "tail_logmass": (n,)}
        )
        if set(tensors) != set(shapes) or any(
            t.shape != shapes[key] or str(t.dtype) != "torch." + DTYPES[kind][key]
            for key, t in tensors.items()
        ):
            raise ValueError(f"Invalid cache tensors in shard {i}")
        if diag:
            diag.read_host_s += time.perf_counter() - start
        with interval("cache_h2d"):
            ref = {key: t.to(model.device) for key, t in tensors.items()}
        with interval("candidate_backbone"):
            h = hidden(model, batch, ident["tokenizer"]["pad_id"])
        for j in range(0, n, meta["chunk"]):
            span = slice(j, j + meta["chunk"])
            with interval("candidate_head_and_normalization"):
                q = proj(h[span])
            if kind == "hidden":
                with interval("reference_head_and_normalization"):
                    p = ref_proj(ref["hidden"][span])
                with interval("kl_and_sum"):
                    total += full_kl(p, q).sum(dtype=torch.float64)
            else:
                with interval("kl_and_sum"):
                    total += binned({k: t[span] for k, t in ref.items()}, q).sum(
                        dtype=torch.float64
                    )
    return total.item() / ident["tokens_scored"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare-reference", "score"):
        p = subs.add_parser(name)
        p.add_argument("--format", choices=list(DTYPES), required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--inputs", type=Path, required=True)
        p.add_argument("--cache", type=Path, required=True)
        p.add_argument("--reference-model", required=True)
        p.add_argument("--reference-revision", required=True)
        p.add_argument("--device", default="cuda:0")
        if name == "prepare-reference":
            p.add_argument("--batch-size", type=int, default=2)
            p.add_argument("--chunk", type=int, default=128)
            p.add_argument("--k", type=int, default=128)
        else:
            p.add_argument(
                "--reference-head-model",
                type=Path,
                help="Teacher checkpoint for a differing output head",
            )
    a = parser.parse_args()
    start = time.perf_counter()
    rows = read_rows(a.inputs)
    model, backend = load_model(a.model, a.device)
    if "source" in backend and (
        backend["source"]["model"],
        backend["source"]["revision"],
    ) != (a.reference_model, a.reference_revision):
        raise ValueError("Candidate lineage mismatch")
    proj = Projection(model)
    ident = make_identity(
        model, rows, tokenizer_info(a.model), a.reference_model, a.reference_revision
    )
    reference = None
    if a.action == "prepare-reference":
        if json.loads((a.model / "config.json").read_text()).get("quantization_config"):
            raise ValueError("Expected an unquantized teacher")
        ident["reference_weights_sha256"] = checkpoint_hashes(a.model)
    elif a.reference_head_model:
        if a.format != "hidden":
            parser.error("--reference-head-model applies to hidden caches only")
        teacher, _ = load_model(a.reference_head_model, "cpu")
        reference = Projection(teacher)
        reference.head.to(a.device)
        del teacher
    sync(a.device)
    ready = time.perf_counter()
    out = {}
    if a.action == "prepare-reference":
        prepare(model, proj, rows, ident, a.cache, a.format, a.k, a.batch_size, a.chunk)
    else:
        out["mean_kl_nats"] = score(
            model, proj, rows, ident, a.cache, a.format, reference
        )
    sync(a.device)
    print(
        json.dumps(
            {
                **out,
                "tokens": ident["tokens_scored"],
                "setup_s": ready - start,
                "stage_s": time.perf_counter() - ready,
                "backend": backend,
            }
        )
    )


if __name__ == "__main__":
    main()
