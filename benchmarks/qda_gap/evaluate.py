"""Exact next-token KL on one fixed anchor per paragraph, plus teacher-forced NLL."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("tokens", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--make-reference", action="store_true")
    args = parser.parse_args()
    data = args.tokens.read_bytes()
    rows = json.loads(data)
    engine = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=4,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
        max_logprobs=-1,
        seed=13,
        disable_log_stats=True,
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        detokenize=False,
        logprobs=-1,
        prompt_logprobs=0,
        flat_logprobs=True,
    )
    args.reference.mkdir(exist_ok=True)
    results = []
    for start in range(0, len(rows), 4):
        batch = rows[start : start + 4]
        outputs = engine.generate(
            [{"prompt_token_ids": r["ids"][:-1]} for r in batch], params, use_tqdm=False
        )
        for row, out in zip(batch, outputs, strict=True):
            ids = row["ids"]
            assert out.prompt_token_ids == ids[:-1]
            lp = out.outputs[0].logprobs
            indices = np.asarray(lp.token_ids, dtype=np.int64)
            # The sampled id may also appear in the all-vocabulary list.
            vocab = engine.get_tokenizer().vocab_size
            width = engine.model_config.get_vocab_size()
            assert (
                indices.min() == 0
                and len(np.unique(indices)) == width
                and width >= vocab
            )
            logq = np.empty(width, dtype=np.float32)
            logq[indices] = lp.logprobs
            assert (
                np.isfinite(logq).all()
                and abs(np.exp(logq.astype("d")).sum() - 1) < 1e-4
            )
            assert logq[out.outputs[0].token_ids[0]] == logq.max()
            ref = args.reference / f"{row['row']}.npy"
            if args.make_reference:
                np.save(ref, logq)
            logp = np.load(ref).astype("d")
            kl = float(np.sum(np.exp(logp) * (logp - logq)))
            nll = -sum(
                out.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids) - 1)
            )
            nll -= float(logq[ids[-1]])
            results.append(
                {"row": row["row"], "kl": kl, "nll_sum": nll, "positions": len(ids) - 1}
            )
    report = {
        "tokens_sha256": hashlib.sha256(data).hexdigest(),
        "kl_mean": float(np.mean([r["kl"] for r in results])),
        "nll_mean": sum(r["nll_sum"] for r in results)
        / sum(r["positions"] for r in results),
        "rows": results,
    }
    args.out.write_text(json.dumps(report) + "\n")


if __name__ == "__main__":
    main()
