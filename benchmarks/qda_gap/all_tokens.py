"""Full-vocabulary KL at every held-out prompt position, validated against vLLM."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from qda_kl.vllm_hooks import TAG, KLWorker
from vllm import LLM, SamplingParams


class ExactWorker(KLWorker):
    """Reuse QDA's request-to-hidden-state mapping. Single unchunked TP1 prefill."""

    def begin_exact(self, key, ids, file, make_reference):
        self._qda_kl().tracks = {key: (ids, file, make_reference)}
        self.exact_result = None

    def _qda_kl_take(self, st, track, hidden, lo, hi, pos0, n_rows, read):
        ids, file, make_reference = track
        assert read and lo == pos0 == 0 and hi == len(ids)
        assert self.exact_result is None
        model = self.model_runner.model
        logits = model.compute_logits(hidden[: n_rows or hi])
        lp = logits[: hi - 1].float().log_softmax(-1)
        target = lp.gather(
            1, torch.tensor(ids[1:], device=lp.device)[:, None]
        ).flatten()
        if make_reference:
            np.save(file, lp.cpu().numpy())
        ref = torch.from_numpy(np.load(file)).to(lp.device, torch.float64)
        kl = (ref.exp() * (ref - lp)).sum(-1)
        self.exact_result = (kl.tolist(), target.tolist())

    def end_exact(self):
        self._qda_kl().tracks = {}
        assert self.exact_result is not None
        return self.exact_result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("tokens", type=Path)
    p.add_argument("cache", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--make-reference", action="store_true")
    a = p.parse_args()
    data = a.tokens.read_bytes()
    rows = json.loads(data)
    a.cache.mkdir(exist_ok=True, parents=True)
    engine = LLM(
        model=a.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
        seed=13,
        worker_extension_cls="all_tokens.ExactWorker",
        disable_log_stats=True,
    )
    runner = engine.collective_rpc("qda_kl_install")
    bound = engine.collective_rpc("qda_kl_kernels")
    results = []
    max_error = 0.0
    for row in rows:
        key, ids = str(row["row"]), row["ids"]
        engine.collective_rpc(
            "begin_exact",
            args=(key, ids, str(a.cache.resolve() / f"{key}.npy"), a.make_reference),
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=0,
            detokenize=False,
            extra_args={TAG: {"key": key, "start": 1}},
        )
        result = engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        [(kls, target)] = engine.collective_rpc("end_exact")
        real = [result.prompt_logprobs[i][t].logprob for i, t in enumerate(ids) if i]
        error = float(np.max(np.abs(np.array(real) - target)))
        if error > 5e-5 or not np.isfinite(kls).all():
            raise RuntimeError(
                f"Invalid score for row {key}, target logprob difference {error}"
            )
        max_error = max(max_error, error)
        results.append(
            {
                "row": row["row"],
                "positions": len(kls),
                "kl_sum": sum(kls),
                "nll_sum": -sum(real),
            }
        )
    n = sum(r["positions"] for r in results)
    a.out.write_text(
        json.dumps(
            {
                "tokens_sha256": hashlib.sha256(data).hexdigest(),
                "runner": runner,
                "bound": bound,
                "max_logprob_difference": max_error,
                "kl": sum(r["kl_sum"] for r in results) / n,
                "nll": sum(r["nll_sum"] for r in results) / n,
                "rows": results,
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
