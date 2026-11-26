"""ARC-Easy zero-shot accuracy from full answer-text log likelihood in vLLM."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ARC_SHA = "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("source_tokenizer")
    p.add_argument("test", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    assert hashlib.sha256(a.test.read_bytes()).hexdigest() == ARC_SHA
    tok = AutoTokenizer.from_pretrained(a.source_tokenizer)
    rows = pq.read_table(a.test).to_pylist()
    examples, scores, answers = [], [], []
    for i, r in enumerate(rows):
        prefix = "Question: " + r["question"].strip() + "\nAnswer:"
        begin = tok.encode(prefix, add_special_tokens=False)
        answers.append(r["choices"]["label"].index(r["answerKey"]))
        scores.append([])
        for choice in r["choices"]["text"]:
            ids = tok.encode(prefix + " " + choice.strip(), add_special_tokens=False)
            assert ids[: len(begin)] == begin and len(ids) > len(begin)
            examples.append((i, len(begin), ids))
    engine = LLM(
        model=a.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=max(len(x[2]) for x in examples) + 1,
        max_num_seqs=32,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=False,
        seed=13,
        disable_log_stats=True,
    )
    params = SamplingParams(
        max_tokens=1,
        temperature=0,
        prompt_logprobs=0,
        detokenize=False,
        flat_logprobs=True,
    )
    for start in range(0, len(examples), 128):
        batch = examples[start : start + 128]
        outs = engine.generate(
            [{"prompt_token_ids": ids} for _, _, ids in batch], params, use_tqdm=False
        )
        for (i, begin, ids), out in zip(batch, outs, strict=True):
            assert out.prompt_token_ids == ids
            scores[i].append(
                sum(
                    out.prompt_logprobs[j][ids[j]].logprob
                    for j in range(begin, len(ids))
                )
            )
    pred = [max(range(len(s)), key=s.__getitem__) for s in scores]
    correct = [x == y for x, y in zip(pred, answers, strict=True)]
    a.out.write_text(
        json.dumps(
            {"total": len(rows), "correct": sum(correct), "correct_by_row": correct}
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
