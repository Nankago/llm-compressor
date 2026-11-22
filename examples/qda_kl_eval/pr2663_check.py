"""Run the pinned PR's actual offline chunk function on Granite pre-norm states."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch

from backend import load_model, sha_file
from evaluate import collate, read_rows
from paired import full_kl


def read_pr(directory):
    for filename, sha in {
        "compute_kl.py": "41a3cbf836039ec9f613e6c2182c7021d5258c141e0c9f7f5f5a9571467b3586",
        "lm_head_utils.py": "7ead8623340131062dbf839e0b3fdc16e9a376c47605cbfee4fd23ba96d3c746",
    }.items():
        if sha_file(directory / filename) != sha:
            raise ValueError("Expected the unmodified #2663 files at 1ac89a7")
    spec = importlib.util.spec_from_file_location(
        "pr_compute_kl", directory / "compute_kl.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._compute_kl_for_chunk


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("pr_directory", "teacher", "w4", "w8", "inputs", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    chunk_fn = read_pr(a.pr_directory)
    rows = read_rows(a.inputs)[:2]
    ids, mask, ri, ti = collate(rows, 0, "cuda:0")
    out = {"positions": len(ri), "chunk": 128, "results": {}}
    for name, path in (("BF16", a.teacher), ("W4", a.w4), ("W8", a.w8)):
        model, _ = load_model(path, "cuda:0")
        norm = model.model.norm
        capture = []
        handle = norm.register_forward_pre_hook(
            lambda _, inputs: capture.append(inputs[0][ri, ti])
        )
        try:
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[
                ri, ti
            ]
        finally:
            handle.remove()
        (h,) = capture
        linear = model.get_output_embeddings()
        weights = {
            "norm_type": "rms_norm",
            "norm_eps": norm.variance_epsilon,
            "norm_weight": norm.weight.float(),
            "lm_head_weight": linear.weight.float(),
            "lm_head_bias": linear.bias,
        }
        logp = logits.double().log_softmax(-1)
        # Isolate projection precision using the checkpoint's own final-normalized activations.
        normed = norm(h)
        head_only_fp32 = normed.float() @ weights["lm_head_weight"].T
        if linear.bias is not None:
            head_only_fp32 += linear.bias.float()
        fp32_head_lp = (head_only_fp32 / model.config.logits_scaling).log_softmax(-1)
        native_replay = linear(normed) / model.config.logits_scaling
        max_logit_difference = (native_replay - logits).abs().max().item()
        if name == "BF16":
            ref, ref_weights, ref_logp = h, weights, logp
            ref_fp32_head_lp = fp32_head_lp
        values = {"native_fp64": full_kl(ref_logp, logp).mean().item()}
        for temp in (1.0, model.config.logits_scaling):
            runs = [
                chunk_fn(r, q, ref_weights, weights, temp)
                for r, q in zip(ref.split(128), h.split(128), strict=True)
            ]
            values[f"pr_temperature_{temp:g}"] = torch.cat(runs).double().mean().item()
        values["native_norm_fp32_head_and_division"] = (
            full_kl(ref_fp32_head_lp, fp32_head_lp).double().mean().item()
        )
        values["native_pre_norm_replay_max_logit_difference"] = max_logit_difference
        out["results"][name] = values
        del model
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
