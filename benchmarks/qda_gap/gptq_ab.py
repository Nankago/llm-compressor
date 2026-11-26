"""Research adapter for the group-128 W4AFP8 GPTQ block.

The optional block uses the pinned QDA package at runtime. Only dense symmetric
group-128 CUTLASS W4AFP8, no activation ordering and block size 128 are tested.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from datasets import Dataset
from qda_quant.solve._triton import gptq_block
from qda_quant.solve.quant import fp8_scale_parts
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier, gptq_quantize

TRAIN_SHA = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"


def calibration(tokenizer, path, seed):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TRAIN_SHA
    texts = pq.read_table(path, columns=["text"]).column(0).to_pylist()
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    rows = []
    for i in order:
        ids = tokenizer.encode(texts[i].strip(), add_special_tokens=False)
        if len(ids) >= 128:
            rows.append(ids[:128])
        if len(rows) == 16:
            break
    assert len(rows) == 16
    return Dataset.from_dict(
        {"input_ids": rows, "attention_mask": [[1] * 128 for _ in rows]}
    ), hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def block_adapter(aware):
    def update(
        W1,
        Hinv1,
        Q1,
        Err1,
        losses1,
        *,
        scale,
        zero_point,
        global_scale,
        g_idx,
        quant_args,
        i1,
    ):
        assert (
            quant_args.num_bits,
            quant_args.symmetric,
            quant_args.group_size,
            quant_args.actorder,
        ) == (4, True, 128, None)
        assert global_scale is None and W1.shape[-1] == 128 and i1 % 128 == 0
        index = i1 // 128
        stored = scale[..., index : index + 1].float()
        work = W1.contiguous()
        codes = torch.empty_like(work)
        factors = None
        effective = stored
        if aware:
            f8, channel = fp8_scale_parts(scale)
            group = f8[..., index : index + 1]
            if (group == 0).any():
                raise ValueError(
                    "This experiment requires nonzero served group factors"
                )
            factors = (group.expand_as(work).contiguous(), channel)
            effective = group * channel
        err = gptq_block(
            work,
            effective.expand_as(work).contiguous(),
            None,
            factors,
            Hinv1.contiguous(),
            codes,
            0,
            128,
            -8,
            7,
        )
        Err1.copy_(err)
        losses1.copy_(err.square())
        # Main's exporter recovers the nibble by dividing by the stored BF16 scale.
        # QDA's error already uses the served level. Keep the packing carrier separate.
        Q1.copy_((codes * stored).to(torch.bfloat16))

    return update


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["main", "control", "aware"])
    p.add_argument("model")
    p.add_argument("train", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--calibration-seed", type=int, default=13)
    a = p.parse_args()
    torch.manual_seed(13)
    tok = AutoTokenizer.from_pretrained(a.model)
    data, digest = calibration(tok, a.train, a.calibration_seed)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    )
    scheme = preset_name_to_scheme("W4AFP8", ["Linear"])
    scheme.weights.observer = "memoryless_minmax"
    recipe = GPTQModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
        actorder=None,
        block_size=128,
        dampening_frac=0.01,
    )
    target = (
        gptq_quantize.gptq_block_update
        if a.mode == "main"
        else block_adapter(a.mode == "aware")
    )
    with patch.object(gptq_quantize, "gptq_block_update", target):
        oneshot(
            model=model,
            tokenizer=tok,
            dataset=data,
            recipe=recipe,
            max_seq_length=128,
            num_calibration_samples=len(data),
            shuffle_calibration_samples=False,
            propagate_error=False,
        )
    torch.cuda.synchronize()
    solved = time.perf_counter()
    model.save_pretrained(
        a.out, save_compressed=True, quantization_format="pack-quantized"
    )
    tok.save_pretrained(a.out)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "mode": a.mode,
                "calibration_sha256": digest,
                "load_and_oneshot_s": solved - start,
                "export_s": time.perf_counter() - solved,
                "wall_s": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
