"""Create the stock data-free W4AFP8 RTN checkpoint."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--packed",
        action="store_true",
        help="Explicitly select the dense CUTLASS checkpoint layout",
    )
    args = parser.parse_args()
    torch.manual_seed(13)
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    recipe = QuantizationModifier(targets="Linear", scheme="W4AFP8", ignore=["lm_head"])
    if args.packed:
        oneshot(model=model, recipe=recipe, tokenizer=tokenizer)
        model.save_pretrained(
            args.out, save_compressed=True, quantization_format="pack-quantized"
        )
        tokenizer.save_pretrained(args.out)
    else:
        oneshot(
            model=model, recipe=recipe, tokenizer=tokenizer, output_dir=str(args.out)
        )
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "load_s": loaded - start,
                "quantize_and_save_s": time.perf_counter() - loaded,
                "wall_s": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )


if __name__ == "__main__":
    main()
