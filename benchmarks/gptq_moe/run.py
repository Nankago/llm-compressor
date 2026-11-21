"""Run one fresh-model replicate. See README.md for the command matrix."""

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from instrument import Trace
from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import get_linearized_moes, linearize_moe
from llmcompressor.modifiers.gptq import GPTQModifier, helpers

BASE = "f7d824cee12ff15bfa719ba8a7ea8ace7566df67"
MODEL = "FlameF0X/TinyMoE-100m-2x8-retrained"
REVISION = "0c1d097c964b949bc0092fa8e99c88c325421167"
WEIGHTS = "feb1cc5a5140f3ec7c357ecb9b13f54b17575d76c792459f563ce061a1527c6a"
SEED = 42
LENGTH = 128
ROOT = Path(__file__).resolve().parents[2]


def data(tokenizer, sample_count):
    corpus = subprocess.check_output(["git", "show", f"{BASE}:README.md"], cwd=ROOT)
    ids = tokenizer.encode(corpus.decode(), add_special_tokens=False)
    offsets = list(range(0, len(ids) - LENGTH + 1, LENGTH))
    random.Random(SEED).shuffle(offsets)
    assert 0 < sample_count <= len(offsets)
    offsets = offsets[:sample_count]
    windows = torch.tensor([ids[i : i + LENGTH] for i in offsets], dtype=torch.long)
    digest = hashlib.sha256(windows.numpy().astype("<i8").tobytes()).hexdigest()
    rows = [
        {"input_ids": row, "attention_mask": torch.ones_like(row)} for row in windows
    ]
    return DataLoader(rows, batch_size=1, shuffle=False), offsets, digest


def gpu_snapshot():
    props = torch.cuda.get_device_properties(0)
    fields = "index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu"
    return (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                f"GPU-{props.uuid}",
                f"--query-gpu={fields}",
                "--format=csv",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", required=True, help="auto, none or positive integer")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--host-profile", action="store_true", help="Additional diagnostic replicate"
    )
    parser.add_argument(
        "--observer-details",
        action="store_true",
        help="Split observer work into its own event intervals",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    mode = {"auto": "auto", "none": None}.get(args.mode, args.mode)
    if mode not in ("auto", None):
        mode = int(mode)
    if torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one GPU")
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    # Bound only this process's PyTorch caching allocator on the shared device.
    torch.cuda.set_per_process_memory_fraction(0.045)
    pre_start = time.perf_counter()
    source = args.model_dir or Path(
        snapshot_download(
            MODEL, revision=REVISION, allow_patterns=["*.json", "model.safetensors"]
        )
    )
    with (source / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHTS
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    loader, offsets, input_digest = data(tokenizer, args.samples)
    model, info = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
        local_files_only=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if info.get(key):
            raise RuntimeError(f"Checkpoint load: {key}={info[key]}")
    model.eval()
    model.config.output_router_logits = False
    model.config.use_cache = False
    linearize_moe(model)
    assert len(get_linearized_moes(model)) == model.config.num_hidden_layers
    recipe = GPTQModifier(
        scheme="W4A16",
        targets="Linear",
        ignore=["lm_head", r"re:.*\.gate$"],
        batched_quantization=mode,
        block_size=128,
        dampening_frac=0.01,
    )
    expected = {
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n != "lm_head" and not n.endswith(".gate")
    }
    trace = Trace()
    before = gpu_snapshot()
    datetime_utc = datetime.now(timezone.utc).isoformat()
    preparation = time.perf_counter() - pre_start
    host_profile = cProfile.Profile() if args.host_profile else None
    with trace.install(observer_details=args.observer_details):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if host_profile:
            host_profile.enable()
        oneshot(
            model=model,
            tokenizer=tokenizer,
            recipe=recipe,
            dataset=loader,
            pipeline="sequential",
            sequential_targets_per_subgraph=1,
            batch_size=1,
            max_seq_length=LENGTH,
            num_calibration_samples=args.samples,
            shuffle_calibration_samples=False,
            moe_calibrate_all_experts=True,
            propagate_error=True,
            tie_word_embeddings=False,
            output_dir=None,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        if host_profile:
            host_profile.disable()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    after = gpu_snapshot()
    measured = trace.seconds()
    seen = [n for batch in trace.batches for n in batch["modules"]]
    assert len(seen) == len(expected) and set(seen) == expected
    assert trace.calls["accumulate"] == len(seen) * args.samples
    assert (
        trace.calls["calib_forward_outside_gptq_hook"]
        == trace.calls["propagate_forward"]
    )
    counts = Counter()
    for batch in trace.batches:
        branch = batch["cholesky"]
        size = batch["size"]
        assert batch["cholesky_ex_calls"] == (size if branch == "per_item" else 1)
        assert (branch == "per_item") == (size < helpers.MIN_BATCHED_CHOLESKY_SIZE)
        counts[branch] += size
    packages = [
        "torch",
        "triton",
        "llmcompressor",
        "compressed-tensors",
        "transformers",
        "datasets",
        "huggingface-hub",
        "accelerate",
        "auto-round",
        "safetensors",
        "tokenizers",
        "numpy",
        "scipy",
        "pydantic",
        "loguru",
        "pillow",
        "pyyaml",
        "requests",
        "tqdm",
        "nvidia-ml-py",
        "setuptools-scm",
    ]
    result = {
        "upstream": BASE,
        "model": MODEL,
        "revision": REVISION,
        "weight_sha256": WEIGHTS,
        "seed": SEED,
        "samples": args.samples,
        "sequence_length": LENGTH,
        "token_offsets": offsets,
        "input_ids_sha256_le_i64": input_digest,
        "mode": mode,
        "start_utc": datetime_utc,
        "gpu_before": before,
        "gpu_after": after,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "torch_runtime": str(torch.__version__),
            "cudnn": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "packages": {p: version(p) for p in packages},
        },
        "setup_wall_s": preparation,
        "oneshot_wall_s": wall,
        "cuda_event_s": measured,
        "unattributed_wall_s": wall - sum(measured.values()),
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "batch_histogram": dict(Counter(batch["size"] for batch in trace.batches)),
        "module_count_by_cholesky": dict(counts),
        "rtn_fallback_modules": trace.fallback,
        "calls": dict(trace.calls),
        "caps_subgraph_representative_limit": trace.caps,
        "batches": trace.batches,
    }
    if args.observer_details:
        assert trace.calls["imatrix_input_hook"] == len(seen) * args.samples
    if host_profile:
        stats = pstats.Stats(host_profile)
        rows = []
        for (filename, lineno, function), (
            _,
            calls,
            own,
            cumulative,
            _,
        ) in stats.stats.items():
            # Keep portable source references. The benchmark runs in a private checkout.
            if "/llmcompressor/" in filename:
                filename = "llmcompressor/" + filename.split("/llmcompressor/", 1)[1]
            else:
                filename = Path(filename).name
            rows.append(
                {
                    "function": f"{filename}:{lineno} {function}",
                    "calls": calls,
                    "own_s": own,
                    "cumulative_s": cumulative,
                }
            )
        result["host_profile"] = {
            key: sorted(rows, key=lambda row: row[key], reverse=True)[:40]
            for key in ("own_s", "cumulative_s")
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out, wall, result["batch_histogram"], counts, flush=True)


if __name__ == "__main__":
    main()
