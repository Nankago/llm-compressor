"""Read actual GPTQ nibbles, compare QDA, and probe both codebooks by CUTLASS GEMM."""

import json
import sys

import torch
from qda_quant.solve.quant import fp8_scale_parts, round_fp8
from safetensors.torch import load_file
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A8Fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    convert_bf16_scales_to_fp8,
)


def apply(packed, scale, inputs, *, global_k=None):
    scheme = CompressedTensorsW4A8Fp8("group", 4, 128)
    layer = torch.nn.Module()
    n, k8 = packed.shape
    with torch.device("cuda"):
        scheme.create_weights(
            layer, n, global_k or k8 * 8, [n], k8 * 8, torch.bfloat16, lambda *a: None
        )
    layer.weight_packed.data.copy_(packed)
    layer.weight_scale.data.copy_(scale)
    factors = convert_bf16_scales_to_fp8(
        scheme.kernel.quant_fp8, layer.weight_scale.data
    )
    scheme.process_weights_after_loading(layer)
    result = scheme.apply_weights(layer, inputs, None)
    return result, factors, type(scheme.kernel).__name__


def main(path):
    tensors = load_file(path, device="cuda:0")
    packed, scale, weight = [
        tensors[k] for k in ("weight_packed", "weight_scale", "source")
    ]
    shifts = torch.arange(8, device="cuda") * 4
    codes = (((packed.unsqueeze(-1) >> shifts) & 15) - 8).flatten(1)
    basis = torch.eye(weight.shape[1], dtype=torch.bfloat16, device="cuda")
    baseline, (f8, ch), kernel = apply(packed, scale, basis)
    assert kernel == "CutlassW4A8LinearKernel"
    ref_f8, ref_ch = fp8_scale_parts(scale)
    torch.testing.assert_close(f8.float(), ref_f8, rtol=0, atol=0)
    torch.testing.assert_close(ch, ref_ch, rtol=0, atol=0)
    factor = ref_f8.repeat_interleave(128, dim=1)
    wanted, _ = round_fp8(weight.float(), factor, ch, -8, 7)
    words = (((wanted.int() + 8).unflatten(1, (-1, 8))) << shifts).sum(-1).int()
    aware, _, _ = apply(words, scale, basis)

    def predict(c):
        return ((c * factor).to(torch.float8_e4m3fn).float() * ch).to(torch.bfloat16).T

    assert torch.equal(baseline, predict(codes)) and torch.equal(aware, predict(wanted))
    changed = (codes != wanted) & (
        (aware.T.float() - weight).abs() < (baseline.T.float() - weight).abs()
    )
    row, col = changed.nonzero()[0].tolist()
    # Reproduce the first row-parallel shard locally. Only its first 128
    # input columns participate; use the same source nibbles and local basis.
    assert col < 128
    local_basis = basis[:128, :128].contiguous()
    local_scale = scale[:, :1].contiguous()

    def shard(words):
        return apply(words[:, :16].contiguous(), local_scale, local_basis, global_k=256)

    local_base, (local_f8, local_ch), _ = shard(packed)
    local_aware, _, _ = shard(words)
    local_codes, _ = round_fp8(
        weight[:, :128].float(), local_f8.float(), local_ch, -8, 7
    )
    local_w = weight[:, :128].float()
    reversals = (
        changed[:, :128]
        & (local_codes == codes[:, :128])
        & (
            (local_base.T.float() - local_w).abs()
            < (local_aware.T.float() - local_w).abs()
        )
    )
    local_row, local_col = reversals.nonzero()[0].tolist()
    print(
        json.dumps(
            dict(
                kernel=kernel,
                gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                row=row,
                col=col,
                w=weight[row, col].item(),
                scale=scale[row].tolist(),
                group_f8=ref_f8[row].tolist(),
                channel_fp32=ch[row].item(),
                gptq_code=codes[row, col].item(),
                qda_code=int(wanted[row, col]),
                main_output=baseline[col, row].item(),
                aware_output=aware[col, row].item(),
                strict_improvements=int(changed.sum()),
                count=codes.numel(),
                rank_local=dict(
                    global_k=256,
                    local_k=128,
                    row=local_row,
                    col=local_col,
                    w=local_w[local_row, local_col].item(),
                    global_nearest=int(wanted[local_row, local_col]),
                    global_baseline=baseline[local_col, local_row].item(),
                    global_aware=aware[local_col, local_row].item(),
                    group_f8=local_f8.float()[local_row].tolist(),
                    channel_fp32=local_ch[local_row].item(),
                    nearest_code=int(local_codes[local_row, local_col]),
                    gptq_output=local_base[local_col, local_row].item(),
                    full_aware_output=local_aware[local_col, local_row].item(),
                    strict_reversals=int(reversals.sum()),
                ),
                basis_matches_both=True,
                qda_scale_matches_loader=True,
            )
        )
    )


if __name__ == "__main__":
    import tempfile

    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with (
        tempfile.TemporaryDirectory() as directory,
        set_current_vllm_config(VllmConfig()),
    ):
        init_distributed_environment(1, 0, f"file://{directory}/rendezvous", 0)
        initialize_model_parallel()
        try:
            with torch.inference_mode():
                main(sys.argv[1])
        finally:
            cleanup_dist_env_and_memory()
