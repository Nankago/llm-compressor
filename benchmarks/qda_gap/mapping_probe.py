"""Screen AWQ mapping discovery on pretrained Qwen3.5-0.8B-Base on CPU."""

import json
import sys
from dataclasses import replace

import torch
from qda_quant.arch import describe
from qda_quant.arch.verify import verify
from transformers import AutoModelForImageTextToText

from llmcompressor.modifiers.transform.awq.dynamic_mappings import (
    get_layer_mappings_from_model,
)


def main(model_dir):
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    mappings = get_layer_mappings_from_model(model)
    graph = describe(model)
    group = next(
        g for g in graph.layers[0].groups if g.norm.endswith("input_layernorm")
    )
    key = group.modules[0]
    readers = graph.extra["readers"]
    incomplete = {**readers, key: tuple(r for r in readers[key] if r != key)}
    assert len(incomplete[key]) + 1 == len(readers[key])
    damaged = replace(graph, extra={**graph.extra, "readers": incomplete})
    rejected = verify(model, damaged)
    assert not graph.extra["dropped"]
    assert any(group.norm in msg for msg in rejected.extra["dropped"])
    print(
        json.dumps(
            {
                "model_class": type(model).__name__,
                "upstream_mapping_count": len(mappings),
                "layer_count": len(graph.layers),
                "verified": graph.verified,
                "norm": group.norm,
                "readers": readers[key],
                "original_dropped": graph.extra["dropped"],
                "removed_reader": key,
                "negative_control_dropped": rejected.extra["dropped"],
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
