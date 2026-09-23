"""Strict loading of AutoTailor's nested shared-weight checkpoints."""


def load_shared_checkpoint(path, target_weights):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not target_weights or state.keys() != target_weights.keys():
        raise ValueError("Checkpoint operator names do not match the parsed supernet")
    count = 0
    # Validate everything before mutating any model parameters. Never fall back
    # silently to ONNX weights for a missing or incompatible checkpoint entry.
    for name, target in target_weights.items():
        source = state[name]
        if not isinstance(source, dict) or source.keys() != target.keys():
            raise ValueError(f"Checkpoint parameter names differ for {name}")
        for key, tensor in target.items():
            value = source[key]
            if (not isinstance(value, torch.Tensor) or value.shape != tensor.shape
                    or value.dtype != tensor.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid checkpoint tensor {name}/{key}")
            count += 1
    with torch.no_grad():
        for name, target in target_weights.items():
            for key, tensor in target.items():
                tensor.copy_(state[name][key])
    return {"epoch": checkpoint.get("epoch"), "operator_count": len(state),
            "tensor_count": count, "strict": True}
