import functools
from typing import Optional

import torch
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.onnx import symbolic_helper, _type_utils

INT64_MAX = 9223372036854775807

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=14)

# Registering custom operation for unflatten
@_onnx_symbolic("aten::unflatten")
@_beartype.beartype
def unflatten(g: jit_utils.GraphContext, input, dim, unflattened_size):
    input_dim = symbolic_helper._get_tensor_rank(input)
    if input_dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )

    # dim could be negative
    input_dim = g.op("Constant", value_t=torch.tensor([input_dim], dtype=torch.int64))
    dim = g.op("Add", input_dim, dim)
    dim = g.op("Mod", dim, input_dim)

    input_size = g.op("Shape", input)

    head_start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    head_end_idx = g.op(
        "Reshape", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    head_part_rank = g.op("Slice", input_size, head_start_idx, head_end_idx)

    dim_plus_one = g.op(
        "Add", dim, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    tail_start_idx = g.op(
        "Reshape",
        dim_plus_one,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
    )
    tail_end_idx = g.op(
        "Constant", value_t=torch.tensor([INT64_MAX], dtype=torch.int64)
    )
    tail_part_rank = g.op("Slice", input_size, tail_start_idx, tail_end_idx)

    final_shape = g.op(
        "Concat", head_part_rank, unflattened_size, tail_part_rank, axis_i=0
    )

    return symbolic_helper._reshape_helper(g, input, final_shape)

@_onnx_symbolic("aten::scaled_dot_product_attention")
@symbolic_helper.parse_args("v", "v", "v", "v", "f", "b", "v")
@_beartype.beartype
def scaled_dot_product_attention(
    g: jit_utils.GraphContext,
    query: torch._C.Value,
    key: torch._C.Value,
    value: torch._C.Value,
    attn_mask: Optional[torch._C.Value] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[torch._C.Value] = None,
):
    assert (not is_causal) or (
        is_causal and symbolic_helper._is_none(attn_mask)
    ), "is_causal and attn_mask cannot be set at the same time"

    scale = symbolic_helper._maybe_get_const(scale, "f")
    if symbolic_helper._is_none(scale):
        scale = _attention_scale(g, query)

    if is_causal:
        attn_mask = _causal_attention_mask(g, query, key)

    # Swap the last two axes of key
    # NOTE: onnx-script has different logic here, because the attribute perms in
    # transpose needs list of ints
    key_shape_builtin = symbolic_helper._get_tensor_rank(key)
    key_transposed_axes = list(range(key_shape_builtin))
    key_transposed_axes[-1], key_transposed_axes[-2] = (
        key_transposed_axes[-2],
        key_transposed_axes[-1],
    )
    key_transposed = g.op("Transpose", key, perm_i=key_transposed_axes)

    # https://github.com/pytorch/pytorch/blob/12da0c70378b5be9135c6fda62a9863bce4a4818/aten/src/ATen/native/transformers/attention.cpp#L653
    # Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    query_scaled = g.op("Mul", query, g.op("Sqrt", scale))
    key_transposed_scaled = g.op("Mul", key_transposed, g.op("Sqrt", scale))
    mul_qk = g.op("MatMul", query_scaled, key_transposed_scaled)

    if symbolic_helper._is_none(attn_mask):
        mul_qk_add = mul_qk
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.BOOL
    ):
        # Turn the Boolean mask to float: attn_mask.masked_fill(not attn_mask, -float('inf'))
        const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
        const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
        attn_mask = g.op("Where", attn_mask, const_zero, const_neg_inf)
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    elif (
        _type_utils.JitScalarType.from_value(attn_mask)
        == _type_utils.JitScalarType.FLOAT
    ):
        mul_qk_add = g.op("Add", mul_qk, attn_mask)
    else:
        raise ValueError(
            f"Unsupported type for attn_mask: {_type_utils.JitScalarType.from_value(attn_mask)}"
        )

    attn_weight = g.op("Softmax", mul_qk_add, axis_i=-1)

    if dropout_p != 0:
        attn_weight = g.op(
            "Dropout",
            attn_weight,
            g.op("Constant", value_t=torch.tensor(dropout_p, dtype=torch.float)),
        )

    return g.op("MatMul", attn_weight, value)

@_beartype.beartype
def _attention_scale(
    g: jit_utils.GraphContext, query: torch._C.Value
) -> torch._C.Value:
    """Calculate the scale factor for the attention result.

    Args:
        query: Tensor of shape [..., L, E]

    Returns:
        Scalar scale factor := 1 / math.sqrt(query.size(-1))
    """
    query_shape = g.op("Shape", query)
    query_shape_last = g.op(
        "Slice",
        query_shape,
        g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)),
        g.op(
            "Constant", value_t=torch.tensor([INT64_MAX], dtype=torch.int64)
        ),
    )
    embedding_size = g.op(
        "Cast",
        query_shape_last,
        to_i=_type_utils.JitScalarType.from_value(query).onnx_type(),
    )
    const_one = g.op("Constant", value_t=torch.tensor([1.0], dtype=torch.float))
    scale = g.op("Div", const_one, g.op("Sqrt", embedding_size))
    return scale

@_beartype.beartype
def _causal_attention_mask(
    g: jit_utils.GraphContext, query: torch._C.Value, key: torch._C.Value
) -> torch._C.Value:
    """Create a causal mask for the given query and key tensors.

    Equivalent to::
        mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_mask = torch.zeros(L, S, dtype=torch.float)
        attn_mask = attn_mask.masked_fill(not mask, -float('inf'))

    Args:
        query: Tensor of shape [..., L, E]
        key: Tensor of shape [..., S, E]

    Returns:
        Tensor of shape [L, S]
    """

    query_shape = g.op("Shape", query)
    key_shape = g.op("Shape", key)

    last_idx = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
    second_last_idx = g.op("Constant", value_t=torch.tensor([-2], dtype=torch.int64))
    target_length = g.op("Slice", query_shape, second_last_idx, last_idx)
    source_length = g.op("Slice", key_shape, second_last_idx, last_idx)
    # attn_mask = torch.ones(L, S) := {
    size = g.op("Concat", target_length, source_length, axis_i=0)
    const_one = g.op("Constant", value_t=torch.tensor([1.0]))
    attn_mask = g.op("Expand", const_one, size)
    # }
    attn_mask = g.op("Trilu", attn_mask, upper_i=0)
    # The causal mask has 0s in the lower triangle and -inf in the upper triangle.
    const_zero = g.op("Constant", value_t=torch.tensor([0.0]))
    const_neg_inf = g.op("Constant", value_t=torch.tensor([-float("inf")]))
    attn_mask = g.op(
        "Where", g.op("Equal", attn_mask, const_zero), const_neg_inf, const_zero
    )
    return attn_mask
