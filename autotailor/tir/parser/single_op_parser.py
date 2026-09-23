import onnx_graphsurgeon as gs

from autotailor.tir.blocks.single_op import *

__all__ = ["parse_single_node"]


def parse_single_node(node: gs.Node):
    """Parse a single op block for a gs node

    Args:
      node: the block node

    Returns:
      Block: the built block
    """
    if isinstance(node, list):
        # the first block
        return None
    if node.op == 'Conv':
        if node.attrs['group'] == 1:
            return ConvOp(node)
        else:
            return DepthConvOp(node)
    elif node.op == 'BatchNormalization':
        return BNOp(node)
    elif node.op == 'LayerNormalization':
        return LNOp(node)
    elif node.op == 'Gemm':
        return LinearOp(node)
    elif node.op == 'MatMul':
        for inp in node.inputs:
            if len(inp.inputs) == 0:
                # matmul as a linear
                return LinearMatMulOp(node)
        return MatMulOp(node)
    elif node.op == 'Add':
        for inp in node.inputs:
            if isinstance(inp, gs.Constant) and inp.shape is not None and len(inp.shape) > 0:
                # add as a bias
                return BiasAddOp(node)
        return SingleOp(node)
    elif node.op == "Mul":
        for inp in node.inputs:
            if len(inp.inputs) == 0:
                if len(inp.shape) == 0:
                    # param is a scalar
                    break
                elif inp.shape[0] > 1:
                    # mul as a scaler, param is a tensor
                    return ScaleMulOp(node)
        else:
            return MulOp(node)
        return SingleOp(node)
    elif node.op == 'AveragePool':
        return AveragePoolOp(node)
    elif node.op == 'MaxPool':
        return MaxPoolOp(node)
    elif node.op == 'GlobalAveragePool':
        return GlobalAveragePoolOp(node)
    elif node.op == 'ReduceMean':
        return ReduceMeanOp(node)
    elif node.op == "Flatten":
        return FlattenOp(node)
    elif node.op == 'Reshape' and isinstance(node.inputs[1], gs.Constant):
        return ReshapeOp(node)
    elif node.op == "Shape":
        return ShapeOp(node)
    elif node.op == "Gather":
        return GatherOp(node)
    elif node.op == 'Transpose':
        return TransposeOp(node)
    elif node.op == 'Slice':
        return SliceOp(node)
    elif node.op == 'Concat':
        return ConcatOp(node)
    elif node.op == 'Split':
        return SplitOp(node)
    elif node.op == "Softmax":
        return SoftmaxOp(node)
    # elif node.op == "Gelu":
    #     return GeluOp(node)
    else:
        return SingleOp(node)
