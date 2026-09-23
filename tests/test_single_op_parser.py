import numpy as np
import onnx_graphsurgeon as gs

from autotailor.tir.blocks.single_op import BiasAddOp, SingleOp
import autotailor.tir.globvar as globvar
from autotailor.tir.parser.single_op_parser import parse_single_node


def add_node(constant):
    source = gs.Variable("source", dtype=np.float32, shape=(1, 3))
    data = gs.Variable("data", dtype=np.float32, shape=(1, 3))
    gs.Node(op="Identity", name="producer", inputs=[source], outputs=[data])
    output = gs.Variable("output", dtype=np.float32, shape=(1, 3))
    return gs.Node(op="Add", name="test_add", inputs=[data, constant], outputs=[output])


def graph_input_add_node(constant):
    data = gs.Variable("data", dtype=np.float32, shape=(1, 3))
    output = gs.Variable("output", dtype=np.float32, shape=(1, 3))
    return gs.Node(op="Add", name="input_add", inputs=[data, constant], outputs=[output])


def parse_without_prebuild(node):
    old_prebuild = globvar.tir_prebuild
    old_sharing = globvar.sharing
    try:
        globvar.tir_prebuild = False
        globvar.sharing = False
        return parse_single_node(node)
    finally:
        globvar.tir_prebuild = old_prebuild
        globvar.sharing = old_sharing


def test_scalar_add_is_not_classified_as_bias():
    scalar = gs.Constant("scalar", values=np.array(3.0, dtype=np.float32))

    parsed = parse_without_prebuild(add_node(scalar))

    assert type(parsed) is SingleOp


def test_vector_add_is_classified_as_bias():
    bias = gs.Constant("bias", values=np.zeros((3,), dtype=np.float32))

    parsed = parse_without_prebuild(add_node(bias))

    assert isinstance(parsed, BiasAddOp)


def test_graph_input_is_not_misclassified_as_bias():
    other_input = gs.Variable("other", dtype=np.float32, shape=(1, 3))

    parsed = parse_without_prebuild(graph_input_add_node(other_input))

    assert type(parsed) is SingleOp
