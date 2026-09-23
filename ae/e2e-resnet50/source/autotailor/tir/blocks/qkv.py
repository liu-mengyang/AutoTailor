
from .block import Block
from .residual import ResidualBlock
from .single_op import *


class QKVBlock(ResidualBlock):
    def __init__(self, start_node, last_node, qk_path, v_path):
        super().__init__(start_node,
                         last_node,
                         qk_path,
                         v_path)
        self.type = "QKVBlock"
        
        assert isinstance(self.start_node, SplitOp) or isinstance(self.start_node, LNOp)
        qk_head_dim = None
        v_head_dim = None
        if isinstance(self.start_node, SplitOp):
            qk_head_dim = self.start_node.features["split"][0]
            assert qk_head_dim == self.start_node.features["split"][1]
            
            v_head_dim = self.start_node.features["split"][-1]
        elif isinstance(self.start_node, LNOp):
            # find reshape op to achieve qk head dim
            qk_head_dim = self.start_node.features["in_channel"]
            v_head_dim = qk_head_dim
        assert qk_head_dim is not None and v_head_dim is not None
        # extract information
        self.features["trans_type"] = 'active'
        self.features["qk_head_dim"] = qk_head_dim
        self.features["max_in_channel"] = v_head_dim + 2 * qk_head_dim
        self.features["max_out_channel"] = v_head_dim + 2 * qk_head_dim
        self.features["in_channel"] = v_head_dim + 2 * qk_head_dim
        self.features["out_channel"] = v_head_dim + 2 * qk_head_dim
        self.features["max_v_scale"] = v_head_dim // qk_head_dim
        self.features["v_scale"] = v_head_dim // qk_head_dim
        
        self.dynamic_dimensions.append("v_scale")
        
        self.q_path = []
        self.k_path = []
        self.qk_path = []
        self.v_path = []
        self.qk_bias = None
        
        for op in self.main_path:
            self.v_path.append(op)
        for op in self.residual_path:
            if isinstance(op, ResidualBlock):
                # residual path is q path
                for node in op.residual_path:
                    self.q_path.append(node)
                for node in op.main_path:
                    self.k_path.append(node)
            else:
                self.qk_path.append(op)
    
    
    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        cur_width = self.features["in_channel"]
        cur_shape = self.features["in_shape"]
        
        # update split node
        v_scale = self.features["v_scale"]
        qk_head_dim = self.features["qk_head_dim"]
        v_head_dim = qk_head_dim * v_scale
        assert self.start_node.type == "Split" or self.start_node.type == "LayerNormalization"
        if self.start_node.type == "Split":
            self.start_node.update({
                "split": tuple([qk_head_dim, qk_head_dim, v_head_dim]),
                "in_channel": cur_width,
                "in_shape": cur_shape
            })
        elif self.start_node.type == "LayerNormalization":
            self.start_node.update({
                "in_channel": cur_width,
                "in_shape": cur_shape,
            })

        qk_width = qk_head_dim
        qk_shape = list(cur_shape)
        qk_shape[-1] = qk_width
        qk_shape = tuple(qk_shape)
        v_width = v_head_dim
        v_shape = list(cur_shape)
        v_shape[-1] = v_width
        v_shape = tuple(v_shape)
        
        # update qk path
        for op in self.residual_path:
            # print(op)
            # print(f"qk shape: {qk_shape}, qk width: {qk_width}")
            op.update({"in_channel": qk_width, "in_shape": qk_shape})
            qk_width = op.features["out_channel"]
            qk_shape = op.features["out_shape"]
            if isinstance(op, ResidualBlock):
                qk_width = qk_shape[-1]
        
        # update v path
        for i, op in enumerate(self.main_path):
            op.update({"in_channel": v_width, "in_shape": v_shape})
            v_width = op.features["out_channel"]
            v_shape = op.features["out_shape"]
            if isinstance(op, ReshapeOp) and isinstance(self.main_path[i+1], ConvOp):
                v_width = op.features["out_shape"][1]
            elif isinstance(op, ConvOp) and isinstance(self.main_path[i+1], ReshapeOp):
                v_width = self.features["out_channel"]
        
        # update matmul node
        assert self.last_node.type == "MatMul"
        cur_shape = tuple([qk_shape, v_shape])
        self.last_node.update({"in_channel": qk_width, "in_shape": cur_shape})
        self.features["out_channel"] = self.last_node.features["out_channel"]
        self.features["out_shape"] = self.last_node.features["out_shape"]
    
    def transform(self, kv_features: dict) -> None:
        super().transform(kv_features)
        
        if "v_scale" in kv_features.keys():
            # update
            cur_width = self.features["in_channel"]
            cur_shape = self.features["in_shape"]

            self.update({})
    
    def build(self, cache=None, block_id=None):
        op_modules = []
        for op in self.main_path:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)
        
        v_path = nn.Sequential(*op_modules)
        
        op_modules = []
        q_modules = []
        k_modules = []
        # qk_bias = None
        for op in self.residual_path:
            if isinstance(op, ResidualBlock):
                # residual path is q path
                for node in op.residual_path:
                    node_module = node.build(cache=cache, block_id=f"{block_id}-{op.id}-{node.id}")
                    q_modules.append(node_module)
                for node in op.main_path:
                    node_module = node.build(cache=cache, block_id=f"{block_id}-{op.id}-{node.id}")
                    k_modules.append(node_module)
                # qk_bias = op.last_node.super_bias
                # assert qk_bias is not None
            else:
                op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
                op_modules.append(op_module)
        q_path = nn.Sequential(*q_modules)
        k_path = nn.Sequential(*k_modules)
        qk_path = nn.Sequential(*op_modules)
        
        class QKVResidualModule(nn.Module):
            def __init__(self, q_path, k_path, qk_tail_path, v_path):
            # def __init__(self, q_path, k_path, qk_tail_path, v_path, qk_bias):
                super().__init__()
                self.q_path = q_path
                self.k_path = k_path
                self.qk_path = qk_tail_path
                self.v_path = v_path
                # self.qk_bias = qk_bias
            
            def forward(self, q, k, v):
                q_o = self.q_path(q)
                k_o = self.k_path(k)
                qk = self.qk_path(q_o @ k_o)
                # qk = self.qk_path(q @ k + qk_bias)
                v = self.v_path(v)
                y = qk @ v
                return y
        self.module = QKVResidualModule(q_path,
                                        k_path,
                                        qk_path,
                                        v_path)
                                        # qk_bias)
        return self.module
    