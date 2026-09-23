
from .residual import ResidualBlock
from .qkv import QKVBlock
from .single_op import *


class AttentionBlock(ResidualBlock):
    def __init__(self, block):
        super().__init__(block.start_node,
                         block.last_node,
                         block.main_path,
                         block.residual_path)
        self.type = "AttentionBlock"
        
        # extract information
        ln = None
        first_linear = None
        last_linear = None
        qkv = None
        self.head_path = []
        self.tail_path = []
        self.qkv_module = None
        
        meet_qkv = False
        for op in self.main_path:
            if isinstance(op, LNOp) and ln is None:
                ln = op
            elif isinstance(op, LinearMatMulOp):
                if first_linear is None:
                    first_linear = op
                elif last_linear is None:
                    last_linear = op
                else:
                    raise NotImplementedError
            elif isinstance(op, QKVBlock) and qkv is None:
                qkv = op
                self.qkv_module = qkv
                meet_qkv = True
                continue
            if meet_qkv:
                self.tail_path.append(op)
            else:
                self.head_path.append(op)
        
        self.features["trans_type"] = 'active'
        self.features["max_in_channel"] = ln.features["in_channel"]
        self.features["max_out_channel"] = ln.features["in_channel"]
        self.features["in_channel"] = ln.features["in_channel"]
        self.features["out_channel"] = ln.features["in_channel"]
        self.features["middle_width"] = first_linear.features["out_channel"]
        self.features["max_v_scale"] = qkv.features["max_v_scale"]
        self.features["v_scale"] = qkv.features["v_scale"]
        
        self.dynamic_dimensions.append("v_scale")
    
    def update(self, kv_features: dict) -> None:
        for k, v in kv_features.items():
            if k in self.features and k in self.update_dimensions:
                self.features[k] = v
            else:
                raise KeyError
        inp_shape = self.features["in_shape"]
        
        cin = self.features["in_channel"]
        self.features["out_channel"] = self.features["in_channel"]
        cout = self.features["out_channel"]
        
        cur_width = cin
        cur_shape = inp_shape
        for op in self.main_path:
            op.update({"in_channel": cur_width,
                        "in_shape": cur_shape})
            cur_width = op.features["out_channel"]
            cur_shape = op.features["out_shape"]
        cur_width = cin
        cur_shape = inp_shape
        for op in self.residual_path:
            op.update({"in_channel": cur_width,
                    "in_shape": cur_shape})
            cur_width = op.features["out_channel"]
            cur_shape = op.features["out_shape"]
        self.features["out_channel"] = cur_width
        self.features["out_shape"] = cur_shape
        self.last_node.features["in_shape"] = cur_shape
        self.last_node.features["out_shape"] = cur_shape
        self.last_node.features["in_channel"] = cur_width
        self.last_node.features["out_channel"] = cur_width
        
    def transform(self, kv_features: dict) -> None:
        super().transform(kv_features)
        
        if "v_scale" in kv_features.keys():
            # update
            cur_width = self.features["in_channel"]
            cur_shape = self.features["in_shape"]
            
            v_scale = kv_features["v_scale"]
            
            middle_width = cur_width * (2+v_scale)
            
            ln = None
            first_linear = None
            last_linear = None
            qkv = None
            for op in self.main_path:
                if isinstance(op, QKVBlock) and qkv is None:
                    qkv = op
                    qkv.transform({"v_scale": v_scale})
                elif isinstance(op, LinearMatMulOp) and first_linear is None:
                    first_linear = op
                    first_linear.update({"in_channel": cur_width,
                                            "in_shape": cur_shape,
                                            "out_channel": middle_width})
                else:
                    op.update({"in_channel": cur_width,
                               "in_shape": cur_shape})
                cur_shape = op.features["out_shape"]
                cur_width = op.features["out_channel"]
    
    def build(self, cache=None, block_id=None):
        main_head_path = []
        main_tail_path = []
        meet_qkv = False
        qkv_module = None
        for op in self.main_path:
            if isinstance(op, QKVBlock):
                qkv_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
                meet_qkv = True
            else:
                op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
                if meet_qkv:
                    main_tail_path.append(op_module)
                else:
                    main_head_path.append(op_module)
        assert qkv_module is not None
        main_head_module = nn.Sequential(*main_head_path)
        main_tail_module = nn.Sequential(*main_tail_path)
        
        class AttentionResidualModule(nn.Module):
            def __init__(self, head_path, tail_path, qkv_module):
                super().__init__()
                self.head_path = head_path
                self.tail_path = tail_path
                self.qkv_module = qkv_module
            
            def forward(self, x):
                # print(f"x shape: {x.shape}")
                if len(self.head_path) > 1:
                    q, k, v = self.head_path(x)
                    # print(f"q shape: {q.shape}")
                    # print(f"k shape: {k.shape}")
                    # print(f"v shape: {v.shape}")
                    qkv = self.qkv_module(q, k, v)
                else:
                    head_o = self.head_path(x)
                    qkv = self.qkv_module(head_o, head_o, head_o)
                y = self.tail_path(qkv)
                y = x + y
                # print(f"y shape: {y.shape}")
                return y
        self.module = AttentionResidualModule(main_head_module, main_tail_module, qkv_module)
        return self.module
