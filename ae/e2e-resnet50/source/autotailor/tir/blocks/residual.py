from .block import Block
from .single_op import *
from .fuse_op import FuseOp

import autotailor.tir.globvar as globvar


class ResidualBlock(Block):
    def __init__(self,
                 start_node,
                 last_node,
                 first_path,
                 second_path):
        super().__init__()
        self.type = 'ResidualBlock'
        self.start_node = start_node

        if len(first_path) > len(second_path):
            self.residual_path = second_path
            self.main_path = first_path
        else:
            self.residual_path = first_path
            self.main_path = second_path

        self.last_node = last_node
        self.paths = {
            "main_path": self.main_path,
            "residual_path": self.residual_path
        }
        self.reduce_type = self.last_node.type

        self.features = {
            'activated': True,
            'trans_type': 'passive',
            'reduce_type': self.reduce_type,
            'main_path': [node.name for node in self.main_path],
            'residual_path': [node.name for node in self.residual_path],
            'in_shape': (),
            'out_shape': (),
            'in_channel': 0,
            'out_channel': 0,
            "max_in_channel": 0,
            "max_out_channel": 0,
        }
        self.recount()

    def recount(self):
        cnt = 0
        for op in self.main_path:
            if isinstance(op, Block):
                op.recount()
            op.id = cnt
            cnt += 1
        for op in self.residual_path:
            if isinstance(op, Block):
                op.recount()
            op.id = cnt
            cnt += 1
        self.last_node.id = cnt

    def update(self, kv_features: dict) -> None:
        """Update features of the block.

        Only in shape and width can be updated.

        Args:
          kv_features: the dictionary of to update features.

        Raises:
          KeyError: the error occured in updating non exist feature.
          NotImplementedError: the error occured in unsupport input shape.
        """
        super().update(kv_features)
        inp_shape = self.features["in_shape"]
        if self.features["in_channel"] == 0:
            # Init channels
            if len(inp_shape) == 4:
                self.features["in_channel"] = inp_shape[1]
                self.features["out_channel"] = inp_shape[1]
                self.features["max_in_channel"] = inp_shape[1]
                self.features["max_out_channel"] = inp_shape[1]
            else:
                self.features["in_channel"] = inp_shape[-1]
                self.features["out_channel"] = inp_shape[-1]
                self.features["max_in_channel"] = inp_shape[-1]
                self.features["max_out_channel"] = inp_shape[-1]

        residual_tensor_shape = inp_shape
        cur_width = self.features['in_channel']

        for node in self.residual_path:
            # update each node in residual path
            # print(node.name)
            # print(residual_tensor_shape)
            # print(cur_width)
            node.update({'in_shape': residual_tensor_shape,
                         'in_channel': cur_width})
            residual_tensor_shape = node.features['out_shape']
            cur_width = node.features['out_channel']

        main_tensor_shape = inp_shape
        cur_width = self.features['in_channel']
        for node in self.main_path:
            # print(node.name)
            # print(residual_tensor_shape)
            # print(cur_width)
            # update each node in main path
            if isinstance(node, ConvOp):
                node.update({"in_shape": main_tensor_shape,
                             "in_channel": cur_width,
                             "out_channel": self.features["out_channel"]})
            else:
                node.update({'in_shape': main_tensor_shape,
                            'in_channel': cur_width})
            main_tensor_shape = node.features['out_shape']
            cur_width = node.features['out_channel']

        if self.reduce_type == 'MatMul':
            # handle matmul reduce residual tensor update
            # print(self.reduce_type)
            # print(main_tensor_shape)
            # print(residual_tensor_shape)
            if main_tensor_shape[-1] == residual_tensor_shape[-2]:
                cur_shape = tuple([residual_tensor_shape, main_tensor_shape])
                cur_width = main_tensor_shape[-1]
            elif main_tensor_shape[-2] == residual_tensor_shape[-1]:
                cur_shape = tuple([main_tensor_shape, residual_tensor_shape])
                cur_width = residual_tensor_shape[-1]
            else:
                raise NotImplementedError
            self.last_node.update({"in_channel": cur_width, "in_shape": cur_shape})
            self.features['out_channel'] = self.last_node.features["out_channel"]
            self.features['out_shape'] = self.last_node.features["out_shape"]
            # print(self.features["out_shape"])
            return
            # if len(main_tensor_shape)==2:
            #     cin_main, cout_main = main_tensor_shape
            #     cin_res, cout_res = residual_tensor_shape

            #     if cin_main == cout_res:
            #         m = residual_tensor_shape
            #         n = main_tensor_shape
            #     elif cout_main == cin_res:
            #         m = main_tensor_shape
            #         n = residual_tensor_shape
            #     else:
            #         raise NotImplementedError
            #     out_shape = [m[0], n[1]]
            # elif len(main_tensor_shape)==3:
            #     b, cin_main, cout_main = main_tensor_shape
            #     b, cin_res, cout_res = residual_tensor_shape

            #     if cin_main == cout_res:
            #         m = residual_tensor_shape
            #         n = main_tensor_shape
            #     elif cout_main == cin_res:
            #         m = main_tensor_shape
            #         n = residual_tensor_shape
            #     else:
            #         raise NotImplementedError
            #     out_shape = [b, m[1], n[2]]
            # elif len(main_tensor_shape)==4:
            #     b, hw, cin_main, cout_main = main_tensor_shape
            #     b, hw, cin_res, cout_res = residual_tensor_shape

            #     if cin_main == cout_res:
            #         m = residual_tensor_shape
            #         n = main_tensor_shape
            #     elif cout_main == cin_res:
            #         m = main_tensor_shape
            #         n = residual_tensor_shape
            #     else:
            #         raise NotImplementedError
            #     out_shape = [b, hw, m[2], n[3]]
            # else:
            #     raise NotImplementedError
            # self.last_node.features['inp_shape1'] = m
            # self.last_node.features['inp_shape2'] = n
        elif self.reduce_type == 'Mul' and len(inp_shape)==4:
            if main_tensor_shape!=residual_tensor_shape:
                assert main_tensor_shape[:2]==residual_tensor_shape[:2]
                out_shape = (main_tensor_shape[0],
                            main_tensor_shape[1],
                            max(main_tensor_shape[2], residual_tensor_shape[2]),
                            max(main_tensor_shape[3], residual_tensor_shape[3]))
            else:
                assert main_tensor_shape==residual_tensor_shape
                out_shape = self.main_path[-1].features['out_shape']
        elif self.reduce_type == "Concat":
            # by default axis is 1
            out_shape = self.main_path[-1].features["out_shape"]
            out_shape[1] += self.residual_path[-1].features["out_shape"][1]

        else:
            # for node in self.main_path:
            #     print(f"{node.type}: {node.info}")
            # print(main_tensor_shape)
            # print(residual_tensor_shape)
            # print(f"{self.reduce_type}")
            assert tuple(main_tensor_shape)==tuple(residual_tensor_shape)
            out_shape = self.main_path[-1].features['out_shape']

        # self.features['in_channel'] = inp_shape[1] # update in width
        self.features['out_channel'] = cur_width # update out width
        self.features['out_shape'] = out_shape # update out shape
        self.last_node.features["in_shape"] = out_shape
        self.last_node.features["out_shape"] = out_shape
        self.last_node.features["in_channel"] = out_shape[1]
        self.last_node.features["out_channel"] = out_shape[1]

    def build(self, cache=None, block_id=None) -> nn.Module:
        """Build torch module for residual block.

        Build operator one by one in each path.

        Retruns:
          Torch module of residual.
        """
        reduce_type = self.features["reduce_type"]

        op_modules = []
        for op in self.main_path:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        main_path_module = nn.Sequential(*op_modules)
        op_modules = []
        for op in self.residual_path:
            op_module = op.build(cache=cache, block_id=f"{block_id}-{op.id}")
            op_modules.append(op_module)

        residual_path_module = nn.Sequential(*op_modules)

        class ResidualModule(nn.Module):
            def __init__(self, main_path, residual_path, reduce_type):
                super().__init__()
                self.main_path = main_path
                self.residual_path = residual_path
                self.reduce_type = reduce_type

            def forward(self, x):
                y = self.main_path(x)
                x = self.residual_path(x)
                if self.reduce_type == "Add":
                    y = y + x
                elif self.reduce_type == "Mul":
                    y = y * x
                elif self.reduce_type == "MatMul":
                    y = x @ v
                return y

        self.module = ResidualModule(main_path_module, residual_path_module,
                                     reduce_type)
        return self.module

    def bind_weight(self):
        for op in self.main_path:
            op.bind_weight()
        for op in self.residual_path:
            op.bind_weight()

    def count_flops_params(self):
        params = 0
        flops = 0
        for op in self.residual_path:
            op_flops, op_params = op.count_flops_params()
            flops += op_flops
            params += op_params
        for op in self.main_path:
            op_flops, op_params = op.count_flops_params()
            flops += op_flops
            params += op_params
        return flops, params

    def get_ops(self):
        op_lst = []
        paths = [self.main_path, self.residual_path]
        for path in paths:
            path_op_lst = []
            for op in path:
                if not isinstance(op, SingleOp):
                    ops = op.get_ops()
                    for subop in ops:
                        subop.id = str(op.id)+'-'+str(subop.id)
                        # print(subop.id)
                        path_op_lst.append(subop)
                    # block will not fuse
                    for op in path_op_lst:
                        op_lst.append(op)
                    path_op_lst = []
                    # print(f"Clear lst due to meet {op}")
                else:
                    path_op_lst.append(op)
                    for fuse_rule in globvar.fusion_rule:
                        # print(path_op_lst)
                        num_ops = len(fuse_rule.split('&'))
                        # print(f"# of ops to match: {num_ops}")
                        for i in range(len(path_op_lst)):
                            for j in range(i, len(path_op_lst)):
                                if len(path_op_lst)-j < num_ops:
                                    # print("Length not enough")
                                    break
                                else:
                                    pattern = '&'.join([path_op_lst[k].type for k in range(j, j+num_ops)])
                                    # print(f"Matching {fuse_rule}")
                                    # print(pattern)
                                    if pattern == fuse_rule:
                                        fused_op = FuseOp(path_op_lst[j:j+num_ops])
                                        # print(f"Fuse {fused_op}")
                                        fused_op.id = str(path_op_lst[j].id)
                                        path_op_lst = path_op_lst[:j] + [fused_op] + path_op_lst[j+num_ops:]
                                        j += num_ops - 1
            if len(path_op_lst) > 0:
                for op in path_op_lst:
                    op_lst.append(op)
        op_lst.append(self.last_node)
        # print(f"Residual ops: {op_lst}")
        return op_lst

    def update_grad(self):
        paths = [self.main_path, self.residual_path]
        for path in paths:
            for op in path:
                op.update_grad()

    @property
    def info_dict(self):
        ret_dict = super().info_dict

        ret_dict['main_path'] = []
        ret_dict['residual_path'] = []

        for op in self.main_path:
            ret_dict['main_path'].append(op.info_dict)
        for op in self.residual_path:
            ret_dict['residual_path'].append(op.info_dict)

        ret_dict['reduce_type'] = self.reduce_type
        return ret_dict

    @property
    def info(self):
        info_list = []
        for op in self.main_path:
            info_list.append(op.info)
        for op in self.residual_path:
            info_list.append(op.info)
        info_tuple = tuple(info_list)
        return info_tuple
