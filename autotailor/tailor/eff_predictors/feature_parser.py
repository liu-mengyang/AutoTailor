import ast

class FeatureParser:
    def __init__(self):
        self.support_op_type = ["Conv", "DepthConv", "MatMul", "LinearMatMul"]
        self.num_features = {
            "Conv": 7,
            "DepthConv": 6,
            "MatMul": 5,
            "LinearMatMul": 4
        }
    
    def parse(self, op_type, feature):
        assert op_type in self.support_op_type
        if isinstance(feature, str):
            obj = ast.literal_eval(feature)
            feature = list(obj)
        if op_type == "Conv":
            # kernel_size, stride, group, has_bias, in_shape, out_shape, in_channel, out_channel
            # -> HW, CIN, COUT, KERNEL_SIZE, STRIDES
            ks, stride, group, has_bias, in_shape, out_shape, in_channel, out_channel = feature
            if len(in_shape) == 4:
                HW = in_shape[-1]
            else:
                raise NotImplementedError
            CIN = in_channel
            COUT = out_channel
            KERNEL_SIZE = ks
            STRIDE = stride
            PARAMS = COUT * KERNEL_SIZE * KERNEL_SIZE * CIN
            FLOPS = 2 * HW / STRIDE * HW / STRIDE * PARAMS
            parsed_feature = [HW, CIN, COUT, KERNEL_SIZE, STRIDE, PARAMS/1e6, FLOPS/2e6]
        elif op_type == "DepthConv":
            # kernel_size, stride, group, has_bias, in_shape, out_shape, in_channel, out_channel
            # -> HW, CHANNEL_SIZE, KERNEL_SIZE, STRIDES
            ks, stride, group, has_bias, in_shape, out_shape, in_channel, out_channel = feature
            if len(in_shape) == 4:
                HW = in_shape[-1]
            else:
                raise NotImplementedError
            CHANNEL_SIZE = in_channel
            KERNEL_SIZE = ks
            STRIDE = stride
            PARAMS = CHANNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
            FLOPS = 2 * HW / STRIDE * HW / STRIDE * PARAMS
            parsed_feature = [HW, CHANNEL_SIZE, KERNEL_SIZE, STRIDE, PARAMS/1e6, FLOPS/2e6]
        elif op_type == "MatMul":
            # input_shape (with 2), output_shape, in_channel, out_channel
            in_shape, out_shape, in_channel, out_channel = feature
            C = in_shape[0][1]
            M = in_shape[0][2]
            N = in_shape[0][3]
            K = in_shape[1][3]
            # PARAMS = CHANNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
            FLOPS = 2 * C * M * N * K
            parsed_feature = [C, M, N, K, FLOPS/2e6]
        elif op_type == "LinearMatMul":
            in_shape, out_shape, in_channel, out_channel = feature
            M = in_shape[1][2]
            N = in_shape[1][1]
            K = in_shape[2][3]
            PARAMS = N * K
            FLOPS = 2 * C * M * N * K
            parsed_feature = [C, M, N, K, FLOPS/2e6]
        else:
            raise NotImplementedError
        
        return parsed_feature