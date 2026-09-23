from pprint import pprint
import warnings
import subprocess

from .nn_modules.torch_networks import blocks as torch_blocks
from .nn_modules.tf_networks import blocks as tf_blocks
import logging
logger = logging.getLogger("autotailor")

__BUILTIN_KERNELS__ = {
    # builtin name: [kernel class name, kernel sampler class name]
    "Conv": ["ConvBlock"],
    # "conv-bn-relu": ["ConvBnRelu", "ConvSampler"],
    # "conv-bn-relu6": ["ConvBnRelu6", "ConvSampler"],
    # "conv-bn": ["ConvBn", "ConvSampler"],
    # "conv-relu": ["ConvRelu", "ConvSampler"],
    # "conv-relu6": ["ConvRelu6", "ConvSampler"],
    # "conv-hswish": ["ConvHswish", "ConvSampler"],
    # "conv-bn-hswish": ["ConvBnHswish", "ConvSampler"],
    # # dwconv
    "DepthConv": ["DepthConvBlock"],
    # "dwconv-bn": ["DwConvBn", "DwConvSampler"],
    # "dwconv-relu": ["DwConvRelu", "DwConvSampler"],
    # "dwconv-relu6": ["DwConvRelu6", "DwConvSampler"],
    # "dwconv-bn-relu": ["DwConvBnRelu", "DwConvSampler"],
    # "dwconv-bn-relu6": ["DwConvBnRelu6", "DwConvSampler"],
    # "dwconv-bn-hswish": ["DwConvBnHswish", "DwConvSampler"],
    # # others
    # "gemm": ["GEMMBlock", "GEMMSampler"],
    "MatMul": ["MatMulBlock"],
    "MaxPool": ["MaxPoolBlock"],
    "AveragePool": ["AvgPoolBlock"],
    "GlobalAveragePool": ["GlobalAvgPoolBlock"],
    "ReduceMean": ["MeanBlock"],
    "Linear": ["FCBlock"],
    "LinearMatMul": ["FCBlock"],
    "Concat": ["ConcatBlock"],
    "Gather": ["GatherBlock"],
    # "split": ["SplitBlock", "CinEvenSampler"],
    # "channelshuffle": ["ChannelShuffle", "CinEvenSampler"],
    # "se": ["SEBlock", "CinEvenSampler"],
    "Reshape": ["ReshapeBlock"],
    "Transpose": ["TransposeBlock"],
    "Flatten": ["FlattenBlock"],
    # "bnrelu": ["BnRelu", "HwCinSampler"],
    "HardSwish": ["HswishBlock"],
    "HardSigmoid": ["HsigmoidBlock"],
    "Relu": ["ReluBlock"],
    "Gelu": ["GeluBlock"],
    # "addrelu": ["AddRelu", "HwCinSampler"],
    "Pad": ["PadBlock"],
    "Clip": ["ClipBlock"],
    "Softmax": ["SoftmaxBlock"],
    "Add": ["AddBlock"],
    "BiasAdd": ["AddBlock"],
    "ScaleMul": ["ScaleMulBlock"],
    "Mul": ["MulBlock"],
    "BatchNormalization": ["BNBlock"],
    "LayerNormalization": ["LNBlock"],
}


class KernelGenerator:
    def generate_model_for_kernel(self,
                                  kernel_type,
                                  config,
                                  save_path,
                                  batch_size=1,
                                  framework='torch',
                                  disk_check=False,
                                  save_model=False):
        """
        Generate and save the single kernel model.

        Args:
            kernel_type (str): the kernel type.
            config (dict): kernel description.
            save_path (str): model saving path.
            batch_size (int, optional): the batchsize of the input tensor.
                Defaults to 1.

        Raises:
            ValueError: no support kernel type

        Returns:
            nn.Module: instantiated model
            list: input tensor shape
            dict: the config dictionary of the kernel
        """
        if disk_check:
            # check disk
            usage_ratio = self.check_disk()
            if usage_ratio >= 95:
                logger.error('Disk free space is not enough.')
                raise OSError('Disk free space is not enough.')
            elif usage_ratio >= 80:
                logger.warn(f"Disk is in high usage ratio: {usage_ratio}")
                warnings.warn(f"Disk is in high usage ratio: {usage_ratio}")
        
        # get kernel class information
        if kernel_type in __BUILTIN_KERNELS__:
            kernel_name = __BUILTIN_KERNELS__[kernel_type][0]
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}. Please register the kernel first.")

        # get kernel class and create kernel instance by needed_config
        if framework == 'torch':
            kernel_class = getattr(torch_blocks, kernel_name)(config)
        # if framework == 'tf':
        #     kernel_class = getattr(tf_blocks, kernel_name)(config, batch_size)
        input_shape = kernel_class.input_shape
        model = kernel_class.get_model()

        # save model file to savepath
        if save_model:
            kernel_class.save_model(save_path)
            logger.info(f"{kernel_type} model is generated and saved to {save_path}.")

        return model, input_shape
        
    
    def list_kernels(self):
        """
        List supported kernels.

        Returns:
            list: _description_
        """
        kernels = list(__BUILTIN_KERNELS__.keys())
        pprint(kernels)
        return kernels

    def check_disk(self):
        ps = subprocess.Popen(('df', '-h'), stdout=subprocess.PIPE)
        output = subprocess.check_output(('grep', '/dev/nvme'), stdin=ps.stdout).decode('utf-8')
        ps.wait()
        
        return int(output.split(' ')[-2].split('%')[0])
        