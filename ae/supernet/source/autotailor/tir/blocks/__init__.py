from .single_op import *
from .seresidual import *
from .bottleneck import *
from .residual import *
from .convresidual import *
from .bottleneckresidual import *
from .mb import *
from .cnresidual import *
from .attention import *
from .qkv import *
from .ffn import *
from .transformer import *
from .fuse_op import *

DYNAMIC_BLOCKS = ["ConvResidualBlock", "BottleneckResidualBlock", "MBBlock", "BottleneckBlock", "CNResidualBlock"]