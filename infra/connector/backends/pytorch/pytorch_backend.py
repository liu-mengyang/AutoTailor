# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import subprocess
import shutil
import logging
from ..interface import BaseBackend

logging = logging.getLogger("nn-Meter")


class pytorchBackend(BaseBackend):
    parser_class = None
    profiler_class = None

    def update_configs(self):
        """update the config parameters for pytorch platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
            'benchmark_fp32_model_path': self.configs['BENCHMARK_FP32_MODEL_PATH'],
            'serial': self.configs['DEVICE_SERIAL'],
            # 'dst_kernel_path': self.configs['KERNEL_PATH']
        })

