# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import shutil
import logging
import subprocess
from ..interface import BaseBackend
logging = logging.getLogger("nn-Meter")


class NcnnBackend(BaseBackend):
    parser_class = None
    profiler_class = None
    benchmark_fp32_model_path = "benchncnn_fp32"
    benchmark_fp16_model_path = "benchncnn_fp16"
    benchmark_int8_model_path = "benchncnn_int8"

    def update_configs(self):
        """update the config parameters for TFLite platform
        """
        super().update_configs()
        self.profiler_kwargs.update({
            'dst_graph_path': self.configs['REMOTE_MODEL_DIR'],
            'default_benchamrk_path': self.configs['default_benchmark_path'],
            'serial': self.configs['DEVICE_SERIAL'],
            'dst_kernel_path': self.configs['KERNEL_PATH'],
        })


    def test_connection(self):
        """check the status of backend interface connection, ideally including open/close/check_healthy...
        """
        from ppadb.client import Client as AdbClient
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.configs['DEVICE_SERIAL']:
            device = client.device(self.configs['DEVICE_SERIAL'])
        else:
            device = client.devices()[0]
        logging.keyinfo(device.shell("echo hello backend !"))
