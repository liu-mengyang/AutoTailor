# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import json
from ..interface import BaseProfiler


class NcnnLocalProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, default_benchamrk_path, graph_path='', dst_graph_path='', num_threads=1, num_runs=50, warm_ups=10):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        """
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._benchmark_model_path = default_benchamrk_path
        self._loop_count = 100 
        self._num_threads = 1
        self._powersave = 0
        self._gpu_device = 0
        self._cooling_down = 1  
        

