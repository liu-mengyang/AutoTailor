# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
from ..interface import BaseProfiler


class pytorchProfiler(BaseProfiler):
    use_gpu = None

    def __init__(self, benchmark_fp32_model_path, graph_path='', dst_graph_path='', serial='', max_parallels_nums=1, num_runs=50, use_nchw_layout=False):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on android device
        kernel_path: dest kernel output file. path on android device
        benchmark_model_path: path to benchmark_model on android device
        max_parallels_nums: max parallel tasks number during inference, default to 1
        """
        self._serial = serial
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._benchmark_model_path = benchmark_fp32_model_path
        self._max_parallels_nums = max_parallels_nums
        self._num_runs = num_runs
        self._use_nchw_layout = use_nchw_layout


