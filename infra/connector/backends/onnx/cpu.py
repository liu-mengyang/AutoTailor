# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
import numpy as np

from .onnx_profiler import OnnxProfiler
from .onnx_backend import OnnxBackend
from ..interface import BaseParser
from ..utils import Latency, ProfiledResults


class OnnxCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content):
        self.total_latency = self._parse_total_latency(content)
        return self

    def _parse_total_latency(self, content):
        # Average inference time cost: 9.19386 ms
        # Min Latency: 0.00904651 s
        # Max Latency: 0.0100804 s
        total_latency_regex = r'Average inference time cost: ([\d.]+|[\d.]+e[\d-]+) (ms|s)'
        total_var_regex_min = r'Min Latency: ([\d.]+|[\d.]+e[\d-]+) (s|ms)'
        total_var_regex_max = r'Max Latency: ([\d.]+|[\d.]+e[\d-]+) (s|ms)'
        total_latency = Latency()
        match_latency = re.search(total_latency_regex, content, re.MULTILINE)
        match_var_min = re.search(total_var_regex_min, content, re.MULTILINE)
        match_var_max = re.search(total_var_regex_max, content, re.MULTILINE)
        if match_latency and match_var_min and match_var_max:
            if match_latency[2]=='s':
                avg = float(match_latency[1]) * 1000
            else:
                avg = float(match_latency[1])
            if match_var_min[2]=='s':
                std_min = float(match_var_min[1]) * 1000
            else:
                std_min = float(match_var_min[1])
            if match_var_max[2]=='s':
                std_max = float(match_var_max[1]) * 1000
            else:
                std_max = float(match_var_max[1])
            std = std_max - std_min
            total_latency = Latency(avg, std)
        else:
            raise ValueError('Parsing error when profiling')
        return total_latency
    
    @property
    def latency(self):
        return self.total_latency

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class OnnxCPUProfiler(OnnxProfiler):
    use_gpu = False


class OnnxCPUBackend(OnnxBackend):
    parser_class = OnnxCPULatencyParser
    profiler_class = OnnxCPUProfiler