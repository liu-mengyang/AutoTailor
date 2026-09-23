# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .ncnn_local_profiler import NcnnLocalProfiler
from .ncnn_local_backend import NcnnLocalBackend
from ..interface import BaseParser
from ..utils import Latency, ProfiledResults


class NcnnCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content):
        self.total_latency = self._parse_total_latency(content)
        return self

    def _parse_total_latency(self, content):
        total_latency_regex = r'time_avg ([\d.\+e-]+)'
        total_var_regex = r'time_var ([\d.\+e-]+)'
        total_latency = Latency()
        match_latency = re.search(total_latency_regex, content, re.MULTILINE)
        match_var = re.search(total_var_regex, content, re.MULTILINE)
        if match_latency and match_var:
            total_latency = Latency(float(match_latency[1]), float(match_var[1]))
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


class NcnnCPUProfiler(NcnnLocalProfiler):
    use_gpu = False


class NcnnLocalCPUBackend(NcnnLocalBackend):
    parser_class = NcnnCPULatencyParser
    profiler_class = NcnnCPUProfiler
