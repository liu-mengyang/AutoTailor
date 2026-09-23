# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .ncnn_profiler import NcnnProfiler
from .ncnn_backend import NcnnBackend
from ..interface import BaseParser
from ..utils import Latency, ProfiledResults


class NcnnGPULatencyParser(BaseParser):
    def __init__(self):
        self.kernels = []
        self.realtime = 0
        self.kernel_sum = 0
        self.block_name = ''
        self.raw_content = ''
        self.before_fused_graph = ''
        self.after_fused_graph = ''

    def parse(self, content):
        self.total_latency = self._parse_total_latency(content)

        return self

    @staticmethod
    def resolve_name(name):
        name = name.split(' ')
        if 'linked' in name:
            ops = []
            name = [x for x in name if x != ':' and x != 'linked']
            for i in range(0, len(name), 2):
                ops.append(name[i])
            return ops
        else:
            return [name[0]]

    @property
    def latency(self):
        """
        On GPU, we currently decide to use kernel_sum instead of realtime (block) as latency
        """
        return self.total_latency

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

    def _parse_error(self, content):
        error_regex = r'ERROR: (.*)'
        errors = []

        for line in content.splitlines():
            match = re.search(error_regex, line)
            if match:
                errors.append(match[1])

        return errors

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class NcnnGPUProfiler(NcnnProfiler):
    use_gpu = True


class NcnnGPUBackend(NcnnBackend):
    parser_class = NcnnGPULatencyParser
    profiler_class = NcnnGPUProfiler
