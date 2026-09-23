# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .ncnn_local_profiler import NcnnLocalProfiler
from .ncnn_local_backend import NcnnLocalBackend
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

    def _parse_total_latency(self, content, loop_count=100, warm_up=10):
        regex_us = r'([\d.\+e-]+)us'
        regex_ms = r'([\d.\+e-]+)ms'
        regex_e2e = r'time_avg ([\d.\+e-]+)'
        total_latency = []
        sum_latency = 0.0
        index = 0
        match_tag = False
        for line in content.splitlines():
            if "Segmentation fault" in line:
                return "Segmentation fault"
            match_us = re.search(regex_us, line.strip())
            match_ms = re.search(regex_ms, line.strip())
            match_e2e = re.search(regex_e2e, line.strip())
            # if "MemoryData" in line:
            #     continue
            if match_us:
                index += 1
                match_tag = True
                total_latency.append(float(match_us[1])/1000.0)
            if match_ms:
                index += 1
                match_tag = True
                total_latency.append(float(match_ms[1]))
            if match_e2e and (match_tag == False):
                return Latency(avg=float(match_e2e[1]), std=0)
        print(index)
        print(str(loop_count + warm_up))
        warm_up_kernel = int(index/(loop_count + warm_up)) * warm_up
        for i in range(warm_up_kernel, len(total_latency)):
            sum_latency += total_latency[i]
            
        avg = float(sum_latency/loop_count)
        print(f"avg is: {avg}")
        print(f"round {loop_count}")
        return Latency(avg=avg, std=0)

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


class NcnnGPUProfiler(NcnnLocalProfiler):
    use_gpu = True


class NcnnLocalGPUBackend(NcnnLocalBackend):
    parser_class = NcnnGPULatencyParser
    profiler_class = NcnnGPUProfiler
