# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from .ncnn_profiler import NcnnProfiler
from .ncnn_backend import NcnnBackend
from ..interface import BaseParser
from ..utils import Latency, ProfiledResults
from loguru import logger


class NcnnCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content, mode=None):
        self.total_latency = self._parse_total_latency(content, mode=mode)
        return self


    def _parse_total_latency(self, content, loop_count=100, warm_up=8, warm_tag=40, mode=None):
        regex_us = r'([\d.\+e-]+)us'
        regex_ms = r'([\d.\+e-]+)ms'
        regex_e2e = r'time_avg ([\d.\+e-]+)'
        total_latency = []
        sum_latency = 0.0
        index = 0
        begin_tag = False
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
            if match_e2e and mode == "avg":
                return Latency(avg=float(match_e2e[1]), std=0)
            
        print(index)
        print(str(loop_count + warm_up))
        warm_up_kernel = int(index/(loop_count + warm_up)) * warm_tag
        for i in range(warm_up_kernel, len(total_latency)):
            sum_latency += total_latency[i]
            
        avg = float(sum_latency/(loop_count+warm_up-warm_tag))
        print(f"avg is: {avg}")
        print(f"round {loop_count}")
        return Latency(avg=avg, std=0)
    
    @property
    def latency(self):
        return self.total_latency

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class NcnnCPUProfiler(NcnnProfiler):
    use_gpu = False


class NcnnCPUBackend(NcnnBackend):
    parser_class = NcnnCPULatencyParser
    profiler_class = NcnnCPUProfiler
