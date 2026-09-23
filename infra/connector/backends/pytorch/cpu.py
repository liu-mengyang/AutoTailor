# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
import numpy as np

from .pytorch_profiler import pytorchProfiler
from .pytorch_backend import pytorchBackend
from ..interface import BaseParser
from ..utils import Latency, ProfiledResults


class pytorchCPULatencyParser(BaseParser):
    def __init__(self):
        self.nodes = []
        self.total_latency = Latency()

    def parse(self, content):
        self.total_latency = self._parse_total_latency(content)
        return self

    def _parse_total_latency(self, content):
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73590"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73680"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73931"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73790"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73485"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73793"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73620"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73600"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73298"}
        # PyTorchObserver {"type": "NET", "unit": "us", "metric": "latency", "value": "73770"}
        # Main run finished. Microseconds per iter: 73657.6. Iters per second: 13.5763
        values = [int(match.group(1)) for match in re.finditer(r'"value": "(\d+)"', content)]
        values_in_m_seconds = np.array(values) / 1000.0

        # 输出最大值、最小值和平均值
        max_value = np.max(values_in_m_seconds)
        min_value = np.min(values_in_m_seconds)
        avg = np.mean(values_in_m_seconds)
        std = max_value - min_value
        total_latency = Latency()
        total_latency = Latency(avg, std)

        return total_latency
    
    @property
    def latency(self):
        return self.total_latency

    @property
    def results(self):
        results = ProfiledResults({'latency': self.latency})
        return results


class pytorchCPUProfiler(pytorchProfiler):
    use_gpu = False


class pytorchCPUBackend(pytorchBackend):
    parser_class = pytorchCPULatencyParser
    profiler_class = pytorchCPUProfiler