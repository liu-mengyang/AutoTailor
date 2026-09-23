
from typing import Optional

import torch

GiB_bytes = 1 << 30


def get_current_memory_usage(device: Optional[torch.types.Device] = None
                             ) -> float:
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.max_memory_allocated(device)