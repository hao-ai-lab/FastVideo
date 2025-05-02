from dataclasses import dataclass, field
from typing import List


@dataclass
class CacheConfig:
    cache_type: str = "none"
    enable_teacache: bool = False
    teacache_thresh: float = 0.0
    use_ret_steps: bool = False
    ret_steps: int = 0
    num_steps: int = 0
    cutoff_steps: int = 0
    coefficients: List[float] = field(default_factory=list)
    

