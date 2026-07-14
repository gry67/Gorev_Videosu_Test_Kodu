
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

@dataclass
class TrackState:
    last_center: Optional[Tuple[int, int]] = None
    consecutive_frames: int = 0
    missed_frames: int = 0
    last_detection: Optional[Detection] = None