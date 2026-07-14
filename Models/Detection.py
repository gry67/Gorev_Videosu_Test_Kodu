
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    color: str
    center: Tuple[int, int]
    area: float

    # Normal yatay kutu, yazı konumu ve yedek merkez için kullanılır.
    bbox: Tuple[int, int, int, int]

    # Döndürülmüş dikdörtgenin dört köşe noktası
    rotated_box: np.ndarray = field(repr=False)

    aspect_ratio: float
    rectangularity: float
    corners: int
    angle: float
    confidence: float

    contour: np.ndarray = field(repr=False)

    verified: bool = False
    consecutive_frames: int = 1