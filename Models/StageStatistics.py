@dataclass
class StageStatistics:
    total_frames: int = 0

    blue_verified_frames: int = 0
    red_verified_frames: int = 0

    blue_events: int = 0
    red_events: int = 0

    previous_blue_verified: bool = False
    previous_red_verified: bool = False