from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from config import *
import numpy as np
from Models.Detection import *
from Models.TrackState import *
from DetectionTracker import *
from live_test import *


class DetectionTracker:
    def __init__(self) -> None:
        self.states: Dict[str, TrackState] = {
            "MAVİ": TrackState(),
            "KIRMIZI": TrackState(),
        }

    def reset(self, color: str) -> None:
        self.states[color] = TrackState()

    def update(
        self,
        color: str,
        detections: List[Detection],
    ) -> Optional[Detection]:
        state = self.states[color]

        # ---------------------------------------------------------------------
        # Bu karede tespit yoksa
        # ---------------------------------------------------------------------

        if not detections:
            state.missed_frames += 1

            if (
                state.missed_frames
                > TRACK_MAX_MISSED_FRAMES
            ):
                self.reset(color)

            return None

        selected_detection: Optional[
            Detection
        ] = None

        # ---------------------------------------------------------------------
        # Önceki hedef merkezine yakın adayı tercih et
        # ---------------------------------------------------------------------

        if state.last_center is not None:
            nearby_candidates = []

            for detection in detections:
                distance = euclidean_distance(
                    state.last_center,
                    detection.center,
                )

                if distance <= TRACK_MAX_DISTANCE_PX:
                    nearby_candidates.append(
                        (
                            distance,
                            -detection.confidence,
                            -detection.area,
                            detection,
                        )
                    )

            if nearby_candidates:
                nearby_candidates.sort(
                    key=lambda item: (
                        item[0],
                        item[1],
                        item[2],
                    )
                )

                selected_detection = (
                    nearby_candidates[0][3]
                )

        # Yakın aday bulunamazsa en güçlü adayı seç
        if selected_detection is None:
            selected_detection = detections[0]

        # ---------------------------------------------------------------------
        # Ardışık kare sayısı
        # ---------------------------------------------------------------------

        if state.last_center is None:
            state.consecutive_frames = 1

        else:
            distance = euclidean_distance(
                state.last_center,
                selected_detection.center,
            )

            if distance <= TRACK_MAX_DISTANCE_PX:
                state.consecutive_frames += 1
            else:
                state.consecutive_frames = 1

        state.last_center = (
            selected_detection.center
        )

        state.last_detection = (
            selected_detection
        )

        state.missed_frames = 0

        selected_detection.consecutive_frames = (
            state.consecutive_frames
        )

        selected_detection.verified = (
            state.consecutive_frames
            >= REQUIRED_CONSECUTIVE_FRAMES
        )

        return selected_detection
