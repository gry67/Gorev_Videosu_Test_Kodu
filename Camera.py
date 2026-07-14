import os
import time
from typing import Optional
import cv2
import numpy as np
from config import *
import numpy as np
from Models.Detection import *
from DetectionTracker import *
from live_test import *

class Camera:
    def __init__(self) -> None:
        self.picamera = None
        self.capture = None
        self.use_picamera = False

    def start(self) -> bool:
        # Raspberry Pi kamerasını dene
        
        if HAS_PICAMERA2:
            try:
                self.picamera = Picamera2()

                camera_config = (
                    self.picamera.create_preview_configuration(
                        main={
                            "size": (
                                CAMERA_WIDTH,
                                CAMERA_HEIGHT,
                            ),
                            "format": "RGB888",
                        }
                    )
                )

                self.picamera.configure(
                    camera_config
                )

                self.picamera.start()

                self.use_picamera = True

                time.sleep(1.0)

                print(
                    "  ✓ Raspberry Pi kamera "
                    f"başlatıldı — "
                    f"{CAMERA_WIDTH}x"
                    f"{CAMERA_HEIGHT}"
                )

                return True

            except Exception as error:
                print(
                    "  Picamera2 kullanılamadı: "
                    f"{error}"
                )

                self.picamera = None
                self.use_picamera = False

        # Windows / USB kamera
        if os.name == "nt":
            self.capture = cv2.VideoCapture(
                CAMERA_INDEX,
                cv2.CAP_DSHOW,
            )
        else:
            self.capture = cv2.VideoCapture(
                CAMERA_INDEX
            )

        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(
                CAMERA_INDEX
            )

        if not self.capture.isOpened():
            print("  ✗ Kamera açılamadı.")
            return False

        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        self.capture.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS,
        )

        self.use_picamera = False

        print(
            f"  ✓ USB kamera başlatıldı — "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )

        return True

    def read(self) -> Optional[np.ndarray]:
        if self.use_picamera:
            if self.picamera is None:
                return None

            rgb_frame = (
                self.picamera.capture_array()
            )

            return cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR,
            )

        if self.capture is None:
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def stop(self) -> None:
        if self.picamera is not None:
            try:
                self.picamera.stop()
                self.picamera.close()
            except Exception:
                pass

            self.picamera = None

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print("  Kamera kapatıldı.")