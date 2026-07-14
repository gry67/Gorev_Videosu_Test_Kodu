# -*- coding: utf-8 -*-
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

import config
from logger_config import setup_logger

logger = setup_logger("vision")

BLUE_HSV_LOWER = np.array([95, 80, 50], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
RED_HSV_LOWER_1 = np.array([0, 140, 70], dtype=np.uint8)
RED_HSV_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_HSV_LOWER_2 = np.array([170, 140, 70], dtype=np.uint8)
RED_HSV_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)

MIN_CONTOUR_AREA_BLUE = 800
MAX_CONTOUR_AREA_BLUE = 190000
MIN_CONTOUR_AREA_RED = 1200
MAX_CONTOUR_AREA_RED = 190000
BLUE_ASPECT_RATIO_MIN = 1.00
BLUE_ASPECT_RATIO_MAX = 1.70
RED_ASPECT_RATIO_MIN = 1.00
RED_ASPECT_RATIO_MAX = 1.70
BLUE_RECTANGULARITY_MIN = 0.60
RED_RECTANGULARITY_MIN = 0.65
BLUE_MIN_CORNERS = 4
BLUE_MAX_CORNERS = 6
RED_MIN_CORNERS = 4
RED_MAX_CORNERS = 6
MIN_CONFIDENCE_BLUE = 70.0
MIN_CONFIDENCE_RED = 72.0
SHAPE_APPROX_EPSILON = 0.04


@dataclass
class DetectionResult:
    color: str
    center_pixel: Tuple[int, int]
    contour: np.ndarray = field(repr=False)
    area: float = 0.0
    bounding_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    rotated_box: Optional[np.ndarray] = field(default=None, repr=False)
    aspect_ratio: float = 0.0
    rectangularity: float = 0.0
    confidence: float = 0.0
    corners: int = 0
    angle: float = 0.0


class VisionProcessor:
    def __init__(self, camera_index: Optional[int] = None, force_usb: bool = False):
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.cap = None
        self.picam = None
        self.use_picamera2 = False
        self.force_usb = force_usb
        self.is_running = False
        if config.SAVE_DEBUG_IMAGES:
            os.makedirs(config.DEBUG_IMAGE_DIR, exist_ok=True)
        logger.info("VisionProcessor başlatıldı - minAreaRect etkin")

    def start_camera(self) -> bool:
        if HAS_PICAMERA2 and not self.force_usb:
            try:
                self.picam = Picamera2()
                cam_config = self.picam.create_preview_configuration(
                    main={
                        "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                        "format": "RGB888",
                    }
                )
                self.picam.configure(cam_config)
                self.picam.start()
                self.use_picamera2 = True
                self.is_running = True
                time.sleep(1.0)
                logger.info("RPi Camera başlatıldı")
                return True
            except Exception as error:
                logger.warning(f"Picamera2 başlatılamadı: {error}")
                self.picam = None
                self.use_picamera2 = False

        try:
            if os.name == "nt":
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error("USB kamera açılamadı")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self.use_picamera2 = False
            self.is_running = True
            logger.info("USB kamera başlatıldı")
            return True
        except Exception as error:
            logger.error(f"Kamera başlatma hatası: {error}")
            return False

    def stop_camera(self) -> None:
        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass
            self.picam = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Kamera durduruldu")

    def capture_frame(self) -> Optional[np.ndarray]:
        try:
            if self.use_picamera2 and self.picam is not None:
                rgb_frame = self.picam.capture_array()
                return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            if self.cap is not None and self.cap.isOpened():
                success, frame = self.cap.read()
                return frame if success else None
            return None
        except Exception as error:
            logger.error(f"Kare yakalama hatası: {error}")
            return None

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (5, 5), 0)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        return mask

    def _create_masks(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(self.preprocess_frame(frame), cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
        blue_mask = self._clean_mask(blue_mask)
        red_1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
        red_2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
        red_mask = self._clean_mask(cv2.bitwise_or(red_1, red_2))
        return blue_mask, red_mask

    def detect_blue_regions(self, frame: np.ndarray) -> List[DetectionResult]:
        blue_mask, _ = self._create_masks(frame)
        return self._find_and_validate_contours(blue_mask, "blue")

    def detect_red_regions(self, frame: np.ndarray) -> List[DetectionResult]:
        _, red_mask = self._create_masks(frame)
        return self._find_and_validate_contours(red_mask, "red")

    def detect_all(self, frame: np.ndarray) -> Dict[str, List[DetectionResult]]:
        blue_mask, red_mask = self._create_masks(frame)
        return {
            "blue": self._find_and_validate_contours(blue_mask, "blue"),
            "red": self._find_and_validate_contours(red_mask, "red"),
        }

    def _find_and_validate_contours(self, mask: np.ndarray, color: str) -> List[DetectionResult]:
        if color == "blue":
            min_area, max_area = MIN_CONTOUR_AREA_BLUE, MAX_CONTOUR_AREA_BLUE
            ar_min, ar_max = BLUE_ASPECT_RATIO_MIN, BLUE_ASPECT_RATIO_MAX
            rect_min = BLUE_RECTANGULARITY_MIN
            corner_min, corner_max = BLUE_MIN_CORNERS, BLUE_MAX_CORNERS
            conf_min = MIN_CONFIDENCE_BLUE
        elif color == "red":
            min_area, max_area = MIN_CONTOUR_AREA_RED, MAX_CONTOUR_AREA_RED
            ar_min, ar_max = RED_ASPECT_RATIO_MIN, RED_ASPECT_RATIO_MAX
            rect_min = RED_RECTANGULARITY_MIN
            corner_min, corner_max = RED_MIN_CORNERS, RED_MAX_CORNERS
            conf_min = MIN_CONFIDENCE_RED
        else:
            raise ValueError(f"Desteklenmeyen renk: {color}")

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[DetectionResult] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            rotated_rect = cv2.minAreaRect(contour)
            _, (rect_w, rect_h), raw_angle = rotated_rect
            if rect_w <= 0 or rect_h <= 0:
                continue

            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            if short_side <= 0:
                continue

            aspect_ratio = long_side / short_side
            if not ar_min <= aspect_ratio <= ar_max:
                continue

            rotated_area = rect_w * rect_h
            rectangularity = area / rotated_area if rotated_area > 0 else 0.0
            if rectangularity < rect_min:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, SHAPE_APPROX_EPSILON * perimeter, True)
            corners = len(approx)
            if not corner_min <= corners <= corner_max:
                continue

            cx, cy = self._calculate_center(contour)
            confidence = self._calculate_confidence(
                area, aspect_ratio, rectangularity, corners, color
            )
            if confidence < conf_min:
                continue

            rotated_box = np.int32(np.round(cv2.boxPoints(rotated_rect)))
            angle = self._normalize_angle(rect_w, rect_h, raw_angle)

            results.append(
                DetectionResult(
                    color=color,
                    center_pixel=(cx, cy),
                    contour=contour,
                    area=area,
                    bounding_rect=(x, y, w, h),
                    rotated_box=rotated_box,
                    aspect_ratio=aspect_ratio,
                    rectangularity=rectangularity,
                    confidence=confidence,
                    corners=corners,
                    angle=angle,
                )
            )

        # Aynı renkten birden fazla geçerli nesne bulunursa yalnızca
        # güven skoru ve alanı en yüksek olan hedefi kullan.
        results.sort(key=lambda d: (d.confidence, d.area), reverse=True)
        return results[:1]

    def _calculate_center(self, contour: np.ndarray) -> Tuple[int, int]:
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            return (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        x, y, w, h = cv2.boundingRect(contour)
        return x + w // 2, y + h // 2

    def _normalize_angle(self, rect_w: float, rect_h: float, raw_angle: float) -> float:
        angle = raw_angle + (90.0 if rect_w < rect_h else 0.0)
        while angle < 0.0:
            angle += 90.0
        while angle >= 90.0:
            angle -= 90.0
        return angle

    def _calculate_confidence(
        self,
        area: float,
        aspect_ratio: float,
        rectangularity: float,
        corners: int,
        color: str,
    ) -> float:
        if color == "blue":
            min_area, ideal_area = MIN_CONTOUR_AREA_BLUE, 5000.0
        else:
            min_area, ideal_area = MIN_CONTOUR_AREA_RED, 6000.0

        if area <= min_area:
            area_score = 0.0
        else:
            normalized = (area - min_area) / max(1.0, ideal_area - min_area)
            area_score = max(0.0, min(1.0, normalized)) * 25.0

        ratio_error = abs(aspect_ratio - 1.0)
        aspect_score = 0.0 if ratio_error >= 0.70 else (1.0 - ratio_error / 0.70) * 30.0
        rect_score = max(0.0, min(1.0, rectangularity)) * 30.0
        corner_score = {4: 15.0, 5: 11.0, 6: 7.0}.get(corners, 0.0)
        return max(0.0, min(100.0, area_score + aspect_score + rect_score + corner_score))

    def get_best_detections(self, frame: np.ndarray) -> Dict[str, Optional[DetectionResult]]:
        detections = self.detect_all(frame)
        return {
            "blue": detections["blue"][0] if detections["blue"] else None,
            "red": detections["red"][0] if detections["red"] else None,
        }

    def draw_detections(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        output = frame.copy()
        colors = {"blue": (255, 120, 0), "red": (0, 0, 255)}
        for color_name, detection_list in detections.items():
            draw_color = colors.get(color_name, (255, 255, 255))
            for det in detection_list:
                cv2.drawContours(output, [det.contour], -1, draw_color, 1)
                if det.rotated_box is not None:
                    cv2.polylines(output, [det.rotated_box], True, draw_color, 2)
                cx, cy = det.center_pixel
                cv2.circle(output, (cx, cy), 6, draw_color, -1)
                x, y, _, _ = det.bounding_rect
                label = (
                    f"{color_name.upper()} C:{det.confidence:.0f}% "
                    f"R:{det.aspect_ratio:.2f} Ac:{det.angle:.0f}"
                )
                cv2.putText(
                    output,
                    label,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    draw_color,
                    2,
                )
        return output

    def save_debug_image(self, frame: np.ndarray, suffix: str = "") -> None:
        if not config.SAVE_DEBUG_IMAGES:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.DEBUG_IMAGE_DIR, f"debug_{timestamp}_{suffix}.jpg")
        cv2.imwrite(path, frame)


if __name__ == "__main__":
    processor = VisionProcessor()
    if not processor.start_camera():
        raise SystemExit("Kamera başlatılamadı")
    try:
        while True:
            frame = processor.capture_frame()
            if frame is None:
                continue
            detections = processor.detect_all(frame)
            output = processor.draw_detections(frame, detections)
            cv2.imshow("Teknofest IHA - Vision", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        processor.stop_camera()
        cv2.destroyAllWindows()
