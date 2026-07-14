# -*- coding: utf-8 -*-
"""
TEKNOFEST İHA - Döndürmeye Dayanıklı Canlı Hedef Tespit Testi

Kullanım:
    python live_test.py --sure 60 --pencere
    python live_test.py --sure 60
    python live_test.py --surekli --pencere

60 saniyelik test:
    0-10 sn   : HEDEF YOK
    10-20 sn  : SADECE MAVİ
    20-30 sn  : HEDEF YOK
    30-40 sn  : SADECE KIRMIZI
    40-50 sn  : HEDEF YOK
    50-60 sn  : MAVİ + KIRMIZI

Durdurma:
    CTRL+C
    Q
    ESC
"""

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# =============================================================================
# RASPBERRY PI KAMERA DESTEĞİ
# =============================================================================

try:
    

    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


# =============================================================================
# KAMERA AYARLARI
# =============================================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_INDEX = 0


# =============================================================================
# HSV RENK ARALIKLARI
# =============================================================================

# Mavi
BLUE_HSV_LOWER = np.array([95, 80, 50], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)

# Kırmızı - iki Hue bölgesi
RED_HSV_LOWER_1 = np.array([0, 140, 70], dtype=np.uint8)
RED_HSV_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)

RED_HSV_LOWER_2 = np.array([170, 140, 70], dtype=np.uint8)
RED_HSV_UPPER_2 = np.array([179, 255, 255], dtype=np.uint8)


# =============================================================================
# MAVİ HEDEF FİLTRELERİ
# =============================================================================

MIN_CONTOUR_AREA_BLUE = 800
MAX_CONTOUR_AREA_BLUE = 190000

# minAreaRect kullanıldığı için oran her zaman uzun/kısa şeklinde >= 1 olur.
BLUE_ASPECT_RATIO_MIN = 1.00
BLUE_ASPECT_RATIO_MAX = 1.70

BLUE_RECTANGULARITY_MIN = 0.60

BLUE_MIN_CORNERS = 4
BLUE_MAX_CORNERS = 6

MIN_CONFIDENCE_BLUE = 70.0


# =============================================================================
# KIRMIZI HEDEF FİLTRELERİ
# =============================================================================

MIN_CONTOUR_AREA_RED = 1200
MAX_CONTOUR_AREA_RED = 190000

RED_ASPECT_RATIO_MIN = 1.00
RED_ASPECT_RATIO_MAX = 1.70

RED_RECTANGULARITY_MIN = 0.65

RED_MIN_CORNERS = 4
RED_MAX_CORNERS = 6

MIN_CONFIDENCE_RED = 72.0


# =============================================================================
# GÖRÜNTÜ İŞLEME AYARLARI
# =============================================================================

GAUSSIAN_BLUR_SIZE = 5

OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 5

OPEN_ITERATIONS = 1
CLOSE_ITERATIONS = 2

SHAPE_APPROX_EPSILON = 0.04


# =============================================================================
# TAKİP AYARLARI
# =============================================================================

REQUIRED_CONSECUTIVE_FRAMES = 5

TRACK_MAX_DISTANCE_PX = 120.0
TRACK_MAX_MISSED_FRAMES = 8


# =============================================================================
# DEBUG AYARLARI
# =============================================================================

DEBUG_DIR = "debug_images"


# =============================================================================
# VERİ SINIFLARI
# =============================================================================










# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def euclidean_distance(
    point_1: Tuple[int, int],
    point_2: Tuple[int, int],
) -> float:
    dx = point_1[0] - point_2[0]
    dy = point_1[1] - point_2[1]

    return float(np.hypot(dx, dy))


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Küçük gürültüleri temizler ve hedef içindeki boşlukları kapatır.
    """

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE),
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE),
    )

    cleaned = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=OPEN_ITERATIONS,
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=CLOSE_ITERATIONS,
    )

    return cleaned


def calculate_center(
    contour: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    """
    Kontur merkezini momentlerle hesaplar.
    Moment başarısız olursa yatay kutunun merkezini kullanır.
    """

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        return center_x, center_y

    x, y, width, height = bbox

    return x + width // 2, y + height // 2


def normalize_rotated_angle(
    rect_width: float,
    rect_height: float,
    raw_angle: float,
) -> float:
    """
    OpenCV minAreaRect açısını daha anlaşılır hâle getirir.

    Sonuç yaklaşık olarak 0-90 derece aralığında tutulur.
    Bu açı hedefin görüntü içindeki dönüşünü gözlemlemek içindir.
    """

    angle = raw_angle

    if rect_width < rect_height:
        angle += 90.0

    while angle < 0:
        angle += 90.0

    while angle >= 90:
        angle -= 90.0

    return angle


def calculate_confidence(
    area: float,
    aspect_ratio: float,
    rectangularity: float,
    corners: int,
    color_name: str,
) -> float:
    """
    Hedef uygunluk puanı hesaplar.

    Bu değer istatistiksel olasılık değildir.

    Puan dağılımı:
        Alan             : 25
        Döndürülmüş oran : 30
        Dikdörtgensellik : 30
        Köşe             : 15
    """

    if color_name == "MAVİ":
        minimum_area = MIN_CONTOUR_AREA_BLUE
        ideal_area = 5000.0
    else:
        minimum_area = MIN_CONTOUR_AREA_RED
        ideal_area = 6000.0

    # -------------------------------------------------------------------------
    # 1. Alan puanı
    # -------------------------------------------------------------------------

    if area <= minimum_area:
        area_score = 0.0
    else:
        normalized_area = (
            (area - minimum_area)
            / max(1.0, ideal_area - minimum_area)
        )

        normalized_area = max(
            0.0,
            min(1.0, normalized_area),
        )

        area_score = normalized_area * 25.0

    # -------------------------------------------------------------------------
    # 2. Döndürülmüş en-boy oranı puanı
    # -------------------------------------------------------------------------
    # Kare için ideal değer 1.0'dır.
    # minAreaRect kullanıldığı için hedef döndüğünde oran fazla bozulmaz.

    ratio_error = abs(aspect_ratio - 1.0)

    if ratio_error >= 0.70:
        aspect_score = 0.0
    else:
        aspect_score = (
            1.0 - ratio_error / 0.70
        ) * 30.0

    # -------------------------------------------------------------------------
    # 3. Dikdörtgensellik puanı
    # -------------------------------------------------------------------------
    # Kontur alanı / döndürülmüş kutu alanı

    normalized_rectangularity = max(
        0.0,
        min(1.0, rectangularity),
    )

    rectangularity_score = (
        normalized_rectangularity * 30.0
    )

    # -------------------------------------------------------------------------
    # 4. Köşe puanı
    # -------------------------------------------------------------------------

    if corners == 4:
        corner_score = 15.0
    elif corners == 5:
        corner_score = 11.0
    elif corners == 6:
        corner_score = 7.0
    else:
        corner_score = 0.0

    total_score = (
        area_score
        + aspect_score
        + rectangularity_score
        + corner_score
    )

    return max(
        0.0,
        min(100.0, total_score),
    )


# =============================================================================
# HEDEF TESPİTİ
# =============================================================================

def find_detections(
    mask: np.ndarray,
    color_name: str,
) -> List[Detection]:
    """
    Maskedeki konturları döndürmeye dayanıklı şekilde değerlendirir.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if color_name == "MAVİ":
        minimum_area = MIN_CONTOUR_AREA_BLUE
        maximum_area = MAX_CONTOUR_AREA_BLUE

        aspect_minimum = BLUE_ASPECT_RATIO_MIN
        aspect_maximum = BLUE_ASPECT_RATIO_MAX

        rectangularity_minimum = (
            BLUE_RECTANGULARITY_MIN
        )

        minimum_corners = BLUE_MIN_CORNERS
        maximum_corners = BLUE_MAX_CORNERS

        minimum_confidence = MIN_CONFIDENCE_BLUE

    elif color_name == "KIRMIZI":
        minimum_area = MIN_CONTOUR_AREA_RED
        maximum_area = MAX_CONTOUR_AREA_RED

        aspect_minimum = RED_ASPECT_RATIO_MIN
        aspect_maximum = RED_ASPECT_RATIO_MAX

        rectangularity_minimum = (
            RED_RECTANGULARITY_MIN
        )

        minimum_corners = RED_MIN_CORNERS
        maximum_corners = RED_MAX_CORNERS

        minimum_confidence = MIN_CONFIDENCE_RED

    else:
        raise ValueError(
            f"Desteklenmeyen renk: {color_name}"
        )

    detections: List[Detection] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        if area > maximum_area:
            continue

        # ---------------------------------------------------------------------
        # Normal yatay kutu
        # ---------------------------------------------------------------------
        # Sadece yazı konumu ve yedek merkez hesabında kullanılır.

        x, y, width, height = cv2.boundingRect(contour)

        if width <= 0 or height <= 0:
            continue

        # ---------------------------------------------------------------------
        # Döndürülmüş minimum alanlı dikdörtgen
        # ---------------------------------------------------------------------

        rotated_rect = cv2.minAreaRect(contour)

        (
            rotated_center,
            rotated_size,
            raw_angle,
        ) = rotated_rect

        rect_width, rect_height = rotated_size

        if rect_width <= 0 or rect_height <= 0:
            continue

        long_side = max(
            rect_width,
            rect_height,
        )

        short_side = min(
            rect_width,
            rect_height,
        )

        if short_side <= 0:
            continue

        # Her zaman >= 1
        aspect_ratio = (
            long_side / short_side
        )

        if aspect_ratio < aspect_minimum:
            continue

        if aspect_ratio > aspect_maximum:
            continue

        rotated_box_area = (
            rect_width * rect_height
        )

        if rotated_box_area <= 0:
            continue

        rectangularity = (
            area / rotated_box_area
        )

        if rectangularity < rectangularity_minimum:
            continue

        # ---------------------------------------------------------------------
        # Köşe sayısı
        # ---------------------------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approximated_contour = cv2.approxPolyDP(
            contour,
            SHAPE_APPROX_EPSILON * perimeter,
            True,
        )

        corners = len(approximated_contour)

        if corners < minimum_corners:
            continue

        if corners > maximum_corners:
            continue

        center = calculate_center(
            contour,
            (x, y, width, height),
        )

        angle = normalize_rotated_angle(
            rect_width,
            rect_height,
            raw_angle,
        )

        confidence = calculate_confidence(
            area=area,
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            corners=corners,
            color_name=color_name,
        )

        if confidence < minimum_confidence:
            continue

        rotated_box = cv2.boxPoints(
            rotated_rect
        )

        rotated_box = np.int32(
            np.round(rotated_box)
        )

        detection = Detection(
            color=color_name,
            center=center,
            area=area,
            bbox=(x, y, width, height),
            rotated_box=rotated_box,
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            corners=corners,
            angle=angle,
            confidence=confidence,
            contour=contour,
        )

        detections.append(detection)

    detections.sort(
        key=lambda detection: (
            detection.confidence,
            detection.area,
        ),
        reverse=True,
    )

    return detections


def detect_all(
    frame: np.ndarray,
) -> Tuple[
    List[Detection],
    List[Detection],
    np.ndarray,
    np.ndarray,
]:
    """
    Bir görüntüde mavi ve kırmızı hedefleri arar.
    """

    blurred = cv2.GaussianBlur(
        frame,
        (
            GAUSSIAN_BLUR_SIZE,
            GAUSSIAN_BLUR_SIZE,
        ),
        0,
    )

    hsv_frame = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV,
    )

    # -------------------------------------------------------------------------
    # Mavi maske
    # -------------------------------------------------------------------------

    blue_mask = cv2.inRange(
        hsv_frame,
        BLUE_HSV_LOWER,
        BLUE_HSV_UPPER,
    )

    blue_mask = clean_mask(
        blue_mask
    )

    # -------------------------------------------------------------------------
    # Kırmızı maske
    # -------------------------------------------------------------------------

    red_mask_1 = cv2.inRange(
        hsv_frame,
        RED_HSV_LOWER_1,
        RED_HSV_UPPER_1,
    )

    red_mask_2 = cv2.inRange(
        hsv_frame,
        RED_HSV_LOWER_2,
        RED_HSV_UPPER_2,
    )

    red_mask = cv2.bitwise_or(
        red_mask_1,
        red_mask_2,
    )

    red_mask = clean_mask(
        red_mask
    )

    blue_detections = find_detections(
        blue_mask,
        "MAVİ",
    )

    red_detections = find_detections(
        red_mask,
        "KIRMIZI",
    )

    return (
        blue_detections,
        red_detections,
        blue_mask,
        red_mask,
    )


# =============================================================================
# TAKİP SİSTEMİ
# =============================================================================



# =============================================================================
# GÖRSEL ÇİZİM
# =============================================================================

def draw_single_detection(
    frame: np.ndarray,
    detection: Detection,
    draw_color: Tuple[int, int, int],
) -> None:
    """
    Döndürülmüş hedef kutusunu ve tespit bilgilerini çizer.
    """

    x, y, width, height = detection.bbox
    center_x, center_y = detection.center

    thickness = (
        3 if detection.verified else 1
    )

    # Gerçek kontur
    cv2.drawContours(
        frame,
        [detection.contour],
        -1,
        draw_color,
        1,
    )

    # Döndürülmüş minimum alanlı kutu
    cv2.polylines(
        frame,
        [detection.rotated_box],
        True,
        draw_color,
        thickness,
    )

    # Merkez
    cv2.circle(
        frame,
        (center_x, center_y),
        6,
        draw_color,
        -1,
    )

    cv2.line(
        frame,
        (center_x - 10, center_y),
        (center_x + 10, center_y),
        (255, 255, 255),
        1,
    )

    cv2.line(
        frame,
        (center_x, center_y - 10),
        (center_x, center_y + 10),
        (255, 255, 255),
        1,
    )

    if detection.verified:
        status_text = "DOGRULANDI"
    else:
        status_text = (
            f"BEKLE "
            f"{detection.consecutive_frames}/"
            f"{REQUIRED_CONSECUTIVE_FRAMES}"
        )

    first_line = (
        f"{detection.color} "
        f"%{detection.confidence:.0f} "
        f"{status_text}"
    )

    second_line = (
        f"A:{detection.area:.0f} "
        f"R:{detection.aspect_ratio:.2f} "
        f"D:{detection.rectangularity:.2f} "
        f"K:{detection.corners} "
        f"Ac:{detection.angle:.0f}"
    )

    text_y_1 = max(
        22,
        y - 25,
    )

    text_y_2 = max(
        42,
        y - 6,
    )

    cv2.putText(
        frame,
        first_line,
        (x, text_y_1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        draw_color,
        2,
    )

    cv2.putText(
        frame,
        second_line,
        (x, text_y_2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        draw_color,
        1,
    )


def detection_status_text(
    detection: Optional[Detection],
) -> str:
    if detection is None:
        return "YOK"

    if detection.verified:
        return "DOGRULANDI"

    return (
        f"{detection.consecutive_frames}/"
        f"{REQUIRED_CONSECUTIVE_FRAMES}"
    )


def draw_results(
    frame: np.ndarray,
    blue_detection: Optional[Detection],
    red_detection: Optional[Detection],
    stage_name: str,
    elapsed: float,
) -> np.ndarray:
    output = frame.copy()

    if blue_detection is not None:
        draw_single_detection(
            output,
            blue_detection,
            (255, 120, 0),
        )

    if red_detection is not None:
        draw_single_detection(
            output,
            red_detection,
            (0, 0, 255),
        )

    blue_status = detection_status_text(
        blue_detection
    )

    red_status = detection_status_text(
        red_detection
    )

    cv2.rectangle(
        output,
        (0, 0),
        (CAMERA_WIDTH, 65),
        (0, 0, 0),
        -1,
    )

    first_info = (
        f"MAVI: {blue_status} | "
        f"KIRMIZI: {red_status}"
    )

    second_info = (
        f"ASAMA: {stage_name} | "
        f"SURE: {elapsed:.1f}s"
    )

    cv2.putText(
        output,
        first_info,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        output,
        second_info,
        (8, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return output


def create_debug_grid(
    frame: np.ndarray,
    annotated: np.ndarray,
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
) -> np.ndarray:
    height, width = frame.shape[:2]

    half_width = width // 2
    half_height = height // 2

    grid = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    original_small = cv2.resize(
        frame,
        (half_width, half_height),
    )

    annotated_small = cv2.resize(
        annotated,
        (half_width, half_height),
    )

    blue_mask_bgr = cv2.cvtColor(
        blue_mask,
        cv2.COLOR_GRAY2BGR,
    )

    red_mask_bgr = cv2.cvtColor(
        red_mask,
        cv2.COLOR_GRAY2BGR,
    )

    blue_mask_small = cv2.resize(
        blue_mask_bgr,
        (half_width, half_height),
    )

    red_mask_small = cv2.resize(
        red_mask_bgr,
        (half_width, half_height),
    )

    grid[
        0:half_height,
        0:half_width,
    ] = original_small

    grid[
        0:half_height,
        half_width:width,
    ] = annotated_small

    grid[
        half_height:height,
        0:half_width,
    ] = blue_mask_small

    grid[
        half_height:height,
        half_width:width,
    ] = red_mask_small

    cv2.putText(
        grid,
        "ORIJINAL",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        grid,
        "TESPIT",
        (half_width + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        grid,
        "MAVI MASKE",
        (8, half_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        grid,
        "KIRMIZI MASKE",
        (half_width + 8, half_height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return grid





# =============================================================================
# TEST AŞAMALARI
# =============================================================================

def get_test_stage(
    elapsed: float,
) -> Tuple[str, str]:
    if elapsed < 10:
        return "HEDEF YOK 1", "empty_1"

    if elapsed < 20:
        return "SADECE MAVI", "blue"

    if elapsed < 30:
        return "HEDEF YOK 2", "empty_2"

    if elapsed < 40:
        return "SADECE KIRMIZI", "red"

    if elapsed < 50:
        return "HEDEF YOK 3", "empty_3"

    return "MAVI + KIRMIZI", "both"


def update_stage_statistics(
    statistics: StageStatistics,
    blue_verified: bool,
    red_verified: bool,
) -> None:
    statistics.total_frames += 1

    if blue_verified:
        statistics.blue_verified_frames += 1

    if red_verified:
        statistics.red_verified_frames += 1

    if (
        blue_verified
        and not statistics.previous_blue_verified
    ):
        statistics.blue_events += 1

    if (
        red_verified
        and not statistics.previous_red_verified
    ):
        statistics.red_events += 1

    statistics.previous_blue_verified = (
        blue_verified
    )

    statistics.previous_red_verified = (
        red_verified
    )


# =============================================================================
# SİNYAL YÖNETİMİ
# =============================================================================

stop_requested = False


def signal_handler(
    _signal_number,
    _frame,
) -> None:
    global stop_requested

    stop_requested = True

    print("\nDurdurma sinyali alındı...")


signal.signal(
    signal.SIGINT,
    signal_handler,
)

signal.signal(
    signal.SIGTERM,
    signal_handler,
)


# =============================================================================
# ANA PROGRAM
# =============================================================================

def main() -> None:
    global stop_requested

    parser = argparse.ArgumentParser(
        description=(
            "Döndürmeye dayanıklı canlı "
            "mavi ve kırmızı hedef testi"
        )
    )

    parser.add_argument(
        "--sure",
        type=int,
        default=60,
        help="Test süresi",
    )

    parser.add_argument(
        "--surekli",
        action="store_true",
        help=(
            "CTRL+C ile durdurulana kadar "
            "çalıştır"
        ),
    )

    parser.add_argument(
        "--kayit-araligi",
        type=int,
        default=30,
        help=(
            "Kaç karede bir debug "
            "görüntüsü kaydedileceği"
        ),
    )

    parser.add_argument(
        "--pencere",
        action="store_true",
        help="Canlı kamera penceresini göster",
    )

    args = parser.parse_args()

    if args.sure <= 0:
        print(
            "HATA: --sure sıfırdan "
            "büyük olmalıdır."
        )
        sys.exit(1)

    if args.kayit_araligi <= 0:
        print(
            "HATA: --kayit-araligi sıfırdan "
            "büyük olmalıdır."
        )
        sys.exit(1)

    os.makedirs(
        DEBUG_DIR,
        exist_ok=True,
    )

    print()
    print(
        "╔════════════════════════════════════════════════════════════╗"
    )
    print(
        "║    DÖNDÜRMEYE DAYANIKLI MAVİ / KIRMIZI HEDEF TESTİ      ║"
    )
    print(
        "╠════════════════════════════════════════════════════════════╣"
    )
    print(
        f"║  Mavi güven eşiği     : "
        f"%{MIN_CONFIDENCE_BLUE:<35.0f}║"
    )
    print(
        f"║  Kırmızı güven eşiği  : "
        f"%{MIN_CONFIDENCE_RED:<35.0f}║"
    )
    print(
        f"║  Mavi minimum alan    : "
        f"{MIN_CONTOUR_AREA_BLUE:<36}║"
    )
    print(
        f"║  Kırmızı minimum alan : "
        f"{MIN_CONTOUR_AREA_RED:<36}║"
    )
    print(
        f"║  Ardışık doğrulama    : "
        f"{REQUIRED_CONSECUTIVE_FRAMES} kare"
        f"{'':<29}║"
    )
    print(
        "║  Şekil hesabı         : minAreaRect                       ║"
    )
    print(
        "║  Durdurmak için       : CTRL+C, Q veya ESC               ║"
    )
    print(
        "╚════════════════════════════════════════════════════════════╝"
    )

    camera = Camera()

    print("\n[1/2] Kamera başlatılıyor...")

    if not camera.start():
        sys.exit(1)

    tracker = DetectionTracker()

    duration = (
        float("inf")
        if args.surekli
        else float(args.sure)
    )

    stage_statistics: Dict[
        str,
        StageStatistics,
    ] = {
        "empty_1": StageStatistics(),
        "blue": StageStatistics(),
        "empty_2": StageStatistics(),
        "red": StageStatistics(),
        "empty_3": StageStatistics(),
        "both": StageStatistics(),
    }

    frame_count = 0
    saved_count = 0

    total_blue_verified_frames = 0
    total_red_verified_frames = 0

    total_blue_events = 0
    total_red_events = 0

    previous_blue_verified = False
    previous_red_verified = False

    start_time = time.time()
    last_print_time = 0.0
    last_stage_name = ""

    print("\n[2/2] Tespit başlıyor...")
    print("─" * 76)

    try:
        while not stop_requested:
            elapsed = (
                time.time() - start_time
            )

            if elapsed >= duration:
                break

            (
                current_stage_name,
                current_stage_key,
            ) = get_test_stage(elapsed)

            if current_stage_name != last_stage_name:
                print()
                print("═" * 76)
                print(
                    "TEST AŞAMASI: "
                    f"{current_stage_name}"
                )
                print("═" * 76)

                tracker.reset("MAVİ")
                tracker.reset("KIRMIZI")

                previous_blue_verified = False
                previous_red_verified = False

                last_stage_name = (
                    current_stage_name
                )

            frame = camera.read()

            if frame is None:
                time.sleep(0.02)
                continue

            frame_count += 1

            (
                blue_detections,
                red_detections,
                blue_mask,
                red_mask,
            ) = detect_all(frame)

            best_blue = tracker.update(
                "MAVİ",
                blue_detections,
            )

            best_red = tracker.update(
                "KIRMIZI",
                red_detections,
            )

            blue_verified = (
                best_blue is not None
                and best_blue.verified
            )

            red_verified = (
                best_red is not None
                and best_red.verified
            )

            if blue_verified:
                total_blue_verified_frames += 1

            if red_verified:
                total_red_verified_frames += 1

            if (
                blue_verified
                and not previous_blue_verified
            ):
                total_blue_events += 1

            if (
                red_verified
                and not previous_red_verified
            ):
                total_red_events += 1

            previous_blue_verified = (
                blue_verified
            )

            previous_red_verified = (
                red_verified
            )

            current_statistics = (
                stage_statistics[
                    current_stage_key
                ]
            )

            update_stage_statistics(
                current_statistics,
                blue_verified,
                red_verified,
            )

            current_time = time.time()

            should_print = (
                blue_verified
                or red_verified
            ) and (
                current_time
                - last_print_time
                >= 0.4
            )

            if should_print:
                last_print_time = current_time

                timestamp = time.strftime(
                    "%H:%M:%S"
                )

                if args.surekli:
                    remaining_text = ""
                else:
                    remaining = max(
                        0.0,
                        duration - elapsed,
                    )

                    remaining_text = (
                        f" | Kalan: "
                        f"{remaining:.0f}s"
                    )

                print()
                print(
                    f"[{timestamp}] "
                    f"Kare #{frame_count}"
                    f"{remaining_text} "
                    f"| Aşama: "
                    f"{current_stage_name}"
                )

                if (
                    blue_verified
                    and best_blue is not None
                ):
                    print(
                        "  🔵 MAVİ DOĞRULANDI | "
                        f"Merkez: {best_blue.center} | "
                        f"Alan: {best_blue.area:.0f}px² | "
                        f"Oran: {best_blue.aspect_ratio:.2f} | "
                        f"Dik.: {best_blue.rectangularity:.2f} | "
                        f"Köşe: {best_blue.corners} | "
                        f"Açı: {best_blue.angle:.0f}° | "
                        f"Güven: %{best_blue.confidence:.0f} | "
                        f"Seri: {best_blue.consecutive_frames}"
                    )

                if (
                    red_verified
                    and best_red is not None
                ):
                    print(
                        "  🔴 KIRMIZI DOĞRULANDI | "
                        f"Merkez: {best_red.center} | "
                        f"Alan: {best_red.area:.0f}px² | "
                        f"Oran: {best_red.aspect_ratio:.2f} | "
                        f"Dik.: {best_red.rectangularity:.2f} | "
                        f"Köşe: {best_red.corners} | "
                        f"Açı: {best_red.angle:.0f}° | "
                        f"Güven: %{best_red.confidence:.0f} | "
                        f"Seri: {best_red.consecutive_frames}"
                    )

            annotated = draw_results(
                frame,
                best_blue,
                best_red,
                current_stage_name,
                elapsed,
            )

            if (
                frame_count
                % args.kayit_araligi
                == 0
            ):
                saved_count += 1

                detect_path = os.path.join(
                    DEBUG_DIR,
                    (
                        f"frame_"
                        f"{saved_count:04d}"
                        f"_detect.jpg"
                    ),
                )

                grid_path = os.path.join(
                    DEBUG_DIR,
                    (
                        f"frame_"
                        f"{saved_count:04d}"
                        f"_grid.jpg"
                    ),
                )

                debug_grid = create_debug_grid(
                    frame,
                    annotated,
                    blue_mask,
                    red_mask,
                )

                cv2.imwrite(
                    detect_path,
                    annotated,
                )

                cv2.imwrite(
                    grid_path,
                    debug_grid,
                )

            if args.pencere:
                cv2.imshow(
                    (
                        "TEKNOFEST IHA - "
                        "Dondurmeye Dayanikli Tespit"
                    ),
                    annotated,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    27,
                    ord("q"),
                    ord("Q"),
                ):
                    stop_requested = True

    except Exception as error:
        print()
        print(
            "Program çalışırken hata oluştu: "
            f"{error}"
        )

        raise

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    total_elapsed = (
        time.time() - start_time
    )

    if total_elapsed > 0:
        average_fps = (
            frame_count / total_elapsed
        )
    else:
        average_fps = 0.0

    print()
    print("═" * 76)
    print("GENEL TEST SONUÇLARI")
    print("═" * 76)
    print(
        f"  Süre                         : "
        f"{total_elapsed:.1f} saniye"
    )
    print(
        f"  İşlenen toplam kare          : "
        f"{frame_count}"
    )
    print(
        f"  Ortalama FPS                 : "
        f"{average_fps:.1f}"
    )
    print(
        f"  Kaydedilen debug çifti       : "
        f"{saved_count}"
    )
    print(
        f"  Mavi doğrulanmış kare        : "
        f"{total_blue_verified_frames}"
    )
    print(
        f"  Kırmızı doğrulanmış kare     : "
        f"{total_red_verified_frames}"
    )
    print(
        f"  Mavi doğrulama olayı         : "
        f"{total_blue_events}"
    )
    print(
        f"  Kırmızı doğrulama olayı      : "
        f"{total_red_events}"
    )
    print(
        f"  Debug klasörü                : "
        f"{os.path.abspath(DEBUG_DIR)}"
    )

    print()
    print("═" * 76)
    print("AŞAMA BAZLI SONUÇLAR")
    print("═" * 76)

    stage_display_names = {
        "empty_1": "HEDEF YOK 1",
        "blue": "SADECE MAVİ",
        "empty_2": "HEDEF YOK 2",
        "red": "SADECE KIRMIZI",
        "empty_3": "HEDEF YOK 3",
        "both": "MAVİ + KIRMIZI",
    }

    for stage_key, stage_name in stage_display_names.items():
        statistics = stage_statistics[
            stage_key
        ]

        print()
        print(f"  {stage_name}")
        print(
            f"    Toplam kare           : "
            f"{statistics.total_frames}"
        )
        print(
            f"    Mavi doğrulanmış kare : "
            f"{statistics.blue_verified_frames}"
        )
        print(
            f"    Kırmızı doğrulanmış   : "
            f"{statistics.red_verified_frames}"
        )
        print(
            f"    Mavi olay             : "
            f"{statistics.blue_events}"
        )
        print(
            f"    Kırmızı olay          : "
            f"{statistics.red_events}"
        )

    empty_blue_false_positives = (
        stage_statistics[
            "empty_1"
        ].blue_verified_frames
        + stage_statistics[
            "empty_2"
        ].blue_verified_frames
        + stage_statistics[
            "empty_3"
        ].blue_verified_frames
    )

    empty_red_false_positives = (
        stage_statistics[
            "empty_1"
        ].red_verified_frames
        + stage_statistics[
            "empty_2"
        ].red_verified_frames
        + stage_statistics[
            "empty_3"
        ].red_verified_frames
    )

    print()
    print("═" * 76)
    print("YANLIŞ POZİTİF ÖZETİ")
    print("═" * 76)
    print(
        f"  Hedefsiz aşamalarda mavi "
        f"doğrulama : "
        f"{empty_blue_false_positives} kare"
    )
    print(
        f"  Hedefsiz aşamalarda kırmızı "
        f"doğrulama: "
        f"{empty_red_false_positives} kare"
    )
    print("═" * 76)


if __name__ == "__main__":
    main()
