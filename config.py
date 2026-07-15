# -*- coding: utf-8 -*-
"""
Teknofest Uluslararası İHA Yarışması - Konfigürasyon Dosyası
=============================================================
Tüm sistem parametreleri bu dosyada tanımlanmıştır.
Orange Cube+ uçuş kontrolcüsü ile uyumludur.
"""

import numpy as np

# =============================================================================
# BAĞLANTI AYARLARI
# =============================================================================


#GÖREV VİDEOSU TEST DURUMU 0=webcam, 1=Raspberry Kamerası
GOREV_KAMERA_SECIM = 0


# MAVLink bağlantı stringi (Orange Cube+ için)
# USB: '/dev/ttyACM0'  |  Telemetri: '/dev/ttyUSB0'  |  TCP: 'tcp:127.0.0.1:5763'
CONNECTION_STRING = '/dev/ttyACM0'
CONNECTION_BAUD = 57600
CONNECTION_TIMEOUT = 30  # saniye

# Mission Planner TCP bağlantısı (opsiyonel, SITL test için)
SITL_CONNECTION = 'udp:0.0.0.0:14550'

# =============================================================================
# KAMERA AYARLARI
# =============================================================================

# Kamera indeksi (USB kamera)
CAMERA_INDEX = 0

# Çözünürlük
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Görüş alanı (Field of View) - derece cinsinden
CAMERA_HFOV = 62.2  # Yatay FOV
CAMERA_VFOV = 48.8  # Dikey FOV

# Kamera intrinsic matrisi (kalibrasyondan elde edilir)
# Varsayılan değerler - KALİBRASYON YAPILDIKTAN SONRA GÜNCELLENMELİDİR
CAMERA_FOCAL_LENGTH_X = 554.25  # piksel cinsinden
CAMERA_FOCAL_LENGTH_Y = 554.25
CAMERA_CENTER_X = CAMERA_WIDTH / 2   # 320
CAMERA_CENTER_Y = CAMERA_HEIGHT / 2  # 240

CAMERA_MATRIX = np.array([
    [CAMERA_FOCAL_LENGTH_X, 0, CAMERA_CENTER_X],
    [0, CAMERA_FOCAL_LENGTH_Y, CAMERA_CENTER_Y],
    [0, 0, 1]
], dtype=np.float64)

# Distorsiyon katsayıları (kalibrasyondan elde edilir)
CAMERA_DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# =============================================================================
# RENK TESPİTİ - HSV ARALIKLARI
# =============================================================================
# OpenCV HSV: H(0-179), S(0-255), V(0-255)

# Mavi bölge HSV aralığı
BLUE_HSV_LOWER = np.array([100, 80, 50])
BLUE_HSV_UPPER = np.array([130, 255, 255])

# Kırmızı bölge HSV aralığı (kırmızı Hue sınırında olduğu için iki aralık)
RED_HSV_LOWER_1 = np.array([0, 80, 50])
RED_HSV_UPPER_1 = np.array([10, 255, 255])
RED_HSV_LOWER_2 = np.array([170, 80, 50])
RED_HSV_UPPER_2 = np.array([179, 255, 255])

# =============================================================================
# ŞEKİL DOĞRULAMA
# =============================================================================

# Minimum kontur alanı (piksel²) - gürültü filtreleme
MIN_CONTOUR_AREA = 500

# Maksimum kontur alanı (piksel²)
MAX_CONTOUR_AREA = 200000

# Kare/dikdörtgen şekil doğrulaması için köşe sayısı toleransı
SHAPE_APPROX_EPSILON = 0.04  # cv2.approxPolyDP epsilon katsayısı

# En-boy oranı toleransı (kare bölgeler için ≈ 1.0)
ASPECT_RATIO_MIN = 0.6
ASPECT_RATIO_MAX = 1.4

# Dikdörtgensellik skoru minimum eşiği (alan / boundingRect alanı)
RECTANGULARITY_MIN = 0.7

# Minimum güven skoru (0-100)
MIN_CONFIDENCE_SCORE = 60

#search süresi
SEARCH_DURATION_SECONDS = 999999.0
DETECTION_SAMPLE_INTERVAL = 0.20
MINIMUM_DETECTION_SAMPLES = 1

# =============================================================================
# SERVO / FAYDALI YÜK BIRAKMA
# =============================================================================

# Servo kanal numarası (Orange Cube+ AUX çıkışları)
# AUX1 = Kanal 9, AUX2 = Kanal 10, vb.
SERVO_CHANNEL = 9

# PWM değerleri (mikrosaniye)
SERVO_NEUTRAL_PWM = 1500    # Nötr pozisyon (yük tutuluyor)
SERVO_PAYLOAD_1_PWM = 1100  # Yük 1 bırakma (mavi bölge)
SERVO_PAYLOAD_2_PWM = 1900  # Yük 2 bırakma (kırmızı bölge)

# Servo hareket sonrası bekleme süresi (saniye)
SERVO_ACTION_DELAY = 2.0

# =============================================================================
# UÇUŞ PARAMETRELERİ
# =============================================================================

# Arama yüksekliği (metre - AGL)
SEARCH_ALTITUDE = 50

# Yük bırakma yüksekliği (metre - AGL)
DROP_ALTITUDE = 30

# Uçuş hızı (m/s)
CRUISE_SPEED = 15  # Seyir hızı
APPROACH_SPEED = 8  # Yaklaşma hızı

# Waypoint'e varış yarıçapı (metre)
WAYPOINT_REACHED_RADIUS = 10

# Waypoint'e yaklaşma yarıçapı - yavaşlama başlangıcı (metre)
WAYPOINT_APPROACH_RADIUS = 30

# Loiter (bekleme) yarıçapı (metre) - sabit kanat için
LOITER_RADIUS = 50

# =============================================================================
# ARAMA ALANI (Seyir Uçuşu Waypoint'leri)
# =============================================================================

# Arama alanı merkez koordinatları (örnek - yarışma alanına göre güncellenmeli)
SEARCH_AREA_CENTER_LAT = 39.925533  # Örnek: Ankara
SEARCH_AREA_CENTER_LON = 32.866287

# Arama alanı boyutları (metre)
SEARCH_AREA_LENGTH = 500  # Uzunluk
SEARCH_AREA_WIDTH = 300   # Genişlik

# Kalkış noktası (home pozisyonu)
HOME_LAT = 39.925533
HOME_LON = 32.866287
HOME_ALT = 0

# =============================================================================
# GÖREV DURUM MAKİNESİ
# =============================================================================

# Durum tanımları
class MissionState:
    INIT = "INIT"
    TAKEOFF = "TAKEOFF"
    SEARCH = "SEARCH"
    PROCESS = "PROCESS"
    NAVIGATE_BLUE = "NAVIGATE_BLUE"
    DROP_BLUE = "DROP_BLUE"
    NAVIGATE_RED = "NAVIGATE_RED"
    DROP_RED = "DROP_RED"
    RTL = "RTL"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"

# =============================================================================
# LOGLAMA
# =============================================================================

LOG_DIR = "logs"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Görüntü kaydetme (debug amaçlı)
SAVE_DEBUG_IMAGES = True
DEBUG_IMAGE_DIR = "debug_images"

# =============================================================================
# GÜVENLİK
# =============================================================================

# Acil durum RTL tetikleme koşulları
BATTERY_FAILSAFE_VOLTAGE = 10.5  # Volt
MAX_MISSION_TIME = 600  # Maksimum görev süresi (saniye)
GEOFENCE_RADIUS = 1000  # Coğrafi sınır yarıçapı (metre)
