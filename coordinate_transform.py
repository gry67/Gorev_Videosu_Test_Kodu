# -*- coding: utf-8 -*-
"""
Teknofest İHA - Piksel → GPS Koordinat Dönüşüm Modülü
=======================================================
Kamera görüntüsündeki piksel koordinatlarını,
İHA'nın GPS konumu ve attitude bilgilerini kullanarak
gerçek dünya GPS koordinatlarına dönüştürür.

Yöntem:
1. Piksel → Kamera koordinatları (intrinsic matris ile)
2. Kamera → NED çerçevesi (roll, pitch, yaw rotasyonu)
3. NED offset → GPS koordinatları (irtifa ile yer projeksiyonu)
"""

import math
from typing import Tuple, Optional

import numpy as np

import config
from logger_config import setup_logger

logger = setup_logger("coordinate_transform")

# Dünya sabitleri
EARTH_RADIUS = 6378137.0  # metre (WGS84 ekvatoral yarıçap)


class CoordinateTransformer:
    """
    Piksel koordinatlarını GPS koordinatlarına dönüştüren sınıf.
    İHA'nın anlık konumu (GPS) ve durumu (attitude) bilgilerine ihtiyaç duyar.
    """

    def __init__(self):
        """
        Kamera intrinsic matrisini yükler.
        """
        self.camera_matrix = config.CAMERA_MATRIX
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)
        self.dist_coeffs = config.CAMERA_DIST_COEFFS

        logger.info("CoordinateTransformer başlatıldı")
        logger.info(f"  Kamera matrisi:\n{self.camera_matrix}")

    def pixel_to_gps(
        self,
        pixel_x: int,
        pixel_y: int,
        uav_lat: float,
        uav_lon: float,
        uav_alt: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Optional[Tuple[float, float]]:
        """
        Piksel koordinatlarını GPS koordinatlarına dönüştürür.

        Args:
            pixel_x: Piksel X koordinatı
            pixel_y: Piksel Y koordinatı
            uav_lat: İHA enlemi (derece)
            uav_lon: İHA boylamı (derece)
            uav_alt: İHA irtifası (metre, AGL)
            roll: İHA roll açısı (radyan)
            pitch: İHA pitch açısı (radyan)
            yaw: İHA yaw açısı (radyan)

        Returns:
            (latitude, longitude) veya dönüşüm başarısız ise None
        """
        try:
            # 1. Piksel → Normalleştirilmiş kamera koordinatları
            pixel_homogeneous = np.array([pixel_x, pixel_y, 1.0])
            camera_coords = self.camera_matrix_inv @ pixel_homogeneous

            # 2. Kamera → Gövde çerçevesi rotasyonu
            # Kameranın aşağı baktığını varsayıyoruz (nadir montaj)
            # Kamera çerçevesi (OpenCV): X-sağ, Y-aşağı, Z-ileri (optik eksen)
            # Nadir montajda optik eksen (Z) → gövde Z-aşağı yönüne eşlenir
            # Kamera X-sağ → gövde Y-sağ
            # Kamera Y-aşağı → gövde X-ileri (kamera döndürüldüğünde)
            # Gövde çerçevesi: X-ileri (kuzey), Y-sağ (doğu), Z-aşağı
            R_cam_to_body = np.array([
                [0,  -1,  0],   # Gövde X (ileri) ← -Kamera Y
                [1,   0,  0],   # Gövde Y (sağ)   ← Kamera X
                [0,   0,  1]    # Gövde Z (aşağı)  ← Kamera Z (optik eksen)
            ], dtype=np.float64)

            body_coords = R_cam_to_body @ camera_coords

            # 3. Gövde → NED (North-East-Down) çerçevesi rotasyonu
            R_body_to_ned = self._rotation_matrix(roll, pitch, yaw)
            ned_coords = R_body_to_ned @ body_coords

            # 4. NED vektörünü yere projekte et
            # ned_coords bir yön vektörüdür, irtifa ile ölçeklendiriyoruz
            if ned_coords[2] <= 0:
                logger.warning(
                    f"Geçersiz NED-Z bileşeni: {ned_coords[2]:.3f} "
                    f"(kamera yukarı bakıyor olabilir)"
                )
                return None

            # Ölçeklendirme faktörü: yer düzlemine ulaşmak için
            scale = uav_alt / ned_coords[2]

            # Yerdeki NED offset (metre)
            north_offset = scale * ned_coords[0]  # Kuzey offset
            east_offset = scale * ned_coords[1]   # Doğu offset

            # 5. NED offset → GPS koordinatları
            target_lat, target_lon = self._ned_to_gps(
                uav_lat, uav_lon, north_offset, east_offset
            )

            logger.debug(
                f"Piksel ({pixel_x}, {pixel_y}) → "
                f"NED ({north_offset:.1f}m N, {east_offset:.1f}m E) → "
                f"GPS ({target_lat:.6f}, {target_lon:.6f})"
            )

            return (target_lat, target_lon)

        except Exception as e:
            logger.error(f"Koordinat dönüşüm hatası: {e}")
            return None

    def _rotation_matrix(
        self, roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """
        Euler açılarından (ZYX sırası) rotasyon matrisi oluşturur.
        Gövde çerçevesinden NED çerçevesine dönüşüm.

        Args:
            roll: Roll açısı (radyan)
            pitch: Pitch açısı (radyan)
            yaw: Yaw/Heading açısı (radyan)

        Returns:
            3x3 rotasyon matrisi
        """
        # Yaw (Z ekseni etrafında)
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        R = np.array([
            [cy * cp,   cy * sp * sr - sy * cr,   cy * sp * cr + sy * sr],
            [sy * cp,   sy * sp * sr + cy * cr,   sy * sp * cr - cy * sr],
            [-sp,       cp * sr,                   cp * cr               ]
        ], dtype=np.float64)

        return R

    def _ned_to_gps(
        self,
        ref_lat: float,
        ref_lon: float,
        north_offset: float,
        east_offset: float,
    ) -> Tuple[float, float]:
        """
        NED offset'i (metre) GPS koordinatlarına dönüştürür.
        Düz dünya yaklaşımı kullanır (kısa mesafeler için yeterli).

        Args:
            ref_lat: Referans enlemi (derece)
            ref_lon: Referans boylamı (derece)
            north_offset: Kuzey yönünde offset (metre)
            east_offset: Doğu yönünde offset (metre)

        Returns:
            (target_lat, target_lon) derece cinsinden
        """
        # Enlem değişimi
        d_lat = north_offset / EARTH_RADIUS
        target_lat = ref_lat + math.degrees(d_lat)

        # Boylam değişimi (enlem düzeltmesi ile)
        d_lon = east_offset / (EARTH_RADIUS * math.cos(math.radians(ref_lat)))
        target_lon = ref_lon + math.degrees(d_lon)

        return (target_lat, target_lon)

    def gps_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """
        İki GPS noktası arasındaki mesafeyi hesaplar (Haversine formülü).

        Args:
            lat1, lon1: Birinci nokta (derece)
            lat2, lon2: İkinci nokta (derece)

        Returns:
            Mesafe (metre)
        """
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return EARTH_RADIUS * c

    def gps_bearing(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """
        İki GPS noktası arasındaki yön açısını (bearing) hesaplar.

        Args:
            lat1, lon1: Başlangıç noktası (derece)
            lat2, lon2: Hedef nokta (derece)

        Returns:
            Yön açısı (derece, 0-360, kuzey=0)
        """
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)

        x = math.sin(dlon) * math.cos(lat2_r)
        y = (
            math.cos(lat1_r) * math.sin(lat2_r)
            - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        )

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360


# =============================================================================
# BAĞIMSIZ TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test: Bilinen piksel koordinatları için GPS dönüşümü doğrulaması.
    """
    print("=" * 60)
    print("CoordinateTransformer Testi")
    print("=" * 60)

    ct = CoordinateTransformer()

    # Test senaryosu: İHA 50m yükseklikte, düz uçuş, kuzey yönünde
    uav_lat = 39.925533
    uav_lon = 32.866287
    uav_alt = 50.0
    roll = 0.0      # radyan
    pitch = 0.0     # radyan
    yaw = 0.0       # radyan (kuzey)

    print(f"\nİHA Konumu: ({uav_lat}, {uav_lon}), İrtifa: {uav_alt}m")
    print(f"Attitude: Roll={math.degrees(roll):.1f}°, "
          f"Pitch={math.degrees(pitch):.1f}°, "
          f"Yaw={math.degrees(yaw):.1f}°")

    # Test pikselleri
    test_pixels = [
        (320, 240, "Merkez (tam altında)"),
        (0, 0, "Sol üst köşe"),
        (639, 479, "Sağ alt köşe"),
        (320, 120, "Üst merkez (kuzeyde)"),
        (500, 240, "Sağ merkez (doğuda)"),
    ]

    print("\n{:<25} {:<15} → {:<30} {:<10}".format(
        "Açıklama", "Piksel", "GPS", "Mesafe"
    ))
    print("-" * 80)

    for px, py, desc in test_pixels:
        result = ct.pixel_to_gps(px, py, uav_lat, uav_lon, uav_alt, roll, pitch, yaw)
        if result:
            target_lat, target_lon = result
            distance = ct.gps_distance(uav_lat, uav_lon, target_lat, target_lon)
            print(f"{desc:<25} ({px:3d},{py:3d})      → "
                  f"({target_lat:.6f}, {target_lon:.6f})  {distance:.1f}m")
        else:
            print(f"{desc:<25} ({px:3d},{py:3d})      → BAŞARISIZ")

    # Mesafe testi
    print("\n--- Mesafe Testi ---")
    dist = ct.gps_distance(39.925533, 32.866287, 39.926533, 32.867287)
    print(f"Test mesafesi: {dist:.1f}m")

    bearing = ct.gps_bearing(39.925533, 32.866287, 39.926533, 32.867287)
    print(f"Test yönü: {bearing:.1f}°")

    print("\nTest tamamlandı.")
