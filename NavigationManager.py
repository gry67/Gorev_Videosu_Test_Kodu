# -*- coding: utf-8 -*-
"""
Teknofest İHA - Navigasyon Modülü
===================================
DroneKit/MAVLink ile Orange Cube+ uçuş kontrolcüsü haberleşmesi.

Özellikler:
- MAVLink bağlantısı yönetimi
- Telemetri okuma (GPS, irtifa, attitude)
- Uçuş modu değiştirme
- Waypoint/Misyon oluşturma ve yükleme
- Pozisyon takibi ve waypoint'e varış kontrolü
"""

import time
import math
import collections
import collections.abc

if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
from typing import List, Tuple, Optional, Callable

from dronekit import connect, VehicleMode, Command, LocationGlobalRelative
from pymavlink import mavutil

import config
from logger_config import setup_logger

logger = setup_logger("navigation")


class NavigationManager:
    """
    İHA navigasyon yöneticisi.
    Orange Cube+ ile MAVLink üzerinden haberleşir.
    """

    def __init__(self):
        """NavigationManager başlatır. Bağlantı henüz yapılmaz."""
        self.vehicle = None
        self.is_connected = False
        self._waypoint_reached_callbacks = []

        logger.info("NavigationManager başlatıldı")

    def connect(self, connection_string: str = None, baud: int = None) -> bool:
        """
        Orange Cube+ uçuş kontrolcüsüne bağlanır.

        Args:
            connection_string: MAVLink bağlantı stringi (None ise config'den alınır)
            baud: Baud rate (None ise config'den alınır)

        Returns:
            Bağlantı başarılı ise True
        """
        conn_str = connection_string or config.CONNECTION_STRING
        baud_rate = baud or config.CONNECTION_BAUD

        logger.info(f"Bağlanılıyor: {conn_str} (baud={baud_rate})...")

        try:
            self.vehicle = connect(
                conn_str,
                baud=baud_rate,
                wait_ready=True,
                heartbeat_timeout=config.CONNECTION_TIMEOUT,
            )
            self.is_connected = True

            logger.info("✓ Bağlantı başarılı!")
            self._print_vehicle_info()

            return True

        except Exception as e:
            logger.error(f"✗ Bağlantı hatası: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Bağlantıyı kapatır."""
        if self.vehicle is not None:
            self.vehicle.close()
            self.is_connected = False
            logger.info("Bağlantı kapatıldı")

    def _print_vehicle_info(self):
        """Araç bilgilerini loglar."""
        if self.vehicle is None:
            return

        logger.info("--- Araç Bilgileri ---")
        logger.info(f"  Firmware: {self.vehicle.version}")
        logger.info(f"  GPS: {self.vehicle.gps_0}")
        logger.info(f"  Batarya: {self.vehicle.battery}")
        logger.info(f"  Mod: {self.vehicle.mode.name}")
        logger.info(f"  Armed: {self.vehicle.armed}")
        logger.info(f"  Sistem durumu: {self.vehicle.system_status.state}")

    # =========================================================================
    # TELEMETRİ
    # =========================================================================

    def get_location(self) -> Optional[Tuple[float, float, float]]:
        """
        İHA'nın mevcut GPS konumunu döndürür.

        Returns:
            (latitude, longitude, altitude_relative) veya None
        """
        if self.vehicle is None:
            return None

        loc = self.vehicle.location.global_relative_frame
        if loc is None:
            return None

        return (loc.lat, loc.lon, loc.alt)

    def get_attitude(self) -> Optional[Tuple[float, float, float]]:
        """
        İHA'nın mevcut attitude bilgisini döndürür.

        Returns:
            (roll, pitch, yaw) radyan cinsinden veya None
        """
        if self.vehicle is None:
            return None

        att = self.vehicle.attitude
        if att is None:
            return None

        return (att.roll, att.pitch, att.yaw)

    def get_heading(self) -> Optional[float]:
        """
        İHA'nın mevcut yön açısını döndürür.

        Returns:
            Heading (derece, 0-360) veya None
        """
        if self.vehicle is None:
            return None
        return self.vehicle.heading

    def get_groundspeed(self) -> Optional[float]:
        """
        İHA'nın yer hızını döndürür.

        Returns:
            Yer hızı (m/s) veya None
        """
        if self.vehicle is None:
            return None
        return self.vehicle.groundspeed

    def get_battery(self) -> Optional[dict]:
        """
        Batarya bilgisini döndürür.

        Returns:
            {'voltage': V, 'current': A, 'level': %} veya None
        """
        if self.vehicle is None:
            return None

        bat = self.vehicle.battery
        return {
            'voltage': bat.voltage,
            'current': bat.current,
            'level': bat.level,
        }

    def get_gps_fix(self) -> Optional[int]:
        """
        GPS fix tipini döndürür.

        Returns:
            GPS fix tipi (0=no fix, 2=2D, 3=3D, vb.)
        """
        if self.vehicle is None:
            return None
        return self.vehicle.gps_0.fix_type

    # =========================================================================
    # UÇUŞ MODU
    # =========================================================================

    def set_mode(self, mode_name: str, timeout: float = 10.0) -> bool:
        """
        Uçuş modunu değiştirir.

        Args:
            mode_name: Mod adı (AUTO, GUIDED, RTL, MANUAL, FBWA, LOITER, vb.)
            timeout: Mod değişikliği bekleme süresi (saniye)

        Returns:
            Başarılı ise True
        """
        if self.vehicle is None:
            logger.error("Vehicle bağlantısı yok!")
            return False

        logger.info(f"Uçuş modu değiştiriliyor: {mode_name}")

        self.vehicle.mode = VehicleMode(mode_name)

        # Mod değişikliğini bekle
        start_time = time.time()
        while self.vehicle.mode.name != mode_name:
            if time.time() - start_time > timeout:
                logger.error(
                    f"Mod değişikliği zaman aşımı! "
                    f"Mevcut mod: {self.vehicle.mode.name}"
                )
                return False
            time.sleep(0.5)

        logger.info(f"✓ Uçuş modu: {mode_name}")
        return True

    def arm(self, timeout: float = 15.0) -> bool:
        """
        İHA'yı arm eder.

        Args:
            timeout: Bekleme süresi (saniye)

        Returns:
            Başarılı ise True
        """
        if self.vehicle is None:
            return False

        logger.info("İHA arm ediliyor...")

        self.vehicle.armed = True

        start_time = time.time()
        while not self.vehicle.armed:
            if time.time() - start_time > timeout:
                logger.error("Arm zaman aşımı!")
                return False
            time.sleep(0.5)

        logger.info("✓ İHA armed")
        return True

    # =========================================================================
    # MİSYON / WAYPOINT YÖNETİMİ
    # =========================================================================

    def create_mission(
        self,
        blue_targets: List[Tuple[float, float]],
        red_targets: List[Tuple[float, float]],
        search_waypoints: List[Tuple[float, float]] = None,
    ) -> bool:
        """
        Görev misyonunu oluşturur ve Orange Cube+'a yükler.

        Misyon sırası:
        1. Kalkış
        2. (Opsiyonel) Arama waypoint'leri
        3. Mavi bölgelere uçuş + yük 1 bırakma
        4. Kırmızı bölgelere uçuş + yük 2 bırakma
        5. Eve dönüş (RTL)

        Args:
            blue_targets: Mavi bölge GPS koordinatları [(lat, lon), ...]
            red_targets: Kırmızı bölge GPS koordinatları [(lat, lon), ...]
            search_waypoints: Arama rotası waypoint'leri (opsiyonel)

        Returns:
            Başarılı ise True
        """
        if self.vehicle is None:
            logger.error("Vehicle bağlantısı yok!")
            return False

        logger.info("=" * 50)
        logger.info("MİSYON OLUŞTURULUYOR")
        logger.info("=" * 50)

        try:
            cmds = self.vehicle.commands
            cmds.clear()

            # Mevcut konum (home olarak kullan)
            home_loc = self.vehicle.location.global_relative_frame

            # ---- 1. Kalkış Komutu ----
            # Takeoff komutu (sadece döner kanat için - sabit kanatta AUTO takeoff)
            # Sabit kanat için NAV_TAKEOFF kullanılır
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0,
                15,     # param1: pitch açısı (derece)
                0, 0, 0,
                home_loc.lat, home_loc.lon,
                config.SEARCH_ALTITUDE,  # hedef irtifa
            ))
            logger.info(f"  [WP 0] Kalkış → {config.SEARCH_ALTITUDE}m")

            wp_index = 1

            # ---- 2. Arama Waypoint'leri (Opsiyonel) ----
            if search_waypoints:
                for lat, lon in search_waypoints:
                    cmds.add(Command(
                        0, 0, 0,
                        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                        0, 0,
                        0,   # param1: hold time
                        config.WAYPOINT_REACHED_RADIUS,  # param2: acceptance radius
                        0, 0,
                        lat, lon,
                        config.SEARCH_ALTITUDE,
                    ))
                    logger.info(f"  [WP {wp_index}] Arama → ({lat:.6f}, {lon:.6f})")
                    wp_index += 1

            # ---- 3. Mavi Bölge Waypoint'leri + Yük 1 Bırakma ----
            for i, (lat, lon) in enumerate(blue_targets):
                # Waypoint'e git
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 0,
                    0,
                    config.WAYPOINT_REACHED_RADIUS,
                    0, 0,
                    lat, lon,
                    config.DROP_ALTITUDE,
                ))
                logger.info(
                    f"  [WP {wp_index}] MAVİ Hedef {i+1} → "
                    f"({lat:.6f}, {lon:.6f}) @ {config.DROP_ALTITUDE}m"
                )
                wp_index += 1

                # Servo komutu - Yük 1 bırak
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                    0, 0,
                    config.SERVO_CHANNEL,         # param1: servo kanalı
                    config.SERVO_PAYLOAD_1_PWM,   # param2: PWM
                    0, 0, 0, 0, 0,
                ))
                logger.info(
                    f"  [WP {wp_index}] SERVO → Yük 1 Bırak "
                    f"(Kanal={config.SERVO_CHANNEL}, PWM={config.SERVO_PAYLOAD_1_PWM})"
                )
                wp_index += 1

            # ---- 4. Kırmızı Bölge Waypoint'leri + Yük 2 Bırakma ----
            for i, (lat, lon) in enumerate(red_targets):
                # Waypoint'e git
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 0,
                    0,
                    config.WAYPOINT_REACHED_RADIUS,
                    0, 0,
                    lat, lon,
                    config.DROP_ALTITUDE,
                ))
                logger.info(
                    f"  [WP {wp_index}] KIRMIZI Hedef {i+1} → "
                    f"({lat:.6f}, {lon:.6f}) @ {config.DROP_ALTITUDE}m"
                )
                wp_index += 1

                # Servo komutu - Yük 2 bırak
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                    0, 0,
                    config.SERVO_CHANNEL,         # param1: servo kanalı
                    config.SERVO_PAYLOAD_2_PWM,   # param2: PWM
                    0, 0, 0, 0, 0,
                ))
                logger.info(
                    f"  [WP {wp_index}] SERVO → Yük 2 Bırak "
                    f"(Kanal={config.SERVO_CHANNEL}, PWM={config.SERVO_PAYLOAD_2_PWM})"
                )
                wp_index += 1

            # ---- 5. Eve Dönüş (RTL) ----
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ))
            logger.info(f"  [WP {wp_index}] RTL (Eve Dönüş)")

            # ---- Misyonu Yükle ----
            logger.info(f"\nToplam {wp_index + 1} komut yükleniyor...")
            cmds.upload()

            logger.info("✓ Misyon başarıyla yüklendi!")
            return True

        except Exception as e:
            logger.error(f"✗ Misyon oluşturma hatası: {e}")
            return False

    def create_search_mission(
        self, waypoints: List[Tuple[float, float]]
    ) -> bool:
        """
        Arama rotası misyonu oluşturur (görüntü işleme sırasında kullanılır).

        Args:
            waypoints: Arama waypoint'leri [(lat, lon), ...]

        Returns:
            Başarılı ise True
        """
        if self.vehicle is None:
            return False

        try:
            cmds = self.vehicle.commands
            cmds.clear()

            home_loc = self.vehicle.location.global_relative_frame

            # Kalkış
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0,
                15, 0, 0, 0,
                home_loc.lat, home_loc.lon,
                config.SEARCH_ALTITUDE,
            ))

            # Arama waypoint'leri
            for lat, lon in waypoints:
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 0,
                    0, config.WAYPOINT_REACHED_RADIUS, 0, 0,
                    lat, lon,
                    config.SEARCH_ALTITUDE,
                ))

            # Loiter (hedefler bulunana kadar bekleme)
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_LOITER_UNLIM,
                0, 0,
                0, config.LOITER_RADIUS, 0, 0,
                home_loc.lat, home_loc.lon,
                config.SEARCH_ALTITUDE,
            ))

            cmds.upload()
            logger.info(f"✓ Arama misyonu yüklendi ({len(waypoints)} waypoint)")
            return True

        except Exception as e:
            logger.error(f"Arama misyonu yükleme hatası: {e}")
            return False

    def generate_search_pattern(
        self,
        center_lat: float,
        center_lon: float,
        length: float,
        width: float,
        spacing: float = 50.0,
    ) -> List[Tuple[float, float]]:
        """
        Çim biçme (lawnmower) paterni ile arama waypoint'leri oluşturur.

        Args:
            center_lat: Arama alanı merkez enlemi
            center_lon: Arama alanı merkez boylamı
            length: Arama alanı uzunluğu (metre, kuzey-güney)
            width: Arama alanı genişliği (metre, doğu-batı)
            spacing: Tarama hattı aralığı (metre)

        Returns:
            Waypoint listesi [(lat, lon), ...]
        """
        waypoints = []

        # Alanı tarama hatlarına böl
        num_lines = int(width / spacing) + 1
        half_length = length / 2
        half_width = width / 2

        for i in range(num_lines):
            # Doğu-batı offset
            east_offset = -half_width + i * spacing

            if i % 2 == 0:
                # Kuzeyden güneye
                north_start = half_length
                north_end = -half_length
            else:
                # Güneyden kuzeye
                north_start = -half_length
                north_end = half_length

            # Başlangıç noktası
            lat1 = center_lat + (north_start / 6378137.0) * (180 / math.pi)
            lon1 = center_lon + (east_offset / (6378137.0 * math.cos(math.radians(center_lat)))) * (180 / math.pi)
            waypoints.append((lat1, lon1))

            # Bitiş noktası
            lat2 = center_lat + (north_end / 6378137.0) * (180 / math.pi)
            lon2 = lon1  # Aynı boylam hattında
            waypoints.append((lat2, lon2))

        logger.info(
            f"Arama paterni oluşturuldu: {len(waypoints)} waypoint, "
            f"{num_lines} tarama hattı"
        )

        return waypoints

    # =========================================================================
    # POZİSYON TAKİBİ
    # =========================================================================

    def get_distance_to(self, target_lat: float, target_lon: float) -> float:
        """
        Mevcut konumdan hedefe olan mesafeyi hesaplar.

        Args:
            target_lat: Hedef enlemi
            target_lon: Hedef boylamı

        Returns:
            Mesafe (metre)
        """
        loc = self.get_location()
        if loc is None:
            return float('inf')

        lat1, lon1, _ = loc
        return self._haversine(lat1, lon1, target_lat, target_lon)

    def is_at_waypoint(
        self, target_lat: float, target_lon: float, radius: float = None
    ) -> bool:
        """
        İHA'nın belirtilen waypoint'e ulaşıp ulaşmadığını kontrol eder.

        Args:
            target_lat: Hedef enlemi
            target_lon: Hedef boylamı
            radius: Kabul yarıçapı (metre). None ise config'den alınır.

        Returns:
            Waypoint'e ulaşıldıysa True
        """
        r = radius or config.WAYPOINT_REACHED_RADIUS
        distance = self.get_distance_to(target_lat, target_lon)
        return distance <= r

    def get_current_waypoint_index(self) -> int:
        """
        Aktif misyondaki mevcut waypoint indeksini döndürür.

        Returns:
            Waypoint indeksi
        """
        if self.vehicle is None:
            return -1
        return self.vehicle.commands.next

    def wait_for_waypoint(
        self,
        target_lat: float,
        target_lon: float,
        timeout: float = 300.0,
        callback: Callable = None,
    ) -> bool:
        """
        İHA'nın belirtilen waypoint'e ulaşmasını bekler.

        Args:
            target_lat: Hedef enlemi
            target_lon: Hedef boylamı
            timeout: Zaman aşımı (saniye)
            callback: Her kontrol döngüsünde çağrılacak fonksiyon

        Returns:
            Waypoint'e ulaşıldıysa True
        """
        logger.info(
            f"Waypoint bekleniyor: ({target_lat:.6f}, {target_lon:.6f})"
        )

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                logger.warning("Waypoint bekleme zaman aşımı!")
                return False

            distance = self.get_distance_to(target_lat, target_lon)

            if distance <= config.WAYPOINT_REACHED_RADIUS:
                logger.info(
                    f"✓ Waypoint'e ulaşıldı! Mesafe: {distance:.1f}m"
                )
                return True

            # Periyodik bilgi
            if int(time.time()) % 5 == 0:
                loc = self.get_location()
                if loc:
                    logger.debug(
                        f"  Mesafe: {distance:.1f}m, "
                        f"Konum: ({loc[0]:.6f}, {loc[1]:.6f}), "
                        f"İrtifa: {loc[2]:.1f}m"
                    )

            # Callback çağır (görüntü işleme vb. için)
            if callback:
                callback()

            time.sleep(0.5)

    def wait_for_mission_item(
        self, target_wp_index: int, timeout: float = 300.0
    ) -> bool:
        """
        Belirtilen misyon maddesine (waypoint indeksi) ulaşılmasını bekler.

        Args:
            target_wp_index: Hedef waypoint indeksi
            timeout: Zaman aşımı (saniye)

        Returns:
            Hedef indekse ulaşıldıysa True
        """
        logger.info(f"Misyon maddesi bekleniyor: WP #{target_wp_index}")

        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                logger.warning("Misyon maddesi bekleme zaman aşımı!")
                return False

            current = self.get_current_waypoint_index()

            if current >= target_wp_index:
                logger.info(
                    f"✓ Misyon maddesi #{target_wp_index}'e ulaşıldı "
                    f"(mevcut: #{current})"
                )
                return True

            time.sleep(0.5)

    # =========================================================================
    # GÜVENLİK
    # =========================================================================

    def check_battery_failsafe(self) -> bool:
        """
        Batarya failsafe kontrolü yapar.

        Returns:
            Failsafe tetiklendi ise True (acil RTL gerekli)
        """
        battery = self.get_battery()
        if battery is None:
            return False

        voltage = battery.get('voltage')
        if voltage and voltage < config.BATTERY_FAILSAFE_VOLTAGE:
            logger.critical(
                f"⚠ BATARYA FAILSAFE! Voltaj: {voltage:.1f}V "
                f"(eşik: {config.BATTERY_FAILSAFE_VOLTAGE}V)"
            )
            return True

        return False

    def emergency_rtl(self) -> bool:
        """
        Acil eve dönüş (RTL) komutu gönderir.

        Returns:
            Başarılı ise True
        """
        logger.critical("⚠ ACİL RTL BAŞLATILIYOR!")
        return self.set_mode("RTL")

    # =========================================================================
    # YARDIMCI
    # =========================================================================

    @staticmethod
    def _haversine(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Haversine formülü ile iki GPS noktası arasındaki mesafe.

        Returns:
            Mesafe (metre)
        """
        R = 6378137.0
        lat1_r = math.radians(lat1)
        lat2_r = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r)
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


# =============================================================================
# BAĞIMSIZ TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test: Bağlantı ve temel fonksiyonları test eder.
    NOT: Gerçek vehicle veya SITL gerektirir.
    """
    print("=" * 60)
    print("NavigationManager Testi")
    print("=" * 60)

    nav = NavigationManager()

    # Arama paterni testi (vehicle olmadan)
    print("\n--- Arama Paterni Testi ---")
    pattern = nav.generate_search_pattern(
        center_lat=39.925533,
        center_lon=32.866287,
        length=300,
        width=200,
        spacing=60,
    )
    print(f"Oluşturulan waypoint sayısı: {len(pattern)}")
    for i, (lat, lon) in enumerate(pattern):
        print(f"  WP {i}: ({lat:.6f}, {lon:.6f})")

    # Mesafe testi
    print("\n--- Mesafe Testi ---")
    d = NavigationManager._haversine(39.925533, 32.866287, 39.926533, 32.867287)
    print(f"Test mesafesi: {d:.1f}m")

    # Bağlantı testi
    print("\n--- Bağlantı Testi ---")
    print(f"Bağlantı stringi: {config.CONNECTION_STRING}")
    print("SITL bağlantısı deneniyor...")

    if nav.connect(config.SITL_CONNECTION):
        print("✓ SITL bağlantısı başarılı!")

        loc = nav.get_location()
        if loc:
            print(f"  Konum: ({loc[0]:.6f}, {loc[1]:.6f}), İrtifa: {loc[2]:.1f}m")

        att = nav.get_attitude()
        if att:
            print(
                f"  Attitude: Roll={math.degrees(att[0]):.1f}°, "
                f"Pitch={math.degrees(att[1]):.1f}°, "
                f"Yaw={math.degrees(att[2]):.1f}°"
            )

        nav.disconnect()
    else:
        print("✗ SITL bağlantısı başarısız (SITL çalışmıyor olabilir)")
        print("  SITL başlatmak için: sim_vehicle.py -v ArduPlane")

    print("\nTest tamamlandı.")
