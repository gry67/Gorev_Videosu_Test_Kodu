import numpy as np
import os
import sys
import time
import signal
import argparse
import math
from typing import Optional, Tuple, List

import config
from config import MissionState
from logger_config import setup_logger
from vision import VisionProcessor, DetectionResult
from coordinate_transform import CoordinateTransformer
from NavigationManager import NavigationManager
from PayloadManager import PayloadManager



logger = setup_logger("mission")

class MissionController:
    """
    Ana görev kontrolcüsü.
    Tüm modülleri koordine eder ve durum makinesi ile görev akışını yönetir.
    """

    def __init__(self, use_sitl: bool = False, use_camera: bool = True):
        """
        Args:
            use_sitl: SITL simülatörü kullanılsın mı
            use_camera: Kamera kullanılsın mı (False = test modu)
        """
        self.use_sitl = use_sitl
        self.use_camera = use_camera

        # Durum makinesi
        self.state = MissionState.INIT
        self.previous_state = None

        # Modüller
        self.vision = VisionProcessor()
        self.coord_transform = CoordinateTransformer()
        self.navigation = NavigationManager()
        self.payload = PayloadManager()

        # Tespit edilen hedefler
        self.blue_targets: List[Tuple[float, float]] = []   # GPS koordinatları
        self.red_targets: List[Tuple[float, float]] = []     # GPS koordinatları

        # Biriktirilen tespitler (doğrulama için)
        self._blue_detections_buffer: List[Tuple[float, float]] = []
        self._red_detections_buffer: List[Tuple[float, float]] = []
        # SEARCH aşaması ayarları
        # Bu değerler config.py içinde tanımlıysa oradan alınır.
        self.search_duration = float(
            getattr(config, "SEARCH_DURATION_SECONDS", 12.0)
        )
        self.detection_sample_interval = float(
            getattr(config, "DETECTION_SAMPLE_INTERVAL", 0.20)
        )
        self.minimum_detection_samples = int(
            getattr(config, "MINIMUM_DETECTION_SAMPLES", 10)
        )
        self._last_detection_sample_time = 0.0
        self._search_window_started = False

        # Görev zamanlaması
        self.mission_start_time = None
        self.state_start_time = None

        # Güvenli kapatma
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("=" * 60)
        logger.info("TEKNOFEST İHA GÖREV SİSTEMİ BAŞLATILDI")
        logger.info("=" * 60)
        logger.info(f"  SITL Modu: {use_sitl}")
        logger.info(f"  Kamera: {use_camera}")

    def _signal_handler(self, signum, frame):
        """Güvenli kapatma sinyal yöneticisi."""
        logger.warning(f"Kapatma sinyali alındı (signal={signum})")
        self._shutdown_requested = True

    # =========================================================================
    # DURUM MAKİNESİ
    # =========================================================================

    def run(self):
        """
        Ana görev döngüsünü başlatır.
        Durum makinesini sürekli işletir.
        """
        self.mission_start_time = time.time()
        logger.info("Görev döngüsü başlatılıyor...")

        while not self._shutdown_requested:
            try:
                # Güvenlik kontrolleri
                if not self._safety_check():
                    self._transition_to(MissionState.RTL)

                # Görev süre kontrolü
                elapsed = time.time() - self.mission_start_time
                if elapsed > config.MAX_MISSION_TIME:
                    logger.warning(
                        f"Maksimum görev süresi aşıldı! ({elapsed:.0f}s)"
                    )
                    self._transition_to(MissionState.RTL)

                # Durum işleyicisi
                self._process_state()

                # Görev tamamlandıysa çık
                if self.state == MissionState.COMPLETE:
                    break

                time.sleep(0.1)  # CPU yükünü azalt

            except Exception as e:
                logger.error(f"Görev döngüsü hatası: {e}", exc_info=True)
                self._transition_to(MissionState.ERROR)

        self._cleanup()

    def _process_state(self):
        """Mevcut duruma göre uygun işleyiciyi çağırır."""
        handlers = {
            MissionState.INIT: self._handle_init,
            MissionState.SEARCH: self._handle_search,
            MissionState.PROCESS: self._handle_process,
            MissionState.NAVIGATE_BLUE: self._handle_navigate_blue,
            MissionState.DROP_BLUE: self._handle_drop_blue,
            MissionState.NAVIGATE_RED: self._handle_navigate_red,
            MissionState.DROP_RED: self._handle_drop_red,
            MissionState.RTL: self._handle_rtl,
            MissionState.ERROR: self._handle_error,
            MissionState.COMPLETE: self._handle_complete,
        }

        handler = handlers.get(self.state)
        if handler:
            handler()
        else:
            logger.error(f"Bilinmeyen durum: {self.state}")
            self._transition_to(MissionState.ERROR)

    def _transition_to(self, new_state: str):
        """
        Durum geçişi yapar.

        Args:
            new_state: Yeni durum
        """
        if self.state == new_state:
            return

        self.previous_state = self.state
        self.state = new_state
        self.state_start_time = time.time()

        logger.info(f"DURUM GEÇİŞİ: {self.previous_state} → {new_state}")

    # =========================================================================
    # DURUM İŞLEYİCİLERİ
    # =========================================================================

    def _handle_init(self):
        """
        INIT: Sistem başlatma, bağlantı kurma, kontroller.
        """
        logger.info("=" * 50)
        logger.info("DURUM: INIT - Sistem Başlatılıyor")
        logger.info("=" * 50)

        # 1. Orange Cube+ bağlantısı
        conn_str = config.SITL_CONNECTION if self.use_sitl else config.CONNECTION_STRING
        if not self.navigation.connect(conn_str):
            logger.error("Uçuş kontrolcüsü bağlantısı başarısız!")
            self._transition_to(MissionState.ERROR)
            return

        # 2. Payload manager'a vehicle bağla
        self.payload.set_vehicle(self.navigation.vehicle)

        # 3. GPS fix kontrolü
        gps_fix = self.navigation.get_gps_fix()
        if gps_fix is None or gps_fix < 3:
            logger.warning(f"GPS fix yetersiz: {gps_fix} (minimum 3D fix gerekli)")
            # Bekleme döngüsü
            for _ in range(60):  # 30 saniye bekle
                gps_fix = self.navigation.get_gps_fix()
                if gps_fix and gps_fix >= 3:
                    break
                time.sleep(0.5)
            else:
                logger.error("GPS 3D fix alınamadı!")
                self._transition_to(MissionState.ERROR)
                return

        logger.info(f"✓ GPS fix: {gps_fix}")

        # 4. Kamera başlatma
        if self.use_camera:
            if not self.vision.start_camera():
                logger.error("Kamera başlatılamadı!")
                self._transition_to(MissionState.ERROR)
                return
            logger.info("✓ Kamera başlatıldı")

        # 5. Servo test (nötr pozisyon)
        self.payload.reset_servo()
        logger.info("✓ Servo nötr pozisyonda")

        # 6. Batarya kontrolü
        battery = self.navigation.get_battery()
        if battery:
            logger.info(
                f"✓ Batarya: {battery['voltage']:.1f}V, "
                f"Seviye: {battery['level']}%"
            )

        logger.info("✓ Tüm sistemler hazır")
        self._transition_to(MissionState.SEARCH)











    def _handle_search(self):
        """
        SEARCH: Görüntü işleme ile hedef arama.

        Süre kısıtlaması kaldırıldı. İHA zaten Mission Planner üzerinden
        AUTO modda rotasını geziyor. Sistem sadece kamerayı izler,
        hem mavi hem de kırmızı hedefler için yeterli örnek (minimum_detection_samples)
        toplandığı anda direkt PROCESS aşamasına geçer.
        """
        now = time.time()

        # İlk kez SEARCH'e girildiğinde buffer'ları temizle ve başlangıç ayarlarını yap
        if not self._search_window_started:
            self._search_window_started = True
            self.state_start_time = now
            self._last_detection_sample_time = 0.0
            self._blue_detections_buffer.clear()
            self._red_detections_buffer.clear()

            logger.info("=" * 50)
            logger.info("DURUM: SEARCH - Süresiz Serbest Hedef Arama")
            logger.info("=" * 50)
            logger.info(
                f"Örnekleme aralığı: {self.detection_sample_interval:.2f}s, "
                f"Minimum örnek sayısı: {self.minimum_detection_samples}"
            )

        # Belirlenen örnekleme aralığı dolmadıysa yeni kare işlemeye gerek yok
        if (
            self._last_detection_sample_time > 0.0
            and now - self._last_detection_sample_time < self.detection_sample_interval
        ):
            return

        self._last_detection_sample_time = now

        # Kamera karesi yakala
        if self.use_camera:
            frame = self.vision.capture_frame()
        else:
            # Test modu: sentetik kare
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if frame is None:
            return

        # Tespitler
        best = self.vision.get_best_detections(frame)

        # İHA'nın mevcut konumu ve durumu
        location = self.navigation.get_location()
        attitude = self.navigation.get_attitude()

        if location is None or attitude is None:
            return

        uav_lat, uav_lon, uav_alt = location
        roll, pitch, yaw = attitude

        # Mavi bölge tespit edildiyse
        if best['blue'] is not None:
            blue_det = best['blue']
            gps = self.coord_transform.pixel_to_gps(
                blue_det.center_pixel[0], blue_det.center_pixel[1],
                uav_lat, uav_lon, uav_alt,
                roll, pitch, yaw
            )
            if gps:
                self._blue_detections_buffer.append(gps)
                logger.info(
                    f"MAVİ tespit #{len(self._blue_detections_buffer)}: "
                    f"GPS=({gps[0]:.6f}, {gps[1]:.6f}), "
                    f"Güven={blue_det.confidence:.1f}%"
                )

        # Kırmızı bölge tespit edildiyse
        if best['red'] is not None:
            red_det = best['red']
            gps = self.coord_transform.pixel_to_gps(
                red_det.center_pixel[0], red_det.center_pixel[1],
                uav_lat, uav_lon, uav_alt,
                roll, pitch, yaw
            )
            if gps:
                self._red_detections_buffer.append(gps)
                logger.info(
                    f"KIRMIZI tespit #{len(self._red_detections_buffer)}: "
                    f"GPS=({gps[0]:.6f}, {gps[1]:.6f}), "
                    f"Güven={red_det.confidence:.1f}%"
                )

        # Debug görüntü kaydet
        if best['blue'] or best['red']:
            detections = self.vision.detect_all(frame)
            annotated = self.vision.draw_detections(frame, detections)
            self.vision.save_debug_image(annotated, "search")

        # Toplanan örnek sayıları
        blue_count = len(self._blue_detections_buffer)
        red_count = len(self._red_detections_buffer)

        # Yeterli sayıya ulaşıldı mı?
        blue_ready = blue_count >= self.minimum_detection_samples
        red_ready = red_count >= self.minimum_detection_samples

        # İkisi de yeterliyse süreyi beklemeden direkt fırla
        if blue_ready and red_ready:
            logger.info("=" * 50)
            logger.info(
                f"✓ HEDEFLER BULUNDU! Mavi={blue_count}/{self.minimum_detection_samples}, "
                f"Kırmızı={red_count}/{self.minimum_detection_samples}"
            )
            logger.info("Arama tamamlandı, işleme (PROCESS) geçiliyor.")
            logger.info("=" * 50)
            
            self._search_window_started = False
            self._transition_to(MissionState.PROCESS)
        else:
            # Çok fazla log spamı yapmamak için 5 saniyede bir durumu bildir
            if int(now) % 5 == 0 and int(now) != int(now - self.detection_sample_interval):
                logger.info(
                    f"Arama sürüyor... Eksik tespitler -> "
                    f"Mavi: {blue_count}/{self.minimum_detection_samples} | "
                    f"Kırmızı: {red_count}/{self.minimum_detection_samples}"
                )












    def _handle_process(self):
        """
        PROCESS: Toplanan tespitleri değerlendir, ortalama GPS hesapla,
        yeni misyon oluştur.
        """
        logger.info("=" * 50)
        logger.info("DURUM: PROCESS - Tespitler Değerlendiriliyor")
        logger.info("=" * 50)

        # Mavi bölge: Tespitlerin ortalamasını al
        if self._blue_detections_buffer:
            blue_lat = sum(d[0] for d in self._blue_detections_buffer) / len(
                self._blue_detections_buffer
            )
            blue_lon = sum(d[1] for d in self._blue_detections_buffer) / len(
                self._blue_detections_buffer
            )
            self.blue_targets.append((blue_lat, blue_lon))
            logger.info(
                f"MAVİ hedef GPS (ortalamalı): ({blue_lat:.6f}, {blue_lon:.6f})"
            )

        # Kırmızı bölge: Tespitlerin ortalamasını al
        if self._red_detections_buffer:
            red_lat = sum(d[0] for d in self._red_detections_buffer) / len(
                self._red_detections_buffer
            )
            red_lon = sum(d[1] for d in self._red_detections_buffer) / len(
                self._red_detections_buffer
            )
            self.red_targets.append((red_lat, red_lon))
            logger.info(
                f"KIRMIZI hedef GPS (ortalamalı): ({red_lat:.6f}, {red_lon:.6f})"
            )

        # Hedefler var mı kontrol et
        if not self.blue_targets:
            logger.warning("Mavi hedef bulunamadı, aramaya dönülüyor")
            self._blue_detections_buffer.clear()
            self._search_window_started = False
            self._transition_to(MissionState.SEARCH)
            return

        if not self.red_targets:
            logger.warning("Kırmızı hedef bulunamadı, aramaya dönülüyor")
            self._red_detections_buffer.clear()
            self._search_window_started = False
            self._transition_to(MissionState.SEARCH)
            return

        # Yeni misyon oluştur (mavi → kırmızı → RTL)
        logger.info("Yük bırakma misyonu oluşturuluyor...")
        success = self.navigation.create_mission(
            blue_targets=self.blue_targets,
            red_targets=self.red_targets,
        )

        if not success:
            logger.error("Misyon oluşturulamadı!")
            self._transition_to(MissionState.ERROR)
            return

        # AUTO moda geç ve misyonu başlat
        self.navigation.set_mode("AUTO")

        logger.info("✓ Misyon yüklendi, mavi hedeflere yönleniliyor")
        self._transition_to(MissionState.NAVIGATE_BLUE)




        

    def _handle_navigate_blue(self):
        """
        NAVIGATE_BLUE: Mavi bölgeye doğru uçuş.
        """
        if self.state_start_time is None:
            self.state_start_time = time.time()
            logger.info("=" * 50)
            logger.info("DURUM: NAVIGATE_BLUE - Mavi Hedefe Uçuş")
            logger.info("=" * 50)

        if not self.blue_targets:
            self._transition_to(MissionState.NAVIGATE_RED)
            return

        target_lat, target_lon = self.blue_targets[0]

        # Hedefe olan mesafe
        distance = self.navigation.get_distance_to(target_lat, target_lon)

        # Periyodik bilgi
        if int(time.time()) % 3 == 0:
            logger.info(f"  Mavi hedefe mesafe: {distance:.1f}m")

        # Waypoint'e ulaşıldı mı?
        if distance <= config.WAYPOINT_REACHED_RADIUS:
            logger.info(f"✓ Mavi hedefe ulaşıldı! Mesafe: {distance:.1f}m")
            self._transition_to(MissionState.DROP_BLUE)

    def _handle_drop_blue(self):
        """
        DROP_BLUE: Mavi bölgede yük 1 bırakma.
        """
        logger.info("=" * 50)
        logger.info("DURUM: DROP_BLUE - Yük 1 Bırakılıyor")
        logger.info("=" * 50)

        success = self.payload.drop_payload_1()

        if success:
            logger.info("✓ Yük 1 başarıyla bırakıldı!")
        else:
            logger.error("✗ Yük 1 bırakma başarısız!")

        # Kırmızı bölgeye geç
        self._transition_to(MissionState.NAVIGATE_RED)

    def _handle_navigate_red(self):
        """
        NAVIGATE_RED: Kırmızı bölgeye doğru uçuş.
        """
        if self.state_start_time is None:
            self.state_start_time = time.time()
            logger.info("=" * 50)
            logger.info("DURUM: NAVIGATE_RED - Kırmızı Hedefe Uçuş")
            logger.info("=" * 50)

        if not self.red_targets:
            self._transition_to(MissionState.RTL)
            return

        target_lat, target_lon = self.red_targets[0]

        # Hedefe olan mesafe
        distance = self.navigation.get_distance_to(target_lat, target_lon)

        # Periyodik bilgi
        if int(time.time()) % 3 == 0:
            logger.info(f"  Kırmızı hedefe mesafe: {distance:.1f}m")

        # Waypoint'e ulaşıldı mı?
        if distance <= config.WAYPOINT_REACHED_RADIUS:
            logger.info(f"✓ Kırmızı hedefe ulaşıldı! Mesafe: {distance:.1f}m")
            self._transition_to(MissionState.DROP_RED)

    def _handle_drop_red(self):
        """
        DROP_RED: Kırmızı bölgede yük 2 bırakma.
        """
        logger.info("=" * 50)
        logger.info("DURUM: DROP_RED - Yük 2 Bırakılıyor")
        logger.info("=" * 50)

        success = self.payload.drop_payload_2()

        if success:
            logger.info("✓ Yük 2 başarıyla bırakıldı!")
        else:
            logger.error("✗ Yük 2 bırakma başarısız!")

        # Eve dönüş
        self._transition_to(MissionState.RTL)

    def _handle_rtl(self):
        """
        RTL: Eve dönüş.
        """
        logger.info("=" * 50)
        logger.info("DURUM: RTL - Eve Dönüş")
        logger.info("=" * 50)

        self.navigation.set_mode("RTL")
        logger.info("RTL modu aktif, eve dönülüyor...")

        # Eve dönüşü bekle (basit kontrol)
        for _ in range(600):  # 5 dakika bekle
            if self._shutdown_requested:
                break

            loc = self.navigation.get_location()
            if loc:
                distance = self.navigation.get_distance_to(
                    config.HOME_LAT, config.HOME_LON
                )
                if distance < 50 and loc[2] < 5:
                    logger.info("✓ Eve dönüş tamamlandı!")
                    break

                if int(time.time()) % 10 == 0:
                    logger.info(
                        f"  Eve mesafe: {distance:.1f}m, İrtifa: {loc[2]:.1f}m"
                    )

            time.sleep(0.5)

        self._transition_to(MissionState.COMPLETE)

    def _handle_error(self):
        """
        ERROR: Hata durumu - güvenli RTL.
        """
        logger.error("=" * 50)
        logger.error("DURUM: ERROR - Hata Oluştu!")
        logger.error("=" * 50)

        # Acil RTL dene
        if self.navigation.is_connected:
            self.navigation.emergency_rtl()

        logger.error(
            f"Hata oluştu. Önceki durum: {self.previous_state}"
        )

        # 10 saniye bekle, sonra tamamla
        time.sleep(10)
        self._transition_to(MissionState.COMPLETE)

    def _handle_complete(self):
        """
        COMPLETE: Görev tamamlandı.
        """
        elapsed = time.time() - self.mission_start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        logger.info("=" * 60)
        logger.info("GÖREV TAMAMLANDI!")
        logger.info("=" * 60)
        logger.info(f"  Toplam süre: {minutes}dk {seconds}sn")

        # Yük durumları
        payload_status = self.payload.get_status()
        logger.info(f"  Yük 1 (Mavi): {payload_status['payload_1']}")
        logger.info(f"  Yük 2 (Kırmızı): {payload_status['payload_2']}")

        # Hedef bilgileri
        if self.blue_targets:
            logger.info(f"  Mavi hedef: ({self.blue_targets[0][0]:.6f}, "
                         f"{self.blue_targets[0][1]:.6f})")
        if self.red_targets:
            logger.info(f"  Kırmızı hedef: ({self.red_targets[0][0]:.6f}, "
                         f"{self.red_targets[0][1]:.6f})")

        logger.info("=" * 60)

    # =========================================================================
    # GÜVENLİK
    # =========================================================================

    def _safety_check(self) -> bool:
        """
        Güvenlik kontrollerini yapar.

        Returns:
            Güvenli ise True, failsafe gerekiyorsa False
        """
        if not self.navigation.is_connected:
            return True  # Bağlantı yoksa kontrol yapamayız

        # Batarya kontrolü
        if self.navigation.check_battery_failsafe():
            logger.critical("⚠ BATARYA FAILSAFE!")
            return False

        return True

    # =========================================================================
    # TEMİZLİK
    # =========================================================================

    def _cleanup(self):
        """Tüm kaynakları serbest bırakır."""
        logger.info("Sistem kapatılıyor...")

        # Kamera kapat
        if self.use_camera:
            self.vision.stop_camera()

        # Servo nötre getir
        self.payload.reset_servo()

        # Bağlantı kapat
        self.navigation.disconnect()

        logger.info("✓ Sistem kapatıldı")