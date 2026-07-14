# -*- coding: utf-8 -*-
"""
Teknofest İHA - Faydalı Yük Bırakma Modülü
=============================================
MAVLink üzerinden servo kontrol ile faydalı yük bırakma.

Çalışma prensibi:
- Tek servo, iki yönde döner
- Bir yöne dönüş → Yük 1 (mavi bölge) bırakılır
- Diğer yöne dönüş → Yük 2 (kırmızı bölge) bırakılır
- Mekanik mekanizma takım tarafından sağlanır
"""

import time
from enum import Enum
from typing import Optional

from pymavlink import mavutil
from Models.PayloadState import *
import config
from logger_config import setup_logger

logger = setup_logger("payload")





class PayloadManager:
    """
    Servo kontrol ile faydalı yük bırakma yöneticisi.

    Servo PWM değerleri:
    - Nötr (1500µs): Her iki yük de tutuluyor
    - Yük 1 (1100µs): Servo bir yöne döner, yük 1 bırakılır
    - Yük 2 (1900µs): Servo diğer yöne döner, yük 2 bırakılır
    """

    def __init__(self, vehicle=None):
        """
        Args:
            vehicle: DroneKit vehicle nesnesi
        """
        self.vehicle = vehicle
        self.servo_channel = config.SERVO_CHANNEL

        # Yük durumları
        self.payload_1_state = PayloadState.LOADED
        self.payload_2_state = PayloadState.LOADED

        # Mevcut servo PWM değeri
        self.current_pwm = config.SERVO_NEUTRAL_PWM

        logger.info(
            f"PayloadManager başlatıldı - "
            f"Servo Kanal: {self.servo_channel}, "
            f"Nötr: {config.SERVO_NEUTRAL_PWM}µs"
        )

    def set_vehicle(self, vehicle):
        """
        DroneKit vehicle nesnesini ayarlar.

        Args:
            vehicle: DroneKit vehicle nesnesi
        """
        self.vehicle = vehicle
        logger.info("Vehicle bağlantısı ayarlandı")

    def _send_servo_command(self, channel: int, pwm: int) -> bool:
        """
        MAVLink üzerinden servo PWM komutu gönderir.

        Args:
            channel: Servo kanal numarası
            pwm: PWM değeri (µs)

        Returns:
            Başarılı ise True
        """
        if self.vehicle is None:
            logger.error("Vehicle bağlantısı yok! Servo komutu gönderilemedi.")
            return False

        try:
            msg = self.vehicle.message_factory.command_long_encode(
                0, 0,                                      # target_system, target_component
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,      # command
                0,                                         # confirmation
                channel,                                   # param1: servo numarası
                pwm,                                       # param2: PWM değeri
                0, 0, 0, 0, 0                              # param3-7: kullanılmıyor
            )
            self.vehicle.send_mavlink(msg)
            self.vehicle.flush()
            self.current_pwm = pwm

            logger.info(f"Servo komutu gönderildi: Kanal={channel}, PWM={pwm}µs")
            return True

        except Exception as e:
            logger.error(f"Servo komutu gönderme hatası: {e}")
            return False

    def reset_servo(self) -> bool:
        """
        Servoyu nötr pozisyona getirir.

        Returns:
            Başarılı ise True
        """
        logger.info("Servo nötr pozisyona getiriliyor...")
        success = self._send_servo_command(
            self.servo_channel,
            config.SERVO_NEUTRAL_PWM
        )
        if success:
            logger.info("Servo nötr pozisyonda")
        return success

    def drop_payload_1(self) -> bool:
        """
        Yük 1'i bırakır (mavi bölge için).
        Servo bir yöne döner (1100µs).

        Returns:
            Başarılı ise True
        """
        if self.payload_1_state == PayloadState.DROPPED:
            logger.warning("Yük 1 zaten bırakılmış!")
            return True

        logger.info("=" * 40)
        logger.info("YÜK 1 BIRAKILIYOR (Mavi Bölge)")
        logger.info("=" * 40)

        success = self._send_servo_command(
            self.servo_channel,
            config.SERVO_PAYLOAD_1_PWM
        )

        if success:
            # Servonun mekanik olarak hareket etmesini bekle
            logger.info(
                f"Servo hareketi bekleniyor ({config.SERVO_ACTION_DELAY}s)..."
            )
            time.sleep(config.SERVO_ACTION_DELAY)

            self.payload_1_state = PayloadState.DROPPED
            logger.info("✓ Yük 1 başarıyla bırakıldı!")

            # Servoyu nötre geri getir (ikinci yük için hazır)
            self.reset_servo()
            time.sleep(0.5)
        else:
            self.payload_1_state = PayloadState.ERROR
            logger.error("✗ Yük 1 bırakma BAŞARISIZ!")

        return success

    def drop_payload_2(self) -> bool:
        """
        Yük 2'yi bırakır (kırmızı bölge için).
        Servo diğer yöne döner (1900µs).

        Returns:
            Başarılı ise True
        """
        if self.payload_2_state == PayloadState.DROPPED:
            logger.warning("Yük 2 zaten bırakılmış!")
            return True

        logger.info("=" * 40)
        logger.info("YÜK 2 BIRAKILIYOR (Kırmızı Bölge)")
        logger.info("=" * 40)

        success = self._send_servo_command(
            self.servo_channel,
            config.SERVO_PAYLOAD_2_PWM
        )

        if success:
            # Servonun mekanik olarak hareket etmesini bekle
            logger.info(
                f"Servo hareketi bekleniyor ({config.SERVO_ACTION_DELAY}s)..."
            )
            time.sleep(config.SERVO_ACTION_DELAY)

            self.payload_2_state = PayloadState.DROPPED
            logger.info("✓ Yük 2 başarıyla bırakıldı!")

            # Servoyu nötre geri getir
            self.reset_servo()
            time.sleep(0.5)
        else:
            self.payload_2_state = PayloadState.ERROR
            logger.error("✗ Yük 2 bırakma BAŞARISIZ!")

        return success

    def get_status(self) -> dict:
        """
        Yük durumlarını döndürür.

        Returns:
            Durum bilgileri sözlüğü
        """
        return {
            'payload_1': self.payload_1_state.value,
            'payload_2': self.payload_2_state.value,
            'current_pwm': self.current_pwm,
            'servo_channel': self.servo_channel,
            'all_dropped': (
                self.payload_1_state == PayloadState.DROPPED
                and self.payload_2_state == PayloadState.DROPPED
            ),
        }

    def is_payload_1_loaded(self) -> bool:
        """Yük 1 hâlâ yüklü mü?"""
        return self.payload_1_state == PayloadState.LOADED

    def is_payload_2_loaded(self) -> bool:
        """Yük 2 hâlâ yüklü mü?"""
        return self.payload_2_state == PayloadState.LOADED






















# =============================================================================
# BAĞIMSIZ TEST
# =============================================================================
if __name__ == "__main__":
    """
    Test: Servo komutlarını simüle eder (vehicle olmadan).
    """
    print("=" * 60)
    print("PayloadManager Testi (Simülasyon)")
    print("=" * 60)

    pm = PayloadManager(vehicle=None)

    print(f"\nBaşlangıç durumu:")
    status = pm.get_status()
    print(f"  Yük 1: {status['payload_1']}")
    print(f"  Yük 2: {status['payload_2']}")
    print(f"  Servo PWM: {status['current_pwm']}µs")

    print(f"\n--- Vehicle bağlantısı olmadan test ---")
    print("Yük 1 bırakma denemesi...")
    result = pm.drop_payload_1()
    print(f"  Sonuç: {'Başarılı' if result else 'Başarısız (beklenen - vehicle yok)'}")

    print(f"\nSon durum:")
    status = pm.get_status()
    print(f"  Yük 1: {status['payload_1']}")
    print(f"  Yük 2: {status['payload_2']}")
    print(f"  Tüm yükler bırakıldı mı: {status['all_dropped']}")

    print("\nTest tamamlandı.")
    print("NOT: Gerçek test için DroneKit vehicle bağlantısı gereklidir.")
