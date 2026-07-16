import Camera as cam
import config as conf
import NavigationManager as nav
import PayloadManager as payload
import MissionController as miscon
import dronekit as dronekit
from vision import VisionProcessor
import time

BLUE_PWM = 1650
RED_PWM = 2200
SERVO_CHANNEL = 6

def main():

    # Pixhawk bağlantısı
    navigation = nav.NavigationManager()

    if not navigation.connect():
        print("Pixhawk bağlantısı başarısız.")
        return

    # Servo yöneticisi
    payload_manager = payload.PayloadManager(navigation.vehicle)

    # Kamera
    vision = VisionProcessor()

    if not vision.start_camera():
        print("Kamera açılamadı.")
        return

    blue_done = False
    red_done = False

    print("Renk bekleniyor...")

    try:

        while True:

            frame = vision.capture_frame()

            if frame is None:
                continue

            detections = vision.get_best_detections(frame)

            # MAVİ
            if detections["blue"] is not None and not blue_done:

                print("Mavi görüldü -> Servo 1650")

                payload_manager._send_servo_command(
                    SERVO_CHANNEL,
                    BLUE_PWM
                )

                blue_done = True

            # KIRMIZI
            elif detections["red"] is not None and not red_done:

                print("Kırmızı görüldü -> Servo 2200")

                payload_manager._send_servo_command(
                    SERVO_CHANNEL,
                    RED_PWM
                )

                red_done = True

            if blue_done and red_done:
                print("İki renk de test edildi.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        pass

    finally:
        vision.stop_camera()
        navigation.disconnect()

if __name__ == "__main__":
    main()