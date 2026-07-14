import Camera as cam
import config as conf
import NavigationManager as nav
import PayloadManager as payloman
import MissionController as miscon
import dronekit as dronekit

def main():

    gorev_mekanizmasi = miscon.MissionController(use_sitl=True)

    gorev_mekanizmasi.run()

if __name__ == "__main__":
    main()