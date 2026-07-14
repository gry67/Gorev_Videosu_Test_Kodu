from enum import Enum

class PayloadState(Enum):
    """Faydalı yük durumları."""
    LOADED = "loaded"       # Yük takılı
    DROPPED = "dropped"     # Yük bırakıldı
    ERROR = "error"         # Hata