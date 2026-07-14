# -*- coding: utf-8 -*-
"""
Teknofest İHA - Loglama Yapılandırması
=======================================
Tüm modüller için merkezi loglama sistemi.
Konsol ve dosya çıktıları desteklenir.
"""

import os
import logging
from datetime import datetime

import config


def setup_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Belirtilen isimle bir logger oluşturur.

    Args:
        name: Logger ismi (genellikle modül adı)
        log_to_file: Dosyaya da log yazılsın mı

    Returns:
        Yapılandırılmış logger nesnesi
    """
    logger = logging.getLogger(name)

    # Zaten handler eklenmişse tekrar ekleme
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # Konsol handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Dosya handler
    if log_to_file:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(config.LOG_DIR, f"{name}_{timestamp}.log")

        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            config.LOG_FORMAT,
            datefmt=config.LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
