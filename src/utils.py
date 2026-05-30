import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_path: Path, logger_name: str):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 防止重复 handler
    if logger.handlers:
        logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # ======================================================
    # ⭐关键：写入本次运行时间 header
    # ======================================================

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info(f"LOG START TIME: {start_time}")
    logger.info("=" * 60)

    return logger
