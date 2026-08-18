"""
logger.py
---------
Simple logging setup for Paddy Yield Predictor.
Every notebook and script imports get_logger() from here.
Logs go to both the console AND a file inside the /logs folder.
"""

import logging
import os
from pathlib import Path
from datetime import datetime

# Figure out where the project root is
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"

# Create logs folder if it doesn't exist yet
LOGS_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to:
      - Console (so you see logs in the notebook output)
      - logs/paddy_YYYY-MM-DD.log (so logs are saved to disk)

    Usage:
        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("Data loaded successfully")
    """

    logger = logging.getLogger(name)

    # Don't add handlers if this logger already has them
    # (avoids duplicate log lines when cells are re-run)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format: timestamp | level | logger name | message
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console handler (shows in notebook) ---
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)   # INFO and above in console
    console.setFormatter(fmt)
    logger.addHandler(console)

    # --- File handler (saves everything to disk) ---
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"paddy_{today}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)   # DEBUG and above in file
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
