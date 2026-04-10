"""
Structured Logging System — Antidote AI
Logs poisoning detections, evasion attempts, and final decisions to rotating files.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)


def _get_logger(name: str, filename: str) -> logging.Logger:
    """Create or retrieve a named logger with a rotating file handler."""
    logger = logging.getLogger(f"antidote.{name}")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            os.path.join(LOG_DIR, filename),
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ── Loggers ──────────────────────────────────────────────────────
_poisoning_logger = _get_logger("poisoning", "poisoning.log")
_evasion_logger = _get_logger("evasion", "evasion.log")
_decision_logger = _get_logger("decisions", "decisions.log")


def log_poisoning(total_rows: int, suspicious_rows: int, cleaned_rows: int, filename: str = ""):
    """Log a poisoning detection event."""
    _poisoning_logger.info(
        "file=%s | total=%d | suspicious=%d | cleaned=%d",
        filename, total_rows, suspicious_rows, cleaned_rows,
    )


def log_evasion(input_summary: str, evasion_flag: bool, decision_score: float):
    """Log an evasion detection event."""
    _evasion_logger.info(
        "input=%s | evasion_flag=%s | score=%.4f",
        input_summary, evasion_flag, decision_score,
    )


def log_decision(
    input_summary: str,
    decision: str,
    risk_score: int,
    severity: str,
    drift_flag: bool = False,
    explanation: str = "",
):
    """Log a final pipeline decision."""
    _decision_logger.info(
        "input=%s | decision=%s | risk=%d | severity=%s | drift=%s | explanation=%s",
        input_summary, decision, risk_score, severity, drift_flag, explanation,
    )
