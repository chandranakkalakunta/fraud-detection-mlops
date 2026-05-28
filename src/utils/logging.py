"""Structured JSON logging compatible with Google Cloud Logging."""

import logging
import os
import sys
from pythonjsonlogger import json as pythonjson


_SEVERITY_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class _CloudFormatter(pythonjson.JsonFormatter):
    """Remaps standard fields to Cloud Logging structured log format."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = _SEVERITY_MAP.get(record.levelno, "DEFAULT")
        log_record.pop("levelname", None)
        log_record.pop("asctime", None)
        component = os.getenv("LOG_COMPONENT", "fraud-detection-mlops")
        log_record["logging.googleapis.com/labels"] = {
            "component": component,
            "environment": os.getenv("ENV", "dev"),
        }


def get_logger(name: str) -> logging.Logger:
    """Return a structured JSON logger for Cloud Logging."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = _CloudFormatter(
        fmt="%(message)s %(severity)s %(name)s %(filename)s %(lineno)d",
        timestamp=True,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)
    logger.propagate = False
    return logger
