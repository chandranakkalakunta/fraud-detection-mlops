"""Structured logging with Google Cloud Logging integration.

- GCP environment (GOOGLE_CLOUD_PROJECT set): CloudLoggingHandler → Cloud Logging API,
  log_name="fraud-detection-mlops", with service/version/environment labels.
- Local development: JSON to stdout in Cloud Logging-compatible format.
"""

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

# Standard labels attached to every log record.
_LABELS = {
    "component": os.getenv("LOG_COMPONENT", "fraud-detection-mlops"),
    "service": os.getenv("SERVICE_NAME", "fraud-detection-api"),
    "version": os.getenv("MODEL_VERSION", "lgb-v8-no-txnid"),
    "environment": os.getenv("ENV", "dev"),
}


class _CloudFormatter(pythonjson.JsonFormatter):
    """Remaps standard fields to Cloud Logging structured log format."""

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = _SEVERITY_MAP.get(record.levelno, "DEFAULT")
        log_record.pop("levelname", None)
        log_record.pop("asctime", None)
        log_record["logging.googleapis.com/labels"] = _LABELS


def get_logger(name: str) -> logging.Logger:
    """Return a structured logger for the given module name.

    In GCP (GOOGLE_CLOUD_PROJECT set): CloudLoggingHandler with background
    transport — non-blocking, sends directly to Cloud Logging API.
    Locally: JSON stdout handler in Cloud Logging-compatible format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.DEBUG if os.getenv("DEBUG") else logging.INFO
    project = os.getenv("GOOGLE_CLOUD_PROJECT")

    if project:
        try:
            import google.cloud.logging as cloud_logging
            from google.cloud.logging.handlers import CloudLoggingHandler

            client = cloud_logging.Client(project=project)
            handler = CloudLoggingHandler(
                client,
                name="fraud-detection-mlops",
                labels=_LABELS,
            )
            logger.addHandler(handler)
            logger.setLevel(level)
            logger.propagate = False
            return logger
        except Exception:
            pass  # fall through to stdout on any import or auth failure

    # Local / fallback: structured JSON to stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = _CloudFormatter(
        fmt="%(message)s %(severity)s %(name)s %(filename)s %(lineno)d",
        timestamp=True,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
