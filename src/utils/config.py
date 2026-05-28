"""Load YAML config with environment variable substitution."""

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

_ROOT = Path(__file__).resolve().parents[2]


def _expand(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} in strings; cast int/float env vars."""
    if isinstance(obj, str):
        expanded = os.path.expandvars(obj)
        # Leave unexpanded markers visible so misconfiguration is obvious.
        if expanded.startswith("$"):
            logger.warning("unresolved_env_var", extra={"raw": obj})
        try:
            return int(expanded)
        except (ValueError, TypeError):
            pass
        try:
            return float(expanded)
        except (ValueError, TypeError):
            pass
        return expanded
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(i) for i in obj]
    return obj


def load_config(env: str = "dev") -> dict:
    """Load config/{env}.yaml with all ${VAR} references resolved."""
    config_path = _ROOT / "config" / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open() as f:
        raw = yaml.safe_load(f)

    config = _expand(raw)
    logger.info("config_loaded", extra={"env": env, "path": str(config_path)})
    return config
