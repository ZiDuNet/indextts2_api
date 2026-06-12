"""
Network detection utility for determining whether the current network
environment needs a proxy to access HuggingFace, to decide whether to
use ModelScope for model downloads.
"""

import os
import socket
import time
import logging

logger = logging.getLogger(__name__)

# Cache the detection result so we only check once per process
_detection_cache = None


def _tcp_latency(host: str, port: int = 443, timeout: float = 3.0):
    """TCP handshake latency in seconds, or None if unreachable."""
    try:
        start = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=timeout)
        latency = time.perf_counter() - start
        sock.close()
        return latency
    except (socket.timeout, socket.error, OSError):
        return None


def need_proxy(timeout: float = 3.0) -> bool:
    """
    Detect if the current network environment needs a proxy to access HF.

    Returns True if a proxy is needed (use ModelScope / hf-mirror),
    False otherwise.

    Detection methods (in order):
    1. Check environment variable ``USE_MODELSCOPE`` for manual override
    2. Default to ModelScope (USE_MODELSCOPE=true)

    The result is cached after the first call so subsequent calls are instant.
    """
    global _detection_cache
    if _detection_cache is not None:
        return _detection_cache

    # Allow manual override via environment variable
    env_override = os.environ.get("USE_MODELSCOPE", "").lower()
    if env_override == "false":
        logger.info("Network detection: forced to direct mode (USE_MODELSCOPE=false)")
        _detection_cache = False
        return False

    # Default to ModelScope
    logger.info("Network detection: using ModelScope as primary source")
    _detection_cache = True
    return True
