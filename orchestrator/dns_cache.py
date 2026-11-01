"""Process-wide DNS cache: serve-stale-while-revalidate, stale-if-error.

The host resolver (systemd-resolved behind Tailscale) intermittently returns
EAI_AGAIN, which surfaced as "Connection error" turns. After install(), a
lookup is answered from memory; expired entries refresh in the background and
the last good answer is reused when the resolver fails.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

TTL_S = 120.0
FIRST_LOOKUP_TRIES = 3

_real_getaddrinfo: Callable[..., Any] = socket.getaddrinfo
_cache: dict[tuple, tuple[float, Any]] = {}
_refreshing: set[tuple] = set()
_lock = threading.Lock()
_installed = False


def _resolve(key: tuple) -> Any:
    host, port, args, kwargs = key[0], key[1], key[2], dict(key[3])
    return _real_getaddrinfo(host, port, *args, **kwargs)


def _refresh(key: tuple) -> None:
    try:
        result = _resolve(key)
        with _lock:
            _cache[key] = (time.monotonic(), result)
    except OSError as e:
        logger.warning("DNS refresh failed for %s (serving stale): %s", key[0], e)
    finally:
        with _lock:
            _refreshing.discard(key)


def cached_getaddrinfo(host, port, *args, **kwargs):
    key = (host, port, args, tuple(sorted(kwargs.items())))
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        stamp, result = hit
        if time.monotonic() - stamp > TTL_S:
            with _lock:
                start = key not in _refreshing
                _refreshing.add(key)
            if start:
                threading.Thread(target=_refresh, args=(key,), daemon=True).start()
        return result
    last: OSError | None = None
    for attempt in range(FIRST_LOOKUP_TRIES):
        try:
            result = _resolve(key)
            with _lock:
                _cache[key] = (time.monotonic(), result)
            return result
        except socket.gaierror as e:
            last = e
            if e.errno != socket.EAI_AGAIN:
                raise
            time.sleep(0.1 * (attempt + 1))
    assert last is not None
    raise last


def install() -> None:
    global _installed
    if not _installed:
        socket.getaddrinfo = cached_getaddrinfo
        _installed = True


def prewarm(host: str, port: int = 443) -> None:
    """Resolve now so the first real request does no DNS work."""
    try:
        socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)  # as socket.create_connection
    except OSError as e:
        logger.warning("DNS prewarm failed for %s: %s", host, e)
