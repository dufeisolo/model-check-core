"""L1 检测核心基础配置与 Python 3.10 兼容层。"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import sys


@dataclass(frozen=True)
class Settings:
    CONCURRENCY: int = 2
    TIMEOUT_S: float = 60.0
    L1_REQUEST_TIMEOUT_S: float = 30.0
    L1_PROBE_TIMEOUT_S: float = 30.0
    L1_PROTOCOL_TIMEOUT_S: float = 30.0
    L1_TOTAL_TIMEOUT_S: float = 180.0
    MAX_UPSTREAM_RESPONSE_BYTES: int = 2_000_000
    RETRIES: int = 2
    RETRY_BACKOFF: tuple = (1.0, 4.0)
    MIN_ARTIFACT_LEN: int = 32
    PARAM_MATRIX_REPEAT: int = 3
    CACHE_PREFIX_MIN: dict = field(default_factory=lambda: {"3.5": 1024, "4": 4096})


_DEFAULT_SETTINGS = Settings()


def get_settings() -> Settings:
    return _DEFAULT_SETTINGS


if sys.version_info >= (3, 11):
    async_timeout = asyncio.timeout
else:
    @asynccontextmanager
    async def async_timeout(delay: float):
        """Python 3.10 兼容实现的 asyncio.timeout。"""
        if delay is None:
            yield
            return
        try:
            loop = asyncio.get_running_loop()
            time_fn = getattr(loop, "time", None)
            call_at_fn = getattr(loop, "call_at", None)
            if time_fn is None or call_at_fn is None:
                yield
                return
            deadline = time_fn() + delay
            task = asyncio.current_task()
            handle = call_at_fn(deadline, task.cancel)
            try:
                yield
            except asyncio.CancelledError:
                if time_fn() >= deadline:
                    raise TimeoutError() from None
                raise
            finally:
                handle.cancel()
        except Exception:
            yield

