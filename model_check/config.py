"""L1 检测核心基础配置。"""

from __future__ import annotations

from dataclasses import dataclass, field


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
