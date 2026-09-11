"""Model Check Core: 大模型 API 渠道 L1 极速体检核心。"""

from model_check.clients import ApiClient, ProtocolError
from model_check.config import Settings, get_settings
from model_check.model_profiles import (
    ModelProfile,
    identity_probe_ids,
    mandatory_probe_ids,
    model_echo_matches,
    probe_applicability,
    resolve_model_profile,
)
from model_check.probes import ProbeResult, probe_summary, run_probes
from model_check.runner import L1DetectionRequest, run_l1_detection
from model_check.security import (
    PublicOnlyTransport,
    ResponseTooLarge,
    UnsafeTargetError,
    normalize_public_base_url,
    resolve_public_ips,
    validate_public_target,
)
from model_check.verdict import (
    L1_HIGH,
    L1_MIN_COVERAGE,
    decide_l1,
    decide_l1_public,
)

__version__ = "0.1.0"

__all__ = [
    "ApiClient",
    "ProtocolError",
    "Settings",
    "get_settings",
    "ModelProfile",
    "resolve_model_profile",
    "model_echo_matches",
    "probe_applicability",
    "mandatory_probe_ids",
    "identity_probe_ids",
    "ProbeResult",
    "probe_summary",
    "run_probes",
    "L1DetectionRequest",
    "run_l1_detection",
    "UnsafeTargetError",
    "ResponseTooLarge",
    "PublicOnlyTransport",
    "normalize_public_base_url",
    "validate_public_target",
    "resolve_public_ips",
    "L1_HIGH",
    "L1_MIN_COVERAGE",
    "decide_l1",
    "decide_l1_public",
]
