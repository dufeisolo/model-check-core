"""L1 检测判定规则。"""

from __future__ import annotations

from model_check.model_profiles import (
    ModelProfile,
    identity_probe_ids,
    mandatory_probe_ids,
)

L1_HIGH = 0.7
L1_MIN_COVERAGE = 0.7
L1_MID = 0.4


def decide_l1(summary: dict) -> str:
    """公网 L1 二值结论：只依据现有加权得分，不调用复杂判定树。"""
    try:
        score = float(summary.get("score", 0.0))
    except (AttributeError, TypeError, ValueError):
        score = 0.0
    return "pass" if score >= L1_HIGH else "fail"


def decide_l1_public(
    summary: dict,
    results: list,
    profile: ModelProfile,
    protocol: str,
) -> str:
    """公网 L1 结论：硬门槛、得分、覆盖率和身份能力证据同时成立。

    返回 pass / fail / inconclusive。
    """
    by_id = {item.id: item for item in results}
    mandatory = [by_id.get(pid) for pid in mandatory_probe_ids(protocol)]
    if any(item is not None and item.status == "fail" for item in mandatory):
        return "fail"
    if any(item is None or item.status == "na" for item in mandatory):
        return "inconclusive"

    identity = [
        by_id[pid]
        for pid in identity_probe_ids(profile, protocol)
        if pid in by_id
    ]
    if profile.known and identity and not any(item.status == "pass" for item in identity):
        if all(item.status == "fail" for item in identity):
            return "fail"
        return "inconclusive"

    try:
        score = float(summary.get("score", 0.0))
        coverage = float(summary.get("coverage", 0.0))
    except (AttributeError, TypeError, ValueError):
        return "inconclusive"
    if score < L1_HIGH:
        return "fail"
    if coverage < L1_MIN_COVERAGE:
        return "inconclusive"
    return "pass"
