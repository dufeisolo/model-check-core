"""L1 模型能力配置：决定探针是否适用，以及协议探测优先级。

这里只维护稳定的模型家族能力，不把渠道前缀当成新模型。未知模型保留
``None``，由运行时探测决定，避免配置库未更新时直接误判。
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    family: str
    known: bool
    preferred_protocol: str
    tools: bool | None = None
    vision: bool | None = None
    document: bool | None = None
    prompt_cache: bool | None = None
    reasoning_effort: bool | None = None


UNKNOWN_PROFILE = ModelProfile("unknown", "unknown", False, "openai")


def _contains_o_series(model: str) -> bool:
    return bool(re.search(r"(?:^|[/_:.-])o(?:1|3|4)(?:$|[/_:.-])", model))


def resolve_model_profile(model: str) -> ModelProfile:
    """将带渠道前缀的声明名称归入 OpenAI / Anthropic 能力家族。"""
    name = (model or "").strip().lower()
    if "claude" in name:
        modern = bool(re.search(r"(?:^|[-_/])(?:3|4|5)(?:[.-]|$)", name))
        return ModelProfile(
            provider="anthropic",
            family="claude",
            known=True,
            preferred_protocol="anthropic",
            tools=True if modern else None,
            vision=True if modern else None,
            document=True if modern else None,
            prompt_cache=True if modern else None,
            reasoning_effort=False,
        )

    if "gpt" in name or _contains_o_series(name):
        modern_gpt = any(token in name for token in ("gpt-4o", "gpt-4.1", "gpt-5"))
        reasoning = "gpt-5" in name or _contains_o_series(name)
        return ModelProfile(
            provider="openai",
            family="openai",
            known=True,
            preferred_protocol="openai",
            tools=True if modern_gpt or reasoning else None,
            vision=True if modern_gpt or reasoning else None,
            document=None,
            prompt_cache=None,
            reasoning_effort=reasoning,
        )

    return UNKNOWN_PROFILE


def _strip_channel_prefix(model: str) -> str:
    """保留从真实模型家族标识开始的部分，兼容 vendor/model-name 前缀。"""
    name = (model or "").strip().lower()
    starts = [pos for token in ("claude-", "gpt-")
              if (pos := name.find(token)) >= 0]
    if starts:
        return name[min(starts):]
    return name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def model_echo_matches(declared: str, echoed: str) -> bool:
    """严格核对模型回显；仅容许渠道前缀和官方日期版本后缀。

    不能再用 ``gpt``/``claude`` 家族前缀匹配，否则声明 GPT-5 而实际回显
    GPT-4o 也会被错误放行。
    """
    expected = _strip_channel_prefix(declared)
    actual = _strip_channel_prefix(echoed)
    if not expected or not actual:
        return False
    if actual == expected:
        return True
    if not actual.startswith(expected + "-"):
        return False
    suffix = actual[len(expected) + 1:]
    return bool(re.fullmatch(r"(?:20\d{2}-?\d{2}-?\d{2}|latest)", suffix))


_CAPABILITY_BY_PROBE = {
    "p6": "prompt_cache",
    "p10": "tools",
    "p13": "vision",
    "p14": "document",
    "o8": "tools",
    "o9": "vision",
    "o10": "reasoning_effort",
}


def probe_applicability(profile: ModelProfile, probe_id: str) -> bool | None:
    """返回 True/False/None：适用、不适用、配置未知需动态探测。"""
    capability = _CAPABILITY_BY_PROBE.get(probe_id)
    return getattr(profile, capability) if capability else True


def mandatory_probe_ids(protocol: str) -> frozenset[str]:
    """真实性判定的硬门槛：基础调用与声明模型回显。"""
    return frozenset(("p1", "p2")) if protocol == "anthropic" \
        else frozenset(("o1", "o2"))


def identity_probe_ids(profile: ModelProfile, protocol: str) -> frozenset[str]:
    """已知模型应具备的能力探针；至少一项通过才形成身份辅助证据。"""
    candidates = ("p6", "p10", "p13", "p14") if protocol == "anthropic" \
        else ("o8", "o9", "o10")
    return frozenset(
        probe_id for probe_id in candidates
        if probe_applicability(profile, probe_id) is True
    )
