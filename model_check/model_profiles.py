"""L1 模型能力配置：决定探针是否适用，以及协议探测优先级。

维护各大主流模型家族能力（Claude, GPT/o, Gemini, DeepSeek, GLM, Kimi, Grok, MiniMax, Qwen）。
未知模型保留 ``None``，由运行时探测决定，避免配置库未更新时直接误判。
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
    """将声明模型名称归入各大主流能力家族。"""
    name = (model or "").strip().lower().replace("_", "-")
    clean_name = _strip_channel_prefix(name)

    # 1. Anthropic Claude 家族
    if "claude" in clean_name:
        modern = bool(re.search(r"(?:^|[-_/])(?:3|4|5)(?:[.-]|$)", clean_name))
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

    # 2. OpenAI GPT / o 系列
    if "gpt" in clean_name or _contains_o_series(clean_name):
        modern_gpt = any(token in clean_name for token in ("gpt-4", "gpt-5", "gpt-6"))
        reasoning = any(token in clean_name for token in ("gpt-5", "gpt-6")) or _contains_o_series(clean_name)
        return ModelProfile(
            provider="openai",
            family="openai",
            known=True,
            preferred_protocol="openai",
            tools=True if (modern_gpt or reasoning) else None,
            vision=True if (modern_gpt or reasoning) else None,
            document=False,
            prompt_cache=None,
            reasoning_effort=reasoning,
        )

    # 3. Google Gemini 家族
    if "gemini" in clean_name:
        return ModelProfile(
            provider="google",
            family="gemini",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True,
            document=False,
            prompt_cache=False,
            reasoning_effort=False,
        )

    # 4. DeepSeek 家族
    if "deepseek" in clean_name:
        is_vision = "-vl" in clean_name or "vision" in clean_name
        is_reasoning = any(tok in clean_name for tok in ("reasoner", "r1", "pro", "v4-pro"))
        return ModelProfile(
            provider="deepseek",
            family="deepseek",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True if is_vision else False,
            document=False,
            prompt_cache=False,
            reasoning_effort=is_reasoning,
        )

    # 5. 智谱 GLM 家族
    if any(tok in clean_name for tok in ("glm", "chatglm", "zhipu")):
        is_vision = any(tok in clean_name for tok in ("-v", "-vl", "vision"))
        return ModelProfile(
            provider="zhipu",
            family="glm",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True if is_vision else False,
            document=False,
            prompt_cache=False,
            reasoning_effort=False,
        )

    # 6. 月之暗面 Kimi 家族
    if any(tok in clean_name for tok in ("kimi", "moonshot")):
        is_vision = "-vl" in clean_name or "vision" in clean_name
        is_reasoning = "k3" in clean_name
        return ModelProfile(
            provider="moonshot",
            family="kimi",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True if is_vision else False,
            document=False,
            prompt_cache=False,
            reasoning_effort=is_reasoning,
        )

    # 7. xAI Grok 家族
    if "grok" in clean_name:
        return ModelProfile(
            provider="xai",
            family="grok",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True,
            document=False,
            prompt_cache=False,
            reasoning_effort=False,
        )

    # 8. MiniMax 家族
    if any(tok in clean_name for tok in ("minimax", "abab")):
        is_vision = "-vl" in clean_name or "vision" in clean_name
        return ModelProfile(
            provider="minimax",
            family="minimax",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True if is_vision else False,
            document=False,
            prompt_cache=False,
            reasoning_effort=False,
        )

    # 9. 阿里通义 Qwen 家族
    if any(tok in clean_name for tok in ("qwen", "qwq", "qvq")):
        is_vision = any(tok in clean_name for tok in ("-vl", "vision", "qvq"))
        return ModelProfile(
            provider="alibaba",
            family="qwen",
            known=True,
            preferred_protocol="openai",
            tools=True,
            vision=True if is_vision else False,
            document=False,
            prompt_cache=False,
            reasoning_effort=False,
        )

    return UNKNOWN_PROFILE


_FAMILY_STARTS = (
    "claude-", "gpt-", "gemini-", "deepseek-", "glm-", "kimi-",
    "grok-", "minimax-", "qwen-", "o1-", "o3-", "o4-", "moonshot-",
)


def _strip_channel_prefix(model: str) -> str:
    """保留从真实模型家族标识开始的部分，兼容 vendor/model-name 前缀与路由标记。"""
    name = (model or "").strip().lower().replace("_", "-")
    name = name.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    starts = [pos for token in _FAMILY_STARTS if (pos := name.find(token)) >= 0]
    if starts:
        return name[min(starts):]
    return name


MODEL_ALIAS_GROUPS: list[set[str]] = [
    # DeepSeek 家族
    {"deepseek-v4-flash", "deepseek-flash", "deepseek-chat"},
    {"deepseek-v4-pro", "deepseek-pro", "deepseek-reasoner"},
    {"deepseek-r1", "deepseek-reasoner"},
    # Claude 家族
    {"claude-3-5-sonnet", "claude-3-5-sonnet-20241022", "claude-sonnet-3-5"},
    {"claude-3-5-haiku", "claude-3-5-haiku-20241022", "claude-haiku-3-5"},
    {"claude-opus-4-8", "claude-4-8-opus"},
    {"claude-sonnet-5-5", "claude-5-5-sonnet"},
    {"claude-fable-5", "claude-5-fable"},
    # Gemini 家族
    {"gemini-3.8-flash", "gemini-flash", "gemini-3.8-flash-preview", "gemini-3.8-flash-tiered"},
    {"gemini-2.0-flash-exp", "gemini-2.0-flash"},
    {"gemini-4-argon", "gemini-argon"},
    # 国产与其它
    {"kimi-k3", "kimi-latest", "moonshot-v1-k3"},
    {"glm-5.3", "glm-5", "glm-5.3-flash"},
    {"gpt-5.6-terra", "gpt-5.6"},
    {"gpt-6-astra", "gpt-6"},
]

_ALLOWED_SUFFIX_PATTERN = (
    r"(?:20\d{2}-?\d{2}-?\d{2}|latest|v\d+(?:\.\d+)?|"
    r"tiered|medium|fast|turbo|flash|pro|standard|free|"
    r"preview|thinking|instruct|chat|online|search|text|fc|exp)"
)

_ALLOWED_SUFFIX_REGEX = re.compile(
    rf"^{_ALLOWED_SUFFIX_PATTERN}(?:-{_ALLOWED_SUFFIX_PATTERN})*$"
)


def _strip_allowed_suffixes(name: str) -> str:
    """递归剥离尾部合规后缀，提取稳定核心名称段。"""
    while True:
        m = re.search(rf"-({_ALLOWED_SUFFIX_PATTERN})$", name)
        if m and len(name[:m.start()].split("-")) >= 2:
            name = name[:m.start()]
        else:
            break
    return name


def model_echo_matches(declared: str, echoed: str) -> bool:
    """核对模型回显：精确匹配、合规后缀白名单容差与官方别名映射。"""
    expected = _strip_channel_prefix(declared)
    actual = _strip_channel_prefix(echoed)
    if not expected or not actual:
        return False
    if actual == expected:
        return True

    # 官方别名互认
    for group in MODEL_ALIAS_GROUPS:
        norm_group = {_strip_channel_prefix(m) for m in group}
        if expected in norm_group and actual in norm_group:
            return True

    # 前缀扩展：actual = expected + "-<allowed_suffix>"
    if actual.startswith(expected + "-"):
        suffix = actual[len(expected) + 1:]
        if _ALLOWED_SUFFIX_REGEX.fullmatch(suffix):
            return True

    # 反向兼容：declared 带有 preview/tiered 等后缀，actual 为基准名
    if expected.startswith(actual + "-"):
        suffix = expected[len(actual) + 1:]
        if _ALLOWED_SUFFIX_REGEX.fullmatch(suffix):
            return True

    # 同一基准名的不同变体后缀归并（例如 gemini-3.8-flash-preview ↔ gemini-3.8-flash-tiered）
    base_exp = _strip_allowed_suffixes(expected)
    base_act = _strip_allowed_suffixes(actual)
    if base_exp and base_act and base_exp == base_act and len(base_exp.split("-")) >= 2:
        return True

    return False


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
    """真实性判定的存活性硬门槛：基础连通性调用。"""
    return frozenset(("p1",)) if protocol == "anthropic" else frozenset(("o1",))


def identity_probe_ids(profile: ModelProfile, protocol: str) -> frozenset[str]:
    """已知模型应具备的能力探针；至少一项通过才形成身份辅助证据。"""
    candidates = ("p6", "p10", "p13", "p14") if protocol == "anthropic" \
        else ("o8", "o9", "o10")
    return frozenset(
        probe_id for probe_id in candidates
        if probe_applicability(profile, probe_id) is True
    )
