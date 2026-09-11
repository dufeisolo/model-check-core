from model_check.model_profiles import (identity_probe_ids, model_echo_matches,
                                probe_applicability, resolve_model_profile)


def test_openai_profile_with_channel_prefix():
    profile = resolve_model_profile("vendor-a/gpt-5.5")
    assert profile.provider == "openai"
    assert profile.preferred_protocol == "openai"
    assert profile.tools is True
    assert profile.vision is True
    assert profile.reasoning_effort is True
    assert identity_probe_ids(profile, "openai") == {"o8", "o9", "o10"}


def test_gpt4o_reasoning_effort_not_applicable():
    profile = resolve_model_profile("gpt-4o")
    assert probe_applicability(profile, "o10") is False
    assert probe_applicability(profile, "o9") is True


def test_modern_claude_capabilities():
    profile = resolve_model_profile("relay/claude-sonnet-4-5")
    assert profile.provider == "anthropic"
    assert profile.preferred_protocol == "anthropic"
    assert profile.tools is True
    assert profile.vision is True
    assert profile.document is True
    assert profile.prompt_cache is True


def test_unknown_model_keeps_dynamic_capabilities():
    profile = resolve_model_profile("custom-model-x")
    assert profile.known is False
    assert probe_applicability(profile, "o9") is None


def test_model_echo_requires_exact_version_but_allows_prefix_and_date():
    assert model_echo_matches("gpt-5.5", "gpt-5.5")
    assert model_echo_matches("relay/gpt-4o", "openai/gpt-4o-2024-11-20")
    assert model_echo_matches("claude-sonnet-4-5", "claude-sonnet-4-5-latest")
    assert not model_echo_matches("gpt-5.5", "gpt-4o")
    assert not model_echo_matches("gpt-5", "gpt-5-mini")
