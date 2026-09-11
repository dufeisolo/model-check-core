import json
from dataclasses import replace

import httpx
import pytest


from model_check.clients import ApiClient, ProtocolError
from model_check.config import Settings

ANTHROPIC_OK = httpx.Response(
    200,
    json={
        "id": "msg_01abc", "type": "message", "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 2},
    },
    headers={"request-id": "req_123", "cf-ray": "ray_9"},
)

OPENAI_OK = httpx.Response(
    200,
    json={
        "id": "chatcmpl-abc", "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": "hello"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    },
)


def _anthropic_transport():
    async def handler(request):
        if request.url.path == "/v1/messages/count_tokens":
            return httpx.Response(200, json={"input_tokens": 42})
        if request.url.path == "/v1/messages":
            return ANTHROPIC_OK
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_detect_protocol_anthropic():
    c = ApiClient("https://api.anthropic.com", "sk-ant-test",
                  transport=_anthropic_transport())
    assert await c.detect_protocol() == "anthropic"
    await c.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("model,protocol,path", [
    ("gpt-5.5", "openai", "/v1/chat/completions"),
    ("claude-sonnet-4", "anthropic", "/v1/messages"),
])
@pytest.mark.parametrize("recovers", [True, False])
async def test_protocol_timeout_retries_preferred_once(model, protocol, path, recovers):
    calls = []

    async def handler(req):
        calls.append(req.url.path)
        assert req.extensions["timeout"]["read"] == 30.0
        if recovers and len(calls) == 2:
            return OPENAI_OK if protocol == "openai" else ANTHROPIC_OK
        raise httpx.ReadTimeout("slow response", request=req)

    c = ApiClient("https://relay.example.com", "test-key", default_model=model,
                  transport=httpx.MockTransport(handler), public_mode=True)
    try:
        assert await c.detect_protocol() == (protocol if recovers else "unknown")
        assert calls[:2] == [path, path]
        assert len(calls) == (2 if recovers else 3)
        assert c.proto_probe_log()[0]["error"] == "ReadTimeout"
        assert await c.detect_protocol() == (protocol if recovers else "unknown")
        assert len(calls) == (2 if recovers else 3)
    finally:
        await c.close()


@pytest.mark.asyncio
async def test_complete_unified_shape_anthropic():
    c = ApiClient("https://api.anthropic.com", "sk-ant-test",
                  transport=_anthropic_transport())
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=50)
    assert r["text"] == "hello"
    assert r["usage_in"] == 5 and r["usage_out"] == 2
    assert r["stop_reason"] == "end_turn"
    assert r["request_id"] == "req_123"
    await c.close()


@pytest.mark.asyncio
async def test_detect_protocol_openai_fallback():
    async def handler(request):
        if request.url.path == "/v1/chat/completions":
            return OPENAI_OK
        return httpx.Response(404, json={"type": "error",
                                         "error": {"type": "not_found_error", "message": "nf"}})

    c = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler))
    assert await c.detect_protocol() == "openai"
    await c.close()


@pytest.mark.asyncio
async def test_complete_openai_shape():
    async def handler(request):
        if request.url.path == "/v1/chat/completions":
            return OPENAI_OK
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler))
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=50)
    assert r["text"] == "hello"
    assert r["usage_in"] == 5
    assert r["request_id"] == "chatcmpl-abc"
    await c.close()


@pytest.mark.asyncio
async def test_detect_unknown():
    async def handler(request):
        return httpx.Response(404, json={})

    c = ApiClient("https://nothing.example.com", "k", transport=httpx.MockTransport(handler))
    assert await c.detect_protocol() == "unknown"
    with pytest.raises(ProtocolError):
        await c.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    await c.close()


@pytest.mark.asyncio
async def test_count_tokens_ok_and_fallback():
    c = ApiClient("https://api.anthropic.com", "sk-ant-test",
                  transport=_anthropic_transport())
    assert await c.count_tokens("hello world") == 42
    await c.close()

    async def handler(request):
        return httpx.Response(500, json={})

    c2 = ApiClient("https://x", "k", transport=httpx.MockTransport(handler))
    assert await c2.count_tokens("hi") == -1
    await c2.close()


@pytest.mark.asyncio
async def test_retry_on_transient_error():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if request.url.path != "/v1/messages":
            return httpx.Response(404, json={})
        if calls["n"] == 1:
            return ANTHROPIC_OK  # 协议探测成功
        if calls["n"] <= 3:
            raise httpx.ConnectError("boom")  # complete: 尝试 1 次 + 重试 1 次失败
        return ANTHROPIC_OK  # 第 3 次尝试成功

    c = ApiClient("https://api.anthropic.com", "k", transport=httpx.MockTransport(handler))
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert r["text"] == "hello"
    assert calls["n"] == 4  # 1 次探测 + 1 次尝试 + 2 次重试(前 2 次失败) + 第 3 次成功
    await c.close()


@pytest.mark.asyncio
async def test_complete_retries_one_rejected_request_after_protocol_success():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return ANTHROPIC_OK  # 协议识别成功
        if calls["n"] == 2:
            return httpx.Response(401, json={"error": {"message": "channel busy"}})
        return ANTHROPIC_OK

    settings = replace(Settings(), RETRY_BACKOFF=(0, 0))
    c = ApiClient("https://api.anthropic.com", "k",
                  transport=httpx.MockTransport(handler), settings=settings)
    result = await c.complete([{"role": "user", "content": "hi"}], max_tokens=5)

    assert result["text"] == "hello"
    assert calls["n"] == 3
    await c.close()


@pytest.mark.asyncio
async def test_detect_protocol_error_shape_required():
    """P3-4：错误响应须符合 Anthropic 错误结构才判协议，避免误判。"""
    async def handler(request):
        if request.url.path == "/v1/messages":
            return httpx.Response(400, json={"type": "weird",
                                             "message": "msg_ 出现在错误消息里误导判定"})
        return httpx.Response(404, json={})

    c = ApiClient("https://x", "k", transport=httpx.MockTransport(handler))
    assert await c.detect_protocol() == "unknown"  # 错误结构不符 → 不判 anthropic
    await c.close()

    async def handler2(request):
        if request.url.path == "/v1/messages":
            return httpx.Response(400, json={"type": "error",
                                             "error": {"type": "invalid_request_error",
                                                       "message": "bad"}})
        return httpx.Response(404, json={})

    c2 = ApiClient("https://x", "k", transport=httpx.MockTransport(handler2))
    assert await c2.detect_protocol() == "anthropic"  # 规范错误结构 → 判 anthropic
    await c2.close()


@pytest.mark.asyncio
async def test_detect_protocol_openai_requires_body_shape():
    """P3-19：openai 探测收紧——仅状态码不符响应体特征的端点不判 openai。"""
    async def handler(request):
        # 返回 400 但响应体不是 OpenAI 错误结构（裸字符串）
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(400, text="Bad Request")
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler))
    # 响应体非 JSON / 无 error 对象 → 不判 openai
    assert await c.detect_protocol() == "unknown"
    await c.close()

    async def handler2(request):
        # 返回 200 但无 choices/object 字段
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"random": "payload"})
        return httpx.Response(404, json={})

    c2 = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler2))
    assert await c2.detect_protocol() == "unknown"
    await c2.close()


@pytest.mark.asyncio
async def test_default_model_propagates_to_probe():
    """未来模型名透传到探测/审计请求（中转站严格校验模型名场景）。"""
    seen = {}

    async def handler(request):
        import json
        seen["path"] = request.url.path
        try:
            seen["model"] = json.loads(request.content).get("model")
        except Exception:
            pass
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json={
                "id": "msg_abc", "type": "message",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2}},
                headers={"request-id": "req_1"})
        return httpx.Response(404, json={})

    c = ApiClient("https://x", "k", transport=httpx.MockTransport(handler),
                  default_model="claude-opus-5-0")
    assert await c.detect_protocol() == "anthropic"
    assert seen["model"] == "claude-opus-5-0"
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert seen["model"] == "claude-opus-5-0"
    assert r["text"] == "hi"
    await c.close()


@pytest.mark.asyncio
async def test_openai_complete_param_mapping():
    """openai 分支参数映射：model 透传、stop_sequences→stop、system→role=system 消息。

    回归锚点：修复前该分支硬编码 "gpt-4o-mini" 且不映射 stop/system，
    kimi 等非 openai 声明模型的检测会路由到错误模型。
    """
    bodies = []

    async def handler(request):
        import json
        if request.url.path == "/v1/chat/completions":
            bodies.append(json.loads(request.content))
            return OPENAI_OK
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any",
                  transport=httpx.MockTransport(handler), default_model="kimi-k3")
    # 预热协议缓存，使 bodies 只含 complete 的请求体
    assert await c.detect_protocol() == "openai"
    bodies.clear()
    r = await c.complete(
        [{"role": "user", "content": "hi"}], max_tokens=50,
        extra={"system": "你是助手", "stop_sequences": ["STOPXYZ"],
               "reasoning_effort": "max"})
    await c.close()

    assert len(bodies) == 1
    b = bodies[0]
    assert b["model"] == "kimi-k3"          # 透传声明模型，不得硬编码
    assert b["max_tokens"] == 50
    assert b["stop"] == ["STOPXYZ"]         # stop_sequences → stop
    assert "stop_sequences" not in b        # anthropic 参数名不得残留
    assert b["messages"][0] == {"role": "system", "content": "你是助手"}
    assert b["messages"][1] == {"role": "user", "content": "hi"}
    assert b["reasoning_effort"] == "max"   # 白名单内 → 透传
    assert r["text"] == "hello"
    assert r["stop_reason"] == "stop"
    assert r["usage_in"] == 5
    assert r["request_id"] == "chatcmpl-abc"


@pytest.mark.asyncio
async def test_openai_reasoning_effort_filtered():
    """reasoning_effort 不在白名单时不透传（避免严格校验的中转站返回 4xx）。"""
    bodies = []

    async def handler(request):
        import json
        if request.url.path == "/v1/chat/completions":
            bodies.append(json.loads(request.content))
            return OPENAI_OK
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler))
    assert await c.detect_protocol() == "openai"
    bodies.clear()
    await c.complete([{"role": "user", "content": "hi"}], max_tokens=5,
                     extra={"reasoning_effort": "ultra_max"})
    await c.close()
    assert "reasoning_effort" not in bodies[0]


@pytest.mark.asyncio
async def test_openai_model_propagates_to_probe():
    """openai 协议探测请求同样透传 default_model（中转站严格校验模型名场景）。"""
    seen = {}

    async def handler(request):
        import json
        try:
            seen["model"] = json.loads(request.content).get("model")
        except Exception:
            pass
        if request.url.path == "/v1/chat/completions":
            return OPENAI_OK
        return httpx.Response(404, json={"detail": "Not Found"})

    c = ApiClient("https://relay.example.com", "sk-any",
                  transport=httpx.MockTransport(handler), default_model="kimi-k3")
    assert await c.detect_protocol() == "openai"
    assert seen["model"] == "kimi-k3"
    await c.close()


@pytest.mark.asyncio
async def test_gpt_protocol_probe_prefers_openai_without_token_limit():
    """GPT-5 先走常规 Chat Completions，协议识别不携带易卡住的 token 参数。"""
    seen = []

    async def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/chat/completions":
            return OPENAI_OK
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any",
                  transport=httpx.MockTransport(handler), default_model="gpt-5.5")
    assert await c.detect_protocol() == "openai"
    await c.close()

    assert [path for path, _ in seen] == ["/v1/chat/completions"]
    body = seen[0][1]
    assert body["model"] == "gpt-5.5"
    assert "max_tokens" not in body
    assert "max_completion_tokens" not in body


@pytest.mark.asyncio
async def test_raw_chat_returns_raw_response():
    """raw_chat 直返原始响应不抛 ProtocolError（o10 依赖其检查 4xx 状态码）。"""
    async def handler(request):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(400, json={"error": {
                "message": "invalid reasoning_effort",
                "type": "invalid_request_error"}})
        return httpx.Response(404, json={})

    c = ApiClient("https://relay.example.com", "sk-any", transport=httpx.MockTransport(handler))
    r = await c.raw_chat({"model": "kimi-k3", "max_tokens": 5,
                          "messages": [{"role": "user", "content": "hi"}],
                          "reasoning_effort": "ultra_max"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    await c.close()


@pytest.mark.asyncio
async def test_base_url_with_v1_suffix_normalized():
    """base_url 以 /v1 结尾时自动剥离，避免打出 /v1/v1/... 双前缀路径。"""
    seen = []

    async def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion",
                                         "choices": [{"index": 0,
                                                      "message": {"role": "assistant",
                                                                  "content": "hi"},
                                                      "finish_reason": "stop"}]})

    c = ApiClient("https://relay.example.com/v1/", "sk-any",
                  transport=httpx.MockTransport(handler))
    assert c.base_url == "https://relay.example.com"
    assert await c.detect_protocol() == "openai"
    await c.close()
    assert seen, "探测应发出请求"
    assert all(p in ("/v1/messages", "/v1/chat/completions") for p in seen)
    assert "/v1/chat/completions" in seen


@pytest.mark.asyncio
async def test_openai_max_completion_tokens_fallback():
    """o 系/GPT-5 推理模型：max_tokens 400 → 换 max_completion_tokens 重试成功。"""
    async def handler(request):
        body = json.loads(request.read())
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unsupported parameter: 'max_tokens' is not supported "
                           "with this model. Use 'max_completion_tokens' instead.",
                "type": "invalid_request_error"}})
        return httpx.Response(200, json={
            "id": "chatcmpl-fallback", "object": "chat.completion",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    c = ApiClient("https://api.openai.com", "k",
                  transport=httpx.MockTransport(handler), default_model="o4-mini")
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=10)
    await c.close()
    assert r["text"] == "ok"
    assert r["stop_reason"] == "stop"


def test_gemini_base_url_normalized():
    """Gemini 官方域名任意写法（裸域 /v1beta /v1beta/openai）都规范化到兼容端点。"""
    canonical = ("https://generativelanguage.googleapis.com"
                 "/v1beta/openai/chat/completions")
    for base in ("https://generativelanguage.googleapis.com",
                 "https://generativelanguage.googleapis.com/",
                 "https://generativelanguage.googleapis.com/v1beta",
                 "https://generativelanguage.googleapis.com/v1beta/openai",
                 "https://generativelanguage.googleapis.com/v1beta/openai/"):
        c = ApiClient(base, "k")
        assert c._chat_url == canonical, base
        # anthropic 路径保持通用拼接（Gemini 无该端点，探测 404 落入 openai）
        assert c._messages_url == "https://generativelanguage.googleapis.com/v1/messages"


def test_non_gemini_base_url_unchanged():
    """普通网关地址不受 Gemini 规范化影响。"""
    c = ApiClient("https://api.moonshot.cn/v1", "k")
    assert c._chat_url == "https://api.moonshot.cn/v1/chat/completions"
    assert c._messages_url == "https://api.moonshot.cn/v1/messages"


@pytest.mark.asyncio
async def test_gemini_protocol_detection_via_compat_endpoint():
    """Gemini 域名：/v1/messages 404 → openai 兼容端点识别成功。"""
    hits = []

    async def handler(request):
        hits.append(request.url.path)
        if request.url.path.endswith("/v1beta/openai/chat/completions"):
            return httpx.Response(200, json={
                "id": "chatcmpl-g", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}})
        return httpx.Response(404, json={"detail": "Not Found"})

    c = ApiClient("https://generativelanguage.googleapis.com", "k",
                  transport=httpx.MockTransport(handler), default_model="gemini-2.5-pro")
    proto = await c.detect_protocol()
    await c.close()
    assert proto == "openai"
    assert "/v1beta/openai/chat/completions" in "".join(hits)


def test_base_url_with_endpoint_path_normalized():
    """用户把完整端点路径当 base_url 填（实测案例）→ 剥离端点后缀，避免双重路径 404。"""
    cases = {
        "https://gw.example.com/v1/chat/completions": "https://gw.example.com",
        "https://gw.example.com/v1/chat/completions/": "https://gw.example.com",
        "https://gw.example.com/v1/messages": "https://gw.example.com",
        "https://gw.example.com/v1/messages/count_tokens": "https://gw.example.com",
        "https://gw.example.com/chat/completions": "https://gw.example.com",
        "https://gw.example.com/v1": "https://gw.example.com",
        "https://api.anthropic.com": "https://api.anthropic.com",
    }
    for raw, expect in cases.items():
        c = ApiClient(raw, "k")
        assert c.base_url == expect, raw
        assert c._chat_url == expect + "/v1/chat/completions", raw
        assert c._messages_url == expect + "/v1/messages", raw


@pytest.mark.asyncio
async def test_anthropic_temperature_fallback():
    """anthropic 分支严格网关：temperature≠1.0 → 400 → 去参重试成功（与 openai 同构）。"""
    seen = []

    async def handler(request):
        body = json.loads(request.read())
        seen.append(body.get("temperature", None))
        if "temperature" in body:
            return httpx.Response(400, json={
                "type": "error",
                "error": {"type": "invalid_request_error",
                          "message": "temperature must be 1.0"}})
        return httpx.Response(200, json={
            "id": "msg_ok", "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
            "stop_reason": "end_turn"})

    c = ApiClient("https://strict-relay.example.com", "k",
                  transport=httpx.MockTransport(handler))
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=10,
                         temperature=0)
    await c.close()
    assert r["text"] == "hello"
    # seen 序列：[协议探测(无temperature), complete 首次(temperature=0 被拒), 重试(无temperature)]
    assert seen[-2:] == [0, None]


@pytest.mark.asyncio
async def test_extra_body_merged_into_openai_requests():
    """extra_body（如 OpenRouter provider 钉扎）合并进每个 openai 请求体。"""
    import json as _json
    seen = []

    async def handler(request):
        seen.append(_json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-x", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    c = ApiClient("https://gw.example.com", "k", transport=httpx.MockTransport(handler),
                  default_model="glm-5.3",
                  extra_body={"provider": {"order": ["z-ai"], "allow_fallbacks": False}})
    r = await c.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    await c.close()
    assert (r["text"] or "") == "ok"
    # seen[0] 是协议探测请求（不合并 extra_body，符合预期）；complete 请求必须携带
    assert seen[-1].get("provider") == {"order": ["z-ai"], "allow_fallbacks": False}
    assert seen[-1]["model"] == "glm-5.3"
