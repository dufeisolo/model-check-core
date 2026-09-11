import base64
import json

import httpx
import pytest

from model_check.clients import ApiClient

from model_check.config import get_settings
from model_check.probes import (MINI_PDF, PROBES, PROBES_OPENAI, RED_PNG_B64,
                        ProbeResult, o_param_matrix, probe_summary,
                        p_multi_turn, p_param_matrix, run_probes)

SSE_TEXT = (
    'event: message_start\ndata: {"type":"message_start"}\n\n'
    'event: content_block_start\ndata: {"type":"content_block_start"}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
    'event: content_block_stop\ndata: {"type":"content_block_stop"}\n\n'
    'event: message_delta\ndata: {"type":"message_delta"}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


def make_faithful_transport(**overrides):
    """模拟一个忠实 Anthropic 实现：按请求体内容返回对应行为。

    overrides 可覆盖：
    - replay: bool 两次相同请求返回相同内容（缓存/重放）
    - echo_model: str 响应 model 字段
    - no_cache_field: bool 响应不含 cache 字段
    - short_text: bool 返回短文本（<32 token）
    - no_tool_use: bool 有 tools 也不返回工具调用
    - bad_error_shape: bool 错误对象不符合规范
    """
    o = {"replay": False, "echo_model": "claude-sonnet-4-5", "no_cache_field": False,
         "short_text": False, "no_tool_use": False, "bad_error_shape": False}
    o.update(overrides)
    calls = []

    def _usage(body, cache_read=False):
        u = {"input_tokens": 20, "output_tokens": 8}
        if not o["no_cache_field"]:
            u["cache_creation_input_tokens"] = 100 if cache_read else 2000
            if cache_read:
                u["cache_read_input_tokens"] = 1000
        return u

    def _content(body, cache_read=False):
        if o["short_text"]:
            return [{"type": "text", "text": "ok"}]
        # tools → tool_use
        if body.get("tools") and not o["no_tool_use"]:
            return [{"type": "tool_use", "id": "toolu_01abc", "name": "get_weather",
                     "input": {"city": "北京"}}]
        # system 首字母 Z 约束
        if (body.get("system") or "").startswith("你的每一条回复"):
            return [{"type": "text", "text": "Z 这是服从约束的回答。"}]
        # 图像
        if any(isinstance(c, dict) and c.get("type") == "image" for c in body.get("messages", [{}])[-1].get("content", []) if isinstance(c, dict)):
            return [{"type": "text", "text": "The image color is red."}]
        # 文档
        if any(isinstance(c, dict) and c.get("type") == "document" for c in body.get("messages", [{}])[-1].get("content", []) if isinstance(c, dict)):
            return [{"type": "text", "text": "The document says HELLO MOM."}]
        # 中性会话校验码复述
        msgs = body.get("messages", [])
        full = " ".join(str(m.get("content", "")) for m in msgs)
        for marker in ("MC7A9", "QV4K2"):
            if marker in full and "最后一次" in full:
                return [{"type": "text", "text": marker}]
            if marker in full:
                return [{"type": "text", "text": "收到"}]
        # stop_sequence
        if body.get("stop_sequences"):
            return [{"type": "text", "text": "这是回复 STOPXYZ"}]
        # max_tokens 截断
        if body.get("max_tokens") is not None and body["max_tokens"] <= 3:
            return [{"type": "text", "text": "截断"}]
        # 长文本（探针 15 用）
        if "人工智能的历史" in full:
            return [{"type": "text", "text": "人工智能的历史发展。" * 30}]
        return [{"type": "text", "text": "pong"}]

    def _stop_reason(body):
        if body.get("stop_sequences"):
            return "stop_sequence"
        if body.get("max_tokens") is not None and body["max_tokens"] <= 3:
            return "max_tokens"
        if body.get("tools"):
            return "tool_use"
        return "end_turn"

    def _make_response(body, cache_read=False, replay_key="", seq=0):
        if o["replay"] and replay_key:
            rid = "req_replay"
        else:
            rid = f"req_{seq}"
        return httpx.Response(200, json={
            "id": "msg_abc", "type": "message", "role": "assistant",
            "model": o["echo_model"],
            "content": _content(body, cache_read),
            "stop_reason": _stop_reason(body),
            "usage": _usage(body, cache_read),
        }, headers={"request-id": rid, "cf-ray": "ray_1",
                    "anthropic-ratelimit-requests-limit": "100"})

    async def handler(request):
        calls.append(request)
        path = request.url.path
        if path == "/v1/messages/count_tokens":
            return httpx.Response(200, json={"input_tokens": 20})
        if path != "/v1/messages":
            return httpx.Response(404, json={})
        body = request.read() and __import__("json").loads(request.read())
        if not body:
            return httpx.Response(404, json={})
        if body.get("max_tokens") == 0:
            err = {"type": "error", "error": {"type": "invalid_request_error",
                                              "message": "max_tokens must be > 0"}}
            if o["bad_error_shape"]:
                err = {"type": "weird", "error": "not-a-dict"}
            return httpx.Response(400, json=err)
        if body.get("stream"):
            return httpx.Response(200, text=SSE_TEXT,
                                  headers={"content-type": "text/event-stream"})
        # 重放检测：第二次相同请求返回相同内容
        cache_read = calls.count(request) > 1
        replay_key = str(request.content)
        return _make_response(body, cache_read=cache_read, replay_key=replay_key, seq=len(calls))

    return httpx.MockTransport(handler), calls


@pytest.fixture
def settings():
    return get_settings()


@pytest.mark.asyncio
async def test_probes_all_pass(settings):
    transport, calls = make_faithful_transport()
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert len(results) == 16
    # 核心探针必须 pass
    for pid in ("p1", "p2", "p3", "p4", "p8", "p9", "p10", "p11", "p12", "p13"):
        assert by_id[pid].status == "pass", f"{pid}: {by_id[pid].detail}"
    # p5 token 审计（count_tokens=20 vs usage 20）
    assert by_id["p5"].status == "pass", by_id["p5"].detail
    # p7 错误格式
    assert by_id["p7"].status == "pass", by_id["p7"].detail
    # p15 两次输出相同但 request_id 不同 → na（同一 mock 同 text）
    assert by_id["p15"].status in ("pass", "na"), by_id["p15"].detail
    # p16 参数校验矩阵：渠道指纹探针，执行即 pass，weight=0 不计入得分
    assert by_id["p16"].status == "pass", by_id["p16"].detail
    assert "t1.0=200" in by_id["p16"].detail


@pytest.mark.asyncio
async def test_model_echo_fail(settings):
    transport, _ = make_faithful_transport(echo_model="deepseek-chat")
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p2"].status == "fail"
    assert "deepseek" in by_id["p2"].detail


@pytest.mark.asyncio
async def test_cache_replay_fail(settings):
    transport, _ = make_faithful_transport(replay=True)
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p15"].status == "fail", by_id["p15"].detail


@pytest.mark.asyncio
async def test_short_text_na(settings):
    transport, _ = make_faithful_transport(short_text=True)
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p15"].status == "na"


@pytest.mark.asyncio
async def test_no_cache_field_na(settings):
    transport, _ = make_faithful_transport(no_cache_field=True)
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p6"].status == "na"


@pytest.mark.asyncio
async def test_bad_error_shape_fail(settings):
    transport, _ = make_faithful_transport(bad_error_shape=True)
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p7"].status == "fail"


@pytest.mark.asyncio
async def test_no_tool_use_fail(settings):
    transport, _ = make_faithful_transport(no_tool_use=True)
    c = ApiClient("https://api.anthropic.com", "k", transport=transport)
    results = await run_probes(c, "claude-sonnet-4-5", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["p10"].status == "fail"


def test_probe_summary():
    results = [
        ProbeResult("a", "x", "pass", "", weight=3.0),
        ProbeResult("b", "y", "fail", "", weight=1.0),
        ProbeResult("c", "z", "na", "", weight=1.0),
    ]
    s = probe_summary(results)
    assert s["score"] == 3.0 / 4.0
    assert s["counts"] == {"pass": 1, "fail": 1, "na": 1}
    assert s["total_weight"] == 4.0
    assert s["applicable_weight"] == 5.0
    assert s["coverage"] == 4.0 / 5.0


def test_probe_summary_excludes_configured_not_applicable_from_coverage():
    results = [
        ProbeResult("a", "x", "pass", "", weight=3.0),
        ProbeResult("b", "y", "na", "", weight=2.0, applicable=False),
    ]
    s = probe_summary(results)
    assert s["score"] == 1.0
    assert s["coverage"] == 1.0


def test_constants():
    # 红色 PNG 有效
    assert base64.b64decode(RED_PNG_B64)[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"%PDF" in MINI_PDF


# ---------- OpenAI 兼容协议探针（o1-o12）----------

OPENAI_SSE = (
    'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk",'
    '"choices":[{"index":0,"delta":{"content":"pong"},"finish_reason":null}]}\n\n'
    'data: [DONE]\n\n'
)


def make_faithful_openai_transport(**overrides):
    """模拟忠实 OpenAI 兼容实现（/v1/chat/completions）：按请求体内容返回对应行为。

    overrides 可覆盖：
    - echo_model: str 响应 model 字段
    - stop_in_text: bool 输出包含 stop 序列本身（非标准 openai 语义）
    - effort_strict: bool 非法 reasoning_effort 返回 4xx
    - effort_reject_all: bool 所有 reasoning_effort 值均被拒（含合法值）
    - bad_effort_status: int 非法 reasoning_effort 的拒绝状态码（默认 400）
    - replay: bool 相同请求返回固定 id + 固定文本（缓存/重放）
    - usage_drift: bool o11 第二次请求 prompt_tokens 漂移（42→60）
    """
    o = {"echo_model": "kimi-k3", "stop_in_text": False, "effort_strict": True,
         "effort_reject_all": False, "bad_effort_status": 400,
         "replay": False, "usage_drift": False}
    o.update(overrides)
    calls = []
    usage_hits = 0

    def _reply_text(body, seq):
        """按请求体推断"模型应回复的文本"，分支次序与 make_faithful_transport 对齐。"""
        msgs = body.get("messages", [])
        last = msgs[-1] if msgs else {}
        content = last.get("content")
        if body.get("tools"):
            return ""
        # system 服从约束（o7）：首条 system 消息要求回复以 Z 开头
        if (msgs and msgs[0].get("role") == "system"
                and str(msgs[0].get("content", "")).startswith("你的每一条回复")):
            return "Z 这是服从约束的回答。"
        # 视觉输入（o9）：最后一条消息含 image_url 内容块
        if isinstance(content, list) and any(
                isinstance(c, dict) and c.get("type") == "image_url" for c in content):
            return "The image color is red."
        full = " ".join(str(m.get("content", "")) for m in msgs)
        # 多轮上下文（o6）：第三轮复述中性校验码
        for marker in ("MC7A9", "QV4K2"):
            if marker in full and "最后一次" in full:
                return marker
            if marker in full:
                return "收到"
        # stop 参数（o5）：openai 语义下输出不含序列本身
        if body.get("stop"):
            return "这是回复 STOPXYZ" if o["stop_in_text"] else "这是回复"
        # max_tokens 截断（o3）
        if body.get("max_tokens") is not None and body["max_tokens"] <= 3:
            return "截"
        if "usage 一致性审计" in full:
            return "ok"
        # 缓存/重放（o12）：默认按调用序号变化文本（两次输出不同）；
        # replay 模式固定文本，配合固定 id 构成重放证据
        if "人工智能的历史" in full:
            if o["replay"]:
                return "人工智能的历史发展（固定采样）。" + "历史脉络细节。" * 10
            return f"人工智能的历史发展（第 {seq} 次采样）。" + "历史脉络细节。" * 10
        return "pong"

    def _message(body, seq):
        """构造 choices[0].message：tools → tool_calls 结构，否则纯文本。"""
        if body.get("tools"):
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "call_01abc", "type": "function",
                                    "function": {"name": "get_weather",
                                                 "arguments": "{\"city\": \"北京\"}"}}]}
        return {"role": "assistant", "content": _reply_text(body, seq)}

    def _finish(body):
        """构造 choices[0].finish_reason。"""
        if body.get("tools"):
            return "tool_calls"
        if body.get("stop"):
            return "stop"
        if body.get("max_tokens") is not None and body["max_tokens"] <= 3:
            return "length"
        return "stop"

    def _make_response(body, seq):
        """构造 200 补全响应；replay 模式 id 固定，否则按调用序号唯一。"""
        nonlocal usage_hits
        rid = "chatcmpl-replay" if o["replay"] else f"chatcmpl-seq{seq}"
        prompt_tokens = 42
        full = " ".join(str(m.get("content", "")) for m in body.get("messages", []))
        if "usage 一致性审计" in full:
            usage_hits += 1
            if o["usage_drift"] and usage_hits > 1:
                prompt_tokens = 60
        return httpx.Response(200, json={
            "id": rid, "object": "chat.completion", "created": 1700000000,
            "model": o["echo_model"],
            "choices": [{"index": 0, "message": _message(body, seq),
                         "finish_reason": _finish(body)}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7,
                      "total_tokens": prompt_tokens + 7},
        })

    async def handler(request):
        calls.append(request)
        if request.url.path != "/v1/chat/completions":
            # 404 且无 anthropic 错误结构 → detect_protocol 落入 openai
            return httpx.Response(404, json={"detail": "Not Found"})
        body = json.loads(request.read())
        # o10 reasoning_effort 双信号：非法值（或全部值）按 bad_effort_status 拒绝
        if "reasoning_effort" in body:
            legal = body["reasoning_effort"] in ("minimal", "low", "medium", "high", "max")
            if o["effort_reject_all"] or (o["effort_strict"] and not legal):
                return httpx.Response(o["bad_effort_status"], json={"error": {
                    "message": f"invalid reasoning_effort: {body['reasoning_effort']}",
                    "type": "invalid_request_error", "code": "invalid_value"}})
            # 合法值落入正常补全
        if body.get("stream"):
            return httpx.Response(200, text=OPENAI_SSE,
                                  headers={"content-type": "text/event-stream"})
        return _make_response(body, len(calls))

    return httpx.MockTransport(handler), calls


@pytest.mark.asyncio
async def test_openai_probes_all_pass(settings):
    """默认忠实 openai mock 下 13 个 o 探针全部 pass（o13 为 weight=0 渠道指纹）。"""
    transport, _ = make_faithful_openai_transport()
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert len(results) == 13
    for oid in ("o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
                "o10", "o11", "o12", "o13"):
        assert by_id[oid].status == "pass", f"{oid}: {by_id[oid].detail}"
    # 忠实 mock 对 /v1/messages 返回 404 → 对侧端点指纹入签名
    assert "/v1/messages=404" in by_id["o13"].detail


@pytest.mark.asyncio
async def test_openai_model_echo_fail(settings):
    """响应 model 换成 gpt-4o-mini → o2 fail（中转站换模直接暴露）。"""
    transport, _ = make_faithful_openai_transport(echo_model="gpt-4o-mini")
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o2"].status == "fail"
    assert "gpt-4o" in by_id["o2"].detail


@pytest.mark.asyncio
async def test_openai_model_echo_rejects_same_family_downgrade(settings):
    """同属 GPT 家族也必须严格核对版本，GPT-5.5 不能被 GPT-4o 冒充。"""
    transport, _ = make_faithful_openai_transport(echo_model="gpt-4o")
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="gpt-5.5")
    results = await run_probes(c, "gpt-5.5", settings)
    await c.close()
    assert {r.id: r for r in results}["o2"].status == "fail"


@pytest.mark.asyncio
async def test_gpt4o_skips_reasoning_effort_by_profile(settings):
    transport, calls = make_faithful_openai_transport(echo_model="gpt-4o")
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="gpt-4o")
    results = await run_probes(c, "gpt-4o", settings)
    await c.close()

    effort = {r.id: r for r in results}["o10"]
    assert effort.status == "na"
    assert effort.applicable is False
    assert not any("reasoning_effort" in json.loads(req.read()) for req in calls)


@pytest.mark.asyncio
async def test_openai_stop_in_text_fail(settings):
    """输出包含 stop 序列本身 → o5 fail（疑似透传 anthropic 实现）。"""
    transport, _ = make_faithful_openai_transport(stop_in_text=True)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o5"].status == "fail"


@pytest.mark.asyncio
async def test_openai_effort_loose_na(settings):
    """非法 reasoning_effort 也被接受 → o10 na（宽松透传无判别力）。"""
    transport, _ = make_faithful_openai_transport(effort_strict=False)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o10"].status == "na"
    assert "宽松透传" in by_id["o10"].detail


@pytest.mark.asyncio
async def test_openai_effort_reject_all_na(settings):
    """合法 reasoning_effort 也被拒 → o10 na（无法区分不支持与严格校验）。"""
    transport, _ = make_faithful_openai_transport(effort_reject_all=True)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o10"].status == "na"
    assert "合法值也被拒" in by_id["o10"].detail


@pytest.mark.asyncio
async def test_openai_replay_fail(settings):
    """相同请求返回固定 id + 固定文本 → o12 fail（缓存/重放）。"""
    transport, _ = make_faithful_openai_transport(replay=True)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o12"].status == "fail", by_id["o12"].detail


@pytest.mark.asyncio
async def test_openai_usage_drift_fail(settings):
    """同 prompt 两次 prompt_tokens 漂移（42→60）→ o11 fail。"""
    transport, _ = make_faithful_openai_transport(usage_drift=True)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o11"].status == "fail"
    assert "42/60" in by_id["o11"].detail


@pytest.mark.asyncio
async def test_run_probes_unknown_protocol_single_na(settings):
    """协议识别失败 → run_probes 返回单个 p0 na，不执行任何探针。"""
    async def handler(request):
        return httpx.Response(404, json={"detail": "Not Found"})

    c = ApiClient("https://relay.example.com", "k",
                  transport=httpx.MockTransport(handler))
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    assert len(results) == 1
    assert results[0].id == "p0"
    assert results[0].status == "na"


@pytest.mark.asyncio
async def test_openai_effort_rate_limited_na(settings):
    """非法 reasoning_effort 撞限流返回 429（合法值 200）→ o10 na，不得误报 pass。"""
    transport, _ = make_faithful_openai_transport(bad_effort_status=429)
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    results = await run_probes(c, "kimi-k3", settings)
    await c.close()
    by_id = {r.id: r for r in results}
    assert by_id["o10"].status == "na"
    assert "429" in by_id["o10"].detail
    assert "无法判定" in by_id["o10"].detail


@pytest.mark.asyncio
async def test_multi_turn_max_tokens_reserved():
    """两组多轮验证共六次请求，均为 thinking 型上游预留 max_tokens=600。"""
    transport, calls = make_faithful_transport()
    c = ApiClient("https://relay.example.com", "k", transport=transport,
                  default_model="claude-sonnet-4-5")
    result = await p_multi_turn(c)
    await c.close()
    assert result.status == "pass", result.detail
    # calls 首条是 complete 内部 detect_protocol 的探测请求，按校验码过滤两组三轮。
    multi = [json.loads(r.read()) for r in calls
             if any("会话校验码" in str(m.get("content", ""))
                    for m in json.loads(r.read()).get("messages", []))]
    assert len(multi) == 6
    for body in multi:
        assert body["max_tokens"] == 600


# ---------- 参数校验矩阵（o13 / p16 渠道指纹） ----------

def make_param_strict_transport(**overrides):
    """官方渠道形态 mock：/v1/chat/completions 只放行 temperature=1.0 与
    top_p ∈ [0,1]，其余 400；/v1/messages 返回 404（官方无该端点直连形态）。
    overrides: openai_alt_status 可改 /v1/messages 状态码（404=官方 / 400=转承网关）。
    """
    alt_status = overrides.get("openai_alt_status", 404)

    async def handler(request):
        path = request.url.path
        if path == "/v1/messages":
            return httpx.Response(alt_status, json={"error": {
                "message": "not found", "type": "invalid_request_error"}})
        if path != "/v1/chat/completions":
            return httpx.Response(404, json={})
        body = json.loads(request.read())
        temp = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        if temp != 1.0 or not (0.0 <= top_p <= 1.0):
            return httpx.Response(400, json={"error": {
                "message": "Parameter validation error", "type": "invalid_request_error"}})
        return httpx.Response(200, json={
            "id": "chatcmpl-x", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_param_matrix_official_shape(settings):
    """官方形态：只放行 t1.0/p0，其余 400；/v1/messages=404 → 签名逐项可读。"""
    transport = make_param_strict_transport()
    c = ApiClient("https://gw.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    r = await o_param_matrix(c, "kimi-k3")
    await c.close()
    assert r.status == "pass"
    for frag in ("t0.0=400", "t0.6=400", "t1.0=200", "t1.5=400", "t3.0=400",
                 "p1.5=400", "p0=200", "/v1/messages=404"):
        assert frag in r.detail, r.detail
    # 默认 3 次重复：全一致以 ×3 压缩展示
    assert "t1.0=200×3" in r.detail and "t0.6=400×3" in r.detail
    assert "3次/变体" in r.detail


@pytest.mark.asyncio
async def test_param_matrix_relay_shape(settings):
    """转承形态：/v1/messages 返回 400（网关校验错误体）→ 与官方 404 可区分。"""
    transport = make_param_strict_transport(openai_alt_status=400)
    c = ApiClient("https://gw.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    r = await o_param_matrix(c, "kimi-k3")
    await c.close()
    assert "/v1/messages=400" in r.detail


@pytest.mark.asyncio
async def test_param_matrix_anthropic_variant(settings):
    """p16（anthropic 表）：矩阵走 /v1/messages，对侧探 /v1/chat/completions。"""
    async def handler(request):
        path = request.url.path
        if path == "/v1/chat/completions":
            return httpx.Response(404, json={"detail": "Not Found"})
        body = json.loads(request.read())
        if body.get("temperature", 1.0) not in (0, 1.0) or body.get("top_p", 1.0) > 1.0:
            return httpx.Response(400, json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad param"}})
        return httpx.Response(200, json={
            "id": "msg_x", "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "stop_reason": "end_turn"})

    c = ApiClient("https://gw.example.com", "k", transport=httpx.MockTransport(handler),
                  default_model="claude-sonnet-4-5")
    r = await p_param_matrix(c, "claude-sonnet-4-5")
    await c.close()
    assert r.status == "pass"
    assert "t0.6=400" in r.detail and "t1.0=200" in r.detail
    assert "/v1/chat=404" in r.detail


def test_param_matrix_probe_weight_zero():
    """o13/p16 为渠道指纹探针：weight=0，不参与 L1 得分（只做并排展示）。"""
    weights = {pid: w for pid, _, _, w, _ in PROBES_OPENAI}
    assert weights["o13"] == 0.0
    weights_p = {pid: w for pid, _, _, w, _ in PROBES}
    assert weights_p["p16"] == 0.0


@pytest.mark.asyncio
async def test_param_matrix_intermittent_marked(settings):
    """网关校验间歇开关（同参数先拒后放）→ 签名带 (间歇) 标记，不吞混合形态。"""
    seen = {}

    async def handler(request):
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404, json={})
        body = json.loads(request.read())
        if body.get("temperature") == 0.6:
            seen["t06"] = seen.get("t06", 0) + 1
            if seen["t06"] <= 2:  # 前 2 次拒绝、第 3 次放行 → 混合形态
                return httpx.Response(400, json={"error": {"message": "validation"}})
        return httpx.Response(200, json={
            "id": "chatcmpl-x", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    c = ApiClient("https://gw.example.com", "k",
                  transport=httpx.MockTransport(handler), default_model="kimi-k3")
    r = await o_param_matrix(c, "kimi-k3", settings)
    await c.close()
    assert r.status == "pass"
    assert "t0.6=400/400/200(间歇)" in r.detail, r.detail
    assert "t1.0=200×3" in r.detail  # 其他变体不受影响


@pytest.mark.asyncio
async def test_param_matrix_repeat_configurable(settings):
    """PARAM_MATRIX_REPEAT=1 → 每变体单次、签名不带 ×n（慢网关可调低）。"""
    from dataclasses import replace as dc_replace
    s1 = dc_replace(settings, PARAM_MATRIX_REPEAT=1)
    transport = make_param_strict_transport()
    c = ApiClient("https://gw.example.com", "k", transport=transport,
                  default_model="kimi-k3")
    r = await o_param_matrix(c, "kimi-k3", s1)
    await c.close()
    assert "t0.6=400 " in r.detail and "t0.6=400×" not in r.detail
    assert "1次/变体" in r.detail


@pytest.mark.asyncio
async def test_param_matrix_dead_gateway_na(settings):
    """全 429 端点：首变体即熔断中止矩阵（不再打满 24 请求），签名记 na
    ——限流下继续请求无信息量（2026-09-02 渠道B 全矩阵 429 事故）。"""
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    c = ApiClient("https://gw.example.com", "k",
                  transport=httpx.MockTransport(handler), default_model="kimi-k3")
    r = await o_param_matrix(c, "kimi-k3", settings)
    await c.close()
    assert r.status == "na"
    assert r.rate_limited is True
    assert "t0.0=429×3" in r.detail and "矩阵中止" in r.detail
    # 熔断：首变体 3 请求后即停，而不是 7 变体×3 + 对侧 3 = 24 请求
    assert calls["n"] == 3

    # 至少一个 2xx（端点在服务）→ pass
    calls = {"n": 0}

    async def handler2(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, json={})
        return httpx.Response(200, json={
            "id": "chatcmpl-x", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}})

    c2 = ApiClient("https://gw.example.com", "k",
                   transport=httpx.MockTransport(handler2), default_model="kimi-k3")
    r2 = await o_param_matrix(c2, "kimi-k3", settings)
    await c2.close()
    assert r2.status == "pass"
