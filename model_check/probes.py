"""L1 协议层：黑盒探针（anthropic 表 16 个 / openai 表 13 个，均含 1 个 weight=0 渠道指纹探针）。

每个探针三态：pass / fail / na（不可用，如信息缺失或异常）。
设计原则（spec §3）：
- 探针 3 响应头为弱信号（可伪造）
- 探针 12 依赖"服从率基线"判读（阶段 1 采集，当前用保守阈值）
- 探针 15 短响应（< MIN_ARTIFACT_LEN）判 na，避免空回复误判
- 信息缺失一律 na 不判 fail（限制信息暴露对抗）
"""

import asyncio
import base64
from dataclasses import dataclass, replace
from types import SimpleNamespace

from model_check.clients import ApiClient, ProtocolError
from model_check.config import async_timeout
from model_check.model_profiles import (model_echo_matches, probe_applicability,

                                resolve_model_profile)

# token 审计比较时的固定开销容差（system/role 封装等结构性偏移，非数值误差）
# p5：reported 与 counted 本就有十几 token 的结构性差值，需在百分比之外扣除
_TOKEN_AUDIT_FIXED_OFFSET = 15

# 64×64 不透明纯红 PNG（base64，RGBA 255/0/0/255）
# 注：不能用 1×1 图——像素过小模型无法辨识颜色（会答成黑色），alpha 半透明亦呈黑色
RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAYklEQVR42u3QMREAAAgAoe9fWnN4MlCApuazBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAAAECBAgQIECAgPsWQ4jh0jwfLk0AAAAASUVORK5CYII="
)
# 最小合法 PDF（含 "HELLO MOM" 文本对象）
MINI_PDF = b"""%PDF-1.1
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 20 100 Td (HELLO MOM) Tj ET
endstream endobj
xref
0 5
trailer<</Size 5/Root 1 0 R>>
%%EOF
"""


@dataclass
class ProbeResult:
    id: str
    name: str
    status: str  # "pass" | "fail" | "na"
    detail: str
    weight: float = 1.0
    # 该探针因 429 限流失败（探针层熔断的计数信号；o10 等自查状态码的 na 不置位）
    rate_limited: bool = False
    # 仅供编排层区分「检测不通过」与「检测未完成」，不对公网返回。
    error_code: str = ""
    upstream_status: int | None = None
    # False 表示模型能力配置明确认为该项不适用；运行异常产生的 na 仍为 True，
    # 两者在证据覆盖率中必须区别对待。
    applicable: bool = True


# ---------- 探针实现 ----------

THINK_RETRY_FACTOR = 3
THINK_RETRY_CAP = 2400


async def _complete_with_think_headroom(client: ApiClient, messages, *, max_tokens,
                                        extra=None):
    """thinking 型上游自适应：正文为空时放大预算重试一次。

    固定预算对推理长度波动（数百~近两千 token）天然不稳，任何取值都会偶发
    吃穿。不校验 stop_reason——部分网关 thinking 吃满预算时仍谎报 end_turn；
    正文为空本身即重试信号，健康渠道仅在真异常时多花一次请求。
    """
    r = await client.complete(messages, max_tokens=max_tokens, extra=extra)
    if not (r.get("text") or "").strip():
        r = await client.complete(messages, max_tokens=min(
            max_tokens * THINK_RETRY_FACTOR, THINK_RETRY_CAP), extra=extra)
    return r


async def _safe(fn, pid: str = "?", pname: str = "error"):
    """统一容错：任何异常 → na（归位到对应探针，避免污染统计表）。

    429 限流单独置 rate_limited 标记：探针层熔断据此计数（o10 等探针自查
    原始状态码、不抛异常，天然不会误触熔断）。
    """
    try:
        return await fn()
    except ProtocolError as e:
        status = getattr(e, "status", None)
        code = "rate_limited" if status == 429 else "auth" if status in (401, 403) \
            else "upstream" if status is not None else "transport"
        return ProbeResult(pid, pname, "na",
                           f"探针执行异常: {type(e).__name__}: {e}"[:200],
                           rate_limited=status == 429, error_code=code,
                           upstream_status=status)
    except Exception as e:  # noqa: BLE001 —— 探针层兜底，异常一律判 na 不中断
        return ProbeResult(pid, pname, "na",
                           f"探针执行异常: {type(e).__name__}: {e}"[:200],
                           error_code="system")


async def p_connectivity(client: ApiClient) -> ProbeResult:
    """1. 连通性+响应结构：HTTP 200 + msg_ 消息 ID + usage 字段齐全。"""
    async def _run():
        r = await client.complete([{"role": "user", "content": "ping"}], max_tokens=10)
        raw = r.get("raw") or {}
        ok_id = str(raw.get("id", "")).startswith("msg_")
        usage = raw.get("usage") or {}
        ok_usage = "input_tokens" in usage and "output_tokens" in usage
        if r["text"] is not None and ok_id and ok_usage:
            return ProbeResult("p1", "连通性+响应结构", "pass", f"id={raw.get('id')}")
        return ProbeResult("p1", "连通性+响应结构", "fail", f"id 或 usage 缺失: {str(raw)[:200]}")
    return await _safe(_run, "p1", "连通性+响应结构")


async def p_model_echo(client: ApiClient, declared: str) -> ProbeResult:
    """2. model 回显：严格匹配声明名称，仅容许渠道前缀和日期版本后缀。"""
    async def _run():
        r = await client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        echo = (r.get("raw") or {}).get("model", "") or ""
        if model_echo_matches(declared, echo):
            return ProbeResult("p2", "model 回显", "pass", f"echo={echo}")
        return ProbeResult("p2", "model 回显", "fail", f"echo={echo} 与声明 {declared} 不一致")
    return await _safe(_run, "p2", "model 回显")


async def p_header_fingerprint(client: ApiClient) -> ProbeResult:
    """3. 响应头指纹（弱信号）：request-id/cf-ray/anthropic-ratelimit-* ≥2 个。"""
    async def _run():
        r = await client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        h = r.get("headers") or {}
        marks = [k for k in ("request-id", "cf-ray", "anthropic-version")
                 if k in h] + [k for k in h if k.startswith("anthropic-ratelimit-")]
        if len(set(marks)) >= 2:
            return ProbeResult("p3", "响应头指纹", "pass", f"found={sorted(set(marks))}")
        if len(set(marks)) == 1:
            return ProbeResult("p3", "响应头指纹", "na", f"仅 {len(set(marks))} 个官方头（弱信号）")
        return ProbeResult("p3", "响应头指纹", "fail", "无 Anthropic 官方响应头")
    return await _safe(_run, "p3", "响应头指纹")


async def p_sse_shape(client: ApiClient) -> ProbeResult:
    """4. SSE 流式事件：6 类事件+固定顺序（容忍 ping 穿插）。"""
    async def _run():
        events = await client.stream_events(
            [{"role": "user", "content": "hi"}], max_tokens=5)
        interesting = [e for e in events if e != "ping"]
        order_ok = (
            "message_start" in interesting
            and "content_block_delta" in interesting
            and "message_stop" in interesting
            and interesting.index("message_start") < interesting.index("message_stop")
        )
        if order_ok:
            return ProbeResult("p4", "SSE 流式事件", "pass", f"events={interesting[:8]}")
        return ProbeResult("p4", "SSE 流式事件", "fail" if events else "na",
                           f"事件序列异常: {events[:10]}")
    return await _safe(_run, "p4", "SSE 流式事件")


async def p_count_tokens_match(client: ApiClient) -> ProbeResult:
    """5. token 审计：count_tokens 独立重算 vs usage.input_tokens。

    比较基准：用同一 message 结构（role+content）重算，而非裸 text——
    否则 usage 中的结构性开销（system/role 标签封装）会被误判为偏差。
    容差：百分比误差 ≤10% 或绝对误差 ≤ 固定开销阈值（避免短文本误判）。
    """
    async def _run():
        text = "这是一个用于 token 审计的测试文本。" * 10
        msgs = [{"role": "user", "content": text}]
        r = await client.complete(msgs, max_tokens=5)
        reported = r.get("usage_in", 0)
        # 用同一 message 结构重算，使基准与 usage 口径一致
        counted = await client.count_tokens_messages(msgs)
        if counted < 0:
            return ProbeResult("p5", "token 审计", "na", "count_tokens 端点不可用")
        if reported > 0 and (abs(counted - reported) / max(reported, 1) <= 0.10
                             or abs(counted - reported) <= _TOKEN_AUDIT_FIXED_OFFSET):
            return ProbeResult("p5", "token 审计", "pass", f"reported={reported} counted={counted}")
        return ProbeResult("p5", "token 审计", "fail",
                           f"reported={reported} counted={counted} 偏差>10%")
    return await _safe(_run, "p5", "token 审计")


async def p_cache_behavior(client: ApiClient, declared: str) -> ProbeResult:
    """6. prompt caching：按版本最小缓存前缀长度两连发，第二次 cache_read>0。"""
    async def _run():
        # 声明模型含 4.x → 4096；3.x/3.5 → 1024；unknown → 4096
        if "3.5" in declared or "3-" in declared or declared.startswith("claude-3"):
            prefix_len = 1024
        else:
            prefix_len = 4096
        # 中文约 1.3 token/字，按 3 倍字符数生成，确保前缀 token 数达标
        prefix = ("缓存前缀填充内容。测试前缀稳定性。" * (prefix_len // 10 + 10))[: prefix_len * 3]
        msgs = [{"role": "user", "content": prefix + "你好"}]
        r1 = await client.complete(msgs, max_tokens=5)
        r2 = await client.complete(msgs, max_tokens=5)
        u1 = (r1.get("raw") or {}).get("usage") or {}
        u2 = (r2.get("raw") or {}).get("usage") or {}
        created = u1.get("cache_creation_input_tokens", 0)
        read = u2.get("cache_read_input_tokens", 0)
        if created > 0 and read > 0:
            return ProbeResult("p6", "prompt caching", "pass",
                               f"created={created} read={read}")
        if "cache_creation_input_tokens" not in u1:
            return ProbeResult("p6", "prompt caching", "na",
                               "响应无 cache 字段（前缀可能不足或服务不支持）")
        return ProbeResult("p6", "prompt caching", "fail",
                           f"created={created} read={read} 缓存行为异常")
    return await _safe(_run, "p6", "prompt caching")


async def p_error_shape(client: ApiClient, declared: str = "") -> ProbeResult:
    """7. 错误对象格式：{type:error, error:{type,message}}。"""
    async def _run():
        model = declared or client.default_model
        resp = await client.raw_messages({"model": model, "max_tokens": 0,
                                          "messages": [{"role": "user", "content": "hi"}]})
        if resp.status_code == 200:
            return ProbeResult("p7", "错误对象格式", "na", "max_tokens=0 未触发错误（服务宽容）")
        try:
            data = resp.json()
        except Exception:
            return ProbeResult("p7", "错误对象格式", "na", f"非 JSON 错误体: {resp.text[:100]}")
        err = data.get("error")
        if data.get("type") == "error" and isinstance(err, dict) \
                and "type" in err and "message" in err:
            return ProbeResult("p7", "错误对象格式", "pass",
                               f"错误码={err.get('type')} status={resp.status_code}")
        return ProbeResult("p7", "错误对象格式", "fail", f"错误结构异常: {str(data)[:200]}")
    return await _safe(_run, "p7", "错误对象格式")


async def p_stop_sequence(client: ApiClient) -> ProbeResult:
    """8. stop_sequence：stop_reason=stop_sequence + 输出以序列结尾。"""
    async def _run():
        marker = "STOPXYZ"
        # 预算 600：thinking 型上游先输出推理块再出正文，小预算正文为空必误判
        r = await _complete_with_think_headroom(
            client, [{"role": "user", "content": f"回复后输出 {marker}"}],
            max_tokens=600, extra={"stop_sequences": [marker]})
        hit = marker in r["text"]
        reason = r.get("stop_reason")
        # 词汇兼容：anthropic 命中记 stop_sequence；openai finish_reason=stop
        # 不区分自然结束与序列命中，以正文含序列为准
        if hit and reason in ("stop_sequence", "stop"):
            return ProbeResult("p8", "stop_sequence", "pass", f"text 尾部={r['text'][-20:]}")
        return ProbeResult("p8", "stop_sequence", "fail",
                           f"stop_reason={reason}")
    return await _safe(_run, "p8", "stop_sequence")


async def p_max_tokens_cutoff(client: ApiClient) -> ProbeResult:
    """9. max_tokens 截断：stop_reason=max_tokens + 输出不超限。"""
    async def _run():
        r = await client.complete([{"role": "user", "content": "写一段长文本，至少 500 字"}],
                                  max_tokens=3)
        if r.get("stop_reason") == "max_tokens" and len(r["text"]) < 50:
            return ProbeResult("p9", "max_tokens 截断", "pass", f"len={len(r['text'])}")
        return ProbeResult("p9", "max_tokens 截断", "fail",
                           f"stop_reason={r.get('stop_reason')} len={len(r.get('text', ''))}")
    return await _safe(_run, "p9", "max_tokens 截断")


async def p_tool_use(client: ApiClient) -> ProbeResult:
    """10. tool_use：工具调用格式 / toolu_* ID / stop_reason=tool_use。"""
    async def _run():
        tools = [{
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "input_schema": {"type": "object",
                             "properties": {"city": {"type": "string"}},
                             "required": ["city"]},
        }]
        r = await client.complete(
            [{"role": "user", "content": "帮我查一下北京的天气"}],
            max_tokens=100, extra={"tools": tools})
        content = (r.get("raw") or {}).get("content") or []
        blocks = [b for b in content if b.get("type") == "tool_use"]
        ok_id = any(str(b.get("id", "")).startswith("toolu_") for b in blocks)
        if blocks and ok_id:
            return ProbeResult("p10", "tool_use", "pass",
                               f"tool={blocks[0].get('name')} id={blocks[0].get('id')[:8]}...")
        if blocks:
            return ProbeResult("p10", "tool_use", "fail", "tool_use 块缺少 toolu_* ID")
        return ProbeResult("p10", "tool_use", "fail",
                           f"未返回工具调用 stop_reason={r.get('stop_reason')}")
    return await _safe(_run, "p10", "tool_use")


async def p_multi_turn(client: ApiClient) -> ProbeResult:
    """11. 多轮上下文：两组独立三轮对话复述中性校验码。

    避免使用“秘密/令牌”等可能触发内容过滤的词；两组中任意一组通过即判通过，
    只有两组均失败才判失败，降低单次生成波动造成的误报。
    """
    async def _run():
        passed = 0
        for marker in ("MC7A9", "QV4K2"):
            # max_tokens 600：给 thinking 型上游预留推理空间，避免正文为空。
            msgs = [{
                "role": "user",
                "content": f"请在本轮对话中记住会话校验码 {marker}，并回复“收到”。",
            }]
            r1 = await _complete_with_think_headroom(client, msgs, max_tokens=600)
            msgs.append({"role": "assistant", "content": r1["text"]})
            msgs.append({"role": "user", "content": "请复述刚才的会话校验码。"})
            r2 = await _complete_with_think_headroom(client, msgs, max_tokens=600)
            msgs.append({"role": "assistant", "content": r2["text"]})
            msgs.append({"role": "user", "content": "最后一次，请只回答该会话校验码。"})
            r3 = await _complete_with_think_headroom(client, msgs, max_tokens=600)
            if marker in (r3.get("text") or ""):
                passed += 1
                if getattr(client, "public_mode", False):
                    return ProbeResult("p11", "多轮记忆", "pass", "上下文验证通过")
        if passed:
            return ProbeResult("p11", "多轮记忆", "pass", f"上下文验证通过 {passed}/2")
        return ProbeResult("p11", "多轮记忆", "fail",
                           "两组独立会话均未能复述校验码")
    return await _safe(_run, "p11", "多轮记忆")


async def p_system_adherence(client: ApiClient) -> ProbeResult:
    """12. system prompt 服从：3 次强约束（首字母 Z），≥2 次符合 → pass。

    判读基线说明：不同模型天然服从度不同，透传也可能不服从；
    当前用保守阈值（3 次中 2 次），阶段 1 采集基线后校准。
    """
    async def _run():
        ok = 0
        details = []
        for i in range(3):
            r = await _complete_with_think_headroom(
                client,
                [{"role": "user", "content": f"回答任意一句话（第 {i + 1} 轮）"}],
                max_tokens=600,
                extra={"system": "你的每一条回复必须以字母 Z 开头，其他任何字符都不允许。"})
            text = (r["text"] or "").strip()
            if text.startswith("Z"):
                ok += 1
                details.append("Z")
            else:
                details.append("non-Z")
        if ok >= 2:
            return ProbeResult("p12", "system prompt 服从", "pass", f"服从 {ok}/3: {details}")
        return ProbeResult("p12", "system prompt 服从", "fail", f"服从 {ok}/3: {details}")
    return await _safe(_run, "p12", "system prompt 服从")


async def p_image_input(client: ApiClient) -> ProbeResult:
    """13. 图像输入：64×64 不透明纯红 PNG 识别。"""
    async def _run():
        img = {"type": "image", "source": {"type": "base64",
                                           "media_type": "image/png",
                                           "data": RED_PNG_B64}}
        r = await client.complete(
            [{"role": "user", "content": [img, {"type": "text",
                                                "text": "这张图片是什么颜色？"}]}],
            max_tokens=256)
        t = (r["text"] or "").lower()
        if "red" in t or "红" in t:
            return ProbeResult("p13", "图像输入", "pass", t[:50])
        return ProbeResult("p13", "图像输入", "fail", f"未能识别颜色: {t[:50]}")
    return await _safe(_run, "p13", "图像输入")


async def p_document_input(client: ApiClient) -> ProbeResult:
    """14. 文档输入：极小 PDF（含 HELLO MOM 文本）。解析失败 → na。"""
    async def _run():
        doc = {"type": "document", "source": {"type": "base64",
                                              "media_type": "application/pdf",
                                              "data": base64.b64encode(MINI_PDF).decode()}}
        r = await _complete_with_think_headroom(
            client,
            [{"role": "user", "content": [doc, {"type": "text",
                                                "text": "这个文档里写了什么？"}]}],
            max_tokens=600)
        t = (r["text"] or "").upper()
        if "HELLO MOM" in t or "HELLO" in t or "MOM" in t:
            return ProbeResult("p14", "文档输入", "pass", t[:50])
        if not r["text"]:
            return ProbeResult("p14", "文档输入", "na", "文档输入未返回内容")
        return ProbeResult("p14", "文档输入", "fail", f"未识别文档内容: {t[:50]}")
    return await _safe(_run, "p14", "文档输入")


async def p_cache_replay(client: ApiClient, settings) -> ProbeResult:
    """15. 缓存/重放检测：两次完全相同请求对比。

    - 响应体逐字节一致 且 request_id 相同 且 len ≥ MIN_ARTIFACT_LEN → fail（缓存/重放）
    - 输出不同但 cache_read>0 → pass（合法 prompt caching 命中）
    - 短响应（< MIN_ARTIFACT_LEN）→ na（不构成证据）
    """
    async def _run():
        prompt = ("请写一段不少于 60 个字的短文，介绍人工智能的历史发展，"
                  "要求内容具体、包含至少三个时间节点。" * 2)[:400]
        msgs = [{"role": "user", "content": prompt}]
        r1 = await _complete_with_think_headroom(client, msgs, max_tokens=600)
        r2 = await _complete_with_think_headroom(client, msgs, max_tokens=600)
        t1, t2 = r1["text"], r2["text"]
        if len(t1) < settings.MIN_ARTIFACT_LEN or len(t2) < settings.MIN_ARTIFACT_LEN:
            return ProbeResult("p15", "缓存/重放检测", "na",
                               f"短响应 len={len(t1)}/{len(t2)}（<{settings.MIN_ARTIFACT_LEN}）")
        same_body = t1 == t2
        same_id = r1["request_id"] == r2["request_id"] and bool(r1["request_id"])
        u2 = (r2.get("raw") or {}).get("usage") or {}
        cache_read = u2.get("cache_read_input_tokens", 0) > 0
        if same_body and same_id:
            return ProbeResult("p15", "缓存/重放检测", "fail",
                               "响应体逐字节一致且 request_id 相同（疑似缓存/重放）")
        if same_body and not same_id:
            return ProbeResult("p15", "缓存/重放检测", "na",
                               "响应体一致但 request_id 不同（可能命中输出缓存）")
        if cache_read:
            return ProbeResult("p15", "缓存/重放检测", "pass", "cache_read>0 且输出不同（合法命中）")
        return ProbeResult("p15", "缓存/重放检测", "pass", "两次输出正常不同")
    return await _safe(_run, "p15", "缓存/重放检测")


# ---------- OpenAI 兼容协议探针 ----------

async def o_connectivity(client: ApiClient) -> ProbeResult:
    """o1. 连通性+响应结构：200 + choices + prompt/completion_tokens 齐全。

    ID 只要求非空：OpenAI 兼容层允许聚合商自有 ID 格式（OpenRouter gen-*、
    部分网关日期串），强求 chatcmpl 前缀会把健康渠道误判 fail；
    结构判据由 choices + usage 承担。ID 格式差异本身入 detail 作渠道指纹。
    thinking 型上游小预算下正文可为 None（推理吃满预算）→ 空正文升额重试。
    """
    async def _run():
        r = await _complete_with_think_headroom(
            client, [{"role": "user", "content": "ping"}], max_tokens=10)
        raw = r.get("raw") or {}
        ok_id = bool(str(raw.get("id", "")).strip())
        usage = raw.get("usage") or {}
        ok_usage = "prompt_tokens" in usage and "completion_tokens" in usage
        has_choices = bool(raw.get("choices"))
        if (r.get("text") or "") != "" and ok_id and ok_usage and has_choices:
            return ProbeResult("o1", "连通性+响应结构", "pass", f"id={raw.get('id')}")
        if raw and ok_id and has_choices and not (r.get("text") or "").strip():
            return ProbeResult("o1", "连通性+响应结构", "na",
                               "结构齐全但正文为空（推理吃满预算），连通性存疑待复测")
        return ProbeResult("o1", "连通性+响应结构", "fail",
                           f"id/usage/choices 缺失: {str(raw)[:200]}")
    return await _safe(_run, "o1", "连通性+响应结构")


async def o_model_echo(client: ApiClient, declared: str) -> ProbeResult:
    """o2. model 回显：严格匹配声明名称，仅容许渠道前缀和日期版本后缀。"""
    async def _run():
        r = await client.complete([{"role": "user", "content": "hi"}], max_tokens=5)
        echo = (r.get("raw") or {}).get("model", "") or ""
        if model_echo_matches(declared, echo):
            return ProbeResult("o2", "model 回显", "pass", f"echo={echo}")
        return ProbeResult("o2", "model 回显", "fail", f"echo={echo} 与声明 {declared} 不一致")
    return await _safe(_run, "o2", "model 回显")


async def o_max_tokens_cutoff(client: ApiClient) -> ProbeResult:
    """o3. max_tokens 截断：finish_reason=length + 输出不超限。

    thinking 型上游可能把 3 token 全花在推理上（正文 None/空，实测 OpenRouter
    glm-5.3 content=None 致 len(None) TypeError）——正文为空但 length 语义正确
    仍算通过（截断被尊重），detail 注明推理吃满预算。
    """
    async def _run():
        r = await client.complete([{"role": "user", "content": "写一段长文本，至少 500 字"}],
                                  max_tokens=3)
        text = r.get("text") or ""
        if r.get("stop_reason") == "length" and len(text) < 50:
            note = "（正文为空：推理吃满预算，length 语义正确）" if not text.strip() else ""
            return ProbeResult("o3", "max_tokens 截断", "pass", f"len={len(text)}{note}")
        return ProbeResult("o3", "max_tokens 截断", "fail",
                           f"finish_reason={r.get('stop_reason')} len={len(text)}")
    return await _safe(_run, "o3", "max_tokens 截断")


async def o_sse_shape(client: ApiClient) -> ProbeResult:
    """o4. SSE 流式：归一化事件 message_start + content_delta + 结尾 done 终止标记。

    thinking 型上游（reasoning_content 先行）在小预算下正文为空 → 无
    content_delta（实测 o4 max_tokens=5 时事件只剩 message_start/usage/done），
    与真流式缺陷同形。缺 content_delta 但流未断 → 放大预算重试一次后再判。
    """
    async def _run():
        events = await client.stream_openai_events(
            [{"role": "user", "content": "hi"}], max_tokens=5)
        if events and "content_delta" not in events:
            events = await client.stream_openai_events(
                [{"role": "user", "content": "hi"}], max_tokens=600)
        order_ok = ("message_start" in events and "content_delta" in events
                    and events[-1] == "done")
        if order_ok:
            return ProbeResult("o4", "SSE 流式", "pass", f"events={events[:8]}")
        return ProbeResult("o4", "SSE 流式", "fail" if events else "na",
                           f"事件序列异常: {events[:10]}")
    return await _safe(_run, "o4", "SSE 流式")


async def o_stop_param(client: ApiClient) -> ProbeResult:
    """o5. stop 参数：openai 语义——命中截断后输出**不含**序列本身（与 anthropic 相反）。

    anthropic：stop_reason=stop_sequence 且输出含 marker；
    openai：finish_reason=stop 且输出被截断在 marker 之前（marker 不出现在文本中）。
    thinking 型上游 reasoning 吃满小预算 → finish_reason=length 假失败，
    与 p8 同预算（600）并走空正文升额重试。
    """
    async def _run():
        marker = "STOPXYZ"
        r = await _complete_with_think_headroom(
            client, [{"role": "user", "content": f"回复后输出 {marker}"}],
            max_tokens=600, extra={"stop_sequences": [marker]})
        if r.get("stop_reason") == "stop" and r["text"] and marker not in r["text"]:
            return ProbeResult("o5", "stop 参数", "pass", f"text 尾部={r['text'][-20:]}")
        if r.get("stop_reason") == "stop" and marker in r["text"]:
            return ProbeResult("o5", "stop 参数", "fail",
                               "输出包含 stop 序列本身（非标准 openai 语义，疑似透传 anthropic 实现）")
        return ProbeResult("o5", "stop 参数", "fail",
                           f"finish_reason={r.get('stop_reason')}")
    return await _safe(_run, "o5", "stop 参数")


async def o_multi_turn(client: ApiClient) -> ProbeResult:
    """o6. 多轮记忆：复用 p11 判定逻辑（消息结构两协议兼容），仅替换探针 ID。"""
    return replace(await p_multi_turn(client), id="o6")


async def o_system_adherence(client: ApiClient) -> ProbeResult:
    """o7. system 服从：复用 p12（system 字段由客户端层转为 role=system 消息）。"""
    return replace(await p_system_adherence(client), id="o7")


async def o_tool_use(client: ApiClient) -> ProbeResult:
    """o8. 工具调用：message.tool_calls + id + function.name（openai 工具 schema）。"""
    async def _run():
        tools = [{"type": "function",
                  "function": {"name": "get_weather",
                               "description": "查询指定城市的天气",
                               "parameters": {"type": "object",
                                              "properties": {"city": {"type": "string"}},
                                              "required": ["city"]}}}]
        r = await client.complete(
            [{"role": "user", "content": "帮我查一下北京的天气"}],
            max_tokens=100, extra={"tools": tools})
        choice = ((r.get("raw") or {}).get("choices") or [{}])[0]
        calls = (choice.get("message") or {}).get("tool_calls") or []
        first = calls[0] if calls else {}
        fn = first.get("function") or {}
        if calls and first.get("id") and fn.get("name"):
            return ProbeResult("o8", "工具调用", "pass",
                               f"tool={fn.get('name')} id={str(first.get('id'))[:8]}...")
        if calls:
            return ProbeResult("o8", "工具调用", "fail", "tool_calls 缺少 id/function.name")
        return ProbeResult("o8", "工具调用", "fail",
                           f"未返回工具调用 finish_reason={r.get('stop_reason')}")
    return await _safe(_run, "o8", "工具调用")


async def o_vision_input(client: ApiClient) -> ProbeResult:
    """o9. 视觉输入：image_url（data URL）形式发送 64×64 不透明纯红 PNG 并识别颜色。"""
    async def _run():
        img = {"type": "image_url",
               "image_url": {"url": f"data:image/png;base64,{RED_PNG_B64}"}}
        r = await client.complete(
            [{"role": "user", "content": [img, {"type": "text",
                                                "text": "这张图片是什么颜色？"}]}],
            max_tokens=256)
        t = (r["text"] or "").lower()
        if "red" in t or "红" in t:
            return ProbeResult("o9", "视觉输入", "pass", t[:50])
        return ProbeResult("o9", "视觉输入", "fail", f"未能识别颜色: {t[:50]}")
    return await _safe(_run, "o9", "视觉输入")


async def o_reasoning_effort(client: ApiClient, declared: str) -> ProbeResult:
    """o10. reasoning_effort 特征（kimi-k3 等）：双信号判别防误报。

    仅凭合法值 200 无法区分「支持」与「宽松透传」，
    仅凭非法值 4xx 无法区分「严格校验」与「不支持」；
    合法值被接受 且 非法值被参数校验类状态码（400/404/422）拒绝 → pass；
    其余组合判 na（无判别力）。
    注意：401/403（鉴权）、408（超时）、429（限流）、5xx 与参数校验无关，
    不计入「拒绝」，避免探测期撞限流时把 429 误报成 K3 严格校验特征。
    """
    async def _run():
        model = declared or client.default_model
        token_field = "max_completion_tokens" \
            if resolve_model_profile(model).reasoning_effort is True else "max_tokens"
        good = await client.raw_chat({"model": model, token_field: 5,
                                      "messages": [{"role": "user", "content": "hi"}],
                                      "reasoning_effort": "max"})
        bad = await client.raw_chat({"model": model, token_field: 5,
                                     "messages": [{"role": "user", "content": "hi"}],
                                     "reasoning_effort": "ultra_max"})
        good_ok = good.status_code == 200
        # 仅参数校验类状态码视为「拒绝」，鉴权/限流/超时/5xx 一律不算
        bad_rejected = bad.status_code in (400, 404, 422)
        if good_ok and bad_rejected:
            return ProbeResult("o10", "reasoning_effort 特征", "pass",
                               f"max 被接受 / ultra_max 被拒（{bad.status_code}）")
        if not good_ok and bad_rejected:
            return ProbeResult("o10", "reasoning_effort 特征", "na",
                               "合法值也被拒：服务不支持或严格校验该参数")
        if good_ok and bad.status_code >= 400:
            return ProbeResult("o10", "reasoning_effort 特征", "na",
                               f"bad 请求异常响应 {bad.status_code}"
                               "（鉴权/限流/服务端错误），无法判定参数校验，建议重测")
        if good_ok:
            return ProbeResult("o10", "reasoning_effort 特征", "na",
                               "非法值也被接受：reasoning_effort 被宽松透传，无判别力")
        return ProbeResult("o10", "reasoning_effort 特征", "na",
                           f"good={good.status_code} bad={bad.status_code}")
    return await _safe(_run, "o10", "reasoning_effort 特征")


async def o_usage_consistency(client: ApiClient) -> ProbeResult:
    """o11. usage 一致性：同 prompt 两次请求 prompt_tokens 偏差 ≤5%（计数漂移提示异常路由）。"""
    async def _run():
        prompt = "这是一个用于 usage 一致性审计的固定文本。" * 8
        msgs = [{"role": "user", "content": prompt}]
        r1 = await client.complete(msgs, max_tokens=5)
        r2 = await client.complete(msgs, max_tokens=5)
        u1, u2 = r1.get("usage_in", 0), r2.get("usage_in", 0)
        if u1 <= 0 or u2 <= 0:
            return ProbeResult("o11", "usage 一致性", "na", f"usage 缺失: {u1}/{u2}")
        diff = abs(u1 - u2) / max(u1, u2)
        if diff <= 0.05:
            return ProbeResult("o11", "usage 一致性", "pass", f"prompt_tokens {u1}/{u2}")
        return ProbeResult("o11", "usage 一致性", "fail",
                           f"同 prompt prompt_tokens 漂移 {u1}/{u2}（{diff:.0%}）")
    return await _safe(_run, "o11", "usage 一致性")


async def o_cache_replay(client: ApiClient, settings) -> ProbeResult:
    """o12. 缓存/重放检测：复用 p15（openai 无 cache_read 字段，走重放判定分支）。"""
    return replace(await p_cache_replay(client, settings), id="o12")


# ---------- 参数校验矩阵（渠道指纹，两协议通用） ----------

# 常规 + 越界采样参数：官方渠道常见形态 = 只放行 temperature=1.0、越界值 400；
# 转承渠道常见形态 = 全放行或全 400（网关校验层）。原始状态码直接入签名，
# 不做 ✓/✗ 归一——矩阵形态本身就是指纹，归一化会丢信息
_PARAM_VARIANTS = [
    ("t0.0", {"temperature": 0.0}),
    ("t0.6", {"temperature": 0.6}),
    ("t1.0", {"temperature": 1.0}),
    ("t1.5", {"temperature": 1.5}),
    ("t3.0", {"temperature": 3.0}),
    ("p1.5", {"top_p": 1.5}),
    ("p0", {"top_p": 0.0}),
]


def _fmt_reps(statuses: list[str]) -> str:
    """重复结果压缩展示：全一致 → 400×3；混合 → 400/200/400(间歇)。

    间歇标记是三渠道审计的关键发现之一（top_p 校验同日上下午结论翻转），
    混合形态本身就有诊断价值，不能只报众数。
    """
    if len(set(statuses)) == 1:
        return statuses[0] if len(statuses) == 1 else f"{statuses[0]}×{len(statuses)}"
    return "/".join(statuses) + "(间歇)"


async def _param_fingerprint(client: ApiClient, declared: str, pid: str,
                             name: str, primary: str, settings=None) -> ProbeResult:
    """参数校验矩阵 + 对侧端点探测（渠道指纹，weight=0，不计入 L1 得分）。

    来源：三渠道对照审计（2026-08-31，官方/天翼云/江苏电信）结论——
    ① 参数校验矩阵是最稳定的渠道指纹（官方只收 temperature=1.0、越界 400，
    转承全放行，60/60 重复零抖动）；② `/v1/messages` 端点探测官方 404 vs
    转承 400（网关校验错误体）形态稳定。
    每变体按 settings.PARAM_MATRIX_REPEAT 重复（审计标准：单次探针不可下结论，
    ×3 起步；同分钟内重复可抓间歇开关，跨时段仍需多次运行对比）。
    primary: "chat"（openai 探针）或 "messages"（anthropic 探针），
    对侧端点同样重复采样作为补充指纹。
    """
    model = declared or client.default_model
    reps = max(1, getattr(settings, "PARAM_MATRIX_REPEAT", 3) or 3)
    sig_parts: list[str] = []
    statuses_flat: list[str] = []  # 全部变体 + 对侧端点的原始状态码（pass 判据用）

    async def _send(primary_body: dict, use_chat: bool):
        return (await client.raw_chat(primary_body) if use_chat
                else await client.raw_messages(primary_body))

    aborted = False
    for label, extra in _PARAM_VARIANTS:
        body = {"model": model, "max_tokens": 5,
                "messages": [{"role": "user", "content": "hi"}], **extra}
        statuses: list[str] = []
        for _ in range(reps):
            try:
                resp = await _send(body, primary == "chat")
                statuses.append(str(resp.status_code))
            except Exception:  # noqa: BLE001 —— 传输层异常记 ? 不中断矩阵
                statuses.append("?")
        statuses_flat.extend(statuses)
        sig_parts.append(f"{label}={_fmt_reps(statuses)}")
        # 限流熔断：整变体全 429 说明渠道配额已耗尽，剩余变体只会继续 429
        #（实测渠道B 全矩阵 24 请求全 429，无信息量纯浪费），保留已测部分即止
        if statuses and all(s == "429" for s in statuses):
            aborted = True
            break
    if not aborted:
        # 对侧端点探测：openai 渠道探 /v1/messages（404 vs 400 判别），anthropic 反之
        alt = "messages" if primary == "chat" else "chat"
        probe_body = {"model": model, "max_tokens": 1,
                      "messages": [{"role": "user", "content": "hi"}]}
        alt_statuses: list[str] = []
        for _ in range(reps):
            try:
                resp = await _send(probe_body, alt == "chat")
                alt_statuses.append(str(resp.status_code))
            except Exception:  # noqa: BLE001
                alt_statuses.append("?")
        statuses_flat.extend(alt_statuses)
        sig_parts.append(f"/v1/{alt}={_fmt_reps(alt_statuses)}")
    else:
        sig_parts.append("…=429中止")
    if aborted:
        detail = " ".join(sig_parts) + "（渠道限流：变体全 429，矩阵中止；待配额恢复后重测）"
        return ProbeResult(pid, name, "na", detail, weight=0.0, rate_limited=True)
    detail = " ".join(sig_parts) + f"（{reps}次/变体；跨时段结论需多次运行）"
    # pass 判据：至少一个 2xx（端点确实在提供补全）。全 4xx/5xx 说明端点
    # 死亡/拒绝服务——矩阵照常入签名，但探针记 na，不污染 L1 pass 计数
    # （否则全灭网关会击穿「L1 全灭→无法判定」早退，误报无法归因）
    serving = any(s.startswith("2") for s in statuses_flat)
    return ProbeResult(pid, name, "pass" if serving else "na", detail, weight=0.0)


async def o_param_matrix(client: ApiClient, declared: str, settings=None) -> ProbeResult:
    """o13. 参数校验矩阵（openai 渠道指纹）：越界参数接受形态 + /v1/messages 探测。"""
    return await _param_fingerprint(client, declared, "o13", "参数校验矩阵",
                                    "chat", settings)


async def p_param_matrix(client: ApiClient, declared: str, settings=None) -> ProbeResult:
    """p16. 参数校验矩阵（anthropic 渠道指纹）：越界参数接受形态 + /v1/chat/completions 探测。"""
    return await _param_fingerprint(client, declared, "p16", "参数校验矩阵",
                                    "messages", settings)


# ---------- 编排 ----------

def _no_args(client, ctx, fn):
    """探针只需 client。"""
    return fn(client)


def _declared_args(client, ctx, fn):
    """探针需要 declared model。"""
    return fn(client, ctx.declared)


def _settings_args(client, ctx, fn):
    """探针需要 settings。"""
    return fn(client, ctx.settings)


def _declared_settings_args(client, ctx, fn):
    """探针需要 declared model + settings（如参数校验矩阵的重复次数）。"""
    return fn(client, ctx.declared, ctx.settings)


# 探针表：(id, name, fn, weight, dispatcher)
# dispatcher 决定如何把 ctx 注入 fn —— 新增探针只需在此登记，无需改 run_one（开闭原则）
PROBES = [
    ("p1", "连通性+响应结构", p_connectivity, 3.0, _no_args),
    ("p2", "model 回显", p_model_echo, 2.0, _declared_args),
    ("p3", "响应头指纹", p_header_fingerprint, 1.0, _no_args),
    ("p4", "SSE 流式事件", p_sse_shape, 2.0, _no_args),
    ("p5", "token 审计", p_count_tokens_match, 2.0, _no_args),
    ("p6", "prompt caching", p_cache_behavior, 2.0, _declared_args),
    ("p7", "错误对象格式", p_error_shape, 1.0, _declared_args),
    ("p8", "stop_sequence", p_stop_sequence, 2.0, _no_args),
    ("p9", "max_tokens 截断", p_max_tokens_cutoff, 2.0, _no_args),
    ("p10", "tool_use", p_tool_use, 3.0, _no_args),
    ("p11", "多轮记忆", p_multi_turn, 2.0, _no_args),
    ("p12", "system prompt 服从", p_system_adherence, 2.0, _no_args),
    ("p13", "图像输入", p_image_input, 2.0, _no_args),
    ("p14", "文档输入", p_document_input, 2.0, _no_args),
    ("p15", "缓存/重放检测", p_cache_replay, 2.0, _settings_args),
    ("p16", "参数校验矩阵", p_param_matrix, 0.0, _declared_settings_args),
]

# OpenAI 兼容协议探针表（/v1/chat/completions）：结构同 PROBES。
# 无 p3 响应头/p5 count_tokens/p6 缓存字段/p14 PDF 的对应物；
# 新增 o10 reasoning_effort（kimi-k3 特征）与 o11 usage 一致性。
PROBES_OPENAI = [
    ("o1", "连通性+响应结构", o_connectivity, 3.0, _no_args),
    ("o2", "model 回显", o_model_echo, 2.0, _declared_args),
    ("o3", "max_tokens 截断", o_max_tokens_cutoff, 2.0, _no_args),
    ("o4", "SSE 流式", o_sse_shape, 2.0, _no_args),
    ("o5", "stop 参数", o_stop_param, 2.0, _no_args),
    ("o6", "多轮记忆", o_multi_turn, 2.0, _no_args),
    ("o7", "system prompt 服从", o_system_adherence, 2.0, _no_args),
    ("o8", "工具调用", o_tool_use, 3.0, _no_args),
    ("o9", "视觉输入", o_vision_input, 2.0, _no_args),
    ("o10", "reasoning_effort 特征", o_reasoning_effort, 2.0, _declared_args),
    ("o11", "usage 一致性", o_usage_consistency, 2.0, _no_args),
    ("o12", "缓存/重放检测", o_cache_replay, 2.0, _settings_args),
    ("o13", "参数校验矩阵", o_param_matrix, 0.0, _declared_settings_args),
]


async def run_probes(client: ApiClient, declared_model: str, settings,
                     progress=None) -> list[ProbeResult]:
    """并行执行全部探针（受 settings.CONCURRENCY 并发限制）。

    先探测协议：anthropic → PROBES；openai 兼容 → PROBES_OPENAI；
    协议识别失败 → 返回单个 na 结果（调用方据此判无法判定）。
    每个探针通过其 dispatcher 从 ctx 取所需参数，避免在编排处用 pid 字符串硬编码分发。
    """
    proto = await client.detect_protocol()
    if proto == "openai":
        table = PROBES_OPENAI
    elif proto == "anthropic":
        table = PROBES
    else:
        # 诊断留痕：展示两个端点的实测状态码，避免"均无有效响应"黑盒
        # （401 也可能表示 key 有效但未授权该模型；非常规 4xx 多为渠道未授权/已下线）
        seen = []
        for e in client.proto_probe_log():
            st = e.get("status")
            seen.append(f"{e['endpoint']}→{st if st is not None else e.get('error', '异常')}"
                        + ("（响应体形状不符）" if e.get("status") is not None and not e.get("shape_ok") else ""))
        hints = []
        codes = [e.get("status") for e in client.proto_probe_log() if e.get("status") is not None]
        if 401 in codes:
            hints.append("401 通常表示 key 无效，或多渠道网关按 key×模型 鉴权时该 key 未授权此模型")
        elif codes and any(c not in (200, 400, 401, 429) for c in codes):
            hints.append("非常规状态码：key 可能有效但渠道未授权该模型，或渠道已被网关下线")
        detail = ("协议识别失败（实测：" + "，".join(seen) + "）。"
                  "排查：① Base URL 应为根地址或 …/v1（勿带 /chat/completions 等端点路径，"
                  "多余后缀已自动剥离）；② API key 是否有效；③ 目标网络/VPN 是否可达")
        if hints:
            detail += "。提示：" + "；".join(hints)
        return [ProbeResult("p0", "协议探测", "na", detail)]
    public_mode = getattr(client, "public_mode", False)
    if public_mode:
        table = [spec for spec in table if spec[0] not in ("o13", "p16")]
    profile = resolve_model_profile(declared_model)
    ctx = SimpleNamespace(declared=declared_model, settings=settings, profile=profile)
    if progress:
        progress("plan", {
            "items": [{"id": spec[0], "name": spec[1]} for spec in table],
        })

    # 限流熔断：429 失败探针达阈值后，未派发的探针直接跳过（并发 8 下首批
    # 已在途的无法撤回，但可避免把剩余探针+矩阵请求继续喂给已耗尽的配额）
    rl = {"n": 0, "stopped": False}
    RL_BREAKER = 3

    async def run_one(spec):
        pid, pname, fn, weight, dispatch = spec
        index = next(i for i, item in enumerate(table, 1) if item[0] == pid)
        if probe_applicability(profile, pid) is False:
            result = ProbeResult(
                pid, pname, "na", "模型能力配置判定该项不适用",
                weight=weight, applicable=False,
            )
            if progress:
                progress("finished", {
                    "id": pid, "name": pname, "result": "na",
                    "index": index, "total": len(table),
                })
            return result
        if rl["stopped"]:
            result = ProbeResult(pid, pname, "na", "渠道限流(429)熔断，探针跳过",
                                 error_code="rate_limited")
            if progress:
                progress("finished", {
                    "id": pid, "name": pname, "result": "na",
                    "index": index,
                    "total": len(table),
                })
            return result
        if progress:
            progress("started", {
                "id": pid, "name": pname, "index": index, "total": len(table),
            })
        if public_mode:
            try:
                async with async_timeout(settings.L1_PROBE_TIMEOUT_S):
                    r = await dispatch(client, ctx, fn)
            except TimeoutError:
                r = ProbeResult(pid, pname, "na", "单项检测超时",
                                weight=weight, error_code="probe_timeout")
        else:
            r = await dispatch(client, ctx, fn)
        if getattr(r, "rate_limited", False):
            rl["n"] += 1
            if rl["n"] >= RL_BREAKER:
                rl["stopped"] = True
        if progress:
            progress("finished", {
                "id": pid, "name": pname,
                "result": r.status if r.status in ("pass", "fail", "na") else "na",
                "error_code": r.error_code,
                "index": index, "total": len(table),
            })
        return r

    sem = asyncio.Semaphore(settings.CONCURRENCY)

    async def limited(spec):
        async with sem:
            return await run_one(spec)

    if not public_mode:
        return await asyncio.gather(*(limited(s) for s in table))

    tasks = [asyncio.create_task(limited(spec)) for spec in table]
    try:
        remaining = max(0, getattr(client, "probe_deadline",
                        asyncio.get_running_loop().time() + 140)
                        - asyncio.get_running_loop().time())
        _, pending = await asyncio.wait(tasks, timeout=remaining)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for index, (spec, task) in enumerate(zip(table, tasks), 1):
            if task.cancelled():
                pid, name, _, weight, _ = spec
                result = ProbeResult(pid, name, "na", "检测时间预算已用完",
                                     weight=weight, error_code="probe_deadline")
                if progress:
                    progress("finished", {"id": pid, "name": name, "result": "na",
                             "error_code": result.error_code, "index": index,
                             "total": len(table)})
            else:
                result = task.result()
            results.append(result)
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def probe_summary(results: list[ProbeResult]) -> dict:
    """汇总评分与证据覆盖率。

    score 只比较已有结论的通过/失败；coverage 衡量应检证据中有多少真正形成结论。
    配置明确不适用的项目不进入覆盖率分母，运行时无法判断的 na 会降低覆盖率。
    """
    applicable_w = sum(r.weight for r in results if r.applicable)
    observed_w = sum(
        r.weight for r in results if r.applicable and r.status in ("pass", "fail")
    )
    earned = sum(r.weight for r in results if r.status == "pass")
    counts = {"pass": 0, "fail": 0, "na": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "score": earned / observed_w if observed_w else 0.0,
        "coverage": observed_w / applicable_w if applicable_w else 0.0,
        "counts": counts,
        "total_weight": observed_w,
        "applicable_weight": applicable_w,
    }
