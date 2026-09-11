import asyncio
import json

import httpx

from model_check.config import get_settings
from model_check.model_profiles import resolve_model_profile
from model_check.security import PublicOnlyTransport


class ProtocolError(Exception):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        # HTTP 状态码（限流熔断按 429 判定；传输层异常为 None）
        self.status = status


ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01", "content-type": "application/json"}


class ApiClient:
    """双协议客户端：Anthropic 原生（/v1/messages）与 OpenAI 兼容（/v1/chat/completions）自动探测。

    所有外部调用统一返回 dict：
    {text, usage_in, usage_out, stop_reason, raw, headers, request_id}
    transport 参数注入 httpx.MockTransport 用于测试。
    """

    def __init__(self, base_url: str, api_key: str, transport=None, settings=None,
                 default_model: str = "claude-sonnet-4-5",
                 extra_body: dict | None = None, public_mode: bool = False,
                 timeout_s: float | None = None):
        self.base_url = base_url.rstrip("/")
        # 用户常把完整端点路径当 base_url 填（实测案例：
        # https://gw/v1/chat/completions）——不剥离会打出 /v1/chat/completions/v1/xxx
        # 双重路径 404 → 协议识别失败。按最长优先反复剥离已知端点后缀
        for suffix in ("/messages/count_tokens", "/chat/completions",
                       "/completions", "/messages"):
            while self.base_url.endswith(suffix):
                self.base_url = self.base_url[: -len(suffix)].rstrip("/")
        # 兼容用户传入以 /v1 结尾的 base_url：内部统一按根地址拼 /v1/xxx
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
        # 协议端点 URL 集中管理：Gemini 官方域名的 OpenAI 兼容层路径是
        # /v1beta/openai/chat/completions（无 /v1 前缀），与通用拼接规则不同；
        # 无论用户填裸域名、/v1beta 还是 /v1beta/openai，都规范化到正确端点，
        # 否则协议探测必 404 → 误报「协议识别失败」
        self._chat_url = f"{self.base_url}/v1/chat/completions"
        self._messages_url = f"{self.base_url}/v1/messages"
        if "generativelanguage.googleapis.com" in self.base_url:
            root = self.base_url.split("generativelanguage.googleapis.com", 1)[0] \
                + "generativelanguage.googleapis.com"
            self._chat_url = f"{root}/v1beta/openai/chat/completions"
            # Gemini 无 /v1/messages：探测固定打根路径（稳定 404 → 落入 openai 分支），
            # 不跟随用户填的子路径避免拼出不存在的嵌套地址
            self._messages_url = f"{root}/v1/messages"
        self.api_key = api_key
        self.public_mode = public_mode
        self.settings = settings or get_settings()
        # 探测/审计请求使用的模型名：跟随被检测的声明模型（中转站可能严格校验模型名）
        self.default_model = default_model
        # trust_env=False：审计探针必须直连目标——macOS 上 urllib 会把系统代理
        # （SystemConfiguration）注入 httpx 且不认内网例外，导致内网网关被代理劫持
        if transport is None and public_mode:
            transport = PublicOnlyTransport(
                max_response_bytes=self.settings.MAX_UPSTREAM_RESPONSE_BYTES
            )
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=timeout_s or self.settings.TIMEOUT_S,
            trust_env=False,
            follow_redirects=False,
        )
        self._proto_cache: str | None = None
        # 附加请求体字段（每个请求合并进 JSON body）：聚合商路由钉扎等，
        # 如 OpenRouter {"provider": {"order": ["z-ai"], "allow_fallbacks": false}}
        # ——OpenRouter 默认按请求在 24 家上游间负载均衡（量化 fp4/fp8 混杂、
        # 甚至杂牌模型），不钉扎则指纹/检测采样在供应商间漂移
        self.extra_body = dict(extra_body or {})

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def _request_with_retry(self, method, url, *, retry_rejected=False, **kw):
        last = None
        retries = min(self.settings.RETRIES, 1) if self.public_mode else self.settings.RETRIES
        for attempt in range(retries + 1):
            try:
                response = await self._client.request(method, url, **kw)
                # 协议识别已成功后，部分聚合网关会把并发/渠道繁忙错误写成
                # 401/403。仅延迟重试一次，避免把真实鉴权失败无限重放。
                if retry_rejected and response.status_code in (401, 403) and attempt == 0:
                    await asyncio.sleep(self.settings.RETRY_BACKOFF[0])
                    continue
                return response
            except httpx.HTTPError as e:
                last = e
                if attempt < retries:
                    await asyncio.sleep(self.settings.RETRY_BACKOFF[attempt])
        raise ProtocolError(f"request failed after retries: {last}")

    async def detect_protocol(self) -> str:
        """按声明模型优先探测协议，失败后回退另一协议。结果缓存。

        判定收紧（审查 P3-4）：错误响应须同时符合 Anthropic 错误对象结构
        {type:"error", error:{type,message}} 才视为协议存在，避免误判。
        """
        if self._proto_cache is not None:
            return self._proto_cache

        # 探测过程留痕：p0 诊断文案需要展示实测状态码（避免"均无有效响应"黑盒）
        self._proto_probe_log: list[dict] = []

        def _anthropic_error_shape(text: str) -> bool:
            try:
                d = json.loads(text)
                err = d.get("error")
                return d.get("type") == "error" and isinstance(err, dict) \
                    and "type" in err and "message" in err
            except Exception:
                return False

        async def probe_anthropic() -> bool:
            try:
                r = await self._client.post(
                    self._messages_url,
                    headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS},
                    json={"model": self.default_model, "max_tokens": 1,
                          "messages": [{"role": "user", "content": "仅回复 OK"}]},
                    timeout=self.settings.L1_PROTOCOL_TIMEOUT_S,
                )
                text = r.text[:2000]
                shape_ok = _anthropic_error_shape(text)
                self._proto_probe_log.append(
                    {"endpoint": "/v1/messages", "status": r.status_code,
                     "shape_ok": shape_ok})
                return (r.status_code == 200 and "msg_" in text) or (
                    r.status_code in (400, 401, 429) and shape_ok
                )
            except httpx.HTTPError as e:
                self._proto_probe_log.append(
                    {"endpoint": "/v1/messages", "status": None,
                     "error": type(e).__name__})
                return False

        async def probe_openai() -> bool:
            try:
                # 不携带 max_tokens：部分 GPT-5 兼容渠道会在协议探测阶段卡住，
                # 而常规 Chat Completions 请求可以正常返回。正式探针仍会测试截断语义。
                r = await self._client.post(
                    self._chat_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.default_model,
                          "messages": [{"role": "user", "content": "仅回复 OK"}]},
                    timeout=self.settings.L1_PROTOCOL_TIMEOUT_S,
                )
                shape_ok = _openai_like(r)
                self._proto_probe_log.append(
                    {"endpoint": "/v1/chat/completions", "status": r.status_code,
                     "shape_ok": shape_ok})
                return r.status_code in (200, 400, 401, 429) and shape_ok
            except httpx.HTTPError as e:
                self._proto_probe_log.append(
                    {"endpoint": "/v1/chat/completions", "status": None,
                     "error": type(e).__name__})
                return False

        probes = {
            "anthropic": probe_anthropic,
            "openai": probe_openai,
        }
        preferred = resolve_model_profile(self.default_model).preferred_protocol
        order = (preferred, "anthropic" if preferred == "openai" else "openai")
        for protocol in order:
            if await probes[protocol]():
                self._proto_cache = protocol
                return protocol
            # 仅优先协议的超时重试一次；HTTP 拒绝或格式不符不重试。
            # 保留两次探测留痕，且仍受调用方的检测总时限约束。
            if protocol == preferred and self._proto_probe_log[-1].get("error") in (
                "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
            ):
                if await probes[protocol]():
                    self._proto_cache = protocol
                    return protocol
        self._proto_cache = "unknown"
        return "unknown"

    def proto_probe_log(self) -> list[dict]:
        """最近一次 detect_protocol 的探测留痕（endpoint/status/shape）。"""
        return list(getattr(self, "_proto_probe_log", []))

    async def complete(self, messages, *, max_tokens, temperature=0, extra=None) -> dict:
        extra = extra or {}
        proto = await self.detect_protocol()
        if proto == "anthropic":
            body = {"model": extra.get("model", self.default_model),
                    "max_tokens": max_tokens, "messages": messages,
                    "temperature": temperature}
            if "system" in extra:
                body["system"] = extra["system"]
            if "tools" in extra:
                body["tools"] = extra["tools"]
            if "stop_sequences" in extra:
                body["stop_sequences"] = extra["stop_sequences"]
            r = await self._request_with_retry(
                "POST", self._messages_url,
                headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS}, json=body,
                retry_rejected=True)
            # 严格网关参数兼容（与 openai 分支同构）：实测存在只放行
            # temperature=1.0 的 anthropic 类端点，400 时去参重试一次；
            # 400 不耗 token，重试无成本
            if r.status_code == 400 and "temperature" in body:
                body = {k: v for k, v in body.items() if k != "temperature"}
                r = await self._request_with_retry(
                    "POST", self._messages_url,
                    headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS}, json=body,
                    retry_rejected=True)
            if r.status_code != 200:
                raise ProtocolError(f"anthropic {r.status_code}: {r.text[:300]}",
                                    status=r.status_code)
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", [])
                           if b.get("type") == "text")
            usage = data.get("usage", {})
            return {"text": text,
                    "usage_in": usage.get("input_tokens", 0),
                    "usage_out": usage.get("output_tokens", 0),
                    "stop_reason": data.get("stop_reason"),
                    "raw": data, "headers": dict(r.headers),
                    "request_id": r.headers.get("request-id", "")}
        if proto == "openai":
            # 模型名跟随声明模型透传（中转站可能严格校验模型名，硬编码默认值会导致路由错误或 4xx）
            request_model = extra.get("model", self.default_model)
            profile = resolve_model_profile(request_model)
            token_field = "max_completion_tokens" \
                if profile.reasoning_effort is True else "max_tokens"
            body = {"model": request_model,
                    token_field: max_tokens,
                    "messages": _openai_messages(messages, extra.get("system")),
                    "temperature": temperature}
            # anthropic → openai 参数映射：stop_sequences→stop；system 由 _openai_messages 转为消息
            if "stop_sequences" in extra:
                body["stop"] = extra["stop_sequences"]
            if "tools" in extra:
                body["tools"] = extra["tools"]
            if extra.get("reasoning_effort") in _REASONING_EFFORTS:
                body["reasoning_effort"] = extra["reasoning_effort"]
            if self.extra_body:
                body.update(self.extra_body)
            r = await self._request_with_retry(
                "POST", self._chat_url,
                headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                retry_rejected=True)
            # 严格网关参数兼容：同网关不同上游渠道对参数校验宽严不一（实测官方渠道
            # 拒绝 temperature → 400，天翼云渠道接受；错误体可能为空或带
            # "Parameter validation error" 字样，故不依赖错误体内容判定）。
            # 去掉该可选参数重试一次——L2 采样扰动由查询模板文本承担（seed 决定
            # 模板填充），不依赖温度，去掉后多样性不受损。400 不耗 token，重试无成本。
            if r.status_code == 400 and "temperature" in body:
                body = {k: v for k, v in body.items() if k != "temperature"}
                r = await self._request_with_retry(
                    "POST", self._chat_url,
                    headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                    retry_rejected=True)
            # 不同兼容层对 GPT-5/o 系 token 参数支持不一致，按模型能力选择首选字段，
            # 400 时双向切换一次，避免把参数方言误判为模型异常。
            if r.status_code == 400 and (
                    "max_tokens" in body or "max_completion_tokens" in body):
                if "max_tokens" in body:
                    body["max_completion_tokens"] = body.pop("max_tokens")
                else:
                    body["max_tokens"] = body.pop("max_completion_tokens")
                r = await self._request_with_retry(
                    "POST", self._chat_url,
                    headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                    retry_rejected=True)
            if r.status_code != 200:
                raise ProtocolError(f"openai {r.status_code}: {r.text[:300]}",
                                    status=r.status_code)
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage", {})
            return {"text": (choice.get("message") or {}).get("content", ""),
                    "usage_in": usage.get("prompt_tokens", 0),
                    "usage_out": usage.get("completion_tokens", 0),
                    "stop_reason": choice.get("finish_reason"),
                    "raw": data, "headers": dict(r.headers),
                    "request_id": data.get("id", "")}
        raise ProtocolError(f"protocol detection failed for {self.base_url}")

    async def raw_messages(self, body: dict) -> httpx.Response:
        """底层 /v1/messages 请求（不抛 ProtocolError），供探针检查错误对象等原始行为。"""
        return await self._client.post(
            self._messages_url,
            headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS}, json=body)

    async def raw_chat(self, body: dict) -> httpx.Response:
        """底层 /v1/chat/completions 请求（不抛 ProtocolError、不重试），
        供 openai 探针检查原始状态码与错误对象（如 reasoning_effort 拒绝行为）。"""
        return await self._client.post(
            self._chat_url,
            headers={"Authorization": f"Bearer {self.api_key}"}, json=body)

    async def count_tokens(self, text: str) -> int:
        """独立 token 核算（Anthropic count_tokens 端点）；不可用时返回 -1。

        对裸 text 重算 —— 仅用于粗略对比，p5 审计请改用 count_tokens_messages。
        """
        try:
            r = await self._client.post(
                f"{self.base_url}/v1/messages/count_tokens",
                headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS},
                json={"model": self.default_model,
                      "messages": [{"role": "user", "content": text}]})
            if r.status_code == 200:
                return r.json().get("input_tokens", 0)
        except httpx.HTTPError:
            pass
        return -1

    async def count_tokens_messages(self, messages: list) -> int:
        """用与 complete 请求一致的 message 结构重算 token（与 usage 口径对齐）。

        p5 审计专用：裸 text 与 usage.input_tokens 有结构性偏移（role/封装开销），
        用同一 message 结构重算可消除该偏移。不可用返回 -1。
        """
        try:
            r = await self._client.post(
                f"{self.base_url}/v1/messages/count_tokens",
                headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS},
                json={"model": self.default_model, "messages": messages})
            if r.status_code == 200:
                return r.json().get("input_tokens", 0)
        except httpx.HTTPError:
            pass
        return -1

    async def stream_events(self, messages, *, max_tokens, extra=None) -> list[str]:
        """SSE 流式请求，返回事件类型序列（如 message_start/content_block_delta/...）。"""
        extra = extra or {}
        if await self.detect_protocol() != "anthropic":
            return []
        body = {"model": extra.get("model", self.default_model),
                "max_tokens": max_tokens, "messages": messages,
                "temperature": extra.get("temperature", 0), "stream": True}
        if "system" in extra:
            body["system"] = extra["system"]
        events = []
        try:
            async with self._client.stream(
                "POST", self._messages_url,
                headers={"x-api-key": self.api_key, **ANTHROPIC_HEADERS}, json=body,
            ) as resp:
                if resp.status_code != 200:
                    return []
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())
        except httpx.HTTPError:
            return []
        return events

    async def stream_openai_events(self, messages, *, max_tokens, extra=None) -> list[str]:
        """OpenAI 兼容 SSE 流式请求，返回归一化事件类型序列。

        openai chat completions 流式无 event: 前缀行，按 data: JSON 分片解析；
        归一化事件：message_start（首 delta.role）/ content_delta（有内容增量）/
        tool_delta（工具调用增量）/ usage（分片携带用量）/ done（[DONE] 终止标记）。
        失败或非 200 返回空列表。
        """
        extra = extra or {}
        request_model = extra.get("model", self.default_model)
        profile = resolve_model_profile(request_model)
        token_field = "max_completion_tokens" \
            if profile.reasoning_effort is True else "max_tokens"
        body = {"model": request_model,
                token_field: max_tokens, "stream": True,
                "messages": _openai_messages(messages, extra.get("system")),
                "temperature": extra.get("temperature", 0)}
        if self.extra_body:
            body.update(self.extra_body)
        events: list[str] = []
        try:
            for attempt in range(2):
                events.clear()
                async with self._client.stream(
                    "POST", self._chat_url,
                    headers={"Authorization": f"Bearer {self.api_key}"}, json=body,
                ) as resp:
                    if resp.status_code == 400 and attempt == 0:
                        if "max_tokens" in body:
                            body["max_completion_tokens"] = body.pop("max_tokens")
                        else:
                            body["max_tokens"] = body.pop("max_completion_tokens")
                        continue
                    if resp.status_code != 200:
                        return []
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line.split(":", 1)[1].strip()
                        if payload == "[DONE]":
                            events.append("done")
                            break
                        try:
                            chunk = json.loads(payload)
                        except Exception:
                            continue
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        if "role" in delta:
                            events.append("message_start")
                        if delta.get("content"):
                            events.append("content_delta")
                        if delta.get("tool_calls"):
                            events.append("tool_delta")
                        if chunk.get("usage"):
                            events.append("usage")
                    return events
        except httpx.HTTPError:
            return []
        return events


# reasoning_effort 合法值白名单（openai o 系列 minimal/low/medium/high；
# kimi-k3 当前仅 max——白名单取并集，探针据此探测档位行为）
_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "max"}


def _openai_messages(messages: list, system) -> list:
    """把 anthropic 风格 messages + system 转为 openai 兼容消息列表。

    openai 协议无独立 system 字段，需作为 role=system 消息插在开头；
    system 为 str 直接使用，为 content blocks 列表时抽取 text 拼接；
    其余消息结构（role/content）两协议兼容，原样保留。
    """
    out: list = []
    if system:
        if isinstance(system, str):
            sys_text = system
        else:
            sys_text = "".join(
                b.get("text", "") for b in system if isinstance(b, dict))
        if sys_text:
            out.append({"role": "system", "content": sys_text})
    out.extend(messages)
    return out


def _openai_like(resp: "httpx.Response") -> bool:
    """OpenAI 协议特征判定：200 看 choices，错误看 error.{message,type,code}。

    detect_protocol 收紧用 —— 单凭状态码会误判任意返回 4xx 的端点。
    """
    try:
        d = resp.json()
    except Exception:
        return False
    if resp.status_code == 200:
        return "choices" in d or "object" in d
    err = d.get("error")
    # OpenAI 错误对象：{"error": {"message":..., "type":..., "code":...}}
    return isinstance(err, dict) and ("message" in err or "type" in err or "code" in err)
