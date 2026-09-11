"""公网检测目标的安全校验与受限 HTTP 传输。

目标域名在每次连接前重新解析，并把请求固定到已验证的公网 IP，避免 DNS
重绑定把 API Key 带入内网。原始 Host 与 TLS SNI 保持不变。
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

import httpx


from model_check.config import async_timeout


class UnsafeTargetError(ValueError):
    """目标 URL 不符合公网检测安全要求。"""


class ResponseTooLarge(httpx.StreamError):
    """上游响应超过允许大小。"""


_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home", ".lan")
DNS_TIMEOUT_S = 5.0


def normalize_public_base_url(value: str) -> str:
    """校验 URL 语法并返回规范化后的 HTTPS Base URL。"""
    raw = (value or "").strip()
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeTargetError("API 地址格式不正确") from exc

    if parsed.scheme.lower() != "https":
        raise UnsafeTargetError("API 地址必须使用 HTTPS")
    if not parsed.hostname:
        raise UnsafeTargetError("API 地址缺少有效域名")
    if parsed.username or parsed.password:
        raise UnsafeTargetError("API 地址不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise UnsafeTargetError("API 地址不能包含查询参数或片段")
    if port is not None and not 1 <= port <= 65535:
        raise UnsafeTargetError("API 地址端口无效")

    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeTargetError("不能检测本机或内网地址")

    try:
        literal = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        literal = None
    if literal is not None and not _is_public_ip(literal):
        raise UnsafeTargetError("不能检测本机、私网或保留地址")

    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    if port is not None and port != 443:
        netloc = f"{netloc}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit(("https", netloc, path, "", ""))


def _is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


async def resolve_public_ips(host: str, port: int = 443) -> tuple[str, ...]:
    """解析域名，要求解析结果全部为公网地址。"""
    try:
        async with async_timeout(DNS_TIMEOUT_S):
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, port, type=socket.SOCK_STREAM
            )

    except (OSError, TimeoutError) as exc:
        raise UnsafeTargetError("API 域名无法解析") from exc

    addresses: list[str] = []
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError as exc:
            raise UnsafeTargetError("API 域名解析结果无效") from exc
        if not _is_public_ip(address):
            raise UnsafeTargetError("API 域名解析到了非公网地址")
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise UnsafeTargetError("API 域名没有可用的公网地址")
    # IPv4 优先，降低部分上游公布 AAAA 但实际 IPv6 不可达造成的失败率。
    addresses.sort(key=lambda item: 1 if ":" in item else 0)
    return tuple(addresses)


async def validate_public_target(value: str) -> str:
    """完成语法和 DNS 两层校验，返回规范化 URL。"""
    normalized = normalize_public_base_url(value)
    parsed = urlsplit(normalized)
    await resolve_public_ips(parsed.hostname or "", parsed.port or 443)
    return normalized


class _LimitedResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: httpx.AsyncByteStream, max_bytes: int):
        self._stream = stream
        self._max_bytes = max_bytes

    async def __aiter__(self):
        received = 0
        async for chunk in self._stream:
            received += len(chunk)
            if received > self._max_bytes:
                await self._stream.aclose()
                raise ResponseTooLarge("upstream response exceeded size limit")
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class PublicOnlyTransport(httpx.AsyncBaseTransport):
    """只连接公网 HTTPS 地址，并限制单个响应的最大体积。"""

    def __init__(self, max_response_bytes: int = 2_000_000):
        self._transport = httpx.AsyncHTTPTransport(trust_env=False)
        self._max_response_bytes = max_response_bytes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != "https":
            raise httpx.ConnectError("public transport requires HTTPS", request=request)

        host = request.url.host
        port = request.url.port or 443
        try:
            ips = await resolve_public_ips(host, port)
        except UnsafeTargetError as exc:
            raise httpx.ConnectError("connection target is not public", request=request) from exc

        # 固定到已验证地址；Host 头与 TLS SNI 继续使用原始域名。
        pinned_url = request.url.copy_with(host=ips[0])
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = host
        outbound_request = httpx.Request(
            request.method,
            pinned_url,
            headers=request.headers,
            stream=request.stream,
            extensions=extensions,
        )
        response = await self._transport.handle_async_request(outbound_request)
        length = response.headers.get("content-length")
        if length and length.isdigit() and int(length) > self._max_response_bytes:
            await response.aclose()
            raise ResponseTooLarge("upstream response exceeded size limit")
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=_LimitedResponseStream(response.stream, self._max_response_bytes),
            extensions=response.extensions,
        )

    async def aclose(self) -> None:
        await self._transport.aclose()
