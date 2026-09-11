import httpx
import pytest

from model_check.security import (
    PublicOnlyTransport,
    ResponseTooLarge,
    UnsafeTargetError,
    normalize_public_base_url,
    resolve_public_ips,
)


@pytest.mark.parametrize("value", [
    "http://api.example.com",
    "https://localhost",
    "https://service.internal/v1",
    "https://127.0.0.1",
    "https://10.0.0.1",
    "https://169.254.169.254/latest/meta-data",
    "https://[::1]",
    "https://user:pass@example.com",
    "https://example.com/v1?target=other",
])
def test_normalize_rejects_unsafe_targets(value):
    with pytest.raises(UnsafeTargetError):
        normalize_public_base_url(value)


def test_normalize_public_target():
    assert normalize_public_base_url(" https://API.Example.com/v1/ ") == \
        "https://api.example.com/v1"


def test_public_transport_ignores_proxy_environment(monkeypatch):
    options = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            options.update(kwargs)

    monkeypatch.setenv("LOCAL_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setattr("model_check.security.httpx.AsyncHTTPTransport", FakeTransport)

    PublicOnlyTransport()

    assert options == {"trust_env": False}


@pytest.mark.asyncio
async def test_resolver_rejects_mixed_public_private_dns(monkeypatch):
    class Loop:
        async def getaddrinfo(self, *args, **kwargs):
            return [
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("10.0.0.8", 443)),
            ]

    monkeypatch.setattr("model_check.security.asyncio.get_running_loop", lambda: Loop())
    with pytest.raises(UnsafeTargetError, match="非公网"):
        await resolve_public_ips("example.com")


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        return None


class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self, response):
        self.request = None
        self.response = response

    async def handle_async_request(self, request):
        self.request = request
        return self.response

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_public_transport_pins_verified_ip_and_keeps_host(monkeypatch):
    async def resolve(host, port=443):
        return ("93.184.216.34",)

    monkeypatch.setattr("model_check.security.resolve_public_ips", resolve)
    inner = _CaptureTransport(httpx.Response(200, stream=_Stream([b"ok"])))
    transport = PublicOnlyTransport()
    transport._transport = inner

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://api.example.com/v1/test")

    assert response.text == "ok"
    assert inner.request.url.host == "93.184.216.34"
    assert inner.request.headers["host"] == "api.example.com"
    assert inner.request.extensions["sni_hostname"] == "api.example.com"


@pytest.mark.asyncio
async def test_public_transport_still_pins_ip_with_legacy_proxy_env(monkeypatch):
    async def resolve(host, port=443):
        return ("93.184.216.34",)

    monkeypatch.setenv("LOCAL_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setattr("model_check.security.resolve_public_ips", resolve)
    inner = _CaptureTransport(httpx.Response(200, stream=_Stream([b"ok"])))
    transport = PublicOnlyTransport()
    transport._transport = inner

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://api.example.com/v1/test")

    assert response.text == "ok"
    assert inner.request.url.host == "93.184.216.34"
    assert inner.request.headers["host"] == "api.example.com"
    assert inner.request.extensions["sni_hostname"] == "api.example.com"


@pytest.mark.asyncio
async def test_public_transport_limits_response_size(monkeypatch):
    async def resolve(host, port=443):
        return ("93.184.216.34",)

    monkeypatch.setattr("model_check.security.resolve_public_ips", resolve)
    inner = _CaptureTransport(httpx.Response(200, stream=_Stream([b"123", b"456"])))
    transport = PublicOnlyTransport(max_response_bytes=5)
    transport._transport = inner

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ResponseTooLarge):
            await client.get("https://api.example.com/v1/test")
