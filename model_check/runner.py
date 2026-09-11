"""L1 检测编排器：协议探测 → 探针执行 → 得分与结论汇总。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from urllib.parse import urlsplit

from model_check.clients import ApiClient
from model_check.config import Settings, async_timeout, get_settings
from model_check.model_profiles import resolve_model_profile
from model_check.probes import probe_summary, run_probes
from model_check.verdict import L1_HIGH, L1_MIN_COVERAGE, decide_l1_public

logger = logging.getLogger("model_check.runner")


@dataclass(frozen=True)
class L1DetectionRequest:
    """L1 极速检测请求。"""

    base_url: str
    api_key: str
    declared_model: str


def _incomplete(code: str, message: str, started_at: float) -> dict:
    return {
        "status": "incomplete",
        "overall": None,
        "protocol": None,
        "declared_model": None,
        "dimensions": [],
        "duration_ms": round((time.monotonic() - started_at) * 1000),
        "error": {"code": code, "message": message},
    }


async def run_l1_detection(
    request: L1DetectionRequest,
    progress=None,
    settings: Settings | None = None,
    detection_id: str = "cli",
) -> dict:
    """执行 L1 极速检测。"""
    settings = settings or get_settings()
    started_at = time.monotonic()

    def finish_incomplete(code: str, message: str) -> dict:
        result = _incomplete(code, message, started_at)
        logger.warning(
            "detection_incomplete id=%s target=%s model=%s code=%s ms=%s",
            detection_id,
            urlsplit(request.base_url).hostname or "unknown",
            request.declared_model,
            code,
            result["duration_ms"],
        )
        return result

    def emit(message: str, **state) -> None:
        if progress:
            progress("progress", {"message": message, **state})

    client = ApiClient(
        request.base_url,
        request.api_key,
        settings=settings,
        default_model=request.declared_model,
        public_mode=True,
        timeout_s=settings.L1_REQUEST_TIMEOUT_S,
    )
    logger.info(
        "detection_started id=%s target=%s model=%s",
        detection_id,
        urlsplit(request.base_url).hostname or "unknown",
        request.declared_model,
    )

    try:
        async with async_timeout(settings.L1_TOTAL_TIMEOUT_S):
            emit("正在识别接口协议", stage="protocol", status="running")
            protocol = await client.detect_protocol()
            logs = client.proto_probe_log()
            for item in logs:
                logger.info(
                    "protocol_probe id=%s endpoint=%s status=%s error=%s",
                    detection_id,
                    item.get("endpoint"),
                    item.get("status"),
                    item.get("error"),
                )

            statuses = [item.get("status") for item in logs]
            if 401 in statuses or 403 in statuses:
                return finish_incomplete(
                    "authentication_failed", "鉴权失败，请检查 API Key 与模型权限"
                )
            if 429 in statuses:
                return finish_incomplete(
                    "rate_limited", "接口触发限流，请稍后重试"
                )
            if protocol not in ("anthropic", "openai"):
                if any(
                    item.get("error")
                    in (
                        "ReadTimeout",
                        "ConnectTimeout",
                        "WriteTimeout",
                        "PoolTimeout",
                    )
                    for item in logs
                ):
                    return finish_incomplete(
                        "protocol_timeout",
                        "接口响应超时，未能完成协议识别，请检查网络或代理后重试；这不代表模型检测不通过",
                    )
                return finish_incomplete(
                    "protocol_unavailable",
                    "无法识别接口协议，请检查 API 地址、API Key 和模型名称",
                )

            protocol_name = (
                "Anthropic Messages"
                if protocol == "anthropic"
                else "OpenAI 兼容协议"
            )
            emit(
                f"已识别 {protocol_name}",
                stage="protocol",
                status="pass",
                protocol=protocol,
            )

            def probe_progress(action: str, data: dict) -> None:
                payload = {"stage": "probes", "action": action, **data}
                if action == "plan":
                    payload["message"] = f"准备执行 {len(data.get('items', []))} 项检测"
                elif action == "started":
                    payload["message"] = f"正在检测：{data.get('name', '检测项目')}"
                elif action == "finished":
                    labels = {"pass": "通过", "fail": "不通过", "na": "不适用"}
                    result_label = labels.get(data.get("result"), "不适用")
                    payload["message"] = f"{data.get('name', '检测项目')}：{result_label}"
                if progress:
                    progress("progress", payload)

            emit("正在准备检测项目", stage="probes", status="running")
            client.probe_deadline = asyncio.get_running_loop().time() + max(
                0,
                min(
                    140,
                    settings.L1_TOTAL_TIMEOUT_S
                    - (time.monotonic() - started_at)
                    - 2,
                ),
            )
            probe_results = await run_probes(
                client, request.declared_model, settings, progress=probe_progress
            )

            for item in probe_results:
                if item.error_code:
                    logger.warning(
                        "probe_issue id=%s probe=%s error=%s status=%s",
                        detection_id,
                        item.id,
                        item.error_code,
                        item.upstream_status,
                    )

            connectivity = next(
                (item for item in probe_results if item.id in ("p1", "o1")), None
            )
            if connectivity is None or (
                connectivity.status == "na"
                and connectivity.error_code not in ("probe_timeout", "probe_deadline")
            ):
                code = connectivity.error_code if connectivity else "upstream"
                if code == "auth":
                    code = "upstream_rejected"
                messages = {
                    "auth": "鉴权失败，请检查 API Key 与模型权限",
                    "upstream_rejected": "渠道拒绝了部分检测请求，请稍后重试",
                    "rate_limited": "接口触发限流，请稍后重试",
                    "transport": "接口暂时不可用，请检查地址和网络后重试",
                    "system": "检测服务暂时异常，请稍后重试",
                }
                return finish_incomplete(
                    code or "upstream",
                    messages.get(code, "接口未能完成基础响应，请检查配置后重试"),
                )

            rate_limited = sum(
                1
                for item in probe_results
                if item.rate_limited or item.error_code == "rate_limited"
            )
            if rate_limited >= 3:
                return finish_incomplete(
                    "rate_limited", "接口触发限流，检测未完成，请稍后重试"
                )

            summary = probe_summary(probe_results)
            profile = resolve_model_profile(request.declared_model)
            overall = decide_l1_public(summary, probe_results, profile, protocol)

            def public_reason(item) -> str:
                if item.status == "pass":
                    return "verified"
                if item.status == "fail":
                    return "behavior_mismatch"
                if not item.applicable:
                    return "model_not_applicable"
                if item.error_code:
                    return "request_unavailable"
                return "insufficient_signal"

            result = {
                "status": "completed",
                "overall": overall,
                "score": round(summary["score"] * 100, 1),
                "pass_threshold": round(L1_HIGH * 100, 1),
                "evidence_coverage": round(summary["coverage"] * 100, 1),
                "coverage_threshold": round(L1_MIN_COVERAGE * 100, 1),
                "protocol": protocol,
                "declared_model": request.declared_model,
                "dimensions": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "result": (
                            item.status
                            if item.status in ("pass", "fail", "na")
                            else "na"
                        ),
                        "reason": public_reason(item),
                        "detail": item.detail,
                    }
                    for item in probe_results
                ],
                "duration_ms": round((time.monotonic() - started_at) * 1000),
                "error": None,
            }
            logger.info(
                "detection_completed id=%s model=%s protocol=%s result=%s coverage=%s ms=%s",
                detection_id,
                request.declared_model,
                protocol,
                overall,
                result["evidence_coverage"],
                result["duration_ms"],
            )
            return result
    except TimeoutError:
        return finish_incomplete("timeout", "检测超时，请稍后重试")
    except asyncio.CancelledError:
        logger.info("detection_cancelled id=%s", detection_id)
        raise
    finally:
        await client.close()
