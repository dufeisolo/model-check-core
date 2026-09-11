import argparse
import json
import pytest
from unittest.mock import AsyncMock, patch

from model_check.cli import _run_cli, main, _render_markdown


def test_render_markdown():
    sample_result = {
        "declared_model": "claude-sonnet-4-5",
        "protocol": "anthropic",
        "overall": "pass",
        "score": 95.0,
        "pass_threshold": 70.0,
        "evidence_coverage": 100.0,
        "duration_ms": 1200,
        "dimensions": [
            {
                "id": "p1",
                "name": "连通性与结构",
                "result": "pass",
                "reason": "verified",
                "detail": "status 200",
            }
        ],
    }
    md = _render_markdown(sample_result, "https://api.example.com/v1")
    assert "# Model Check 诊断报告" in md
    assert "https://api.example.com/v1" in md
    assert "claude-sonnet-4-5" in md
    assert "PASS" in md
    assert "model-check.org" in md


@pytest.mark.asyncio
async def test_run_cli_invalid_url():
    args = argparse.Namespace(
        url="http://insecure-http.com",
        key="test-key",
        model="gpt-4o",
        output=None,
        format="table",
        no_color=True,
    )
    code = await _run_cli(args)
    assert code == 1


@pytest.mark.asyncio
async def test_run_cli_missing_key():
    args = argparse.Namespace(
        url="https://api.openai.com/v1",
        key=None,
        model="gpt-4o",
        output=None,
        format="table",
        no_color=True,
    )
    with patch("sys.stdin.isatty", return_value=False), \
         patch("sys.stdin.readline", return_value=""), \
         patch.dict("os.environ", {}, clear=True):
        code = await _run_cli(args)
        assert code == 1


@pytest.mark.asyncio
async def test_run_cli_mock_success(tmp_path):
    report_file = str(tmp_path / "report.md")
    args = argparse.Namespace(
        url="https://api.example.com/v1",
        key="sk-test-key-123456",
        model="claude-sonnet-4-5",
        output=report_file,
        format="table",
        no_color=True,
    )

    mock_result = {
        "status": "completed",
        "overall": "pass",
        "score": 100.0,
        "pass_threshold": 70.0,
        "evidence_coverage": 100.0,
        "protocol": "anthropic",
        "declared_model": "claude-sonnet-4-5",
        "dimensions": [
            {
                "id": "p1",
                "name": "连通性与结构",
                "result": "pass",
                "reason": "verified",
                "detail": "ok",
            }
        ],
        "duration_ms": 850,
        "error": None,
    }

    with patch("model_check.cli.validate_public_target", AsyncMock(return_value="https://api.example.com/v1")), \
         patch("model_check.cli.run_l1_detection", AsyncMock(return_value=mock_result)):
        code = await _run_cli(args)
        assert code == 0
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Model Check 诊断报告" in content
