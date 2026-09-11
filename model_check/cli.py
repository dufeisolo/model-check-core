"""Model Check 命令行工具（CLI）。

提供终端彩色体检报告，并导流至 https://model-check.org 免费在线版。
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from typing import TextIO

from model_check.runner import L1DetectionRequest, run_l1_detection
from model_check.security import normalize_public_base_url, validate_public_target


class Colors:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, text: str) -> str:
        return self._wrap("32;1", text)

    def red(self, text: str) -> str:
        return self._wrap("31;1", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33;1", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36;1", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)


BANNER = """
======================================================================
                  Model Check - LLM API Inspector                     
            大模型 API 渠道黑盒体检与兼容性诊断工具                     
     🌐 免安装在线体检 / 渠道徽章: https://model-check.org            
======================================================================
"""


def _print_banner(c: Colors) -> None:
    print(c.cyan(BANNER.strip()) + "\n")


def _render_markdown(result: dict, target_url: str) -> str:
    lines = [
        "# Model Check 诊断报告",
        "",
        f"- **目标地址**: `{target_url}`",
        f"- **声明模型**: `{result.get('declared_model')}`",
        f"- **识别协议**: `{result.get('protocol')}`",
        f"- **总体结论**: **{result.get('overall', 'UNKNOWN').upper()}**",
        f"- **L1 加权得分**: `{result.get('score', 0)}%` (通过线: `{result.get('pass_threshold', 70)}%`)",
        f"- **有效证据覆盖率**: `{result.get('evidence_coverage', 0)}%`",
        f"- **耗时**: `{result.get('duration_ms', 0)} ms`",
        "",
        "## 检测维度明细",
        "",
        "| ID | 检测项目 | 结果 | 状态原因 | 说明 |",
        "|:---|:---|:---:|:---|:---|",
    ]
    labels = {"pass": "✅ 通过", "fail": "❌ 未通过", "na": "➖ 不适用"}
    for d in result.get("dimensions", []):
        res_label = labels.get(d["result"], d["result"])
        detail = d.get("detail", "").replace("\n", " ").replace("|", "\\|")
        lines.append(f"| {d['id']} | {d['name']} | {res_label} | {d['reason']} | {detail} |")

    lines.extend([
        "",
        "---",
        "*报告由 [Model Check](https://model-check.org) 开源体检核心自动生成。*",
    ])
    return "\n".join(lines)


async def _run_cli(args: argparse.Namespace) -> int:
    use_color = not args.no_color and sys.stdout.isatty()
    c = Colors(enabled=use_color)

    if args.format != "json":
        _print_banner(c)

    base_url = args.url.strip()
    try:
        base_url = normalize_public_base_url(base_url)
    except ValueError as e:
        print(c.red(f"❌ 目标地址校验失败: {e}"), file=sys.stderr)
        return 1

    api_key = args.key
    if not api_key:
        api_key = (
            os.getenv("MODEL_CHECK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
        )
    if not api_key:
        if sys.stdin.isatty():
            api_key = getpass.getpass("请输入 API Key (输入时隐藏): ").strip()
        else:
            api_key = sys.stdin.readline().strip()

    if not api_key:
        print(c.red("❌ 缺少 API Key，请通过 -k/--key 或环境变量提供"), file=sys.stderr)
        return 1

    if args.format != "json":
        print(f"🎯 目标渠道: {c.bold(base_url)}")
        print(f"📦 声明模型: {c.bold(args.model)}")
        print(c.dim("⏳ 正在进行公网 DNS 解析与安全防御检查..."))

    try:
        base_url = await validate_public_target(base_url)
    except Exception as e:
        print(c.red(f"❌ 安全防御拦截: {e}"), file=sys.stderr)
        return 1

    req = L1DetectionRequest(
        base_url=base_url,
        api_key=api_key,
        declared_model=args.model,
    )

    last_probe = ""

    def on_progress(event: str, data: dict):
        nonlocal last_probe
        if args.format == "json":
            return
        stage = data.get("stage")
        msg = data.get("message", "")
        if stage == "protocol":
            print(f"🔍 协议探测: {msg}")
        elif stage == "probes":
            action = data.get("action")
            if action == "started":
                name = data.get("name", "")
                sys.stdout.write(f"\r  ⏳ 正在检测: {name:<30}")
                sys.stdout.flush()
            elif action == "finished":
                name = data.get("name", "")
                res = data.get("result")
                res_str = (
                    c.green("PASS")
                    if res == "pass"
                    else c.red("FAIL")
                    if res == "fail"
                    else c.yellow("N/A ")
                )
                sys.stdout.write(f"\r  [{res_str}] {name:<30}\n")
                sys.stdout.flush()

    if args.format != "json":
        print(c.bold("\n🚀 开始 L1 极速黑盒体检:"))

    result = await run_l1_detection(req, progress=on_progress)

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("overall") == "pass" else 1

    print("\n" + "=" * 70)
    overall = result.get("overall")
    if overall == "pass":
        verdict_str = c.green("【通过 PASS】")
    elif overall == "fail":
        verdict_str = c.red("【未通过 FAIL】")
    else:
        verdict_str = c.yellow(f"【{overall.upper() if overall else '未完成'}】")

    print(f"📊 体检总评: {verdict_str}")
    if result.get("error"):
        print(c.red(f"❌ 错误详情: {result['error'].get('message')}"))
    else:
        print(
            f"📈 L1 加权得分: {c.bold(str(result.get('score')) + '%')} "
            f"(通过线: {result.get('pass_threshold')}%) | "
            f"有效证据覆盖率: {c.bold(str(result.get('evidence_coverage')) + '%')}"
        )
        print(f"⏱️ 耗时: {result.get('duration_ms')} ms")

    print("=" * 70)

    # 导出文件
    if args.output:
        out_path = args.output
        if out_path.endswith(".json") or args.format == "json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            md_content = _render_markdown(result, base_url)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)
        print(c.green(f"📁 诊断报告已导出至: {out_path}"))

    print("\n" + c.cyan("🌐 访问在线可视化体验站 / 获取中转站信誉徽章: https://model-check.org"))
    print(c.dim("💡 提醒: 本地 API Key 仅保存在当前内存进程中，测试完毕后建议在控制台删除临时 Key。"))

    return 0 if overall == "pass" else 1


def main():
    parser = argparse.ArgumentParser(
        prog="model-check",
        description="Model Check - 大模型 API 渠道黑盒体检与兼容性诊断工具",
    )
    parser.add_argument("-u", "--url", required=True, help="API Base URL (例如: https://api.openai.com/v1)")
    parser.add_argument("-k", "--key", help="API Key (默认读取环境变量或提示输入)")
    parser.add_argument("-m", "--model", default="claude-sonnet-4-5", help="声明的模型名称 (默认: claude-sonnet-4-5)")
    parser.add_argument("-o", "--output", help="导出报告文件路径 (.md 或 .json)")
    parser.add_argument(
        "--format",
        choices=["table", "json", "markdown"],
        default="table",
        help="控制台输出格式",
    )
    parser.add_argument("--no-color", action="store_true", help="禁用彩色输出")

    args = parser.parse_args()
    sys.exit(asyncio.run(_run_cli(args)))


if __name__ == "__main__":
    main()
