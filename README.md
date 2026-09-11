# Model Check Core (`model-check`)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Online Service](https://img.shields.io/badge/Web%20App-model--check.org-green.svg)](https://model-check.org)

**Model Check** 是一个极速、轻量、客观的开源大模型 API 渠道黑盒诊断与体检工具（LLM API Inspector）。

通过真实黑盒 API 探测，在 **15 ~ 30 秒** 内排查 API 中转站/代理渠道是否存在功能阉割、协议不符、假冒回显或参数异常。

> 🌐 **免安装在线体检**：如果你不想在本地配置环境，欢迎直接使用免费在线体检站：[**model-check.org**](https://model-check.org)

---

## ✨ 核心特性

- ⚡ **15~30 秒极速诊断**：单次体检仅消耗几千 Token（几分钱成本），即可快速出具完整报告。
- 🔍 **双协议智能识别**：自动嗅探 Anthropic Messages 原生协议与 OpenAI 兼容格式，自适应 `max_completion_tokens` 等推理模型新参数。
- 🛡️ **深层协议与功能探针**：
  - **Anthropic 协议（16 项探针）**：SSE 流式事件、Tool Use、Prompt Caching、System Prompt 服从性、多轮上下文、图像/PDF 输入识别、错误对象结构、模型回显一致性、参数校验矩阵等。
  - **OpenAI 兼容协议（13 项探针）**：Tools 调用、Vision 视觉多模态、Reasoning effort（o1/o3/o4/GPT-5 兼容）、流式一致性、Usage 结构核对、参数校验矩阵等。
- 🔒 **严格安全防御**：
  - **零保存保证**：API Key 仅在单次内存会话中调用，绝不写入磁盘、日志或网络上报。
  - **防 SSRF 保护**：内置公网 IP 校验与 DNS 防重绑定保护，拒绝回环地址与内网探测。
- 📊 **多端可用**：支持终端 CLI 命令行、Python SDK 编程调用、CI/CD 自动化集成。

---

## 🚀 快速开始

### 方式 1：免安装直接运行（推荐 uvx）

如果你安装了 [uv](https://docs.astral.sh/uv/)，无需手动安装即可直接运行：

```bash
uvx --from model-check-core model-check --url https://api.your-relay.com/v1 --key sk-xxxxxx --model claude-sonnet-4-5
```

### 方式 2：使用 pip 安装

```bash
pip install model-check-core
```

安装后直接在终端运行：

```bash
model-check --url https://api.your-relay.com/v1 --key sk-xxxxxx --model gpt-4o
```

你也可以将 API Key 设置为环境变量，避免在终端命令历史中留下记录：

```bash
export OPENAI_API_KEY="sk-xxxxxx"
model-check --url https://api.your-relay.com/v1 --model gpt-4o
```

---

## 🖥️ 命令行参数

```text
用法: model-check [选项]

选项:
  -u, --url TEXT        API Base URL (例如: https://api.openai.com/v1) [必需]
  -k, --key TEXT        API Key (默认自动读取 MODEL_CHECK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
  -m, --model TEXT      声明的模型名称 (默认: claude-sonnet-4-5)
  -o, --output PATH     导出检测报告文件路径 (.md 或 .json)
  --format [table|json|markdown]
                        输出格式 (默认: table)
  --no-color            禁用彩色终端输出
  -h, --help            显示帮助信息并退出
```

### 导出 Markdown 体检报告

```bash
model-check -u https://api.your-relay.com/v1 -m claude-sonnet-4-5 -o report.md --format markdown
```

---

## 🐍 Python SDK 使用

你可以在自己的测试套件、定时健康检查脚本中直接导入 `model_check`：

```python
import asyncio
from model_check import ApiClient, L1DetectionRequest, run_l1_detection

async def main():
    request = L1DetectionRequest(
        base_url="https://api.your-relay.com/v1",
        api_key="sk-xxxxxx",
        declared_model="claude-sonnet-4-5",
    )

    def on_progress(event, data):
        print(f"[{data.get('stage')}] {data.get('message')}")

    result = await run_l1_detection(request, progress=on_progress)
    print("总体结论:", result["overall"])  # pass / fail / inconclusive
    for dim in result["dimensions"]:
        print(f" - {dim['name']}: {dim['status']}")

asyncio.run(main())
```

---

## 🧭 开源与商业化定位

- **本仓库 (`model-check-core`)**：作为完全开源、透明的 L1 协议与功能体检核心，保持中立客观，为广大开发者提供开箱即用的排障工具。
- **在线平台 ([model-check.org](https://model-check.org))**：提供免安装的 Web 可视化极速体检、中转站状态看板，以及后续演进的模型行为指纹（L2）与基准评测能力。

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
