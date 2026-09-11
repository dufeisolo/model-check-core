# Model Check Core (`model-check`)

<div align="center">

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](tests/)
[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)](tests/)
[![Online Service](https://img.shields.io/badge/Online%20Web-model--check.org-00b894.svg)](https://model-check.org)

**极速、轻量、专业的大模型 API 渠道与中转站黑盒体检工具（LLM API Inspector）**  
*像给 API 做一次全套“深度体检”：30 秒排查功能阉割、假冒回显、流式假死与参数异常。*

[在线免安装体检](https://model-check.org) • [快速上手](#-快速上手) • [检测模式](#-检测模式分层) • [检测项目全景](#-检测了哪些内容29-项探针全景) • [Python SDK](#-python-sdk-使用)

</div>

---

## 🎯 为什么需要 Model Check？

接入第三方 API 中转站或聚合网关时，很多开发者常遇到这样的困扰：
- ❌ **“为什么我的 Cursor / Cline / NextChat 一用就报错？”** —— 表面返回 HTTP 200，但实际工具调用（Tools）或流式输出被中间件阉割。
- ❌ **“我买的 Claude 3.5 Sonnet 是不是被套壳了？”** —— 拿廉价小模型套壳回显，或者删掉了 Prompt Caching 导致开销剧增。
- ❌ **“这个中转商到底稳不稳？”** —— 接口报错格式混乱、Token 计费暗中虚标、视觉多模态丢包。

**Model Check** 通过非侵入式的黑盒主动探测技术，在 **15 ~ 30 秒** 内下发 29 组工程级高灵敏探针，直观给出每个维度的 **通过 / 未通过 / 不适用** 结论。

> 🌐 **免安装在线体检**：如果你不想在本地配置环境，欢迎直接使用免费在线体检站：[**model-check.org**](https://model-check.org)

---

## ✨ 四大核心特点

<table>
  <tr>
    <td width="50%">
      <h3>⚡ 极速低耗，立等可取</h3>
      <p>单次体检仅需 <b>15 ~ 30 秒</b>，总消耗约几千 Token（成本不足几分钱），采用智能并发与熔断保护，不浪费你的宝贵额度。</p>
    </td>
    <td width="50%">
      <h3>🔄 双协议全自动适配</h3>
      <p>自动嗅探 <b>Anthropic Messages 原生协议</b> 与 <b>OpenAI 兼容协议</b>，内置 <code>max_completion_tokens</code> 等新版推理模型参数自适应与智能回退。</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔒 零落库安全防御</h3>
      <p>API Key 仅在当前单次内存进程中使用，<b>绝不写入磁盘、日志或外部网络</b>；内置公网 IP 白名单校验与 <b>DNS 防重绑定（防 SSRF）</b>。</p>
    </td>
    <td width="50%">
      <h3>🛠️ 多场景自由接入</h3>
      <p>支持开箱即用的终端彩色 CLI、一行命令免安装运行（<code>uvx</code>）、Python SDK 代码级嵌入，以及用于团队监控的 CI/CD 自动化工作流。</p>
    </td>
  </tr>
</table>

---

## 🧭 检测模式分层与研发进度

Model Check 规划了三层梯度检测架构，兼顾“日常秒级体检”与“深度防伪鉴真”：

| 检测层级 | 研发状态 | 模式定位 | 核心目标 | 耗时与 Token 消耗 | 开放形态 |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **L1 极速黑盒体检** | 🟢 **已发布开源** | **功能与协议排障** | 协议规范、SSE 流式、Tools、视觉多模态、Prompt Caching、参数容差矩阵 | **15~30 秒**<br>约 3K~5K Tokens | **完全开源（本项目）**<br>CLI / 本地 / CI 均可跑 |
| **L2 行为指纹比对** | 🟡 **正在研发中** | **模型防伪鉴真** | 8组判别知识盲区、指令遵从边界、对齐风格，提取特征向量比对 | **1~2 分钟**<br>约 2.5 万 Tokens | 依托云端官方金标指纹库持续更新比对 |
| **L3 深度能力曲线** | 🔵 **正在调研中** | **性能与降智筛查** | 参数化数理、代码逻辑、长文本对抗评测，识别量化缩水与模型降智 | **3~5 分钟**<br>约 10 万 Tokens | 防刷题动态题库，后续上线深度评测站 |


---

## 📋 检测了哪些内容？（29 项探针全景）

L1 极速体检覆盖了 API 渠道日常运行最容易踩坑的全部核心维度。

> 💡 **代号说明**：
> - `p1 ~ p16`：**Anthropic 协议探针代号**（**P** = **P**robe，对应 Claude 官方 `/v1/messages` 格式）。
> - `o1 ~ o13`：**OpenAI 协议探针代号**（**O** = **O**penAI，对应 `/v1/chat/completions` 兼容格式）。  
> 对应体检报告与排障日志中每一项测试维度的唯一追踪 ID。


### 1. 基础协议与数据流传输（通信基石）
| 探针项目 | 说明与排查痛点 | Anthropic | OpenAI |
| :--- | :--- | :---: | :---: |
| **连通性与结构** | 校验接口能否返回合法的标准 JSON 响应结构 | ✅ (p1) | ✅ (o1) |
| **模型回显严格核对** | 检查返回的 `model` 字段是否与声明一致（防低劣换壳冒充） | ✅ (p2) | ✅ (o2) |
| **SSE 流式传输规范** | 校验打字机流式事件流是否完整，是否出现断流、漏 chunk | ✅ (p4) | ✅ (o4) |
| **错误对象规范性** | 故意触发错误，校验返回的错误格式是否符合官方 SDK 规范 | ✅ (p7) | — |
| **Stop 停止词与截断** | 验证 `stop` 序列与 `max_tokens` 参数是否真实生效截断 | ✅ (p8, p9) | ✅ (o3, o5) |

### 2. 高阶模型能力支持（功能是否被中转站阉割）
| 探针项目 | 说明与排查痛点 | Anthropic | OpenAI |
| :--- | :--- | :---: | :---: |
| **Tool Use (工具调用)** | 验证函数调用（Function Calling）能否正常触发并传参 | ✅ (p10) | ✅ (o8) |
| **Vision (视觉多模态)** | 传入 Base64 图像，验证模型能否正确辨析视觉内容 | ✅ (p13) | ✅ (o9) |
| **Document (文档解析)** | 传入 PDF 等原生文档结构，检验多模态文档解析能力 | ✅ (p14) | — |
| **Prompt Caching** | 测试提示词缓存机制是否生效（直接影响中转使用成本与速度） | ✅ (p6) | — |
| **Reasoning Effort** | 检验 o1/o3/o4/GPT-5 等新一代推理模型的思考参数兼容性 | — | ✅ (o10) |

### 3. 一致性与稳定性校验（暗坑检测）
| 探针项目 | 说明与排查痛点 | Anthropic | OpenAI |
| :--- | :--- | :---: | :---: |
| **Token Usage 计数审计** | 审计返回的 input/output token 与实际文本偏差（防虚标扣费） | ✅ (p5) | ✅ (o11) |
| **System Prompt 遵从** | 检验前置系统指令是否被网关中间件吞掉或篡改 | ✅ (p12) | ✅ (o7) |
| **多轮上下文记忆** | 检验两轮连续对话中的上下文保持能力（首次成功即短路结束） | ✅ (p11) | ✅ (o6) |
| **响应头与重放缓存** | 探测响应头指纹特征，检验是否存在陈旧数据缓存直接重放 | ✅ (p3, p15)| ✅ (o12) |

### 4. 渠道底层指纹探针（高阶识别）
* **参数校验边界矩阵**（Anthropic p16 / OpenAI o13）：
  探测网关对 `temperature` 越界值（0.0, 1.5, 3.0 等）、`top_p` 越界形态的拦截态度，并嗅探对侧端点（如 `/v1/messages`）。该探针权重为 0（不影响得分），专门用于识别中转网关背后的底层路由架构。

---

## 🚀 快速上手

### 方式 1：免安装直接运行（推荐 uvx）

无需配置 Python 环境，一行命令直接对目标渠道做体检：

```bash
uvx --from model-check-core model-check --url https://api.your-relay.com/v1 --key sk-xxxxxx --model claude-sonnet-4-5
```

### 方式 2：使用 pip 安装并运行

```bash
pip install model-check-core
```

安装完成后在终端运行：

```bash
model-check --url https://api.your-relay.com/v1 --key sk-xxxxxx --model gpt-4o
```

> 💡 **小技巧**：你也可以通过环境变量传入 API Key，避免在终端命令历史中留下记录：
> ```bash
> export OPENAI_API_KEY="sk-xxxxxx"
> model-check --url https://api.your-relay.com/v1 --model gpt-4o
> ```

---

## 🖥️ 命令行参数与报告导出

```text
用法: model-check [选项]

选项:
  -u, --url TEXT        API Base URL (例如: https://api.openai.com/v1) [必需]
  -k, --key TEXT        API Key (默认自动读取 MODEL_CHECK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
  -m, --model TEXT      声明的模型名称 (默认: claude-sonnet-4-5)
  -o, --output PATH     导出检测报告文件路径 (.md 或 .json)
  --format [table|json|markdown]
                        控制台输出格式 (默认: table)
  --no-color            禁用彩色终端输出
  -h, --help            显示帮助信息并退出
```

### 一键导出 Markdown 渠道体检报告

```bash
model-check -u https://api.your-relay.com/v1 -m claude-sonnet-4-5 -o channel_report.md --format markdown
```
导出的 Markdown 报告会自动包含排版完整的检测表格与结论，非常适合直接粘贴到 GitHub Issue、社区论坛或团队内部分享。

---

## 🐍 Python SDK 使用

你可以在自己的自动化测试套件或渠道定时健康检查脚本中直接引入：

```python
import asyncio
from model_check import L1DetectionRequest, run_l1_detection

async def main():
    request = L1DetectionRequest(
        base_url="https://api.your-relay.com/v1",
        api_key="sk-xxxxxx",
        declared_model="claude-sonnet-4-5",
    )

    def on_progress(event, data):
        # 实时监听探测进度
        if data.get("action") == "finished":
            print(f"[{data.get('result').upper()}] {data.get('name')}")

    result = await run_l1_detection(request, progress=on_progress)
    
    print("\n--- 体检总评 ---")
    print("总体结论:", result["overall"])  # pass / fail / inconclusive
    print(f"得分: {result['score']}% (通过线: {result['pass_threshold']}%)")
    print(f"证据覆盖率: {result['evidence_coverage']}%")

asyncio.run(main())
```

---

## 🧭 开源协同与在线平台

- **开源公共仓库 (`model-check-core`)**：作为完全透明、客观中立的 L1 协议与功能检测核心，欢迎社区开发者提交 PR，共同扩充模型能力表（`model_profiles.py`）及新协议适配。
- **在线平台 ([model-check.org](https://model-check.org))**：提供免安装的 Web 可视化体检、中转站信誉看板，并持续跟踪维护官方基准指纹库。

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
