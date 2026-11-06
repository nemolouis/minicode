# 开发指南

## 环境与日常命令

```bash
python3 -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev,serve]"
make check          # ruff + pytest + 黄金轨迹的格式检查（CI 跑的就是这个，另加 Docker 和 MCP 互操作两个作业）
```

支持 Python 3.10–3.13（在 3.10、3.11、3.12、3.13 上本地跑过全量测试；GitHub Actions 的矩阵配置见 `.github/workflows/ci.yml`，**这份工作流本身没有在 GitHub 上跑过**，只验证过 YAML 能解析）。不调用任何模型就能跑完全部测试；需要 Docker 沙箱镜像的测试在没有镜像时自动跳过（`make sandbox` 构建镜像后会真的运行）。

## 保证实验可信的几道防线

| 防线 | 位置 | 防什么 |
|---|---|---|
| 评测配置严格校验 | `miniagent/eval/config.py`，`tests/test_config.py` | 配置里的拼写错误（如 `require_reed`）曾经会被静默忽略，实验照跑但什么也没测；现在未知键、错类型、越界值、不存在的功能名/工具名/文件都直接报错，所有 `configs/*.json` 有测试保证合法 |
| 运行溯源 | `miniagent/provenance.py` | `meta.json` 和每条 `run_start` 记录 git 提交号、工作区是否有未提交改动、Python 版本；数字能对回产生它的代码 |
| 轨迹格式 | `miniagent/trajectory.py`（`SCHEMA_VERSION`、`validate_events`）| 格式是可执行的：`tests/conftest.py` 让**每个测试**产生的轨迹自动过一遍校验；`miniagent-eval check-trajectories <路径>` 可检查任意轨迹（没有 `schema_version` 的旧轨迹只做宽松检查） |
| 黄金轨迹回放 | `tests/golden/`，`scripts/make_golden.py`，`tests/test_golden_replay.py` | 4 条脚本化的轨迹（直路径、错误与恢复、两种压缩、步数耗尽）在真实任务上录制，每次测试都用**当前**主循环回放，调用、错误、压缩时机、结束方式、工具输出都必须一致。里面没有第三方模型输出（模型是脚本，答案取自任务的参考解）。故意改变行为后用 `make golden` 重新生成并审阅 diff |
| 变异检验 | 无脚本，是做法 | 写完测试后逐个破坏实现（去掉一道检查、改一个阈值），确认测试会失败；活下来的变异要么补测试，要么证明是冗余防线。最近一轮：执行器、压缩、会话、权限、MCP HTTP 约 100 处，最终幸存的都是冗余检查 |
| 文档链接 | `tests/test_docs_links.py` | README 拆分后相对链接不能断 |

## 代码结构（agent 核心）

| 模块 | 职责 |
|---|---|
| `agent.py` | 主循环：每步一次模型请求、记录轨迹、组装消息；不关心工具怎么跑、历史怎么压 |
| `executor.py` | 一轮里的工具调用：参数校验 → 权限规则 → 确认 → 执行 → 记录；只读调用并发，其余顺序（屏障） |
| `compaction.py` | 历史压缩：先抹旧输出，不够再让模型摘要；`compact_trigger` 决定何时压 |
| `session.py` | 会话保存/恢复、工作区检查点、回退（`Recorder` 是循环里用的那部分） |
| `features.py` | 可选功能：AGENTS.md 记忆、todo 工具、explore 子 agent |
| `permissions.py` / `schema.py` / `safety.py` | allow/ask/deny 规则；工具参数的 JSON schema 检查；bash 硬黑名单 |
| `trajectory.py` / `provenance.py` | 事件格式和校验；运行来源 |

## 常见操作

- **加一个评测配置键**：在 `miniagent/eval/config.py` 的 `_spec()` 里登记（类型和检查），否则配置会被拒绝；在 `runner.py` 里读取；写测试。
- **加一种轨迹事件或字段**：改 `trajectory.py` 的文档和 `_REQUIRED` / `_OPTIONAL`，必要时升 `SCHEMA_VERSION`；让查看器（`eval/viewer/index.html`）认识它；`check-trajectories` 和 conftest 会替你检查所有产生方。
- **改了主循环的行为**：先看 `tests/test_golden_replay.py` 是否失败。失败说明行为变了，确认是有意的再 `make golden`，并在提交说明里写出原因。
- **新增 results 目录后**：`make results-index`。
- **MCP 互操作**：官方 SDK 装在单独的环境里（`python3 -m venv .venv-sdk && .venv-sdk/bin/pip install mcp && .venv-sdk/bin/pip install -e .`），然后 `make interop`。
