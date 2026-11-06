# MCP 服务器与 HTTP 服务（M4）

设计与取舍见 [m4_design.md](m4_design.md)。

## MCP 服务器（M4）

一个手写的、只读为主的 MCP 笔记服务器，加上 agent 侧的 MCP 客户端。设计与取舍见 [docs/m4_design.md](m4_design.md)。

```bash
python3 scripts/mcp_demo.py        # 不需要模型或 API key：演示边界（含笔记里的提示注入、符号链接、FIFO、越权调用）

# 让 agent 通过 MCP 读笔记（把内置的读文件/命令工具隐藏，只剩 MCP 工具）
minicode "规划会议上定的月度预算是多少？" -w examples -y -m deepseek-flash \
  --mcp-config examples/mcp_notes.json --disable-tools bash,grep,read_file,write_file,edit_file
```

- **服务端**（`minicode/mcp/notes_server.py`，标准库）：工具 `list_notes` / `read_note` / `search_notes`，以及默认**不存在**的写工具 `append_note`（只有 `--allow-write` 才启用）。同时支持新版协议（2026-07-28，每个请求自带版本和能力元数据，实现了 `server/discover`）和旧版握手（`initialize`）。stdout 只写 JSON-RPC，日志写 stderr，stdin 关闭即退出。
- **路径安全**（`minicode/mcp/safepath.py`）：工具参数一律当作攻击者输入。目录逐级用文件描述符 `openat` 打开且带 `O_NOFOLLOW`（检查和打开是同一个原子操作，没有"检查后被替换"的窗口，任何一级是符号链接就失败）；最终文件必须是单链接的普通文件（拒绝 FIFO、设备、硬链接）；只认 `.md/.txt/.rst`；`.` 开头的任何分量、`..`、绝对路径、`~`、NUL、反斜杠一律拒绝；不做 URL 解码；"不存在"和"不允许"返回同样的话，报错里不出现真实路径；有大小、数量、耗时上限；搜索是字面量，没有 ReDoS。
- **越权**：只读模式下 `append_note` 不在 `tools/list` 里，直接调用也只得到"未知工具"，磁盘上不会出现任何文件；`--allow-tools` 之外的工具同样不存在。agent 侧只暴露配置里 `allow` 列出的工具（名字加前缀 `mcp__notes__`），模型调用没暴露的名字时根本不会联系服务端；MCP 工具默认需要确认（规范说工具注解不可信）。
- **客户端**（`minicode/mcp/client.py`）：按规范先探测 `server/discover`，失败再回退到旧版握手；请求超时会发取消通知；独立进程组启动、关闭时按"关 stdin → SIGTERM → SIGKILL"收尾；stderr 单独读取，不会因日志太多而死锁；服务端只拿到最小环境变量，**拿不到我们的 API key**。
- **测试**（166 个，`tests/test_mcp_*.py`）：协议两代、各错误码和畸形输入；路径攻击（数十种写法，用真实的符号链接、硬链接、FIFO）；用钩子在"检查"和"打开"之间换文件的 TOCTOU；越权调用；客户端对 8 种行为异常的假服务端（旧版、静默、版本不匹配、超时、stderr 洪水、崩溃、拒绝退出、分页）；agent 集成，包括"被笔记诱导的模型"。我还做了**变异检验**：逐个去掉防御共 18 次，15 次让测试失败，3 次存活（都是因为有冗余防线，详见设计文档）。

不是沙箱：服务端以你的用户身份运行，上面都是应用层的检查，一个绕过检查的漏洞仍可读到你能读的文件；根目录应使用专用目录；硬链接检测靠 `st_nlink`（会误伤极少数合法情况）；提示注入没有被"解决"，只是让服务端的边界不依赖模型的判断。

## HTTP 服务（可选）

把 `Agent.run` 包成一个 FastAPI 服务（`minicode/server.py`）。agent 本身仍然零依赖，服务是可选依赖：`pip install -e ".[serve]"`。

```bash
export LLM_API_KEY="..."                       # 模型密钥只放环境变量
minicode-serve --workspace demo=path/to/repo  # 默认只监听 127.0.0.1，bash 跑在 Docker 沙箱里
```

| 接口 | 作用 |
|---|---|
| 浏览器打开 `http://127.0.0.1:8000/docs` | FastAPI 自带的交互页面，列出下面所有接口，可以直接填参数试（根地址 `/` 会跳转到它） |
| `POST /runs` `{task, workspace, model?, max_steps?}` | 提交任务，返回 `run_id`（202） |
| `GET /runs/{id}` | 状态、最终回复、token 数、改动的文件 |
| `GET /runs/{id}/events?after=N` | 轨迹事件（JSON，可轮询） |
| `GET /runs/{id}/stream` | 同样的事件，用 SSE 实时推送到运行结束 |
| `GET /runs/{id}/patch` | agent 改动的 unified diff |

一个会让模型执行命令的服务就是远程执行入口，所以默认值偏保守：客户端**不能传路径**，`workspace` 只是启动时白名单里的名字，每次运行在该目录的**私有拷贝**上进行，原目录不会被改；bash 只在 Docker 沙箱里跑，`--no-sandbox` 时 bash 直接被拒绝；非本机地址必须设置 `MINICODE_API_KEY`；任务长度、步数、模型名、排队数量都有上限。19 个测试覆盖这些行为，并对服务代码做了 9 处故意破坏（改成在原目录上运行、去掉认证、放开 bash 等），测试全部能发现。

**局限**：没有用真实模型做过端到端调用（没有密钥，只用脚本化的假模型测过，并用真实的 uvicorn 进程验证了启动、认证、参数校验和建任务）；运行记录保存在内存里，重启后接口里查不到（轨迹文件还在磁盘上）；只有单进程、没有持久化队列；不是生产级服务，它的作用是把循环服务化，并展示这类入口该有的安全边界。
