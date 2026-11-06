# 特性

项目总览见 [README](../README.md)。下面是 agent 与周边工具已经实现的全部功能；还没做的见 README 的"已知局限"和路线图。

## 特性（已实现）

- **agent 主循环**：直接调用模型的 function calling 接口，最大步数限制，可配置模型、temperature、系统提示词、可用工具
- **工具**：`read_file` / `write_file` / `edit_file`（要求匹配唯一）/ `bash`（超时、输出截断，超时会杀掉整个进程组）/ `grep`；文件路径限制在工作区内
- **权限与安全**：写文件和 shell 需确认；危险命令硬拦截（`--yes` 也无法绕过）
- **上下文管理**：先抹掉旧的工具输出，仍超限再让模型做摘要；保证 tool_calls 与对应 tool 消息成对、不被切断；压缩后仍超阈值时不再逐步重试，而是等输入增长 `compact_regrowth`（默认翻倍）后再压（案例 2）
- **轨迹落盘**：每次运行写一份 JSONL，包含模型输出、工具调用与结果、耗时、token（含缓存与推理 token）、压缩事件、服务端自报的模型名
- **录制与回放**（`miniagent/replay.py`、`miniagent-eval replay`）：`CassetteClient` 按请求哈希缓存模型响应（`run --cassette 文件 --cassette-mode auto|record|replay`，replay 模式不需要 API key，没录过的请求直接报错而不是联网）；`replay` 命令把旧轨迹里的模型回答按顺序喂给**当前**的主循环，在新的任务副本里真实执行工具，逐步比对工具调用、压缩时机和结束状态，用来零成本检查"改了 loop 之后行为变没变"。局限：回放不能预测新模型或新提示词的行为；压缩决策依赖按字符数估算的历史长度，所以工作区路径长度、Gemini 思考签名这类细节都会影响结果（旧的 Gemini 轨迹没记签名，回放时会跳过并说明原因）。
- **SFT 数据导出**（`miniagent-eval export`）：把通过的运行写成聊天格式 JSONL（system、tools、带 tool_calls 的 assistant、tool 消息）。只留隐藏测试通过且自己结束、没改测试、没发生压缩、没有非法工具参数的运行；默认只导出 `train` 划分（`--splits` 可改；dev、test 和没有划分的手写题都不会进训练数据）；精确重复去重；每种跳过原因都计数。`run_start` 现在记录系统提示词全文和工具定义；`log_reasoning: true` 时记录模型的可见思考。
- **Docker 沙箱**（`miniagent/sandbox.py`，设计见 [docs/m5_sandbox_taskgen_design.md](m5_sandbox_taskgen_design.md)）：`miniagent --sandbox`、`miniagent-eval run --sandbox`（或配置里 `"sandbox": true`）让 `bash` 和判分用的测试都跑在容器里：断网、根文件系统只读、去掉全部 capability、非 root、不传任何环境变量（拿不到 API key）、限制进程数/内存/CPU，只挂载工作区。每个运行一个长驻容器、每条命令一次 `docker exec`；命令超时或容器出问题时换新容器；输出有上限，洪水不会撑爆宿主内存。镜像用 `miniagent-eval sandbox-build` 构建，`sandbox-clean` 清理崩溃遗留的容器。沙箱自身故障（Docker 挂了等）让运行以 error 结束、不计入通过率，不会当作工具结果显示给模型。21 个测试（其中 18 个对真实容器做逃逸检查），对每个安全选项做过变异检验。**不是虚拟机**：容器与宿主共享内核。
- **变异式任务生成**（`miniagent-eval gen`，`miniagent/taskgen/`）：对"能通过自带测试的种子库"注入一个小 bug（比较/算术/布尔运算符、条件取反、整数常量±1、break↔continue、删除语句），只在能通过全部筛选时才保留为任务：能编译、原代码通过全部测试、可见测试至少失败一个、失败数不多、两次结果一致且不超时、同一失败模式的任务有数量上限、构造出的任务本身再核对一遍。可见测试是种子测试的随机子集，隐藏测试是完整文件。不调用任何模型。生成的任务放在 `gen_tasks/`（不进入 `tasks/` 与既有的 dev/test 划分），**按种子库划分** train/dev/test，导出 SFT 数据默认只取 train。
- **工具参数校验**（`miniagent/schema.py`，默认开）：调用前按工具的 JSON schema 检查 `type` / `required` / `enum` / `items`，错误信息指出哪个字段错了并给出期望的签名（如 `offset: expected integer, got str '5'. Expected {path: string, offset?: integer, ...}`），工具不会被执行。未知字段和不认识的 schema 关键字不报错。`--no-validate` / 配置 `validate_args: false` 可关闭。
- **并行工具调用**（默认开）：同一轮里**连续**的只读工具（`read_file`、`grep`、`glob`、`explore`，无需确认）并发执行；写文件、`bash`、需要确认的工具单独按顺序执行，相当于屏障。返回给模型的消息和轨迹里的 `tool_result` 顺序始终是模型发出调用的顺序，并发的调用带 `parallel: N` 标记。`--no-parallel` / `parallel_tools: false` 关闭。
- **可选功能**（默认全关，所以旧配置、旧轨迹、回放不受影响；`--features memory,todo,explore,glob` 或配置里 `"features": [...]`，实现在 `miniagent/features.py`）：
  - `memory`：把工作区根目录的 `AGENTS.md`（没有则 `CLAUDE.md`）接到系统提示词后面，最多 8000 字符，符号链接不读；提示里写明这只是仓库主人的说明、不改变工具和权限。
  - `todo`：`todo_write` 工具，每次调用替换整张清单，最多一项 `in_progress`；摘要式压缩时当前清单会被带进新的开头消息，不会随历史丢失。
  - `explore`：只读子 agent。父 agent 调用 `explore(question)`，子 agent 在**全新的上下文**里只能用 `read_file` / `grep` / `glob`（`write_file`、`edit_file`、`bash` 被禁用；不能再开子 agent），最多 12 步，只把一份简短报告（≤6000 字符）返回给父 agent。子 agent 的轨迹写在父轨迹旁的 `explore-N/`，父轨迹里有一条 `subagent` 事件；子 agent 的 token 计入 `run_end` 和 `trajectory_stats`（按未缓存输入算），同一轮的多个 `explore` 并发执行。含 `explore` 的轨迹不能被确定性回放（子 agent 的模型调用不在父轨迹里）。
  - `glob`：按模式列文件（最多 200 条，跳过 `.git` / 虚拟环境，拒绝 `..` 和指向工作区外的符号链接）。
- **权限规则**（`miniagent/permissions.py`，`--permissions rules.json` 或配置 `permissions`）：`{"deny": [...], "ask": [...], "allow": [...]}`，规则写成 `tool` 或 `tool(模式)`，如 `bash(curl*)`、`write_file(.env*)`、`edit_file(**/secrets/**)`、`mcp__notes__*`。优先级 deny > ask > allow > 工具默认；`--yes` 不能覆盖 deny，allow 也绕不过 `safety.py` 的硬黑名单。路径先规范化（`src/../.env`、工作区内的绝对路径都按 `.env` 匹配）。shell 命令按 `;` `&&` `||` `|` `&` 换行拆段：deny 命中任一段即拒绝；allow 要求每一段都被覆盖，且命令里没有 `$(...)`、反引号、进程替换、重定向，所以 `bash(pytest*)` 不会放行 `pytest; rm -rf x` 或 `pytest > ~/.bashrc`（退回到询问）。轨迹的 `tool_result` 记录 `permission` 和命中的 `rule`。这是便利过滤器，**不是沙箱**；符号链接不会被路径规则跟随。
- **先读后改与改后诊断**（默认关，`--require-read` / `--diagnostics`，配置里 `require_read` / `diagnostics`）：`edit_file` 拒绝本次运行没有 `read_file`/`write_file` 过的已有文件（针对弱模型凭空写 `old_string`）；改完 `.py`/`.json` 后如果文件不再能解析，在工具结果里追加一行警告（只 `compile`，不执行代码）。
- **会话恢复与回退**（`miniagent/session.py`；`miniagent` CLI 默认保存，`--no-session` 关闭）：每步把消息列表写入 `session.json`，在可能改文件的步骤开始前把工作区拷贝到 `checkpoints/step-N/`。`miniagent "后续指令" -w repo --resume runs/<id>` 接着聊；`miniagent --rewind runs/<id> --to-step N -w repo` 把文件和对话都退回第 N 步开始时（删除之后新建的文件、恢复被改的文件），之后可再 `--resume`。限制：压缩过的会话只能退到最近一次压缩之后；快照跳过 >1MB 的文件、总量超 50MB 则拒绝回退；不拷贝 `.git` 和虚拟环境；只还原工作区，命令在工作区外造成的副作用不管。运行目录在工作区内时自动排除，不会被快照或删除。
- **HTTP 服务**（可选，`miniagent/server.py`，见 [mcp-and-service.md](mcp-and-service.md)）：把 `Agent.run` 包成 FastAPI 接口，路径白名单、在拷贝上运行、bash 只进沙箱、默认只监听本机。
- **模型接入**：标准库实现，无第三方 SDK。两种协议：OpenAI 兼容的 chat completions（DeepSeek、智谱 GLM 已验证）和 **Gemini 原生 `generateContent`**（含思考签名的保存与回放，可经转发型中转使用）。配置文件里用 `base_url` / `base_url_env` / `api_key_env` / `extra_body` / `api_style` 指定，密钥永远从环境变量读

**尚未实现**（避免误解）：按条件导出轨迹到网页之外的格式（SFT 导出已有，见上）；和主流 coding agent 相比还缺交互界面（TUI / 流式 / 中途打断）、hooks / skills / 插件、git 集成（分支 / 提交 / PR）、真正的 LSP 与符号索引（只有改后的语法检查）、Anthropic 原生协议、plan 模式（只有 todo 工具）、MCP 的 OAuth / resources / prompts。**本节新增的功能有单元测试和变异检验；其中 todo / explore / 先读后改 / 诊断 / 权限在 flash 上做过一次评测，但几乎没被触发，结果在噪声内（见"可选功能的评测结果"）；memory 和会话恢复/回退没有用真实模型评测过。**
