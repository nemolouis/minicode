# M5：从可验证任务到 SFT

逐步记录和每个决定的理由见 [next_phase_plan.md](next_phase_plan.md)，沙箱与生成器的设计见 [m5_sandbox_taskgen_design.md](m5_sandbox_taskgen_design.md)。

## M5：从可验证任务到 SFT（闭环跑通一圈，已收工）

目标是补上"下层"的一条完整链路：**可验证任务 → teacher 轨迹 → SFT → 用同一套 harness 评测前后差异**。逐步记录、每个决定的理由和被推翻的假设见 [docs/next_phase_plan.md](next_phase_plan.md)，沙箱与生成器的设计见 [docs/m5_sandbox_taskgen_design.md](m5_sandbox_taskgen_design.md)。

```
种子库(seeds/) ─▶ 变异+筛选 ─▶ 352 道题(按种子库分 train/dev/test) ─▶ teacher 采样 ─▶ 导出(只取 train) ─▶ LoRA ─▶ 评测(dev/test: 未见过的库)
                  (沙箱内)       每道题都过 verify                     (DeepSeek flash)    510 条            (第3次成功)  (dev 0→4/48,不稳定)
```

```bash
minicode-eval sandbox-build                                              # 沙箱镜像(需要 Docker)
minicode-eval gen --seeds seeds --out gen_tasks --per-seed 24            # 生成任务
minicode-eval --tasks-root gen_tasks verify --sandbox --workers 4        # 每题:初始必须失败、参考解必须通过
minicode-eval --tasks-root gen_tasks run --config configs/gen-synth-flash.json --split train --sandbox --workers 4 --repeat 2
minicode-eval export results/<run-dir> --out data/sft_train_v2.jsonl --models deepseek-flash
# 训练在单独的环境里(Python venv + mlx-lm),脚本见 scripts/sft_mlx.py:
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install --only-binary :all: mlx-lm
.venv-mlx/bin/python scripts/sft_mlx.py check --model models/<m> --data data/sft_train_v2.jsonl     # 验证 token 数与助手轮掩码
.venv-mlx/bin/python scripts/sft_mlx.py train --model models/<m> --data data/sft_train_v2.jsonl --out adapters/<name> --max-len 8192 --epochs 2 --accum 4
```

已经测到的事实(全部带样本量):

- **题目**:352 道,来自 16 个种子库(14 个 PyPI 小库 + 2 个手写)。对种子库注入一个小 bug,只有同时满足下列条件才保留:能编译、原代码通过全部测试、可见测试至少失败一个、失败数不多、两次结果一致且不超时、同一失败模式只留少数、构造出的任务本身再核对一遍。划分**按种子库**:train 257 题(12 个库)、dev 48 题(colorama、schema)、test 47 题(sqlparse、xmltodict)。
- **teacher**:30 道题的试点上,DeepSeek flash 30/30、平均 8.3 步,Gemini 3.5-flash-lite 25/30、15.0 步(配对差 +16.7 点,区间 3.3~30.0,样本小)。放量用 DeepSeek:257 题 × 2 次,513/514 通过;价格表估算每次约 $0.002~0.004(价格表里的列对应关系标注了"待核对",以账户余额为准)。
- **训练数据**:510 条,覆盖 257 题、12 个库;真实 token 数共 371 万,中位 6.2k,p90 13.1k,最大 32k;只有 9.1% 的 token 有 loss(助手轮)。≤8192 token 的有 330 条(65%)。训练 `--max-len 8192`,更长的**整条丢弃、不截断**,所以实际训练集 313 条,**偏向短而简单的题**。
- **训练前基线**:未训练的 Qwen2.5-Coder-1.5B-Instruct 在 dev 48 题上 **0/48**,48 次都是 `no_edit`、1 步:它不按 `<tool_call>` 格式调用工具(输出 JSON 代码块或计划文字),服务端因此没有解析出工具调用。这个 0% 说的是"格式不会",不是"修不好 bug"。训练后如果有提升,要把"学会了格式"和"真的会修"分开看(失败类别、修改的文件),不能只报一个通过率。
- **沙箱**:见 [features.md](features.md)。验证 352 道题时发现过一个间歇性问题:新容器偶尔读到宿主刚替换过的文件的旧版本(判分时约 8/100),已用"判分用新副本 + 每条命令前对齐文件清单,对不上就报错"修复(0/300)。

**训练与评测的实际情况**（完整记录、数字和局限见 [docs/next_phase_plan.md](next_phase_plan.md)）：

- **训练**：前两次发散（第一次学习率 1e-4、无监控，白跑 5.5 小时；第二次被自动止损守卫在第 88 步停下）。第三次打开了 Adam 偏差修正（MLX 的 AdamW 默认关闭，会把前 100 次更新的步长放大 3~6 倍，已用数值验证），并只训练最后 16 层（第 0 层的 LoRA 梯度比其他层大约 14 个数量级），456 步没有发散。这和"偏差修正是原因"一致，但只有一次运行、同时改了几样，**因果是推断，不是证明**。调试显存时还触发过一次 macOS 内核崩溃。
- **一个有意思的根因**：Qwen2.5-Coder-1.5B 的 `<tool_call>`/`</tool_call>` 两个 token 的嵌入是没训练过的占位值（与 124 号 token 相差约一个 bf16 舍入单位），输出层与嵌入共享权重，所以微调后的模型学会了整个格式，却只能在标记位置输出 124，服务端收不到工具调用。客户端因此提供 `text_tool_calls`（宽容地读回复里的 JSON 工具调用，默认关闭，有测试）；训练前基线用同样的解析重测。
- **结果**：dev 48 题，基础 0/48，训练后 4/48（95% 区间 3.3%~19.6%，Fisher p=0.117，**不显著**）。训练前一次工具调用都没有，训练后 48 次都有、8 次改了文件。失败主要是 `edit_file` 抄不准要替换的代码（31 次编辑 29 次报错）、以及编辑失败后仍声称已修好（20 个）。验证损失最低的第 300 步检查点端到端是 0/48，说明用 7 条验证样本的损失挑检查点不可靠。

**还没做**：数据消融（带/不带思考文本、数据量、是否过滤）、一次 test 集评测、只改 Adam 偏差修正的对照实验、每题多次重复以缩小区间、按步切窗的训练（只做了统计）。

### 后续实验（2026-10-25～26）：都没有带来通过数提升

训练后想改进 `edit_file` 失败的问题，做了三轮，**结果都是负面或无效**，完整记录在 `docs/next_phase_plan.md`。以下都是 dev 48 题、每题 1 次、温度 0，与上面 4/48 的基线逐题配对。

- **给 `edit_file` 的失败信息加提示**（返回文件里最相近的真实行和行号，`edit_hints`，默认关）：4/48 → 3/48，退步 1 题、改进 0 题，没有帮助。原因：失败之后模型下一次调用与上一次完全相同的有 51/58；94 次编辑里 76 次的 `old_string` 在之前的任何工具输出里都找不到，是凭空写的。训练数据里 95% 的首次编辑是照着读到的内容抄的，但没有任何"失败后纠正"的样本（老师的 513 次编辑调用也是 0 次失败）。
- **把 `old_string` 缩短后重训（v4）**：0/48，48 次运行都是说一句话就结束、没有发出工具调用，所以这一轮**不能说明缩短有用或没用**。
- **根因是工具调用标记 token**：`<tool_call>` 的嵌入没训练过，模型能否发出标记取决于权重里很小的差异，不同检查点会随机翻转。因此上面基线的 4/48 依赖一个不稳定的状态，**不应当作稳定能力引用**。
- **改成普通文本工具调用格式（`[TOOL_CALL]`，v5）**：格式层的问题解决了——48/48 次运行都发出了工具调用（共 623 次，基线 333 次）；但通过 0/48，`edit_file` 全程只用了 1 次，54% 的调用与上一次完全相同（反复跑同一条 pytest）。也就是说下一层问题是行为：学生不会从探索转到修改。基线 4/48 对 0/48 的差异（McNemar 精确检验 p=0.125）不显著，也不能由此推出新格式更差——两次训练的数据条数不同（152 对 144），也没有重复运行。
- **数据量的瓶颈**：510 条里只有约 144～160 条放得下 4096 token（本机 16 GB 内存的限制）。把过长的工具输出截断只多留约 17% 的可用样本，而且会让 `old_string` 在前文不可见，不值得做；按步切窗能得到约 244 条可用的编辑样本（约 +74%），但总 token 量约是现在的 14 倍，本机跑不动，没有做。
