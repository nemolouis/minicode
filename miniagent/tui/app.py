"""A Claude-Code-style terminal application for the agent, built on Textual.

    ┌ ● miniagent · deepseek-chat · ~/proj · ⎇ miniagent/run1 ───────────────────────────────┐
    │  > fix the failing test                                                                 │
    │  I'll look at the test first.                                                           │
    │  ▸ ✓ read_file stats.py                                                                 │
    │  ▾ ✓ edit_file stats.py                                                                 │
    │      - return sum(xs[i:i+k]) / k                                                        │
    │      + return sum(xs[i:i+k]) / len(xs[i:i+k])                                           │
    │  ⠹ Working… 12s · esc to interrupt                                                      │
    │ ╭─────────────────────────────────────────────────────────────────────────────────────╮ │
    │ │ › _                                                                                 │ │
    │ ╰─────────────────────────────────────────────────────────────────────────────────────╯ │
    │  enter send · ctrl+j newline · / commands · esc interrupt · ctrl+b sidebar    1.2k tok  │
    └─────────────────────────────────────────────────────────────────────────────────────────┘

The agent runs in a worker thread; everything it reports (streamed text, tool calls, results, approvals) is marshalled to the UI thread.
All session logic (turns, /diff, /undo, /rewind, cost) lives in `miniagent.controller.SessionController`, shared with the line REPL.
"""
from __future__ import annotations

import time
from concurrent.futures import Future

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Markdown, OptionList, Static, TextArea

from ..controller import ControllerError, SessionController
from ..repl import describe_action

COMMANDS = {
    "/help": "keys and commands",
    "/diff": "everything changed since the session started",
    "/steps": "pick a step to rewind to",
    "/undo": "take back the last turn (files and conversation)",
    "/rewind": "rewind to step N  (/rewind 3)",
    "/cost": "tokens and cost so far",
    "/status": "model, workspace, mode",
    "/todo": "show or hide the side panel (tasks and changed files)",
    "/model": "switch model  (/model name)",
    "/yes": "approve everything without asking  (/yes on|off)",
    "/clear": "start a new conversation",
    "/exit": "leave",
}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
LOGO = r"""
  ┏┳┓ ┳ ┏┓┓ ┳  ┏┓ ┏┓ ┏┓ ┳┓ ╋
  ┃┃┃ ┃ ┃┃┃ ┃  ┣┫ ┃┓ ┣  ┃┃ ┃
  ┛ ┗ ┻ ┛┗┻ ┻  ┛┗ ┗┛ ┗┛ ┛┗ ┗
"""
FILE_TOOLS = {"edit_file", "write_file", "apply_patch"}


def diff_text(lines: list[str]) -> Text:
    out = Text()
    for line in lines:
        style = "green" if line.startswith("+") and not line.startswith("+++") else "red" if line.startswith("-") and not line.startswith("---") \
            else "cyan" if line.startswith("@@") else "bold" if line.startswith(("+++", "---")) else ""
        out.append(line + "\n", style=style)
    return out


def tool_brief(name: str, args: dict) -> str:
    if name == "apply_patch":
        return "patch"
    return str(args.get("command") or args.get("path") or args.get("pattern") or args.get("question") or args.get("query") or "")[:100]


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


# ---------------------------------------------------------------- widgets
class StatusBar(Static):
    pass


class UserMessage(Static):
    pass


class Note(Static):
    pass


class AssistantMessage(Markdown):
    """Assistant text; filled piece by piece through a MarkdownStream while the model is generating."""

    def __init__(self) -> None:
        super().__init__("")
        self._stream = None

    async def write(self, piece: str) -> None:
        if self._stream is None:
            self._stream = Markdown.get_stream(self)
        await self._stream.write(piece)

    async def finish(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None


class ToolCard(Collapsible):
    """One tool call: a title line that turns from … to ✓/✗ when the result arrives, with the diff / output folded inside."""

    def __init__(self, name: str, args: dict, call_id: str, preview: list[str] | None) -> None:
        self.tool_name, self.tool_args, self.call_id = name, args, call_id
        self.body = Static("")
        brief = tool_brief(name, args)
        self.base_title = f"{name}  {brief}".rstrip()
        super().__init__(self.body, title=f"… {self.base_title}", collapsed=name not in FILE_TOOLS)
        self.add_class("tool", "pending")
        self.has_preview = bool(preview)
        if preview:
            self.body.update(diff_text(preview))

    def set_result(self, d: dict) -> None:
        out = str(d["output"]).rstrip()
        err = d["is_error"]
        mark = "✗" if err else "✓"
        extra = f"  ·  {d['duration_s']:.1f}s" if d.get("duration_s", 0) >= 0.1 else ""
        if d.get("parallel"):
            extra += f"  ·  ∥{d['parallel']}"
        if d.get("permission") == "deny":
            extra += "  ·  denied by rule"
        self.title = f"{mark} {self.base_title}{extra}"
        self.remove_class("pending")
        self.add_class("error" if err else "ok")
        if err:
            self.collapsed = False
            self.body.update(Text(out[:3000], style="red"))
        elif self.tool_name in FILE_TOOLS:
            if not self.has_preview:                                   # a resumed session: only the result text is known
                self.body.update(Text(out))
        else:
            lines = out.splitlines()
            shown = "\n".join(lines[:40]) + (f"\n… ({len(lines) - 40} more lines)" if len(lines) > 40 else "")
            self.body.update(Text(shown))
            self.collapsed = len(lines) > 8 or self.tool_name not in ("bash",)


class ApprovalPanel(Vertical):
    """The question above the input box: what is about to happen (with a diff), and the three answers."""
    can_focus = True
    BINDINGS = [Binding("y", "answer('yes')", "allow"), Binding("enter", "answer('yes')", "allow"), Binding("a", "answer('all')", "allow all"),
                Binding("n", "answer('no')", "deny"), Binding("escape", "answer('no')", "deny")]

    class Answered(Message):
        def __init__(self, answer: str) -> None:
            super().__init__()
            self.answer = answer

    def __init__(self, name: str, lines: list[str]) -> None:
        super().__init__()
        self._name, self._lines = name, lines

    def compose(self) -> ComposeResult:
        yield Static(Text(f"Allow {self._name}?", style="bold yellow"), classes="approval-title")
        more = [f"… ({len(self._lines) - 40} more lines)"] if len(self._lines) > 40 else []
        yield Static(diff_text(self._lines[:40] + more), classes="approval-body")
        yield Static(Text.from_markup("[b green]y[/] allow   [b green]a[/] allow everything this session   [b red]n[/] deny   "
                                      "[dim](enter = allow, esc = deny)[/]"), classes="approval-keys")

    def on_mount(self) -> None:
        self.focus()

    def action_answer(self, answer: str) -> None:
        self.post_message(self.Answered(answer))


class PromptInput(TextArea):
    """The input box: Enter sends, Ctrl+J (or Alt+Enter) inserts a newline, Up/Down walk the history when the cursor is on the first/last line."""
    BINDINGS = [Binding("enter", "submit", "send", priority=True), Binding("ctrl+j", "newline", "newline", priority=True),
                Binding("alt+enter", "newline", "newline", priority=True), Binding("tab", "complete", "complete", priority=True, show=False)]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class Complete(Message):
        pass

    class History(Message):
        def __init__(self, direction: int) -> None:
            super().__init__()
            self.direction = direction

    def __init__(self) -> None:
        super().__init__("", show_line_numbers=False, soft_wrap=True, tab_behavior="focus", placeholder="Ask anything, or type / for commands")

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")

    def action_complete(self) -> None:
        self.post_message(self.Complete())

    def action_cursor_up(self, select: bool = False) -> None:
        if self.cursor_location[0] == 0 and not select:
            self.post_message(self.History(-1))
        else:
            super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if self.cursor_location[0] >= self.document.line_count - 1 and not select:
            self.post_message(self.History(1))
        else:
            super().action_cursor_down(select)


class TextScreen(ModalScreen):
    """A scrollable full-screen text (the diff, the help)."""
    BINDINGS = [Binding("escape", "dismiss", "close"), Binding("q", "dismiss", "close")]

    def __init__(self, title: str, body: Text) -> None:
        super().__init__()
        self._title, self._body = title, body

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(Text(self._title, style="bold"), id="modal-title")
            with VerticalScroll(id="modal-body"):
                yield Static(self._body)
            yield Static(Text("esc / q to close", style="dim"), id="modal-hint")


class StepsScreen(ModalScreen):
    """Pick the step to go back to."""
    BINDINGS = [Binding("escape", "dismiss", "close"), Binding("q", "dismiss", "close")]

    def __init__(self, steps) -> None:
        super().__init__()
        self._steps = steps

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(Text("Rewind to the start of which step?", style="bold"), id="modal-title")
            yield OptionList(*[f"step {s.step:>3}   {s.what}" + ("   (files changed)" if s.files_changed else "") for s in self._steps], id="steps")
            yield Static(Text("enter to rewind · esc to cancel", style="dim"), id="modal-hint")

    @on(OptionList.OptionSelected, "#steps")
    def chosen(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._steps[event.option_index].step)


CSS = """
Screen { background: $surface; }
#top { height: 1; background: $boost; color: $text; padding: 0 1; }
#main { height: 1fr; }
#conversation { padding: 0 2; }
#sidebar { width: 36; border-left: tall $primary 30%; padding: 0 1; display: none; }
#sidebar.show { display: block; }
.welcome { color: $accent; padding: 1 0 0 0; }
.tips { color: $text-muted; padding: 0 0 1 0; }
UserMessage { margin: 1 0 0 0; padding: 0 1; border-left: thick $primary; background: $boost; }
AssistantMessage { margin: 1 0 0 0; padding: 0 1; }
Note { color: $text-muted; margin: 1 0 0 0; padding: 0 1; }
ToolCard { margin: 0 0 0 2; padding: 0; border: none; background: transparent; }
ToolCard > Contents { padding: 0 0 0 2; }
ToolCard.pending > CollapsibleTitle { color: $text-muted; }
ToolCard.ok > CollapsibleTitle { color: $success; }
ToolCard.error > CollapsibleTitle { color: $error; }
CollapsibleTitle { padding: 0 1; background: transparent; }
#working { height: 1; padding: 0 2; color: $warning; display: none; }
#working.show { display: block; }
ApprovalPanel { height: auto; max-height: 24; margin: 0 1; padding: 0 1; border: round $warning; }
.approval-title { height: 1; }
.approval-body { height: auto; max-height: 18; overflow-y: auto; }
.approval-keys { height: 1; }
#slashmenu { display: none; height: auto; max-height: 12; margin: 0 1; border: round $primary 50%; }
#slashmenu.show { display: block; }
#inputbox { height: auto; min-height: 3; max-height: 10; margin: 0 1; border: round $primary 60%; padding: 0 1; }
#inputbox:focus-within { border: round $primary; }
PromptInput { height: auto; min-height: 1; max-height: 8; border: none; padding: 0; background: transparent; }
#hints { height: 1; padding: 0 2; color: $text-muted; }
TextScreen, StepsScreen { align: center middle; }
#modal { width: 90%; height: 85%; border: round $primary; background: $surface; padding: 0 1; }
#modal-title { height: 1; }
#modal-body { height: 1fr; }
#modal-hint { height: 1; }
#steps { height: 1fr; }
.side-title { text-style: bold; color: $accent; margin: 1 0 0 0; }
"""


class MiniAgentApp(App):
    CSS = CSS
    TITLE = "miniagent"
    BINDINGS = [Binding("escape", "interrupt", "interrupt", show=False), Binding("ctrl+c", "interrupt_or_quit", "interrupt", show=False, priority=True),
                Binding("ctrl+d", "quit_if_empty", "quit", show=False, priority=True), Binding("ctrl+q", "quit_app", "quit", show=False),
                Binding("ctrl+b", "toggle_sidebar", "sidebar", show=False), Binding("ctrl+l", "clear_screen", "redraw", show=False)]

    def __init__(self, ctl: SessionController, *, auto_approve: bool = False, workspace_label: str = "") -> None:
        super().__init__()
        self.ctl = ctl
        self.auto_approve = auto_approve
        self.workspace_label = workspace_label or str(ctl.workspace)
        self.busy = False
        self._turn_t0 = 0.0
        self._pending_approval: Future | None = None
        self._cards: dict[str, ToolCard] = {}
        self._md: AssistantMessage | None = None
        self._streamed_step = False
        self._history: list[str] = []
        self._hist_pos: int | None = None
        self._last_ctrl_c = 0.0
        self._buffer: list[str] = []
        self._turn_worker = None

    # ---------------------------------------------------------------- layout
    @property
    def _base(self):
        """The main screen: queries must not be answered by a modal (help, diff, steps) that happens to be on top."""
        return self.screen_stack[0]

    def _w(self, selector, expect_type=None):
        return self._base.query_one(selector, expect_type) if expect_type else self._base.query_one(selector)

    def _q(self, selector):
        return self._base.query(selector)

    def compose(self) -> ComposeResult:
        yield StatusBar(id="top")
        with Horizontal(id="main"):
            yield VerticalScroll(id="conversation")
            with VerticalScroll(id="sidebar"):
                yield Static(Text("Tasks", style="bold"), classes="side-title")
                yield Static("", id="todos")
                yield Static(Text("Changes", style="bold"), classes="side-title")
                yield Static("", id="changes")
        yield Static("", id="working")
        yield OptionList(id="slashmenu")
        with Vertical(id="inputbox"):
            yield PromptInput()
        yield Static("", id="hints")

    def on_mount(self) -> None:
        self.ctl.open(on_event=self._agent_event, on_text=self._agent_text, confirm=self._agent_confirm)
        conv = self._w("#conversation", VerticalScroll)
        conv.mount(Static(LOGO, classes="welcome"))
        conv.mount(Static("Type a task and press enter.  /help for commands · esc interrupts · ctrl+b side panel · ctrl+d quits.", classes="tips"))
        if self.ctl.state and self.ctl.state.get("messages"):
            self.call_after_refresh(self._render_history)
        self._w(PromptInput).focus()
        self.set_interval(0.12, self._tick)
        self._refresh_chrome()
        if self.size.width >= 130:
            self._w("#sidebar").add_class("show")

    async def _render_history(self) -> None:
        """A resumed session: show what was said before."""
        conv = self._w("#conversation", VerticalScroll)
        results = {m["tool_call_id"]: m["content"] for m in self.ctl.state["messages"] if m["role"] == "tool"}
        await conv.mount(Note("— resumed session —"))
        for m in self.ctl.state["messages"]:
            if m["role"] == "user":
                await conv.mount(UserMessage(m["content"] if isinstance(m["content"], str) else str(m["content"])))
            elif m["role"] == "assistant":
                if m.get("content"):
                    md = AssistantMessage()
                    await conv.mount(md)
                    await md.write(m["content"])
                    await md.finish()
                for c in m.get("tool_calls") or []:
                    import json as _json
                    try:
                        args = _json.loads(c["function"]["arguments"] or "{}")
                    except ValueError:
                        args = {}
                    card = ToolCard(c["function"]["name"], args, c["id"], None)
                    await conv.mount(card)
                    out = results.get(c["id"], "")
                    card.set_result({"output": out, "is_error": out.startswith("error:"), "duration_s": 0})
        conv.scroll_end(animate=False)

    # ---------------------------------------------------------------- chrome
    def _refresh_chrome(self) -> None:
        a = self.ctl.agent
        git = self.ctl.git
        parts = ["● miniagent", a.model, self.workspace_label]
        parts += [f"⎇ {git.branch}"] if git is not None else []
        parts += ["auto-approve"] if self.auto_approve else []
        self._w("#top", StatusBar).update(Text(" · ".join(parts), style="bold"))
        cost = self.ctl.cost_usd()
        toks = f"{fmt_tokens(self.ctl.tokens['in'])} in / {fmt_tokens(self.ctl.tokens['out'])} out" + (f" · ${cost:.3f}" if cost is not None else "")
        keys = "enter send · ctrl+j newline · / commands · esc interrupt · ctrl+b panel · ctrl+d quit"
        self._w("#hints", Static).update(Text(f"{keys}    {toks}", style="dim"))
        self._refresh_sidebar()

    def _refresh_sidebar(self) -> None:
        from ..features import render_todos
        todos = self.ctl.agent.todos
        empty = "(none; the agent fills this\nin with --features todo)"
        self._w("#todos", Static).update(Text(render_todos(todos) if todos else empty, style="" if todos else "dim"))
        files: dict[str, list[int]] = {}
        old = cur = None
        for line in self.ctl.diff_lines():
            if line.startswith("--- "):
                old = line[4:]
            elif line.startswith("+++ "):
                name = line[4:] if line[4:] != "/dev/null" else old
                cur = name[2:] if name and name[:2] in ("a/", "b/") else name
                files.setdefault(cur, [0, 0])
            elif line.startswith("+") and cur:
                files[cur][0] += 1
            elif line.startswith("-") and cur:
                files[cur][1] += 1
        text = Text()
        for f, (add, rem) in sorted(files.items()):
            text.append(f"{f}  ", style="")
            text.append(f"+{add} ", style="green")
            text.append(f"-{rem}\n", style="red")
        self._w("#changes", Static).update(text if files else Text("(no changes yet)", style="dim"))

    def _alive(self) -> bool:
        """False once the app is shutting down: late callbacks from the worker thread or timers must not touch widgets that are gone."""
        return self.is_running and not self._exit

    def _tick(self) -> None:
        if not self._alive():
            return
        w = self._w("#working", Static)
        if self.busy:
            frame = SPINNER[int(time.time() * 10) % len(SPINNER)]
            state = "Interrupting…" if self.ctl.agent.cancel_event.is_set() else "Working…"
            w.update(Text(f"{frame} {state} {int(time.time() - self._turn_t0)}s · esc to interrupt", style="yellow"))
            w.add_class("show")
        else:
            w.remove_class("show")

    def action_toggle_sidebar(self) -> None:
        self._w("#sidebar").toggle_class("show")

    def action_clear_screen(self) -> None:
        self.refresh(layout=True)

    # ---------------------------------------------------------------- input
    @on(PromptInput.Submitted)
    async def submitted(self, event: PromptInput.Submitted) -> None:
        text = event.text.strip()
        box = self._w(PromptInput)
        menu = self._w("#slashmenu", OptionList)
        if menu.has_class("show") and text.startswith("/") and " " not in text and text not in COMMANDS and menu.highlighted is not None:
            self._complete()                                       # enter on a half-typed command completes it first
            return
        if not text:
            return
        if self.busy and not text.startswith("/"):
            self.notify("The agent is still working: wait, or press esc to interrupt it.", severity="warning")
            return
        self._history.append(text)
        self._hist_pos = None
        box.clear()
        self._hide_menu()
        if text.startswith("/"):
            await self._command(text)
        else:
            await self._start_turn(text)

    @on(PromptInput.History)
    def history(self, event: PromptInput.History) -> None:
        if not self._history:
            return
        pos = len(self._history) if self._hist_pos is None else self._hist_pos
        pos = max(0, min(len(self._history), pos + event.direction))
        self._hist_pos = None if pos == len(self._history) else pos
        box = self._w(PromptInput)
        box.load_text("" if self._hist_pos is None else self._history[self._hist_pos])
        box.move_cursor(box.document.end)

    @on(TextArea.Changed)
    def text_changed(self, event: TextArea.Changed) -> None:
        if not self._alive():                                      # a change event can arrive while the app is shutting down
            return
        text = event.text_area.text
        if text.startswith("/") and "\n" not in text and " " not in text:
            matches = [c for c in COMMANDS if c.startswith(text.lower())]
            menu = self._w("#slashmenu", OptionList)
            if matches:
                menu.clear_options()
                menu.add_options([f"{c:<9} {COMMANDS[c]}" for c in matches])
                menu.highlighted = 0
                menu.add_class("show")
                return
        self._hide_menu()

    def _hide_menu(self) -> None:
        self._w("#slashmenu").remove_class("show")

    @on(PromptInput.Complete)
    def complete_requested(self) -> None:
        self._complete()

    def _complete(self) -> None:
        menu = self._w("#slashmenu", OptionList)
        if menu.has_class("show") and menu.highlighted is not None:
            label = str(menu.get_option_at_index(menu.highlighted).prompt)
            box = self._w(PromptInput)
            box.load_text(label.split()[0] + " ")
            box.move_cursor(box.document.end)
            self._hide_menu()

    @on(OptionList.OptionSelected, "#slashmenu")
    def menu_chosen(self, event: OptionList.OptionSelected) -> None:
        self._complete()
        self._w(PromptInput).focus()

    # ---------------------------------------------------------------- a turn (worker thread)
    async def _start_turn(self, text: str) -> None:
        conv = self._w("#conversation", VerticalScroll)
        await conv.mount(UserMessage(text))
        conv.scroll_end(animate=False)
        self.busy = True
        self._turn_t0 = time.time()
        self._md, self._streamed_step = None, False
        self._turn_worker = self.run_worker(lambda: self._turn_thread(text), thread=True, exit_on_error=False, name="turn")
        self._refresh_chrome()

    def _turn_thread(self, text: str) -> None:
        try:
            res = self.ctl.run_turn(text)
        except Exception as e:                                    # an unexpected bug must not take the whole UI down
            self.call_from_thread(self._turn_failed, repr(e))
            return
        self.call_from_thread(self._turn_finished, res)

    async def _turn_finished(self, res) -> None:
        if not self._alive():
            return
        await self._finish_md()
        self._flush_buffer_sync()
        self.busy = False
        conv = self._w("#conversation", VerticalScroll)
        colour = {"done": "green", "interrupted": "yellow"}.get(res.status, "red")
        note = Note(Text(f"{res.status} · {res.steps} steps · {fmt_tokens(res.input_tokens)} in / {fmt_tokens(res.output_tokens)} out", style=colour))
        await conv.mount(note)
        if res.status == "error" and res.error:
            await conv.mount(Note(Text(res.error[:500], style="red")))
        conv.scroll_end(animate=False)
        self._refresh_chrome()
        self._w(PromptInput).focus()

    async def _turn_failed(self, err: str) -> None:
        self.busy = False
        if not self._alive():
            return
        await self._w("#conversation", VerticalScroll).mount(Note(Text(f"internal error: {err}", style="red")))
        self._refresh_chrome()

    # callbacks from the agent (worker thread) -> UI thread
    def _agent_text(self, piece: str) -> None:
        self._buffer.append(piece)
        if sum(len(p) for p in self._buffer) >= 24 or piece.endswith("\n"):
            self._flush_buffer_threaded()

    def _flush_buffer_threaded(self) -> None:
        text, self._buffer = "".join(self._buffer), []
        if text:
            self.call_from_thread(self._append_text, text)

    def _flush_buffer_sync(self) -> None:
        pass

    async def _append_text(self, text: str) -> None:
        if not self._alive():
            return
        conv = self._w("#conversation", VerticalScroll)
        at_bottom = conv.scroll_y >= conv.max_scroll_y - 2
        if self._md is None:
            self._md = AssistantMessage()
            await conv.mount(self._md)
        self._streamed_step = True
        await self._md.write(text)
        if at_bottom:
            conv.scroll_end(animate=False)

    async def _finish_md(self) -> None:
        if self._md is not None:
            await self._md.finish()
            self._md = None

    def _agent_event(self, kind: str, d: dict) -> None:
        if kind == "model_call" and self._buffer:
            self._flush_buffer_threaded()
        self.call_from_thread(self._handle_event, kind, d)

    async def _handle_event(self, kind: str, d: dict) -> None:
        if not self._alive():
            return
        conv = self._w("#conversation", VerticalScroll)
        at_bottom = conv.scroll_y >= conv.max_scroll_y - 2
        if kind == "model_call":
            await self._finish_md()
            if d.get("text") and not self._streamed_step:                 # a client that does not stream: show the text now
                md = AssistantMessage()
                await conv.mount(md)
                await md.write(d["text"])
                await md.finish()
            self._streamed_step = False
            for c in d.get("tool_calls", []):
                preview = describe_action(c["name"], c["input"], self.ctl.workspace)[1:] if c["name"] in FILE_TOOLS else None
                card = ToolCard(c["name"], c["input"], c["id"], preview)
                self._cards[c["id"]] = card
                await conv.mount(card)
        elif kind == "tool_result":
            card = self._cards.get(d["tool_call_id"])
            if card is not None:
                card.set_result(d)
            if d["name"] == "todo_write":
                self._refresh_sidebar()
        elif kind == "stuck":
            await conv.mount(Note(Text(f"stuck detector: {d['kind']} ×{d['count']} → {d['action']}", style="yellow")))
        elif kind == "budget":
            await conv.mount(Note(Text(f"budget: {d['reason']}", style="yellow")))
        elif kind == "git_commit" and d.get("sha"):
            await conv.mount(Note(Text(f"git {d['sha'][:8]}  {', '.join(d['files'][:4])}", style="dim")))
        elif kind == "compaction" and d.get("kind") == "summary":
            await conv.mount(Note(Text("history summarised to save context", style="dim")))
        if at_bottom:
            conv.scroll_end(animate=False)

    # ---------------------------------------------------------------- approvals
    def _agent_confirm(self, name: str, args: dict) -> bool:
        if self.auto_approve:
            return True
        fut: Future = Future()
        self.call_from_thread(self._ask, name, args, fut)
        return fut.result()

    async def _ask(self, name: str, args: dict, fut: Future) -> None:
        if not self._alive():
            fut.set_result(False)
            return
        self._pending_approval = fut
        lines = describe_action(name, args, self.ctl.workspace)
        panel = ApprovalPanel(name, lines)
        self._w(PromptInput).disabled = True
        await self._base.mount(panel, before=self._w("#slashmenu"))
        self._w("#conversation", VerticalScroll).scroll_end(animate=False)

    @on(ApprovalPanel.Answered)
    async def answered(self, event: ApprovalPanel.Answered) -> None:
        fut, self._pending_approval = self._pending_approval, None
        for panel in self._q(ApprovalPanel):
            await panel.remove()
        box = self._w(PromptInput)
        box.disabled = False
        box.focus()
        if event.answer == "all":
            self.auto_approve = True
            self._refresh_chrome()
        if fut is not None and not fut.done():
            fut.set_result(event.answer in ("yes", "all"))

    # ---------------------------------------------------------------- interrupt / quit
    def action_interrupt(self) -> None:
        if self.busy:
            if self._pending_approval is not None and not self._pending_approval.done():
                self._w(ApprovalPanel).action_answer("no")
            self.ctl.cancel()

    def action_interrupt_or_quit(self) -> None:
        if self.busy:
            self.action_interrupt()
            return
        now = time.time()
        if now - self._last_ctrl_c < 2.0:
            self.action_quit_app()
        else:
            self._last_ctrl_c = now
            self.notify("Press ctrl+c again to quit.", timeout=2)

    def action_quit_if_empty(self) -> None:
        box = self._w(PromptInput)
        if not box.text:
            self.action_quit_app()
        else:
            box.action_delete_right()                              # with text in the box ctrl+d keeps its editing meaning

    def action_quit_app(self) -> None:
        if self.busy:
            self.ctl.cancel()
            if self._pending_approval is not None and not self._pending_approval.done():
                self._pending_approval.set_result(False)
        self.exit()

    def on_unmount(self) -> None:
        self.ctl.close()

    # ---------------------------------------------------------------- slash commands
    async def _command(self, line: str) -> None:
        name, _, arg = line.partition(" ")
        arg = arg.strip()
        conv = self._w("#conversation", VerticalScroll)
        if name in ("/exit", "/quit"):
            self.action_quit_app()
        elif name in ("/help", "/?"):
            self.push_screen(TextScreen("miniagent: keys and commands", self._help_text()))
        elif name == "/diff":
            lines = self.ctl.diff_lines()
            body = diff_text(lines) if lines else Text("no changes yet", style="dim")
            self.push_screen(TextScreen("Changes since the session started", body))
        elif name == "/steps":
            steps = self.ctl.steps()
            if not steps:
                self.notify("Nothing to rewind to yet.")
            else:
                self.push_screen(StepsScreen(steps), self._rewind_chosen)
        elif name == "/rewind":
            if not arg.isdigit():
                self.notify("usage: /rewind N  (or /steps to pick)", severity="warning")
            else:
                await self._do(lambda: self.ctl.rewind(int(arg)))
        elif name == "/undo":
            await self._do(self.ctl.undo)
        elif name == "/cost":
            await conv.mount(Note(self.ctl.cost_text()))
        elif name == "/status":
            await conv.mount(Note("\n".join(self.ctl.status_lines(["auto-approve"] if self.auto_approve else []))))
        elif name == "/todo":
            self.action_toggle_sidebar()
        elif name == "/model":
            if arg:
                self.ctl.set_model(arg)
                self._refresh_chrome()
                await conv.mount(Note(f"model for the next turns: {arg}"))
            else:
                await conv.mount(Note(f"model: {self.ctl.agent.model}"))
        elif name == "/yes":
            if arg.lower() in ("on", "off"):
                self.auto_approve = arg.lower() == "on"
                self._refresh_chrome()
                await conv.mount(Note(f"auto-approve: {arg.lower()}"))
            else:
                self.notify("usage: /yes on|off", severity="warning")
        elif name == "/clear":
            if self.busy:
                self.notify("Wait for the current turn (or interrupt it) first.", severity="warning")
                return
            msg = self.ctl.clear()
            await conv.remove_children()
            self._cards.clear()
            await conv.mount(Note(msg))
            self._refresh_chrome()
        else:
            self.notify(f"unknown command {name}: /help lists them", severity="error")
        conv.scroll_end(animate=False)

    async def _rewind_chosen(self, step) -> None:
        if step is not None:
            await self._do(lambda: self.ctl.rewind(step))

    async def _do(self, action) -> None:
        conv = self._w("#conversation", VerticalScroll)
        if self.busy:
            self.notify("Wait for the current turn (or interrupt it) first.", severity="warning")
            return
        try:
            msg = action()
        except ControllerError as e:
            self.notify(str(e), severity="error")
            return
        await conv.mount(Note(Text("↩ " + msg, style="yellow")))
        self._refresh_chrome()

    def _help_text(self) -> Text:
        t = Text()
        t.append("Keys\n", style="bold")
        for k, v in (("enter", "send the message"), ("ctrl+j / alt+enter", "new line in the message"), ("up / down", "earlier messages"),
                     ("tab", "complete a / command"), ("esc / ctrl+c", "interrupt the agent (ctrl+c twice when idle quits)"),
                     ("ctrl+b", "show or hide the side panel (tasks, changed files)"), ("ctrl+d", "quit (when the box is empty)"),
                     ("y / a / n", "answer an approval: allow, allow all, deny")):
            t.append(f"  {k:<20}", style="cyan")
            t.append(v + "\n")
        t.append("\nCommands\n", style="bold")
        for c, d in COMMANDS.items():
            t.append(f"  {c:<10}", style="cyan")
            t.append(d + "\n")
        return t
