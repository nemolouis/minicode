"""The full-screen terminal UI, driven headlessly with Textual's pilot (scripted model, real workspace, real session files)."""
import asyncio
import json
import threading

import pytest

pytest.importorskip("textual")

from rich.text import Text  # noqa: E402
from textual.widgets import OptionList  # noqa: E402

from minicode.agent import Agent  # noqa: E402
from minicode.client import ModelResponse, ToolCall  # noqa: E402
from minicode.controller import SessionController  # noqa: E402
from minicode.tools import Workspace  # noqa: E402
from minicode.tui.app import (COMMANDS, ApprovalPanel, AssistantMessage, MiniCodeApp, Note, PromptInput, StepsScreen, TextScreen,  # noqa: E402
                               ToolCard, UserMessage)


def call(cid, name, **inp): return ToolCall(cid, name, json.dumps(inp))


class Script:
    """Scripted model. Items: (content, tokens), or a callable() -> (content, tokens) that may block (to simulate a slow request)."""
    def __init__(self, script, stream=True):
        self.script, self.calls, self.stream = list(script), [], stream

    def chat(self, on_text=None, **kw):
        import copy
        self.calls.append(copy.deepcopy(kw))
        item = self.script.pop(0)
        content, tok = item() if callable(item) else item
        text = "".join(c for c in content if isinstance(c, str))
        if on_text and self.stream and text:
            for i in range(0, len(text), 5):
                on_text(text[i:i + 5])
        calls = [c for c in content if isinstance(c, ToolCall)]
        return ModelResponse(text, calls, "tool_calls" if calls else "stop", {"input_tokens": tok, "output_tokens": 7})


@pytest.fixture
def env(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    (w / "a.txt").write_text("one\ntwo\n")
    return tmp_path, w


def make_ctl(env, script, *, stream=True, resume=None, price=None, run_dir=None, features=(), **kw):
    tmp, w = env
    client = Script(script, stream)

    def make_agent(trajectory=None, on_text=None, confirm=None):
        return Agent(client, "fake-model", Workspace(w), trajectory=trajectory, on_text=on_text, confirm=confirm, session=True,
                     features=features, keep_background=True, **kw)

    return SessionController(make_agent=make_agent, run_dir=run_dir or (tmp / "runs" / "r1"), workspace=w,
                             new_run_dir=lambda: tmp / "runs" / "r-new", resume=resume, price=price), client


class Snap:
    """What the screen showed at the end of a test (widgets are gone once the headless app has shut down, so it is captured before)."""
    def __init__(self, app):
        self.users = [w.content if isinstance(w.content, str) else str(w.content) for w in app.query(UserMessage)]
        self.assistants = [w.source for w in app.query(AssistantMessage)]
        self.notes = [w.content.plain if isinstance(w.content, Text) else str(w.content) for w in app.query(Note)]
        self.cards = [dict(name=c.tool_name, title=c.title, error=c.has_class("error"), ok=c.has_class("ok"), collapsed=c.collapsed,
                           body=c.body.content) for c in app.query(ToolCard)]
        self.hints = app.query_one("#hints").content.plain
        self.top = app.query_one("#top").content.plain
        self.panels = len(app.query(ApprovalPanel))
        self.input = app.query_one(PromptInput).text

    def card(self, name):
        return next(c for c in self.cards if c["name"] == name)


def run(coro_fn, env, script, *, size=(120, 40), auto=False, **kw):
    """Run `coro_fn(app, pilot)` inside the headless app and return (snapshot, controller, client)."""
    ctl, client = make_ctl(env, script, **kw)
    app = MiniCodeApp(ctl, auto_approve=auto)
    box = {}

    async def main():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await coro_fn(app, pilot)
            await pilot.pause()
            box["snap"] = Snap(app)

    asyncio.run(main())
    return box["snap"], ctl, client


async def idle(app, pilot, seconds=10.0):
    waited = 0.0
    while app.busy and waited < seconds:
        await pilot.pause(0.05)
        waited += 0.05
    assert not app.busy, "the turn did not finish"
    await pilot.pause(0.05)


async def say(app, pilot, text):
    box = app.query_one(PromptInput)
    box.load_text(text)
    await pilot.press("enter")
    await pilot.pause(0.05)


# ---------------------------------------------------------------- start up
def test_startup_shows_the_banner_status_and_an_empty_focused_input(env):
    seen = {}

    async def go(app, pilot):
        seen["status"] = str(app.query_one("#top").content)
        seen["hints"] = str(app.query_one("#hints").content)
        seen["focused"] = isinstance(app.focused, PromptInput)
        seen["tips"] = [str(w.content) for w in app.query(".tips")]

    run(go, env, [])
    assert "minicode" in seen["status"] and "fake-model" in seen["status"] and str(env[1]) in seen["status"]
    assert "enter send" in seen["hints"] and "esc interrupt" in seen["hints"] and seen["focused"] and "/help" in seen["tips"][0]


# ---------------------------------------------------------------- a turn
def test_a_message_gets_a_streamed_markdown_answer(env):
    async def go(app, pilot):
        await say(app, pilot, "hello agent")
        await idle(app, pilot)

    snap, ctl, _ = run(go, env, [(["Here is **bold** text and `code`."], 120)])
    assert snap.users == ["hello agent"]
    assert snap.assistants == ["Here is **bold** text and `code`."]                    # all the streamed pieces, in order
    assert any("done · 1 steps" in n for n in snap.notes)
    assert snap.input == "" and ctl.turns == 1


def test_an_empty_or_blank_message_does_nothing(env):
    async def go(app, pilot):
        await say(app, pilot, "   ")

    snap, ctl, client = run(go, env, [])
    assert ctl.turns == 0 and client.calls == [] and not snap.users


def test_a_client_that_does_not_stream_still_shows_the_text_once(env):
    async def go(app, pilot):
        await say(app, pilot, "hi")
        await idle(app, pilot)

    snap, _, _ = run(go, env, [(["a whole reply at once"], 10)], stream=False)
    assert snap.assistants == ["a whole reply at once"]


def test_tokens_and_cost_appear_in_the_footer(env):
    async def go(app, pilot):
        await say(app, pilot, "hi")
        await idle(app, pilot)

    snap, _, _ = run(go, env, [(["x"], 2_000_000)], price={"input": 1.0, "output": 2.0})
    assert "2000.0k" in snap.hints and "$2.000" in snap.hints


def test_tool_cards_show_progress_results_and_errors(env):
    script = [(["Looking. ", call("1", "read_file", path="a.txt"), call("2", "grep", pattern="two"), call("3", "read_file", path="nope.txt"),
                call("4", "bash", command="echo hi")], 100), (["done"], 100)]

    async def go(app, pilot):
        await say(app, pilot, "go")
        await asyncio.sleep(0)                                   # let the first approval (bash) appear, then allow it
        for _ in range(100):
            if app.query(ApprovalPanel):
                await pilot.press("y")
                break
            await pilot.pause(0.05)
        await idle(app, pilot)

    snap, _, _ = run(go, env, script)
    titles = {c["name"]: c["title"] for c in snap.cards}
    assert titles["grep"].startswith("✓ grep  two") and titles["bash"].startswith("✓ bash  echo hi")
    errs = [c for c in snap.cards if c["error"]]
    assert len(errs) == 1 and errs[0]["title"].startswith("✗ read_file  nope.txt") and not errs[0]["collapsed"]
    assert "not a file" in errs[0]["body"].plain
    assert any("∥3" in c["title"] for c in snap.cards)                                                # the three adjacent read-only calls ran together


def test_file_edits_show_their_diff_inside_the_card(env):
    tmp, w = env
    script = [([call("1", "edit_file", path="a.txt", old_string="two", new_string="TWO")], 10), (["ok"], 10)]

    async def go(app, pilot):
        await say(app, pilot, "edit")
        for _ in range(100):
            if app.query(ApprovalPanel):
                await pilot.press("y")
                break
            await pilot.pause(0.05)
        await idle(app, pilot)

    snap, _, _ = run(go, env, script)
    c = snap.card("edit_file")
    assert isinstance(c["body"], Text) and "-two" in c["body"].plain and "+TWO" in c["body"].plain and not c["collapsed"]
    assert (w / "a.txt").read_text() == "one\nTWO\n"


# ---------------------------------------------------------------- approvals
def approval_script():
    return [([call("1", "write_file", path="b.txt", content="B\n")], 10), (["ok"], 10)]


async def wait_for_panel(app, pilot):
    for _ in range(200):
        if app.query(ApprovalPanel):
            return app.query_one(ApprovalPanel)
        await pilot.pause(0.05)
    raise AssertionError("no approval panel appeared")


@pytest.mark.parametrize("key,expect_file", [("y", True), ("enter", True), ("n", False), ("escape", False), ("a", True)])
def test_approval_keys(env, key, expect_file):
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "write it")
        panel = await wait_for_panel(app, pilot)
        seen["title"] = str(panel.query(".approval-title").first().content)
        seen["body"] = str(panel.query(".approval-body").first().content.plain)
        seen["input_disabled"] = app.query_one(PromptInput).disabled
        await pilot.press(key)
        await idle(app, pilot)
        seen["panel_gone"] = not app.query(ApprovalPanel)
        seen["auto"] = app.auto_approve

    run(go, env, approval_script())
    assert (env[1] / "b.txt").exists() is expect_file
    assert "Allow write_file" in seen["title"] and "+B" in seen["body"] and seen["input_disabled"] and seen["panel_gone"]
    assert seen["auto"] is (key == "a")


def test_allow_all_stops_asking(env):
    script = [([call("1", "write_file", path="b.txt", content="B\n")], 10), ([call("2", "write_file", path="c.txt", content="C\n")], 10), (["ok"], 10)]
    asked = []

    async def go(app, pilot):
        await say(app, pilot, "write two")
        await wait_for_panel(app, pilot)
        asked.append(1)
        await pilot.press("a")
        await idle(app, pilot)

    run(go, env, script)
    assert (env[1] / "b.txt").exists() and (env[1] / "c.txt").exists() and len(asked) == 1


def test_a_denied_action_is_reported_to_the_model(env):
    async def go(app, pilot):
        await say(app, pilot, "write it")
        await wait_for_panel(app, pilot)
        await pilot.press("n")
        await idle(app, pilot)

    _, _, client = run(go, env, approval_script())
    assert "user denied" in [m["content"] for m in client.calls[1]["messages"] if m["role"] == "tool"][0]


def test_auto_approve_never_shows_the_panel(env):
    async def go(app, pilot):
        await say(app, pilot, "write it")
        await idle(app, pilot)
        assert not app.query(ApprovalPanel)

    run(go, env, approval_script(), auto=True)
    assert (env[1] / "b.txt").exists()


# ---------------------------------------------------------------- interrupting
def test_escape_interrupts_a_running_turn_and_the_conversation_survives(env):
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return (["too late"], 10)

    async def go(app, pilot):
        await say(app, pilot, "do something slow")
        for _ in range(50):
            if app.busy and app.ctl.agent is not None:
                break
            await pilot.pause(0.02)
        await pilot.pause(0.2)
        working = app.query_one("#working")
        assert working.has_class("show")
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert "Interrupting" in str(working.content)
        gate.set()
        await idle(app, pilot)
        await say(app, pilot, "try again")
        await idle(app, pilot)

    snap, ctl, client = run(go, env, [slow, (["recovered"], 10)])
    assert any("interrupted" in n for n in snap.notes) and any("done" in n for n in snap.notes)
    assert [m["content"] for m in client.calls[-1]["messages"] if m["role"] == "user"] == ["do something slow", "try again"]


def test_interrupt_while_waiting_for_approval_denies_it(env):
    async def go(app, pilot):
        await say(app, pilot, "write it")
        await wait_for_panel(app, pilot)
        await pilot.press("ctrl+c")
        await idle(app, pilot)

    snap, ctl, _ = run(go, env, approval_script() + [(["unused"], 1)])
    assert not (env[1] / "b.txt").exists() and snap.panels == 0


def test_sending_while_busy_is_refused_not_queued(env):
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return (["done"], 10)

    async def go(app, pilot):
        await say(app, pilot, "first")
        await pilot.pause(0.3)
        await say(app, pilot, "second")
        assert len(app.query(UserMessage)) == 1
        gate.set()
        await idle(app, pilot)

    _, ctl, client = run(go, env, [slow])
    assert ctl.turns == 1 and len(client.calls) == 1


# ---------------------------------------------------------------- the input box
def test_slash_menu_filters_and_tab_completes(env):
    seen = {}

    async def go(app, pilot):
        box = app.query_one(PromptInput)
        box.load_text("/di")
        await pilot.pause(0.1)
        menu = app.query_one("#slashmenu", OptionList)
        seen["shown"] = menu.has_class("show")
        seen["options"] = [str(menu.get_option_at_index(i).prompt) for i in range(menu.option_count)]
        await pilot.press("tab")
        await pilot.pause(0.05)
        seen["text"] = box.text
        seen["hidden"] = not menu.has_class("show")
        box.load_text("not a command")
        await pilot.pause(0.05)
        seen["hidden_for_text"] = not menu.has_class("show")

    run(go, env, [])
    assert seen["shown"] and len(seen["options"]) == 1 and seen["options"][0].startswith("/diff") and seen["text"] == "/diff " and seen["hidden"]
    assert seen["hidden_for_text"]


def test_enter_on_a_half_typed_command_completes_instead_of_sending(env):
    async def go(app, pilot):
        box = app.query_one(PromptInput)
        box.load_text("/he")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert box.text == "/help " and not isinstance(app.screen, TextScreen)

    run(go, env, [])


def test_ctrl_j_inserts_a_newline_and_enter_sends_the_whole_text(env):
    async def go(app, pilot):
        box = app.query_one(PromptInput)
        box.load_text("line one")
        box.move_cursor(box.document.end)
        await pilot.press("ctrl+j")
        await pilot.press("l", "i", "n", "e", " ", "t", "w", "o")
        await pilot.press("enter")
        await idle(app, pilot)

    snap, _, client = run(go, env, [(["ok"], 10)])
    assert client.calls[0]["messages"][-1]["content"] == "line one\nline two"
    assert snap.users == ["line one\nline two"]


def test_up_arrow_recalls_earlier_messages(env):
    seen = []

    async def go(app, pilot):
        await say(app, pilot, "first message")
        await idle(app, pilot)
        await say(app, pilot, "second message")
        await idle(app, pilot)
        box = app.query_one(PromptInput)
        await pilot.press("up")
        seen.append(box.text)
        await pilot.press("up")
        seen.append(box.text)
        await pilot.press("down")
        seen.append(box.text)
        await pilot.press("down")
        seen.append(box.text)

    run(go, env, [(["a"], 1), (["b"], 1)])
    assert seen == ["second message", "first message", "second message", ""]


# ---------------------------------------------------------------- slash commands
def test_help_opens_a_screen_and_escape_closes_it(env):
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "/help")
        await pilot.pause(0.1)
        seen["open"] = isinstance(app.screen, TextScreen)
        seen["text"] = app.screen.query_one("#modal-body Static").content.plain
        await pilot.press("escape")
        await pilot.pause(0.1)
        seen["closed"] = not isinstance(app.screen, TextScreen)

    run(go, env, [])
    assert seen["open"] and seen["closed"] and "ctrl+j" in seen["text"] and all(c in seen["text"] for c in COMMANDS)


def test_diff_screen_shows_what_changed_since_the_session_started(env):
    script = [([call("1", "write_file", path="a.txt", content="one\nTWO\n")], 10), (["ok"], 10)]
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "/diff")
        await pilot.pause(0.1)
        seen["empty"] = app.screen.query_one("#modal-body Static").content.plain
        await pilot.press("q")
        await pilot.pause(0.1)
        await say(app, pilot, "change it")
        await idle(app, pilot)
        await say(app, pilot, "/diff")
        await pilot.pause(0.1)
        seen["diff"] = app.screen.query_one("#modal-body Static").content.plain

    run(go, env, script, auto=True)
    assert "no changes yet" in seen["empty"] and "--- a/a.txt" in seen["diff"] and "-two" in seen["diff"] and "+TWO" in seen["diff"]


def test_undo_restores_files_and_the_conversation(env):
    script = [([call("1", "write_file", path="a.txt", content="CHANGED\n")], 10), (["done"], 10), (["after undo"], 10)]

    async def go(app, pilot):
        await say(app, pilot, "change a")
        await idle(app, pilot)
        assert (env[1] / "a.txt").read_text() == "CHANGED\n"
        await say(app, pilot, "/undo")
        await pilot.pause(0.2)
        assert (env[1] / "a.txt").read_text() == "one\ntwo\n"
        await say(app, pilot, "something else")
        await idle(app, pilot)

    snap, ctl, client = run(go, env, script, auto=True)
    assert [m["content"] for m in client.calls[-1]["messages"] if m["role"] == "user"] == ["something else"]
    assert any("rewound to the start of step 1" in n for n in snap.notes)


def test_steps_picker_rewinds_to_the_selected_step(env):
    script = [([call("1", "write_file", path="a.txt", content="X\n")], 10), ([call("2", "write_file", path="b.txt", content="Y\n")], 10), (["done"], 10)]
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "two edits")
        await idle(app, pilot)
        await say(app, pilot, "/steps")
        await pilot.pause(0.1)
        seen["is_steps"] = isinstance(app.screen, StepsScreen)
        steps = app.screen.query_one("#steps", OptionList)
        seen["options"] = [str(steps.get_option_at_index(i).prompt) for i in range(steps.option_count)]
        steps.highlighted = 1                                           # step 2
        await pilot.press("enter")
        await pilot.pause(0.3)

    run(go, env, script, auto=True)
    assert seen["is_steps"] and "write_file" in seen["options"][0] and "(files changed)" in seen["options"][0] and "final answer" in seen["options"][2]
    assert (env[1] / "a.txt").read_text() == "X\n" and not (env[1] / "b.txt").exists()                 # step 2's edit is undone, step 1's kept


def test_other_commands(env):
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "hi")
        await idle(app, pilot)
        await say(app, pilot, "/cost")
        await say(app, pilot, "/status")
        await say(app, pilot, "/model other-model")
        seen["model"] = app.ctl.agent.model
        seen["top"] = str(app.query_one("#top").content)
        await say(app, pilot, "/yes on")
        seen["yes"] = app.auto_approve
        await say(app, pilot, "/todo")
        seen["sidebar"] = app.query_one("#sidebar").has_class("show")
        await say(app, pilot, "/nonsense")
        await say(app, pilot, "/rewind x")

    snap, _, _ = run(go, env, [(["x"], 5)])
    notes = "\n".join(snap.notes)
    assert "input +" in notes and "model fake-model" in notes and "model for the next turns: other-model" in notes and "auto-approve: on" in notes
    assert seen["model"] == "other-model" and "other-model" in seen["top"] and seen["yes"] and seen["sidebar"]


def test_clear_starts_a_new_conversation_in_a_new_session_dir(env):
    async def go(app, pilot):
        await say(app, pilot, "first")
        await idle(app, pilot)
        await say(app, pilot, "/clear")
        await pilot.pause(0.3)
        await say(app, pilot, "second")
        await idle(app, pilot)

    snap, ctl, client = run(go, env, [(["one"], 1), (["two"], 1)])
    assert snap.users == ["second"] and ctl.run_dir.name == "r-new"
    assert [m["content"] for m in client.calls[1]["messages"] if m["role"] == "user"] == ["second"]


def test_exit_command_and_ctrl_d_and_double_ctrl_c_quit(env):
    for how in ("/exit", "ctrl+d", "ctrl+c ctrl+c", "ctrl+q"):
        ctl, _ = make_ctl(env, [])
        app = MiniCodeApp(ctl)
        result = {}

        async def main(app=app, how=how, result=result):
            async with app.run_test(size=(100, 30)) as pilot:
                await pilot.pause()
                if how.startswith("/"):
                    await say(app, pilot, how)
                else:
                    for k in how.split():
                        await pilot.press(k)
                await pilot.pause(0.2)
                result["exited"] = app._exit

        asyncio.run(main())
        assert result["exited"], how


def test_ctrl_d_does_not_quit_when_there_is_text(env):
    async def go(app, pilot):
        app.query_one(PromptInput).load_text("draft")
        await pilot.press("ctrl+d")
        await pilot.pause(0.1)
        assert not app._exit

    run(go, env, [])


# ---------------------------------------------------------------- side panel
def test_sidebar_lists_changed_files_and_todos(env):
    script = [([call("1", "todo_write", todos=[{"content": "fix a", "status": "in_progress"}]),
                call("2", "write_file", path="a.txt", content="one\nTWO\nthree\n"), call("3", "write_file", path="n.txt", content="x\n")], 10), (["ok"], 10)]
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "work")
        await idle(app, pilot)
        seen["todos"] = app.query_one("#todos").content.plain
        seen["changes"] = app.query_one("#changes").content.plain
        await pilot.press("ctrl+b")
        seen["shown"] = app.query_one("#sidebar").has_class("show")

    run(go, env, script, auto=True, features=("todo",), size=(100, 30))
    assert "[~] fix a" in seen["todos"] and "a.txt  +2 -1" in seen["changes"] and "n.txt  +1 -0" in seen["changes"] and seen["shown"]


def test_wide_terminals_open_the_sidebar_by_default(env):
    seen = {}

    async def go(app, pilot):
        seen["wide"] = app.query_one("#sidebar").has_class("show")

    run(go, env, [], size=(150, 40))
    assert seen["wide"]


# ---------------------------------------------------------------- resume
def test_resumed_session_shows_the_earlier_conversation(env):
    tmp, w = env
    script = [([call("1", "read_file", path="a.txt")], 10), (["It says one and two."], 10)]
    ctl1, _ = make_ctl(env, script)
    app1 = MiniCodeApp(ctl1, auto_approve=True)

    async def first():
        async with app1.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await say(app1, pilot, "read a.txt")
            await idle(app1, pilot)

    asyncio.run(first())
    from minicode.session import load_session
    state = load_session(ctl1.run_dir)
    seen = {}

    async def go(app, pilot):
        await pilot.pause(0.3)
        seen["users"] = [w.content for w in app.query(UserMessage)]
        seen["assistant"] = [w.source for w in app.query(AssistantMessage)]
        seen["tools"] = [c.title for c in app.query(ToolCard)]
        seen["notes"] = [str(n.content) for n in app.query(Note)]
        await say(app, pilot, "and again?")
        await idle(app, pilot)

    _, ctl2, client = run(go, env, [(["second answer"], 5)], resume=state, run_dir=ctl1.run_dir)
    assert seen["users"] == ["read a.txt"] and seen["assistant"] == ["It says one and two."] and seen["tools"][0].startswith("✓ read_file  a.txt")
    assert "— resumed session —" in seen["notes"]
    assert [m["content"] for m in client.calls[0]["messages"] if m["role"] == "user"] == ["read a.txt", "and again?"]


# ---------------------------------------------------------------- the whole screen
def test_screenshot_contains_the_visible_pieces(env):
    seen = {}

    async def go(app, pilot):
        await say(app, pilot, "show me")
        await idle(app, pilot)
        seen["svg"] = app.export_screenshot()

    run(go, env, [(["Everything is fine."], 10)], size=(110, 30))
    svg = seen["svg"]
    for piece in ("minicode", "show me", "Everything", "fake-model", "enter send"):
        assert piece in svg.replace("&#160;", " ").replace("&nbsp;", " ") or piece.replace(" ", "&#160;") in svg


# ---------------------------------------------------------------- the command line
def test_cli_launches_the_tui_with_a_controller(env, monkeypatch, capsys):
    import minicode.tui as tui
    from minicode import cli
    tmp, w = env
    got = {}

    def fake_run(ctl, auto_approve=False, **kw):
        got["ctl"], got["auto"] = ctl, auto_approve
        ctl.turns = 1                                                         # pretend a turn happened, to see the exit summary

    monkeypatch.setattr(tui, "run", fake_run)
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([]))
    assert cli.main(["-w", str(w), "-m", "fake", "-y", "--ui", "tui", "--runs-dir", str(tmp / "runs")]) == 0
    assert isinstance(got["ctl"], SessionController) and got["auto"] is True
    out = capsys.readouterr().out
    assert "session: " in out and "--resume" in out


def test_cli_auto_picks_the_line_repl_when_not_a_terminal(env, monkeypatch, capsys):
    import io

    from minicode import cli
    tmp, w = env
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([(["plain repl answer"], 5)]))
    monkeypatch.setattr("sys.stdin", io.StringIO("hello\n"))
    assert cli.main(["-w", str(w), "-m", "fake", "--runs-dir", str(tmp / "runs")]) == 0
    assert "plain repl answer" in capsys.readouterr().out


def test_cli_tui_without_textual_explains_how_to_install(env, monkeypatch, capsys):
    import minicode.tui as tui
    from minicode import cli
    monkeypatch.setattr(tui, "available", lambda: False)
    monkeypatch.setattr(cli, "ChatClient", lambda: Script([]))
    assert cli.main(["-w", str(env[1]), "-m", "fake", "--ui", "tui", "--runs-dir", str(env[0] / "runs")]) == 2
    assert "pip install 'minicode[tui]'" in capsys.readouterr().err


# ---------------------------------------------------------------- Chinese (and other IME) input
def test_chinese_text_can_be_typed_and_sent(env):
    async def go(app, pilot):
        await pilot.press("你", "好")
        assert app.query_one(PromptInput).text == "你好"
        box = app.query_one(PromptInput)
        box.insert("，请帮我修复这个 bug")
        await pilot.press("enter")
        await idle(app, pilot)

    snap, ctl, client = run(go, env, [(["好的，我来看看。"], 10)])
    assert snap.users == ["你好，请帮我修复这个 bug"] and snap.assistants == ["好的，我来看看。"]
    assert client.calls[0]["messages"][-1]["content"] == "你好，请帮我修复这个 bug"


def test_history_and_menu_work_with_wide_characters(env):
    seen = []

    async def go(app, pilot):
        await say(app, pilot, "修复测试")
        await idle(app, pilot)
        await pilot.press("up")
        seen.append(app.query_one(PromptInput).text)

    run(go, env, [(["好"], 1)])
    assert seen == ["修复测试"]


def run_in_pty(extra_env, seconds=3.0):
    """Start the real UI in a pseudo terminal and return everything it wrote to the terminal in the first seconds."""
    import fcntl
    import os
    import pty
    import select
    import struct
    import subprocess
    import sys
    import tempfile
    import termios
    import time
    from pathlib import Path
    w = Path(tempfile.mkdtemp()) / "proj"
    w.mkdir()
    env = {**os.environ, "LLM_BASE_URL": "http://127.0.0.1:9", "LLM_API_KEY": "k", "TERM": "xterm-256color", **extra_env}
    for k in ("TEXTUAL_DISABLE_KITTY_KEY", "MINICODE_TUI_KITTY"):
        if k not in extra_env:
            env.pop(k, None)
    m, s = pty.openpty()
    fcntl.ioctl(s, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 100, 0, 0))
    p = subprocess.Popen([sys.executable, "-m", "minicode.cli", "-w", str(w), "-m", "fake", "--runs-dir", str(w.parent / "runs"), "--ui", "tui"],
                         stdin=s, stdout=s, stderr=s, env=env, close_fds=True)
    os.close(s)
    out, end = b"", time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([m], [], [], 0.1)
        if r:
            try:
                out += os.read(m, 65536)
            except OSError:
                break
    p.kill()
    p.wait()
    return out


def test_the_kitty_keyboard_protocol_is_off_so_that_iterm2_delivers_ime_text():
    import re
    out = run_in_pty({})
    assert b"minicode" in out                                                 # it started
    assert not re.search(rb"\x1b\[>\d+u", out), "the kitty keyboard protocol was enabled: iTerm2 loses text committed by an input method"


def test_the_kitty_keyboard_protocol_can_be_turned_back_on():
    import re
    out = run_in_pty({"MINICODE_TUI_KITTY": "1"})
    assert re.search(rb"\x1b\[>\d+u", out)


def test_importing_the_ui_package_decides_before_textual_reads_the_variable():
    import subprocess
    import sys
    code = "import minicode.tui, textual.constants as c; print(c.DISABLE_KITTY_KEY)"
    import os
    base = {k: v for k, v in os.environ.items() if k not in ("TEXTUAL_DISABLE_KITTY_KEY", "MINICODE_TUI_KITTY")}
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=base).stdout.strip() == "True"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**base, "MINICODE_TUI_KITTY": "1"}).stdout.strip() == "False"


# ---------------------------------------------------------------- survivors of the mutation run
def test_a_single_ctrl_c_when_idle_does_not_quit(env):
    async def go(app, pilot):
        await pilot.press("ctrl+c")
        await pilot.pause(0.1)
        assert not app._exit

    run(go, env, [])


def test_text_of_a_non_streaming_step_is_shown_after_a_streaming_step(env):
    class Mixed(Script):
        def chat(self, on_text=None, **kw):
            self.stream = len(self.calls) == 0                         # the first request streams, the second does not
            return super().chat(on_text=on_text, **kw)

    ctl, client = make_ctl(env, [])
    client = Mixed([(["Step one text. ", call("1", "read_file", path="a.txt")], 10), (["Step two text."], 10)])
    ctl.make_agent = lambda trajectory=None, on_text=None, confirm=None: Agent(
        client, "fake-model", Workspace(env[1]), trajectory=trajectory, on_text=on_text, confirm=confirm, session=True)
    app = MiniCodeApp(ctl, auto_approve=True)
    box = {}

    async def main():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await say(app, pilot, "go")
            await idle(app, pilot)
            box["snap"] = Snap(app)

    asyncio.run(main())
    assert box["snap"].assistants == ["Step one text. ", "Step two text."]


def test_quitting_in_the_middle_of_a_turn_does_not_crash_the_ui(env):
    gate = threading.Event()

    def slow():
        gate.wait(5)
        return (["late"], 10)

    async def go(app, pilot):
        await say(app, pilot, "slow one")
        await pilot.pause(0.3)
        await pilot.press("ctrl+q")
        gate.set()
        await pilot.pause(0.5)

    run(go, env, [slow])


# ---------------------------------------------------------------- an approval must not look like "working"
def test_waiting_for_approval_is_shown_as_such_and_rings_the_bell(env):
    seen = {}
    rang = []

    async def go(app, pilot):
        app.bell = lambda: rang.append(1)
        await say(app, pilot, "write it")
        await wait_for_panel(app, pilot)
        await pilot.pause(0.3)
        seen["waiting"] = app.query_one("#working").content.plain
        await pilot.press("y")
        await idle(app, pilot)

    run(go, env, approval_script())
    assert "Waiting for YOUR approval" in seen["waiting"] and "Working…" not in seen["waiting"] and rang
