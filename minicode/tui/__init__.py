"""The full-screen terminal UI (an optional extra: `pip install -e ".[tui]"`, needs Textual). The core of minicode does not import it."""
from __future__ import annotations

import os

# Textual asks the terminal for the kitty keyboard protocol with "report all keys" and "associated text". In iTerm2 that makes the text an
# input method commits (system Pinyin, Japanese, Korean, ...) fail to reach the application: Chinese cannot be typed. Without the protocol
# the terminal sends committed text as plain UTF-8, which works everywhere. The cost: a terminal then cannot tell shift+enter from enter
# (ctrl+j and alt+enter insert a newline instead). MINICODE_TUI_KITTY=1 turns the protocol back on. This must be decided before Textual is
# imported (it reads the variable at import time), which is why it is here, at the top of the package.
if os.environ.get("MINICODE_TUI_KITTY") != "1":
    os.environ.setdefault("TEXTUAL_DISABLE_KITTY_KEY", "1")


def available() -> bool:
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def run(controller, *, auto_approve: bool = False, workspace_label: str = "") -> int:
    """Run the UI until the user quits. `controller` is a minicode.controller.SessionController (not yet opened)."""
    from .app import MiniCodeApp
    app = MiniCodeApp(controller, auto_approve=auto_approve, workspace_label=workspace_label)
    app.run()
    return 0
