"""Speaking unprompted: results that arrive while the owner is not mid-conversation (a slow agent's
answer, a long job finishing). The voice loop polls this; it never interrupts a turn or the owner."""
from __future__ import annotations

from typing import Any, Callable


class Announcer:
    def __init__(self, tools: Any, brain: Any, speak: Callable[[str], None], ready: Callable[[], bool]) -> None:
        self.tools = tools
        self.brain = brain
        self.speak = speak
        self.ready = ready

    def poll(self) -> bool:
        """Speak one waiting announcement if the floor is free. True when something was said."""
        if self.tools is None or not self.tools.has_announcement() or not self.ready():
            return False
        text = self.tools.take_announcement()
        if not text:
            return False
        self.speak(text)
        note = getattr(self.brain, "note_background", None)
        if note:
            note(text)  # so "tell me more about that" has something to refer to
        return True
