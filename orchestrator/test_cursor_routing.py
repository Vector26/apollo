"""Composer first, OpenRouter then local behind it: who answers, how fast the fallback joins, who gets cut."""
import threading
import time
import unittest

from agent_tools import Plan
from test_brain_tools import FakeTools
from test_vself_brain import FakeRoute, brain


class FakeCursor:
    """Stands in for CursorRoute: scripted tokens, delay or failure; records what the brain does to it."""

    model = "composer-2.5-fast"

    def __init__(self, tokens=("Composer ", "here."), first_delay=0.0, fail=None, is_available=True):
        self.tokens, self.first_delay, self.fail, self.is_available = list(tokens), first_delay, fail, is_available
        self.calls, self.missed, self.warmed, self.closed_handle = 0, 0, 0, threading.Event()
        self.closed = False

    def available(self):
        return self.is_available

    def warm(self):
        self.warmed += 1

    def note_missed(self):
        self.missed += 1

    def new_conversation(self):
        pass

    def prime(self, system):
        self.primed = system

    def close(self):
        self.closed = True

    def _race(self, messages, on_open=None):
        self.calls += 1
        if on_open is not None:
            on_open(type("H", (), {"close": lambda s: self.closed_handle.set()})())
        end = time.monotonic() + self.first_delay
        while time.monotonic() < end:
            if self.closed_handle.is_set():
                return
            time.sleep(0.01)
        if self.fail:
            raise self.fail
        for t in self.tokens:
            yield t


def make(cursor, cloud, local, hedge=0.2, tools=None):
    b = brain(local, cloud)
    b.prefer, b.cursor, b.hedge_after_s = "cursor", cursor, hedge
    b.tools = tools
    return b


def run(b, text="hello"):
    return "".join(b._cursor_first([{"role": "system", "content": "s"}, {"role": "user", "content": text}]))


class WhoAnswersTest(unittest.TestCase):
    def test_composer_answers_and_the_fallback_is_never_touched(self):
        cur, cloud, local = FakeCursor(), FakeRoute("cloud", ["cloud"]), FakeRoute("local", ["local"])
        b = make(cur, cloud, local)
        self.assertEqual(run(b), "Composer here.")
        self.assertEqual((cloud.calls, local.calls), (0, 0))
        self.assertEqual(b.last_route, "cursor:composer-2.5-fast")
        self.assertEqual(cur.missed, 0)

    def test_a_silent_composer_is_joined_by_openrouter_and_the_first_to_speak_wins(self):
        cur, cloud = FakeCursor(first_delay=2.0), FakeRoute("cloud", ["OpenRouter ", "here."], 0.02)
        b = make(cur, cloud, FakeRoute("local", ["local"]), hedge=0.2)
        t0 = time.monotonic()
        self.assertEqual(run(b), "OpenRouter here.")
        self.assertLess(time.monotonic() - t0, 1.0)  # about hedge + the cloud's own speed, not Composer's 2 s
        self.assertEqual(b.last_route, "cloud:cloud")
        self.assertTrue(cur.closed_handle.wait(1.0))  # the loser is cut
        self.assertEqual(cur.missed, 1)  # and told it missed this turn

    def test_a_composer_that_speaks_before_the_fallback_does_wins_the_race(self):
        cur, cloud = FakeCursor(first_delay=0.35), FakeRoute("cloud", ["cloud"], 1.5)
        b = make(cur, cloud, FakeRoute("local", ["local"]), hedge=0.2)
        self.assertEqual(run(b), "Composer here.")
        self.assertEqual(b.last_route, "cursor:composer-2.5-fast")
        self.assertEqual(cloud.calls, 1)  # it had joined, and was cut

    def test_a_failed_composer_starts_the_fallback_at_once_not_after_the_hedge_wait(self):
        cur = FakeCursor(fail=RuntimeError("boom"))
        b = make(cur, FakeRoute("cloud", ["cloud ", "ok"], 0.0), FakeRoute("local", ["local"]), hedge=5.0)
        t0 = time.monotonic()
        self.assertEqual(run(b), "cloud ok")
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_an_unavailable_composer_is_skipped_and_restarted_in_the_background(self):
        cur = FakeCursor(is_available=False)
        b = make(cur, FakeRoute("cloud", ["cloud ", "ok"], 0.0), FakeRoute("local", ["local"]))
        self.assertEqual(run(b), "cloud ok")
        self.assertEqual(cur.calls, 0)
        self.assertEqual(cur.warmed, 1)

    def test_composer_and_openrouter_down_means_the_local_brain_answers(self):
        cur = FakeCursor(fail=RuntimeError("down"))
        cloud = FakeRoute("cloud", [], fail=RuntimeError("openrouter down"))
        local = FakeRoute("local", ["local ", "answer"])
        b = make(cur, cloud, local)
        b.local_available = lambda: True
        self.assertEqual(run(b), "local answer")
        self.assertEqual(b.last_route, "local:local")

    def test_everything_down_raises_before_a_word_so_the_caller_can_retry(self):
        cur = FakeCursor(fail=RuntimeError("down"))
        b = make(cur, FakeRoute("cloud", [], fail=RuntimeError("x")), FakeRoute("local", [], fail=RuntimeError("y")))
        b.local_available = lambda: True
        with self.assertRaises(Exception):
            run(b)

    def test_a_failure_after_the_first_word_is_raised_so_what_was_said_is_kept(self):
        class Dies(FakeCursor):
            def _race(self, messages, on_open=None):
                yield "Half a "
                raise RuntimeError("mid-sentence")
        b = make(Dies(), FakeRoute("cloud", ["cloud"]), FakeRoute("local", ["local"]))
        got = []
        with self.assertRaises(RuntimeError):
            for piece in b._cursor_first([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]):
                got.append(piece)
        self.assertEqual("".join(got), "Half a ")


class RouteLabelTest(unittest.TestCase):
    def test_a_cut_fallback_cannot_rename_the_route_after_composer_won(self):
        cur = FakeCursor(first_delay=0.3)
        cloud = FakeRoute("cloud", ["cloud"], 0.4)  # joins at 0.2 s and would speak at 0.6 s, after Composer won
        b = make(cur, cloud, FakeRoute("local", ["local"]), hedge=0.2)
        self.assertEqual(run(b), "Composer here.")
        time.sleep(0.8)
        self.assertEqual(b.last_route, "cursor:composer-2.5-fast")

    def test_the_fallback_names_itself_only_when_it_wins(self):
        cur = FakeCursor(first_delay=2.0)
        b = make(cur, FakeRoute("cloud", ["cloud"], 0.0), FakeRoute("local", ["local"]), hedge=0.1)
        run(b)
        self.assertEqual(b.last_route, "cloud:cloud")


class EndToEndTest(unittest.TestCase):
    def test_a_composer_call_line_becomes_a_real_agent_call(self):
        tools = FakeTools()
        cur = FakeCursor(tokens=["[[ask:dume|how much ", "ram is in use]]"])
        b = make(cur, FakeRoute("cloud", ["cloud"]), FakeRoute("local", ["local"]), tools=tools)
        out = "".join(b.chat_stream("how much ram is in use", commit=False))
        self.assertEqual(tools.calls, [("dume", "how much ram is in use")])
        self.assertEqual(out, "dume says: 4 gigabytes.")
        self.assertEqual(b.last_route, "tool:ask")
        self.assertEqual(tools.by, "cursor:composer-2.5-fast")  # the audit learns which brain chose the call

    def test_the_banner_names_composer_first_then_openrouter_then_local(self):
        from vself_brain import VselfBrain
        from llm_client import LLMClient
        local = LLMClient(base_url="http://x/v1", model="local-m", system_prompt="", max_retries=0)
        cloud = LLMClient(base_url="http://x/v1", model="cloud-m", system_prompt="", max_retries=0)
        b = VselfBrain("http://x", "t", local, cloud, prefer="cursor", cursor=FakeCursor())
        self.assertEqual(b.model, "vself → composer-2.5-fast | cloud-m | local-m")
        self.assertEqual(b.prefer, "cursor")

    def test_switching_composer_off_falls_back_to_the_cloud_first_order(self):
        from vself_brain import VselfBrain
        from llm_client import LLMClient
        local = LLMClient(base_url="http://x/v1", model="local-m", system_prompt="", max_retries=0)
        cloud = LLMClient(base_url="http://x/v1", model="cloud-m", system_prompt="", max_retries=0)
        b = VselfBrain("http://x", "t", local, cloud, prefer="cursor", cursor=None)
        self.assertEqual((b.prefer, b.model), ("cloud", "vself → cloud-m | local-m"))

    def test_close_stops_the_composer_agent(self):
        cur = FakeCursor()
        b = make(cur, FakeRoute("cloud", ["c"]), FakeRoute("local", ["l"]))
        b.close()
        self.assertTrue(cur.closed)


if __name__ == "__main__":
    unittest.main()
