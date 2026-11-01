"""The Composer route, against a fake `cursor-agent acp`: speed guards, boxing-in, cleanup, failover signals."""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from cursor_brain import CursorRoute, CursorSlow, CursorUnavailable, _Clean

FAKE = str(Path(__file__).with_name("fake_acp_server.py"))
SYSTEM = {"role": "system", "content": "I am Prometheus."}


def msgs(user, history=()):
    return [SYSTEM, *history, {"role": "user", "content": user}]


class RouteCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cursor-brain-"))
        self.log = self.tmp / "fake.log"
        os.environ["FAKE_LOG"] = str(self.log)
        self.routes = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for r in self.routes:
            r.close()
        os.environ.pop("FAKE_LOG", None)
        os.environ.pop("FAKE_MODEL", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def route(self, **kw):
        kw.setdefault("give_up_s", 3.0)
        kw.setdefault("ready_wait_s", 10.0)
        r = CursorRoute(self.tmp / "sandbox", self.tmp / "state", agent_bin=FAKE, **kw)
        self.routes.append(r)
        return r

    def events(self, kind=None):
        if not self.log.exists():
            return []
        rows = [json.loads(x) for x in self.log.read_text().splitlines()]
        return [r for r in rows if kind is None or r["event"] == kind]

    def say(self, r, user, history=()):
        return "".join(r._race(msgs(user, history)))


class CleanTest(unittest.TestCase):
    def test_markdown_is_removed_even_when_split_across_chunks(self):
        c = _Clean()
        out = "".join(c.feed(x) for x in ["**Bold", "** and `code`", "\n- item one\n", "# Heading\n", "done."])
        self.assertEqual(out, "Bold and code item one Heading done.")

    def test_leading_space_dropped_and_runs_collapsed(self):
        c = _Clean()
        self.assertEqual(c.feed("  \n Hello   "), "Hello")
        self.assertEqual(c.feed("   there"), " there")

    def test_an_agent_call_line_survives(self):
        c = _Clean()
        self.assertEqual("".join(c.feed(x) for x in ["[[ask:dume|how much", " ram?]]"]), "[[ask:dume|how much ram?]]")


class TurnTest(RouteCase):
    def test_a_normal_turn_streams_clean_text(self):
        r = self.route()
        self.assertEqual(self.say(r, "hello"), "Hello there. Second sentence.")

    def test_it_is_boxed_in_ask_mode_in_an_empty_sandbox_with_the_right_model(self):
        r = self.route()
        self.say(r, "hello")
        self.assertEqual(self.events("set_mode")[0]["modeId"], "ask")
        self.assertTrue(self.events("session_new")[0]["cwd"].endswith("sandbox"))
        self.assertEqual(list((self.tmp / "sandbox").iterdir()), [])

    def test_the_fast_model_is_selected_when_the_default_is_something_else(self):
        os.environ["FAKE_MODEL"] = "auto"
        r = self.route()
        self.say(r, "hello")
        self.assertEqual(self.events("set_model")[0]["modelId"], "composer-2.5[fast=true]")

    def test_the_persona_is_sent_once_then_only_the_message_and_a_reminder(self):
        r = self.route()
        self.say(r, "first")
        self.say(r, "second")
        first, second = [e["text"] for e in self.events("prompt")]
        self.assertIn("I am Prometheus.", first)
        self.assertIn("at most two short plain sentences", first)
        self.assertNotIn("Prometheus", second)
        self.assertIn("no tools", second)
        self.assertTrue(second.endswith("Owner: second"))
        self.assertLess(len(second), 200)

    def test_history_is_sent_with_the_persona_on_a_fresh_session(self):
        r = self.route()
        hist = [{"role": "user", "content": "earlier thing"}, {"role": "assistant", "content": "earlier answer"}]
        self.say(r, "now", hist)
        self.assertIn("user: earlier thing", self.events("prompt")[0]["text"])

    def test_a_conversation_that_only_this_session_has_had_adds_no_history_block(self):
        r = self.route()
        self.say(r, "one")
        hist = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "Hello there. Second sentence."}]
        self.say(r, "two", hist)
        self.assertNotIn("Conversation so far", self.events("prompt")[1]["text"])

    def test_a_turn_that_happened_without_this_session_is_recovered_on_the_next(self):
        r = self.route()
        self.say(r, "one")
        hist = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "Hello there. Second sentence."},
                {"role": "user", "content": "how much ram"}, {"role": "assistant", "content": "DUM-E says: 14 of 30 gigabytes."}]
        self.say(r, "and the gpu?", hist)
        text = self.events("prompt")[1]["text"]
        self.assertIn("DUM-E says: 14 of 30 gigabytes.", text)  # a tool's answer the session never saw
        self.say(r, "thanks", hist + [{"role": "user", "content": "and the gpu?"},
                                      {"role": "assistant", "content": "Hello there. Second sentence."}])
        self.assertNotIn("Conversation so far", self.events("prompt")[2]["text"])  # caught up again

    def test_note_missed_forces_the_history_block_on_the_next_turn(self):
        r = self.route()
        self.say(r, "one")
        r.note_missed()
        hist = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "Hello there. Second sentence."}]
        self.say(r, "two", hist)
        self.assertIn("Conversation so far", self.events("prompt")[1]["text"])

    def test_markdown_in_a_reply_never_reaches_the_voice(self):
        r = self.route()
        self.assertEqual(self.say(r, "MARKDOWN please"), "Bold and code item one Heading done.")

    def test_an_agent_call_line_is_passed_through_for_the_brain_to_run(self):
        r = self.route()
        self.assertEqual(self.say(r, "MARKER"), "[[ask:dume|how much ram is in use]]")


class VselfShapeTest(RouteCase):
    """vself ends a turn with the owner's words (memories folded in) and then its own reminder, both as user messages."""

    def vself_messages(self):
        return [SYSTEM,
                {"role": "user", "content": "earlier thing"}, {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "[Memory] he likes music theory.\n\nUser: Help me write an opening line about grief."},
                {"role": "user", "content": "Reminder: reply to the last thing he said, in plain speech."}]

    def test_the_owners_words_reach_the_model_not_just_the_reminder(self):
        r = self.route()
        "".join(r._race(self.vself_messages()))
        text = self.events("prompt")[0]["text"]
        self.assertIn("User: Help me write an opening line about grief.", text)
        self.assertIn("[Memory] he likes music theory.", text)
        self.assertIn("Reminder: reply to the last thing he said", text)
        self.assertIn("user: earlier thing", text)  # what came before the turn is history
        self.assertEqual(text.count("Help me write an opening line about grief."), 1)  # and not also in the history

    def test_later_turns_carry_the_whole_turn_and_only_a_short_reminder(self):
        r = self.route()
        "".join(r._race(self.vself_messages()))
        "".join(r._race(self.vself_messages()))
        second = self.events("prompt")[1]["text"]
        self.assertIn("User: Help me write an opening line about grief.", second)
        self.assertNotIn("Prometheus", second)

    def test_a_single_trailing_user_message_still_works(self):
        r = self.route()
        self.assertEqual(self.say(r, "hello"), "Hello there. Second sentence.")
        self.assertIn("Owner: hello", self.events("prompt")[0]["text"])


class PrimeTest(RouteCase):
    def test_priming_sends_the_persona_in_the_background_so_the_first_real_turn_is_light(self):
        r = self.route()
        r.prime("I am Prometheus.")
        deadline = time.time() + 10
        while len(self.events("prompt")) < 1 and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)
        self.assertIn("I am Prometheus.", self.events("prompt")[0]["text"])
        self.assertEqual(self.say(r, "hello"), "Hello there. Second sentence.")
        real = self.events("prompt")[1]["text"]
        self.assertNotIn("Prometheus", real)  # already in the session
        self.assertLess(len(real), 200)


class SpeedGuardTest(RouteCase):
    def test_no_word_in_time_raises_before_anything_is_spoken_and_cancels_the_turn(self):
        r = self.route(give_up_s=0.4)
        with self.assertRaises(CursorSlow):
            self.say(r, "SLOW one")
        time.sleep(0.2)
        self.assertTrue(self.events("cancel"))

    def test_reaching_for_a_tool_aborts_the_turn_before_a_word(self):
        r = self.route()
        with self.assertRaises(CursorSlow):
            self.say(r, "TOOL please")
        self.assertTrue(self.events("cancel"))

    def test_a_backend_error_sent_as_reply_text_is_a_route_failure_not_speech(self):
        r = self.route()
        spoken = []
        with self.assertRaises(CursorUnavailable) as ctx:
            for piece in r._race(msgs("NETFAIL please")):
                spoken.append(piece)
        self.assertEqual(spoken, [])
        self.assertIn("RetriableError", str(ctx.exception))
        self.assertIsNone(r._last_reply)

    def test_an_ordinary_reply_that_mentions_errors_is_still_spoken(self):
        r = self.route()
        self.assertTrue(self.say(r, "hello"))

    def test_a_permission_request_is_refused_and_aborts(self):
        r = self.route()
        with self.assertRaises(CursorSlow):
            self.say(r, "PERM please")
        time.sleep(0.3)
        ans = self.events("permission_answer")
        self.assertEqual(ans[0]["answer"]["outcome"]["optionId"], "no")  # the reject option, never the allow

    def test_a_long_reply_is_cut_at_a_sentence_end_and_the_turn_cancelled(self):
        r = self.route(max_words=20)
        out = self.say(r, "LONG story")
        self.assertTrue(out.endswith("."))
        self.assertLess(len(out.split()), 30)
        time.sleep(0.2)
        self.assertTrue(self.events("cancel"))

    def test_closing_the_stream_early_cancels_the_turn(self):
        r = self.route()
        g = r._race(msgs("LONG story"))
        next(g)
        g.close()  # a barge-in, or a hedge lost to the other route
        time.sleep(0.3)
        self.assertTrue(self.events("cancel"))
        self.assertEqual(self.say(r, "hello"), "Hello there. Second sentence.")  # the session is still good


class OverlapTest(RouteCase):
    def test_a_newer_turn_replaces_one_still_running(self):
        r = self.route()
        first = r._race(msgs("LONG story"))
        next(first)
        got = []
        t = threading.Thread(target=lambda: got.append(self.say(r, "hello")))
        t.start()
        t.join(8)
        self.assertEqual(got, ["Hello there. Second sentence."])
        self.assertEqual(list(first), [])  # the replaced turn just ends, quietly


class LifecycleTest(RouteCase):
    def test_not_ready_in_time_means_unavailable_not_a_hang(self):
        r = self.route(ready_wait_s=0.0)
        t0 = time.time()
        with self.assertRaises(CursorUnavailable):
            self.say(r, "hello")
        self.assertLess(time.time() - t0, 1.0)

    def test_a_crash_is_unavailable_then_it_recovers(self):
        r = self.route()
        self.say(r, "hello")
        with self.assertRaises(CursorUnavailable):
            self.say(r, "CRASH now")
        r._next_boot = 0
        deadline = time.time() + 15
        out = None
        while time.time() < deadline:
            try:
                out = self.say(r, "hello again")
                break
            except CursorUnavailable:
                time.sleep(0.3)
        self.assertEqual(out, "Hello there. Second sentence.")

    def test_a_missing_binary_is_unavailable_and_backs_off(self):
        r = CursorRoute(self.tmp / "sb", self.tmp / "st", agent_bin="/nonexistent/cursor-agent", ready_wait_s=2.0)
        self.routes.append(r)
        with self.assertRaises(CursorUnavailable):
            self.say(r, "hello")
        self.assertGreater(r._next_boot, time.monotonic())

    def test_close_leaves_no_process_behind(self):
        r = self.route()
        self.say(r, "hello")
        pid = r._conn.proc.pid
        r.close()
        time.sleep(0.3)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_a_stale_agent_from_a_killed_earlier_run_is_ended_at_start(self):
        r1 = self.route()
        self.say(r1, "hello")
        proc = r1._conn.proc
        r2 = self.route()  # a second Apollo starts while the first was left behind
        self.say(r2, "hello")
        self.assertIsNotNone(proc.wait(timeout=5))  # r1's agent was ended by r2's start-up sweep (reaped here)

    def test_the_session_rotates_in_the_background_and_the_persona_is_resent(self):
        r = self.route(rotate_turns=2)
        self.say(r, "one")
        self.say(r, "two")
        deadline = time.time() + 5
        while r._pending_session is None and time.time() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(r._pending_session)
        self.say(r, "three")
        self.assertEqual(len(self.events("session_new")), 2)
        self.assertIn("I am Prometheus.", self.events("prompt")[2]["text"])


if __name__ == "__main__":
    unittest.main()
