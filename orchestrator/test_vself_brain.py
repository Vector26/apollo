import threading
import time
import unittest
from types import SimpleNamespace

import tempfile
from pathlib import Path

import vself_brain
from apollo_self import ApolloSelf
from vself_brain import BrainUnavailable, VselfBrain, local_opening_ok


class FakeRoute:
    """Stands in for LLMClient: ``_race`` yields tokens after a delay, or raises."""

    def __init__(self, model, tokens, first_delay=0.0, fail=None):
        self.model = model
        self.tokens = tokens
        self.first_delay = first_delay
        self.fail = fail
        self.calls = 0
        self.closed = threading.Event()
        self.stream_closed = threading.Event()

    def _race(self, messages, on_open=None):
        self.calls += 1
        if on_open is not None:
            on_open(SimpleNamespace(close=self.stream_closed.set))
        try:
            time.sleep(self.first_delay)
            if self.fail:
                raise self.fail
            for t in self.tokens:
                yield t
                time.sleep(0.005)
        finally:
            self.closed.set()


class FakeApi:
    def __init__(self, fail=False):
        self.fail = fail
        self.posts = []

    def post(self, path, json):
        self.posts.append((path, json))
        if self.fail:
            raise ConnectionError("vself down")
        if path.endswith("prepare"):
            body = {"system": "I am Sam.", "messages": [{"role": "user", "content": json["query"]}]}
            if json["query"].startswith("take a note"):
                body["direct_reply"] = "Got it, noted."
                body["action"] = "note"  # the real API sends the intent kind as a string
        elif path.endswith("end"):
            body = {"job_id": "j-1"}
        else:
            body = {"session_id": "s-1"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)


def brain(local, cloud, threshold_ms=200, api=None):
    b = VselfBrain.__new__(VselfBrain)
    b.api = api or FakeApi()
    b.local, b.cloud = local, cloud
    b.threshold_s = threshold_ms / 1000
    b.queue_url = None
    b.model = "test"
    b.history, b.last_route, b.session_id, b._last_system = [], None, None, None
    b._convo, b._trace, b._pstate, b._step = [], [], "composed", None
    import collections
    b._local_misses = collections.deque(maxlen=3)
    b.owns_memory = True
    b.prefer = "local"
    b.last_activity = time.monotonic()
    b._pending, b._lock = [], threading.Lock()
    return b


CLEAN = ["I spent ", "most of it ", "rebuilding the rig."]


class VselfRoutingTest(unittest.TestCase):
    def test_fast_clean_local_wins_and_cloud_never_sees_the_prompt(self):
        local, cloud = FakeRoute("local", CLEAN, 0.02), FakeRoute("cloud", ["cloud"])
        b = brain(local, cloud)
        self.assertEqual("".join(b.chat_stream("how was the week?", commit=False)), "".join(CLEAN))
        self.assertEqual(cloud.calls, 0)
        self.assertTrue(b.last_route.startswith("local"))

    def test_slow_local_hands_over_to_cloud_at_threshold(self):
        local, cloud = FakeRoute("local", CLEAN, 1.0), FakeRoute("cloud", ["Cloud ", "answer here, quick."])
        b = brain(local, cloud, threshold_ms=150)
        t = time.monotonic()
        out = "".join(b.chat_stream("q", commit=False))
        self.assertEqual(out, "Cloud answer here, quick.")
        self.assertLess(time.monotonic() - t, 0.6)
        self.assertTrue(b.last_route.startswith("cloud"))

    def test_defective_local_opening_goes_to_cloud(self):
        local = FakeRoute("local", ["<think>", "hmm let me think about it"], 0.01)
        cloud = FakeRoute("cloud", ["Fine answer from the cloud."], 0.05)
        b = brain(local, cloud)
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "Fine answer from the cloud.")

    def test_local_error_goes_to_cloud_immediately(self):
        local = FakeRoute("local", [], 0.01, fail=ConnectionError("modelq down"))
        cloud = FakeRoute("cloud", ["Cloud reply that is fine."], 0.02)
        b = brain(local, cloud, threshold_ms=5000)
        t = time.monotonic()
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "Cloud reply that is fine.")
        self.assertLess(time.monotonic() - t, 1.0)

    def test_busy_local_skips_straight_to_cloud(self):
        local, cloud = FakeRoute("local", CLEAN), FakeRoute("cloud", ["Cloud reply that is fine."])
        b = brain(local, cloud, threshold_ms=5000)
        b.local_available = lambda: False
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "Cloud reply that is fine.")
        self.assertEqual(local.calls, 0)

    def test_without_cloud_slow_local_still_answers(self):
        b = brain(FakeRoute("local", CLEAN, 0.3), None, threshold_ms=100)
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "".join(CLEAN))

    def test_three_local_misses_start_cloud_without_head_start(self):
        local, cloud = FakeRoute("local", CLEAN, 0.6), FakeRoute("cloud", ["Cloud reply that is fine."], 0.01)
        b = brain(local, cloud, threshold_ms=400)
        b._local_misses.extend([True, True, True])
        t = time.monotonic()
        "".join(b.chat_stream("q", commit=False))
        self.assertLess(time.monotonic() - t, 0.55)  # cloud picked at the threshold, not after it + its own latency


class FlakyRoute(FakeRoute):
    """Fails the first ``fails`` calls, then answers."""

    def __init__(self, model, tokens, fails):
        super().__init__(model, tokens)
        self.fails_left = fails

    def _race(self, messages, on_open=None):
        if self.fails_left > 0:
            self.fails_left -= 1
            self.calls += 1
            raise OSError("Connection error.")
        yield from super()._race(messages, on_open)


class ResilienceTest(unittest.TestCase):
    def setUp(self):
        self._pause = vself_brain.RETRY_PAUSE_S
        vself_brain.RETRY_PAUSE_S = 0.01

    def tearDown(self):
        vself_brain.RETRY_PAUSE_S = self._pause

    def test_one_network_blip_is_retried_silently(self):
        cloud = FlakyRoute("cloud", ["Cloud reply that is fine."], fails=1)
        b = brain(FakeRoute("local", [], fail=OSError("paused")), cloud, threshold_ms=5000)
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "Cloud reply that is fine.")

    def test_two_failures_raise_brain_unavailable(self):
        b = brain(FakeRoute("local", [], fail=OSError("down")), FakeRoute("cloud", [], fail=OSError("net")))
        with self.assertRaises(BrainUnavailable):
            "".join(b.chat_stream("q", commit=False))

    def test_paused_local_is_not_used_as_the_fallback(self):
        local, cloud = FakeRoute("local", CLEAN), FakeRoute("cloud", [], fail=OSError("net"))
        b = brain(local, cloud)
        b.prefer = "cloud"
        b.local_available = lambda: False
        with self.assertRaises(BrainUnavailable):
            "".join(b.chat_stream("q", commit=False))
        self.assertEqual(local.calls, 0)

    def test_failure_after_words_is_not_retried(self):
        class Dies(FakeRoute):
            def _race(self, messages, on_open=None):
                yield "I was saying that "
                raise OSError("peer closed connection")

        b = brain(Dies("cloud", []), None)
        got = []
        with self.assertRaises(OSError):
            for d in b.chat_stream("q", commit=False):
                got.append(d)
        self.assertEqual(got, ["I was saying that "])

    def test_losing_route_stream_is_closed_not_left_hanging(self):
        local, cloud = FakeRoute("local", CLEAN, 5.0), FakeRoute("cloud", ["Cloud reply that is fine."], 0.01)
        b = brain(local, cloud, threshold_ms=50)
        "".join(b.chat_stream("q", commit=False))
        self.assertTrue(local.stream_closed.wait(1.0))


class CloudFirstTest(unittest.TestCase):
    def test_cloud_answers_and_local_is_never_asked(self):
        local, cloud = FakeRoute("local", CLEAN), FakeRoute("cloud", ["Cloud reply that is fine."], 0.3)
        b = brain(local, cloud, threshold_ms=10)
        b.prefer = "cloud"
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "Cloud reply that is fine.")
        self.assertEqual(local.calls, 0)
        self.assertTrue(b.last_route.startswith("cloud"))

    def test_cloud_failure_falls_back_to_local(self):
        local, cloud = FakeRoute("local", CLEAN), FakeRoute("cloud", [], fail=OSError("net"))
        b = brain(local, cloud)
        b.prefer = "cloud"
        self.assertEqual("".join(b.chat_stream("q", commit=False)), "".join(CLEAN))
        self.assertTrue(b.last_route.startswith("local"))

    def test_config_order_is_composer_then_openrouter_then_local(self):
        import os
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.example.yaml")))
        os.environ["VSM_BOOTSTRAP_TOKEN"] = "t"
        os.environ.setdefault("OPENROUTER_API_KEY", "k")
        try:
            b = VselfBrain.from_config(cfg["brains"]["vself"], cfg["brains"])
        finally:
            del os.environ["VSM_BOOTSTRAP_TOKEN"]
        self.assertEqual(b.prefer, "cursor")
        self.assertEqual(b.cursor.model_id, "composer-2.5[fast=true]")  # strictly the fast variant
        self.assertEqual(b.model.split(" | ")[0], "vself → composer-2.5-fast")
        self.assertEqual(b.model.split(" | ")[1], cfg["brains"]["openrouter"]["model"])  # OpenRouter second
        self.assertEqual(b.model.split(" | ")[2], cfg["brains"]["vself"]["local"]["model"])  # local last


class VselfMemoryTest(unittest.TestCase):
    def test_commit_records_turn_in_vself(self):
        api = FakeApi()
        b = brain(FakeRoute("local", CLEAN, 0.01), None, api=api)
        "".join(b.chat_stream("I started running again", commit=False))
        b.commit("I started running again", "".join(CLEAN))
        for _ in range(100):
            if b.session_id:
                break
            time.sleep(0.01)
        commits = [j for p, j in api.posts if p.endswith("commit")]
        self.assertEqual(commits[0]["user"], "I started running again")
        self.assertEqual(b.session_id, "s-1")
        self.assertEqual(len(b.history), 2)

    def test_direct_reply_is_spoken_without_a_model_and_still_committed(self):
        api = FakeApi()
        local, cloud = FakeRoute("local", CLEAN), FakeRoute("cloud", ["x"])
        b = brain(local, cloud, api=api)
        b.history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}]
        out = "".join(b.chat_stream("take a note that the rig needs a new fan", commit=True))
        self.assertEqual(out, "Got it, noted.")
        self.assertIsNotNone(b._last_system)  # prepare succeeded; it did not fall back to the stale persona
        self.assertEqual((local.calls, cloud.calls), (0, 0))
        self.assertEqual(b.last_route, "vself:direct")
        for _ in range(100):
            if b.session_id:
                break
            time.sleep(0.01)
        commit = [j for p, j in api.posts if p.endswith("commit")][0]
        self.assertEqual(commit["assistant"], "Got it, noted.")
        self.assertEqual(commit["history"], [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}])
        self.assertRegex(commit["now"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")

    def test_prepare_sends_the_clock_and_the_thread(self):
        api = FakeApi()
        b = brain(FakeRoute("local", CLEAN), None, api=api)
        b.history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        b.prepare("yeah")
        sent = api.posts[0][1]
        self.assertEqual(len(sent["history"]), 2)
        self.assertIn("now", sent)

    def test_end_consolidates_and_starts_fresh(self):
        api = FakeApi()
        b = brain(FakeRoute("local", CLEAN, 0.01), None, api=api)
        "".join(b.chat_stream("I started running again", commit=True))
        self.assertTrue(b.end())  # waits for the commit that creates the session
        ends = [j for p, j in api.posts if p.endswith("end")]
        self.assertEqual(ends, [{"session_id": "s-1"}])
        self.assertIsNone(b.session_id)
        self.assertEqual(b.history, [])

    def test_end_keeping_history_for_a_mid_conversation_checkpoint(self):
        b = brain(FakeRoute("local", CLEAN, 0.01), None)
        "".join(b.chat_stream("hello there", commit=True))
        b.end(keep_history=True)
        self.assertEqual(len(b.history), 2)

    def test_end_without_a_session_does_nothing(self):
        api = FakeApi()
        b = brain(FakeRoute("local", CLEAN), None, api=api)
        self.assertFalse(b.end())
        self.assertEqual(api.posts, [])

    def test_prepare_outage_reuses_last_persona(self):
        b = brain(FakeRoute("local", CLEAN), None)
        b.prepare("hello")
        b.api = FakeApi(fail=True)
        msgs = b.prepare("still there?")
        self.assertTrue(msgs[0]["content"].startswith("I am Sam."))
        self.assertIn("respond to every one of them", msgs[0]["content"])  # bursts of speech are all answered
        self.assertEqual(msgs[-1]["content"], "still there?")

    def test_prepare_outage_before_any_success_is_an_error(self):
        b = brain(FakeRoute("local", CLEAN), None, api=FakeApi(fail=True))
        with self.assertRaises(RuntimeError):
            b.prepare("hello")

    def test_prepare_injects_apollo_self_block(self):
        b = brain(FakeRoute("local", CLEAN), None)
        b.apollo_self = ApolloSelf(Path(tempfile.mkdtemp()) / "self.json")
        msgs = b.prepare("hello")
        self.assertIn("Who I am becoming", msgs[0]["content"])
        self.assertIn("I am Sam.", msgs[0]["content"])


class FromConfigTest(unittest.TestCase):
    def test_real_config_builds_local_and_cloud_routes(self):
        import os
        import yaml
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.example.yaml")))
        os.environ["VSM_BOOTSTRAP_TOKEN"] = "t"
        os.environ.setdefault("OPENROUTER_API_KEY", "k")
        try:
            b = VselfBrain.from_config(cfg["brains"]["vself"], cfg["brains"])
        finally:
            del os.environ["VSM_BOOTSTRAP_TOKEN"]
        self.assertEqual(b.local.model, "local-qwen3.6-35b")
        self.assertEqual(b.cloud.model, cfg["brains"]["openrouter"]["model"])
        self.assertIsNone(b.cloud.fallback)
        self.assertTrue(b.owns_memory)


class OpeningGuardTest(unittest.TestCase):
    def test_guard(self):
        self.assertTrue(local_opening_ok("I went hiking"))
        for bad in ("", "  ", "<think>ok", "**Sure**", "# Week", "- item one", "The user went", "As an AI I"):
            self.assertFalse(local_opening_ok(bad), bad)


if __name__ == "__main__":
    unittest.main()


class SelfReflectionFlowTest(unittest.TestCase):
    def test_reflection_sees_the_whole_conversation_not_the_capped_history(self):
        b = brain(FakeRoute("local", CLEAN, 0.0), None)
        for i in range(20):
            "".join(b.chat_stream(f"point number {i}", commit=True))
        self.assertGreater(len(b._convo), len(b.history))
        seen = []
        b._reflect_self = lambda convo, trace=None: seen.append(len(convo))
        b.end()
        for t in threading.enumerate():
            if t.name == "apollo-self":
                t.join(2)
        self.assertEqual(seen, [40])
        self.assertEqual(b._convo, [])

    def test_reflection_never_uses_the_local_brain(self):
        b = brain(FakeRoute("local", CLEAN), None)
        b.apollo_self = ApolloSelf(Path(tempfile.mkdtemp()) / "self.json")
        called = []
        b.local.complete = lambda *a, **k: called.append(1) or "{}"
        b._reflect_self([{"role": "user", "content": "x" * 100}] * 4)
        self.assertEqual(called, [])


class MannerFlowTest(unittest.TestCase):
    def make(self):
        import random
        b = brain(FakeRoute("local", CLEAN, 0.0), None)
        store = ApolloSelf(Path(tempfile.mkdtemp()) / "self.json")
        b.apollo_self = store
        store.automaton = lambda rng=None, _o=store.automaton: _o(rng=random.Random(3))
        return b

    def test_each_reply_gets_a_manner_line_for_what_was_just_said(self):
        b = self.make()
        msgs = b.prepare("how much vram is free")
        self.assertTrue(msgs[-1]["content"].startswith("[Manner for this reply, "))

    def test_preparing_twice_does_not_move_the_state_until_commit(self):
        b = self.make()
        first = b.prepare("check the gpu")[-1]["content"]
        again = b.prepare("check the gpu")[-1]["content"]
        self.assertEqual(first, again)
        self.assertEqual(b._pstate, "composed")
        "".join(b.chat_stream("check the gpu", commit=True))
        self.assertEqual(len(b._trace), 1)
        self.assertEqual(b._pstate, b._trace[0][2])

    def test_end_learns_from_the_trace_and_resets_to_composed(self):
        b = self.make()
        "".join(b.chat_stream("check the gpu", commit=True))
        "".join(b.chat_stream("thanks", commit=True))
        learned = []
        b.apollo_self.learn = lambda trace: learned.append(list(trace))
        b.end()
        for t in threading.enumerate():
            if t.name == "apollo-self":
                t.join(2)
        self.assertEqual(len(learned[0]), 2)
        self.assertEqual((b._pstate, b._trace), ("composed", []))


class AsideFlowTest(unittest.TestCase):
    def make(self, cloud=True):
        b = brain(FakeRoute("local", CLEAN, 0.0), None)
        b.apollo_self = ApolloSelf(Path(tempfile.mkdtemp()) / "self.json")
        if cloud:
            b.cloud = SimpleNamespace(calls=[], complete=None)
            b.cloud.complete = lambda system, user, max_tokens=None: (
                b.cloud.calls.append((system, user, max_tokens)) or "Rather late, sir.")
        return b

    def test_the_line_comes_from_the_cloud_with_persona_tone_and_the_recent_exchange(self):
        b = self.make()
        "".join(b.chat_stream("still debugging the rig", commit=True))
        line = b.aside("solicitous", "Local time 02:10.")
        self.assertEqual(line, "Rather late, sir.")
        system, user, max_tokens = b.cloud.calls[0]
        self.assertIn("Jarvis", system)
        self.assertIn("still debugging the rig", user)
        self.assertIn("Local time 02:10.", user)
        self.assertLessEqual(max_tokens, 120)

    def test_vself_memories_are_not_sent_with_it(self):
        b = self.make()
        b._last_system = "SECRET MEMORY: his bank pin"
        "".join(b.chat_stream("hello there", commit=True))
        b.aside("wry", "facts")
        self.assertNotIn("SECRET MEMORY", " ".join(b.cloud.calls[0][:2]))

    def test_without_a_cloud_brain_or_a_conversation_there_is_nothing_and_local_is_never_used(self):
        b = self.make(cloud=False)
        "".join(b.chat_stream("hello there", commit=True))
        called = []
        b.local.complete = lambda *a, **k: called.append(1) or "x"
        self.assertIsNone(b.aside("wry", "facts"))
        b2 = self.make()
        self.assertIsNone(b2.aside("wry", "facts"))  # nothing said yet
        self.assertEqual(called, [])

    def test_cloud_in_use_follows_the_route_that_answered(self):
        b = self.make()
        for route, expected in (("cursor:composer-2.5", True), ("cloud:gemma", True), ("local:qwen", False),
                                ("tool:notes", False), ("vself:direct", False), (None, False)):
            b.last_route = route
            self.assertIs(b.cloud_in_use(), expected, route)

    def test_session_age_and_owner_symbols(self):
        b = self.make()
        self.assertEqual(b.session_age_s(), 0.0)
        "".join(b.chat_stream("i am so tired", commit=True))
        self.assertEqual(b.owner_symbols(), ["tired"])
        self.assertGreater(b.session_age_s(), 0.0)
        b.end()
        for t in threading.enumerate():
            if t.name == "apollo-self":
                t.join(2)
        self.assertEqual((b.session_age_s(), b.owner_symbols()), (0.0, []))

    def test_an_aside_stays_in_the_thread_but_not_in_what_gets_reflected_on(self):
        b = self.make()
        "".join(b.chat_stream("hello there", commit=True))
        convo = len(b._convo)
        b.note_aside("Rather late, sir.")
        self.assertEqual(b.history[-1], {"role": "assistant", "content": "Rather late, sir."})
        self.assertEqual(len(b._convo), convo)
