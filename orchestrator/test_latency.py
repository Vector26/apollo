"""Latency-path tests: clause splitting and caption-after-audio ordering.

No GPU, no mic, no LLM required.
"""

import threading
import unittest

import numpy as np

from cli import deliver_clause
from stt_stream import pick_device
from tts_engine import ClauseSplitter, split_token_stream


class ClauseSplitterTest(unittest.TestCase):
    def test_hard_stop_emits_immediately(self):
        s = ClauseSplitter()
        self.assertIsNone(s.feed("Hello"))
        self.assertEqual(s.feed(" there."), "Hello there.")

    def test_mid_buffer_sentence_splits_and_keeps_remainder(self):
        s = ClauseSplitter()
        out = s.feed("First done. Second is still run")
        self.assertEqual(out, "First done.")
        self.assertEqual(s.feed("ning."), "Second is still running.")

    def test_soft_stop_needs_min_words(self):
        s = ClauseSplitter()
        self.assertIsNone(s.feed("Yes, "))  # too short for a comma split
        self.assertEqual(s.feed("I think so, and more"), "Yes, I think so,")

    def test_first_clause_splits_at_short_comma(self):
        s = ClauseSplitter()
        self.assertEqual(s.feed("I am Ultron, an artificial"), "I am Ultron,")
        self.assertIsNone(s.feed(" mind, born of"))  # later clauses keep the 4-word minimum
        self.assertEqual(s.feed(" ambition."), "an artificial mind, born of ambition.")

    def test_bare_punctuation_is_not_a_clause(self):
        s = ClauseSplitter()
        s.feed("one two three four five six seven eight nine ten ")
        self.assertIsNone(s.feed("."))
        self.assertIsNone(s.flush())

    def test_force_split_at_word_limit(self):
        s = ClauseSplitter()
        clause = s.feed("one two three four five six seven eight nine ten ")
        self.assertEqual(clause, "one two three four five six seven eight nine ten")

    def test_forced_split_breaks_before_a_conjunction(self):
        s = ClauseSplitter()
        clause = s.feed("the hardware is fine for now and the driver is the real problem here ")
        self.assertEqual(clause, "the hardware is fine for now")
        self.assertEqual(s.flush(), "and the driver is the real problem here")

    def test_forced_split_never_ends_on_a_dangling_word(self):
        s = ClauseSplitter()
        self.assertIsNone(s.feed("one two three four five six seven eight nine the "))
        self.assertEqual(s.feed("ocean "), "one two three four five six seven eight nine the ocean")

    def test_forced_split_waits_for_the_last_word_to_finish(self):
        s = ClauseSplitter()
        self.assertIsNone(s.feed("one two three four five six seven eight nine ten"))  # "ten" may still grow
        self.assertEqual(s.feed("th "), "one two three four five six seven eight nine tenth")

    def test_forced_split_has_a_hard_ceiling(self):
        s = ClauseSplitter()
        text = "to " * 14
        self.assertEqual(s.feed(text), text.strip())

    def test_flush_returns_tail(self):
        s = ClauseSplitter()
        self.assertIsNone(s.feed("no punctuation here"))
        self.assertEqual(s.flush(), "no punctuation here")
        self.assertIsNone(s.flush())

    def test_token_stream_yields_in_order(self):
        tokens = ["Hi there. ", "How are ", "you doing", "? I am ", "fine."]
        clauses = list(split_token_stream(tokens))
        self.assertEqual(clauses[0], "Hi there.")
        self.assertEqual(clauses[-1], "I am fine.")
        self.assertEqual(" ".join(clauses), "Hi there. How are you doing? I am fine.")


class CaptionOrderTest(unittest.TestCase):
    def test_caption_never_before_audio(self):
        calls = []

        started = []

        class FakeSpeaker:
            def write_pcm(self, pcm, sr, on_start=None):
                calls.append(("pcm", len(pcm)))
                started.append(on_start)

        class FakeUI:
            def line(self, who, text):
                calls.append(("line", who, text))

        def fake_synth(text):
            return np.zeros(2400, dtype=np.float32), 24000

        ok = deliver_clause(FakeSpeaker(), FakeUI(), "Hello there.", fake_synth)
        self.assertTrue(ok)
        self.assertEqual([c[0] for c in calls], ["pcm"])  # queued, not yet playing
        started[0]()  # speaker reports playback start
        self.assertEqual([c[0] for c in calls], ["pcm", "line"])

    def test_no_caption_if_cancelled_before_playback(self):
        calls = []
        started = []
        cancel = threading.Event()

        class FakeSpeaker:
            def write_pcm(self, pcm, sr, on_start=None):
                started.append(on_start)

        class FakeUI:
            def line(self, who, text):
                calls.append(text)

        def fake_synth(text):
            return np.zeros(100, dtype=np.float32), 24000

        self.assertTrue(deliver_clause(FakeSpeaker(), FakeUI(), "late", fake_synth, cancel))
        cancel.set()  # barge-in before the audio started
        started[0]()
        self.assertEqual(calls, [])

    def test_cancelled_clause_writes_nothing(self):
        calls = []

        class FakeSpeaker:
            def write_pcm(self, pcm, sr, on_start=None):
                calls.append("pcm")

        class FakeUI:
            def line(self, who, text):
                calls.append("line")

        cancel = threading.Event()
        cancel.set()

        def fake_synth(text):
            return np.zeros(100, dtype=np.float32), 24000

        ok = deliver_clause(FakeSpeaker(), FakeUI(), "stale", fake_synth, cancel)
        self.assertFalse(ok)
        self.assertEqual(calls, [])


class DevicePickTest(unittest.TestCase):
    def test_forced_cpu(self):
        self.assertEqual(pick_device("cpu"), "cpu")

    def test_forced_cuda(self):
        self.assertEqual(pick_device("cuda"), "cuda")


if __name__ == "__main__":
    unittest.main()
