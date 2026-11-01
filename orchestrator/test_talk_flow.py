"""The real `_run_talk` loop, headless: fake mic, STT, voice, speaker and brain.

Covers the turn-taking and session behaviour that lives in the loop itself: held fragments,
filler turns, goodbye, a brain that is down, and ending the conversation (exit and idle).
"""

import os
import queue
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import cli
from acks import Acks
from vself_brain import BrainUnavailable


class FakeBrain:
    owns_memory = True
    model = "fake"

    def __init__(self, fail=False, delay=0.0):
        self.fail = fail
        self.delay = delay  # seconds before the first token
        self.heard: list[str] = []
        self.history: list[dict] = []
        self.last_route = "fake"
        self.last_activity = time.monotonic()
        self.session = False
        self.ended: list[dict] = []
        self.committed: list[tuple[str, str]] = []

    def warm(self):
        pass

    def reset(self):
        self.history.clear()

    def chat_stream(self, text, memory_block="", commit=False):
        self.heard.append(text)
        time.sleep(self.delay)
        if self.fail:
            raise BrainUnavailable("Connection error.")
        yield f"Reply to {text}."

    def commit(self, user, reply):
        self.committed.append((user, reply))
        self.session = True
        self.last_activity = time.monotonic()

    def has_session(self):
        return self.session

    def idle_s(self):
        return time.monotonic() - self.last_activity

    def end(self, keep_history=False):
        self.ended.append({"keep_history": keep_history})
        saved, self.session = self.session, False
        return saved


class AsideBrain(FakeBrain):
    """FakeBrain that can speak up unprompted, like VselfBrain on a cloud route."""

    def __init__(self, cloud=True, line="Rather late, sir. Might I suggest sleep.", **kw):
        super().__init__(**kw)
        self.cloud, self.line = cloud, line
        self.asked: list[tuple[str, str]] = []
        self.noted: list[str] = []

    def cloud_in_use(self): return self.cloud
    def owner_symbols(self): return ["tired"]
    def session_age_s(self): return 60.0

    def aside(self, tone, facts):
        self.asked.append((tone, facts))
        return self.line

    def note_aside(self, text): self.noted.append(text)


class Harness:
    def __init__(self, brain, hold_ms=250, idle_end_s=600.0, llm_extra=None):
        self.brain = brain
        self.texts: queue.Queue = queue.Queue()
        self.spoken: list[str] = []
        self.evq = None
        self.stop = threading.Event()
        self.tmp = tempfile.mkdtemp(prefix="vox-flow-")
        self.error: Exception | None = None
        h = self

        class FakeSTTWorker:
            def __init__(self, stt, evq):
                h.evq = evq

            def start(self): pass
            def stop(self): pass
            def submit_partial(self, pcm): pass

            def submit_final(self, audio, dur, tag=None):
                h.evq.put(("final_text", h.texts.get_nowait(), 0.0, tag))

        class FakeTTS:
            display_name = "Fake"
            device = "cpu"

            def prerender(self, lines): pass
            def synthesize(self, text): return np.zeros(8, dtype=np.float32), 16000

            def phrase(self, text):
                h.spoken.append(text)
                return np.zeros(8, dtype=np.float32), 16000

        class FakeSpeaker:
            def __init__(self, *a, **k): pass
            def start(self): pass
            def close(self): pass
            def flush(self): pass
            def drain(self, timeout=0): pass
            def voice_active(self, tail): return False

            def write_pcm(self, pcm, sr, on_start=None):
                if on_start:
                    on_start()

            def after_playback(self, cb): cb()

        class FakeCapture:
            def __init__(self, **kw): pass
            def start(self): pass

        self.patches = [
            patch.object(cli, "apply_usb_mic_linux_fixes"),
            patch.object(cli, "talk_llm_config", return_value=(
                {"kind": "vself", "idle_end_s": idle_end_s, "api_url": "x", **(llm_extra or {})}, "sys")),
            patch("voice.presets.load_profile", return_value=SimpleNamespace(brain="vself", persona="p", profile_id="ultron")),
            patch("vself_brain.VselfBrain.from_config", return_value=brain),
            patch.object(cli.dns_cache, "install"),
            patch.object(cli.dns_cache, "prewarm"),
            patch.object(cli, "resolve_devices", return_value=("Spk", None, "none", None, "")),
            patch.object(cli, "find_input_device", return_value=None),
            patch.object(cli, "build_stt", return_value=MagicMock(device="cpu")),
            patch.object(cli, "ChatterboxEngine", return_value=FakeTTS()),
            patch.object(cli, "WarmSpeaker", FakeSpeaker),
            patch.object(cli, "ContinuousCapture", FakeCapture),
            patch.object(cli, "STTWorker", FakeSTTWorker),
            patch.object(cli, "wav_bytes_to_f32", lambda b: b),
        ]
        self.cfg = {
            "llm": {"base_url": "x", "model": "m", "system_prompt": "s"},
            "notes": {"output_dir": os.path.join(self.tmp, "notes")},
            "memory": {"dir": os.path.join(self.tmp, "mem")},
            "bluetooth": {"mic_name": "none"},
            "audio": {"sample_rate": 16000, "vad_threshold": 0.5, "hold_fragment_ms": hold_ms, "device_watch": False},
            "stt": {}, "tts": {"talk_profile": "ultron"},
        }

    def __enter__(self):
        for p in self.patches:
            p.start()

        def run():
            try:
                cli._run_talk(self.cfg, MagicMock(), self.stop, voice="ultron")
            except Exception as e:  # surfaced by the test, not swallowed by the thread
                self.error = e

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        from cli import STARTUP_LINES
        self.until(lambda: self.evq is not None and any(line in self.spoken for line in STARTUP_LINES))
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(5)
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp, True)

    def until(self, cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.error:
                raise self.error
            if cond():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached in time")

    def say(self, text):
        self.texts.put(text)
        self.evq.put(("speech_end", b"", 1.0, 0.2))


class TalkFlowTest(unittest.TestCase):
    def test_a_normal_turn_is_answered_and_committed(self):
        brain = FakeBrain()
        with Harness(brain) as h:
            h.say("How is the rig doing?")
            h.until(lambda: brain.committed)
        self.assertEqual(brain.heard, ["How is the rig doing?"])

    def test_a_dangling_fragment_waits_and_is_joined_to_what_follows(self):
        brain = FakeBrain()
        with Harness(brain, hold_ms=3000) as h:
            h.say("I just")
            time.sleep(0.4)
            self.assertEqual(brain.heard, [])  # not answered
            h.say("wanted to ask about the rig.")
            h.until(lambda: brain.committed)
        self.assertEqual(brain.heard, ["I just wanted to ask about the rig."])

    def test_an_abandoned_fragment_is_answered_after_the_window(self):
        brain = FakeBrain()
        with Harness(brain, hold_ms=250) as h:
            h.say("And then")
            h.until(lambda: brain.heard == ["And then"], timeout=3)

    def test_filler_only_turn_is_ignored(self):
        brain = FakeBrain()
        with Harness(brain) as h:
            h.say("Um...")
            time.sleep(0.4)
            self.assertEqual(brain.heard, [])
            h.say("Okay, what about the fan?")
            h.until(lambda: brain.committed)
        self.assertEqual(brain.heard, ["Okay, what about the fan?"])

    def test_brain_down_says_so_out_loud_and_the_loop_keeps_going(self):
        brain = FakeBrain(fail=True)
        with Harness(brain) as h:
            h.say("Tell me something.")
            h.until(lambda: cli.FAIL_LINE in h.spoken)
            brain.fail = False
            h.say("Are you back?")
            h.until(lambda: brain.committed)
        self.assertEqual(brain.heard, ["Tell me something.", "Are you back?"])

    def test_goodbye_variant_exits_and_ends_the_session_once(self):
        brain = FakeBrain()
        brain.session = True
        with Harness(brain) as h:
            h.say("Okay, bye-bye.")
            h.thread.join(5)
            self.assertFalse(h.thread.is_alive())
            self.assertIn("Goodbye.", h.spoken)
        self.assertEqual(brain.ended, [{"keep_history": False}])
        self.assertEqual(brain.heard, [])

    def test_take_notes_checkpoints_through_the_brain_and_keeps_the_conversation(self):
        brain = FakeBrain()
        brain.session = True
        with Harness(brain) as h:
            h.say("Take notes.")
            h.until(lambda: "Notes saved." in h.spoken)
            self.assertEqual(brain.ended, [{"keep_history": True}])
            md = [f for _, _, fs in os.walk(h.cfg["notes"]["output_dir"]) for f in fs if f.endswith(".md")]
            self.assertEqual(md, [])  # no loose markdown: vself files the notes

    def test_a_sentence_that_asks_for_notes_goes_to_the_brain(self):
        brain = FakeBrain()
        with Harness(brain) as h:
            h.say("Can you take notes on this idea?")
            h.until(lambda: brain.committed)
        self.assertEqual(brain.heard, ["Can you take notes on this idea?"])

    def test_a_slow_ops_question_gets_an_ack_but_small_talk_does_not(self):
        brain = FakeBrain(delay=0.6)
        h = Harness(brain)
        h.cfg["audio"]["acks"] = {"enabled": True, "after_ms": 100}
        with h:
            h.say("Check the GPU status.")
            h.until(lambda: brain.committed)
            self.assertTrue(set(h.spoken) & set(Acks({"enabled": True}).all_lines()))
            n = len(h.spoken)
            time.sleep(0.3)
            h.say("How are you today?")
            h.until(lambda: len(brain.committed) == 2)
        self.assertEqual(len(h.spoken), n, h.spoken)  # no ack for small talk

    def test_a_fast_brain_gets_no_ack(self):
        brain = FakeBrain(delay=0.0)
        h = Harness(brain)
        h.cfg["audio"]["acks"] = {"enabled": True, "after_ms": 400}
        with h:
            h.say("Check the GPU status.")
            h.until(lambda: brain.committed)
            time.sleep(0.6)
        self.assertFalse(set(h.spoken) & set(Acks({"enabled": True}).all_lines()))

    def test_a_transcript_of_its_own_voice_is_not_a_turn(self):
        brain = FakeBrain()
        h = Harness(brain)
        h.cfg["audio"]["echo_gate"] = {"enabled": True}
        with h:
            h.say("Check the GPU status please.")
            h.until(lambda: brain.committed)
            time.sleep(0.3)
            h.say("reply to check the gpu status please")  # the assistant's own words coming back
            time.sleep(0.4)
            h.say("Now what about the fan?")
            h.until(lambda: len(brain.committed) == 2)
        self.assertEqual(brain.heard, ["Check the GPU status please.", "Now what about the fan?"])

    ASIDE_CFG = {"initiative": {"enabled": True, "min_idle_s": 0.3, "max_idle_s": 30, "min_gap_s": 0.1,
                                "roll_every_s": 0.1, "after_none_s": 0.1}}

    def aside_harness(self, brain):
        import random
        from initiative import Initiative

        class Win(random.Random):
            def random(self): return 0.0  # every roll succeeds

        h = Harness(brain, llm_extra=self.ASIDE_CFG)
        h.patches.append(patch.object(cli, "Initiative", lambda cfg: Initiative(cfg, rng=Win())))
        return h

    def test_on_a_cloud_route_it_speaks_up_after_a_quiet_spell(self):
        brain = AsideBrain()
        with self.aside_harness(brain) as h:
            h.say("I am so tired, been up all night.")
            h.until(lambda: brain.committed)
            h.until(lambda: brain.noted, timeout=4)
        self.assertEqual(brain.noted, [brain.line])
        self.assertEqual(brain.asked[0][0], "solicitous")
        self.assertIn("tired", brain.asked[0][1])

    def test_on_the_local_brain_it_never_speaks_up(self):
        brain = AsideBrain(cloud=False)
        with self.aside_harness(brain) as h:
            h.say("Check the GPU status.")
            h.until(lambda: brain.committed)
            time.sleep(1.2)
        self.assertEqual((brain.asked, brain.noted), ([], []))

    def test_a_model_that_answers_none_says_nothing(self):
        brain = AsideBrain(line="NONE")
        with self.aside_harness(brain) as h:
            h.say("Check the GPU status.")
            h.until(lambda: brain.committed)
            h.until(lambda: brain.asked, timeout=4)
            time.sleep(0.3)
        self.assertEqual(brain.noted, [])

    def test_it_stays_quiet_if_the_owner_spoke_while_the_model_was_thinking(self):
        brain = AsideBrain()
        aside = brain.aside

        def slow_aside(tone, facts):
            out = aside(tone, facts)
            brain.last_activity = time.monotonic() + 1  # an exchange landed meanwhile
            return out
        brain.aside = slow_aside
        with self.aside_harness(brain) as h:
            h.say("Check the GPU status.")
            h.until(lambda: brain.committed)
            h.until(lambda: brain.asked, timeout=4)
            time.sleep(0.5)
        self.assertEqual(brain.noted, [])

    def test_a_quiet_spell_ends_the_conversation(self):
        brain = FakeBrain()
        with Harness(brain, idle_end_s=0.3) as h:
            h.say("Hello there friend.")
            h.until(lambda: brain.committed)
            h.until(lambda: brain.ended, timeout=3)
        self.assertEqual(brain.ended[0], {"keep_history": False})


if __name__ == "__main__":
    unittest.main()
