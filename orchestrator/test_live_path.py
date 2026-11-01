"""Live-path tests: non-blocking speaker, brain selection, API key loading.

No GPU, audio device, or network required.
"""

import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import speaker as speaker_mod
from cli import talk_llm_config
from llm_client import LLMClient, read_env_var
from speaker import PLAY_RATE, WarmSpeaker
from voice.presets import load_profile


class _FakeStdin:
    def __init__(self):
        self.bytes = 0
        self.lock = threading.Lock()

    def write(self, b):
        with self.lock:
            self.bytes += len(b)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProc:
    def __init__(self, *a, **k):
        self.stdin = _FakeStdin()

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def _fake_speaker() -> WarmSpeaker:
    with patch.object(speaker_mod.subprocess, "Popen", _FakeProc):
        spk = WarmSpeaker("33", preroll_ms=0)
        spk.start()
    return spk


class WarmSpeakerQueueTest(unittest.TestCase):
    def test_write_returns_immediately_and_drain_waits_for_playback(self):
        spk = _fake_speaker()
        try:
            t0 = time.monotonic()
            spk.write_pcm(np.zeros(24000, dtype=np.float32), 24000)  # 1.0 s
            self.assertLess(time.monotonic() - t0, 0.05)
            self.assertTrue(spk.drain(timeout=5))
            self.assertGreater(time.monotonic() - t0, 0.9)
        finally:
            spk.close()

    def test_paced_not_dumped(self):
        spk = _fake_speaker()
        try:
            before = spk.proc.stdin.bytes
            spk.write_pcm(np.zeros(PLAY_RATE * 2, dtype=np.int16), PLAY_RATE)  # 2.0 s
            time.sleep(0.3)
            sent_s = (spk.proc.stdin.bytes - before) / 2 / PLAY_RATE
            self.assertLess(sent_s, 0.9)  # ~0.3 s played + 0.2 s lead
        finally:
            spk.close()

    def test_on_start_fires_and_flush_drops_queued(self):
        spk = _fake_speaker()
        fired = []
        try:
            spk.write_pcm(np.zeros(PLAY_RATE, dtype=np.int16), PLAY_RATE, on_start=lambda: fired.append("a"))
            spk.write_pcm(np.zeros(PLAY_RATE, dtype=np.int16), PLAY_RATE, on_start=lambda: fired.append("b"))
            time.sleep(0.3)
            spk.flush()
            t0 = time.monotonic()
            self.assertTrue(spk.drain(timeout=3))
            self.assertLess(time.monotonic() - t0, 0.5)  # flushed audio is not waited on
            time.sleep(0.3)
            self.assertEqual(fired, ["a"])
        finally:
            spk.close()


class WarmSpeakerEchoGateTest(unittest.TestCase):
    def test_voice_active_spans_playback_plus_tail(self):
        spk = _fake_speaker()
        try:
            time.sleep(0.1)
            self.assertFalse(spk.voice_active(0.0))  # keepalive silence is not voice
            spk.write_pcm(np.zeros(PLAY_RATE // 2, dtype=np.int16), PLAY_RATE)  # 0.5 s
            self.assertTrue(spk.voice_active(0.0))
            time.sleep(0.75)
            self.assertFalse(spk.voice_active(0.0))
            self.assertTrue(spk.voice_active(0.5))
        finally:
            spk.close()

    def test_after_playback_fires_when_audio_ends(self):
        spk = _fake_speaker()
        done = threading.Event()
        try:
            t0 = time.monotonic()
            spk.write_pcm(np.zeros(PLAY_RATE // 2, dtype=np.int16), PLAY_RATE)
            spk.after_playback(done.set)
            self.assertTrue(done.wait(2))
            self.assertGreater(time.monotonic() - t0, 0.45)
        finally:
            spk.close()

    def test_after_playback_dropped_by_flush(self):
        spk = _fake_speaker()
        done = threading.Event()
        try:
            spk.write_pcm(np.zeros(PLAY_RATE, dtype=np.int16), PLAY_RATE)
            spk.after_playback(done.set)
            time.sleep(0.1)
            spk.flush()
            self.assertFalse(done.wait(1.3))
        finally:
            spk.close()


class SpeakerBufferingTest(unittest.TestCase):
    def test_bluetooth_sink_gets_deep_buffer(self):
        import speaker
        with patch.object(speaker.subprocess, "run",
                               return_value=SimpleNamespace(stdout='node.name = "bluez_output.B0_38.1"')):
            spk = WarmSpeaker("184", preroll_ms=0)
        self.assertEqual(spk.buffering, "bluetooth")
        self.assertGreaterEqual(spk.lead_s, 0.5)
        self.assertGreaterEqual(spk.latency_ms, 200)

    def test_numeric_sink_is_targeted_by_node_name(self):
        import speaker
        inspect = SimpleNamespace(stdout='  * node.name = "alsa_output.pci-0000_11_00.6.playback.0.0"\n')
        with patch.object(speaker.subprocess, "run", return_value=inspect), \
                patch.object(speaker.subprocess, "Popen") as popen:
            spk = WarmSpeaker("199", preroll_ms=0)
            spk.start()
            spk.close()
        cmd = popen.call_args[0][0]
        self.assertEqual(cmd[cmd.index("--target") + 1], "alsa_output.pci-0000_11_00.6.playback.0.0")

    def test_wired_sink_keeps_low_latency(self):
        import speaker
        with patch.object(speaker.subprocess, "run",
                               return_value=SimpleNamespace(stdout='node.name = "alsa_output.pci-0000"')):
            spk = WarmSpeaker("57", preroll_ms=0)
        self.assertEqual((spk.buffering, spk.latency_ms, spk.lead_s), ("wired", 60, 0.2))


class PickSpeakerTest(unittest.TestCase):
    CFG = {"bluetooth": {"speaker_names": ["realme Buds Air7", "Family 17h/19h HD Audio Controller"]}}
    AUX_ONLY = (
        "Audio\n ├─ Sinks:\n │  *  175. USB2.0 Device                       [vol: 1.00]\n"
        " │     199. Family 17h/19h HD Audio Controller  [vol: 1.00]\n │  \n"
        " ├─ Sources:\n │  *  200. USB2.0 Device                       [vol: 1.00]\n"
    )
    WITH_BUDS = AUX_ONLY.replace(" │     199.", " │     184. realme Buds Air7                    [vol: 0.80]\n │     199.")

    def pick(self, status):
        from cli import pick_speaker
        with patch("cli.subprocess.check_output", return_value=status):
            return pick_speaker(self.CFG)

    def test_buds_when_connected(self):
        self.assertEqual(self.pick(self.WITH_BUDS), ("realme Buds Air7", "184"))

    def test_dash4_aux_when_buds_are_not_connected(self):
        self.assertEqual(self.pick(self.AUX_ONLY), ("Family 17h/19h HD Audio Controller", "199"))

    def test_old_single_speaker_name_still_works(self):
        from cli import speaker_preferences
        self.assertEqual(speaker_preferences({"bluetooth": {"speaker_name": "Dash 4"}}), ["Dash 4"])


class SpokenLogTest(unittest.TestCase):
    def test_repeated_sentence_is_dropped_but_short_words_are_not(self):
        from cli import SpokenLog
        log = SpokenLog()
        log.record("Right. The packet grabber's still running. Sam's tweaking the feed next week—should help.")
        self.assertTrue(log.is_repeat("The packet grabber’s still running."))
        self.assertTrue(log.is_repeat("sam's tweaking the feed next week—"))
        self.assertFalse(log.is_repeat("Right."))
        self.assertFalse(log.is_repeat("The packet grabber is finished now."))


class ReadEnvVarTest(unittest.TestCase):
    def test_reads_file_without_env(self):
        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# c\nOTHER=1\nexport VOX_TEST_KEY='abc123'\n")
        try:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VOX_TEST_KEY", None)
                self.assertEqual(read_env_var("VOX_TEST_KEY", f.name), "abc123")
        finally:
            os.unlink(f.name)

    def test_env_wins_and_missing_is_none(self):
        with patch.dict(os.environ, {"VOX_TEST_KEY": "fromenv"}):
            self.assertEqual(read_env_var("VOX_TEST_KEY", "/nonexistent"), "fromenv")
        os.environ.pop("VOX_TEST_KEY", None)
        self.assertIsNone(read_env_var("VOX_TEST_KEY", "/nonexistent"))

    def test_from_config_fails_loudly_without_key(self):
        os.environ.pop("VOX_TEST_KEY", None)
        with self.assertRaises(RuntimeError):
            LLMClient.from_config({
                "base_url": "https://example.invalid/v1", "model": "m", "system_prompt": "s",
                "api_key_env": "VOX_TEST_KEY", "api_key_file": "/nonexistent",
            })


class TalkBrainTest(unittest.TestCase):
    CFG = {
        "llm": {"base_url": "http://127.0.0.1:8081/v1", "model": "local", "system_prompt": "generic",
                "max_tokens": 512},
        "brains": {"openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "remote",
                                  "max_tokens": 160},
                   "vself": {"kind": "vself", "api_url": "http://vself", "cloud": "openrouter"}},
    }

    def test_ultron_is_only_a_voice_on_the_vself_brain(self):
        profile = load_profile("ultron")
        self.assertIsNone(profile.persona)
        llm_cfg, prompt = talk_llm_config(self.CFG, profile)
        self.assertEqual(llm_cfg["kind"], "vself")
        self.assertNotIn("Ultron", prompt)

    def test_openrouter_brain_overrides_local_settings(self):
        prof = SimpleNamespace(profile_id="x", brain="openrouter", persona=None)
        llm_cfg, _ = talk_llm_config(self.CFG, prof)
        self.assertEqual(llm_cfg["model"], "remote")
        self.assertEqual(llm_cfg["max_tokens"], 160)

    def test_every_voice_talks_with_vself(self):
        for name in ("jarvis", "ultron", "robert_california"):
            llm_cfg, _ = talk_llm_config(self.CFG, load_profile(name))
            self.assertEqual(llm_cfg.get("kind"), "vself", name)

    def test_profile_without_brain_keeps_local_llm(self):
        prof = SimpleNamespace(profile_id="x", brain=None, persona=None)
        llm_cfg, prompt = talk_llm_config(self.CFG, prof)
        self.assertEqual(llm_cfg["model"], "local")
        self.assertEqual(prompt, "generic")

    def test_missing_brain_is_clear_error(self):
        prof = SimpleNamespace(profile_id="x", brain="nope", persona=None)
        with self.assertRaises(KeyError):
            talk_llm_config(self.CFG, prof)


if __name__ == "__main__":
    unittest.main()
