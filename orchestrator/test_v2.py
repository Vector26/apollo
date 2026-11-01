"""Vox/Apollo 2.0: acks, echo gate, stall instrumentation, version, and the DUM-E local-only guard."""

import os
import unittest
from pathlib import Path

import yaml

import speaker
import stutter_report
from acks import Acks
from cli import log_clause_timing
from echo_gate import EchoGate
from version import VERSION_FILE, version

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = Path(os.environ.get("APOLLO_TEST_REGISTRY", "/nonexistent/registry.yaml"))


class AcksTest(unittest.TestCase):
    def test_disabled_gives_nothing(self):
        self.assertIsNone(Acks({"enabled": False}).pick("check the gpu status"))
        self.assertIsNone(Acks().pick("check the gpu status"))

    def test_ops_and_deep_get_a_line_small_talk_does_not(self):
        a = Acks({"enabled": True})
        self.assertIn(a.pick("check the gpu status"), a.all_lines())
        self.assertIn(a.pick("what is your opinion on this design"), a.all_lines())
        self.assertIsNone(a.pick("how are you today"))
        self.assertIsNone(a.pick("thanks that was great"))
        self.assertIsNone(a.pick(None))
        self.assertIsNone(a.pick(""))

    def test_lines_rotate_and_never_repeat_back_to_back(self):
        a = Acks({"enabled": True})
        picks = [a.pick("check the disk") for _ in range(6)]
        self.assertTrue(all(x != y for x, y in zip(picks, picks[1:])))
        self.assertGreater(len(set(picks)), 1)

    def test_config_overrides_lines_and_delay(self):
        a = Acks({"enabled": True, "after_ms": 700, "lines": {"ops": ["Right away."]}})
        self.assertEqual(a.after_s, 0.7)
        self.assertEqual(a.pick("restart the container"), "Right away.")
        self.assertIn("Right away.", a.all_lines())


class EchoGateTest(unittest.TestCase):
    def setUp(self):
        self.t = 100.0
        self.g = EchoGate(clock=lambda: self.t)

    def test_own_voice_coming_back_is_dropped(self):
        self.g.heard("The GPU is at forty percent and the disk is fine.")
        self.assertTrue(self.g.is_echo("the gpu is at forty percent and the disk is fine"))
        self.assertTrue(self.g.is_echo("gpu is at forty percent and the disk is"))  # garbled edges

    def test_a_reply_that_reuses_words_is_kept(self):
        self.g.heard("The plan is to rebuild the image and restart the service.")
        self.assertFalse(self.g.is_echo("what is the plan"))
        self.assertFalse(self.g.is_echo("restart the service and the image please"))

    def test_short_utterances_are_never_dropped(self):
        self.g.heard("Yes. Stop.")
        self.assertFalse(self.g.is_echo("yes stop"))
        self.assertFalse(self.g.is_echo("stop"))

    def test_expires_after_the_window(self):
        self.g.heard("The GPU is at forty percent and the disk is fine.")
        self.t += self.g.window_s + 1
        self.assertFalse(self.g.is_echo("the gpu is at forty percent and the disk is fine"))

    def test_nothing_spoken_means_nothing_dropped(self):
        self.assertFalse(self.g.is_echo("the gpu is at forty percent and the disk is fine"))


class GapTest(unittest.TestCase):
    def test_gap_inside_speech_is_reported(self):
        self.assertEqual(speaker.voice_gap_ms(10.4, 10.0), 400)

    def test_first_audio_jitter_and_new_utterances_are_not_gaps(self):
        self.assertIsNone(speaker.voice_gap_ms(10.0, 0.0))
        self.assertIsNone(speaker.voice_gap_ms(10.02, 10.0))
        self.assertIsNone(speaker.voice_gap_ms(10.0, 10.0))
        self.assertIsNone(speaker.voice_gap_ms(20.0, 10.0))
        self.assertIsNone(speaker.voice_gap_ms(9.0, 10.0))  # chunk queued ahead of playback: no gap


class StutterReportTest(unittest.TestCase):
    LOG = [
        "2026-11-01 14:30:11,358 [INFO] voice.rvc: Loading RVC Ultron on cuda:0",
        "2026-11-01 14:30:15,987 [INFO] speaker: WarmSpeaker started target=199",
        "2026-11-01 14:31:00,000 [INFO] cli: clause synth 1800ms for audio 1200ms (rtf 1.50)",
        "2026-11-01 14:31:02,000 [INFO] speaker: speaker gap 650ms between chunks",
        "2026-11-01 14:31:03,000 [INFO] cli: clause synth 600ms for audio 1500ms (rtf 0.40)",
        "2026-11-01 14:47:45,803 [INFO] speaker: WarmSpeaker started target=199",
        "2026-11-01 14:48:00,000 [INFO] cli: clause synth 300ms for audio 1500ms (rtf 0.20)",
    ]

    def test_sessions_split_on_speaker_start_and_carry_rvc(self):
        a, b = stutter_report.parse(self.LOG)
        self.assertTrue(a.rvc)
        self.assertFalse(b.rvc)
        self.assertEqual(a.synth_ms, [1800, 600])
        self.assertEqual(a.gaps_ms, [650])
        self.assertEqual(b.synth_ms, [300])

    def test_reading_names_the_bottleneck(self):
        a, b = stutter_report.parse(self.LOG)
        self.assertIn("TTS/RVC is the bottleneck", stutter_report.summarize(a))
        self.assertIn("no audible mid-reply stalls", stutter_report.summarize(b))

    def test_stalls_with_fast_synthesis_point_at_the_brain(self):
        s = stutter_report.Session(synth_ms=[300], audio_ms=[1500], rtf=[0.2], gaps_ms=[900])
        self.assertIn("token cadence", stutter_report.summarize(s))

    def test_clause_timing_line_round_trips(self):
        import logging
        import numpy as np
        with self.assertLogs(level=logging.INFO) as cm:
            log_clause_timing(0.5, np.zeros(16000, dtype=np.float32), 16000)
        s, = stutter_report.parse(["x WarmSpeaker started"] + [r.getMessage() for r in cm.records])
        self.assertEqual((s.synth_ms, s.audio_ms, s.rtf), ([500], [1000], [0.5]))


class VersionTest(unittest.TestCase):
    def test_major_bump_and_single_source(self):
        self.assertEqual(VERSION_FILE, ROOT / "VERSION")
        self.assertRegex(version(), r"^2\.\d+\.\d+$")


class DumeStaysLocalTest(unittest.TestCase):
    """DUM-E must only ever use the local model server. 2.0 does not touch it; this keeps it so."""

    @unittest.skipUnless(REGISTRY.exists(), "agent registry not present")
    def test_registry_routes_dume_to_local_endpoints(self):
        reg = yaml.safe_load(REGISTRY.read_text())
        ep = reg["llm_endpoints"]
        for key in ("pi_agents_dume_oda", "brain", "gateway_adapters", "model_queue"):
            url = ep.get(key) or reg[key]["url"]
            self.assertRegex(url, r"^https?://(127\.0\.0\.1|localhost)[:/]", key)
        dume = next(a for a in reg["agents"] if a["id"] == "dume")
        self.assertEqual(dume["uses_brain"], "hauhaucs")  # the local model; never a cloud brain

    def test_dume_adapter_and_tools_do_not_reference_cloud_brains(self):
        src = (Path(__file__).parent / "agent_tools.py").read_text()
        self.assertNotRegex(src, r"openrouter|api\.openai|anthropic|composer", "agent_tools must stay local-only")


if __name__ == "__main__":
    unittest.main()
