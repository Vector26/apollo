"""CLI helper tests (no audio / no LLM)."""

import inspect
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from cli import _run_talk, cmd_talk, is_dump_request, is_exit, parse_wpctl_id, vu_bar
from speaker import resample_s16, silence_s16

WPCTL = """
Audio
 ├─ Devices:
 │      49. USB Composite Device                [alsa]
 │      52. Dash 4                              [bluez5]
 ├─ Sinks:
 │  *   53. Dash 4                              [vol: 0.70]
 ├─ Sources:
 │  *   50. USB Composite Device Mono           [vol: 1.00]
"""


class VuBarTest(unittest.TestCase):
    def test_empty_and_full(self):
        self.assertEqual(vu_bar(0, 4), "░░░░")
        self.assertEqual(vu_bar(1, 4), "████")
        self.assertEqual(len(vu_bar(0.5, 10)), 10)


WPCTL_AUX = """
PipeWire 'pipewire-0' [1.0.5, user@host, cookie:1]
 └─ Clients:
        33. Family 17h/19h HD Audio Controller fake client

Audio
 ├─ Devices:
 │      48. Family 17h/19h HD Audio Controller  [alsa:pcm]
 │     197. USB2.0 Device                       [alsa:pcm]
 │  
 ├─ Sinks:
 │  *   33. Family 17h/19h HD Audio Controller  [vol: 1.00]
 │      53. Rembrandt Radeon High Definition Audio Controller (HDMI 0) [vol: 1.00]
 │     198. USB2.0 Device                       [vol: 1.00]
 │  
 ├─ Sink endpoints:
 │  
 ├─ Sources:
 │      40. Family 17h/19h HD Audio Controller (ALC897 Alt Analog) [vol: 1.00]
 │  *   42. Family 17h/19h HD Audio Controller  [vol: 1.00]
 │     200. USB2.0 Device                       [vol: 1.00]
 │  
 └─ Streams:

Video
 ├─ Sources:
 │  *   99. USB2.0 Device                       [v4l2]
"""


class WpctlParseTest(unittest.TestCase):
    def test_prefers_starred_sink(self):
        self.assertEqual(parse_wpctl_id(WPCTL, "Dash 4"), "53")

    def test_mic_source(self):
        self.assertEqual(parse_wpctl_id(WPCTL, "USB Composite Device", "sources"), "50")

    def test_missing(self):
        self.assertIsNone(parse_wpctl_id(WPCTL, "Nope"))

    def test_aux_sink_not_onboard_source(self):
        self.assertEqual(parse_wpctl_id(WPCTL_AUX, "Family 17h/19h HD Audio Controller"), "33")

    def test_onboard_source_exact_name_wins(self):
        self.assertEqual(
            parse_wpctl_id(WPCTL_AUX, "Family 17h/19h HD Audio Controller", "sources"), "42"
        )

    def test_usb_mic_source_not_sink_or_video(self):
        self.assertEqual(parse_wpctl_id(WPCTL_AUX, "USB2.0 Device", "sources"), "200")
        self.assertEqual(parse_wpctl_id(WPCTL_AUX, "USB2.0 Device", "sinks"), "198")


class TalkVoiceWireTest(unittest.TestCase):
    def test_run_talk_declares_voice(self):
        sig = inspect.signature(_run_talk)
        self.assertIn("voice", sig.parameters)

    @patch("cli._run_talk")
    @patch("cli.TalkUI")
    def test_cmd_talk_forwards_voice(self, mock_ui_cls, mock_run_talk):
        ui = MagicMock()
        mock_ui_cls.return_value = ui
        mock_run_talk.side_effect = KeyboardInterrupt()

        cfg = {
            "llm": {"base_url": "http://127.0.0.1:1/v1", "model": "x", "system_prompt": "hi"},
            "notes": {"output_dir": "/tmp/vox-notes-test"},
            "memory": {"dir": "/tmp/vox-mem-test"},
            "bluetooth": {"speaker_name": "Dash 4", "mic_name": "USB Composite Device"},
            "audio": {"sample_rate": 16000, "vad_threshold": 0.5},
            "stt": {},
            "tts": {"profile": "jarvis"},
        }
        # cmd_talk attaches a log file to the root logger: keep it (and the rest of the suite's
        # log lines) out of the real talk.log
        tmp = tempfile.mkdtemp(prefix="vox-log-test-")
        root = logging.getLogger()
        before = list(root.handlers)
        self.addCleanup(lambda: [root.removeHandler(h) for h in root.handlers if h not in before])
        self.addCleanup(shutil.rmtree, tmp, True)
        with patch.dict(os.environ, {"VOX_LOG_DIR": tmp}):
            cmd_talk(cfg, voice="ultron")
        self.assertTrue(os.path.exists(os.path.join(tmp, "talk.log")))
        mock_run_talk.assert_called_once()
        _args, kwargs = mock_run_talk.call_args
        self.assertEqual(kwargs.get("voice"), "ultron")
        self.assertEqual(_args[0], cfg)
        self.assertIs(_args[1], ui)
        self.assertIsNotNone(_args[2])  # stop Event


class GoodbyeTest(unittest.TestCase):
    def test_goodbye_variants_end_the_session(self):
        for text in ("Goodbye.", "Bye.", "Bye-bye.", "bye bye!", "Okay, good night.", "Good night, Prometheus.",
                     "See you later.", "That\u2019s all.", "Alright, I'm done.", "Stop", "Exit."):
            self.assertTrue(is_exit(text), text)

    def test_a_sentence_that_mentions_bye_does_not(self):
        for text in ("Say bye to the guests for me", "I don't want to stop thinking about it",
                     "That's all well and good but why?", "Goodbye is a hard word to say"):
            self.assertFalse(is_exit(text), text)

    def test_notes_request_only_when_said_on_its_own(self):
        for text in ("Take notes.", "Please save the notes now.", "Okay, write notes"):
            self.assertTrue(is_dump_request(text), text)
        for text in ("Can you take notes on this idea?", "take a note that I need milk", "notes are boring"):
            self.assertFalse(is_dump_request(text), text)


class SpeakerDspTest(unittest.TestCase):
    def test_preroll_is_half_second(self):
        self.assertEqual(silence_s16(500, 48000).shape[0], 24000)

    def test_resample_doubles(self):
        x = np.arange(100, dtype=np.int16)
        y = resample_s16(x, 24000, 48000)
        self.assertGreater(len(y), 180)
        self.assertLess(len(y), 220)


if __name__ == "__main__":
    unittest.main()
