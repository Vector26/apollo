"""STT engine selection: Parakeet vs Whisper."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from stt_stream import ParakeetSTT, StreamingSTT, _filter_transcript, _too_quiet, build_stt


class QuietTest(unittest.TestCase):
    def test_silence_never_hits_the_model(self):
        stt = ParakeetSTT.__new__(ParakeetSTT)
        stt.model = SimpleNamespace(recognize=lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")))
        self.assertEqual(stt.transcribe(np.zeros(32000, dtype=np.float32)), "")


class FilterTest(unittest.TestCase):
    def test_noise_phrases_become_empty(self):
        self.assertEqual(_filter_transcript("Thanks for watching!"), "")
        self.assertEqual(_filter_transcript("What time is it?"), "What time is it?")


class BuildSttTest(unittest.TestCase):
    def test_whisper_is_default(self):
        with patch.object(StreamingSTT, "__init__", return_value=None) as init:
            build_stt({})
            init.assert_called_once()

    def test_parakeet_engine(self):
        with patch.object(ParakeetSTT, "__init__", return_value=None) as init:
            build_stt({"engine": "parakeet", "model": "nemo-parakeet-tdt-0.6b-v3", "threads": 2})
            init.assert_called_once()
            _, kw = init.call_args
            self.assertEqual(kw["model"], "nemo-parakeet-tdt-0.6b-v3")
            self.assertEqual(kw["threads"], 2)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            build_stt({"engine": "vosk"})


class ParakeetTranscribeTest(unittest.TestCase):
    def test_empty_pcm_is_empty(self):
        stt = ParakeetSTT.__new__(ParakeetSTT)
        stt.model = SimpleNamespace(recognize=lambda *a, **k: "hello")
        self.assertEqual(stt.transcribe(np.zeros(0, dtype=np.float32)), "")

    def test_recognize_output_is_filtered(self):
        stt = ParakeetSTT.__new__(ParakeetSTT)
        stt.model = SimpleNamespace(recognize=lambda *a, **k: "you")
        self.assertEqual(stt.transcribe(np.ones(1600, dtype=np.float32)), "")

    def test_beam_size_is_ignored(self):
        stt = ParakeetSTT.__new__(ParakeetSTT)
        stt.model = SimpleNamespace(recognize=lambda *a, **k: "check the gpu")
        self.assertEqual(stt.transcribe(np.ones(1600, dtype=np.float32), beam_size=5), "check the gpu")
