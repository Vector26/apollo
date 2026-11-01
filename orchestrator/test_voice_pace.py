"""Pacing helpers."""

import unittest

import numpy as np

from voice.pace import apply_speaking_rate, pace_text
from voice.presets import load_profile


class PaceTest(unittest.TestCase):
    def test_measured_inserts_pauses(self):
        out = pace_text("Hello, world. Why?", "measured")
        self.assertIn("…", out)

    def test_slower_longer(self):
        sr = 24000
        x = np.sin(np.linspace(0, 8, sr, dtype=np.float32))
        slow, _ = apply_speaking_rate(x, sr, 0.5)
        self.assertGreater(len(slow), len(x))

    def test_ultron_profile_rate(self):
        p = load_profile("ultron")
        self.assertLess(p.speaking_rate, 1.0)
        self.assertEqual(p.text_pace, "measured")


if __name__ == "__main__":
    unittest.main()
