import unittest

from voice_interrupt import is_interrupt_command


class InterruptKeywordTest(unittest.TestCase):
    def test_yes(self):
        for t in (
            "wait",
            "stop",
            "stop talking",
            "hold on",
            "listen to me",
            "hang on",
            "pause",
            "enough",
            "okay stop",
            "let me speak",
        ):
            self.assertTrue(is_interrupt_command(t), t)

    def test_no(self):
        for t in ("what are my plane tickets", "wait for the build", "stop the detective job", "hello"):
            self.assertFalse(is_interrupt_command(t), t)


if __name__ == "__main__":
    unittest.main()
