"""speak_cli helpers (no GPU)."""

import argparse
import io
import sys
import unittest
from unittest.mock import patch

from speak_cli import _resolve_text


class ResolveTextTest(unittest.TestCase):
    def test_words(self):
        args = argparse.Namespace(text=None, file=None, words=["Hello", "world"])
        self.assertEqual(_resolve_text(args), "Hello world")

    def test_text_flag(self):
        args = argparse.Namespace(text="Hi", file=None, words=[])
        self.assertEqual(_resolve_text(args), "Hi")

    def test_stdin_when_piped(self):
        args = argparse.Namespace(text=None, file=None, words=[])
        with patch.object(sys.stdin, "isatty", return_value=False):
            with patch.object(sys.stdin, "read", return_value="piped\n"):
                self.assertEqual(_resolve_text(args), "piped\n")


class ApolloCliTest(unittest.TestCase):
    def test_voices_lists_profiles(self):
        from apollo_cli import cmd_voices_list
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_voices_list(argparse.Namespace())
        out = buf.getvalue()
        self.assertIn("jarvis", out)
        self.assertIn("ultron", out)


if __name__ == "__main__":
    unittest.main()
