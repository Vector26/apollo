"""Apollo honors the global halt flag: it won't start, and a running one stops by itself."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cli
import halt
from test_talk_flow import FakeBrain, Harness


class HaltFlagTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vox-halt-")
        p = patch.dict(os.environ, {"FORTRESS_HALT_STATE": self.tmp})
        p.start()
        self.addCleanup(p.stop)

    def flag(self) -> Path:
        return Path(self.tmp) / "HALT"

    def test_flag_path_follows_the_state_dir(self):
        self.assertEqual(halt.halt_flag(), self.flag())
        self.assertFalse(halt.is_halted())
        self.flag().write_text("x")
        self.assertTrue(halt.is_halted())

    def test_apollo_refuses_to_start_while_halted(self):
        self.flag().write_text("x")
        with self.assertRaises(SystemExit) as ctx:
            cli.cmd_talk({})
        self.assertEqual(ctx.exception.code, 1)

    def test_running_apollo_stops_itself_within_two_seconds(self):
        brain = FakeBrain()
        with Harness(brain) as h:
            self.assertTrue(h.thread.is_alive())
            self.flag().write_text("x")
            h.thread.join(3.0)
            self.assertFalse(h.thread.is_alive(), "Apollo kept running with the halt flag present")

    def test_a_halted_apollo_does_not_file_the_session_on_the_way_out(self):
        brain = FakeBrain()
        brain.session = True
        with Harness(brain) as h:
            self.flag().write_text("x")
            h.thread.join(3.0)
        self.assertEqual(brain.ended, [])  # vself is being stopped too: no 9 second hang


if __name__ == "__main__":
    unittest.main()
