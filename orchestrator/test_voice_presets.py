"""Voice preset resolution (no GPU)."""

import unittest
from pathlib import Path

from voice.presets import VOX_ROOT, get_fetch_spec, list_profiles, load_profile


class VoicePresetsTest(unittest.TestCase):
    def test_list_profiles(self):
        names = list_profiles()
        self.assertIn("jarvis", names)
        self.assertIn("ultron", names)

    @unittest.skipUnless((Path(__file__).resolve().parent.parent / "models").is_dir(),
                         "needs local model weights and a reference clip (not shipped)")
    def test_jarvis_paths(self):
        p = load_profile("jarvis")
        self.assertTrue(p.ref_audio.is_file(), p.ref_audio)
        self.assertTrue(p.model_dir.is_dir(), p.model_dir)
        self.assertFalse(p.rvc_enabled)

    def test_ultron_rvc_dir(self):
        p = load_profile("ultron")
        self.assertTrue(p.rvc_enabled)
        self.assertIsNotNone(p.rvc)
        assert p.rvc is not None
        self.assertEqual(p.rvc.model_dir, (VOX_ROOT / "models/rvc/ultron").resolve())

    def test_fetch_spec(self):
        spec = get_fetch_spec("ultron")
        self.assertIn("primary_url", spec)
        self.assertTrue(str(spec["dest_dir"]).endswith("ultron"))


if __name__ == "__main__":
    unittest.main()
