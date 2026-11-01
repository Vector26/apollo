"""usb_mic_linux helpers (no hardware)."""

import unittest
from unittest.mock import patch

from usb_mic_linux import _card_index, _pipewire_usb_source_id, apply_usb_mic_linux_fixes


class CardIndexTest(unittest.TestCase):
    @patch("usb_mic_linux.subprocess.check_output")
    def test_finds_usb_card(self, mock_out):
        mock_out.return_value = (
            "card 2: Generic [HD-Audio Generic]\n"
            "card 3: Device [USB Composite Device]\n"
        )
        self.assertEqual(_card_index(), 3)


class PipewireParseTest(unittest.TestCase):
    @patch("usb_mic_linux.subprocess.check_output")
    def test_finds_usb_source_id_not_device_row(self, mock_out):
        mock_out.return_value = (
            " │      49. USB Composite Device                [alsa]\n"
            " │  *   50. USB Composite Device Mono           [vol: 1.00 MUTED]\n"
        )
        self.assertEqual(_pipewire_usb_source_id(), "50")

    @patch("usb_mic_linux.subprocess.check_output")
    def test_finds_capture_node_without_mono_label(self, mock_out):
        mock_out.return_value = (
            " │      41. USB Composite Device                [alsa:pcm]\n"
            " │  *   48. USB Composite Device                [vol: 1.00]\n"
        )
        self.assertEqual(_pipewire_usb_source_id(), "48")


class ApplyFixesTest(unittest.TestCase):
    @patch("usb_mic_linux._card_index", return_value=None)
    def test_no_card_skips(self, _):
        r = apply_usb_mic_linux_fixes()
        self.assertIn("skip:no_usb_mic_card", r["steps"])

    @patch("usb_mic_linux._pipewire_usb_source_id", return_value="50")
    @patch("usb_mic_linux._card_index", return_value=3)
    @patch("usb_mic_linux._amixer", return_value=True)
    @patch("usb_mic_linux._wpctl", return_value=True)
    def test_apply_sets_alsa_on(self, *_mocks):
        r = apply_usb_mic_linux_fixes()
        self.assertIn("alsa_switch_on_agc_on", r["steps"])
        self.assertEqual(r.get("pipewire_source"), "50")


if __name__ == "__main__":
    unittest.main()
