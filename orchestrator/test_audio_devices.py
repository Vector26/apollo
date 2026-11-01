"""Mic and speaker survive a cold boot: waiting, fallbacks, Bluetooth, levels, hot-unplug."""
import queue
import threading
import time
import unittest
from unittest.mock import patch

import audio_devices as ad

# layout of the real `wpctl status` on this machine
REAL = """PipeWire 'pipewire-0' [1.0.5, user@host, cookie:1032343943]
 └─ Clients:
        32. pipewire                            [1.0.5, user@host, pid:1807]

Audio
 ├─ Devices:
 │      64. USB2.0 Device                       [alsa:pcm]
 │  
 ├─ Sinks:
 │      36. Rembrandt Radeon High Definition Audio Controller (HDMI 0) [vol: 1.00]
 │  *  175. USB2.0 Device                       [vol: 1.00]
 │     199. Family 17h/19h HD Audio Controller  [vol: 1.00]
 │  
 ├─ Sink endpoints:
 │  
 ├─ Sources:
 │     174. Family 17h/19h HD Audio Controller  [vol: 1.00]
 │  *  182. USB2.0 Device                       [vol: 1.00]
 │  
 ├─ Source endpoints:
 │  
 └─ Streams:
"""
NO_USB = REAL.replace(" │  *  182. USB2.0 Device                       [vol: 1.00]\n", "").replace(
    " │     174. Family", " │  *  174. Family")
NO_SOURCES = REAL.replace(" │     174. Family 17h/19h HD Audio Controller  [vol: 1.00]\n", "").replace(
    " │  *  182. USB2.0 Device                       [vol: 1.00]\n", "")
BLANK = "PipeWire 'pipewire-0' [1.0.5]\n"
MICS = ["USB2.0 Device", "Family 17h/19h HD Audio Controller"]


class ParseTest(unittest.TestCase):
    def test_nodes_ids_and_default(self):
        srcs = ad.list_nodes(REAL, "sources")
        self.assertEqual([(n.id, n.default) for n in srcs], [("174", False), ("182", True)])
        self.assertEqual(ad.parse_wpctl_id(REAL, "USB2.0 Device", "sources"), "182")
        self.assertEqual(ad.parse_wpctl_id(REAL, "USB2.0 Device", "sinks"), "175")
        self.assertIsNone(ad.parse_wpctl_id(REAL, "Test Earbuds", "sinks"))

    def test_audio_up_needs_an_output_and_an_input(self):
        self.assertTrue(ad.audio_up(REAL))
        self.assertFalse(ad.audio_up(BLANK))
        self.assertFalse(ad.audio_up(NO_SOURCES))


class ChooseMicTest(unittest.TestCase):
    def test_the_configured_mic_wins(self):
        c = ad.choose_mic(REAL, MICS)
        self.assertEqual((c.name, c.id, c.how), ("USB2.0 Device", "182", "preferred"))

    def test_second_preference_when_the_first_is_missing(self):
        c = ad.choose_mic(NO_USB, MICS)
        self.assertEqual((c.id, c.how), ("174", "preferred"))

    def test_any_usb_mic_then_the_default_then_anything(self):
        usb = REAL.replace("182. USB2.0 Device", "182. Blue Yeti USB Mic")
        self.assertEqual(ad.choose_mic(usb, ["Nope"]).how, "usb")
        self.assertEqual(ad.choose_mic(NO_USB, ["Nope"]).how, "default")
        no_default = NO_USB.replace("*  174", "   174")
        self.assertEqual(ad.choose_mic(no_default, ["Nope"]).how, "any")

    def test_no_mic_at_all(self):
        self.assertIsNone(ad.choose_mic(NO_SOURCES, MICS))


class WaitForAudioTest(unittest.TestCase):
    def run_wait(self, reads, timeout=30.0):
        seq, now, calls = list(reads), [0.0], []

        def read():
            calls.append(1)
            return seq.pop(0) if len(seq) > 1 else seq[0]

        def sleep(s):
            now[0] += s

        return ad.wait_for_audio(MICS, timeout, poll_s=1.0, read=read, sleep=sleep, clock=lambda: now[0]), len(calls), now[0]

    def test_already_up_returns_at_once(self):
        (status, have), calls, waited = self.run_wait([REAL])
        self.assertTrue(have)
        self.assertEqual((calls, waited), (1, 0.0))

    def test_waits_for_pipewire_then_the_mic_to_appear(self):
        (status, have), calls, waited = self.run_wait([BLANK, BLANK, NO_USB, NO_USB, REAL])
        self.assertTrue(have)
        self.assertEqual(status, REAL)
        self.assertGreaterEqual(waited, 4.0)

    def test_gives_up_at_the_timeout_and_never_hangs(self):
        (status, have), calls, waited = self.run_wait([BLANK], timeout=5.0)
        self.assertFalse(have)
        self.assertLessEqual(waited, 6.0)

    def test_a_missing_preferred_mic_returns_the_status_for_fallback_after_the_timeout(self):
        gone = NO_USB.replace("Family 17h/19h HD Audio Controller  [", "Other Mic  [")
        (status, have), _, waited = self.run_wait([gone], timeout=3.0)
        self.assertFalse(have)
        self.assertTrue(ad.audio_up(status))


class LevelsTest(unittest.TestCase):
    def run_levels(self, volume_line):
        calls = []

        def run(cmd, timeout=3.0):
            calls.append(cmd)
            return volume_line if cmd[1] == "get-volume" else ""

        return ad.ensure_audible("182", run=run), calls

    def test_a_muted_device_is_unmuted(self):
        changed, calls = self.run_levels("Volume: 0.60 [MUTED]")
        self.assertTrue(changed)
        self.assertIn(["wpctl", "set-mute", "182", "0"], calls)
        self.assertFalse(any(c[1] == "set-volume" for c in calls))  # the chosen level is left alone

    def test_a_zeroed_device_is_lifted(self):
        changed, calls = self.run_levels("Volume: 0.00")
        self.assertTrue(changed)
        self.assertIn(["wpctl", "set-volume", "182", "1.0"], calls)

    def test_a_normal_device_is_untouched(self):
        changed, calls = self.run_levels("Volume: 0.40")
        self.assertFalse(changed)
        self.assertEqual(len(calls), 1)


class BluetoothTest(unittest.TestCase):
    def make_run(self, connected, connect_out="Connection successful"):
        calls = []

        def run(cmd, timeout=5.0):
            calls.append(cmd)
            if cmd[:3] == ["bluetoothctl", "devices", "Paired"]:
                return "Device AA:BB:CC:DD:EE:01 Test Earbuds\nDevice 11:22:33:44:55:66 Other Thing\n"
            if cmd[1] == "info":
                return f"Name: x\n\tConnected: {'yes' if connected else 'no'}\n"
            if cmd[1] == "connect":
                return connect_out
            return ""

        return run, calls

    def test_a_paired_speaker_that_is_not_connected_gets_connected(self):
        run, calls = self.make_run(connected=False)
        self.assertEqual(ad.connect_bluetooth(["Test Earbuds", "Dash 4"], run=run), ["Test Earbuds"])
        self.assertIn(["bluetoothctl", "connect", "AA:BB:CC:DD:EE:01"], calls)

    def test_already_connected_is_left_alone(self):
        run, calls = self.make_run(connected=True)
        self.assertEqual(ad.connect_bluetooth(["Test Earbuds"], run=run), [])
        self.assertFalse(any(c[1] == "connect" for c in calls))

    def test_out_of_range_is_skipped_quietly(self):
        run, _ = self.make_run(connected=False, connect_out="Failed to connect: org.bluez.Error.Failed")
        self.assertEqual(ad.connect_bluetooth(["Test Earbuds"], run=run), [])

    def test_unpaired_names_are_never_connected(self):
        run, calls = self.make_run(connected=False)
        self.assertEqual(ad.connect_bluetooth(["Dash 4"], run=run), [])
        self.assertFalse(any(c[1] == "connect" for c in calls))


class EnsureTest(unittest.TestCase):
    CFG = {"bluetooth": {"speaker_names": ["Test Earbuds", "Family 17h/19h HD Audio Controller"],
                         "mic_name": "USB2.0 Device"}}

    def test_boot_summary_picks_mic_and_speaker_and_fixes_levels(self):
        with patch.object(ad, "read_status", return_value=REAL), \
             patch.object(ad, "connect_bluetooth", return_value=[]), \
             patch.object(ad, "ensure_audible", return_value=True):
            s = ad.ensure(self.CFG, wait_s=0.0)
        self.assertTrue(s["waited_for_audio"])
        self.assertEqual(s["mic"], {"name": "USB2.0 Device", "how": "preferred"})
        self.assertEqual(s["speaker"], "Family 17h/19h HD Audio Controller")
        self.assertEqual(len(s["fixed_levels"]), 2)

    def test_nothing_up_yet_is_a_summary_not_a_crash(self):
        with patch.object(ad, "read_status", return_value=BLANK), patch.object(ad, "connect_bluetooth", return_value=[]):
            s = ad.ensure(self.CFG, wait_s=0.0)
        self.assertFalse(s["waited_for_audio"])
        self.assertIsNone(s["mic"])

    def test_a_bluetooth_failure_never_fails_the_boot(self):
        with patch.object(ad, "read_status", return_value=REAL), \
             patch.object(ad, "connect_bluetooth", side_effect=OSError("no bluetoothctl")), \
             patch.object(ad, "ensure_audible", return_value=False):
            s = ad.ensure(self.CFG, wait_s=0.0)
        self.assertEqual(s["bluetooth_error"], "OSError")
        self.assertEqual(s["mic"]["name"], "USB2.0 Device")


class DeviceWatcherTest(unittest.TestCase):
    def make(self, statuses, how="preferred", current="182"):
        seq, defaults = list(statuses), []
        w = ad.DeviceWatcher(MICS, current, how, read=lambda: seq.pop(0) if len(seq) > 1 else seq[0],
                             set_default=defaults.append)
        return w, defaults

    def test_quiet_while_the_mic_is_fine(self):
        w, defaults = self.make([REAL])
        self.assertIsNone(w.step())
        self.assertEqual(defaults, [])

    def test_unplugging_the_mic_moves_to_the_next_best(self):
        w, defaults = self.make([NO_USB])
        with patch.object(ad, "ensure_audible"):
            msg = w.step()
        self.assertIn("mic lost", msg)
        self.assertEqual(defaults, ["174"])
        self.assertIsNone(w.step())  # said once

    def test_the_preferred_mic_coming_back_takes_over_again(self):
        w, defaults = self.make([REAL], how="fallback", current="174")
        # the fallback is still there, but the preferred one is back
        with patch.object(ad, "ensure_audible"):
            msg = w.step()
        self.assertIn("back on", msg)
        self.assertEqual(defaults, ["182"])

    def test_a_blank_read_never_moves_anything(self):
        w, defaults = self.make([BLANK])
        self.assertIsNone(w.step())
        self.assertEqual(defaults, [])

    def test_no_mic_at_all_is_said_once_then_picked_up_when_plugged_in(self):
        w, defaults = self.make([NO_SOURCES, NO_SOURCES, REAL], current=None, how="any")
        self.assertIn("no microphone", w.step())
        self.assertIsNone(w.step())
        with patch.object(ad, "ensure_audible"):
            msg = w.step()
        self.assertIn("USB2.0 Device", msg)
        self.assertEqual(defaults, ["182"])


class SpeakerRespawnTest(unittest.TestCase):
    def test_a_dead_player_is_restarted_on_wherever_audio_can_go_now(self):
        import speaker as speaker_mod

        spawned, changes = [], []

        class Proc:
            def __init__(self, cmd, **kw):
                spawned.append(cmd)
                self.dead = False
                self.stdin = type("S", (), {"write": lambda s, b: None, "flush": lambda s: None, "close": lambda s: None})()

            def poll(self):
                return 1 if self.dead else None

            def kill(self):
                self.dead = True

            def wait(self, timeout=None):
                return 0

        with patch.object(speaker_mod.subprocess, "Popen", Proc), \
             patch.object(speaker_mod, "sink_is_bluetooth", return_value=False), \
             patch.object(speaker_mod, "sink_node_name", side_effect=lambda t: f"node-{t}"):
            spk = speaker_mod.WarmSpeaker("199", preroll_ms=0, buffering="wired",
                                          resolve_target=lambda: None, on_change=changes.append)
            spk.RESPAWN_EVERY_S = 0.1
            spk.start()
            self.assertIn("--target", spawned[0])
            spk.proc.dead = True  # Bluetooth dropped / sink unplugged
            deadline = time.monotonic() + 3
            while len(spawned) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            spk.close()
        self.assertEqual(len(spawned), 2)
        self.assertNotIn("--target", spawned[1])  # fell back to the default output
        self.assertEqual(changes, [None])


class CaptureStallTest(unittest.TestCase):
    def test_a_mic_that_goes_silent_is_reported_not_waited_on_forever(self):
        import audio_io

        class DeadStream:
            def __init__(self, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False

        events = queue.Queue()
        with patch.object(audio_io.sd, "InputStream", DeadStream):
            cap = audio_io.ContinuousCapture(sample_rate=16000, capture_rate=16000, vad_threshold=0.5,
                                             event_queue=events, stop_event=threading.Event())
            cap.STALL_S = 0.4
            cap.start()
            ev = events.get(timeout=5)
            cap.stop_event.set()
        self.assertEqual(ev[0], "capture_error")


if __name__ == "__main__":
    unittest.main()
