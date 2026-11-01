"""Linux setup for Jieli / Digitek USB wireless receivers (4c4a:4155)."""

from __future__ import annotations

import logging
import re
import signal
import subprocess
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

JIELI_USB_ID = "4c4a:4155"


def _card_index() -> int | None:
    try:
        out = subprocess.check_output(["/usr/bin/arecord", "-l"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "USB Composite Device" in line or "Jieli" in line:
            if line.strip().startswith("card "):
                return int(line.split(":", 1)[0].split()[1])
    return None


def _pipewire_usb_device_id() -> str | None:
    try:
        out = subprocess.check_output(["wpctl", "status"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "alsa:pcm" not in line or "USB Composite Device" not in line:
            continue
        m = re.search(r"(\d+)\.\s+USB Composite Device", line)
        if m:
            return m.group(1)
    return None


def _jieli_pw_node_name() -> str | None:
    try:
        out = subprocess.check_output(["pw-cli", "list-objects"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    mono: str | None = None
    any_cap: str | None = None
    for line in out.splitlines():
        if "node.name" not in line or "alsa_input.usb-Jieli" not in line:
            continue
        name = line.split("=", 1)[1].strip().strip('"')
        if "mono-fallback" in name:
            mono = name
        elif any_cap is None:
            any_cap = name
    return mono or any_cap


def _pipewire_usb_source_id() -> str | None:
    """wpctl source id for Jieli capture (Mono or capture.0.0)."""
    try:
        status = subprocess.check_output(["wpctl", "status"], text=True, errors="replace")
    except (OSError, subprocess.CalledProcessError):
        return None
    best: tuple[int, str] | None = None
    for line in status.splitlines():
        if "USB Composite Device" not in line:
            continue
        if "Family 17h" in line or "ALC897" in line or "alsa:pcm" in line:
            continue
        m = re.search(r"(\d+)\.\s+(.+)", line.strip().lstrip("│* "))
        if not m:
            continue
        nid, label = m.group(1), m.group(2)
        if "USB Composite Device" not in label:
            continue
        score = (10 if "Mono" in label else 0) + (5 if "*" in line else 0)
        if best is None or score > best[0]:
            best = (score, nid)
    return best[1] if best else None


def _pw_record_target(source: str | None) -> str:
    if source:
        return source
    name = _jieli_pw_node_name()
    return name or "@DEFAULT_AUDIO_SOURCE@"


def _wpctl(args: list[str]) -> bool:
    try:
        subprocess.run(["wpctl", *args], check=False, capture_output=True, text=True)
        return True
    except OSError:
        return False


def _amixer(card: int, args: list[str]) -> bool:
    try:
        subprocess.run(
            ["amixer", "-c", str(card), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return True
    except OSError as e:
        logger.warning("amixer failed: %s", e)
        return False


def _record_peak(
    seconds: float = 1.5,
    rate: int = 48000,
    target: str | None = None,
) -> float:
    """Peak 0..1 from a short PipeWire capture (pw-record must be stopped with INT)."""
    import numpy as np

    tgt = _pw_record_target(target)
    path = Path(tempfile.mktemp(suffix=".wav"))
    try:
        proc = subprocess.Popen(
            [
                "pw-record",
                "--target",
                str(tgt),
                "--rate",
                str(rate),
                "--channels",
                "1",
                "--format",
                "s16",
                str(path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=seconds + 2)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=3)
        if not path.is_file() or path.stat().st_size < 44:
            return 0.0
        with wave.open(str(path), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if pcm.size == 0:
            return 0.0
        return float(np.max(np.abs(pcm))) / 32768.0
    except (OSError, wave.Error):
        return 0.0
    finally:
        path.unlink(missing_ok=True)


def calibrate_alsa_mic(card: int, pw_target: str | None, seconds: float = 1.2) -> dict:
    """
    Try INV_BOOLEAN capture switch + AGC both ways; pick the loudest (speak during this).
    """
    _amixer(card, ["cset", "numid=3", "147"])
    tried: list[dict] = []
    best_switch, best_agc, best_peak = "on", "on", -1.0
    for switch in ("on", "off"):
        for agc in ("on", "off"):
            _amixer(card, ["cset", "numid=2", switch])
            _amixer(card, ["cset", "numid=4", agc])
            peak = _record_peak(seconds, target=pw_target or "@DEFAULT_AUDIO_SOURCE@")
            row = {"switch": switch, "agc": agc, "peak": peak}
            tried.append(row)
            if peak > best_peak:
                best_peak = peak
                best_switch, best_agc = switch, agc
    _amixer(card, ["cset", "numid=2", best_switch])
    _amixer(card, ["cset", "numid=4", best_agc])
    return {
        "best_switch": best_switch,
        "best_agc": best_agc,
        "best_peak": best_peak,
        "tried": tried,
    }


def apply_usb_mic_linux_fixes(card: int | None = None, calibrate: bool = False) -> dict:
    report: dict = {"card": None, "steps": []}
    if card is None:
        card = _card_index()
    if card is None:
        report["steps"].append("skip:no_usb_mic_card")
        return report
    report["card"] = card

    src = _pipewire_usb_source_id()
    if src:
        _wpctl(["set-mute", src, "0"])
        _wpctl(["set-volume", src, "1.0"])
        _wpctl(["set-default", src])
        report["pipewire_source"] = src
        report["steps"].append("pipewire_unmute")

    _amixer(card, ["cset", "numid=3", "147"])
    report["steps"].append("volume_max")

    # Jieli INV_BOOLEAN: empirically "on"+"on" often passes RF; calibrate when asked.
    _amixer(card, ["cset", "numid=2", "on"])
    _amixer(card, ["cset", "numid=4", "on"])
    report["steps"].append("alsa_switch_on_agc_on")

    if calibrate and src:
        report["calibration"] = calibrate_alsa_mic(card, src)
        report["steps"].append("alsa_calibrated")

    for dev in Path("/sys/bus/usb/devices").glob("*"):
        try:
            vid = (dev / "idVendor").read_text().strip()
            pid = (dev / "idProduct").read_text().strip()
        except OSError:
            continue
        if f"{vid}:{pid}" != JIELI_USB_ID:
            continue
        ctrl = dev / "power" / "control"
        if ctrl.exists():
            try:
                ctrl.write_text("on\n")
                report["steps"].append(f"usb_autosuspend_off:{dev.name}")
            except OSError:
                report["steps"].append("usb_power:permission_denied")

    logger.info("usb_mic_linux fixes: %s", report)
    return report
