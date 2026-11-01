#!/usr/bin/env python3
"""Offline speech generation (Apollo merged into Vox)."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import yaml

from voice.fetch_models import fetch_ultron
from voice.pipeline import VoicePipeline
from voice.presets import VOX_ROOT, list_profiles, load_profile

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"

_HARD_STOP = re.compile(r"[.!?][\"')\]]*(\s|$)")


def _load_cfg(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if parts:
        return parts
    return [text.strip()] if text.strip() else []


def _split_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        buf += (" " if buf else "") + line
        while True:
            m = _HARD_STOP.search(buf)
            if not m:
                break
            chunks.append(buf[: m.end()].strip())
            buf = buf[m.end() :].strip()
    if buf:
        chunks.append(buf)
    return chunks or ([text.strip()] if text.strip() else [])


def _resolve_text(args: argparse.Namespace) -> str | None:
    if getattr(args, "file", None):
        p = Path(args.file)
        if str(p) == "-":
            return sys.stdin.read()
        return p.read_text(encoding="utf-8")
    if args.text:
        return args.text
    words = getattr(args, "words", None)
    if words:
        return " ".join(words)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def cmd_speak(args: argparse.Namespace) -> int:
    cfg = _load_cfg(Path(args.config))
    tts_cfg = cfg.get("tts") or {}
    profile_id = args.voice or tts_cfg.get("profile", "jarvis")

    text = _resolve_text(args)
    if not text or not text.strip():
        print("Provide text (args, -t, -f, or stdin)", file=sys.stderr)
        return 2
    text = text.strip()

    if _rate_from_args(args, tts_cfg) is not None:
        tts_cfg = {**tts_cfg, "speaking_rate": _rate_from_args(args, tts_cfg)}
    pipeline = VoicePipeline(profile_id, tts_cfg=tts_cfg)
    rate = _rate_from_args(args, tts_cfg)
    out = Path(args.output)
    segments = _split_sentences(text) if args.by_sentence else _split_paragraphs(text)

    if len(segments) == 1 and "%" not in str(out):
        audio, sr = pipeline.synthesize(segments[0], speaking_rate=rate)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, sr)
        print(out)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, seg in enumerate(segments):
        target = Path(str(out) % i) if "%" in str(out) else out.parent / f"{out.stem}_{i:03d}{out.suffix}"
        audio, sr = pipeline.synthesize(seg, speaking_rate=rate)
        sf.write(str(target), audio, sr)
        paths.append(target)
        print(target)
    return 0


def cmd_echo(args: argparse.Namespace) -> int:
    """Synthesize and play on the default Bluetooth speaker (echo-style)."""
    from cli import pick_speaker
    from speaker import WarmSpeaker

    cfg = _load_cfg(Path(args.config))
    tts_cfg = cfg.get("tts") or {}
    profile_id = args.voice or tts_cfg.get("profile", "jarvis")

    text = _resolve_text(args)
    if not text or not text.strip():
        print("Usage: echo [words...] | echo -t TEXT | echo -f file.txt", file=sys.stderr)
        return 2
    text = text.strip()

    _speaker_name, sink = pick_speaker(cfg)

    if _rate_from_args(args, tts_cfg) is not None:
        tts_cfg = {**tts_cfg, "speaking_rate": _rate_from_args(args, tts_cfg)}
    pipeline = VoicePipeline(profile_id, tts_cfg=tts_cfg)
    rate = _rate_from_args(args, tts_cfg)
    segments = _split_sentences(text) if args.by_sentence or len(text) > 280 else [text]

    lead_ms = int((cfg.get("audio") or {}).get("lead_silence_ms", 1000))
    spk = WarmSpeaker(sink, preroll_ms=lead_ms)
    spk.start()
    try:
        for seg in segments:
            pcm, sr = pipeline.synthesize(seg, speaking_rate=rate)
            spk.write_pcm(pcm, sr)
        spk.drain(timeout=600.0)
        time.sleep(0.1)  # pw-cat's own 60ms device latency
    finally:
        spk.close()
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    cfg = _load_cfg(Path(args.config))
    tts_cfg = cfg.get("tts") or {}
    profile_id = args.voice or tts_cfg.get("profile", "ultron")
    pipeline = VoicePipeline(profile_id, tts_cfg=tts_cfg)
    pipeline.convert_wav(Path(args.input), Path(args.output))
    print(args.output)
    return 0


def cmd_models_fetch(args: argparse.Namespace) -> int:
    if args.name == "ultron":
        dest = fetch_ultron(vox_root=VOX_ROOT)
        print(dest)
        return 0
    print(f"Unknown model pack {args.name!r}", file=sys.stderr)
    return 2


def _rate_from_args(args: argparse.Namespace, tts_cfg: dict) -> float | None:
    if getattr(args, "rate", None) is not None:
        return float(args.rate)
    return None


def cmd_presets_show(args: argparse.Namespace) -> int:
    prof = load_profile(args.name, tts_cfg=_load_cfg(Path(args.config)).get("tts"))
    print(f"id: {prof.profile_id}")
    print(f"display_name: {prof.display_name}")
    print(f"speaking_rate: {prof.speaking_rate}")
    print(f"text_pace: {prof.text_pace or 'none'}")
    print(f"ref_audio: {prof.ref_audio} ({'ok' if prof.ref_audio.is_file() else 'MISSING'})")
    print(f"model_dir: {prof.model_dir}")
    print(f"rvc_enabled: {prof.rvc_enabled}")
    if prof.rvc:
        print(f"rvc.model_path: {prof.rvc.model_path}")
        print(f"rvc.index_path: {prof.rvc.index_path}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="speak_cli", description="Vox offline voice generation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_speak = sub.add_parser("speak", help="text to WAV")
    p_speak.add_argument("-t", "--text")
    p_speak.add_argument("-f", "--file")
    p_speak.add_argument("-o", "--output", required=True)
    p_speak.add_argument("--voice", choices=list_profiles())
    p_speak.add_argument("--by-sentence", action="store_true", help="split on sentence boundaries")
    p_speak.add_argument(
        "--rate",
        type=float,
        metavar="FACTOR",
        help="speaking rate (1.0=normal, 0.82=slower)",
    )
    p_speak.set_defaults(func=cmd_speak)

    p_echo = sub.add_parser("echo", help="speak text aloud on the configured speaker (like shell echo)")
    p_echo.add_argument("words", nargs="*", help="words to speak")
    p_echo.add_argument("-t", "--text")
    p_echo.add_argument("-f", "--file", help="file path, or - for stdin")
    p_echo.add_argument("--voice", choices=list_profiles())
    p_echo.add_argument(
        "--by-sentence",
        action="store_true",
        help="split long text into sentences (auto when text > 280 chars)",
    )
    p_echo.add_argument(
        "--rate",
        type=float,
        metavar="FACTOR",
        help="speaking rate (1.0=normal, 0.82=slower)",
    )
    p_echo.set_defaults(func=cmd_echo)

    p_conv = sub.add_parser("convert", help="RVC-only wav conversion")
    p_conv.add_argument("-i", "--input", required=True)
    p_conv.add_argument("-o", "--output", required=True)
    p_conv.add_argument("--voice", choices=list_profiles())
    p_conv.set_defaults(func=cmd_convert)

    p_mod = sub.add_parser("models", help="download model packs")
    p_mod_sub = p_mod.add_subparsers(dest="models_cmd", required=True)
    p_fetch = p_mod_sub.add_parser("fetch")
    p_fetch.add_argument("name", choices=["ultron"])
    p_fetch.set_defaults(func=cmd_models_fetch)

    p_pre = sub.add_parser("presets", help="show preset details")
    p_pre_sub = p_pre.add_subparsers(dest="presets_cmd", required=True)
    p_show = p_pre_sub.add_parser("show")
    p_show.add_argument("name", choices=list_profiles())
    p_show.set_defaults(func=cmd_presets_show)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
