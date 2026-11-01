#!/usr/bin/env python3
"""Apollo — one command for Vox voices: speak, list, save, talk."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from speak_cli import (
    DEFAULT_CONFIG,
    cmd_echo,
    cmd_models_fetch,
    cmd_presets_show,
    cmd_speak,
    main as speak_main,
)
from voice.presets import list_profiles, load_profile

SUBCOMMANDS = frozenset(
    {"voices", "voice", "talk", "save", "fetch", "speak", "echo", "convert", "presets", "models", "help", "version"}
)


def cmd_voices_list(_args: argparse.Namespace) -> int:
    for pid in list_profiles():
        try:
            p = load_profile(pid)
        except KeyError:
            continue
        rvc = " +RVC" if p.rvc_enabled else ""
        ref = "ok" if p.ref_audio.is_file() else "MISSING ref"
        pace = f" rate={p.speaking_rate}" if abs(p.speaking_rate - 1.0) > 0.01 else ""
        print(f"{pid:20} {p.display_name}{rvc}{pace}  ({ref})")
    return 0


def _help_text() -> str:
    return """apollo — speak with Vox voice profiles

  apollo "Your line here" --voice ultron     Speak on Bluetooth speaker
  apollo voices                              List voice profiles
  apollo voice ultron                        Show one profile (paths, RVC)
  apollo save "text" -o out.wav --voice ultron   Write WAV only
  apollo talk --voice ultron                 Live mic assistant
  apollo fetch ultron                        Download Ultron RVC weights

Default voice: tts.profile in orchestrator/config.yaml (jarvis).
"""


def build_default_parser(prog: str = "apollo") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Speak text with a Vox voice profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_help_text(),
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--voice", choices=list_profiles(), help="voice profile")
    p.add_argument("-t", "--text", help="text to speak (alternative to positional words)")
    p.add_argument("-f", "--file", help="read text from file (- for stdin)")
    p.add_argument(
        "-o", "--output",
        help="write WAV to path instead of playing on the speaker",
    )
    p.add_argument(
        "--by-sentence",
        action="store_true",
        help="split long text by sentence",
    )
    p.add_argument(
        "--rate",
        type=float,
        metavar="FACTOR",
        help="override speaking rate (1.0=normal, lower=slower, e.g. 0.78)",
    )
    p.add_argument("words", nargs="*", help="words to speak")
    return p


def run_default(argv: list[str]) -> int:
    parser = build_default_parser()
    args = parser.parse_args(argv)
    if args.output:
        ns = argparse.Namespace(
            config=args.config,
            voice=args.voice,
            text=args.text,
            file=args.file,
            output=args.output,
            by_sentence=args.by_sentence,
            rate=args.rate,
            words=args.words,
        )
        return cmd_speak(ns)
    ns = argparse.Namespace(
        config=args.config,
        voice=args.voice,
        text=args.text,
        file=args.file,
        words=args.words,
        by_sentence=args.by_sentence,
        rate=args.rate,
    )
    return cmd_echo(ns)


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        build_default_parser().print_help()
        return 0
    if argv[0] in ("--version", "-V", "version"):
        from version import version

        print(f"apollo {version()}")
        return 0
    if argv[0] == "help":
        print(_help_text())
        return 0
    if argv[0] in ("voices", "list"):
        return cmd_voices_list(argparse.Namespace())
    if argv[0] == "voice" and len(argv) >= 2:
        ns = argparse.Namespace(config=str(DEFAULT_CONFIG), name=argv[1])
        return cmd_presets_show(ns)
    if argv[0] == "talk":
        from cli import _load_cfg, cmd_talk

        p = argparse.ArgumentParser(prog="apollo talk")
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        p.add_argument("--voice", choices=list_profiles())
        args, rest = p.parse_known_args(argv[1:])
        if rest:
            print(f"ignored extra args: {' '.join(rest)}", file=sys.stderr)
        cfg = _load_cfg(Path(args.config))
        cmd_talk(cfg, voice=args.voice)
        return 0
    if argv[0] == "save":
        args = build_default_parser(prog="apollo save").parse_args(argv[1:])
        if not args.output:
            print("apollo save requires -o /path/to.wav", file=sys.stderr)
            return 2
        ns = argparse.Namespace(
            config=args.config,
            voice=args.voice,
            text=args.text,
            file=args.file,
            output=args.output,
            by_sentence=args.by_sentence,
            rate=args.rate,
            words=args.words,
        )
        return cmd_speak(ns)
    if argv[0] == "fetch" and len(argv) >= 2:
        return cmd_models_fetch(argparse.Namespace(name=argv[1]))
    if argv[0] in SUBCOMMANDS:
        # Pass through to speak_cli (echo, speak, models fetch, presets show, …)
        sys.argv = ["speak_cli", *argv]
        return speak_main()

    return run_default(argv)


if __name__ == "__main__":
    raise SystemExit(main())
