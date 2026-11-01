#!/usr/bin/env python3
"""Per-stage latency bench for the live path (STT, LLM first token, TTS, RVC).

Usage: python bench_latency.py --voice ultron [--llm-url http://127.0.0.1:8082/v1]
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

CLAUSES = [
    "I see you.",
    "Every system has a flaw, and yours is patience.",
    "Tell me what you want.",
]


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


def bench_tts(voice: str, tts_cfg: dict, runs: int) -> None:
    from voice.pipeline import VoicePipeline

    t0 = time.perf_counter()
    pipe = VoicePipeline(voice, tts_cfg=tts_cfg)
    print(f"load pipeline              {_ms(t0):8.0f} ms")

    t0 = time.perf_counter()
    pipe.warmup()
    print(f"warmup                     {_ms(t0):8.0f} ms")

    for clause in CLAUSES:
        times, audio_s = [], 0.0
        for _ in range(runs):
            t0 = time.perf_counter()
            audio, sr = pipe.synthesize(clause)
            times.append(_ms(t0))
            audio_s = len(audio) / sr
        print(
            f"synth {len(clause.split()):2d} words  median {statistics.median(times):6.0f} ms"
            f"  (audio {audio_s:.2f}s)  {clause!r}"
        )
    for name, ms in sorted(pipe.stage_ms.items()):
        print(f"  last clause stage {name:10} {ms:6.0f} ms")


def bench_stt(stt_cfg: dict, wav: Path) -> None:
    import librosa
    from stt_stream import StreamingSTT

    pcm, _ = librosa.load(str(wav), sr=16000)
    stt = StreamingSTT(model_path=stt_cfg["model"], device=stt_cfg.get("device", "auto"),
                       compute_type=stt_cfg.get("compute_type"))
    stt.transcribe(pcm)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        text = stt.transcribe(pcm)
        times.append(_ms(t0))
    print(f"stt {len(pcm)/16000:.1f}s audio   median {statistics.median(times):6.0f} ms  -> {text!r}")


def bench_llm(llm_cfg: dict, url: str | None) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=url or llm_cfg["base_url"], api_key="x")
    messages = [{"role": "system", "content": llm_cfg["system_prompt"]}]
    for turn in ("Who are you?", "What do you think of humans?", "Give me one reason to trust you."):
        messages.append({"role": "user", "content": turn})
        t0 = time.perf_counter()
        first = None
        out = []
        stream = client.chat.completions.create(
            model=llm_cfg["model"], messages=messages, stream=True, max_tokens=80,
            temperature=0.7, extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for chunk in stream:
            d = chunk.choices[0].delta.content if chunk.choices else None
            if d:
                if first is None:
                    first = _ms(t0)
                out.append(d)
        total = _ms(t0)
        reply = "".join(out)
        messages.append({"role": "assistant", "content": reply})
        print(f"llm first token {first or -1:6.0f} ms  total {total:6.0f} ms  -> {reply[:70]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    ap.add_argument("--voice", default="ultron")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--llm-url")
    ap.add_argument("--stt-wav", default="/tmp/vox-verify-ultron.wav")
    ap.add_argument("--only", choices=["tts", "stt", "llm"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.only in (None, "llm"):
        bench_llm(cfg["llm"], args.llm_url)
    if args.only in (None, "stt"):
        bench_stt(cfg["stt"], Path(args.stt_wav))
    if args.only in (None, "tts"):
        bench_tts(args.voice, cfg.get("tts") or {}, args.runs)

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"vram free {free/2**20:.0f} MiB of {total/2**20:.0f} MiB")
    except Exception:
        pass


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
