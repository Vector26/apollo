<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/header-dark.svg">
  <img src="docs/readme/header-light.svg" width="100%" alt="Apollo, a voice-first agent hub">
</picture>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-465_passing-34d399?style=flat-square)](orchestrator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux_·_PipeWire-lightgrey?style=flat-square)](https://pipewire.org/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/divider-dark.svg">
  <img src="docs/readme/divider-light.svg" width="100%" alt="">
</picture>

</div>

**Apollo is a talk-to-it-like-a-person assistant that can also *do* things:** list and change your tickets, set reminders that ping your phone and speak out loud, ask a research agent to browse the web, check on your server. It is built so that **no single model has to be smart, trusted, or expensive.**

The speech loop, the tool routing, the safety rules and the memory are ordinary deterministic code. A language model is called only to phrase a reply or to pick one line from a short menu. That makes the brain **swappable and cheap**: a fast agent, a small cloud model and a local model all race for each turn, and the first to speak wins.

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/architecture-dark.svg">
  <img src="docs/architecture-light.svg" width="100%" alt="Apollo architecture: real-time voice loop, intent router, brain race, capability layer, identity and memory">
</picture>

</div>

## What makes it different

To my knowledge there is no other open project that combines all of these. Each piece exists elsewhere; the combination is the point.

| | |
|---|---|
| **The brain is a pluggable role, not the product** | Up to three routes answer in parallel (an agent in Ask mode, a cheap cloud model, a local model). First token wins, the losers are cut mid-stream. Swap a route in `config.yaml`; nothing else changes. |
| **Models never touch the machine** | A model can only write one marker line such as `[[plane:list\|]]`. Code parses it, checks an allowlist, reads risky actions back and waits for a spoken *yes*, then runs it. There is no shell, no file access, no free-form tool call. |
| **Most requests never reach a model for routing** | "Remind me in ten minutes", "list my tickets", "ask the detective" are matched by a deterministic planner first, tolerant of speech-to-text mangling ("dummy" resolves to DUM-E). The model is the fallback, not the router. |
| **Identity and memory live outside the brain** | A separate service compiles persona and memories for every turn. Replace the model and the assistant is still *the same assistant*. |
| **It knows when to shut up, and when to speak** | An LLM-free gate decides if an unprompted remark is even worth a model call (idle window, hourly cap, late hour, one weighted roll). Due reminders and late agent results are spoken when you are not mid-sentence. |
| **Voice is a first-class, tuned loop** | Speculative start at 96 ms of silence, hold on half-sentences, a lexical echo gate for open speakers, keyword interrupt (*wait*, *stop*, *listen to me*) while it talks, reconnecting audio devices, latency metrics per turn. |

## Agent-agnostic by construction

Every box in the diagram is a role with a narrow contract. Nothing depends on a particular vendor.

| Role | Contract | What fills it today | To swap |
|---|---|---|---|
| Brain | text in, text stream out, may emit a marker line | Composer 2.5 Fast (an agent used as a plain brain), OpenRouter, a local MoE behind a queue | edit `brains:` in `config.yaml` |
| Speech in | audio in, text out | Parakeet on CPU (Whisper supported) | `stt.engine` |
| Speech out | text in, audio out, per clause | Chatterbox Turbo, optional voice conversion | `tts:` and `orchestrator/voice/presets.yaml` |
| Tools | one allowlisted function per marker | agents via a gateway, Plane, reminders, notes, phone notify | add a handler in `agent_tools.py` |
| Memory | `prepare`, `commit`, `end` over HTTP | Virtual Self | any service with those three calls |

"Agent-agnostic" runs in both directions. Apollo can *use* any agent behind the gateway, and any agent can *be* the brain, because the contract is a line of text.

## Why it is cheap

Cost is a design property here, not a tuning pass.

- **No per-minute speech bill.** Voice activity detection, speech-to-text and text-to-speech all run on the local machine. STT runs on the CPU and leaves the GPU free for the voice. On my machine Parakeet on CPU measured about 20x faster than Whisper on CPU at similar accuracy (see [`CHANGELOG.md`](CHANGELOG.md)).
- **Tokens are spent only where language is needed.** Routing, confirmation, reminders, ticket operations and agent dispatch are code. A turn costs one short reply, capped to a couple of spoken sentences.
- **The model's job is tiny.** It needs good manners and the ability to emit one line from a menu. That is within reach of small and cheap models, which is why the cheap cloud route and a 3B-active local model are first-class citizens instead of emergency fallbacks.
- **Context is built outside the model.** The memory service hands the brain exactly what a turn needs, so you do not need a large context window or an expensive model to remember things.
- **Idle is free.** Unprompted speech passes a non-LLM gate before any model is asked.
- **The expensive thing is optional.** The fast agent route is a latency choice. Remove it and the other routes carry on.

I have not benchmarked dollars per hour against other assistants, so there is no number here. The mechanisms above are what keep the bill small, and each one is visible in the code.

## Safety model

- **Allowlist, not sandbox.** Tools are a fixed set of functions. Anything not in `agent_tools.py` cannot happen.
- **Spoken confirmation** for writes that matter: phone messages, appending to existing notes, ticket changes and deletes, reminder create/update/delete, long agent jobs. A misheard word cannot send something or delete something on its own.
- **An ops agent that is pinned local.** The agent that can inspect the machine is routed to local models only; a test guards this.
- **Audit trail** of every tool side effect, and a **global halt switch** that stops tool execution.
- **No secrets in the repo.** Keys are read from environment variables or key files named in config.

## Repository map

```
orchestrator/
  cli.py                 live loop: capture, turns, barge-in, playback, announcer
  audio_io.py            VAD capture and playback        stt_stream.py   STT workers
  tts_engine.py          clause-level TTS                echo_gate.py    drop its own voice
  endpoint_hold.py       don't answer half a sentence    voice_interrupt.py  stop words
  vself_brain.py         brain race, marker peeling, memory calls
  cursor_brain.py        an agent session used as a plain brain
  agent_tools.py         planner, allowlist, confirmations, audit
  plane_client.py        ticket CRUD                     reminder_client.py  reminder CRUD
  initiative.py          when to speak unprompted        persona_automaton.py  growing temperament
  apollo_self.py         the assistant's own persona     announce.py     late results
  config.example.yaml    every knob, with comments
scripts/                 apollo, talk, speak, audio diagnostics
docs/                    ARCHITECTURE.md, REMINDER_API.md, measurements
```

## Run it

```bash
git clone https://github.com/Vector26/apollo && cd apollo
python3 -m venv orchestrator/.venv && source orchestrator/.venv/bin/activate
pip install -r orchestrator/requirements.txt
cp orchestrator/config.example.yaml orchestrator/config.yaml   # then edit
scripts/apollo talk
```

What is **not** included, on purpose: model weights, any reference voice clip, and any personal data. Put a 10 to 30 second reference clip of a voice you have the rights to at `models/refs/default.wav`, and place the Chatterbox Turbo weights in the directory named by `tts.model_dir` (they are loaded with `from_local`, not fetched for you). Paths in config accept `~` and are otherwise relative to the repo (or `APOLLO_HOME`). I have not verified a from-scratch install on a clean machine.

Tests need no hardware or network:

```bash
cd orchestrator && python -m unittest discover
```

## Honest status

This is a personal system, published as it runs. It is tested on Ubuntu with PipeWire and one consumer GPU.

- The tool layer rides on the Virtual Self brain path. Without that service you get a plain OpenAI-compatible brain and no tools.
- Reminders need a service that speaks the small HTTP contract in [`docs/REMINDER_API.md`](docs/REMINDER_API.md). The server is not in this repository.
- The Plane tool currently assumes the ticket prefix `LP` (hard-coded in the reference parser and the model prompt); making it configurable is on the list.
- Two tests are skipped on a clean checkout because they need model weights or a private agent registry.
- Latency is real but not instant: on the fast-agent route I measured end-of-speech to first audio at p50 2846 ms and p90 3609 ms (n=32, 2026-11-01); most of that is the model's first token. Details in [`docs/research`](docs/research/04-local-measurements-and-prompts.md).
- The voice presets point at community voice-conversion weights for fictional characters. Check their licences, or use a voice of your own.

## License

MIT. See [`LICENSE`](LICENSE).
