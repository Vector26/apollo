# Local measurements, POC verdicts, agent prompts

**Host:** Ubuntu, RTX 5070 Ti 16 GB · **Voice:** `apollo talk` / Vox orchestrator  
**Archived:** 2026-11-01

---

## Measured baseline (talk.log, route contains `cursor`, n=32, 2026-11-01)

| Metric | p50 | p90 |
|--------|-----|-----|
| End of speech → **first audio** | 2846 ms | 3609 ms |
| **STT** (`stt=`) | 186 ms | 210 ms |
| **LLM first token** (`llm_token=`) | 2463 ms | 3018 ms |
| **Token → first audio** (gap) | 426 ms | 630 ms |
| **Local TTS clause** (Chatterbox + RVC, POC) | 337 ms | 431 ms |

**5% latency bar on first audio:** +142 ms (p50), +180 ms (p90).

Full POC write-up: [`../../poc/cloud_audio/PLAN.md`](../../poc/cloud_audio/PLAN.md).

---

## Cloud STT/TTS (OpenRouter, `provider.zdr=true`) — FAIL for drop-in

| Arm | vs bar |
|-----|--------|
| STT `deepgram/nova-3` | +15–21% (batch upload, not streaming) |
| TTS `elevenlabs/eleven-flash-v2.5` (full clause) | +6% |
| TTS first byte only (needs streaming playback) | barely PASS; not current architecture |

**Conclusion:** keep STT local; fix GPU/speaker stutter before cloud TTS; cloud does not fix **2.5 s Composer** wait.

---

## What already exists in Vox (don't re-build blindly)

| Idea from research | Repo reality |
|--------------------|--------------|
| Clause streaming TTS | `tts_engine.py` + `ClauseSplitter` |
| Speculative pipeline | `speculate_ms` in `config.yaml`, `cli.py` |
| Fixed-line PCM | `FIXED_LINES`, `tts.prerender()` |
| No generative filler | `filler_after_ms: 0` |
| Tiered brains | Composer → OpenRouter → local; DUM-E via gateway |
| User vs companion memory | vself + `apollo_self` / `persona_automaton` |
| Tool abstraction | `agent_tools.py`, gateway, no raw shell on voice path |
| Plane | **Done (v2.3.0):** voice CRUD via `plane_client.py` and `agent_tools.py`, confirm-gated writes |
| Timed reminders | **Done (v2.4.0):** Telemon SQLite store, notifier scheduler, voice CRUD and spoken delivery |
| STT off the GPU | **Done (v2.2.0):** Parakeet int8 on CPU |
| Interrupt by voice | **Done (v2.5):** keyword interrupt while the assistant speaks |

---

## Gaps prompt — generic Jarvis advisor

Paste to an external agent that gave the decoupled-architecture essay:

```
You advised a Jarvis decoupled voice stack. Ground the next answer in THIS deployment only:

Host: headless Ubuntu, RTX 5070 Ti 16GB. Voice = `apollo talk` / Vox orchestrator (`orchestrator/`).

Measured baseline (2026-11-01, n=32, cursor route): end-of-speech→first-audio p50 2846ms p90 3609ms; stt~186ms; llm_token~2463ms; token→audio gap~426ms; local TTS clause p50~337ms. Cloud STT/TTS via OpenRouter+ZDR failed +5% latency bar.

Hard constraints: DUM-E local Qwen ~8.6GB on :8181 MUST stay up; Apollo may die for VRAM; no cloud STT/TTS unless privacy (ZDR) and latency proven on live paired shadow runs. Personality: vself (user memory) + apollo_self automaton (Jarvis registers). Tools: agent gateway/registry, NOT Plane on the hot voice path.

Do NOT recommend swapping DUM-E to 7B or monolithic Redis without justification. Propose: (1) root-cause plan for audio stutter (GPU vs WarmSpeaker), (2) reflex/filler layer that doesn’t lie, (3) optional lighter TTS A/B vs Chatterbox+RVC, (4) async delegation matching agent_tools, (5) what Plane would add vs cost. Cite which numbers you’re improving and how you’d measure success.
```

---

## Gaps prompt — Reddit / TTS research agent

```
Research ONLY implementations verifiable on GitHub/HF with latency numbers. Target: Ubuntu + RTX 5070 Ti, DUM-E Qwen ~8.6GB MUST stay on GPU, Apollo voice = Chatterbox Turbo + optional RVC (Ultron) OR CPU zero-shot clone (Pocket TTS, Kitten-TTS 2). Baseline: TTS clause p50 337ms, llm_token p50 2463ms, stt ~186ms, first_audio p50 2846ms. Success = no WarmSpeaker underruns in 30min talk AND first_audio not worse than +5% OR masked by zero-GPU PCM ack within 200ms of end-of-speech. Compare: (1) drop RVC use ref WAV only, (2) CPU TTS+clone, (3) serialize GPU (pause TTS when :8181 busy). Cite primary sources; ignore DUM-E downsizing and Redis unless voice is distributed.
```

---

## Suggested next engineering order (this project)

_Status checked against code and CHANGELOG on 2026-11-01._

1. **A/B voice:** ultron (CB+RVC) vs Chatterbox-only vs CPU clone POC — underruns + quality. — **OPEN.** No A/B results exist under `poc/`.
2. ~~**Pin STT to CPU** at talk time if GPU STT still possible during speculate.~~ — **DONE (v2.2.0).** Parakeet int8 on CPU is the default (`stt.engine: parakeet`).
3. ~~**Intent PCM sprites** for Tier-0/tool intents (zero GPU during Composer think).~~ — **DONE (v2.0.0).** `acks.py`, pre-rendered, intent-keyed.
4. ~~**Lexical echo gate** before enabling `barge_in` on open speaker.~~ — **DONE (v2.0.0)** (`echo_gate.py`). `barge_in` itself stays off; v2.5 adds keyword interrupts (wait / stop / listen…) instead.
5. **Tool tier 0/1/2** in registry for voice confirmations. — **PARTIAL.** Voice confirmation exists (`tools.confirm`, "yes" gates on notify, notes, Plane and reminder writes, jobs, deletes). No per-tool tier field in `registry.yaml` yet.
6. **`nvidia-smi dmon` + talk.log** correlation for DUM-E ↔ WarmSpeaker restarts. — **PARTIAL.** `stutter_report.py` summarizes stalls from talk.log; there is no GPU-side correlation.
7. **Phone as a remote voice terminal (S25)** — **PARKED, NEXT.** Detail below.

### Item 7 — Phone as a remote voice terminal (parked 2026-11-01)

**Goal.** Talk to Apollo from the Samsung S25 and work through tasks by voice, instead of using Gemini Live. The phone is only a mic and speaker; the whole Apollo stack stays on the server.

**Why not move Apollo onto the phone.** The brains are already cloud and STT is already CPU, so moving logic saves little. The real load is Chatterbox TTS on the GPU, which a phone cannot run (a small on-device TTS would lose the Jarvis voice). The orchestrator is also Linux/PipeWire specific. Keeping the server whole preserves vself memory, Plane, reminders, notes, the agent gateway and DUM-E (which must stay local-only).

**Design.**
- The phone streams mic audio to the server over a WebSocket on a private network (Tailscale), with a token. The server runs the existing loop (VAD, Parakeet STT, vself prepare/commit, brains, tools, Chatterbox) and streams TTS audio back.
- Server side: network audio source and sink adapters in place of `ContinuousCapture` and `WarmSpeaker`. These are the existing I/O seam in `cli.py`; the turn logic should not change.
- One live session at a time (a session lock), so the desk mic and the phone cannot both be active.
- Phone-side echo cancellation allows real barge-in for phone sessions (`barge_in` on for that source only). Keyword interrupts stay for the desk setup.
- Reminders and announcements are delivered to the active phone session. Telegram still fires regardless.
- Unchanged: "yes" confirmations, the halt switch, DUM-E local-only, tool allowlist.

**Client.** Browser page (PWA) first: fastest to build, uses `getUserMedia` with echo cancellation, works on the S25 over Tailscale. A native Android app is optional later, mainly if the OS kills background audio.

**Build order.**
1. Audio source/sink interface behind `cli.py`, with the current PipeWire path as one implementation. Tests with fakes.
2. WebSocket server: auth token, session lock, PCM framing, clean disconnect and reconnect.
3. PWA client: mic capture, playback, push-to-talk and open-mic modes, a clear connected/halted indicator.
4. Per-source barge-in on phone sessions.
5. Reminder and announcement delivery to the phone session.
6. Optional: native Android client.

**Success criteria.**
- A full conversation from the S25 over Tailscale, including a tool call with a spoken "yes".
- First audio within about 300 ms of the desk p50 (2846 ms baseline above), measured from `talk.log`.
- Interrupting mid-reply works by voice alone, with no keyword needed.
- A due reminder is spoken on the phone.
- The endpoint is unreachable off the private network and refuses connections when halted.

**Unverified / risks.** S25 browser mic and echo-cancellation quality; whether a locked screen kills the audio session; added network latency on mobile data; the token and endpoint exposure.

---

## Run cloud POC again

```bash
cd $APOLLO_HOME/poc/cloud_audio
PY=$APOLLO_HOME/orchestrator/.venv/bin/python
$PY bench.py baseline --since 2026-11-01 --route cursor
$PY bench.py tts --model elevenlabs/eleven-flash-v2.5 --voice george
$PY bench.py stt --model deepgram/nova-3
$PY bench.py tts --local --yes-gpu
$PY bench.py verdict --stt-model deepgram/nova-3 --tts-model elevenlabs/eleven-flash-v2.5
```
