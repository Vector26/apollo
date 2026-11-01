# Changelog

## 2.5.0 — 2026-11-01

### Added
- **Keyword interrupt** (`voice_interrupt.py`, `audio.interrupt_keywords`): while Apollo speaks, the mic listens only for
  short stop phrases (wait, stop, hold on, listen to me, pause, enough, …) and cuts playback. Whole short utterance only,
  so "wait for the build" does not interrupt. Full open-mic barge-in stays off (echo).
- **Varied start-up line**: one of six pre-rendered lines instead of always "I'm listening."

### Fixed
- A Plane question could be answered as a reminder lookup: model-emitted reminder calls are now honoured only when the
  owner said remind / reminder / alarm, and "what are the plane projects that are open" routes to Plane.
- Plane was silently disabled: `~` in `api_key_file` was never expanded.
- The voice loop polled the notifier for due reminders about 25 times a second; now every 2 s.

## 2.4.0 — 2026-11-01

### Added
- **Timed reminders (Telemon)**: self-hosted SQLite store (`reminders.db`, indexed due queries, transactional CRUD;
  one-time import from legacy `reminders.json`). Notifier on `:8001` exposes `/reminders/*`, runs the scheduler
  (Telegram + voice queue), and Apollo voice has full CRUD with yes on create/update/delete.

## 2.1.0 — 2026-11-01

### Added
- **Jarvis speaks up** (`initiative.py`, `brains.vself.initiative`): in a live conversation, only while Composer or
  OpenRouter is the one answering (never the local brain, never DUM-E), Apollo may say one unprompted line. A tiny
  decision maker gates it (owner quiet 15-240 s, 4 min between asides, 3 per hour, floor free, not halted), then
  makes one weighted roll per quiet spell. The weight rises with late hour, a long session, "tired", or a risky
  request. Only then is one cloud call made, with the persona and the last few turns (no vself memories). NONE is the
  expected answer; rambling, markup, bot talk and repeats are discarded. If anything happened while the model was
  thinking, the line is dropped.
- **Concern and sarcasm**: new register `solicitous` (Jarvis worrying about Tony Stark: say it once, one practical
  step), a sharper `wry` (dry sarcasm, never at him when he is down), new symbols `tired` and `risk`, and a seed core
  that says so. Saved personalities gain the new register and keep their learned weights; an untouched old core is upgraded.
- **More spoken confirmation** (`tools.confirm`, default `notify` and `notes_write`): a message to the phone and words
  added to an existing note are read back and wait for "yes" (a new note, reads and status checks do not). Delete,
  start/cancel job, flowsmith and DUM-E approval were already gated. Remove an entry in `config.yaml` to undo it.
- **Parakeet test** (`poc/parakeet/RESULTS.md`): on CPU it ran 20x faster than Whisper on CPU at similar accuracy.
  Not adopted.
- Tests: `test_initiative.py`, `ConfirmMoreTest`, brain and loop tests.

## 2.0.0 — 2026-11-01

Major bump. Everything from 1.x is kept; nothing was removed. The state just before this release is
archived at `releases/v1-pre-v2-20261101-152802.tar.gz` (sha256 beside it). See `releases/RESTORE.md`.

### Added
- **Intent-keyed acks** (`acks.py`, `audio.acks`): when the brain is still silent after `after_ms`, an ops or
  deep question gets a short pre-rendered line ("Checking.", "Let me think that through."). Small talk gets none.
  No GPU at play time; lines rotate and never repeat back to back.
- **Lexical echo gate** (`echo_gate.py`, `audio.echo_gate`): a transcript that repeats the assistant's last ~8 s
  of speech (word-pair overlap ≥ 70%, ≥ 4 words) is dropped instead of becoming a turn.
- **Stall instrumentation**: `clause synth Nms for audio Nms (rtf X)` per clause and
  `speaker gap Nms between chunks` per silent gap inside a reply. `python stutter_report.py` summarizes sessions
  (RVC on/off, synth vs audio, stalls) and says whether TTS or the brain's token cadence is the bottleneck.
- `VERSION`, `version.py`, `apollo --version`, version shown in the startup line.
- Tests: `test_v2.py`, three loop tests in `test_talk_flow.py`.

### Unchanged on purpose
- **DUM-E and the agent gateway are local-only and were not touched.** `test_v2.DumeStaysLocalTest` fails if the
  registry ever routes DUM-E off the local model server.
- Tool gating already existed (read-only runs at once; changes become DUM-E proposals or ask "yes" first).
- Cloud TTS/STT stays off (POC missed the 5% bar); Parakeet STT not adopted (new dependency, unmeasured).
- Chatterbox + RVC voices and the `jarvis` (no RVC) profile are as before.

## 1.x — before 2026-11-01
Apollo self-memory and weighted personality automaton, vself retrieval referent fix, backend errors treated as route
failures, halt switch, gateway hardening, cloud-audio POC, research notes under `docs/research/`.
