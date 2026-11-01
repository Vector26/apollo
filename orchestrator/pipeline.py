"""Main conversation pipeline: listen → transcribe → think → speak → note."""

from apollo_home import apollo_path
import logging
import re
import signal
import sys
from typing import Optional

import yaml

from audio_io import AudioPlayer, VADCapture
from llm_client import LLMClient
from memory import MemoryStore
from notes import NotesManager
from voicebox_client import VoiceboxClient

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split text into speakable sentences."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


class ConversationPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.vbx = VoiceboxClient(
            self.cfg["voicebox"]["base_url"],
        )
        self.llm = LLMClient(
            base_url=self.cfg["llm"]["base_url"],
            model=self.cfg["llm"]["model"],
            system_prompt=self.cfg["llm"]["system_prompt"],
            temperature=self.cfg["llm"].get("temperature", 0.7),
            max_tokens=self.cfg["llm"].get("max_tokens", 512),
        )
        self.notes = NotesManager(
            output_dir=apollo_path(self.cfg["notes"]["output_dir"]),
            llm_client=self.llm,
            auto_dump_minutes=self.cfg["notes"].get("auto_dump_interval_minutes", 5),
        )
        mem_cfg = self.cfg.get("memory") or {}
        self.memory = MemoryStore(
            root=apollo_path(mem_cfg.get("dir", "memory")),
            notes_dir=apollo_path(self.cfg["notes"]["output_dir"]),
            keep_turns=int(mem_cfg.get("keep_turns", 8)),
            fact_limit=int(mem_cfg.get("fact_limit", 8)),
        )
        self.player = AudioPlayer()
        self.capture = VADCapture(
            sample_rate=self.cfg["audio"]["sample_rate"],
            capture_rate=self.cfg["audio"].get("capture_rate", 48000),
            vad_threshold=self.cfg["audio"]["vad_threshold"],
            silence_ms=self.cfg["audio"]["silence_ms"],
            min_speech_ms=self.cfg["audio"]["min_speech_ms"],
            on_speech_start=self._on_barge_in,
        )
        self._running = True
        self._profile_name = self.cfg["voicebox"].get("tts_profile", "vox-voice")
        self._tts_engine = self.cfg["voicebox"].get("tts_engine", "kokoro")

    def _on_barge_in(self):
        if self.player.is_playing:
            logger.info("Barge-in: stopping playback")
            self.player.stop()

    def _wait_for_services(self):
        import time
        logger.info("Waiting for Voicebox…")
        for i in range(120):
            if self.vbx.health():
                break
            if i % 10 == 0:
                logger.info("  Voicebox not ready (%ds)…", i)
            time.sleep(2)
        else:
            raise RuntimeError("Voicebox not available at " + self.cfg["voicebox"]["base_url"])

        self.vbx.ensure_kokoro_profile(
            self._profile_name,
            self.cfg["voicebox"].get("tts_profile", "af_heart"),
        )
        logger.info("Services ready")

    def _speak_reply(self, text: str):
        for sentence in split_sentences(text):
            if not self._running:
                return
            try:
                wav = self.vbx.speak(
                    sentence,
                    profile=self._profile_name,
                    engine=self._tts_engine,
                )
                self.player.play(wav)
            except Exception as e:
                logger.error("TTS failed for '%s': %s", sentence[:40], e)

    def _handle_turn(self, user_text: str):
        lower = user_text.lower().strip()

        if lower in ("stop", "quit", "exit", "goodbye"):
            self.notes.add("user", user_text)
            self._speak_reply("Goodbye! I'll save your notes now.")
            path = self.notes.dump()
            logger.info("Session ended. Notes: %s", path)
            self._running = False
            return

        if self.notes.is_dump_command(user_text):
            self.notes.add("user", user_text)
            path = self.notes.dump()
            self._speak_reply(f"Notes saved to {path.name}.")
            return

        if self.memory.is_forget_all(user_text):
            self.notes.add("user", user_text)
            self.memory.forget_all()
            self.llm.reset()
            self._speak_reply("I've cleared everything I was remembering.")
            return

        if self.memory.is_forget_last(user_text):
            self.notes.add("user", user_text)
            dropped = self.memory.forget_last()
            if dropped:
                self._speak_reply("Okay, I'll forget that.")
            else:
                self._speak_reply("I didn't have a last fact to drop.")
            return

        self.notes.add("user", user_text)

        # Stream LLM reply, speak sentence-by-sentence for low latency
        buffer = ""
        full_reply = []
        memory_block = self.memory.context_for(user_text)
        for token in self.llm.chat_stream(user_text, memory_block=memory_block):
            buffer += token
            full_reply.append(token)
            if re.search(r'[.!?]\s*$', buffer):
                sentence = buffer.strip()
                if sentence:
                    self._speak_reply(sentence)
                buffer = ""

        if buffer.strip():
            self._speak_reply(buffer.strip())

        reply = "".join(full_reply)
        self.notes.add("assistant", reply)
        self.memory.observe(self.llm, user_text, reply)
        self.llm.history = self.memory.compact(self.llm, self.llm.history)

        if self.notes.should_dump():
            path = self.notes.dump()
            logger.info("Auto-dump: %s", path)

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_running", False))
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_running", False))

        self._wait_for_services()
        self._speak_reply("Hello! I'm ready to chat. Say stop when you're done.")

        logger.info("Pipeline running — speak into the mic")
        while self._running:
            wav = self.capture.listen_once()
            if wav is None:
                continue
            try:
                text = self.vbx.transcribe(
                    wav,
                    model=self.cfg["voicebox"].get("whisper_model", "turbo"),
                )
            except Exception as e:
                logger.error("Transcription failed: %s", e)
                continue

            if not text:
                continue

            logger.info("USER: %s", text)
            self._handle_turn(text)

        if self.notes.entries:
            self.notes.dump()
