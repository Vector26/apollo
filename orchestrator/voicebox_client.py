"""Voicebox REST client — STT, TTS, profile management."""

import io
import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class VoiceboxClient:
    def __init__(self, base_url: str, client_id: str = "vox"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-Voicebox-Client-Id": client_id}
        self._profile_id: Optional[str] = None

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/profiles", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def ensure_kokoro_profile(self, name: str, voice_id: str) -> str:
        """Create or reuse a Kokoro preset profile."""
        r = requests.get(f"{self.base_url}/profiles", timeout=10)
        r.raise_for_status()
        for p in r.json():
            if p.get("name") == name:
                self._profile_id = p["id"]
                logger.info("Reusing profile %s (%s)", name, self._profile_id)
                return self._profile_id

        r = requests.post(
            f"{self.base_url}/profiles",
            json={
                "name": name,
                "voice_type": "preset",
                "language": "en",
                "preset_engine": "kokoro",
                "preset_voice_id": voice_id,
            },
            timeout=30,
        )
        r.raise_for_status()
        self._profile_id = r.json()["id"]
        logger.info("Created profile %s (%s)", name, self._profile_id)
        return self._profile_id

    @property
    def profile_id(self) -> str:
        if not self._profile_id:
            raise RuntimeError("No profile configured — call ensure_kokoro_profile first")
        return self._profile_id

    def transcribe(self, wav_bytes: bytes, model: str = "turbo", language: str = "en") -> str:
        """Transcribe WAV audio to text."""
        for attempt in range(30):
            r = requests.post(
                f"{self.base_url}/transcribe",
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={"model": model, "language": language},
                timeout=120,
            )
            if r.status_code == 202:
                logger.info("Whisper model downloading, waiting… (attempt %d)", attempt + 1)
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json()["text"].strip()
        raise RuntimeError("Whisper model not ready after 30 attempts")

    def speak(self, text: str, profile: Optional[str] = None, engine: str = "kokoro") -> bytes:
        """Generate speech and return WAV bytes."""
        body = {"text": text, "engine": engine}
        if profile:
            body["profile"] = profile

        r = requests.post(
            f"{self.base_url}/speak",
            json=body,
            headers=self.headers,
            timeout=30,
        )
        r.raise_for_status()
        gen = r.json()
        gen_id = gen["id"]

        deadline = time.time() + 120
        while time.time() < deadline:
            status = self._poll_generation(gen_id)
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"TTS generation failed for: {text[:60]}")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"TTS timed out for: {text[:60]}")

        r = requests.get(f"{self.base_url}/audio/{gen_id}", timeout=30)
        r.raise_for_status()
        return r.content

    def _poll_generation(self, gen_id: str) -> str:
        r = requests.get(
            f"{self.base_url}/generate/{gen_id}/status",
            stream=True,
            timeout=130,
        )
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            status = payload.get("status", "completed")
            if status in ("completed", "failed", "not_found"):
                return status
        return "completed"
