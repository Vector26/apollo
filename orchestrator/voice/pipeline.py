"""Profile-aware TTS (+ optional RVC) for live and batch paths."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from voice.chatterbox import ChatterboxSynth
from voice.pace import apply_speaking_rate, pace_text
from voice.presets import VoiceProfile, load_profile
from voice.rvc import RvcConverter

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Single synthesize() entry for talk + speak CLIs."""

    def __init__(
        self,
        profile_id: str = "jarvis",
        *,
        tts_cfg: Optional[dict[str, Any]] = None,
        vox_root: Optional[Path] = None,
    ):
        from voice.presets import VOX_ROOT

        root = vox_root or VOX_ROOT
        self.profile = load_profile(profile_id, tts_cfg=tts_cfg, vox_root=root)
        self.profile_id = profile_id
        self.device = self.profile.device
        self.sample_rate = 24000

        if not self.profile.ref_audio.is_file() and self.profile_id == "robert_california":
            raise FileNotFoundError(
                f"Reference audio missing: {self.profile.ref_audio}\n"
                "Add a Robert California clip at that path (10–30 s WAV)."
            )
        if not self.profile.ref_audio.is_file():
            raise FileNotFoundError(f"Reference audio missing: {self.profile.ref_audio}")

        self._chatterbox = ChatterboxSynth(self.profile.model_dir, device=self.profile.device)
        self.sample_rate = self._chatterbox.sample_rate
        self._rvc: Optional[RvcConverter] = None
        if self.profile.rvc_enabled and self.profile.rvc:
            self._rvc = RvcConverter(self.profile.rvc, default_device=self._chatterbox.device)
        self.stage_ms: dict[str, float] = {}

    def warmup(self) -> None:
        """Load every lazy model (RVC, HuBERT, RMVPE) before the first real clause."""
        self.synthesize("Ready.")

    @property
    def display_name(self) -> str:
        return self.profile.display_name

    def synthesize(self, text: str, *, speaking_rate: float | None = None) -> tuple[np.ndarray, int]:
        text = pace_text(text, self.profile.text_pace)
        cb = self.profile.chatterbox
        t0 = time.perf_counter()
        audio, sr = self._chatterbox.generate(
            text,
            self.profile.ref_audio,
            temperature=cb.temperature,
            top_k=cb.top_k,
            top_p=cb.top_p,
            repetition_penalty=cb.repetition_penalty,
            exaggeration=cb.exaggeration,
        )
        t1 = time.perf_counter()
        if self._rvc is not None:
            audio, sr = self._rvc.convert_array(audio, sr)
        t2 = time.perf_counter()
        rate = speaking_rate if speaking_rate is not None else self.profile.speaking_rate
        audio, sr = apply_speaking_rate(audio, sr, rate)
        t3 = time.perf_counter()
        self.stage_ms = {
            "tts": (t1 - t0) * 1000,
            "rvc": (t2 - t1) * 1000,
            "stretch": (t3 - t2) * 1000,
        }
        return audio, sr

    def convert_wav(self, input_path: Path, output_path: Path) -> None:
        if self._rvc is None:
            raise RuntimeError(f"Profile {self.profile_id!r} has RVC disabled")
        self._rvc.convert_file(input_path, output_path)
