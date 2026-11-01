"""Delivery pacing: slower, controlled speech without changing pitch."""

from __future__ import annotations

import re

import numpy as np


def pace_text(text: str, mode: str | None) -> str:
    """Light text shaping for deliberate delivery."""
    if not mode or mode == "none":
        return text
    if mode == "measured":
        # Breathing room at phrase boundaries (Chatterbox reads punctuation timing).
        t = re.sub(r"\s+", " ", text.strip())
        t = re.sub(r",\s*", ", … ", t)
        t = re.sub(r"\.\s+", ". … ", t)
        t = re.sub(r"\?\s+", "? … ", t)
        t = re.sub(r"!\s+", "! … ", t)
        t = re.sub(r"…\s*…+", "… ", t)
        return t.strip()
    return text


def apply_speaking_rate(
    audio: np.ndarray,
    sample_rate: int,
    speaking_rate: float,
) -> tuple[np.ndarray, int]:
    """Stretch time to slow delivery. speaking_rate 1.0 = normal, 0.82 ≈ 18% slower."""
    if speaking_rate <= 0:
        raise ValueError("speaking_rate must be positive")
    if abs(speaking_rate - 1.0) < 0.015:
        return audio, sample_rate

    import librosa

    mono = np.asarray(audio, dtype=np.float32).squeeze()
    if mono.ndim > 1:
        mono = mono.mean(axis=-1)
    # librosa: rate < 1 slows down, > 1 speeds up
    stretched = librosa.effects.time_stretch(mono, rate=speaking_rate)
    return stretched.astype(np.float32), sample_rate
