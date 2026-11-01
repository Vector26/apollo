"""Lazy-loaded RVC v2 conversion."""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from voice.presets import RvcSettings

logger = logging.getLogger(__name__)


def _patch_torch_load_for_rvc() -> None:
    """Fairseq HuBERT checkpoints need full pickle load on PyTorch 2.6+."""
    import torch

    if getattr(torch.load, "_vox_rvc_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._vox_rvc_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


class RvcConverter:
    """Wrap rvc_python for ndarray in/out."""

    def __init__(self, settings: RvcSettings, default_device: str = "auto"):
        self.settings = settings
        self._default_device = default_device
        self._rvc = None
        self._loaded_path: Optional[Path] = None
        self._lock = threading.Lock()

    def _device(self) -> str:
        import torch

        d = self.settings.device
        if d != "auto":
            return d if d.startswith("cuda") else f"cuda:0" if d == "cuda" else d
        base = self._default_device
        if base == "auto":
            base = "cuda:0" if torch.cuda.is_available() else "cpu"
        if base == "cuda":
            base = "cuda:0"
        return base

    def _ensure_loaded(self) -> None:
        if self._rvc is not None:
            return
        if not self.settings.model_path or not self.settings.model_path.is_file():
            raise FileNotFoundError(
                f"RVC model not found under {self.settings.model_dir}. "
                "Run: python -m speak_cli models fetch ultron"
            )
        import torch

        _patch_torch_load_for_rvc()
        from rvc_python.infer import RVCInference

        device = self._device()
        logger.info("Loading RVC %s on %s", self.settings.model_path.name, device)
        self._rvc = RVCInference(device=device)
        index_path = str(self.settings.index_path) if self.settings.index_path else ""
        # PyTorch 2.x: RVC weight init uses in-place zero_ on parameters that may require grad.
        with torch.inference_mode():
            self._rvc.load_model(
                str(self.settings.model_path),
                version=self.settings.version,
                index_path=index_path,
            )
        self._rvc.set_params(
            f0up_key=self.settings.pitch,
            f0method=self.settings.method,
            index_rate=self.settings.index_rate,
            filter_radius=self.settings.filter_radius,
            rms_mix_rate=self.settings.rms_mix_rate,
            protect=self.settings.protect,
        )
        self._loaded_path = self.settings.model_path

    def convert_array(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Convert float32 mono PCM; returns (audio, sr)."""
        with self._lock:
            self._ensure_loaded()
            assert self._rvc is not None
            with tempfile.TemporaryDirectory(prefix="vox-rvc-") as tmp:
                inp = Path(tmp) / "in.wav"
                out = Path(tmp) / "out.wav"
                mono = np.asarray(audio, dtype=np.float32).squeeze()
                if mono.ndim > 1:
                    mono = mono.mean(axis=-1)
                sf.write(str(inp), mono, sample_rate)
                self._rvc.infer_file(str(inp), str(out))
                converted, out_sr = sf.read(str(out), dtype="float32")
                return converted.squeeze(), int(out_sr)

    def convert_file(self, input_path: Path, output_path: Path) -> None:
        with self._lock:
            self._ensure_loaded()
            assert self._rvc is not None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._rvc.infer_file(str(input_path), str(output_path))
