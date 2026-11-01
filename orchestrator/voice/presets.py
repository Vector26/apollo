"""Load voice profiles from presets.yaml + config overrides."""

from __future__ import annotations

from apollo_home import APOLLO_HOME
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

VOX_ROOT = APOLLO_HOME
PRESETS_PATH = Path(__file__).resolve().parent / "presets.yaml"


@dataclass
class RvcSettings:
    model_dir: Path
    model_path: Optional[Path] = None
    index_path: Optional[Path] = None
    pitch: int = 0
    index_rate: float = 0.75
    filter_radius: int = 3
    rms_mix_rate: float = 0.25
    protect: float = 0.5
    method: str = "rmvpe"
    version: str = "v2"
    device: str = "auto"


@dataclass
class ChatterboxSettings:
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    exaggeration: Optional[float] = None


@dataclass
class VoiceProfile:
    profile_id: str
    display_name: str
    ref_audio: Path
    model_dir: Path
    device: str
    rvc_enabled: bool
    speaking_rate: float = 1.0
    text_pace: Optional[str] = None
    rvc: Optional[RvcSettings] = None
    chatterbox: ChatterboxSettings = field(default_factory=ChatterboxSettings)
    brain: Optional[str] = None
    persona: Optional[str] = None


def _resolve_path(raw: str, root: Path = VOX_ROOT) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _find_rvc_weights(model_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    if not model_dir.is_dir():
        return None, None
    pth_files = sorted(model_dir.glob("*.pth"))
    index_files = sorted(model_dir.glob("*.index"))
    if not pth_files:
        for sub in model_dir.iterdir():
            if sub.is_dir():
                pth_files = sorted(sub.glob("*.pth"))
                index_files = sorted(sub.glob("*.index"))
                if pth_files:
                    break
    model_path = pth_files[0] if pth_files else None
    index_path = index_files[0] if index_files else None
    return model_path, index_path


def _load_presets_doc() -> dict[str, Any]:
    with open(PRESETS_PATH) as f:
        return yaml.safe_load(f) or {}


def list_profiles() -> list[str]:
    doc = _load_presets_doc()
    return list((doc.get("profiles") or {}).keys())


def load_profile(
    profile_id: str,
    *,
    tts_cfg: Optional[dict[str, Any]] = None,
    vox_root: Path = VOX_ROOT,
) -> VoiceProfile:
    doc = _load_presets_doc()
    profiles = doc.get("profiles") or {}
    if profile_id not in profiles:
        raise KeyError(f"Unknown voice profile {profile_id!r}; choose from {list(profiles)}")

    raw = profiles[profile_id]
    defaults = doc.get("defaults") or {}
    tts_cfg = tts_cfg or {}

    model_dir = _resolve_path(
        tts_cfg.get("model_dir") or defaults.get("model_dir", "models/chatterbox-turbo"),
        vox_root,
    )
    device = tts_cfg.get("device") or defaults.get("device", "auto")

    ref_raw = raw.get("ref_audio")
    if profile_id == "jarvis" and tts_cfg.get("ref_audio"):
        ref_raw = tts_cfg["ref_audio"]
    ref_audio = _resolve_path(ref_raw, vox_root)

    cb_raw = raw.get("chatterbox") or {}
    chatterbox = ChatterboxSettings(
        temperature=float(cb_raw.get("temperature", 0.8)),
        top_k=int(cb_raw.get("top_k", 1000)),
        top_p=float(cb_raw.get("top_p", 0.95)),
        repetition_penalty=float(cb_raw.get("repetition_penalty", 1.2)),
        exaggeration=cb_raw.get("exaggeration"),
    )
    if chatterbox.exaggeration is not None:
        chatterbox.exaggeration = float(chatterbox.exaggeration)

    rvc_enabled = bool(raw.get("rvc_enabled", False))
    rvc_settings: Optional[RvcSettings] = None
    if rvc_enabled:
        rvc_raw = raw.get("rvc") or {}
        rvc_dir = _resolve_path(rvc_raw.get("model_dir", "models/rvc/ultron"), vox_root)
        model_path, index_path = _find_rvc_weights(rvc_dir)
        rvc_settings = RvcSettings(
            model_dir=rvc_dir,
            model_path=model_path,
            index_path=index_path,
            pitch=int(rvc_raw.get("pitch", 0)),
            index_rate=float(rvc_raw.get("index_rate", 0.75)),
            filter_radius=int(rvc_raw.get("filter_radius", 3)),
            rms_mix_rate=float(rvc_raw.get("rms_mix_rate", 0.25)),
            protect=float(rvc_raw.get("protect", 0.5)),
            method=str(rvc_raw.get("method", "rmvpe")),
            version=str(rvc_raw.get("version", "v2")),
            device=str(rvc_raw.get("device", "auto")),
        )

    speaking_rate = float(raw.get("speaking_rate", 1.0))
    if tts_cfg.get("speaking_rate") is not None:
        speaking_rate = float(tts_cfg["speaking_rate"])

    text_pace = raw.get("text_pace")
    if tts_cfg.get("text_pace") is not None:
        text_pace = tts_cfg["text_pace"]

    return VoiceProfile(
        profile_id=profile_id,
        display_name=str(raw.get("display_name", profile_id)),
        ref_audio=ref_audio,
        model_dir=model_dir,
        device=device,
        rvc_enabled=rvc_enabled,
        speaking_rate=speaking_rate,
        text_pace=str(text_pace) if text_pace else None,
        rvc=rvc_settings,
        chatterbox=chatterbox,
        brain=str(raw["brain"]) if raw.get("brain") else None,
        persona=str(raw["persona"]).strip() if raw.get("persona") else None,
    )


def get_fetch_spec(name: str, vox_root: Path = VOX_ROOT) -> dict[str, Any]:
    doc = _load_presets_doc()
    fetch = (doc.get("fetch") or {}).get(name)
    if not fetch:
        raise KeyError(f"No fetch spec for {name!r}")
    dest = _resolve_path(fetch["dest_dir"], vox_root)
    return {
        "primary_url": fetch["primary_url"],
        "fallback_url": fetch.get("fallback_url"),
        "dest_dir": dest,
    }
