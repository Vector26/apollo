"""Download and extract community RVC weights."""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from voice.presets import VOX_ROOT, get_fetch_spec, load_profile

logger = logging.getLogger(__name__)


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, dest)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extracted to %s", dest_dir)


def fetch_ultron(*, vox_root: Path = VOX_ROOT) -> Path:
    spec = get_fetch_spec("ultron", vox_root)
    dest_dir: Path = spec["dest_dir"]
    urls = [spec["primary_url"]]
    if spec.get("fallback_url"):
        urls.append(spec["fallback_url"])

    last_err: Exception | None = None
    for url in urls:
        try:
            with tempfile.TemporaryDirectory(prefix="vox-fetch-") as tmp:
                zip_path = Path(tmp) / "model.zip"
                _download(url, zip_path)
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                _extract_zip(zip_path, dest_dir)
            prof = load_profile("ultron", vox_root=vox_root)
            if prof.rvc and prof.rvc.model_path and prof.rvc.model_path.is_file():
                logger.info("RVC weights: %s", prof.rvc.model_path)
                return dest_dir
            raise RuntimeError(f"No .pth found in {dest_dir} after extract")
        except Exception as e:
            last_err = e
            logger.warning("Fetch failed for %s: %s", url, e)
    raise RuntimeError(f"All fetch URLs failed: {last_err}") from last_err
