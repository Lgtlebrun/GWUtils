import os
from pathlib import Path

# Defaults to a per-user data directory; override with env vars to use a
# shared/custom location.
_DEFAULT_DATA_DIR = Path.home() / ".gwutils"

SKYMAP_FITS_DIRECTORY = Path(
    os.environ.get("GWUTILS_SKYMAP_DIR", _DEFAULT_DATA_DIR / "skymaps")
)
EVENTS_DIRECTORY = Path(
    os.environ.get("GWUTILS_EVENTS_DIR", _DEFAULT_DATA_DIR / "events")
)

SKYMAP_FITS_DIRECTORY.mkdir(parents=True, exist_ok=True)
EVENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
