"""
Tiny manual .env loader -- deliberately not a dependency on python-dotenv.
Only supports simple `KEY=VALUE` lines (blank lines and lines starting with
`#` are ignored). Existing environment variables always win over the .env
file, so real deployment env vars are never clobbered by a stray local file.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_env_file(path: Path | str = DEFAULT_ENV_PATH) -> None:
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
