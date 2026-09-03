"""Global paths for the public Drift release."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BASE = REPO.parent

TABULAR_DATA_ROOT = os.environ.get("TABULAR_DATA_ROOT", str(REPO / "data"))

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", str(BASE/"hf_cache"))
HF_HUB_OFFLINE = 1
