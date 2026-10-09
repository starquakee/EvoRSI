from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOGGING_DIR", "/tmp")

OPENMLE_EVO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = OPENMLE_EVO_ROOT.parent
AIRA_EVO_SRC = OPENMLE_EVO_ROOT / "third_party" / "aira-evo" / "src"

for path in (REPO_ROOT, OPENMLE_EVO_ROOT, AIRA_EVO_SRC):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)
