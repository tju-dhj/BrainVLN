"""Record the command and Git state next to an experiment directory."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _git(args):
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def write_provenance(output_dir: str, argv) -> None:
    payload = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "argv": list(argv),
        "python": sys.version,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status": _git(["status", "--short"]),
    }
    Path(output_dir, "provenance.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
