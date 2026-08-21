#!/usr/bin/env python3
"""Fail on common anonymity, secret, archive, and hosting mistakes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".tex", ".yml", ".yaml"}
FORBIDDEN = [
    re.compile(r"/" + "home" + r"/", re.I),
    re.compile(r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
]
MAX_GIT_FILE = 95 * 1024 * 1024


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / p.decode() for p in result.stdout.split(b"\0") if p]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = tracked_files()
    failures: list[str] = []
    for path in files:
        size = path.stat().st_size
        rel = path.relative_to(ROOT)
        if size > MAX_GIT_FILE:
            failures.append(f"oversized tracked file: {rel} ({size} bytes)")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Makefile", "LICENSE", ".gitignore"}:
            text = path.read_text(errors="replace")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    failures.append(f"forbidden pattern {pattern.pattern!r}: {rel}")

    pred_dir = ROOT / "filtering/figures/ieee_tps_2026/aligned_predictions"
    try:
        import numpy as np
        for path in sorted(pred_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    archive[key]
    except Exception as exc:
        failures.append(f"portable prediction archive check failed: {exc}")

    manifest_path = ROOT / "filtering/figures/ieee_tps_2026/artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest.get("generated_outputs_before_manifest", []):
        target = ROOT / "filtering/figures/ieee_tps_2026" / Path(record["path"]).name
        if target.is_file() and record.get("sha256") and sha256(target) != record["sha256"]:
            failures.append(f"manifest hash mismatch: {target.relative_to(ROOT)}")

    if failures:
        print("release audit FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"release audit passed ({len(files)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
