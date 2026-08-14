#!/usr/bin/env python
"""
Build <run_dir>/failure_videos/<reason>/ containing ONLY the episode videos
(one .mp4 symlink per failed episode) for easy browsing.

Non-destructive: episode folders stay where the benchmark put them (safe to
run while a fleet is writing). Re-run anytime to refresh.

Usage:
    python scripts/collect_failures.py output/vibegeo_sam3_cosmos3_cosmos3
"""

import re
import shutil
import sys
from pathlib import Path


def collect(run_dir: Path) -> None:
    out_root = run_dir / "failure_videos"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir()
    # drop the older episode-folder browser if present (superseded)
    legacy = run_dir / "failures_by_reason"
    if legacy.exists():
        shutil.rmtree(legacy)

    count = 0
    reasons = {}
    for ep_dir in sorted(run_dir.glob("*/failure/episode-*")):
        video = ep_dir / "metrics.mp4"
        if not video.exists():
            continue
        scene = ep_dir.parent.parent.name
        m = re.match(r"episode-(\d+)-(.+?)-([a-z_]+?)/?$", ep_dir.name)
        reason = m.group(3) if m else "unknown"
        reason_dir = out_root / reason
        reason_dir.mkdir(exist_ok=True)
        link = reason_dir / f"{scene}__{ep_dir.name}.mp4"
        target = Path("..", "..") / scene / "failure" / ep_dir.name / "metrics.mp4"
        if not link.exists():
            link.symlink_to(target)
        reasons[reason] = reasons.get(reason, 0) + 1
        count += 1

    print(f"{run_dir.name}: {count} failure videos")
    for reason, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:<24} {c}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run = Path(sys.argv[1])
    if not run.is_dir():
        sys.exit(f"not a directory: {run}")
    collect(run)
