"""Read vibemax3 results across plain and episode-sharded metrics files.

The work-stealing pool writes episode shards to
`metrics/<scene>.shard<i>of<n>.csv` alongside the original `metrics/<scene>.csv`.
A naive glob over `metrics/*.csv` would count an episode twice if it appears in
both, so de-duplicate on (scene, episode) before computing anything.

    python eval/vibemax3_report.py [run_dir]
"""
import csv
import glob
import os
import sys
from collections import Counter

DEFAULT = ("/home/ashed/Documents/cosmos3_nav/OpenFrontier_vibe"
           "/output/vibemax3_sam3_gemini_gemini")


def scene_of(path: str) -> str:
    """`<scene>.csv` and `<scene>.shard0of2.csv` both belong to `<scene>`."""
    return os.path.basename(path).split(".")[0]


def load(run_dir: str):
    """{(scene, episode): row}, last writer wins."""
    episodes = {}
    files = sorted(glob.glob(os.path.join(run_dir, "metrics", "*.csv")))
    sharded = [f for f in files if ".shard" in os.path.basename(f)]
    for f in files:
        scene = scene_of(f)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                episodes[(scene, row["episode"])] = row
    return episodes, files, sharded


def main() -> int:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    episodes, files, sharded = load(run_dir)
    rows = list(episodes.values())
    n = len(rows)
    if not n:
        print("no episodes yet")
        return 1

    scenes = {s for s, _ in episodes}
    succ = [r for r in rows if float(r["success"]) == 1.0]
    sr = len(succ) / n
    spl = sum(float(r["spl"]) for r in rows) / n
    sr01 = sum(1 for r in succ if float(r["distance_to_goal"]) <= 0.1) / n

    print(f"=== vibemax3 ===  n={n} unique episodes  scenes={len(scenes)}/36")
    print(f"    csv files: {len(files)} ({len(sharded)} sharded)")
    print(f"    SR@1.0 {sr:.4f}   SPL {spl:.4f}   SR@0.1 {sr01:.4f}")

    fails = [r for r in rows if float(r["success"]) == 0.0]
    print(f"    failures n={len(fails)}:",
          dict(Counter(r["termination_reason"] for r in fails).most_common()))

    per_goal = {}
    for r in rows:
        g = per_goal.setdefault(r["object_goal"], [0, 0])
        g[0] += float(r["success"])
        g[1] += 1
    print("    per-goal:", ", ".join(
        f"{k} {v[0]:.0f}/{v[1]}={v[0]/v[1]:.2f}"
        for k, v in sorted(per_goal.items(), key=lambda x: -x[1][0] / x[1][1])))

    per_scene = {}
    for (s, _), r in episodes.items():
        c = per_scene.setdefault(s, [0, 0])
        c[0] += float(r["success"])
        c[1] += 1
    incomplete = {s: c for s, c in per_scene.items() if c[1] < 28}
    print(f"    scenes at 28 episodes: {len(per_scene) - len(incomplete)}/{len(per_scene)}")
    if incomplete:
        print("    still filling:", ", ".join(
            f"{s}({c[1]})" for s, c in sorted(incomplete.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
