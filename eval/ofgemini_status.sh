#!/bin/bash
# Progress + ETA for the parallel ofgemini run.
cd /home/ashed/Documents/cosmos3_nav || exit 1
echo "workers alive: $(pgrep -fc 'benchmark.py --benchmark hm3d --nickname ofgemini')"
python3 - <<'EOF'
import csv, glob, os, time
D = 'OpenFrontier/output/ofgemini_sam3_gemini_gemini/metrics'
files = sorted(glob.glob(D + '/*.csv'))
rows = [r for f in files for r in csv.DictReader(open(f))]
n = len(rows)
if not n:
    print('no episodes scored yet'); raise SystemExit
s = sum(float(r['success']) for r in rows)
p = sum(float(r['spl']) for r in rows)
print(f'{n}/1000 episodes over {len(files)}/36 scenes   SR={s/n*100:.1f}%  SPL={p/n*100:.1f}%')
from collections import Counter
print('  reasons:', dict(Counter(r['termination_reason'] for r in rows).most_common()))
# ETA from the newest/oldest metrics-file mtimes (wall-clock of the parallel run)
ts = [os.path.getmtime(f) for f in files]
span = max(ts) - min(ts)
if span > 60 and n > 5:
    rate = n / span                     # episodes/sec aggregate
    print(f'  aggregate {rate*3600:.0f} ep/h -> ETA {(1000-n)/rate/3600:.1f} h remaining')
EOF
echo "429s so far: $(cat eval/ofgemini_logs/worker_*.log 2>/dev/null | grep -c '429\|RESOURCE_EXHAUSTED')"
