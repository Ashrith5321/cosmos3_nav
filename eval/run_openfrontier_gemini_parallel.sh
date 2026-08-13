#!/bin/bash
# Scenario A, parallel. Splits the 36 HM3D val scenes across N workers via
# benchmark.py --split, spread over both GPUs with one SAM3 server per GPU.
#
# Workers own disjoint scenes, so they write disjoint metrics/<scene>.csv files
# and never race. Fully resumable: re-running skips episodes already scored.
#
#   bash eval/run_openfrontier_gemini_parallel.sh [N_WORKERS]   # default 8
#
# Measured headroom that makes this safe:
#   - Gemini: 1723 calls/min at concurrency 32, zero 429s (run needs ~120/min at N=8)
#   - GPU   : ~1 GB per worker + 4.6 GB per SAM3 server, on 2 x 48 GB
#   - CPU   : 16 physical cores / 32 threads
set -uo pipefail

N=${1:-8}
REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python
SAM3_PY=$REPO/sam3-venv/bin/python
KEY_ENV=${GEMINI_KEY_ENV:-$HOME/.config/ofgemini/gemini.env}
LOGDIR=$REPO/eval/ofgemini_logs

NICKNAME=ofgemini
CONFIG=config/navigation_gemini.yaml
EPISODES=28
MAX_STEPS=500
MODEL=gemini-3.1-flash-lite

export OPENFRONTIER_DATA_ROOT=$OF/data
export HABITAT_LAB_ROOT=$OF/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
# Keep per-worker thread pools small; N workers x 16 threads would thrash 32 cores.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

[ -f "$KEY_ENV" ] || { echo "FATAL: no key file at $KEY_ENV"; exit 1; }
set -a; . "$KEY_ENV"; set +a
mkdir -p "$LOGDIR"
cd "$OF" || exit 1

# --- preflight: fail before touching habitat --------------------------------
echo "== preflight: $MODEL =="
OF_MODEL=$MODEL $PY - <<'EOF' || { echo "PREFLIGHT FAILED - not launching."; exit 1; }
import os, sys, numpy as np
from google import genai
from google.genai import types
from PIL import Image
try:
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    r = c.models.generate_content(
        model=os.environ["OF_MODEL"],
        contents=['Return {"probability": 0.0, "reason": "t"}',
                  Image.fromarray(np.zeros((480, 960, 3), dtype=np.uint8))],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    print(f"   OK in={r.usage_metadata.prompt_token_count}")
except Exception as e:
    print("   FAIL:", str(e)[:300]); sys.exit(1)
EOF

# --- one SAM3 server per GPU ------------------------------------------------
start_sam3 () {  # $1=port  $2=gpu
  if ss -tln 2>/dev/null | grep -q ":$1 "; then
    echo "   SAM3 already up on $1"; return 0
  fi
  echo "   starting SAM3 on port $1 (GPU $2)"
  # sam3_server.py takes the port as argv[1] (or $SAM3_PORT); OF_SAM3_PORT is
  # the *client*-side variable and is ignored by the server.
  ( cd "$OF" && CUDA_VISIBLE_DEVICES=$2 \
      nohup "$SAM3_PY" sam3_server.py "$1" > "$LOGDIR/sam3_$1.log" 2>&1 & )
  for _ in $(seq 1 90); do ss -tln 2>/dev/null | grep -q ":$1 " && return 0; sleep 5; done
  echo "FATAL: SAM3 on $1 never came up (see $LOGDIR/sam3_$1.log)"; return 1
}
echo "== SAM3 servers =="
start_sam3 12184 0 || exit 1
start_sam3 12190 1 || exit 1

# --- workers ----------------------------------------------------------------
echo "== launching $N workers over 36 scenes =="
for i in $(seq 1 "$N"); do
  if [ $(( (i - 1) % 2 )) -eq 0 ]; then GPU=0; SAM3=12184; else GPU=1; SAM3=12190; fi
  OF_CUDA_DEVICE=$GPU OF_SAM3_PORT=$SAM3 \
    nohup $PY benchmark.py \
      --benchmark hm3d --nickname "$NICKNAME" --config "$CONFIG" \
      --output-path output --eval_episodes $EPISODES --max_steps $MAX_STEPS \
      --split "$i" "$N" > "$LOGDIR/worker_${i}of${N}.log" 2>&1 &
  echo "   worker $i/$N -> GPU $GPU, SAM3 $SAM3, pid $!"
  sleep 3            # stagger habitat-sim startup
done

echo
echo "logs: $LOGDIR/worker_*.log"
echo "progress: bash $REPO/eval/ofgemini_status.sh"
wait
