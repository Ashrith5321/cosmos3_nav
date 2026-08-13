#!/bin/bash
# OpenFrontier_vibe world-model evaluation on HM3D v2 val, parallel.
#
# Same protocol as of1000_sam3_cosmos3_cosmos3 (28 eps/scene = ~1000 episodes,
# SAM3 segmentation + cosmos3-nano for both VLM roles) so runs are
# episode-paired with the existing baseline, PLUS the frontier-conditioned
# world model (zero-shot CLIP backend) and --wm-oracle ranking logging.
#
#   bash eval/run_ofvibe_wm_parallel.sh [N_WORKERS]   # default 4
#
# Requirements already running (started separately):
#   - cosmos3-nano VLM server on 12185 (GPU 0)
#   - SAM3 servers on 12184 (GPU 0) and 12190 (GPU 1)
# Fully resumable: re-running skips episodes already scored.
set -uo pipefail

N=${1:-4}
REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier_vibe
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python
LOGDIR=$REPO/eval/ofvibe_wm_logs

NICKNAME=ofvibewm
CONFIG=config/navigation_worldmodel.yaml
EPISODES=28
MAX_STEPS=500

export OPENFRONTIER_DATA_ROOT=$REPO/OpenFrontier/data
export HABITAT_LAB_ROOT=$REPO/OpenFrontier/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OF_VLM_PORT=12185

mkdir -p "$LOGDIR"
cd "$OF" || exit 1

# --- preflight: local servers must answer -----------------------------------
echo "== preflight: VLM (12185) and SAM3 (12184/12190) =="
for port in 12185 12184 12190; do
  ss -tln 2>/dev/null | grep -q ":$port " \
    || { echo "FATAL: nothing listening on $port"; exit 1; }
done
$PY - <<'EOF' || { echo "PREFLIGHT FAILED - VLM server not answering."; exit 1; }
import numpy as np, sys
sys.path.insert(0, ".")
from vlm.client import VLMClient
c = VLMClient("vlm", port=12185)
r = c.send_request(prompt='Reply with the JSON {"ok": true}',
                   image=np.zeros((32, 32, 3), dtype=np.uint8))
print("   VLM OK:", str(r.get("response", ""))[:80].replace("\n", " "))
EOF

# --- workers ----------------------------------------------------------------
# All main workers pinned to GPU 0 (with the VLM server); GPU 1 is reserved
# for the cosmos_gen generator pilot. Override with MAIN_GPU/MAIN_SAM3.
echo "== launching $N workers over 36 scenes =="
for i in $(seq 1 "$N"); do
  GPU=${MAIN_GPU:-0}; SAM3=${MAIN_SAM3:-12184}
  OF_CUDA_DEVICE=$GPU OF_SAM3_PORT=$SAM3 OF_VLM_PORT=12185 \
    nohup $PY benchmark.py \
      --benchmark hm3d --nickname "$NICKNAME" --config "$CONFIG" \
      --output-path output --eval_episodes $EPISODES --max_steps $MAX_STEPS \
      --wm-oracle \
      --split "$i" "$N" > "$LOGDIR/worker_${i}of${N}.log" 2>&1 &
  echo "   worker $i/$N -> GPU $GPU, SAM3 $SAM3, pid $!"
  sleep 3            # stagger habitat-sim startup
done

echo
echo "logs:     $LOGDIR/worker_*.log"
echo "metrics:  $OF/output/${NICKNAME}_sam3_cosmos3_cosmos3/metrics/"
echo "ranking:  python scripts/analyze_wm_ranking.py '$OF/output/${NICKNAME}_*/**/wm_state.jsonl'"
wait
