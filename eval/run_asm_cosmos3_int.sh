#!/bin/bash
# OpenFrontier_cosmos3_int with the ASM side-channel enabled.
#
#   bash eval/run_asm_cosmos3_int.sh              # smoke: 2 episodes, 1 worker
#   bash eval/run_asm_cosmos3_int.sh 28           # 28 episodes per scene
#   bash eval/run_asm_cosmos3_int.sh 28 3 8       # episodes, split i, of N
#
# Starts its own SAM3 servers on ports 12191 (main detection path) and 12192
# (ASM), so it does not contend with an eval already running on 12184/12190.
# Both land on GPU 1.
#
# The ASM writes to output/<nickname>_*/<scene>/{success,failure}/<ep>/asm/.
set -uo pipefail

EPISODES=${1:-2}
SPLIT_I=${2:-}
SPLIT_N=${3:-}

REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier_cosmos3_int
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python
SAM3_PY=$REPO/sam3-venv/bin/python
KEY_ENV=${GEMINI_KEY_ENV:-$HOME/.config/ofgemini/gemini.env}
LOGDIR=$REPO/eval/asm_logs

NICKNAME=asmint
CONFIG=config/navigation_gemini.yaml
MAX_STEPS=${MAX_STEPS:-500}
MAIN_SAM3_PORT=12191
ASM_PORT=12192
GPU=${GPU:-1}

export OPENFRONTIER_DATA_ROOT=$OF/data
export HABITAT_LAB_ROOT=$OF/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# --- ASM knobs (see asm/config.py) ------------------------------------------
export OF_ASM=1
export ASM_SAM3_PORT=$ASM_PORT
export ASM_SEGMENT_EVERY=${ASM_SEGMENT_EVERY:-8}
export ASM_RESOLUTION_M=${ASM_RESOLUTION_M:-0.05}
# Trim this list to cut ASM cost -- it is one SAM3 request per entry per keyframe.
export ASM_CATEGORIES=${ASM_CATEGORIES:-"chair,bed,potted plant,toilet,tv,sofa"}

[ -f "$KEY_ENV" ] || { echo "FATAL: no key file at $KEY_ENV"; exit 1; }
set -a; . "$KEY_ENV"; set +a
mkdir -p "$LOGDIR"
cd "$OF" || { echo "FATAL: no $OF"; exit 1; }

# --- preflight: the ASM package imports cleanly ------------------------------
echo "== preflight: asm package =="
$PY -c "
import sys; sys.path.insert(0, '$OF')
from asm import ASMConfig
cfg = ASMConfig.from_env()
print('   categories      :', cfg.categories)
print('   resolution      :', cfg.resolution_m, 'm |  extent', cfg.extent_m, 'm')
print('   segment every   :', cfg.segment_every_n_frames, 'frames')
print('   ASM SAM3 port   :', cfg.sam3_port)
" || { echo "PREFLIGHT FAILED"; exit 1; }

# --- SAM3 servers ------------------------------------------------------------
start_sam3 () {  # $1=port $2=gpu
  if ss -tln 2>/dev/null | grep -q ":$1 "; then
    echo "   SAM3 already up on $1"; return 0
  fi
  echo "   starting SAM3 on port $1 (GPU $2)"
  ( cd "$OF" && CUDA_VISIBLE_DEVICES=$2 \
      nohup "$SAM3_PY" sam3_server.py "$1" > "$LOGDIR/sam3_$1.log" 2>&1 & )
  for _ in $(seq 1 90); do ss -tln 2>/dev/null | grep -q ":$1 " && return 0; sleep 5; done
  echo "FATAL: SAM3 on $1 never came up (see $LOGDIR/sam3_$1.log)"; return 1
}
echo "== SAM3 servers =="
start_sam3 $MAIN_SAM3_PORT $GPU || exit 1
start_sam3 $ASM_PORT       $GPU || exit 1

# --- run ---------------------------------------------------------------------
SPLIT_ARGS=()
[ -n "$SPLIT_I" ] && [ -n "$SPLIT_N" ] && SPLIT_ARGS=(--split "$SPLIT_I" "$SPLIT_N")

LOG=$LOGDIR/asmint_$(date +%Y%m%d_%H%M%S).log
echo "== launching (episodes=$EPISODES, max_steps=$MAX_STEPS, GPU=$GPU) =="
echo "   log: $LOG"

OF_CUDA_DEVICE=$GPU OF_SAM3_PORT=$MAIN_SAM3_PORT \
  nohup $PY -m asm.run_with_asm benchmark \
    --benchmark hm3d --nickname "$NICKNAME" --config "$CONFIG" \
    --output-path output --eval_episodes "$EPISODES" --max_steps "$MAX_STEPS" \
    "${SPLIT_ARGS[@]}" > "$LOG" 2>&1 &

echo "   pid $!"
echo
echo "watch:   tail -f $LOG"
echo "asm out: find $OF/output/${NICKNAME}_* -name latest_asm.json | head"
