#!/bin/bash
# OpenFrontier (original) evaluated with the fine-tuned Qwen3-VL checkpoint in
# place of gemini / gemma / cosmos3.
#
#   bash eval/run_openfrontier_qwen3.sh [N_WORKERS]     # default 2
#
# Placement, given the box is busy:
#   GPU 1 (idle)  - Qwen3-VL server (~6 GB), SAM3 :12190 (already loaded), workers
#   GPU 0         - left alone; the ofvibewm eval is at 98% util there
#
# No Gemini API key is needed: every VLM role is local.
set -uo pipefail

N=${1:-2}
REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python
VLM_PY=$REPO/.venv/bin/python          # transformers 5.x, needed for qwen3_vl
SAM3_PY=$REPO/sam3-venv/bin/python
LOGDIR=$REPO/eval/qwen3_logs

NICKNAME=ofqwen3
CONFIG=config/navigation_qwen3.yaml
EPISODES=${EPISODES:-28}
MAX_STEPS=${MAX_STEPS:-500}
MODEL=qwen3-vl-sft-local
VLM_PORT=${VLM_PORT:-12187}
SAM3_PORT=${SAM3_PORT:-12190}
GPU=${GPU:-1}

export OPENFRONTIER_DATA_ROOT=$OF/data
export HABITAT_LAB_ROOT=$OF/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export QWEN3_MODEL_ID=${QWEN3_MODEL_ID:-$OF/model_weights/qwen3_sft_single_action}
# Mandatory for this checkpoint -- see vlm/inference_qwen3vl.py.
export QWEN3_REPETITION_PENALTY=${QWEN3_REPETITION_PENALTY:-1.15}
export QWEN3_MAX_NEW_TOKENS=${QWEN3_MAX_NEW_TOKENS:-512}

mkdir -p "$LOGDIR"
cd "$OF" || exit 1

wait_port () {  # $1=port $2=label $3=tries
  for _ in $(seq 1 "${3:-90}"); do
    ss -tln 2>/dev/null | grep -q ":$1 " && return 0
    sleep 5
  done
  echo "FATAL: $2 on $1 never came up (see $LOGDIR)"; return 1
}

echo "== SAM3 on $SAM3_PORT =="
if ss -tln 2>/dev/null | grep -q ":$SAM3_PORT "; then
  echo "   already up"
else
  ( cd "$OF" && CUDA_VISIBLE_DEVICES=$GPU \
      nohup "$SAM3_PY" sam3_server.py "$SAM3_PORT" > "$LOGDIR/sam3_$SAM3_PORT.log" 2>&1 & )
  wait_port "$SAM3_PORT" SAM3 || exit 1
fi

echo "== Qwen3-VL server on $VLM_PORT (GPU $GPU) =="
if ss -tln 2>/dev/null | grep -q ":$VLM_PORT "; then
  echo "   already up"
else
  ( cd "$OF" && CUDA_VISIBLE_DEVICES=$GPU OF_VLM_PORT=$VLM_PORT \
      nohup "$VLM_PY" vlm_server.py --model "$MODEL" \
        > "$LOGDIR/vlm_$VLM_PORT.log" 2>&1 & )
  wait_port "$VLM_PORT" "Qwen3-VL" 120 || exit 1
fi

echo "== launching $N workers =="
for i in $(seq 1 "$N"); do
  OF_CUDA_DEVICE=$GPU OF_SAM3_PORT=$SAM3_PORT OF_VLM_PORT=$VLM_PORT \
    nohup $PY benchmark.py \
      --benchmark hm3d --nickname "$NICKNAME" --config "$CONFIG" \
      --output-path output --eval_episodes "$EPISODES" --max_steps "$MAX_STEPS" \
      --split "$i" "$N" > "$LOGDIR/worker_${i}of${N}.log" 2>&1 &
  echo "   worker $i/$N -> pid $!"
  sleep 5
done

echo
echo "watch:   tail -f $LOGDIR/worker_1of${N}.log"
echo "metrics: $OF/output/${NICKNAME}_sam3_qwen3_qwen3/metrics/"
