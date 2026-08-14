#!/bin/bash
# Work-stealing worker pool for the vibemax3 eval.
#
# benchmark.py has no load balancing: a worker owns one --split for its whole
# life and exits when that split is done, even if other splits still have
# hundreds of episodes left. This wraps it in a shared task queue so a worker
# that finishes immediately pulls the next pending task. All workers stay busy
# until the queue is empty.
#
#   bash eval/vibemax3_pool.sh [N_WORKERS]      # default 14
#
# Tasks are (split, ep-shard) pairs. Episode shards write to
# metrics/<scene>.shard<i>of<n>.csv, so tasks never write to the same file and
# `already_done` skips episodes the earlier unsharded run already scored.
# Use eval/vibemax3_report.py to read results (it de-duplicates).
set -uo pipefail

N=${1:-14}
SPLITS=12
SHARDS=2

REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier_vibe
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python
RUNDIR=$REPO/eval/vibemax3_logs
QUEUE=$RUNDIR/queue.txt
LOCK=$RUNDIR/queue.lock

NICKNAME=vibemax3
CONFIG=config/navigation_worldmodel_max_gemini.yaml
EPISODES=28
MAX_STEPS=500

export OPENFRONTIER_DATA_ROOT=$REPO/OpenFrontier/data
export HABITAT_LAB_ROOT=$REPO/OpenFrontier/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -a; . ${GEMINI_KEY_ENV:-$HOME/.config/ofgemini/gemini.env}; set +a

mkdir -p "$RUNDIR"; touch "$LOCK"
cd "$OF" || exit 1

# --- build the queue -------------------------------------------------------
# Interleave shards so the two shards of one split land on different workers
# (and therefore different GPUs) rather than queueing back to back.
: > "$QUEUE"
for sh in $(seq 0 $((SHARDS-1))); do
  for sp in $(seq 1 $SPLITS); do
    echo "$sp $sh" >> "$QUEUE"
  done
done
echo "queued $(wc -l < "$QUEUE") tasks ($SPLITS splits x $SHARDS shards)"

# --- one worker: pop a task, run it, repeat until the queue is empty --------
worker () {
  local id=$1
  local gpu=$(( (id - 1) % 2 ))
  local sam3; [ "$gpu" -eq 0 ] && sam3=12184 || sam3=12190
  local log="$RUNDIR/pool_w${id}.log"
  echo "[worker $id] gpu=$gpu sam3=$sam3" > "$log"

  while true; do
    # Atomically pop the first line.
    local task
    task=$(flock "$LOCK" -c "head -n1 '$QUEUE'; sed -i '1d' '$QUEUE'")
    [ -z "$task" ] && break

    local sp sh
    sp=$(echo "$task" | awk '{print $1}')
    sh=$(echo "$task" | awk '{print $2}')
    echo "[worker $id] === split $sp/$SPLITS shard $sh/$SHARDS  $(date +%H:%M:%S) ===" >> "$log"

    OF_CUDA_DEVICE=$gpu OF_SAM3_PORT=$sam3 \
      "$PY" benchmark.py \
        --benchmark hm3d --nickname "$NICKNAME" --config "$CONFIG" \
        --output-path output --eval_episodes $EPISODES --max_steps $MAX_STEPS \
        --wm-oracle --agent-radius 0.05 \
        --split "$sp" $SPLITS --ep-shard "$sh" $SHARDS >> "$log" 2>&1

    echo "[worker $id] done split $sp shard $sh rc=$? $(date +%H:%M:%S)" >> "$log"
  done
  echo "[worker $id] queue empty, exiting $(date +%H:%M:%S)" >> "$log"
}

echo "starting $N workers ($(( (N+1)/2 )) on GPU 0, $(( N/2 )) on GPU 1)"
for i in $(seq 1 "$N"); do
  worker "$i" &
  echo "  worker $i -> pid $! (gpu $(( (i-1) % 2 )))"
  sleep 3
done

echo
echo "queue:   wc -l $QUEUE"
echo "logs:    tail -f $RUNDIR/pool_w1.log"
echo "results: $PY $REPO/eval/vibemax3_report.py"
wait
