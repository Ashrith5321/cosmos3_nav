#!/usr/bin/env bash
# Waits for >= NEED_MIB free VRAM (stable across 3 checks), then launches the
# Cosmos3 server, the 1000-episode HM3D v2 driver, and the VRAM watchdog —
# all nohup'd so they survive terminal/session restarts.
set -u
EVAL_DIR="/home/ashed/Documents/cosmos3_nav/eval"
VENV_PY="/home/ashed/Documents/cosmos3_nav/.venv/bin/python"
HAB_PY="/home/ashed/miniconda3/envs/habitat033/bin/python"
NEED_MIB=20000
STABLE=0

echo "Waiting for ${NEED_MIB} MiB free VRAM (3 stable checks, 60s apart)..."
while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( FREE >= NEED_MIB )); then
        STABLE=$((STABLE + 1))
        echo "free=${FREE} MiB (stable ${STABLE}/3)"
        (( STABLE >= 3 )) && break
    else
        (( STABLE > 0 )) && echo "free=${FREE} MiB — below threshold, reset"
        STABLE=0
    fi
    sleep 60
done

echo "GPU free — starting Cosmos3 server"
cd /home/ashed/Documents/cosmos3_nav
nohup "$VENV_PY" eval/cosmos3_server.py > eval/server.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 150); do
    curl -sf http://127.0.0.1:8399/health >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED during load"; exit 1; }
    sleep 2
done
curl -sf http://127.0.0.1:8399/health >/dev/null || { echo "SERVER never became healthy"; exit 1; }
echo "Server healthy (pid $SERVER_PID)"

nohup bash "$EVAL_DIR/vram_watchdog.sh" "cosmos3_server.py" 20000 \
    > "$EVAL_DIR/watchdog.log" 2>&1 &
echo "Watchdog armed (log: eval/watchdog.log)"

nohup "$HAB_PY" eval/run_habitat_smoke_eval.py \
    --limit 1000 --max-steps 100 \
    --skip-results "$EVAL_DIR/done_so_far.jsonl" \
    --output-dir "$EVAL_DIR/habitat_1000_out" \
    > eval/habitat_1000_run.log 2>&1 &
echo "Driver launched (pid $!, log: eval/habitat_1000_run.log)"
echo "LAUNCHED"
