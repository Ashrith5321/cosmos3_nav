#!/usr/bin/env bash
# Kills the process named in $1 (cmdline match) if its GPU memory exceeds $2 MiB.
# Emits a status line every poll while the process exists; exits when it's gone.
PATTERN="${1:?usage: vram_watchdog.sh <cmdline-pattern> <limit-mib>}"
LIMIT_MIB="${2:-20000}"
LAST_EMIT=0

while true; do
    # pgrep can match wrapper shells whose cmdline contains the pattern, so
    # pick the matching pid that nvidia-smi actually reports as a compute app.
    CANDIDATES=$(pgrep -f "$PATTERN")
    if [[ -z "$CANDIDATES" ]]; then
        echo "WATCHDOG: target process gone — exiting"
        exit 0
    fi
    GPU_APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    TARGET_PID=""
    USED=0
    for pid in $CANDIDATES; do
        mem=$(awk -F', ' -v p="$pid" '$1==p {print $2}' <<<"$GPU_APPS")
        if [[ -n "$mem" ]]; then
            TARGET_PID="$pid"
            USED="$mem"
            break
        fi
    done
    if [[ -z "$TARGET_PID" ]]; then
        # process exists but not on the GPU yet (still loading)
        TARGET_PID=$(head -1 <<<"$CANDIDATES")
    fi
    NOW=$(date +%s)
    if (( USED > LIMIT_MIB )); then
        echo "WATCHDOG: pid $TARGET_PID at ${USED} MiB > ${LIMIT_MIB} MiB limit — KILLING"
        kill -9 "$TARGET_PID"
        echo "WATCHDOG: killed pid $TARGET_PID"
        exit 1
    elif (( USED > LIMIT_MIB - 1500 )); then
        echo "WATCHDOG warning: pid $TARGET_PID at ${USED} MiB (limit ${LIMIT_MIB})"
    elif (( NOW - LAST_EMIT >= 3600 )); then
        echo "WATCHDOG: pid $TARGET_PID using ${USED} MiB / limit ${LIMIT_MIB} MiB"
        LAST_EMIT=$NOW
    fi
    sleep 3
done
