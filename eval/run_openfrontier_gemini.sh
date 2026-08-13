#!/bin/bash
# Scenario A: reproduce OpenFrontier Table I on HM3D val with Gemini for both
# VLM roles (frontier reweighting + goal verification), SAM3 for segmentation.
#
# 28 episodes/scene = 1000 episodes, matching of1000_sam3_cosmos3_cosmos3 and
# ofgemma_sam3_gemma_gemma exactly, so the three runs are episode-paired.
#
#   bash eval/run_openfrontier_gemini.sh
#
# Resumable: benchmark.py skips episodes already present in metrics/<scene>.csv.
set -uo pipefail

REPO=/home/ashed/Documents/cosmos3_nav
OF=$REPO/OpenFrontier
PY=/home/ashed/miniconda3/envs/openfrontier/bin/python   # habitat + google.genai + open3d
SAM3_PY=$REPO/sam3-venv/bin/python
KEY_ENV=${GEMINI_KEY_ENV:-/tmp/claude-995200228/-home-ashed-Documents-cosmos3-nav/c750294f-69ff-4a05-ba42-f5b120c52f3d/scratchpad/gemini.env}

NICKNAME=ofgemini          # -> output/ofgemini_sam3_gemini_gemini
                           # NOT "of1000": that collides with the corrupted
                           # July of1000_sam3_gemini_gemini directory.
CONFIG=config/navigation_gemini.yaml
EPISODES=28
MAX_STEPS=500

export OPENFRONTIER_DATA_ROOT=$OF/data
# The openfrontier env installs habitat-lab editable from third_party, so the
# config search root is the checkout, NOT site-packages.
export HABITAT_LAB_ROOT=$OF/third_party/habitat-lab-0.2.4/habitat-lab
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

[ -f "$KEY_ENV" ] || { echo "FATAL: no key file at $KEY_ENV"; exit 1; }
set -a; . "$KEY_ENV"; set +a

cd "$OF" || exit 1

# --- preflight: one real Gemini call BEFORE touching habitat -----------------
# benchmark.py only aborts on errors containing "exhausted"; a billing or
# model-availability error is caught as `exception_occurred` and the run keeps
# grinding through episodes writing garbage. That is exactly how the July
# of1000_sam3_gemini_gemini directory got produced. Fail fast instead.
echo "== preflight: verifying gemini-3.1-flash-lite is callable =="
$PY - <<'EOF' || { echo "PREFLIGHT FAILED - not launching. Fix the above, then re-run."; exit 1; }
import os, sys, numpy as np
from google import genai
from google.genai import types
from PIL import Image
try:
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    img = Image.fromarray(np.zeros((480, 960, 3), dtype=np.uint8))
    r = c.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=['Return {"probability": 0.0, "reason": "test"}', img],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    u = r.usage_metadata
    print(f"   OK  in={u.prompt_token_count} out={u.candidates_token_count}")
except Exception as e:
    print("   FAIL:", str(e)[:300]); sys.exit(1)
EOF

# --- SAM3 server ------------------------------------------------------------
if ss -tln 2>/dev/null | grep -q ':12184 '; then
  echo "== SAM3 already listening on 12184 =="
else
  echo "== starting SAM3 server (port 12184) =="
  ( cd "$OF" && nohup "$SAM3_PY" sam3_server.py > "$REPO/eval/sam3_server.log" 2>&1 & )
  for i in $(seq 1 90); do
    ss -tln 2>/dev/null | grep -q ':12184 ' && break
    sleep 5
  done
  ss -tln 2>/dev/null | grep -q ':12184 ' \
    || { echo "FATAL: SAM3 never came up; see eval/sam3_server.log"; exit 1; }
  echo "   SAM3 up"
fi

# --- benchmark --------------------------------------------------------------
echo "== launching HM3D benchmark: $NICKNAME =="
exec $PY benchmark.py \
  --benchmark hm3d \
  --nickname "$NICKNAME" \
  --config "$CONFIG" \
  --output-path output \
  --eval_episodes $EPISODES \
  --max_steps $MAX_STEPS
