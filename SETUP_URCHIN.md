# Task: Set up the Cosmos3 navigation stack on this machine (urchin, 2x RTX A6000 48GB, Ubuntu 22.04, NVIDIA driver 550 = CUDA 12.x)

Everything was rsynced from the old machine (ROB-TIAN-05U, RTX 5090) into ~/Documents. Nothing is
installed yet - both Python environments must be rebuilt from the frozen specs.

## What's already on disk (do NOT re-download)
- ~/Documents/Cosmos3-Nano - full nvidia/Cosmos3-Nano checkpoint (33 GB, reasoner + generator towers)
- ~/Documents/cosmos3_nav - main repo:
  - sft/ - LoRA SFT of the reasoner (train_sft_cosmos3.py + utils/)
  - eval/ - closed-loop HM3D eval stack: cosmos3_server.py (port 8399), run_habitat_smoke_eval.py,
    run_habitat_map_eval.py, vram_watchdog.sh, plus all past results (habitat_*_out/, *.jsonl)
  - src/ - frontiernet_server.py (port 12186), global_map.py, etc.
  - OpenFrontier/ + FrontierNet/ + model_weights/ - learned frontier detector + weights
  - requirements-venv.txt (pip freeze of the model venv)
  - habitat033-env.yml + requirements-habitat033-pip.txt (conda env export + pip freeze)
- ~/Documents/spatial_training - lab code the drivers import (src/longnav), habitat_configs/,
  data/datasets/objectnav/hm3d/v2 (episodes), data/scene_datasets/hm3d_v0.2 (val scenes, dereferenced)

## Step 1 - model venv (cosmos3_nav/.venv)
- Install uv if missing (curl -LsSf https://astral.sh/uv/install.sh | sh).
- cd ~/Documents/cosmos3_nav && uv venv .venv --python 3.12 --seed
- IMPORTANT: driver 550 -> CUDA 12.x. Install torch with cu128 backend, NOT the cu130 in the freeze:
  uv pip install --torch-backend=cu128 torch torchvision
- Install the rest pinned from requirements-venv.txt (skip/adjust torch lines): transformers>=5.11
  (5.13.0 in freeze; has Cosmos3OmniForConditionalGeneration), trl==1.8.0, peft, datasets, accelerate,
  bitsandbytes, pillow, av, tensorboard, flask, opencv-python-headless, segmentation-models-pytorch.
- Verify: .venv/bin/python -c "from transformers import Cosmos3OmniForConditionalGeneration; import torch; print(torch.cuda.get_device_name(0))"

## Step 2 - habitat conda env
- Install Miniconda to ~/miniconda3 if missing.
- conda env create -n habitat033 -f ~/Documents/cosmos3_nav/habitat033-env.yml
  (if exact pins fail, relax build strings but keep habitat-sim 0.3.3 withbullet headless,
  then pip install -r requirements-habitat033-pip.txt)
- Verify: ~/miniconda3/envs/habitat033/bin/python -c "import sys; sys.path.insert(0,'/home/ashed/Documents/spatial_training/src'); import habitat, requests; import longnav.utils.measures; print(habitat.__version__)"  # 0.3.3

## Step 3 - smoke tests (in order)
1. Model loads on GPU 0 (bf16, ~17.6 GB): AutoModelForImageTextToText.from_pretrained('/home/ashed/Documents/Cosmos3-Nano', dtype=bfloat16, device_map={'':0})
2. SFT: .venv/bin/python sft/train_sft_cosmos3.py --smoke_test --lora_r 16   # 4 steps, loss falls, <21 GB
3. Servers: nohup .venv/bin/python eval/cosmos3_server.py > eval/server.log 2>&1 &
            nohup .venv/bin/python src/frontiernet_server.py > eval/frontiernet.log 2>&1 &
   curl http://127.0.0.1:8399/health ; port 12186 listening
4. One episode: ~/miniconda3/envs/habitat033/bin/python eval/run_habitat_smoke_eval.py --scene 4ok3usBNeis --limit 1 --max-steps 20

## Step 4 - resume the interrupted experiment
Paused: map+frontier eval on 3 hardest scenes, 48/84 done, 8 wins, results in
eval/habitat_3scene_lowsr_map_out/results.jsonl. With BOTH servers up:
  ~/miniconda3/envs/habitat033/bin/python eval/run_habitat_map_eval.py \
    --scene DYehNKdT76V,GLAQ4DNUx5U,eF36g7L6Z9M \
    --skip-results /home/ashed/Documents/cosmos3_nav/eval/habitat_3scene_lowsr_map_out/results.jsonl \
    --output-dir /home/ashed/Documents/cosmos3_nav/eval/habitat_3scene_lowsr_map_out
Then report matched comparison vs eval/hm3dv2_val1000_cosmos3_h16.jsonl (same episode keys).

## Known gotchas (all bit us before)
- Both drivers os.chdir() to ~/Documents/spatial_training at import -> ALWAYS absolute paths
  for --skip-results / --output-dir.
- pgrep/pkill -f patterns match their own wrapper shell -> kill by exact PID; cross-check PIDs
  against nvidia-smi --query-compute-apps.
- vram_watchdog.sh kills the server above 20000 MiB (arg 2); raise on 48 GB cards but keep it running.
- Two GPUs: model server on GPU 0 (CUDA_VISIBLE_DEVICES=0), habitat rendering on GPU 1
  (habitat_sim_v0.gpu_device_id in build_config).
- Never run two drivers on the same --output-dir (driver.lock enforces for the map driver).
- If FrontierNet 500s, check eval/frontiernet.log - the driver silently disables it after the
  first failure and the run degrades to geometric frontiers only.

Report: torch/transformers versions, 4 smoke test results, VRAM readings, resumed-eval status.
