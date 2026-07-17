"""SFT for the Cosmos3-Nano reasoner, ported from
spatial_training/src/longnav/habitat_training/train_sft.py.

Same recipe: TRL SFTTrainer + LoRA (r=128) on all attention/MLP projections,
bf16, gradient checkpointing, paged 8-bit AdamW, ActionMaskingVLMCollator
with action dropout. The unused PoseTrainer/spatial-head path was dropped.

Local smoke test (RTX 5090 32GB):
    .venv/bin/python sft/train_sft_cosmos3.py --smoke_test

Cluster run:
    python sft/train_sft_cosmos3.py \
        --model_id <path-to-Cosmos3-Nano-or-reasoner-checkpoint> \
        --train_dataset_dir <load_from_disk dir> \
        --eval_dataset_dir <load_from_disk dir> \
        --output_dir <dump dir>
"""
import os
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

from sft.utils.collators import ActionMaskingVLMCollator
from sft.utils.data_misc import make_dynamic_resize_transform

# --- CONFIGURATION (defaults; override via CLI) ---
MODEL_ID = "/home/ashed/Documents/Cosmos3-Nano"
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING = True

SYSTEM_TOKENS = 190
TURN_TOKENS = 30
ORIG_H = 480
ORIG_W = 640
TOTAL_BUDGET = 32000
TRAIN_DATASET_DIR = ""
EVAL_DATASET_DIR = ""
OUTPUT_DIR = str(_ROOT / "dump" / "sft_cosmos3")
EVAL_MAX_SAMPLES = 40


def get_peak_memory_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def parse_args():
    p = argparse.ArgumentParser(description="Cosmos3-Nano reasoner SFT")
    p.add_argument("--model_id", type=str, default=MODEL_ID, help="HF model id or local path")
    p.add_argument("--train_dataset_dir", type=str, default=TRAIN_DATASET_DIR, help="load_from_disk() dir")
    p.add_argument("--eval_dataset_dir", type=str, default=EVAL_DATASET_DIR,
                   help="load_from_disk() dir (empty string disables eval)")
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Trainer output_dir")
    p.add_argument("--eval_max_samples", type=int, default=EVAL_MAX_SAMPLES, help="Eval subset size")
    p.add_argument("--total_budget", type=int, default=TOTAL_BUDGET, help="Token budget for dynamic resize")
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--action_dropout", type=float, default=0.6)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--smoke_test", action="store_true",
                   help="Train a few steps on a tiny synthetic dataset to validate the pipeline")
    p.add_argument("--print_config", action="store_true", help="Print config then exit")
    return p.parse_args()


def build_smoke_dataset(n_samples=4, n_turns=3, h=ORIG_H, w=ORIG_W):
    """Tiny synthetic dataset with the same schema as the real one:
    'messages' (multi-turn chat) + 'images' (one PIL image per user turn)."""
    import numpy as np
    from PIL import Image as PILImage
    from datasets import Dataset

    actions = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    records = []
    for i in range(n_samples):
        messages = [{"role": "system", "content": [
            {"type": "text", "text": "You are a navigation agent. Given the current observation, output the next action."}]}]
        images = []
        for t in range(n_turns):
            messages.append({"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Step {t}: choose the next action."}]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": actions[(i + t) % len(actions)]}]})
            rng = np.random.default_rng(i * 100 + t)
            images.append(PILImage.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)))
        records.append({"messages": messages, "images": images})
    return Dataset.from_list(records)


def main():
    args = parse_args()

    if args.print_config:
        print(vars(args))
        return

    print(f"Model: {args.model_id}")
    print(f"Batch Size: {BATCH_SIZE}")

    # 1. Load model in bfloat16 (no quantization).
    # AutoModelForImageTextToText maps cosmos3_omni -> Cosmos3OmniForConditionalGeneration
    # (reasoner tower only: LM + vision encoder; the diffusion tower lives in diffusers).
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.enable_input_require_grads()
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # 2. LoRA
    print("Applying LoRA...")
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # 3. Data
    dynamic_resize_transform = make_dynamic_resize_transform(
        SYSTEM_TOKENS, TURN_TOKENS, ORIG_H, ORIG_W, args.total_budget - 600)

    if args.smoke_test:
        train_dataset = build_smoke_dataset()
        eval_dataset = None
        args.max_steps = min(args.max_steps, 4)
    else:
        from datasets import load_from_disk
        train_dataset = load_from_disk(args.train_dataset_dir)
        train_dataset.set_transform(dynamic_resize_transform)

        if args.eval_dataset_dir:
            eval_dataset = load_from_disk(args.eval_dataset_dir)
            if args.eval_max_samples is not None and args.eval_max_samples > 0:
                eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))
            eval_dataset.set_transform(dynamic_resize_transform)
        else:
            eval_dataset = None

    # 4. Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        logging_steps=1,
        max_length=None,
        packing=False,  # FALSE is critical to strictly enforce batch_size x seq_len shape
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        max_steps=args.max_steps,
        report_to="tensorboard",
        assistant_only_loss=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        data_collator=ActionMaskingVLMCollator(
            processor=processor,
            max_length=args.total_budget,
            dropout=args.action_dropout,
        ),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,
    )

    # 6. Train
    torch.cuda.reset_peak_memory_stats()
    print("Starting training loop...")
    trainer.train()

    peak_mem = get_peak_memory_gb()
    print("\n" + "=" * 30)
    print(f"Peak VRAM Used: {peak_mem:.2f} GB")
    print("=" * 30)


if __name__ == "__main__":
    main()
salloc --account=entr475s100y26_class \
  --partition=spgpu \
  --nodes=1 \
  --gres=gpu:a40:3 \
  --cpus-per-task=8 \
  --mem=180G \
  --time=4-00:00:00
