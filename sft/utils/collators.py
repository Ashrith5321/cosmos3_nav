"""Collators ported from spatial_training/src/longnav/utils/collators.py.

Token IDs verified identical for the Cosmos3-Nano reasoner tokenizer
(Qwen lineage): <|im_start|>assistant\n = [151644, 77091, 198],
<|im_end|> = 151645, pad = 151643.
"""
import torch
from dataclasses import dataclass
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling


def mask_user_turns(input_ids, pad_token_id=151643):
    """
    input_ids: Tensor of shape (Batch, Seq_Len)
    Returns: labels Tensor of shape (Batch, Seq_Len) with User/System tokens set to -100.
    """
    START_SEQ = torch.tensor([151644, 77091, 198], device=input_ids.device)  # <|im_start|> assistant \n
    END_TOKEN = 151645  # <|im_end|>

    labels = torch.full_like(input_ids, -100)
    batch_size, seq_len = input_ids.shape

    for i in range(batch_size):
        row = input_ids[i]

        # Sliding windows of size 3, check equality against START_SEQ
        windows = row.unfold(0, 3, 1)
        matches = (windows == START_SEQ).all(dim=1).nonzero(as_tuple=True)[0]

        for start_idx in matches:
            content_start = start_idx + 3

            future_tokens = row[content_start:]
            end_offsets = (future_tokens == END_TOKEN).nonzero(as_tuple=True)[0]

            if len(end_offsets) > 0:
                content_end = content_start + end_offsets[0]
                # Include content_end so the model learns to predict <|im_end|>
                labels[i, content_start : content_end + 1] = row[content_start : content_end + 1]
    return labels


def apply_action_dropout(labels, attention_mask, dropout_rate=0.5):
    """
    Randomly drops contiguous chunks of valid labels (Assistant Actions).

    Args:
        labels: (Batch, Seq) Tensor. User turns must already be -100.
        attention_mask: (Batch, Seq) Tensor. Modified in-place.
        dropout_rate: Probability of dropping an action chunk.

    Returns:
        labels: New labels tensor with dropped chunks set to -100.
    """
    new_labels = labels.clone()
    batch_size, seq_len = labels.shape

    for i in range(batch_size):
        row_labels = new_labels[i]

        valid_mask = (row_labels != -100)
        if not valid_mask.any():
            continue

        valid_int = valid_mask.int()
        padded = torch.cat([torch.tensor([0], device=labels.device), valid_int, torch.tensor([0], device=labels.device)])
        diffs = padded.diff()

        starts = (diffs == 1).nonzero(as_tuple=True)[0]
        ends = (diffs == -1).nonzero(as_tuple=True)[0]
        assert len(starts) == len(ends)

        for start, end in zip(starts, ends):
            if torch.rand(1) < dropout_rate:
                new_labels[i, start:end] = -100
                attention_mask[i, start:end] = 0

    return new_labels


@dataclass
class ActionMaskingVLMCollator(DataCollatorForVisionLanguageModeling):
    """
    Extension of TRL's VLM collator that masks ALL User/System tokens in a multi-turn conversation.
    It identifies Assistant turns based on specific start/end token IDs and masks everything else.
    """
    dropout: float = -1
    length_warning: int = 40000

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "image_grid_thw" not in batch:
            print("!!! CRITICAL ERROR: 'image_grid_thw' MISSING in batch! Model is blind! !!!")

        input_ids = batch["input_ids"]
        if input_ids.shape[1] >= self.length_warning:
            print(f"warning! sequence too long! input ids: {input_ids.shape}")

        labels = mask_user_turns(input_ids)

        if self.dropout > 0:
            labels = apply_action_dropout(
                labels,
                batch["attention_mask"],
                dropout_rate=self.dropout,
            )

        # Padding positions (attention_mask == 0) must stay -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch
