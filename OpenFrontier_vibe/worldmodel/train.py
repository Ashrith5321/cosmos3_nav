"""
Train FrontierWorldModelNet on recorded beyond-frontier data
(design sections 26-27; multi-hypothesis winner-takes-all objectives).

Data comes from episodes run with world_model.record_dataset: true, which
write <save_dir>/wm_dataset/beyond_frontier.npz per episode.

Usage:
    python -m worldmodel.train \
        --data 'output/*/*/wm_dataset/beyond_frontier.npz' \
        --out model_weights/frontier_wm.pth --epochs 50

The winner-takes-all scheme trains, per sample, only the hypothesis whose
predicted embedding best matches the realized future (keeping hypothesis
diversity), while the weight head learns which hypothesis wins - the
standard multi-choice-learning recipe for multi-modal futures.
"""

import argparse
import glob
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from worldmodel.dataset import BeyondFrontierDataset
from worldmodel.net import FrontierWorldModelNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate(batch):
    return {
        k: torch.as_tensor(np.stack([b[k] for b in batch]), dtype=torch.float32)
        for k in batch[0]
    }


def loss_fn(out: dict, batch: dict, lambdas: dict) -> dict:
    """Multi-task, winner-takes-all across the K hypotheses (section 26)."""
    z_gt = F.normalize(batch["future_embedding"], dim=-1)          # (B, D)
    emb = out["embedding"]                                          # (B, K, D)
    cos = torch.einsum("bkd,bd->bk", emb, z_gt)                     # (B, K)

    winner = cos.argmax(dim=1)                                      # (B,)
    B = cos.shape[0]
    idx = torch.arange(B, device=cos.device)

    l_embed = (1.0 - cos[idx, winner]).mean()

    room_logits = out["room_logits"][idx, winner]                   # (B, R)
    room_target = batch["room_label"].argmax(dim=-1)
    l_room = F.cross_entropy(room_logits, room_target)

    object_logits = out["object_logits"][idx, winner]               # (B, O)
    l_objects = F.binary_cross_entropy_with_logits(
        object_logits, batch["object_labels"]
    )

    gain = out["gain"][idx, winner]
    l_gain = F.smooth_l1_loss(torch.log1p(gain), torch.log1p(batch["gain_label"]))

    # the weight head predicts which hypothesis wins
    l_weight = F.cross_entropy(torch.log(out["weights"] + 1e-8), winner)

    # diversity: discourage hypothesis collapse (mean pairwise embedding cosine)
    K = emb.shape[1]
    if K > 1:
        sim = torch.einsum("bkd,bjd->bkj", emb, emb)
        off_diag = sim - torch.eye(K, device=sim.device)[None]
        l_div = off_diag.clamp(min=0).sum(dim=(1, 2)).mean() / (K * (K - 1))
    else:
        l_div = torch.zeros((), device=emb.device)

    total = (
        lambdas["embed"] * l_embed
        + lambdas["room"] * l_room
        + lambdas["objects"] * l_objects
        + lambdas["gain"] * l_gain
        + lambdas["weight"] * l_weight
        + lambdas["diversity"] * l_div
    )
    return {
        "total": total,
        "embed": l_embed.detach(),
        "room": l_room.detach(),
        "objects": l_objects.detach(),
        "gain": l_gain.detach(),
        "weight": l_weight.detach(),
        "diversity": l_div.detach(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="glob of beyond_frontier.npz files")
    p.add_argument("--out", type=str, required=True, help="checkpoint output path")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hypotheses", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lambda-embed", type=float, default=1.0)
    p.add_argument("--lambda-room", type=float, default=0.5)
    p.add_argument("--lambda-objects", type=float, default=0.5)
    p.add_argument("--lambda-gain", type=float, default=0.25)
    p.add_argument("--lambda-weight", type=float, default=0.25)
    p.add_argument("--lambda-diversity", type=float, default=0.1)
    args = p.parse_args()

    paths = sorted(glob.glob(args.data))
    if not paths:
        raise SystemExit(f"No dataset files match {args.data}")
    logger.info("Loading %d dataset files", len(paths))
    dataset = BeyondFrontierDataset(paths)
    logger.info("Total samples: %d", len(dataset))

    n_val = max(int(len(dataset) * args.val_frac), 1)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = dataset.data["future_embedding"].shape[1]
    net = FrontierWorldModelNet(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_hypotheses=args.num_hypotheses,
        num_layers=args.num_layers,
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    lambdas = {
        "embed": args.lambda_embed,
        "room": args.lambda_room,
        "objects": args.lambda_objects,
        "gain": args.lambda_gain,
        "weight": args.lambda_weight,
        "diversity": args.lambda_diversity,
    }

    best_val = float("inf")
    for epoch in range(args.epochs):
        net.train()
        train_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = net(batch["crop_embedding"], batch["scene_embedding"], batch["geom"])
            losses = loss_fn(out, batch, lambdas)
            opt.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            train_losses.append(float(losses["total"]))

        net.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = net(
                    batch["crop_embedding"], batch["scene_embedding"], batch["geom"]
                )
                val_losses.append(float(loss_fn(out, batch, lambdas)["total"]))

        tr, va = float(np.mean(train_losses)), float(np.mean(val_losses))
        logger.info("epoch %03d train %.4f val %.4f", epoch, tr, va)

        if va < best_val:
            best_val = va
            torch.save(
                {"state_dict": net.state_dict(), "net_kwargs": net.net_kwargs},
                args.out,
            )
            logger.info("saved checkpoint to %s (val %.4f)", args.out, va)

    logger.info("done; best val %.4f", best_val)


if __name__ == "__main__":
    main()
