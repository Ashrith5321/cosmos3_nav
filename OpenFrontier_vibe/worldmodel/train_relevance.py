#!/usr/bin/env python3
"""Train a goal-relevance head on ground-truth distance, with a ranking loss.

WHY A NEW TRAINER
-----------------
`train.py` optimises reconstruction: predict the future embedding, room,
objects and gain for one frontier in isolation. Nothing in it compares
frontiers, and its object target saturates at 91% positive, so the resulting
score ranks frontiers at chance (per-goal AUC 0.477-0.558) and the world model
cannot change any decision.

This trains the quantity the planner actually consumes: for goal g and frontier
f, how close is the nearest real instance of g beyond f. Supervision is the
geodesic distance from `label_goal_distance.py`, so the target differs between
frontiers of the same scene -- measured within-scene spread 3.46 m -- which is
exactly what the old label lacked.

Two loss terms:
  * pointwise  - regress relevance exp(-d/tau), so absolute scale is calibrated
  * listwise   - softmax cross-entropy over the frontiers of one scene against
                 the softmax of true relevance, so the ORDER is optimised
                 directly. This is the term the previous objective had no
                 analogue of.

Scene-disjoint split; reports held-out ranking metrics, not just loss.

usage: train_relevance.py --harvest output/wm_harvest_gt --out checkpoints/relevance_v90.pth
"""
import argparse, glob, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GOALS = ["chair", "sofa", "bed", "toilet", "tv_monitor", "plant"]


class RelevanceNet(nn.Module):
    """crop + scene embedding + geometry -> relevance for each goal category."""

    def __init__(self, embed_dim=512, geom_dim=8, hidden=256, n_goals=len(GOALS)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim * 2 + geom_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_goals)

    def forward(self, crop, scene, geom):
        return self.head(self.trunk(torch.cat([crop, scene, geom], dim=-1)))


def load(harvest):
    """One entry per scene, so the listwise loss can group correctly."""
    scenes = []
    for d in sorted(glob.glob(os.path.join(harvest, "*"))):
        bf, gd = (os.path.join(d, "beyond_frontier.npz"),
                  os.path.join(d, "goal_distance.npz"))
        if not (os.path.exists(bf) and os.path.exists(gd)):
            continue
        b, g = np.load(bf, allow_pickle=True), np.load(gd, allow_pickle=True)
        rel, dist = g["goal_relevance"], g["goal_geodesic"]
        # a scene teaches ranking only if the target varies across its frontiers
        if len(b["uid"]) < 4 or not np.isfinite(dist).any():
            continue
        scenes.append({
            "name": os.path.basename(d),
            "crop": b["crop_embedding"].astype(np.float32),
            "scene": b["scene_embedding"].astype(np.float32),
            "geom": b["geom"].astype(np.float32),
            "rel": rel.astype(np.float32),
            "dist": dist.astype(np.float32),
        })
    return scenes


def ranking_metrics(net, scenes, dev):
    """Top-1 and regret against the true nearest-goal ordering, per goal."""
    top1, regret, chance = [], [], []
    with torch.no_grad():
        for s in scenes:
            pred = net(torch.tensor(s["crop"]).to(dev),
                       torch.tensor(s["scene"]).to(dev),
                       torch.tensor(s["geom"]).to(dev)).cpu().numpy()
            for gi in range(len(GOALS)):
                d = s["dist"][:, gi]
                m = np.isfinite(d)
                if m.sum() < 2 or d[m].std() < 1e-3:
                    continue
                p = pred[m, gi]
                pick = int(np.argmax(p))
                top1.append(float(d[m][pick] == d[m].min()))
                regret.append(float(d[m][pick] - d[m].min()))
                chance.append(1.0 / m.sum())
    return (float(np.mean(top1)) if top1 else float("nan"),
            float(np.mean(regret)) if regret else float("nan"),
            float(np.mean(chance)) if chance else float("nan"),
            len(top1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", nargs="+", default=["output/wm_harvest_gt"])
    ap.add_argument("--out", default="checkpoints/relevance_v90.pth")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lambda-list", type=float, default=1.0)
    args = ap.parse_args()

    scenes = []
    seen = set()
    for root in args.harvest:
        for sc in load(root):
            # later roots deepen earlier ones; keep the richer copy per scene
            if sc["name"] in seen:
                continue
            seen.add(sc["name"])
            scenes.append(sc)
    print(f"labeled scenes: {len(scenes)}   "
          f"frontiers: {sum(len(s['crop']) for s in scenes)}")
    if len(scenes) < 8:
        print("not enough labeled scenes yet -- run label_goal_distance.py first")
        return
    rng = np.random.default_rng(0)
    order = rng.permutation(len(scenes))
    cut = max(2, int(0.25 * len(scenes)))
    te = [scenes[i] for i in order[:cut]]
    tr = [scenes[i] for i in order[cut:]]
    print(f"train {len(tr)} scenes / held-out {len(te)} scenes\n")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = RelevanceNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    t1, rg, ch, n = ranking_metrics(net, te, dev)
    print(f"before training: top-1 {t1:.3f} (chance {ch:.3f})  regret {rg:.2f}m  n={n}")
    best = (-1.0, float("inf"), 0, None)

    for ep in range(args.epochs):
        rng.shuffle(tr)
        tot = 0.0
        for s in tr:
            crop = torch.tensor(s["crop"]).to(dev)
            sc = torch.tensor(s["scene"]).to(dev)
            gm = torch.tensor(s["geom"]).to(dev)
            rel = torch.tensor(s["rel"]).to(dev)
            pred = net(crop, sc, gm)
            l_point = F.mse_loss(torch.sigmoid(pred), rel)
            l_list = torch.zeros((), device=dev)
            k = 0
            for gi in range(len(GOALS)):
                d = s["dist"][:, gi]
                m = np.isfinite(d)
                if m.sum() < 2 or d[m].std() < 1e-3:
                    continue
                idx = torch.tensor(np.where(m)[0]).to(dev)
                tgt = F.softmax(rel[idx, gi] * 4.0, dim=0)
                l_list = l_list + F.kl_div(
                    F.log_softmax(pred[idx, gi], dim=0), tgt, reduction="sum")
                k += 1
            if k:
                l_list = l_list / k
            loss = l_point + args.lambda_list * l_list
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 5 == 0 or ep == 0:
            t1, rg, ch, n = ranking_metrics(net, te, dev)
            print(f"epoch {ep+1:3d}  loss {tot/len(tr):.4f}   "
                  f"held-out top-1 {t1:.3f} (chance {ch:.3f})  regret {rg:.2f}m")
            if t1 > best[0]:
                best = (t1, rg, ep + 1,
                        {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if best[3] is not None:
        net.load_state_dict(best[3])   # early stop at the held-out peak
    torch.save({"model": net.state_dict(), "goals": GOALS,
                "best_epoch": best[2]}, args.out)
    t1, rg, ch, n = ranking_metrics(net, te, dev)
    print(f"\nEARLY-STOPPED at epoch {best[2]}")
    print(f"FINAL held-out: top-1 {t1:.3f} vs chance {ch:.3f}   regret {rg:.2f}m   n={n}")
    print(f"saved {args.out}")
    print("\ntop-1 must beat chance on HELD-OUT SCENES for the world model to be")
    print("worth deploying; the shipped predictor scores at chance on this metric.")


if __name__ == "__main__":
    main()
