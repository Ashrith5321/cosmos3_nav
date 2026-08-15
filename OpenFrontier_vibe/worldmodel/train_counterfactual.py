#!/usr/bin/env python3
"""Action-conditioned counterfactual model + the gates that decide if it is real.

The previous formulation predicted p(object beyond frontier | crop) -- semantic
guessing, 91% positive, chance-level ranking. This predicts the CONSEQUENCE of
executing a specific trajectory:

    f(state, action_i) -> (delta_goal_distance, new_area, stuck)

where the action is the executed pose sequence, not a frontier id.

Two gates decide whether this is foresight or bookkeeping:

  RANKING   within-decision top-1 against the branch that truly reduced goal
            distance most, on HELD-OUT SCENES, versus chance.

  ACTION    the shuffle test. Re-score each decision with another branch's
            action sequence. If ranking barely moves, the model is reading
            state (or trajectory geometry) and ignoring the action -- which is
            the exact failure that killed the previous world model, and no
            amount of downstream tuning fixes it.

usage: train_counterfactual.py --data output/cf --epochs 40
"""
import argparse, glob, os, sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(root):
    """Group branches by decision; scenes stay separable for the split."""
    groups = []
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        scene = os.path.basename(f)[:-4]
        d = np.load(f, allow_pickle=True)
        nbr = d["n_branch"]
        emb, occ = d["emb_mean"], d["occ"]
        dd, na, st = d["d_delta"], d["new_area"], d["stuck"]
        cands, p0 = d["cands"], d["p0"]
        i = 0
        for gi, k in enumerate(nbr):
            k = int(k)
            if k >= 2:
                groups.append({
                    "scene": scene,
                    "emb": emb[i:i + k].astype(np.float32),
                    "occ": occ[i:i + k].astype(np.float32),
                    # the ACTION is the executed trajectory, not the endpoint.
                    # relative to its own first pose so it encodes shape, not
                    # absolute position (which the state already carries).
                    "act": (d["acts"][i:i + k]
                            - d["acts"][i:i + k][:, :1, :]).astype(np.float32)
                            if "acts" in d else
                            (cands[i:i + k] - p0[gi]).astype(np.float32),
                    "dd": dd[i:i + k].astype(np.float32),
                    "na": na[i:i + k].astype(np.float32),
                    "st": st[i:i + k].astype(np.float32),
                })
            i += k
    return groups


class CFNet(nn.Module):
    """(state, action) -> predicted consequences. Action enters explicitly."""

    def __init__(self, hidden=256):
        super().__init__()
        self.occ = nn.Sequential(
            nn.Conv2d(1, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2, 2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(32 * 16, 128), nn.ReLU())
        # GRU over the executed pose sequence: a 3-dim endpoint offset is
        # barely more than the frontier id, which is exactly what the spec says
        # the action must not be.
        self.act_rnn = nn.GRU(3, 64, batch_first=True)
        self.act = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(128 + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, occ, act):
        z = self.occ(occ.unsqueeze(1))
        if act.dim() == 3:
            _, h = self.act_rnn(act)
            a = self.act(h[-1])
        else:
            a = self.act(F.pad(act, (0, 61)))
        return self.head(torch.cat([z, a], -1))     # dd, na, stuck


def rank_metrics(net, groups, dev, shuffle=False, rng=None):
    """Top-1 on the branch that truly reduced goal distance most."""
    hit, chance = [], []
    with torch.no_grad():
        for g in groups:
            occ = torch.tensor(g["occ"]).to(dev)
            act = torch.tensor(g["act"]).to(dev)
            if shuffle:                       # break the state-action pairing
                perm = torch.tensor(rng.permutation(len(act))).to(dev)
                act = act[perm]
            p = net(occ, act)[:, 0].cpu().numpy()
            truth = g["dd"]
            if truth.std() < 1e-3:
                continue
            hit.append(float(truth[int(np.argmax(p))] == truth.max()))
            chance.append(1.0 / len(truth))
    return (float(np.mean(hit)) if hit else float("nan"),
            float(np.mean(chance)) if chance else float("nan"), len(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/cf")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="checkpoints/cf_v100.pth")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    groups = load(args.data)
    scenes = sorted({g["scene"] for g in groups})
    print(f"decisions={len(groups)}  scenes={len(scenes)}")

    results = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        sc = rng.permutation(scenes)
        a, b = int(0.2 * len(sc)), int(0.4 * len(sc))
        te = {s for s in sc[:a]}; va = {s for s in sc[a:b]}
        TE = [g for g in groups if g["scene"] in te]
        VA = [g for g in groups if g["scene"] in va]
        TR = [g for g in groups if g["scene"] not in te and g["scene"] not in va]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(seed)
        net = CFNet().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        best = (-1, None)
        for ep in range(args.epochs):
            rng.shuffle(TR)
            for g in TR:
                occ = torch.tensor(g["occ"]).to(dev)
                act = torch.tensor(g["act"]).to(dev)
                tgt = torch.tensor(np.stack([g["dd"], g["na"] / 100.0, g["st"]], 1)).to(dev)
                pred = net(occ, act)
                loss = F.smooth_l1_loss(pred, tgt)
                # within-decision ranking on the goal-distance channel
                if g["dd"].std() > 1e-3:
                    t = F.softmax(torch.tensor(g["dd"]).to(dev) * 2.0, 0)
                    loss = loss + F.kl_div(F.log_softmax(pred[:, 0], 0), t, reduction="sum")
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 5 == 0:
                v, _, _ = rank_metrics(net, VA, dev)
                if v > best[0]:
                    best = (v, {k: t.detach().cpu().clone()
                                for k, t in net.state_dict().items()})
        if best[1]:
            net.load_state_dict(best[1])
        t1, ch, n = rank_metrics(net, TE, dev)
        sh, _, _ = rank_metrics(net, TE, dev, shuffle=True,
                                rng=np.random.default_rng(seed + 99))
        results.append((t1, ch, sh))
        print(f"  seed {seed}: TEST top-1 {t1:.3f} (chance {ch:.3f})  "
              f"action-shuffled {sh:.3f}  n={n}")

    t = np.array([r[0] for r in results]); c = np.array([r[1] for r in results])
    s = np.array([r[2] for r in results])
    print(f"\nTEST top-1        {t.mean():.3f} +/- {t.std():.3f}   chance {c.mean():.3f}")
    print(f"action-shuffled   {s.mean():.3f} +/- {s.std():.3f}")
    print(f"lift over chance  {t.mean()/max(c.mean(),1e-9):.2f}x")
    print(f"action sensitivity (real - shuffled)  {t.mean()-s.mean():+.3f}")
    print("\nGATE: lift must clear chance AND shuffling must collapse it.")
    print("If shuffled ~= real, the model ignores the action and this is not")
    print("action-conditioned foresight, whatever the ranking number says.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": net.state_dict()}, args.out)


if __name__ == "__main__":
    main()
