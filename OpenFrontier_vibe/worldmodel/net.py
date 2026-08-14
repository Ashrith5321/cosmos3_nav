"""
FrontierWorldModelNet: small frontier-conditioned transformer with K
hypothesis queries and multi-head outputs (design sections 6.4, 22, 26).

Inputs are frozen-encoder features (crop + scene embeddings) plus a
geometric feature vector; outputs, per hypothesis query:
  - future semantic embedding (unit-norm, encoder space)
  - room distribution, object probabilities
  - information gain (m^3), hypothesis weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from worldmodel.predictor import GEOM_DIM
from worldmodel.vocab import OBJECT_VOCAB, ROOM_TYPES


class FrontierWorldModelNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_hypotheses: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        geom_dim: int = GEOM_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hypotheses = num_hypotheses

        self.crop_proj = nn.Linear(embed_dim, hidden_dim)
        self.scene_proj = nn.Linear(embed_dim, hidden_dim)
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.queries = nn.Parameter(torch.randn(num_hypotheses, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.emb_head = nn.Linear(hidden_dim, embed_dim)
        self.room_head = nn.Linear(hidden_dim, len(ROOM_TYPES))
        self.object_head = nn.Linear(hidden_dim, len(OBJECT_VOCAB))
        self.gain_head = nn.Linear(hidden_dim, 1)
        self.weight_head = nn.Linear(hidden_dim, 1)

    @property
    def net_kwargs(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.crop_proj.out_features,
            "num_hypotheses": self.num_hypotheses,
            "num_layers": len(self.transformer.layers),
            "num_heads": self.transformer.layers[0].self_attn.num_heads,
        }

    def forward(
        self,
        crop_emb: torch.Tensor,   # (B, D)
        scene_emb: torch.Tensor,  # (B, D)
        geom: torch.Tensor,       # (B, GEOM_DIM)
    ) -> dict:
        B = crop_emb.shape[0]
        tokens = torch.stack(
            [self.crop_proj(crop_emb), self.scene_proj(scene_emb), self.geom_proj(geom)],
            dim=1,
        )  # (B, 3, H)
        queries = self.queries[None].expand(B, -1, -1)  # (B, K, H)
        x = torch.cat([tokens, queries], dim=1)
        x = self.transformer(x)
        q = x[:, tokens.shape[1] :]  # (B, K, H)

        emb = F.normalize(self.emb_head(q), dim=-1)          # (B, K, D)
        room_logits = self.room_head(q)                      # (B, K, R)
        object_logits = self.object_head(q)                  # (B, K, O)
        gain = F.softplus(self.gain_head(q)).squeeze(-1)     # (B, K)
        weights = torch.softmax(self.weight_head(q).squeeze(-1), dim=-1)  # (B, K)

        return {
            "embedding": emb,
            "room_logits": room_logits,
            "room_probs": torch.softmax(room_logits, dim=-1),
            "object_logits": object_logits,
            "object_probs": torch.sigmoid(object_logits),
            "gain": gain,
            "weights": weights,
        }
