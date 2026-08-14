"""
Frozen vision/text encoders providing the shared semantic embedding space.

Backends:
- "clip": HuggingFace transformers CLIP (default openai/clip-vit-base-patch32).
- "dummy": deterministic hash-based embeddings for tests / dry runs
  (no torch/transformers required).

All embeddings are unit-normalized so cosine similarity is a dot product.
"""

import hashlib
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseEncoder:
    embed_dim: int = 512

    def encode_images(self, images: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]


def _l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-8)


class ClipEncoder(BaseEncoder):
    """Frozen CLIP with an in-memory text cache."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP encoder %s on %s", model_name, self.device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim = int(self.model.config.projection_dim)
        self._text_cache = {}

    def encode_images(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        pil_ready = [np.ascontiguousarray(img[..., :3]) for img in images]
        inputs = self.processor(images=pil_ready, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return _l2norm(feats.float().cpu().numpy())

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            inputs = self.processor(
                text=missing, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with self.torch.no_grad():
                feats = self.model.get_text_features(**inputs)
            feats = _l2norm(feats.float().cpu().numpy())
            for t, f in zip(missing, feats):
                self._text_cache[t] = f
        return np.stack([self._text_cache[t] for t in texts], axis=0)


class DummyEncoder(BaseEncoder):
    """Deterministic pseudo-embeddings; images hash their bytes, texts their words.

    Word-level hashing gives texts sharing words correlated embeddings, so
    cosine matching remains meaningful enough for unit tests.
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim

    def _vec_from_seed(self, seed: bytes) -> np.ndarray:
        h = int.from_bytes(hashlib.md5(seed).digest()[:8], "little")
        rng = np.random.default_rng(h)
        return rng.standard_normal(self.embed_dim).astype(np.float32)

    def encode_images(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out = []
        for img in images:
            arr = np.asarray(img)
            # subsample for speed and stability
            sig = arr[:: max(arr.shape[0] // 8, 1), :: max(arr.shape[1] // 8, 1)]
            out.append(self._vec_from_seed(sig.tobytes()))
        return _l2norm(np.stack(out, axis=0))

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out = []
        for t in texts:
            words = t.lower().replace(",", " ").split()
            vecs = [self._vec_from_seed(w.encode()) for w in words] or [
                self._vec_from_seed(t.encode())
            ]
            out.append(np.mean(vecs, axis=0))
        return _l2norm(np.stack(out, axis=0))


def build_encoder(params: Optional[dict] = None) -> BaseEncoder:
    p = params or {}
    backend = p.get("backend", "clip")
    if backend == "dummy":
        return DummyEncoder(embed_dim=int(p.get("embed_dim", 64)))
    if backend == "clip":
        return ClipEncoder(
            model_name=p.get("model_name", "openai/clip-vit-base-patch32"),
            device=p.get("device"),
        )
    raise ValueError(f"Unknown encoder backend: {backend}")
