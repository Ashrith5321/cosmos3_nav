"""Multi-category SAM3 segmentation for the ASM.

The SAM3 server (`sam3_server.py`) takes one text prompt per request, so a
`k`-category map costs `k` requests per keyframe. That is why the builder only
segments every Nth frame and gates on agent motion.

Unlike the main detection path -- which discards SAM3's confidences at
vlm/utils.py:176-179 -- this module keeps `scores` and thresholds on them.
"""
from typing import Dict, List, Optional

import numpy as np

from vlm.client import VLMClient

from .mapnav import SAM3_PROMPT_NAME


class Sam3MultiSegmenter:
    """Thin, failure-tolerant wrapper around the running SAM3 server."""

    def __init__(self, categories: List[str], port: int,
                 score_threshold: float = 0.5, logger=None):
        self.categories = list(categories)
        self.score_threshold = float(score_threshold)
        self.client = VLMClient("sam3", port=int(port))
        self._log = logger
        self.consecutive_failures = 0

    def _warn(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def segment(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Return {category: bool mask (H, W)} for categories that were found.

        Never raises: a dead or busy server yields an empty dict, and the ASM
        simply keeps its previous semantics.
        """
        out: Dict[str, np.ndarray] = {}
        image = np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8))

        for category in self.categories:
            mask = self._segment_one(image, category)
            if mask is not None and mask.any():
                out[category] = mask
        return out

    def _segment_one(self, image: np.ndarray, category: str) -> Optional[np.ndarray]:
        # MapNav used a COCO Mask R-CNN; SAM3 takes text, so the hyphenated
        # category name is spelled out (see asm/mapnav.py SAM3_PROMPT_NAME).
        prompt = SAM3_PROMPT_NAME.get(category, category)
        try:
            response = self.client.send_request(image=image, prompt=prompt)
            self.consecutive_failures = 0
        except Exception as exc:  # server down, busy, OOM -- all non-fatal here
            self.consecutive_failures += 1
            if self.consecutive_failures in (1, 10, 100):
                self._warn(f"ASM: SAM3 request for '{category}' failed: {exc}")
            return None

        masks = response.get("masks")
        scores = response.get("scores")
        if not masks:
            return None

        masks_arr = np.asarray(masks, dtype=bool)
        if masks_arr.ndim < 2:
            return None
        # SAM3 returns (N, 1, H, W) in practice, but (N, H, W) and a bare
        # (H, W) are both plausible; normalise all three to (N, H, W).
        masks_arr = masks_arr.reshape(-1, *masks_arr.shape[-2:])

        # Keep the confidence the main pipeline throws away.
        if scores is not None and len(scores) == masks_arr.shape[0]:
            scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
            keep = scores_arr >= self.score_threshold
            if not keep.any():
                return None
            masks_arr = masks_arr[keep]

        merged = np.any(masks_arr, axis=0)
        if merged.shape != image.shape[:2]:
            self._warn(
                f"ASM: mask shape {merged.shape} != image {image.shape[:2]}; skipping"
            )
            return None
        return merged
