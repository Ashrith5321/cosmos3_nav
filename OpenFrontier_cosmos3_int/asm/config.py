"""Configuration for the annotated semantic map (ASM) side-channel.

Every default here is chosen so that enabling the ASM cannot slow the navigation
loop: the agent thread only ever does a non-blocking queue put, and the worker
drops frames when it falls behind.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import os

from .mapnav import MAPNAV_CATEGORIES

# MapNav's eleven categories, in palette order. Kept as the default so the ASM
# is directly comparable to the paper. SAM3 is prompted once per entry, so cost
# is linear in this list -- trim it for cheaper runs at the price of fidelity.
DEFAULT_CATEGORIES: List[str] = list(MAPNAV_CATEGORIES)


@dataclass
class ASMConfig:
    # --- what to look for ---
    categories: List[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    sam3_score_threshold: float = 0.5   # SAM3 returns per-instance scores; unlike
                                        # the main pipeline we actually read them

    # --- grid geometry (world frame is Z-up; see utils/transform.py) ---
    resolution_m: float = 0.05          # metres per cell
    extent_m: float = 40.0              # square map side; origin anchored at the
                                        # first observed pose
    max_depth_m: float = 3.5            # matches config_utils.py:94 sensor clip
    min_depth_m: float = 0.3

    # Height bands, relative to the floor level under the camera.
    floor_band_m: float = 0.25          # points below this count as free space
    obstacle_band_lo_m: float = 0.25
    obstacle_band_hi_m: float = 1.8     # matches filter_bbox z-range in the yaml

    # --- cadence: how often a queued frame is actually segmented ---
    segment_every_n_frames: int = 8     # geometry is integrated on every frame;
                                        # semantics only every Nth
    min_translation_m: float = 0.25     # skip segmentation if the agent has not
    min_rotation_deg: float = 20.0      # meaningfully moved since the last one

    # --- threading ---
    queue_size: int = 12                # the worker drains the whole queue each
                                        # cycle, so this is headroom to ride out
                                        # a slow segmentation pass without
                                        # dropping geometry (~2 MB per slot)
    sam3_port: int = int(os.environ.get("ASM_SAM3_PORT",
                                        os.environ.get("OF_SAM3_PORT", "12184")))
    sam3_timeout_s: float = 20.0

    # --- annotation ---
    # With mapnav_faithful the annotation is the port in asm/mapnav.py: render
    # the local window through MapNav's palette, decode colours back to names
    # with a +/-5 tolerance, and apply its `area < 50` px filter (toilet and
    # potted plant exempted). Set False for the exact-grid variant, which skips
    # the colour round-trip and is not capped at eleven categories.
    mapnav_faithful: bool = True
    local_window_cells: int = 240       # MapNav's local map window; with
                                        # render_size 480 this reproduces its
                                        # 2x upscale, so `area < 50` px means
                                        # the same thing it does there
    render_size: int = 480
    min_blob_area_cells: int = 12       # exact-grid path only (~0.03 m^2)
    min_votes_per_cell: int = 2         # a cell needs this many hits to be labelled
    font_size: int = 11                 # huatu3.py:82 uses 11

    # --- output ---
    out_subdir: str = "asm"
    write_every_n_updates: int = 5      # PNG/JSON snapshot cadence
    keep_snapshots: bool = False        # False -> only latest_asm.{png,json}
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be > 0")
        if self.extent_m <= 0:
            raise ValueError("extent_m must be > 0")

    @property
    def grid_size(self) -> int:
        return int(round(self.extent_m / self.resolution_m))

    @classmethod
    def from_env(cls, **overrides) -> "ASMConfig":
        """Build a config, letting a few knobs be set without touching code."""
        kwargs = {}
        cats = os.environ.get("ASM_CATEGORIES")
        if cats:
            kwargs["categories"] = [c.strip() for c in cats.split(",") if c.strip()]
        res = os.environ.get("ASM_RESOLUTION_M")
        if res:
            kwargs["resolution_m"] = float(res)
        every = os.environ.get("ASM_SEGMENT_EVERY")
        if every:
            kwargs["segment_every_n_frames"] = int(every)
        kwargs.update(overrides)
        return cls(**kwargs)


def asm_enabled() -> Optional[bool]:
    """True when OF_ASM is set to a truthy value."""
    raw = os.environ.get("OF_ASM", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")
