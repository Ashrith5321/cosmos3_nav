"""Faithful port of MapNav's annotated-semantic-map format (ACL 2025).

Reproduces, exactly:

* the label indexing of `MapNav/r2rnav_agent_nohis.py:495-511`
  (`sem_map += 5`; 0 unexplored, 1 obstacle, 2 explored, 3 visited/edge,
  5+ semantic),
* the `color_palette` of `MapNav/constants.py:96-117`,
* the eleven categories `MapNav/huatu3.py:23-35` decodes,
* the annotation pass of `MapNav/huatu3.py:20-122` -- colour decode with a
  +/-5 tolerance, `connectedComponentsWithStats`, the `area < 50` filter with
  its toilet / potted-plant exemptions, and the orange rounded label boxes,
* the prompt sentence of `MapNav/r2rnav_agent_nohis.py:189`.

`process_semantic_map` below takes an array rather than a file path; it is
otherwise line-for-line equivalent to MapNav's.
"""
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MapNav/constants.py:96-117, verbatim.
COLOR_PALETTE = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
]

# Label slots, from r2rnav_agent_nohis.py:495-511.
UNEXPLORED, OBSTACLE, EXPLORED, VISITED = 0, 1, 2, 3
SEMANTIC_BASE = 5  # `sem_map += 5`

# The eleven categories huatu3.py:23-35 recovers, in palette order. These are
# coco_categories 0-9 and 11 -- MapNav skips `book` (10) and stops at `clock`.
MAPNAV_CATEGORIES: List[str] = [
    "chair",          # palette 5
    "sofa",           # palette 6   (coco "couch")
    "potted plant",   # palette 7
    "bed",            # palette 8
    "toilet",         # palette 9
    "tv",             # palette 10
    "dining-table",   # palette 11
    "oven",           # palette 12
    "sink",           # palette 13
    "refrigerator",   # palette 14
    # palette 15 is `book`; huatu3 does not decode it.
    "clock",          # palette 16
]

# Palette slot for each category above (note the jump over `book`).
CATEGORY_PALETTE_INDEX: Dict[str, int] = {
    "chair": 5, "sofa": 6, "potted plant": 7, "bed": 8, "toilet": 9,
    "tv": 10, "dining-table": 11, "oven": 12, "sink": 13,
    "refrigerator": 14, "clock": 16,
}

# SAM3 is prompted with text, where MapNav used a COCO Mask R-CNN. Only the
# hyphenated name needs adjusting.
SAM3_PROMPT_NAME: Dict[str, str] = {"dining-table": "dining table"}


def palette_rgb() -> np.ndarray:
    """The palette as (N, 3) uint8, matching PIL's `putpalette` rounding."""
    vals = [int(x * 255.0) for x in COLOR_PALETTE]
    return np.asarray(vals, dtype=np.uint8).reshape(-1, 3)


def rgb_to_object() -> Dict[Tuple[int, int, int], str]:
    """huatu3.py:23-35, rebuilt from the palette so the two cannot drift."""
    pal = palette_rgb()
    return {tuple(int(v) for v in pal[idx]): name
            for name, idx in CATEGORY_PALETTE_INDEX.items()}


def render_label_grid(grid: np.ndarray, size: int = 480) -> np.ndarray:
    """MapNav's render: palette lookup, flipud, nearest-neighbour to `size`.

    Mirrors r2rnav_agent_nohis.py:520-530. Returns RGB.
    """
    pal = palette_rgb()
    idx = np.clip(grid, 0, len(pal) - 1).astype(np.uint8)
    rgb = pal[idx]
    rgb = np.flipud(rgb)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)


def draw_rounded_rectangle(draw, bbox, radius, fill) -> None:
    """huatu3.py:6-18, verbatim."""
    x1, y1, x2, y2 = bbox
    draw.rectangle([(x1 + radius, y1), (x2 - radius, y2)], fill=fill)
    draw.rectangle([(x1, y1 + radius), (x2, y2 - radius)], fill=fill)
    draw.ellipse([(x1, y1), (x1 + 2 * radius, y1 + 2 * radius)], fill=fill)
    draw.ellipse([(x2 - 2 * radius, y1), (x2, y1 + 2 * radius)], fill=fill)
    draw.ellipse([(x1, y2 - 2 * radius), (x1 + 2 * radius, y2)], fill=fill)
    draw.ellipse([(x2 - 2 * radius, y2 - 2 * radius), (x2, y2)], fill=fill)


def process_semantic_map(img_rgb: np.ndarray, font_size: int = 11
                         ) -> Tuple[np.ndarray, List[dict], List[str]]:
    """Port of huatu3.py:20-122, taking an array instead of a path.

    Returns (annotated RGB, labels, objects) with the same semantics as
    MapNav: `objects` is one entry per accepted blob, duplicates included,
    which is what its prompt joins.
    """
    mapping = rgb_to_object()
    height, width = img_rgb.shape[:2]
    output_img = img_rgb.copy()
    objects: List[str] = []
    labels: List[dict] = []

    for rgb, object_name in mapping.items():
        color_tolerance = 5
        lower_bound = np.array([max(0, x - color_tolerance) for x in rgb])
        upper_bound = np.array([min(255, x + color_tolerance) for x in rgb])
        mask = cv2.inRange(output_img, lower_bound, upper_bound)

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # huatu3.py:64 -- reproduced exactly, operator precedence and all.
            if area < 50 and object_name != 'toilet' \
                    and object_name != 'potted plant' \
                    or (object_name == 'potted plant' and area < 10):
                continue

            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])

            # MapNav appends only {'object': ...}; the centroid and area are
            # additive here (its `labels` is never consumed downstream) so the
            # caller can recover world coordinates. Nothing else changes.
            labels.append({'object': object_name,
                           'centroid_px': (center_x, center_y),
                           'area_px': int(area)})
            objects.append(object_name)

            pil_image = Image.fromarray(output_img)
            draw = ImageDraw.Draw(pil_image)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), object_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            x = center_x - text_width // 2
            y = center_y - text_height // 2
            x = max(0, min(x, width - text_width))
            y = max(0, min(y, height - text_height))

            padding = 4
            radius = 5
            bg_bbox = (x - padding, y - padding,
                       x + text_width + padding, y + text_height + padding)
            draw_rounded_rectangle(draw, bg_bbox, radius, fill=(255, 165, 0, 230))
            draw.text((x, y), object_name, fill=(0, 0, 0), font=font)

            output_img = np.array(pil_image)

    return output_img, labels, objects


def prompt_sentence(objects: List[str]) -> str:
    """r2rnav_agent_nohis.py:189, including its `including` rather than
    `includes` and the duplicate-preserving join."""
    if not objects:
        return ""
    object_str = ', '.join(objects)
    return f"   - As shown, this semantic map including objects such as {object_str}. \n"


def observation_block(instruction: str, objects: List[str]) -> str:
    """The full `<semantic_map>` block MapNav feeds its VLM
    (r2rnav_agent_nohis.py:182-198), so ASM output is drop-in comparable."""
    lines = [
        "<semantic_map>2. Top-down semantic map, this map has been gradually "
        "built from all observations collected during your navigation process "
        "since the beginning.\n",
    ]
    if objects:
        lines.append(prompt_sentence(objects))
    lines.append(
        "   - Dark gray areas represent obstacles, light gray areas indicate "
        "traversable spaces, and white areas show unexplored regions. \n"
    )
    lines.append(
        "   - The red arrow shows your current position, orientation and the "
        "red line represents your past trajectory.\n</semantic_map>"
    )
    return "".join(lines)
