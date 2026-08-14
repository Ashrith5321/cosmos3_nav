"""
Semantic vocabularies and priors used by the frontier world model.

Rooms, objects, and affordances form the structured prediction targets
(design doc section 19). ROOM_OBJECT_PRIOR encodes room->object
co-occurrence used by the zero-shot predictor to hypothesize what a
predicted room is likely to contain.
"""

import numpy as np

ROOM_TYPES = [
    "kitchen",
    "living room",
    "dining room",
    "bedroom",
    "bathroom",
    "hallway",
    "office",
    "closet",
    "laundry room",
    "garage",
    "staircase",
    "balcony",
    "gym",
    "empty room",
]

# HM3D/MP3D ObjectNav categories first, then common open-vocabulary items.
OBJECT_VOCAB = [
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv_monitor",
    "sofa",
    "table",
    "refrigerator",
    "microwave",
    "oven",
    "stove",
    "sink",
    "counter",
    "cabinet",
    "shower",
    "bathtub",
    "mirror",
    "towel",
    "wardrobe",
    "dresser",
    "nightstand",
    "pillow",
    "lamp",
    "desk",
    "computer",
    "bookshelf",
    "book",
    "picture",
    "window",
    "door",
    "washer",
    "dryer",
    "treadmill",
    "fireplace",
    "stairs",
    "rug",
    "curtain",
    "clothes",
    "cushion",
    "shelves",
]

AFFORDANCES = [
    "traversable",
    "enterable_room",
    "corridor",
    "doorway",
    "dead_end",
    "stairs",
    "open_space",
    "cluttered",
]

# Per-room object likelihoods, P(object visible | room). Unlisted -> BASE_OBJECT_PRIOR.
BASE_OBJECT_PRIOR = 0.02
ROOM_OBJECT_PRIOR = {
    "kitchen": {
        "refrigerator": 0.85, "microwave": 0.65, "oven": 0.70, "stove": 0.75,
        "sink": 0.85, "counter": 0.90, "cabinet": 0.85, "table": 0.35,
        "chair": 0.35, "window": 0.40, "door": 0.35, "shelves": 0.25,
    },
    "living room": {
        "sofa": 0.85, "tv_monitor": 0.65, "table": 0.65, "chair": 0.60,
        "plant": 0.40, "lamp": 0.45, "rug": 0.45, "cushion": 0.60,
        "fireplace": 0.15, "picture": 0.45, "window": 0.55, "bookshelf": 0.25,
        "curtain": 0.35,
    },
    "dining room": {
        "table": 0.90, "chair": 0.90, "cabinet": 0.35, "picture": 0.35,
        "window": 0.45, "lamp": 0.30, "plant": 0.25, "rug": 0.25,
    },
    "bedroom": {
        "bed": 0.92, "nightstand": 0.60, "wardrobe": 0.45, "dresser": 0.45,
        "pillow": 0.80, "lamp": 0.50, "mirror": 0.30, "window": 0.55,
        "curtain": 0.40, "desk": 0.25, "chair": 0.30, "picture": 0.35,
        "tv_monitor": 0.20, "clothes": 0.30,
    },
    "bathroom": {
        "toilet": 0.90, "sink": 0.88, "mirror": 0.70, "shower": 0.55,
        "bathtub": 0.35, "towel": 0.65, "cabinet": 0.30, "window": 0.25,
    },
    "hallway": {
        "door": 0.75, "picture": 0.35, "rug": 0.20, "stairs": 0.15,
        "table": 0.10, "plant": 0.12, "lamp": 0.15, "window": 0.20,
    },
    "office": {
        "desk": 0.85, "chair": 0.85, "computer": 0.60, "bookshelf": 0.50,
        "book": 0.55, "lamp": 0.40, "window": 0.40, "picture": 0.30,
        "shelves": 0.35, "cabinet": 0.25,
    },
    "closet": {
        "clothes": 0.75, "shelves": 0.60, "wardrobe": 0.30, "door": 0.50,
    },
    "laundry room": {
        "washer": 0.85, "dryer": 0.70, "sink": 0.30, "cabinet": 0.30,
        "shelves": 0.30, "clothes": 0.35,
    },
    "garage": {
        "door": 0.50, "shelves": 0.40, "cabinet": 0.20,
    },
    "staircase": {
        "stairs": 0.95, "picture": 0.20, "window": 0.20, "door": 0.25,
    },
    "balcony": {
        "plant": 0.40, "chair": 0.35, "table": 0.25, "window": 0.45,
        "door": 0.45,
    },
    "gym": {
        "treadmill": 0.55, "mirror": 0.40, "tv_monitor": 0.20, "window": 0.25,
    },
    "empty room": {
        "window": 0.30, "door": 0.40,
    },
}

# Room prompts for zero-shot classification of what a view looks INTO.
ROOM_CONTEXT_PROMPTS = {
    room: f"a doorway or opening leading into a {room}" for room in ROOM_TYPES
}
ROOM_SCENE_PROMPTS = {room: f"a photo of a {room}" for room in ROOM_TYPES}

# Generic distractor texts for cosine-score normalization in the evaluator.
DISTRACTOR_TEXTS = [
    "an empty wall",
    "a blurry indoor photo",
    "a plain floor",
    "a ceiling",
    "a dark corner of a room",
    "a random household room",
]


def object_prior_vector(room: str) -> np.ndarray:
    """P(object | room) over OBJECT_VOCAB, with BASE_OBJECT_PRIOR fill."""
    prior = ROOM_OBJECT_PRIOR.get(room, {})
    return np.array(
        [prior.get(obj, BASE_OBJECT_PRIOR) for obj in OBJECT_VOCAB], dtype=np.float32
    )


def room_event_text(room: str, max_objects: int = 4) -> str:
    """Language-event proposition for a hypothesized room (design 19.5)."""
    prior = ROOM_OBJECT_PRIOR.get(room, {})
    top = sorted(prior.items(), key=lambda kv: -kv[1])[:max_objects]
    if top:
        objs = ", ".join(name.replace("_", " ") for name, _ in top)
        return f"exploring further likely leads to a {room} containing {objs}"
    return f"exploring further likely leads to a {room}"


def match_goal_to_vocab(goal: str):
    """Best-effort map of a free-form goal string onto OBJECT_VOCAB.

    Returns the vocab index or None.
    """
    g = goal.strip().lower().replace(" ", "_")
    aliases = {
        "tv": "tv_monitor", "television": "tv_monitor", "monitor": "tv_monitor",
        "couch": "sofa", "potted_plant": "plant", "houseplant": "plant",
        "fridge": "refrigerator", "wc": "toilet", "bookcase": "bookshelf",
    }
    g = aliases.get(g, g)
    if g in OBJECT_VOCAB:
        return OBJECT_VOCAB.index(g)
    # substring fallback ("armchair" -> "chair")
    for i, obj in enumerate(OBJECT_VOCAB):
        if obj in g or g in obj:
            return i
    return None
