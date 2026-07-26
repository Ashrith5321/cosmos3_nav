"""Semantic mapping and derived room categories.

Two things the occupancy map cannot answer: which categories the agent has
seen where, and what kind of room it is standing in. Both are part of the
Phase 4 revelation.

Room labels are DERIVED, not ground truth. HM3D v0.2 ships region membership
per object instance but no room-type name -- every region comes back with
category None and a degenerate bounding box. So a region is labelled by the
objects inside it (a bed implies a bedroom, a toilet a bathroom), which is a
heuristic and must be reported as one. MP3D would give real region categories
if a ground-truth room label ever becomes necessary.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from frontierworld.mapping.occupancy import MapGeometry

UNLABELLED = -1

# Indicator objects for room types, weighted by how strongly they imply the
# room. A sink appears in kitchens and bathrooms, so it is weak; a toilet is
# decisive.
ROOM_INDICATORS: dict[str, dict[str, float]] = {
    "bedroom": {"bed": 3.0, "pillow": 0.5, "wardrobe": 1.0, "nightstand": 1.5},
    "bathroom": {
        "toilet": 3.0,
        "shower": 2.5,
        "bathtub": 2.5,
        "towel": 1.0,
        "sink": 0.5,
    },
    "kitchen": {
        "refrigerator": 3.0,
        "oven": 2.5,
        "stove": 2.5,
        "microwave": 2.0,
        "dishwasher": 2.0,
        "kitchen cabinet": 1.5,
        "sink": 0.5,
    },
    "living room": {"sofa": 2.5, "couch": 2.5, "tv": 1.5, "tv_monitor": 1.5, "coffee table": 1.5},
    "dining room": {"dining table": 3.0, "chair": 0.3},
    "office": {"desk": 2.0, "computer": 1.5, "office chair": 1.5},
    "hallway": {"stairs": 1.0},
    "garage": {"car": 3.0, "garage door": 2.0},
}


class SemanticMap:
    """Top-down grid of observed semantic categories.

    Stores one category id per cell rather than per-cell counts: the
    revelation only needs which cells gained a label and what that label is,
    and a full category histogram per cell would be far larger than the
    occupancy map for no extra signal.
    """

    def __init__(self, geometry: MapGeometry, floor_y: float = 0.0) -> None:
        self.geometry = geometry
        self.floor_y = floor_y
        self.category_grid = np.full(
            (geometry.size_cells, geometry.size_cells), UNLABELLED, dtype=np.int16
        )
        self.vocabulary: dict[str, int] = {}
        self._names: list[str] = []

    def category_id(self, name: str) -> int:
        if name not in self.vocabulary:
            self.vocabulary[name] = len(self._names)
            self._names.append(name)
        return self.vocabulary[name]

    def category_name(self, category_id: int) -> str | None:
        if 0 <= category_id < len(self._names):
            return self._names[category_id]
        return None

    def integrate(
        self,
        semantic: np.ndarray,
        points_world: np.ndarray,
        valid: np.ndarray,
        instance_to_category: dict[int, str],
        height_range: tuple[float, float] = (-0.5, 2.0),
    ) -> None:
        """Fold one semantic frame into the map.

        `points_world` and `valid` come from OccupancyMap.unproject, so the
        semantic map is registered to exactly the same geometry as occupancy.
        """
        semantic = np.squeeze(np.asarray(semantic)).astype(np.int64)
        if semantic.shape != points_world.shape[:2]:
            return

        heights = points_world[..., 1] - self.floor_y
        usable = valid & (heights >= height_range[0]) & (heights <= height_range[1])
        if not usable.any():
            return

        rows, cols = self.geometry.world_to_cell(
            points_world[..., 0][usable], points_world[..., 2][usable]
        )
        keep = self.geometry.in_bounds(rows, cols)
        instances = semantic[usable][keep]
        rows, cols = rows[keep], cols[keep]

        for instance in np.unique(instances):
            name = instance_to_category.get(int(instance))
            if name is None:
                continue
            mask = instances == instance
            self.category_grid[rows[mask], cols[mask]] = self.category_id(name)

    def labelled_mask(self) -> np.ndarray:
        return self.category_grid != UNLABELLED

    def copy_counts(self) -> np.ndarray:
        return self.category_grid.copy()

    def restore_counts(self, grid: np.ndarray) -> None:
        self.category_grid[...] = grid

    def categories_in(self, mask: np.ndarray) -> dict[str, int]:
        """Category name -> cell count, over a boolean region of the map."""
        selected = self.category_grid[mask & self.labelled_mask()]
        return {
            self._names[cid]: int(count)
            for cid, count in Counter(selected.tolist()).items()
            if 0 <= cid < len(self._names)
        }


class RoomLabeller:
    """Assigns a derived room type to each HM3D region, and to world points.

    HM3D gives object-to-region membership but no room names, so the label is
    inferred from the objects a region contains. Reported as derived.
    """

    def __init__(self, sim) -> None:
        self.region_categories: dict[str, str] = {}
        self.region_of_instance: dict[int, str] = {}
        self.instance_positions: dict[int, np.ndarray] = {}
        self._build(sim)

    def _build(self, sim) -> None:
        scene = sim.semantic_scene
        if scene is None:
            return

        objects_by_region: dict[str, list[str]] = defaultdict(list)
        for obj in scene.objects:
            if obj is None or obj.category is None:
                continue
            try:
                instance_id = int(obj.semantic_id)
            except (TypeError, ValueError):
                continue
            region = getattr(obj, "region", None)
            region_id = str(region.id) if region is not None else None
            name = obj.category.name()
            if region_id is not None:
                objects_by_region[region_id].append(name)
                self.region_of_instance[instance_id] = region_id
            try:
                self.instance_positions[instance_id] = np.asarray(
                    obj.aabb.center, dtype=np.float64
                )
            except Exception:  # noqa: BLE001 - some instances lack a bbox
                pass

        for region_id, names in objects_by_region.items():
            self.region_categories[region_id] = self._score(names)

    @staticmethod
    def _score(object_names: list[str]) -> str:
        counts = Counter(n.lower() for n in object_names)
        best_room, best_score = "unknown", 0.0
        for room, indicators in ROOM_INDICATORS.items():
            score = sum(
                weight * counts.get(indicator, 0)
                for indicator, weight in indicators.items()
                if isinstance(weight, (int, float))
            )
            if score > best_score:
                best_room, best_score = room, score
        return best_room

    def room_of_instance(self, instance_id: int) -> str | None:
        region_id = self.region_of_instance.get(int(instance_id))
        if region_id is None:
            return None
        return self.region_categories.get(region_id)

    def rooms_from_instances(self, instance_ids) -> dict[str, int]:
        """Room types implied by a set of observed instances, with counts."""
        rooms: Counter = Counter()
        for instance_id in instance_ids:
            room = self.room_of_instance(int(instance_id))
            if room and room != "unknown":
                rooms[room] += 1
        return dict(rooms)

    def dominant_room(self, instance_ids) -> str | None:
        rooms = self.rooms_from_instances(instance_ids)
        if not rooms:
            return None
        return max(rooms.items(), key=lambda item: item[1])[0]
