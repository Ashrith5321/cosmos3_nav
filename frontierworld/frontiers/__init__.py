"""Frontier extraction from the occupancy map."""

from frontierworld.frontiers.extraction import (
    Frontier,
    boundary_mask,
    extract_frontiers,
    information_gain,
)

__all__ = ["Frontier", "extract_frontiers", "boundary_mask", "information_gain"]
