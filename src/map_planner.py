"""External planner for the planning-branch objectnav stack.

Cosmos picks WHICH frontier to head for; this module makes the PLAN: an A* path
over the agent's *built* occupancy map (fog-of-war free space only -- no
privileged navmesh) from the current pose to the chosen frontier, converted into
the next discrete habitat action (forward / left / right).

Honest by construction: it can only route through cells the agent has actually
observed as FREE. OCCUPIED and UNKNOWN cells are non-traversable.
"""
import heapq
import cv2
import numpy as np

from global_map import FREE, OCCUPIED   # 0=UNKNOWN, 1=FREE, 2=OCCUPIED

_NEI = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356)]
_DILATE_KERNEL = np.ones((5, 5), np.uint8)   # ~2-cell (0.1m) dilation per iteration


def traversable_mask(grid, dilate_iter=3):
    """Boolean traversable grid for planning.

    FREE space dilated to bridge the small gaps left by sparse depth projection
    (e.g. the radial 'spokes' from rotating in place), then walls carved back
    out so we never plan through an OCCUPIED cell.
    """
    free = (grid == FREE).astype(np.uint8)
    dil = cv2.dilate(free, _DILATE_KERNEL, iterations=dilate_iter).astype(bool)
    return dil & (grid != OCCUPIED)


def _nearest_true(mask, rc, max_r=16):
    """Snap a (row,col) to the closest True cell within max_r (Chebyshev)."""
    r0, c0 = int(rc[0]), int(rc[1])
    nr, nc = mask.shape
    if 0 <= r0 < nr and 0 <= c0 < nc and mask[r0, c0]:
        return (r0, c0)
    best, bestd = None, 1e9
    for rad in range(1, max_r + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                r, c = r0 + dr, c0 + dc
                if 0 <= r < nr and 0 <= c < nc and mask[r, c]:
                    d = dr * dr + dc * dc
                    if d < bestd:
                        best, bestd = (r, c), d
        if best is not None:
            return best
    return None


def _astar(mask, start_rc, goal_rc, max_expand=300000):
    """8-connected A* over a boolean traversable mask. Returns list[(row,col)] or None."""
    start = _nearest_true(mask, start_rc)
    goal = _nearest_true(mask, goal_rc)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    gr, gc = goal
    open_heap = [(0.0, start)]
    g = {start: 0.0}
    came = {}
    nr, nc = mask.shape
    expand = 0
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        expand += 1
        if expand > max_expand:
            return None
        cr, cc = cur
        base = g[cur]
        for dr, dc, w in _NEI:
            r, c = cr + dr, cc + dc
            if 0 <= r < nr and 0 <= c < nc and mask[r, c]:
                ng = base + w
                nxt = (r, c)
                if ng < g.get(nxt, 1e18):
                    g[nxt] = ng
                    h = ((r - gr) ** 2 + (c - gc) ** 2) ** 0.5
                    heapq.heappush(open_heap, (ng + h, nxt))
                    came[nxt] = cur
    return None


def _signed_angle_deg(pose, target_xz):
    """Signed angle (deg) from the agent's heading to target_xz; <0 = left."""
    R, t = pose[:3, :3], pose[:3, 3]
    fwd = R @ np.array([0.0, 0.0, -1.0])
    heading = np.arctan2(fwd[0], -fwd[2])
    dx = target_xz[0] - t[0]
    dz = target_xz[1] - t[2]
    return np.degrees((np.arctan2(dx, -dz) - heading + np.pi) % (2 * np.pi) - np.pi)


def plan_action(gm, pose, goal_xz, turn_angle_deg=30.0, lookahead_m=0.5):
    """Next discrete action toward goal_xz.

    Primary: A* over the (dilated) built free space. Fallback: if free space
    can't yet connect the agent to the goal, head greedily toward goal_xz so the
    agent keeps moving and grows the map until a real path exists. Always returns
    an action (never None) so the caller never stalls.

    Returns {action, dist_to_goal, planned(bool), path_cells, waypoint}.
    """
    m = gm.active
    t = pose[:3, 3]
    ax, az = float(t[0]), float(t[2])
    dist_to_goal = float(np.hypot(goal_xz[0] - ax, goal_xz[1] - az))

    mask = traversable_mask(m.grid)
    path = _astar(mask, m.world_to_grid((ax, az)), m.world_to_grid((goal_xz[0], goal_xz[1])))

    planned = bool(path and len(path) >= 2)
    if planned:
        pts = [m.grid_to_world(r, c) for r, c in path]      # each -> [x, z]
        wp = pts[-1]
        for p in pts[1:]:
            if np.hypot(p[0] - ax, p[1] - az) >= lookahead_m:
                wp = p
                break
        path_cells = len(path)
    else:
        wp = (goal_xz[0], goal_xz[1])                        # greedy: straight at the frontier
        path_cells = 0

    ang = _signed_angle_deg(pose, wp)
    if abs(ang) <= turn_angle_deg / 2.0:
        action = "forward"
    elif ang < 0:
        action = "left"
    else:
        action = "right"
    return {"action": action, "dist_to_goal": dist_to_goal, "planned": planned,
            "path_cells": path_cells, "waypoint": (float(wp[0]), float(wp[1]))}
