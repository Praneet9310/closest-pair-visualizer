"""
Closest Pair Algorithms — Brute Force & Divide and Conquer
All logic is pure Python + numpy.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Point:
    x: float
    y: float
    id: int = 0

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


@dataclass
class AlgorithmResult:
    point_a: Point
    point_b: Point
    distance: float
    algorithm: str
    time_ms: float
    comparisons: int
    steps: list = field(default_factory=list)


def euclidean(p1: Point, p2: Point) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ─────────────────────────────────────────────
#  BRUTE FORCE  O(n²)
# ─────────────────────────────────────────────
def brute_force(points: list) -> AlgorithmResult:
    n = len(points)
    if n < 2:
        raise ValueError("Need at least 2 points")

    comparisons = 0
    steps = []
    best_dist = float("inf")
    best_pair = (points[0], points[1])

    t0 = time.perf_counter()

    for i in range(n):
        for j in range(i + 1, n):
            comparisons += 1
            d = euclidean(points[i], points[j])
            step = {
                "type": "compare",
                "a": points[i],
                "b": points[j],
                "dist": d,
                "is_best": False,
            }
            if d < best_dist:
                best_dist = d
                best_pair = (points[i], points[j])
                step["is_best"] = True
            steps.append(step)

    elapsed = (time.perf_counter() - t0) * 1000

    return AlgorithmResult(
        point_a=best_pair[0],
        point_b=best_pair[1],
        distance=best_dist,
        algorithm="Brute Force",
        time_ms=elapsed,
        comparisons=comparisons,
        steps=steps,
    )


# ─────────────────────────────────────────────
#  DIVIDE & CONQUER  O(n log n)
# ─────────────────────────────────────────────
_dc_state = {"steps": [], "comparisons": 0}


def _strip_closest(strip, d):
    best = d
    best_pair = None
    local_steps = []
    strip_sorted = sorted(strip, key=lambda p: p.y)

    for i in range(len(strip_sorted)):
        j = i + 1
        while j < len(strip_sorted) and (strip_sorted[j].y - strip_sorted[i].y) < best:
            _dc_state["comparisons"] += 1
            dist = euclidean(strip_sorted[i], strip_sorted[j])
            local_steps.append({
                "type": "strip_compare",
                "a": strip_sorted[i],
                "b": strip_sorted[j],
                "dist": dist,
                "is_best": dist < best,
            })
            if dist < best:
                best = dist
                best_pair = (strip_sorted[i], strip_sorted[j])
            j += 1

    return best, best_pair, local_steps


def _closest_recursive(pts_x, depth=0):
    n = len(pts_x)

    if n <= 3:
        best = float("inf")
        best_pair = (pts_x[0], pts_x[1]) if n >= 2 else None
        local_steps = []
        for i in range(n):
            for j in range(i + 1, n):
                _dc_state["comparisons"] += 1
                d = euclidean(pts_x[i], pts_x[j])
                local_steps.append({
                    "type": "base_compare",
                    "a": pts_x[i],
                    "b": pts_x[j],
                    "dist": d,
                    "depth": depth,
                    "is_best": d < best,
                })
                if d < best:
                    best = d
                    best_pair = (pts_x[i], pts_x[j])
        return best, best_pair, local_steps

    mid = n // 2
    mid_point = pts_x[mid]

    dl, pair_l, left_steps = _closest_recursive(pts_x[:mid], depth + 1)
    dr, pair_r, right_steps = _closest_recursive(pts_x[mid:], depth + 1)

    all_steps = left_steps + right_steps
    all_steps.append({
        "type": "split",
        "mid_x": mid_point.x,
        "depth": depth,
    })

    if dl <= dr:
        d, best_pair = dl, pair_l
    else:
        d, best_pair = dr, pair_r

    strip = [p for p in pts_x if abs(p.x - mid_point.x) < d]
    strip_d, strip_pair, strip_steps = _strip_closest(strip, d)
    all_steps.extend(strip_steps)

    if strip_d < d and strip_pair:
        return strip_d, strip_pair, all_steps
    return d, best_pair, all_steps


def divide_and_conquer(points: list) -> AlgorithmResult:
    _dc_state["steps"] = []
    _dc_state["comparisons"] = 0

    if len(points) < 2:
        raise ValueError("Need at least 2 points")

    t0 = time.perf_counter()
    pts_sorted = sorted(points, key=lambda p: p.x)
    dist, pair, steps = _closest_recursive(pts_sorted)
    elapsed = (time.perf_counter() - t0) * 1000

    if pair is None:
        pair = (points[0], points[1])

    return AlgorithmResult(
        point_a=pair[0],
        point_b=pair[1],
        distance=dist,
        algorithm="Divide & Conquer",
        time_ms=elapsed,
        comparisons=_dc_state["comparisons"],
        steps=steps,
    )


# ─────────────────────────────────────────────
#  SVG BUILDER
# ─────────────────────────────────────────────
def build_svg(
    points: list,
    result=None,
    active_step: int = -1,
    show_splits: bool = False,
    width: int = 720,
    height: int = 500,
) -> str:
    PAD = 65
    if not points:
        return _empty_svg(width, height)

    xs = [p.x for p in points]
    ys = [p.y for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1

    def tx(x):
        return PAD + (x - min_x) / span_x * (width - 2 * PAD)

    def ty(y):
        return height - PAD - (y - min_y) / span_y * (height - 2 * PAD)

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="border-radius:16px;overflow:hidden;">'
    )

    # ── defs ──────────────────────────────────────────────────────────
    lines.append("""  <defs>
    <radialGradient id="bg" cx="40%" cy="40%" r="80%">
      <stop offset="0%" stop-color="#0e0b2e"/>
      <stop offset="100%" stop-color="#020210"/>
    </radialGradient>
    <filter id="glow">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="strongGlow">
      <feGaussianBlur in="SourceGraphic" stdDeviation="7" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="softGlow">
      <feGaussianBlur in="SourceGraphic" stdDeviation="1.5" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="neonLine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#f72585"/>
      <stop offset="50%" stop-color="#00f5d4"/>
      <stop offset="100%" stop-color="#f72585">
        <animate attributeName="stop-color" values="#f72585;#7209b7;#f72585" dur="3s" repeatCount="indefinite"/>
      </stop>
    </linearGradient>
  </defs>""")

    # background
    lines.append(f'  <rect width="{width}" height="{height}" fill="url(#bg)"/>')

    # grid
    for gx in range(PAD, width - PAD + 1, 55):
        lines.append(
            f'  <line x1="{gx}" y1="{PAD//2}" x2="{gx}" y2="{height-PAD//2}" '
            f'stroke="#ffffff" stroke-opacity="0.03" stroke-width="1"/>'
        )
    for gy in range(PAD, height - PAD + 1, 55):
        lines.append(
            f'  <line x1="{PAD//2}" y1="{gy}" x2="{width-PAD//2}" y2="{gy}" '
            f'stroke="#ffffff" stroke-opacity="0.03" stroke-width="1"/>'
        )

    # axes
    lines.append(
        f'  <line x1="{PAD}" y1="{PAD//2}" x2="{PAD}" y2="{height-PAD//2}" '
        f'stroke="#4a4a8a" stroke-opacity="0.4" stroke-width="1.5"/>'
    )
    lines.append(
        f'  <line x1="{PAD//2}" y1="{height-PAD}" x2="{width-PAD//2}" y2="{height-PAD}" '
        f'stroke="#4a4a8a" stroke-opacity="0.4" stroke-width="1.5"/>'
    )

    # axis labels
    for i, val in enumerate([min_x, (min_x + max_x) / 2, max_x]):
        lx = PAD + i * (width - 2 * PAD) / 2
        lines.append(
            f'  <text x="{lx:.0f}" y="{height - PAD//2 + 5}" text-anchor="middle" '
            f'font-family="monospace" font-size="9" fill="#555588">{val:.1f}</text>'
        )
    for i, val in enumerate([min_y, (min_y + max_y) / 2, max_y]):
        ly = height - PAD - i * (height - 2 * PAD) / 2
        lines.append(
            f'  <text x="{PAD//2 - 2}" y="{ly:.0f}" text-anchor="end" dominant-baseline="middle" '
            f'font-family="monospace" font-size="9" fill="#555588">{val:.1f}</text>'
        )

    # ── D&C split lines ────────────────────────────────────────────────
    if result and show_splits and active_step >= 0:
        for step in result.steps[: active_step + 1]:
            if step["type"] == "split":
                mx = tx(step["mid_x"])
                alpha = max(0.05, 0.35 - step["depth"] * 0.07)
                lines.append(
                    f'  <line x1="{mx:.1f}" y1="{PAD//2}" x2="{mx:.1f}" y2="{height-PAD//2}" '
                    f'stroke="#9b5de5" stroke-opacity="{alpha:.2f}" stroke-width="1.2" '
                    f'stroke-dasharray="6,4"/>'
                )

    # ── active step line (comparison flash) ───────────────────────────
    if result and 0 <= active_step < len(result.steps):
        step = result.steps[active_step]
        if step["type"] in ("compare", "base_compare", "strip_compare") and "a" in step:
            ax_, ay_ = tx(step["a"].x), ty(step["a"].y)
            bx_, by_ = tx(step["b"].x), ty(step["b"].y)
            color = "#ffe566" if step.get("is_best") else "#444466"
            opac = "0.85" if step.get("is_best") else "0.35"
            width_s = "2" if step.get("is_best") else "1"
            lines.append(
                f'  <line x1="{ax_:.1f}" y1="{ay_:.1f}" x2="{bx_:.1f}" y2="{by_:.1f}" '
                f'stroke="{color}" stroke-opacity="{opac}" stroke-width="{width_s}" '
                f'stroke-dasharray="5,3" filter="url(#softGlow)"/>'
            )

    # ── all-pair faint web (if few points) ────────────────────────────
    if len(points) <= 18:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                x1, y1 = tx(points[i].x), ty(points[i].y)
                x2, y2 = tx(points[j].x), ty(points[j].y)
                lines.append(
                    f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="#22224a" stroke-opacity="0.25" stroke-width="0.6"/>'
                )

    # ── closest pair neon line ────────────────────────────────────────
    if result:
        ax_ = tx(result.point_a.x)
        ay_ = ty(result.point_a.y)
        bx_ = tx(result.point_b.x)
        by_ = ty(result.point_b.y)
        lines.append(
            f'  <line x1="{ax_:.1f}" y1="{ay_:.1f}" x2="{bx_:.1f}" y2="{by_:.1f}" '
            f'stroke="url(#neonLine)" stroke-width="3" stroke-linecap="round" '
            f'filter="url(#strongGlow)">'
            f'  <animate attributeName="stroke-opacity" values="0.5;1;0.5" dur="2s" repeatCount="indefinite"/>'
            f'  </line>'
        )
        # distance badge
        mx_ = (ax_ + bx_) / 2
        my_ = (ay_ + by_) / 2 - 16
        dist_txt = f"d = {result.distance:.4f}"
        bw = len(dist_txt) * 7 + 16
        lines.append(
            f'  <rect x="{mx_-bw/2:.1f}" y="{my_-14:.1f}" width="{bw}" height="18" rx="6" '
            f'fill="#000028" fill-opacity="0.85" stroke="#00f5d4" stroke-width="0.8"/>'
        )
        lines.append(
            f'  <text x="{mx_:.1f}" y="{my_-1:.1f}" text-anchor="middle" '
            f'font-family="monospace" font-size="10.5" fill="#00f5d4" font-weight="bold">'
            f'{dist_txt}</text>'
        )

    # ── points ────────────────────────────────────────────────────────
    closest_ids = set()
    if result:
        closest_ids = {result.point_a.id, result.point_b.id}

    for p in points:
        cx_, cy_ = tx(p.x), ty(p.y)
        is_cl = p.id in closest_ids
        dur = f"{1.6 + (p.id % 5) * 0.3:.1f}s"

        if is_cl:
            # pulse ring
            lines.append(
                f'  <circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="14" '
                f'fill="none" stroke="#f72585" stroke-width="1.5">'
                f'  <animate attributeName="r" values="10;20;10" dur="{dur}" repeatCount="indefinite"/>'
                f'  <animate attributeName="stroke-opacity" values="0.8;0;0.8" dur="{dur}" repeatCount="indefinite"/>'
                f'  </circle>'
            )
            col = "#f72585"
            r = 7
            flt = 'filter="url(#strongGlow)"'
        else:
            col = "#4cc9f0"
            r = 5
            flt = 'filter="url(#glow)"'

        lines.append(
            f'  <circle cx="{cx_:.1f}" cy="{cy_:.1f}" r="{r}" fill="{col}" {flt}>'
            f'  <animate attributeName="opacity" values="0.75;1;0.75" dur="{dur}" repeatCount="indefinite"/>'
            f'  </circle>'
        )

        # label
        lbl = f"({p.x:.1f},{p.y:.1f})"
        lx = cx_ + 10
        ly = cy_ - 8
        if lx + len(lbl) * 6.2 > width - 5:
            lx = cx_ - len(lbl) * 6.2 - 6
        lines.append(
            f'  <text x="{lx:.1f}" y="{ly:.1f}" font-family="monospace" font-size="9" '
            f'fill="#8888bb" opacity="0.8">{lbl}</text>'
        )

    # ── step badge ────────────────────────────────────────────────────
    if result and active_step >= 0:
        total = len(result.steps)
        badge = f"Step {min(active_step+1,total)}/{total}"
        bw2 = len(badge) * 7.5 + 16
        lines.append(
            f'  <rect x="{width-bw2-10}" y="8" width="{bw2}" height="20" rx="7" '
            f'fill="#12103a" fill-opacity="0.9" stroke="#9b5de5" stroke-width="0.8"/>'
        )
        lines.append(
            f'  <text x="{width-bw2/2-10}" y="22" text-anchor="middle" font-family="monospace" '
            f'font-size="10.5" fill="#c5a0f0">{badge}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _empty_svg(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="border-radius:16px;overflow:hidden;">'
        f'<defs>'
        f'<radialGradient id="bg" cx="40%" cy="40%" r="80%">'
        f'<stop offset="0%" stop-color="#0e0b2e"/>'
        f'<stop offset="100%" stop-color="#020210"/>'
        f'</radialGradient></defs>'
        f'<rect width="{width}" height="{height}" fill="url(#bg)"/>'
        f'<text x="{width//2}" y="{height//2-12}" text-anchor="middle" '
        f'font-family="monospace" font-size="17" fill="#2e2e6a">No points yet</text>'
        f'<text x="{width//2}" y="{height//2+14}" text-anchor="middle" '
        f'font-family="monospace" font-size="12" fill="#1e1e4a">Add points using the sidebar →</text>'
        f'</svg>'
    )