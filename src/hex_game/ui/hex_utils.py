from __future__ import annotations

from math import cos, pi, sin, sqrt

import numpy as np


def get_hex_points(
    center: tuple[float, float], radius: float, percent: float = 0.06
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Calculate the points for outer and inner hexagons."""
    points = []
    points_inside = []
    for angle in np.arange(pi / 2, -3 * pi / 2, -pi / 3):
        points.append(
            (
                radius * (1 + percent) * cos(angle) + center[0],
                radius * (1 + percent) * sin(angle) + center[1],
            ),
        )
        points_inside.append(
            (
                radius * (1 - percent) * cos(angle) + center[0],
                radius * (1 - percent) * sin(angle) + center[1],
            ),
        )
    return points, points_inside


def get_arc_points(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    n_points: int = 10,
) -> list[tuple[float, float]]:
    """Generate points for an arc."""
    points = []
    for angle in np.linspace(start_angle, end_angle, n_points):
        points.append(
            (
                center[0] + radius * cos(angle),
                center[1] + radius * sin(angle),
            ),
        )
    return points


def get_arc_points_rotated(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    n_points: int = 10,
) -> list[tuple[float, float]]:
    """Generate points for an arc with rotated coordinates."""
    points = []
    for angle in np.linspace(start_angle, end_angle, n_points):
        points.append(
            (
                center[0] - radius * sin(angle),
                center[1] - radius * cos(angle),
            ),
        )
    return points


def calculate_diamond_layout(
    width: int,
    height: int,
    n: int,
    safey: float = 0.12,
    safex: float = 0.02,
) -> dict:
    """Calculate all geometric properties for the diamond board layout."""
    safe_height = int((1 - safey) * height)
    safe_width = int((1 - safex) * width)
    diamond_height = safe_height
    x_val = int(diamond_height * 1.1547005383792517)
    if 1.5 * x_val > safe_width:
        x_val = int(safe_width / 1.5)
        diamond_height = int(x_val / 1.1547005383792517)

    radius = x_val / (n + 2)
    center = (width // 2, height // 2)

    diamond_top_left = (center[0] - x_val * 3 / 4, center[1] - diamond_height / 2)
    diamond_top_right = (center[0] + x_val / 4, center[1] - diamond_height / 2)
    diamond_bottom_left = (center[0] - x_val / 4, center[1] + diamond_height / 2)
    diamond_bottom_right = (center[0] + x_val * 3 / 4, center[1] + diamond_height / 2)

    hex_radius = x_val / (n + 1) / sqrt(3)

    return {
        "radius": radius,
        "x_val": x_val,
        "diamond_height": diamond_height,
        "center": center,
        "hex_radius": hex_radius,
        "top_left": diamond_top_left,
        "top_right": diamond_top_right,
        "bottom_left": diamond_bottom_left,
        "bottom_right": diamond_bottom_right,
    }


def find_closest_hex(
    mouse_pos: tuple[int, int],
    hex_positions: np.ndarray,
    hex_radius: float,
) -> tuple[int, int] | None:
    """Find the index (i, j) of the hex closest to mouse_pos, if within hex_radius."""
    mouse_arr = np.array(mouse_pos)
    diffs = hex_positions - mouse_arr
    diffs = np.sum(diffs**2, axis=2)
    min_index = np.argmin(diffs)
    i, j = np.unravel_index(min_index, diffs.shape)
    if diffs[i][j] <= hex_radius**2:
        return (int(i), int(j))
    return None
