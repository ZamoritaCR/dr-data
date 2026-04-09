"""
Layout Engine -- Deterministic visual positioning for Power BI pages.

compute_layout() returns pixel positions for N visuals on a PBI canvas.
Layout rules optimized for dashboard readability:
  1 visual  = full canvas
  2 visuals = side by side
  3 visuals = 2 top, 1 bottom (full width)
  4 visuals = 2x2 grid
  5-6       = 3x2 grid
  7+        = KPI cards (200x120) top row, charts below in grid

Spec drafted by deepseek-coder-v2 via Ollama (truncated at 190 lines),
fixed and completed by Claude review.
"""

import math


def compute_layout(n_visuals, canvas_w=1280, canvas_h=720, padding=20):
    """Compute pixel positions for N visuals on a PBI canvas.

    Args:
        n_visuals: number of visuals to position.
        canvas_w: canvas width in pixels.
        canvas_h: canvas height in pixels.
        padding: gap between visuals and canvas edges.

    Returns:
        list of dicts: [{"x": int, "y": int, "width": int, "height": int}, ...]
    """
    if n_visuals <= 0:
        return []

    usable_w = canvas_w - 2 * padding
    usable_h = canvas_h - 2 * padding

    if n_visuals == 1:
        return [{"x": padding, "y": padding, "width": usable_w, "height": usable_h}]

    if n_visuals == 2:
        w = (usable_w - padding) // 2
        return [
            {"x": padding, "y": padding, "width": w, "height": usable_h},
            {"x": padding + w + padding, "y": padding, "width": w, "height": usable_h},
        ]

    if n_visuals == 3:
        # 2 top, 1 bottom full width
        w = (usable_w - padding) // 2
        h_top = (usable_h - padding) // 2
        h_bot = usable_h - h_top - padding
        return [
            {"x": padding, "y": padding, "width": w, "height": h_top},
            {"x": padding + w + padding, "y": padding, "width": w, "height": h_top},
            {"x": padding, "y": padding + h_top + padding, "width": usable_w, "height": h_bot},
        ]

    if n_visuals == 4:
        # 2x2 grid
        w = (usable_w - padding) // 2
        h = (usable_h - padding) // 2
        return [
            {"x": padding, "y": padding, "width": w, "height": h},
            {"x": padding + w + padding, "y": padding, "width": w, "height": h},
            {"x": padding, "y": padding + h + padding, "width": w, "height": h},
            {"x": padding + w + padding, "y": padding + h + padding, "width": w, "height": h},
        ]

    if n_visuals <= 6:
        # 3x2 grid
        cols = 3
        rows = 2
        w = (usable_w - (cols - 1) * padding) // cols
        h = (usable_h - (rows - 1) * padding) // rows
        layout = []
        for i in range(n_visuals):
            col = i % cols
            row = i // cols
            x = padding + col * (w + padding)
            y = padding + row * (h + padding)
            layout.append({"x": x, "y": y, "width": w, "height": h})
        return layout

    # 7+ visuals: KPI cards top row, charts below in grid
    return _layout_kpi_plus_grid(n_visuals, canvas_w, canvas_h, padding)


def _layout_kpi_plus_grid(n_visuals, canvas_w, canvas_h, padding):
    """Layout for 7+ visuals: KPI cards top row + chart grid below."""
    usable_w = canvas_w - 2 * padding

    # KPI cards: up to 6 cards in the top row
    kpi_count = min(n_visuals // 3, 6)  # ~1/3 of visuals as KPIs, max 6
    if kpi_count < 1:
        kpi_count = 1
    chart_count = n_visuals - kpi_count

    # KPI row
    kpi_h = 120
    kpi_w = min(200, (usable_w - (kpi_count - 1) * padding) // kpi_count)
    kpi_y = padding

    layout = []
    # Center KPI cards
    total_kpi_w = kpi_count * kpi_w + (kpi_count - 1) * padding
    kpi_start_x = padding + (usable_w - total_kpi_w) // 2

    for i in range(kpi_count):
        x = kpi_start_x + i * (kpi_w + padding)
        layout.append({"x": x, "y": kpi_y, "width": kpi_w, "height": kpi_h})

    # Chart grid below KPIs
    chart_y_start = kpi_y + kpi_h + padding
    chart_area_h = canvas_h - chart_y_start - padding

    if chart_count <= 0:
        return layout

    # Determine grid dimensions
    cols = min(3, chart_count)
    rows = math.ceil(chart_count / cols)
    chart_w = (usable_w - (cols - 1) * padding) // cols
    chart_h = (chart_area_h - (rows - 1) * padding) // rows
    chart_h = max(chart_h, 150)  # Minimum chart height

    for i in range(chart_count):
        col = i % cols
        row = i // cols
        x = padding + col * (chart_w + padding)
        y = chart_y_start + row * (chart_h + padding)
        layout.append({"x": x, "y": y, "width": chart_w, "height": chart_h})

    return layout
