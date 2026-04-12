"""
Design Translator -- Pure data transformation from Tableau design metadata
to Power BI theme and visual formatting structures.

No side effects, no AI calls, no file I/O. All functions are deterministic
and operate only on the dicts passed in.
"""

# Professional categorical color palette (Tableau 10 inspired).
# High contrast, colorblind-safe, visually appealing in Power BI.
_DEFAULT_DATA_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]

# Tableau font family -> Power BI font family mapping.
_FONT_MAP = {
    "Tableau Book": "Segoe UI",
    "Tableau Regular": "Segoe UI",
    "Tableau Bold": "Segoe UI Semibold",
    "Tableau Semibold": "Segoe UI Semibold",
    "Tableau Light": "Segoe UI Light",
    "Tableau Medium": "Segoe UI",
}

# Tableau mark type -> Power BI visual type mapping.
# Tableau mark types -> PBI visual types.
# PBI naming: clusteredBarChart = HORIZONTAL bars, clusteredColumnChart = VERTICAL columns.
# Tableau "bar" mark = HORIZONTAL bars by convention.
_CHART_TYPE_MAP = {
    "bar": "clusteredBarChart",           # horizontal bars
    "line": "lineChart",
    "area": "areaChart",
    "circle": "scatterChart",
    "square": "matrix",                   # Tableau filled-square grid → PBI matrix
    "text": "tableEx",
    "map": "clusteredBarChart",
    "filled-map": "clusteredBarChart",
    "polygon": "clusteredBarChart",
    "multipolygon": "clusteredBarChart",
    "pie": "pieChart",
    "gantt-bar": "clusteredBarChart",     # horizontal gantt -> horizontal bar
    "automatic": "clusteredColumnChart",  # vertical columns (safe default)
    "shape": "scatterChart",             # fallback; disambiguation usually overrides
    "density": "scatterChart",
    "heatmap": "matrix",
    # Inferred from shelf bindings / disambiguation
    "ban": "cardVisual",
    "kpi": "cardVisual",
    "skip": "skip",                       # UI decoration — excluded from output
}


def translate_colors(tableau_design):
    """Translate Tableau color palettes to PBI theme color structure.

    Args:
        tableau_design: spec["design"] dict from the enhanced parser,
            containing color_palettes, global_fonts, datasource_color_maps.

    Returns:
        dict with dataColors, background, foreground, tableAccent.
    """
    if not tableau_design or not isinstance(tableau_design, dict):
        return {
            "dataColors": list(_DEFAULT_DATA_COLORS),
            "background": "#FFFFFF",
            "foreground": "#333333",
            "tableAccent": _DEFAULT_DATA_COLORS[0],
        }

    palettes = tableau_design.get("color_palettes", [])
    global_fonts = tableau_design.get("global_fonts", {})

    # Extract colors from palettes.
    # ONLY use a Tableau palette if it is explicitly typed as "regular"
    # (categorical). Sequential/diverging palettes are for maps/heatmaps
    # and look terrible as bar chart colors in PBI.
    # Most Tableau workbooks store categorical colors implicitly (not in XML),
    # so the professional default palette is used in the vast majority of cases.
    data_colors = []
    for p in palettes:
        ptype = p.get("type", "").lower()
        if ptype == "regular":
            raw = p.get("colors", [])
            if len(raw) >= 3:
                for c in raw:
                    if c and isinstance(c, str):
                        hex_color = c.strip()
                        if not hex_color.startswith("#"):
                            hex_color = "#" + hex_color
                        data_colors.append(hex_color)
                break  # use the first regular palette found

    # Datasource color maps store per-value color assignments defined in the
    # Tableau workbook (via Edit Colors dialog). These are the REAL colors the
    # workbook author chose. Use them when no explicit regular palette was found.
    if not data_colors:
        ds_maps = tableau_design.get("datasource_color_maps", [])
        if ds_maps:
            seen_ds = set()
            ds_colors = []
            for dsm in ds_maps:
                for mapping in dsm.get("mappings", []):
                    color = mapping.get("color", "").strip()
                    if color and color not in seen_ds:
                        seen_ds.add(color)
                        if not color.startswith("#"):
                            color = "#" + color
                        ds_colors.append(color)
            if len(ds_colors) >= 3:
                data_colors = ds_colors

    if not data_colors:
        data_colors = list(_DEFAULT_DATA_COLORS)

    # Background: from global font settings or worksheet background
    background = "#FFFFFF"

    # Foreground: from global font color
    font_color = global_fonts.get("font-color", "")
    foreground = font_color if font_color else "#333333"

    # Table accent: first color
    table_accent = data_colors[0] if data_colors else _DEFAULT_DATA_COLORS[0]

    return {
        "dataColors": data_colors,
        "background": background,
        "foreground": foreground,
        "tableAccent": table_accent,
    }


def translate_positions(zones, tableau_canvas, pbi_canvas=(1280, 720),
                        margin=12, gap=8):
    """Scale Tableau zone positions to PBI canvas coordinates.

    Args:
        zones: list of zone dicts, each with zone["layout"] = {x, y, w, h}
            and zone["name"].
        tableau_canvas: dict with width and height (integers).
        pbi_canvas: tuple (width, height) for the PBI canvas.
        margin: edge margin in pixels.
        gap: gap between visuals in pixels.

    Returns:
        list of dicts with name, x, y, width, height scaled to PBI canvas.
    """
    if not zones or not tableau_canvas:
        return []

    t_w = tableau_canvas.get("width", 0)
    t_h = tableau_canvas.get("height", 0)

    # Tableau zone coordinates can use an internal coordinate system
    # (twips or other units) that differs from the canvas <size>.
    # Auto-detect the actual bounding box from zone data to get the
    # real coordinate range.
    max_right = 0
    max_bottom = 0
    for zone in zones:
        layout = zone.get("layout", {})
        r = layout.get("x", 0) + layout.get("w", 0)
        b = layout.get("y", 0) + layout.get("h", 0)
        if r > max_right:
            max_right = r
        if b > max_bottom:
            max_bottom = b

    # Use the larger of canvas size vs detected bounds
    t_w = max(t_w, max_right) if max_right > 0 else t_w
    t_h = max(t_h, max_bottom) if max_bottom > 0 else t_h

    if t_w <= 0 or t_h <= 0:
        return []

    pbi_w, pbi_h = pbi_canvas
    usable_w = pbi_w - 2 * margin
    usable_h = pbi_h - 2 * margin

    results = []
    for zone in zones:
        layout = zone.get("layout", {})
        name = zone.get("name", "")

        zx = layout.get("x", 0)
        zy = layout.get("y", 0)
        zw = layout.get("w", 0)
        zh = layout.get("h", 0)

        # Skip zones with no size
        if zw <= 0 or zh <= 0:
            continue

        # Proportional scaling
        new_x = int((zx / t_w) * usable_w + margin)
        new_y = int((zy / t_h) * usable_h + margin)
        new_w = int((zw / t_w) * usable_w)
        new_h = int((zh / t_h) * usable_h)

        # Clamp to canvas bounds
        new_x = max(margin, min(new_x, pbi_w - margin - 1))
        new_y = max(margin, min(new_y, pbi_h - margin - 1))
        new_w = max(1, min(new_w, pbi_w - margin - new_x))
        new_h = max(1, min(new_h, pbi_h - margin - new_y))

        results.append({
            "name": name,
            "x": new_x,
            "y": new_y,
            "width": new_w,
            "height": new_h,
        })

    # Collision detection: if two visuals overlap, shrink the later one
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            if _rects_overlap(a, b):
                _resolve_collision(a, b, pbi_w, pbi_h, margin, gap)

    return results


def _rects_overlap(a, b):
    """Check if two rectangles overlap. Each has x, y, width, height."""
    return (
        a["x"] < b["x"] + b["width"]
        and a["x"] + a["width"] > b["x"]
        and a["y"] < b["y"] + b["height"]
        and a["y"] + a["height"] > b["y"]
    )


def _resolve_collision(a, b, canvas_w, canvas_h, margin, gap):
    """Resolve overlap by shrinking rect b (the later one).

    Strategy: determine the overlap dimensions and shrink b by moving its
    edge inward along the axis with smaller overlap.
    """
    # Calculate overlap amounts on each axis
    overlap_x = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    overlap_y = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])

    if overlap_x <= 0 or overlap_y <= 0:
        return  # No actual overlap

    # Shrink along the axis with smaller overlap (less disruption)
    if overlap_x <= overlap_y:
        # Horizontal: push b right of a + gap
        if b["x"] >= a["x"]:
            new_x = a["x"] + a["width"] + gap
            width_loss = new_x - b["x"]
            b["x"] = min(new_x, canvas_w - margin - 1)
            b["width"] = max(1, b["width"] - width_loss)
        else:
            # b is to the left, shrink its right edge
            b["width"] = max(1, a["x"] - gap - b["x"])
    else:
        # Vertical: push b below a + gap
        if b["y"] >= a["y"]:
            new_y = a["y"] + a["height"] + gap
            height_loss = new_y - b["y"]
            b["y"] = min(new_y, canvas_h - margin - 1)
            b["height"] = max(1, b["height"] - height_loss)
        else:
            # b is above, shrink its bottom edge
            b["height"] = max(1, a["y"] - gap - b["y"])


def translate_fonts(tableau_fonts):
    """Translate Tableau global font settings to PBI font specs.

    Args:
        tableau_fonts: global_fonts dict from parser with keys like
            font-family, font-size, font-color, font-style, font-weight.

    Returns:
        dict with fontFamily, fontSize, fontColor keys suitable for PBI.
    """
    if not tableau_fonts or not isinstance(tableau_fonts, dict):
        return {
            "fontFamily": "Segoe UI",
            "fontSize": 10,
            "fontColor": "#333333",
        }

    raw_family = tableau_fonts.get("font-family", "")
    mapped_family = _FONT_MAP.get(raw_family, "Segoe UI")

    raw_size = tableau_fonts.get("font-size", "")
    try:
        font_size = int(raw_size.replace("pt", "").replace("px", "").strip())
    except (ValueError, AttributeError):
        font_size = 10

    font_color = tableau_fonts.get("font-color", "#333333")
    if not font_color:
        font_color = "#333333"

    return {
        "fontFamily": mapped_family,
        "fontSize": font_size,
        "fontColor": font_color,
    }


def translate_chart_type(tableau_mark_type):
    """Translate a Tableau mark type string to PBI visual type string.

    Args:
        tableau_mark_type: Tableau mark type (bar, line, area, circle, etc.)

    Returns:
        PBI visual type string.
    """
    if not tableau_mark_type or not isinstance(tableau_mark_type, str):
        return "clusteredColumnChart"
    return _CHART_TYPE_MAP.get(tableau_mark_type.lower().strip(), "clusteredColumnChart")


def build_pbi_theme(tableau_design, dashboard_title="Dashboard"):
    """Build a complete PBI theme JSON structure from Tableau design metadata.

    Combines translate_colors + translate_fonts into a valid Power BI theme.

    Args:
        tableau_design: spec["design"] dict from enhanced parser.
        dashboard_title: name for the theme.

    Returns:
        dict: valid Power BI theme JSON structure.
    """
    colors = translate_colors(tableau_design)
    global_fonts = {}
    if tableau_design and isinstance(tableau_design, dict):
        global_fonts = tableau_design.get("global_fonts", {})
    fonts = translate_fonts(global_fonts)

    theme = {
        "name": dashboard_title,
        "dataColors": colors["dataColors"],
        "background": colors["background"],
        "foreground": colors["foreground"],
        "tableAccent": colors["tableAccent"],
        "visualStyles": {
            "*": {
                "*": {
                    "background": [{
                        "color": {"solid": {"color": colors["background"]}}
                    }],
                    "title": [{
                        "fontFamily": fonts["fontFamily"],
                        "fontSize": fonts["fontSize"],
                        "color": {"solid": {"color": fonts["fontColor"]}},
                    }],
                }
            }
        },
    }

    return theme


def build_visual_formatting(ws_design):
    """Build PBI visual-level formatting objects from worksheet design data.

    Args:
        ws_design: ws_info["design"] dict from enhanced parser with keys
            mark_colors, background_color, title_font, axis_config,
            mark_style, border.

    Returns:
        dict with keys for visual.json "objects": general, title,
        categoryAxis, valueAxis, dataPoint, labels.
    """
    if not ws_design or not isinstance(ws_design, dict):
        return {}

    formatting = {}

    # General: background and border
    bg_color = ws_design.get("background_color", "")
    border = ws_design.get("border", {})
    general = {}
    if bg_color:
        general["background"] = [{
            "properties": {
                "color": {"solid": {"color": bg_color}},
                "show": {"expr": {"Literal": {"Value": "true"}}},
            }
        }]
    if border:
        border_color = border.get("border-color", "")
        if border_color:
            general["outlineColor"] = [{
                "properties": {
                    "color": {"solid": {"color": border_color}},
                }
            }]
    if general:
        formatting["general"] = [{"properties": {}}]
        if bg_color:
            formatting["general"][0]["properties"]["color"] = {
                "solid": {"color": bg_color}
            }

    # Title font
    title_font = ws_design.get("title_font", {})
    if title_font:
        font_name = title_font.get("fontname", "")
        mapped_font = _FONT_MAP.get(font_name, font_name or "Segoe UI")
        font_size_raw = title_font.get("fontsize", "")
        try:
            font_size = int(str(font_size_raw).replace("pt", "").replace("px", "").strip())
        except (ValueError, TypeError):
            font_size = 12
        font_color = title_font.get("fontcolor", "#333333")

        formatting["title"] = [{
            "properties": {
                "show": {"expr": {"Literal": {"Value": "true"}}},
                "fontFamily": mapped_font,
                "fontSize": font_size,
                "fontColor": {"solid": {"color": font_color}},
            }
        }]

    # Axis configuration
    axis_config = ws_design.get("axis_config", [])
    has_hidden_axis = any(ax.get("hidden") for ax in axis_config)

    formatting["categoryAxis"] = [{
        "properties": {
            "show": {"expr": {"Literal": {"Value": "true" if not has_hidden_axis else "false"}}},
        }
    }]
    formatting["valueAxis"] = [{
        "properties": {
            "show": {"expr": {"Literal": {"Value": "true"}}},
        }
    }]

    # Data point fill colors from mark_colors
    mark_colors = ws_design.get("mark_colors", [])
    if mark_colors:
        fill_colors = [mc.get("color", "") for mc in mark_colors if mc.get("color")]
        if fill_colors:
            formatting["dataPoint"] = [{
                "properties": {
                    "fill": {"solid": {"color": fill_colors[0]}},
                }
            }]

    # Labels visibility from mark_style
    mark_style = ws_design.get("mark_style", {})
    labels_visible = mark_style.get("labels_visible", False)
    formatting["labels"] = [{
        "properties": {
            "show": {"expr": {"Literal": {"Value": "true" if labels_visible else "false"}}},
            "fontFamily": "Segoe UI",
        }
    }]

    return formatting
