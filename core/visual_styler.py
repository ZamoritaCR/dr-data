"""
Visual Styler -- Enhanced PBI visual formatting engine.

Builds on design_translator by producing richer per-visual formatting:
conditional formatting, gradient fills, legend/axis/gridline/label styling,
and page-level formatting. All output is compatible with PBI visual.json
objects schema for direct merge into pbip_generator output.

Pure functions. No AI calls. No file I/O. No side effects.
"""

from core.color_extractor import _normalize_hex, _hex_to_hsl


# Semantic color defaults (good/bad/neutral) derived from palette
def _derive_semantic_colors(palette):
    """Derive good/bad/neutral semantic colors from a palette.

    Looks for green-ish (good), red-ish (bad), and gray/blue (neutral).
    Falls back to professional defaults.
    """
    good = "#59a14f"
    bad = "#e15759"
    neutral = "#bab0ac"

    for c in palette:
        hex6 = _normalize_hex(c)
        if not hex6:
            continue
        h, s, l = _hex_to_hsl(hex6)
        if s < 0.2:
            continue
        # Green range: 90-150
        if 90 <= h <= 150 and s > 0.3:
            good = f"#{hex6}"
        # Red range: 0-30 or 330-360
        elif (h <= 30 or h >= 330) and s > 0.3:
            bad = f"#{hex6}"
        # Blue/gray range: 200-250
        elif 200 <= h <= 250 and s < 0.5:
            neutral = f"#{hex6}"

    return {"good": good, "bad": bad, "neutral": neutral}


def style_visual(ws_design, color_palette, visual_type):
    """Build enhanced PBI visual 'objects' dict from Tableau design metadata.

    Args:
        ws_design: per-worksheet design dict from enhanced parser
            (mark_colors, background_color, title_font, axis_config,
            mark_style, border).
        color_palette: list of hex colors from build_unified_palette().
        visual_type: PBI visual type string (e.g. "clusteredColumnChart").

    Returns:
        dict suitable for merging into visual.json["visual"]["objects"].
    """
    if not ws_design:
        ws_design = {}
    if not color_palette:
        color_palette = []

    objects = {}

    # -- General: background + border --
    bg_color = ws_design.get("background_color", "")
    border = ws_design.get("border", {})
    general_props = {}
    if bg_color:
        general_props["color"] = {"solid": {"color": bg_color}}
        general_props["show"] = {"expr": {"Literal": {"Value": "true"}}}
    if general_props:
        objects["background"] = [{"properties": general_props}]

    if border:
        border_color = border.get("border-color", "")
        border_weight = border.get("border-width", "")
        outline_props = {}
        if border_color:
            outline_props["color"] = {"solid": {"color": border_color}}
        if border_weight:
            try:
                outline_props["weight"] = int(border_weight.replace("px", "").strip())
            except (ValueError, AttributeError):
                pass
        if outline_props:
            objects["outline"] = [{"properties": outline_props}]

    # -- Title formatting --
    title_font = ws_design.get("title_font", {})
    if title_font:
        from core.design_translator import _FONT_MAP
        font_name = title_font.get("fontname", "")
        mapped_font = _FONT_MAP.get(font_name, font_name or "Segoe UI")
        font_size_raw = title_font.get("fontsize", "")
        try:
            font_size = int(str(font_size_raw).replace("pt", "").replace("px", "").strip())
        except (ValueError, TypeError):
            font_size = 12
        font_color = title_font.get("fontcolor", "#333333")
        bold = title_font.get("bold", False)

        title_props = {
            "show": {"expr": {"Literal": {"Value": "true"}}},
            "fontFamily": mapped_font,
            "fontSize": font_size,
            "fontColor": {"solid": {"color": font_color}},
        }
        if bold:
            title_props["bold"] = {"expr": {"Literal": {"Value": "true"}}}
        objects["title"] = [{"properties": title_props}]

    # -- Data labels --
    mark_style = ws_design.get("mark_style", {})
    labels_visible = mark_style.get("labels_visible", False)
    label_props = {
        "show": {"expr": {"Literal": {"Value": "true" if labels_visible else "false"}}},
        "fontFamily": "Segoe UI",
    }
    # Position data labels based on chart type
    if visual_type in ("clusteredColumnChart", "clusteredBarChart", "stackedBarChart"):
        label_props["labelPosition"] = "outsideEnd"
    elif visual_type in ("lineChart", "areaChart"):
        label_props["labelPosition"] = "above"
    if labels_visible:
        label_props["fontSize"] = 9
    objects["labels"] = [{"properties": label_props}]

    # -- Data point fill colors --
    mark_colors = ws_design.get("mark_colors", [])
    if mark_colors:
        fill_colors = [mc.get("color", "") for mc in mark_colors if mc.get("color")]
        if fill_colors:
            objects["dataPoint"] = [{
                "properties": {
                    "fill": {"solid": {"color": fill_colors[0]}},
                }
            }]
    elif color_palette:
        # Use first palette color as data point fill
        objects["dataPoint"] = [{
            "properties": {
                "fill": {"solid": {"color": color_palette[0]}},
            }
        }]

    # -- Axis formatting --
    axis_config = ws_design.get("axis_config", [])
    has_hidden_axis = any(ax.get("hidden") for ax in axis_config)

    cat_axis_props = {
        "show": {"expr": {"Literal": {"Value": "true" if not has_hidden_axis else "false"}}},
        "fontFamily": "Segoe UI",
        "fontSize": 9,
    }
    val_axis_props = {
        "show": {"expr": {"Literal": {"Value": "true"}}},
        "fontFamily": "Segoe UI",
        "fontSize": 9,
    }

    # Axis titles
    for ax in axis_config:
        ax_type = ax.get("type", "")
        title = ax.get("title", "")
        if ax_type == "x":
            if title:
                cat_axis_props["titleText"] = title
        elif ax_type == "y":
            if title:
                val_axis_props["titleText"] = title

    objects["categoryAxis"] = [{"properties": cat_axis_props}]
    objects["valueAxis"] = [{"properties": val_axis_props}]

    # -- Gridlines --
    gridline_props = {
        "show": {"expr": {"Literal": {"Value": "true"}}},
    }
    if color_palette:
        # Light version of first color for gridlines
        hex6 = _normalize_hex(color_palette[0])
        if hex6:
            h, s, l = _hex_to_hsl(hex6)
            # Very light gridline color
            gridline_props["color"] = {"solid": {"color": "#e8e8e8"}}
    objects["gridlines"] = [{"properties": gridline_props}]

    # -- Legend formatting --
    if visual_type not in ("cardVisual", "tableEx", "matrix"):
        legend_props = {
            "show": {"expr": {"Literal": {"Value": "true"}}},
            "position": "right",
            "fontFamily": "Segoe UI",
            "fontSize": 9,
        }
        objects["legend"] = [{"properties": legend_props}]

    return objects


def style_page(page_visuals, page_bg_color="", dashboard_title=""):
    """Build page-level formatting config.

    Args:
        page_visuals: list of visual config dicts for this page.
        page_bg_color: hex color for page background.
        dashboard_title: title text for the page header.

    Returns:
        dict with page_formatting (for page.json) and optional
        title_visual (a textbox visual config to add to the page).
    """
    result = {
        "page_formatting": {},
        "title_visual": {},
    }

    # Page background
    if page_bg_color:
        result["page_formatting"]["background"] = {
            "color": {"solid": {"color": page_bg_color}},
            "transparency": 0,
        }

    # Title textbox visual (appears at top of page)
    if dashboard_title:
        result["title_visual"] = {
            "name": "title_textbox",
            "position": {
                "x": 12, "y": 8, "z": 0,
                "width": 400, "height": 40,
                "tabOrder": 0,
            },
            "visual": {
                "visualType": "textbox",
                "objects": {
                    "general": [{
                        "properties": {
                            "paragraphs": [{
                                "textRuns": [{
                                    "value": dashboard_title,
                                    "textStyle": {
                                        "fontFamily": "Segoe UI Semibold",
                                        "fontSize": 18,
                                    },
                                }],
                            }],
                        }
                    }],
                },
            },
        }

    return result


def build_enhanced_theme(color_palette, fonts=None, dashboard_title="Dashboard"):
    """Build a PBI theme JSON with enhanced styling from extracted palette.

    Args:
        color_palette: list of hex color strings from build_unified_palette().
        fonts: dict with fontFamily, fontSize, fontColor (optional).
        dashboard_title: theme name.

    Returns:
        dict: complete Power BI theme JSON structure.
    """
    if not fonts:
        fonts = {"fontFamily": "Segoe UI", "fontSize": 10, "fontColor": "#333333"}

    # Use palette or professional defaults
    data_colors = list(color_palette) if color_palette else [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
        "#9C755F", "#BAB0AC",
    ]

    # Derive semantic colors from the palette
    semantic = _derive_semantic_colors(data_colors)

    # Determine background/foreground from palette context
    background = "#FFFFFF"
    foreground = fonts.get("fontColor", "#333333")
    table_accent = data_colors[0] if data_colors else "#4E79A7"

    theme = {
        "name": dashboard_title,
        "dataColors": data_colors,
        "background": background,
        "foreground": foreground,
        "tableAccent": table_accent,
        "good": semantic["good"],
        "bad": semantic["bad"],
        "neutral": semantic["neutral"],
        "maximum": semantic["good"],
        "minimum": semantic["bad"],
        "center": semantic["neutral"],
        "visualStyles": {
            "*": {
                "*": {
                    "background": [{
                        "color": {"solid": {"color": background}},
                        "transparency": 0,
                    }],
                    "title": [{
                        "fontFamily": fonts.get("fontFamily", "Segoe UI"),
                        "fontSize": max(fonts.get("fontSize", 10), 11),
                        "color": {"solid": {"color": foreground}},
                        "bold": True,
                    }],
                    "labels": [{
                        "fontFamily": fonts.get("fontFamily", "Segoe UI"),
                        "fontSize": 9,
                        "color": {"solid": {"color": foreground}},
                    }],
                    "categoryAxis": [{
                        "fontFamily": fonts.get("fontFamily", "Segoe UI"),
                        "fontSize": 9,
                    }],
                    "valueAxis": [{
                        "fontFamily": fonts.get("fontFamily", "Segoe UI"),
                        "fontSize": 9,
                    }],
                }
            },
            "card": {
                "*": {
                    "labels": [{
                        "fontSize": 24,
                        "color": {"solid": {"color": table_accent}},
                        "fontFamily": fonts.get("fontFamily", "Segoe UI"),
                    }],
                }
            },
            "slicer": {
                "*": {
                    "general": [{
                        "outlineColor": {"solid": {"color": "#e8e8e8"}},
                        "outlineWeight": 1,
                    }],
                }
            },
        },
    }

    return theme
