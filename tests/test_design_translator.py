"""Tests for core.design_translator -- pure data transformation module."""
import pytest

from core.design_translator import (
    translate_colors,
    translate_positions,
    translate_fonts,
    translate_chart_type,
    build_pbi_theme,
    build_visual_formatting,
    _rects_overlap,
    _DEFAULT_DATA_COLORS,
)


# ------------------------------------------------------------------ #
#  translate_colors                                                    #
# ------------------------------------------------------------------ #

class TestTranslateColors:
    """Color translation from Tableau palettes to PBI dataColors."""

    def test_real_palette(self):
        design = {
            "color_palettes": [{
                "name": "Sales Palette",
                "type": "regular",
                "colors": ["#FF6600", "#0099FF", "#33CC33"],
            }],
            "global_fonts": {},
        }
        result = translate_colors(design)
        assert result["dataColors"] == ["#FF6600", "#0099FF", "#33CC33"]
        assert result["tableAccent"] == "#FF6600"
        assert result["background"] == "#FFFFFF"

    def test_no_palette_fallback(self):
        design = {"color_palettes": [], "global_fonts": {}}
        result = translate_colors(design)
        assert result["dataColors"] == list(_DEFAULT_DATA_COLORS)
        assert result["tableAccent"] == _DEFAULT_DATA_COLORS[0]

    def test_none_input(self):
        result = translate_colors(None)
        assert result["dataColors"] == list(_DEFAULT_DATA_COLORS)
        assert result["foreground"] == "#333333"

    def test_empty_dict(self):
        result = translate_colors({})
        assert result["dataColors"] == list(_DEFAULT_DATA_COLORS)

    def test_font_color_as_foreground(self):
        design = {
            "color_palettes": [{"colors": ["#AABBCC"]}],
            "global_fonts": {"font-color": "#112233"},
        }
        result = translate_colors(design)
        assert result["foreground"] == "#112233"

    def test_datasource_color_maps_fallback(self):
        """When no palette but datasource color maps exist, use those."""
        design = {
            "color_palettes": [],
            "global_fonts": {},
            "datasource_color_maps": [{
                "field": "Category",
                "mappings": [
                    {"color": "#AA0000", "value": "A"},
                    {"color": "#00BB00", "value": "B"},
                ],
            }],
        }
        result = translate_colors(design)
        assert result["dataColors"] == ["#AA0000", "#00BB00"]

    def test_colors_without_hash_prefix(self):
        design = {
            "color_palettes": [{"colors": ["FF6600", "0099FF"]}],
            "global_fonts": {},
        }
        result = translate_colors(design)
        assert result["dataColors"] == ["#FF6600", "#0099FF"]


# ------------------------------------------------------------------ #
#  translate_positions                                                 #
# ------------------------------------------------------------------ #

class TestTranslatePositions:
    """Position scaling from Tableau canvas to PBI canvas."""

    def test_basic_scaling(self):
        zones = [{
            "name": "Sales Chart",
            "layout": {"x": 0, "y": 0, "w": 500, "h": 400},
        }]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas)
        assert len(result) == 1
        r = result[0]
        assert r["name"] == "Sales Chart"
        # x = (0/1000) * (1280-24) + 12 = 12
        assert r["x"] == 12
        assert r["y"] == 12
        # w = (500/1000) * 1256 = 628
        assert r["width"] == 628
        # h = (400/800) * 696 = 348
        assert r["height"] == 348

    def test_different_canvas_sizes(self):
        zones = [{
            "name": "Chart",
            "layout": {"x": 100, "y": 100, "w": 200, "h": 200},
        }]
        # Large Tableau canvas
        canvas = {"width": 2000, "height": 1000}
        result = translate_positions(zones, canvas)
        assert len(result) == 1
        r = result[0]
        assert r["x"] > 12  # offset from margin
        assert r["width"] > 0
        assert r["height"] > 0

    def test_clamping_to_bounds(self):
        """A zone that would exceed the PBI canvas is clamped."""
        zones = [{
            "name": "Oversized",
            "layout": {"x": 900, "y": 700, "w": 200, "h": 200},
        }]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas)
        assert len(result) == 1
        r = result[0]
        # x + width should not exceed pbi_w - margin
        assert r["x"] + r["width"] <= 1280 - 12
        assert r["y"] + r["height"] <= 720 - 12

    def test_empty_zones(self):
        result = translate_positions([], {"width": 1000, "height": 800})
        assert result == []

    def test_zero_canvas_with_zones(self):
        """Canvas 0x0 but zones have data -- auto-detect coordinate bounds."""
        zones = [{"name": "A", "layout": {"x": 0, "y": 0, "w": 100, "h": 100}}]
        result = translate_positions(zones, {"width": 0, "height": 0})
        assert len(result) == 1
        assert result[0]["name"] == "A"

    def test_zero_canvas_no_zones(self):
        result = translate_positions([], {"width": 0, "height": 0})
        assert result == []

    def test_none_canvas(self):
        result = translate_positions([], None)
        assert result == []

    def test_zones_with_zero_size_skipped(self):
        zones = [
            {"name": "NoSize", "layout": {"x": 0, "y": 0, "w": 0, "h": 0}},
            {"name": "HasSize", "layout": {"x": 0, "y": 0, "w": 100, "h": 100}},
        ]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas)
        assert len(result) == 1
        assert result[0]["name"] == "HasSize"

    def test_collision_detection(self):
        """Two overlapping zones should be resolved so they no longer overlap."""
        zones = [
            {"name": "A", "layout": {"x": 0, "y": 0, "w": 600, "h": 400}},
            {"name": "B", "layout": {"x": 100, "y": 100, "w": 600, "h": 400}},
        ]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas)
        assert len(result) == 2
        a, b = result[0], result[1]
        # After collision resolution, they should not overlap
        overlaps = _rects_overlap(a, b)
        assert not overlaps, f"Rects still overlap after resolution: {a} vs {b}"

    def test_non_overlapping_zones_untouched(self):
        zones = [
            {"name": "Left", "layout": {"x": 0, "y": 0, "w": 400, "h": 400}},
            {"name": "Right", "layout": {"x": 500, "y": 0, "w": 400, "h": 400}},
        ]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas)
        assert len(result) == 2

    def test_custom_pbi_canvas(self):
        zones = [{
            "name": "Full",
            "layout": {"x": 0, "y": 0, "w": 1000, "h": 800},
        }]
        canvas = {"width": 1000, "height": 800}
        result = translate_positions(zones, canvas, pbi_canvas=(1920, 1080))
        r = result[0]
        # Should scale to ~1920x1080 bounds
        assert r["width"] > 1200


# ------------------------------------------------------------------ #
#  translate_fonts                                                     #
# ------------------------------------------------------------------ #

class TestTranslateFonts:
    """Font mapping from Tableau families to PBI equivalents."""

    def test_tableau_book(self):
        result = translate_fonts({"font-family": "Tableau Book", "font-size": "12pt"})
        assert result["fontFamily"] == "Segoe UI"
        assert result["fontSize"] == 12

    def test_tableau_bold(self):
        result = translate_fonts({"font-family": "Tableau Bold"})
        assert result["fontFamily"] == "Segoe UI Semibold"

    def test_tableau_regular(self):
        result = translate_fonts({"font-family": "Tableau Regular"})
        assert result["fontFamily"] == "Segoe UI"

    def test_unknown_font_passthrough(self):
        result = translate_fonts({"font-family": "Arial"})
        # Unknown fonts default to Segoe UI
        assert result["fontFamily"] == "Segoe UI"

    def test_none_input(self):
        result = translate_fonts(None)
        assert result["fontFamily"] == "Segoe UI"
        assert result["fontSize"] == 10
        assert result["fontColor"] == "#333333"

    def test_empty_dict(self):
        result = translate_fonts({})
        assert result["fontFamily"] == "Segoe UI"

    def test_font_color_preserved(self):
        result = translate_fonts({"font-color": "#FF0000"})
        assert result["fontColor"] == "#FF0000"

    def test_font_size_px(self):
        result = translate_fonts({"font-size": "14px"})
        assert result["fontSize"] == 14


# ------------------------------------------------------------------ #
#  translate_chart_type                                                #
# ------------------------------------------------------------------ #

class TestTranslateChartType:
    """Chart type mapping for every known Tableau mark type."""

    @pytest.mark.parametrize("tableau,expected", [
        ("bar", "clusteredBarChart"),
        ("line", "lineChart"),
        ("area", "areaChart"),
        ("circle", "scatterChart"),
        ("square", "matrix"),
        ("text", "tableEx"),
        ("map", "map"),
        ("polygon", "filledMap"),
        ("multipolygon", "filledMap"),
        ("pie", "pieChart"),
        ("gantt-bar", "treemap"),
        ("automatic", "clusteredBarChart"),
        ("shape", "scatterChart"),
        ("density", "scatterChart"),
        ("heatmap", "matrix"),
    ])
    def test_known_types(self, tableau, expected):
        assert translate_chart_type(tableau) == expected

    def test_unknown_type_default(self):
        assert translate_chart_type("unknown_type_xyz") == "clusteredBarChart"

    def test_none_input(self):
        assert translate_chart_type(None) == "clusteredBarChart"

    def test_empty_string(self):
        assert translate_chart_type("") == "clusteredBarChart"

    def test_case_insensitive(self):
        assert translate_chart_type("Bar") == "clusteredBarChart"
        assert translate_chart_type("LINE") == "lineChart"

    def test_whitespace_handling(self):
        assert translate_chart_type("  bar  ") == "clusteredBarChart"


# ------------------------------------------------------------------ #
#  build_pbi_theme                                                     #
# ------------------------------------------------------------------ #

class TestBuildPbiTheme:
    """Full theme JSON structure validation."""

    def test_structure_keys(self):
        design = {
            "color_palettes": [{"colors": ["#AA0000", "#00BB00"]}],
            "global_fonts": {"font-family": "Tableau Book", "font-size": "11pt"},
        }
        theme = build_pbi_theme(design, "Test Dashboard")
        assert theme["name"] == "Test Dashboard"
        assert "dataColors" in theme
        assert "background" in theme
        assert "foreground" in theme
        assert "tableAccent" in theme
        assert "visualStyles" in theme
        assert "*" in theme["visualStyles"]
        assert "*" in theme["visualStyles"]["*"]

    def test_visual_styles_structure(self):
        design = {
            "color_palettes": [{"colors": ["#123456"]}],
            "global_fonts": {"font-family": "Tableau Bold", "font-size": "14pt"},
        }
        theme = build_pbi_theme(design)
        vs = theme["visualStyles"]["*"]["*"]
        assert "background" in vs
        assert "title" in vs
        assert vs["title"][0]["fontFamily"] == "Segoe UI Semibold"
        assert vs["title"][0]["fontSize"] == 14

    def test_fallback_with_no_design(self):
        theme = build_pbi_theme(None)
        assert theme["name"] == "Dashboard"
        assert theme["dataColors"] == list(_DEFAULT_DATA_COLORS)

    def test_all_values_are_serializable(self):
        """Theme must be JSON-serializable."""
        import json
        design = {
            "color_palettes": [{"colors": ["#FF0000"]}],
            "global_fonts": {"font-family": "Tableau Regular"},
        }
        theme = build_pbi_theme(design, "Serialization Test")
        serialized = json.dumps(theme)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["name"] == "Serialization Test"


# ------------------------------------------------------------------ #
#  build_visual_formatting                                             #
# ------------------------------------------------------------------ #

class TestBuildVisualFormatting:
    """Visual-level formatting from worksheet design data."""

    def test_with_mark_colors(self):
        ws_design = {
            "mark_colors": [{"color": "#FF0000", "value": "A"}],
            "background_color": "#EEEEEE",
            "title_font": {"fontname": "Tableau Bold", "fontsize": "16", "fontcolor": "#000000"},
            "axis_config": [],
            "mark_style": {"labels_visible": True},
            "border": {},
        }
        fmt = build_visual_formatting(ws_design)
        assert "dataPoint" in fmt
        assert fmt["dataPoint"][0]["properties"]["fill"]["solid"]["color"] == "#FF0000"
        assert "labels" in fmt
        assert fmt["labels"][0]["properties"]["show"]["expr"]["Literal"]["Value"] == "true"

    def test_hidden_axis(self):
        ws_design = {
            "mark_colors": [],
            "background_color": "",
            "title_font": {},
            "axis_config": [{"field": "X", "hidden": True}],
            "mark_style": {},
            "border": {},
        }
        fmt = build_visual_formatting(ws_design)
        assert fmt["categoryAxis"][0]["properties"]["show"]["expr"]["Literal"]["Value"] == "false"

    def test_title_font_mapping(self):
        ws_design = {
            "mark_colors": [],
            "background_color": "",
            "title_font": {"fontname": "Tableau Book", "fontsize": "14", "fontcolor": "#222222"},
            "axis_config": [],
            "mark_style": {},
            "border": {},
        }
        fmt = build_visual_formatting(ws_design)
        assert "title" in fmt
        assert fmt["title"][0]["properties"]["fontFamily"] == "Segoe UI"
        assert fmt["title"][0]["properties"]["fontSize"] == 14

    def test_none_input(self):
        assert build_visual_formatting(None) == {}

    def test_empty_dict(self):
        assert build_visual_formatting({}) == {}

    def test_labels_default_hidden(self):
        ws_design = {
            "mark_colors": [],
            "background_color": "",
            "title_font": {},
            "axis_config": [],
            "mark_style": {},
            "border": {},
        }
        fmt = build_visual_formatting(ws_design)
        assert fmt["labels"][0]["properties"]["show"]["expr"]["Literal"]["Value"] == "false"
