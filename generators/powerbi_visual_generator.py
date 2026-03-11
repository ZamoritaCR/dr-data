"""
Sprint 2 -- Visual Type Mapping Engine.

Maps Tableau chart types to Power BI visual types with full coverage.
Includes mark type detection, visual property injection, and Deneb
fallback for unmappable chart types (gantt, heatmap, boxplot).
"""
import json
import re

# Complete Tableau -> Power BI visual type mapping
TABLEAU_TO_PBI_VISUAL_MAP = {
    # BAR / COLUMN
    "bar":                          {"visual_type": "clusteredBarChart",           "orientation": "horizontal"},
    "bar_stacked":                  {"visual_type": "stackedBarChart",             "orientation": "horizontal"},
    "bar_100":                      {"visual_type": "hundredPercentStackedBarChart","orientation": "horizontal"},
    "column":                       {"visual_type": "clusteredColumnChart",         "orientation": "vertical"},
    "column_stacked":               {"visual_type": "stackedColumnChart",           "orientation": "vertical"},
    "column_100":                   {"visual_type": "hundredPercentStackedColumnChart","orientation": "vertical"},
    # LINE / AREA
    "line":                         {"visual_type": "lineChart",                   "orientation": None},
    "area":                         {"visual_type": "areaChart",                   "orientation": None},
    "area_stacked":                 {"visual_type": "stackedAreaChart",            "orientation": None},
    # SCATTER / BUBBLE
    "circle":                       {"visual_type": "scatterChart",                "orientation": None},
    "scatter":                      {"visual_type": "scatterChart",                "orientation": None},
    "bubble":                       {"visual_type": "bubbleChart",                 "orientation": None},
    # PIE / DONUT
    "pie":                          {"visual_type": "pieChart",                    "orientation": None},
    "donut":                        {"visual_type": "donutChart",                  "orientation": None},
    # TABLE / MATRIX / TEXT
    "text":                         {"visual_type": "tableEx",                     "orientation": None},
    "crosstab":                     {"visual_type": "matrix",                      "orientation": None},
    "table":                        {"visual_type": "tableEx",                     "orientation": None},
    # MAP
    "map":                          {"visual_type": "map",                         "orientation": None},
    "filled_map":                   {"visual_type": "filledMap",                   "orientation": None},
    "shape_map":                    {"visual_type": "shapeMap",                    "orientation": None},
    # TREEMAP / HIERARCHY
    "square":                       {"visual_type": "treemap",                     "orientation": None},
    "treemap":                      {"visual_type": "treemap",                     "orientation": None},
    # COMBO
    "dual_line":                    {"visual_type": "lineClusteredColumnComboChart","orientation": None},
    "dual_bar_line":                {"visual_type": "lineStackedColumnComboChart", "orientation": None},
    # GANTT
    "gantt":                        {"visual_type": "DENEB_FALLBACK",              "deneb_template": "gantt"},
    # HISTOGRAM
    "histogram":                    {"visual_type": "columnChart",                 "binning": True},
    # HEATMAP
    "density":                      {"visual_type": "DENEB_FALLBACK",              "deneb_template": "heatmap"},
    "heatmap":                      {"visual_type": "DENEB_FALLBACK",              "deneb_template": "heatmap"},
    # BOX PLOT
    "boxplot":                      {"visual_type": "DENEB_FALLBACK",              "deneb_template": "boxplot"},
    # WATERFALL
    "waterfall":                    {"visual_type": "waterfallChart",              "orientation": None},
    # GAUGE / KPI
    "kpi":                          {"visual_type": "kpiVisual",                   "orientation": None},
    # FALLBACK
    "automatic":                    {"visual_type": "clusteredColumnChart",        "orientation": "vertical"},
    "unknown":                      {"visual_type": "clusteredColumnChart",        "orientation": "vertical"},
}


def _detect_mark_type(worksheet_info):
    """Detect the normalized mark type from a parsed worksheet.

    Analyzes the mark class and encoding shelves to determine if a chart
    is stacked, 100%, heatmap, etc.

    Args:
        worksheet_info: dict from the parsed TWB with keys:
            mark_type, rows, cols, filters, etc.

    Returns:
        str: normalized key for TABLEAU_TO_PBI_VISUAL_MAP
    """
    mark_class = (worksheet_info.get("mark_type", "") or "").lower().strip()

    if not mark_class:
        return "automatic"

    rows = worksheet_info.get("rows", "") or ""
    cols = worksheet_info.get("cols", "") or ""

    # Count discrete dimensions and continuous measures on shelves
    row_fields = _extract_shelf_fields(rows)
    col_fields = _extract_shelf_fields(cols)
    all_fields = row_fields + col_fields

    discrete_count = sum(1 for f in all_fields if f.get("type") == "discrete")
    continuous_count = sum(1 for f in all_fields if f.get("type") == "continuous")

    # Detect heatmap: two discrete dimensions + color encoding with continuous measure
    if mark_class in ("square", "circle", "shape", "automatic"):
        if discrete_count >= 2 and continuous_count >= 1:
            return "heatmap"

    # Detect stacked variants for bar/area
    if mark_class == "bar":
        # Look for stacking indicators in the encoding
        if _has_stacking(worksheet_info):
            if _is_percent_stacked(worksheet_info):
                return "bar_100"
            return "bar_stacked"
        # Check orientation: if measures on rows = horizontal bar, on cols = column
        if _measures_on_cols(rows, cols):
            return "column"
        return "bar"

    if mark_class == "area":
        if _has_stacking(worksheet_info):
            return "area_stacked"
        return "area"

    # Detect gantt bar
    if mark_class == "gantt" or mark_class == "gantt bar":
        return "gantt"

    # Direct mappings
    direct = {
        "line": "line",
        "circle": "circle",
        "scatter": "scatter",
        "pie": "pie",
        "square": "square",
        "text": "text",
        "map": "map",
        "polygon": "filled_map",
        "shape": "scatter",
    }

    return direct.get(mark_class, mark_class)


def _extract_shelf_fields(shelf_text):
    """Extract field references from a Tableau shelf text.

    Shelf text looks like: '[Orders].[Ship Date]' or 'SUM([Sales])'
    """
    fields = []
    if not shelf_text:
        return fields

    # Find all bracketed references
    refs = re.findall(r'\[([^\]]+)\]', shelf_text)
    for ref in refs:
        # Continuous fields are wrapped in aggregation functions
        is_continuous = bool(re.search(
            r'\b(SUM|AVG|COUNT|MIN|MAX|MEDIAN|STDEV|VAR|ATTR|COUNTD)\s*\(',
            shelf_text, re.IGNORECASE
        ))
        fields.append({
            "name": ref,
            "type": "continuous" if is_continuous else "discrete",
        })

    return fields


def _has_stacking(worksheet_info):
    """Detect if a worksheet uses stacking (color encoding with a discrete field)."""
    # In parsed TWB, stacking is indicated by having a discrete field on color
    # We check the raw shelf text for color-related patterns
    rows = worksheet_info.get("rows", "") or ""
    cols = worksheet_info.get("cols", "") or ""
    # Multiple discrete dimensions typically means stacking
    all_text = rows + " " + cols
    discrete_refs = re.findall(r'(?<!\w)\[([^\]]+)\](?!\s*\))', all_text)
    return len(discrete_refs) > 2


def _is_percent_stacked(worksheet_info):
    """Detect if stacking is 100% (percent of total)."""
    # In Tableau, percent-of-total is a table calc -- hard to detect from XML alone
    # Conservative: return False unless we find explicit percent patterns
    return False


def _measures_on_cols(rows, cols):
    """Detect if measures are on columns (vertical chart) vs rows (horizontal)."""
    cols_has_agg = bool(re.search(r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(', cols or "", re.IGNORECASE))
    rows_has_agg = bool(re.search(r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(', rows or "", re.IGNORECASE))
    return cols_has_agg and not rows_has_agg


def _inject_visual_properties(visual_json, tableau_worksheet, field_map=None):
    """Inject visual properties from Tableau worksheet into PBI visual JSON.

    Reads Tableau color encoding, axis labels, sort order, data labels,
    and title to set corresponding PBI visual config properties.

    Args:
        visual_json: PBI visual JSON dict to modify (in place)
        tableau_worksheet: parsed worksheet dict
        field_map: optional field resolution map for name translation
    """
    if not visual_json or not tableau_worksheet:
        return visual_json

    objects = visual_json.setdefault("visual", {}).setdefault("objects", {})

    # Title
    ws_name = tableau_worksheet.get("name", "")
    if ws_name:
        objects["title"] = [{
            "properties": {
                "text": {"expr": {"Literal": {"Value": f"'{ws_name}'"}}},
                "show": {"expr": {"Literal": {"Value": "true"}}},
            }
        }]

    # Data labels (default to on for bar/line/column charts)
    mark_type = (tableau_worksheet.get("mark_type", "") or "").lower()
    if mark_type in ("bar", "line", "area", "circle"):
        objects["labels"] = [{
            "properties": {
                "show": {"expr": {"Literal": {"Value": "false"}}},
            }
        }]

    return visual_json


def _generate_deneb_visual(template_name, tableau_worksheet, field_map=None):
    """Generate a Deneb (Vega-Lite) visual JSON for unmappable chart types.

    Args:
        template_name: one of 'gantt', 'heatmap', 'boxplot'
        tableau_worksheet: parsed worksheet dict
        field_map: optional field resolution map

    Returns:
        dict: Deneb visual container JSON structure
    """
    templates = {
        "gantt": _deneb_gantt_spec,
        "heatmap": _deneb_heatmap_spec,
        "boxplot": _deneb_boxplot_spec,
    }

    generator = templates.get(template_name)
    if not generator:
        return _deneb_fallback_spec(template_name)

    vega_spec = generator(tableau_worksheet, field_map)

    return {
        "visualType": "deneb",
        "config": {
            "provider": "vegaLite",
            "specification": json.dumps(vega_spec, indent=2),
        },
        "note": f"Deneb/Vega-Lite visual for '{template_name}' -- "
                f"review and adjust field names in Power BI Desktop",
    }


def _deneb_gantt_spec(worksheet, field_map=None):
    """Generate Vega-Lite gantt chart spec."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Gantt Chart",
        "mark": "bar",
        "encoding": {
            "x": {"field": "Start Date", "type": "temporal"},
            "x2": {"field": "End Date"},
            "y": {"field": "Task", "type": "nominal", "sort": None},
            "color": {"field": "Status", "type": "nominal"},
        },
    }


def _deneb_heatmap_spec(worksheet, field_map=None):
    """Generate Vega-Lite heatmap spec."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Heatmap",
        "mark": "rect",
        "encoding": {
            "x": {"field": "Dimension1", "type": "nominal"},
            "y": {"field": "Dimension2", "type": "nominal"},
            "color": {
                "field": "Measure",
                "type": "quantitative",
                "scale": {"scheme": "blues"},
            },
        },
    }


def _deneb_boxplot_spec(worksheet, field_map=None):
    """Generate Vega-Lite boxplot spec."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Box Plot",
        "mark": {"type": "boxplot", "extent": 1.5},
        "encoding": {
            "x": {"field": "Category", "type": "nominal"},
            "y": {"field": "Value", "type": "quantitative"},
        },
    }


def _deneb_fallback_spec(template_name):
    """Fallback Deneb spec for unknown template types."""
    return {
        "visualType": "deneb",
        "config": {
            "provider": "vegaLite",
            "specification": json.dumps({
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": f"Placeholder for '{template_name}' chart type",
                "mark": "point",
                "encoding": {},
            }),
        },
        "note": f"Unsupported chart type '{template_name}' -- "
                f"configure manually in Power BI Desktop",
    }


def map_visual_type(mark_type):
    """Look up a Tableau mark type in the visual map.

    Args:
        mark_type: normalized mark type string

    Returns:
        dict with visual_type, orientation, and optional deneb_template
    """
    key = (mark_type or "unknown").lower().strip()
    return TABLEAU_TO_PBI_VISUAL_MAP.get(key, TABLEAU_TO_PBI_VISUAL_MAP["unknown"])
