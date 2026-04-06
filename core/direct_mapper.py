"""
Direct Tableau-to-PBIP Mapper.

Deterministic mapping from Tableau spec to PBIP config dict.
Bypasses Claude and GPT-4 for all visual structure -- chart types,
positions, field bindings, colors. AI is only used for DAX translation
of complex calculated fields.

This module produces the exact same config dict format that
PBIPGenerator.generate() expects, so it plugs in as a drop-in
replacement for the OpenAI engine output.
"""

import re
import json

from core.design_translator import (
    translate_chart_type,
    translate_positions,
    translate_colors,
    translate_fonts,
)


# ------------------------------------------------------------------ #
#  Field name cleaning                                                 #
# ------------------------------------------------------------------ #

# Tableau shelf prefixes: aggregation + dimension qualifiers.
_SHELF_PREFIX_RE = re.compile(
    r'^(sum|avg|count|min|max|countd|median|attr|'
    r'none|usr|yr|mn|qr|tqr|tmn|twk|day|'
    r'mdy|wk|md|qy|my)\:(.+?)(?:\:[a-z]+)?$',
    re.IGNORECASE,
)

# Simple Tableau aggregation formulas that can be converted to DAX
# without AI assistance.
_SIMPLE_AGG_RE = re.compile(
    r'^\s*(SUM|AVG|COUNT|COUNTD|MIN|MAX|MEDIAN)\s*\(\s*\[([^\]]+)\]\s*\)\s*$',
    re.IGNORECASE,
)

# DAX equivalents for simple Tableau aggregations.
_DAX_AGG_MAP = {
    "sum": "SUM",
    "avg": "AVERAGE",
    "count": "COUNT",
    "countd": "DISTINCTCOUNT",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
}


def _clean_field_name(name):
    """Strip Tableau prefixes, brackets, quotes from a field reference.

    Handles patterns like:
      federated.xxx   -> skip (datasource ID)
      none:Region:nk  -> Region
      sum:Sales:qk    -> Sales
      "Region"        -> Region
      [Region]        -> Region
    """
    if not name:
        return ""
    name = name.strip()

    # Skip datasource IDs
    if name.startswith("federated.") or name.startswith(":"):
        return ""

    # Strip brackets
    if name.startswith("[") and name.endswith("]") and name.count("[") == 1:
        name = name[1:-1]

    # Parse prefix:FieldName:suffix pattern
    m = _SHELF_PREFIX_RE.match(name)
    if m:
        name = m.group(2)
    elif ":" in name:
        parts = name.split(":")
        if len(parts) >= 2:
            candidate = parts[1].strip().strip('"').strip("'")
            if candidate:
                name = candidate

    # Strip quotes
    name = name.strip()
    if name.startswith('"') and name.endswith('"') and len(name) >= 2:
        name = name[1:-1].strip()
    if name.startswith("'") and name.endswith("'") and len(name) >= 2:
        name = name[1:-1].strip()

    # Reject names that are only punctuation
    if not name or not re.search(r'[a-zA-Z0-9]', name):
        return ""
    return name


def _clean_shelf_fields(field_refs):
    """Clean a list of Tableau shelf field references to plain names."""
    result = []
    seen = set()
    for ref in field_refs:
        name = _clean_field_name(ref)
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _resolve_field_against_profile(field_name, profile_col_names):
    """Try to match a field name against profile columns (case-insensitive).

    Returns the matched profile column name, or the original if no match.
    """
    if field_name in profile_col_names:
        return field_name
    lower_map = {c.lower(): c for c in profile_col_names}
    matched = lower_map.get(field_name.lower())
    if matched:
        return matched
    # Try partial match (field name contained in column name or vice versa)
    for col in profile_col_names:
        if field_name.lower() in col.lower() or col.lower() in field_name.lower():
            return col
    return field_name


# ------------------------------------------------------------------ #
#  Worksheet -> Visual mapping                                         #
# ------------------------------------------------------------------ #

def _classify_fields_for_chart(ws, chart_type, profile_col_names, col_types):
    """Determine category, values, and series fields for a worksheet.

    Uses the worksheet shelf fields and chart type to assign PBI roles.

    Args:
        ws: worksheet dict from the parser
        chart_type: PBI visual type string
        profile_col_names: set of column names from data profile
        col_types: dict of col_name -> semantic_type (dimension/measure)

    Returns:
        dict with category, values, series lists
    """
    rows = _clean_shelf_fields(ws.get("rows_fields", []))
    cols = _clean_shelf_fields(ws.get("cols_fields", []))
    color_field = _clean_field_name(ws.get("color_field", ""))

    # Resolve all fields against profile
    rows = [_resolve_field_against_profile(f, profile_col_names) for f in rows]
    cols = [_resolve_field_against_profile(f, profile_col_names) for f in cols]
    if color_field:
        color_field = _resolve_field_against_profile(color_field, profile_col_names)

    # Classify each field as dimension or measure based on:
    # 1. Profile semantic type (most reliable)
    # 2. Shelf position heuristic
    def is_measure(f):
        st = col_types.get(f, "")
        return st == "measure"

    # Separate into dimensions and measures from both shelves
    all_dims = []
    all_measures = []
    for f in rows + cols:
        if is_measure(f):
            all_measures.append(f)
        else:
            all_dims.append(f)

    # Deduplicate while preserving order
    def dedup(lst):
        seen = set()
        out = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    all_dims = dedup(all_dims)
    all_measures = dedup(all_measures)

    category = []
    values = []
    series = []

    # Chart-type-specific field assignment
    if chart_type in ("card", "cardVisual", "multiRowCard", "kpi"):
        # Cards: just values (first measure)
        values = all_measures[:1] or all_dims[:1]

    elif chart_type in ("tableEx", "pivotTable", "table", "matrix"):
        # Tables: all fields as values
        values = all_dims + all_measures

    elif chart_type == "slicer":
        # Slicers: first dimension as category
        category = all_dims[:1]

    elif chart_type in ("pieChart", "donutChart"):
        # Pie/donut: first dimension -> category, first measure -> values
        category = all_dims[:1]
        values = all_measures[:1]
        if not values and len(all_dims) > 1:
            values = all_dims[1:2]

    elif chart_type in ("map", "filledMap"):
        # Maps: geographic field -> category, measure -> values
        category = all_dims[:1]
        values = all_measures[:1]

    elif chart_type in ("scatterChart",):
        # Scatter: dims -> category, first two measures -> values
        category = all_dims[:1]
        values = all_measures[:2]

    else:
        # Bar, column, line, area, combo, etc.
        # Dimensions -> category, measures -> values
        category = all_dims[:1] if all_dims else []
        values = all_measures if all_measures else []

        # If we have no measures, promote remaining dims to values
        if not values and len(all_dims) > 1:
            values = all_dims[1:]

    # Series: color field if it's a dimension and not already used
    used = set(category + values)
    if color_field and color_field not in used and not is_measure(color_field):
        series = [color_field]

    return {
        "category": category,
        "values": values,
        "series": series,
    }


# ------------------------------------------------------------------ #
#  Measure generation                                                  #
# ------------------------------------------------------------------ #

def _build_measures_from_spec(tableau_spec, table_name, profile_col_names):
    """Build DAX measures from Tableau calculated fields and shelf usage.

    Simple aggregations are translated deterministically.
    Complex formulas are tagged for AI translation (placeholder DAX).

    Returns:
        list of measure dicts: {name, dax, format, needs_ai}
    """
    measures = []
    measure_names = set()

    # 1. Calculated fields
    for cf in tableau_spec.get("calculated_fields", []):
        cf_name = _clean_field_name(cf.get("name", ""))
        formula = cf.get("formula", "")
        if not cf_name or not formula:
            continue
        if cf_name in measure_names:
            continue

        # Try simple aggregation translation
        m = _SIMPLE_AGG_RE.match(formula)
        if m:
            agg = m.group(1).lower()
            field = m.group(2)
            dax_agg = _DAX_AGG_MAP.get(agg, "SUM")
            clean_field = _clean_field_name(field)
            resolved = _resolve_field_against_profile(clean_field, profile_col_names)
            dax = f"{dax_agg}('{table_name}'[{resolved}])"
            measures.append({
                "name": cf_name,
                "dax": dax,
                "format": _guess_format(dax_agg),
                "needs_ai": False,
            })
            measure_names.add(cf_name)
        else:
            # Complex formula -- generate placeholder, flag for AI
            dax = _translate_formula_heuristic(formula, table_name, profile_col_names)
            measures.append({
                "name": cf_name,
                "dax": dax,
                "format": "#,0",
                "needs_ai": True,
                "original_formula": formula,
            })
            measure_names.add(cf_name)

    # 2. Implicit measures from shelf usage
    # Fields on measure shelves that are not calculated fields
    used_on_values = set()
    for ws in tableau_spec.get("worksheets", []):
        # Fields from rows and cols that look like measures
        for shelf_key in ("rows", "cols"):
            shelf_text = ws.get(shelf_key, "")
            if not shelf_text:
                continue
            # Find aggregated references: SUM([Field]), AVG([Field]), etc.
            agg_refs = re.findall(
                r'(SUM|AVG|COUNT|COUNTD|MIN|MAX|MEDIAN)\s*\(\s*\[([^\]]+)\]\s*\)',
                shelf_text, re.IGNORECASE,
            )
            for agg, field_ref in agg_refs:
                clean = _clean_field_name(field_ref)
                if clean:
                    used_on_values.add((agg.lower(), clean))

        # Also check cols_fields/rows_fields for fields with agg prefixes
        for field_list_key in ("rows_fields", "cols_fields"):
            for ref in ws.get(field_list_key, []):
                m = _SHELF_PREFIX_RE.match(ref)
                if m:
                    prefix = m.group(1).lower()
                    field = m.group(2)
                    if prefix in _DAX_AGG_MAP:
                        clean = _clean_field_name(field)
                        if clean:
                            used_on_values.add((prefix, clean))

    for agg, field in used_on_values:
        resolved = _resolve_field_against_profile(field, profile_col_names)
        measure_name = f"Total {resolved}" if agg == "sum" else f"{agg.capitalize()} {resolved}"
        if measure_name in measure_names:
            continue

        dax_agg = _DAX_AGG_MAP.get(agg, "SUM")
        dax = f"{dax_agg}('{table_name}'[{resolved}])"
        measures.append({
            "name": measure_name,
            "dax": dax,
            "format": _guess_format(dax_agg),
            "needs_ai": False,
        })
        measure_names.add(measure_name)

    return measures


def _translate_formula_heuristic(formula, table_name, profile_col_names):
    """Best-effort heuristic translation of Tableau formulas to DAX.

    This handles common patterns without AI. For truly complex formulas
    (LOD, CASE, nested IFs), the output will be approximate and flagged
    for AI review.
    """
    result = formula

    # Replace [FieldName] with 'Table'[FieldName]
    def replace_field_ref(match):
        field = match.group(1)
        clean = _clean_field_name(field)
        if not clean:
            return match.group(0)
        resolved = _resolve_field_against_profile(clean, profile_col_names)
        return f"'{table_name}'[{resolved}]"

    result = re.sub(r'\[([^\]]+)\]', replace_field_ref, result)

    # Common function translations
    replacements = [
        (r'\bCOUNTD\s*\(', 'DISTINCTCOUNT('),
        (r'\bATTR\s*\(', ''),  # ATTR is Tableau-specific, drop it
        (r'\bZN\s*\(', ''),  # ZN -> just the value (COALESCE handled differently in DAX)
        (r'\bIFNULL\s*\(', 'COALESCE('),
        (r'\bDATEPART\s*\(', 'DATEPART('),  # Similar syntax
    ]
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result


def _guess_format(dax_agg):
    """Guess a DAX format string based on aggregation type."""
    if dax_agg in ("COUNT", "DISTINCTCOUNT"):
        return "#,0"
    if dax_agg in ("AVERAGE", "MEDIAN"):
        return "#,0.00"
    return "#,0"


# ------------------------------------------------------------------ #
#  Dashboard -> Page mapping                                           #
# ------------------------------------------------------------------ #

def _build_page_from_dashboard(dashboard, worksheets_by_name, profile_col_names,
                               col_types, table_name):
    """Build a PBI page (section) from a Tableau dashboard.

    Args:
        dashboard: dashboard dict from parser
        worksheets_by_name: dict mapping ws name -> ws dict
        profile_col_names: set of column names
        col_types: dict col_name -> semantic_type
        table_name: PBI table name

    Returns:
        dict: PBI section with displayName, width, height, visualContainers
    """
    canvas = dashboard.get("canvas", {})
    zones = dashboard.get("zones", [])

    # Scale zone positions to PBI canvas
    pbi_positions = {}
    if zones and canvas.get("width") and canvas.get("height"):
        scaled = translate_positions(zones, canvas)
        for p in scaled:
            pbi_positions[p["name"]] = p

    visual_containers = []

    for zone in zones:
        zone_name = zone.get("name", "")
        zone_type = zone.get("type", "")

        # Look up worksheet for this zone -- try multiple matching strategies
        ws = worksheets_by_name.get(zone_name)
        if not ws:
            # Try case-insensitive match
            for ws_name, ws_obj in worksheets_by_name.items():
                if ws_name.lower().strip() == zone_name.lower().strip():
                    ws = ws_obj
                    break
        if not ws:
            # Try partial/substring match (zone name contains ws name or vice versa)
            zone_lower = zone_name.lower().strip()
            for ws_name, ws_obj in worksheets_by_name.items():
                ws_lower = ws_name.lower().strip()
                if ws_lower in zone_lower or zone_lower in ws_lower:
                    ws = ws_obj
                    break
        if not ws:
            # Try matching by worksheets_used list order to zones order
            ws_used = dashboard.get("worksheets_used", [])
            zone_idx = zones.index(zone)
            if zone_idx < len(ws_used):
                ws = worksheets_by_name.get(ws_used[zone_idx])
        if not ws:
            # Zone is a title, blank, or layout container -- skip
            continue

        chart_type = translate_chart_type(ws.get("chart_type", "automatic"))
        data_roles = _classify_fields_for_chart(
            ws, chart_type, profile_col_names, col_types
        )

        # Position from scaled zones
        pos = pbi_positions.get(zone_name, {})
        x = pos.get("x", 0)
        y = pos.get("y", 0)
        w = pos.get("width", 300)
        h = pos.get("height", 200)

        config = {
            "visualType": chart_type,
            "title": ws.get("name", zone_name),
            "dataRoles": data_roles,
            "worksheet_name": ws.get("name", ""),
        }

        visual_containers.append({
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "config": config,
        })

    # Handle filters from the dashboard zones that are categorical
    # (create slicer visuals for them)
    filter_zones = []
    for zone in zones:
        zone_name = zone.get("name", "")
        zone_type = zone.get("type", "")
        if zone_type in ("filter", "quick-filter"):
            ws = worksheets_by_name.get(zone_name)
            if ws and ws.get("filters"):
                for filt in ws["filters"]:
                    field = _clean_field_name(filt.get("field", ""))
                    if field:
                        filter_zones.append(field)

    return {
        "displayName": dashboard.get("name", "Dashboard"),
        "width": 1280,
        "height": 720,
        "visualContainers": visual_containers,
    }


def _build_page_for_orphan_worksheets(worksheets, profile_col_names,
                                       col_types, table_name):
    """Build a PBI page for worksheets not assigned to any dashboard.

    Uses a deterministic grid layout.

    Returns:
        dict: PBI section, or None if no orphans.
    """
    if not worksheets:
        return None

    visual_containers = []
    n = len(worksheets)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols

    usable_w = 1280 - 24  # 12px margin each side
    usable_h = 720 - 24
    cell_w = (usable_w - (cols - 1) * 8) // cols
    cell_h = max(100, (usable_h - (rows - 1) * 8) // rows)

    for idx, ws in enumerate(worksheets):
        row = idx // cols
        col = idx % cols

        chart_type = translate_chart_type(ws.get("chart_type", "automatic"))
        data_roles = _classify_fields_for_chart(
            ws, chart_type, profile_col_names, col_types
        )

        config = {
            "visualType": chart_type,
            "title": ws.get("name", f"Visual {idx + 1}"),
            "dataRoles": data_roles,
            "worksheet_name": ws.get("name", ""),
        }

        visual_containers.append({
            "x": 12 + col * (cell_w + 8),
            "y": 12 + row * (cell_h + 8),
            "width": cell_w,
            "height": cell_h,
            "config": config,
        })

    return {
        "displayName": "Additional Worksheets",
        "width": 1280,
        "height": 720,
        "visualContainers": visual_containers,
    }


# ------------------------------------------------------------------ #
#  Main entry point                                                    #
# ------------------------------------------------------------------ #

def build_pbip_config_from_tableau(tableau_spec, data_profile, table_name="Data"):
    """Build a complete PBIP config directly from Tableau spec.

    No AI calls for visual structure. Deterministic mapping only.
    AI is only used for DAX translation of calculated fields (flagged
    in measures with needs_ai=True).

    Args:
        tableau_spec: dict from enhanced_tableau_parser.parse_twb()
        data_profile: dict from DataAnalyzer.analyze() or DeepAnalyzer.profile()
        table_name: str for the PBI table name

    Returns:
        tuple: (config, dashboard_spec) where:
            config = {"report_layout": {...}, "tmdl_model": {...}}
            dashboard_spec = {"dashboard_title": "...", "pages": [...],
                              "measures": [...], "design": {...}, ...}
    """
    # Build lookup structures
    worksheets_by_name = {}
    for ws in tableau_spec.get("worksheets", []):
        worksheets_by_name[ws["name"]] = ws

    profile_col_names = set()
    col_types = {}
    for col_info in data_profile.get("columns", []):
        profile_col_names.add(col_info["name"])
        col_types[col_info["name"]] = col_info.get("semantic_type", "dimension")

    # -- Build measures --
    measures = _build_measures_from_spec(
        tableau_spec, table_name, profile_col_names
    )

    # -- Build pages from dashboards --
    sections = []
    dashboards = tableau_spec.get("dashboards", [])
    used_ws_names = set()

    for db in dashboards:
        section = _build_page_from_dashboard(
            db, worksheets_by_name, profile_col_names, col_types, table_name
        )
        # Safety: if zone matching produced 0 visuals but the dashboard
        # has worksheets_used, create visuals from those worksheets directly
        if (not section.get("visualContainers")
                and db.get("worksheets_used")):
            ws_list = [
                worksheets_by_name[n]
                for n in db.get("worksheets_used", [])
                if n in worksheets_by_name
            ]
            if ws_list:
                fallback = _build_page_for_orphan_worksheets(
                    ws_list, profile_col_names, col_types, table_name
                )
                if fallback and fallback.get("visualContainers"):
                    section["visualContainers"] = fallback["visualContainers"]
                    print(f"    [DIRECT-MAPPER] Zone matching failed for "
                          f"'{db.get('name','')}' -- used worksheets_used fallback "
                          f"({len(section['visualContainers'])} visuals)")

        sections.append(section)
        used_ws_names.update(db.get("worksheets_used", []))

    # -- Handle orphan worksheets (not in any dashboard) --
    orphan_ws = [
        ws for ws in tableau_spec.get("worksheets", [])
        if ws["name"] not in used_ws_names
    ]
    if orphan_ws:
        orphan_section = _build_page_for_orphan_worksheets(
            orphan_ws, profile_col_names, col_types, table_name
        )
        if orphan_section:
            sections.append(orphan_section)

    # If no dashboards and no worksheets, create an empty page
    if not sections:
        sections.append({
            "displayName": "Dashboard",
            "width": 1280,
            "height": 720,
            "visualContainers": [],
        })

    # -- Build the config dict matching PBIPGenerator.generate() expectations --
    config = {
        "report_layout": {
            "sections": sections,
        },
        "tmdl_model": {
            "tables": [
                {
                    "name": table_name,
                    "measures": [
                        {
                            "name": m["name"],
                            "dax": m["dax"],
                            "format": m.get("format", "#,0"),
                        }
                        for m in measures
                    ],
                }
            ],
        },
    }

    # -- Build dashboard_spec for downstream (title, theme, design metadata) --
    dashboard_title = "Dashboard"
    if dashboards:
        dashboard_title = dashboards[0].get("name", "Dashboard")
    elif tableau_spec.get("worksheets"):
        dashboard_title = tableau_spec["worksheets"][0].get("name", "Dashboard")

    # Design metadata passthrough
    design = tableau_spec.get("design", {})
    colors = translate_colors(design)

    # Worksheet designs for visual formatting
    ws_designs = {}
    ws_chart_types = {}
    for ws in tableau_spec.get("worksheets", []):
        if ws.get("design"):
            ws_designs[ws["name"]] = ws["design"]
        if ws.get("chart_type"):
            ws_chart_types[ws["name"]] = ws["chart_type"]

    # Page info for the dashboard_spec
    pages = []
    for section in sections:
        visuals = []
        for vc in section.get("visualContainers", []):
            cfg = vc.get("config", {})
            if isinstance(cfg, str):
                try:
                    cfg = json.loads(cfg)
                except (json.JSONDecodeError, TypeError):
                    cfg = {}
            visuals.append({
                "type": cfg.get("visualType", "unknown"),
                "title": cfg.get("title", ""),
                "source_worksheet": cfg.get("worksheet_name", ""),
                "data_roles": cfg.get("dataRoles", {}),
            })
        pages.append({
            "name": section["displayName"],
            "visuals": visuals,
        })

    dashboard_spec = {
        "dashboard_title": dashboard_title,
        "source": "direct_tableau_mapper",
        "pages": pages,
        "measures": [
            {"name": m["name"], "dax": m["dax"], "format": m.get("format", "#,0")}
            for m in measures
        ],
        "design": design,
        "worksheet_designs": ws_designs,
        "worksheet_chart_types": ws_chart_types,
        "tableau_worksheets": tableau_spec.get("worksheets", []),
        "tableau_dashboards": dashboards,
        "tableau_calcs": tableau_spec.get("calculated_fields", []),
        "migration_warnings": [],
        "theme": {
            "background": colors.get("background", "#FFFFFF"),
            "foreground": colors.get("foreground", "#333333"),
            "accent": colors.get("tableAccent", "#118DFF"),
            "dataColors": colors.get("dataColors", []),
        },
    }

    return config, dashboard_spec
