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
# Matches prefix:FieldName or prefix:FieldName:suffix or prefix:FieldName:suffix:N
_SHELF_PREFIX_RE = re.compile(
    r'^(sum|avg|count|min|max|countd|median|attr|'
    r'none|usr|yr|mn|qr|tqr|tmn|twk|day|'
    r'mdy|wk|md|qy|my)\:(.+?)(?:\:[a-z]+(?:\:\d+)?)?$',
    re.IGNORECASE,
)

# Double-prefix pattern: e.g. pcto:sum:FieldName:suffix
_DOUBLE_PREFIX_RE = re.compile(
    r'^(pcto|pctd|pcti|pctf|ctd|ctds)\:'
    r'(sum|avg|count|min|max|countd|median|attr|none|usr)\:'
    r'(.+?)(?:\:[a-z]+(?:\:\d+)?)?$',
    re.IGNORECASE,
)

# Tableau meta-fields that should never appear as visual fields.
_TABLEAU_META_FIELDS = {
    "multiple values",
    "measure names",
    "measure values",
    "number of records",
    "latitude (generated)",
    "longitude (generated)",
}

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


def _is_datasource_name(name):
    """Check if a string looks like a Tableau datasource/table name.

    Datasource names appear in shelf fields as the first part of compound
    references like [Sample - Superstore].[sum:Sales:qk]. They contain
    no colons and are not actual data fields.

    Heuristic: datasource names are typically multi-word strings with
    spaces or dashes, or contain 'federated.' or ' - '.
    """
    if not name:
        return False
    # Definitely datasource IDs
    if name.startswith("federated."):
        return True
    # Common Tableau datasource name patterns:
    # "Sample - Superstore", "Sheet1 (my_file)", "Extract", etc.
    # A datasource name has NO colon and is not a simple single-word field.
    if ":" in name:
        return False
    # If it contains " - " it is almost certainly a datasource name
    if " - " in name:
        return True
    # If the name ends with a parenthesized portion like "Sheet1 (my_file)"
    # and the portion looks like a datasource qualifier (not "copy" or "generated")
    if "(" in name and name.endswith(")") and name[0].isupper():
        paren_content = name[name.rindex("(") + 1:-1].lower()
        if paren_content not in ("generated", "copy"):
            return True
    return False


def _clean_field_name(name):
    """Strip Tableau prefixes, brackets, quotes from a field reference.

    Handles patterns like:
      federated.xxx                      -> skip (datasource ID)
      Sample - Superstore                -> skip (datasource name)
      none:Region:nk                     -> Region
      sum:Sales:qk                       -> Sales
      usr:Calculation_123:qk:1           -> Calculation_123
      pcto:sum:Calculation_123:qk        -> Calculation_123
      "Region"                           -> Region
      [Region]                           -> Region
      Multiple Values                    -> skip (Tableau meta-field)
      Measure Names                      -> skip (Tableau meta-field)
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

    # Skip Tableau meta-fields
    if name.lower() in _TABLEAU_META_FIELDS:
        return ""

    # Skip datasource/table names (must check BEFORE prefix parsing)
    if _is_datasource_name(name):
        return ""

    # Try double-prefix pattern first: pcto:sum:Field:suffix
    m = _DOUBLE_PREFIX_RE.match(name)
    if m:
        name = m.group(3)
    else:
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

    # Final check: reject Tableau meta-fields after cleaning
    if name.lower() in _TABLEAU_META_FIELDS:
        return ""

    return name


def _clean_shelf_fields(field_refs):
    """Clean a list of Tableau shelf field references to plain names.

    Filters out datasource names, Tableau meta-fields, and duplicates.
    """
    result = []
    seen = set()
    for ref in field_refs:
        name = _clean_field_name(ref)
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _detect_aggregated_fields(field_refs):
    """Detect which field names appear with aggregation prefixes in shelf refs.

    Returns a set of clean field names that were used with SUM/AVG/COUNT/etc.
    This is important because the data profiler may classify a numeric column
    as 'dimension' (low cardinality), but Tableau uses it as a measure.
    """
    agg_prefixes = {"sum", "avg", "count", "countd", "min", "max", "median"}
    aggregated = set()
    for ref in field_refs:
        if not ref:
            continue
        ref = ref.strip()
        m = _SHELF_PREFIX_RE.match(ref)
        if m:
            prefix = m.group(1).lower()
            field = m.group(2)
            if prefix in agg_prefixes and field:
                aggregated.add(field)
    return aggregated


def _detect_aggregated_color(color_ref):
    """Detect if color_field reference has an aggregation prefix.

    Handles compound refs like [federated.xxx].[sum:Sales:qk].
    Returns the clean field name if aggregated, else empty string.
    """
    if not color_ref:
        return ""
    # Handle compound pattern: [table].[prefix:field:suffix]
    inner = color_ref
    if "].[" in inner:
        parts = inner.split("].[", 1)
        if len(parts) == 2:
            inner = parts[1].rstrip("]")
    elif inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    m = _SHELF_PREFIX_RE.match(inner)
    if m:
        prefix = m.group(1).lower()
        field = m.group(2)
        agg_prefixes = {"sum", "avg", "count", "countd", "min", "max", "median"}
        if prefix in agg_prefixes and field:
            return field
    return ""


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

def _is_generated_field(name):
    """Check if a field is a Tableau-generated field (e.g. Latitude/Longitude (generated))."""
    return bool(name and "(generated)" in name.lower())


def _infer_fields_from_worksheet_name(ws_name, profile_col_names, col_types):
    """Infer likely category and value fields from the worksheet name.

    Parses worksheet titles like 'sales by region and sub regions' to find
    matching profile columns.

    Returns:
        (category_fields, value_fields) -- lists of column names
    """
    ws_lower = ws_name.lower()
    dims = [c for c in profile_col_names if col_types.get(c) != "measure"]
    measures = [c for c in profile_col_names if col_types.get(c) == "measure"]

    matched_dims = []
    matched_measures = []

    # Check which profile columns appear (case-insensitive) in the worksheet name
    for col in dims:
        col_words = col.lower().replace("-", " ").replace("_", " ")
        if col_words in ws_lower or col.lower() in ws_lower:
            matched_dims.append(col)

    for col in measures:
        col_words = col.lower().replace("-", " ").replace("_", " ")
        if col_words in ws_lower or col.lower() in ws_lower:
            matched_measures.append(col)

    return matched_dims, matched_measures


def _classify_fields_for_chart(ws, chart_type, profile_col_names, col_types):
    """Determine category, values, and series fields for a worksheet.

    Uses the worksheet shelf fields and chart type to assign PBI roles.
    When shelf fields are empty or contain only Tableau-generated fields,
    infers fields from the worksheet name and available profile columns.

    Args:
        ws: worksheet dict from the parser
        chart_type: PBI visual type string
        profile_col_names: set of column names from data profile
        col_types: dict of col_name -> semantic_type (dimension/measure)

    Returns:
        dict with category, values, series lists
    """
    raw_rows = ws.get("rows_fields", [])
    raw_cols = ws.get("cols_fields", [])
    raw_color = ws.get("color_field", "")

    rows = _clean_shelf_fields(raw_rows)
    cols = _clean_shelf_fields(raw_cols)
    color_field = _clean_field_name(raw_color)

    # Detect fields that Tableau treats as measures (aggregation prefix)
    # even if the profiler classifies them as dimensions (low cardinality).
    tableau_agg_fields = _detect_aggregated_fields(raw_rows) | _detect_aggregated_fields(raw_cols)
    agg_color = _detect_aggregated_color(raw_color)
    if agg_color:
        tableau_agg_fields.add(agg_color)

    # Filter out Tableau-generated fields (Latitude/Longitude (generated))
    # These don't exist in the actual data.
    rows = [f for f in rows if not _is_generated_field(f)]
    cols = [f for f in cols if not _is_generated_field(f)]

    # Resolve all fields against profile
    rows = [_resolve_field_against_profile(f, profile_col_names) for f in rows]
    cols = [_resolve_field_against_profile(f, profile_col_names) for f in cols]
    if color_field:
        # Clean color field: handle compound refs like [federated.xxx].[sum:Sales:qk]
        if _is_generated_field(color_field):
            color_field = ""
        else:
            color_field = _resolve_field_against_profile(color_field, profile_col_names)

    # Classify each field as dimension or measure based on:
    # 1. Profile semantic type (most reliable)
    # 2. Tableau aggregation prefix (if the profiler missed it)
    def is_measure(f):
        st = col_types.get(f, "")
        if st == "measure":
            return True
        # If Tableau used this field with SUM/AVG/COUNT etc., treat as measure
        # even if the profiler says dimension (common for low-cardinality numerics)
        if f in tableau_agg_fields:
            return True
        # Also check case-insensitive match against tableau agg fields
        f_lower = f.lower()
        for taf in tableau_agg_fields:
            if taf.lower() == f_lower:
                return True
        return False

    # Separate into dimensions and measures from both shelves
    all_dims = []
    all_measures = []
    for f in rows + cols:
        if is_measure(f):
            all_measures.append(f)
        else:
            all_dims.append(f)

    # If color_field is a measure and not already captured, add it to measures
    if color_field and color_field not in all_measures and is_measure(color_field):
        all_measures.append(color_field)

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

    # ------------------------------------------------------------------ #
    # ENRICH: Cross-reference worksheet name with profile columns to add  #
    # worksheet-specific fields that the parser missed. Even when shelves #
    # have some fields, the worksheet title often reveals the intended     #
    # category dimension (e.g. "sales by customer type" -> Customer Type). #
    # ------------------------------------------------------------------ #
    ws_name = ws.get("name", "")
    if profile_col_names:
        inferred_dims, inferred_measures = _infer_fields_from_worksheet_name(
            ws_name, profile_col_names, col_types
        )
        if not all_dims and not all_measures:
            # Fully empty -- use all inferred fields
            if inferred_dims or inferred_measures:
                all_dims = inferred_dims
                all_measures = inferred_measures
                print(f"    [DIRECT-MAPPER] Inferred fields from worksheet name "
                      f"'{ws_name}': dims={all_dims}, measures={all_measures}")
        else:
            # Have some fields from shelves. Add name-inferred dims that are
            # missing (e.g. "Customer Type" from "sales by customer type")
            # but don't replace what the parser found.
            existing = set(f.lower() for f in all_dims + all_measures)
            added = False
            for d in inferred_dims:
                if d.lower() not in existing:
                    all_dims.append(d)
                    added = True
            for m in inferred_measures:
                if m.lower() not in existing:
                    all_measures.append(m)
                    added = True
            # If the worksheet name has "by <column>", promote that column
            # to the front of dims -- it's the intended category axis.
            ws_lower = ws_name.lower()
            if " by " in ws_lower and inferred_dims:
                by_part = ws_lower.split(" by ", 1)[1].strip()
                for d in inferred_dims:
                    if d.lower() in by_part:
                        # Move this dim to front
                        if d in all_dims:
                            all_dims.remove(d)
                            all_dims.insert(0, d)
                        break
            if added:
                print(f"    [DIRECT-MAPPER] Enriched fields from worksheet name "
                      f"'{ws_name}': dims={all_dims}, measures={all_measures}")

    # Second fallback: if STILL no fields, pick the first dimension and
    # first measure from the profile. Every visual should show *something*.
    if not all_dims and not all_measures and profile_col_names:
        profile_dims = [c for c in profile_col_names
                        if col_types.get(c) != "measure"
                        and col_types.get(c) != "date"]
        profile_measures = [c for c in profile_col_names
                            if col_types.get(c) == "measure"]
        if profile_dims:
            all_dims = [sorted(profile_dims)[0]]
        if profile_measures:
            all_measures = [sorted(profile_measures)[0]]
        if all_dims or all_measures:
            print(f"    [DIRECT-MAPPER] Auto-assigned fallback fields for "
                  f"'{ws_name}': dims={all_dims}, measures={all_measures}")

    # For map visuals and table fallbacks, pick geographic-looking columns
    # as dimensions so the visual shows location data
    if chart_type in ("tableEx", "filledMap", "map", "shapeMap") and not all_dims and profile_col_names:
        geo_keywords = ["state", "region", "country", "city", "province",
                        "county", "zip", "postal", "location", "area", "territory",
                        "geography", "district", "nation"]
        geo_dims = []
        for col in sorted(profile_col_names):
            if col_types.get(col) != "measure":
                col_lower = col.lower()
                if any(kw in col_lower for kw in geo_keywords):
                    geo_dims.append(col)
        if geo_dims:
            all_dims = geo_dims[:2]
            print(f"    [DIRECT-MAPPER] Auto-assigned geographic field(s) "
                  f"{all_dims} for '{chart_type}' visual '{ws_name}'")

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

    elif chart_type in ("map", "filledMap", "shapeMap", "azureMap"):
        # Map visuals: geo dimension -> category (location), measures -> values
        category = all_dims[:1]
        values = all_measures[:3] or all_dims[1:2]

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

    # Series: color field if it's a dimension and not already used.
    # Also promote unused dimensions to series for charts that only take 1
    # category (bar, line, etc.) -- this gives "sales by customer type"
    # the Customer Type color grouping even when Tableau color shelf is empty.
    used = set(category + values)
    if color_field and color_field not in used and not is_measure(color_field):
        series = [color_field]
    elif (not series and len(all_dims) > 1
          and chart_type not in ("card", "cardVisual", "multiRowCard", "kpi",
                                 "slicer", "filledMap", "map", "shapeMap")):
        unused_dims = [d for d in all_dims if d not in used]
        if unused_dims:
            series = unused_dims[:1]

    # Defensive fallback: charts (not card/table) with category but no values
    # get the first available measure from the profile. Charts with values but
    # no category get the first available dimension. Prevents blank visuals
    # when the parser only captured partial shelf fields.
    _non_axis_types = {"card", "cardVisual", "multiRowCard", "kpi",
                       "tableEx", "pivotTable", "table", "matrix", "slicer"}
    if chart_type not in _non_axis_types and profile_col_names:
        if category and not values:
            profile_measures = [c for c in profile_col_names
                                if col_types.get(c) == "measure"
                                and c not in set(category)]
            if profile_measures:
                values = [sorted(profile_measures)[0]]
                print(f"    [DIRECT-MAPPER] Fallback: assigned measure "
                      f"{values} for '{ws_name}' (had category, no values)")
        elif values and not category:
            profile_dims = [c for c in profile_col_names
                            if col_types.get(c) not in ("measure", "date")
                            and c not in set(values)]
            if profile_dims:
                category = [sorted(profile_dims)[0]]
                print(f"    [DIRECT-MAPPER] Fallback: assigned dimension "
                      f"{category} for '{ws_name}' (had values, no category)")

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
            # Complex formula -- use BLANK() placeholder to avoid invalid
            # TMDL syntax (Tableau IF/THEN/ELSE/END is not valid DAX).
            # The original formula is preserved for AI translation later.
            measures.append({
                "name": cf_name,
                "dax": "BLANK()",
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


def _translate_spatial_formula(formula, table_name):
    """Translate Tableau spatial functions to Power BI equivalents.

    Returns (dax_expression, is_map_visual) or (None, False) if not spatial.
    """
    f_upper = formula.strip().upper()

    # MAKEPOINT(lat, lon) -> map visual with lat/lon columns
    mp = re.match(r'MAKEPOINT\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)', formula, re.IGNORECASE)
    if mp:
        lat_field = _clean_field_name(mp.group(1).strip().strip('[]'))
        lon_field = _clean_field_name(mp.group(2).strip().strip('[]'))
        dax = f"/* MAP_POINT: lat='{table_name}'[{lat_field}] lon='{table_name}'[{lon_field}] */"
        return dax, True

    # MAKELINE(point1, point2) -> map visual with route layer
    ml = re.match(r'MAKELINE\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)', formula, re.IGNORECASE)
    if ml:
        dax = (f"/* MAP_LINE: connect {ml.group(1).strip()} to {ml.group(2).strip()} "
               f"- use Azure Maps route layer */")
        return dax, True

    # DISTANCE(p1, p2, units) -> Haversine DAX
    dist = re.match(
        r'DISTANCE\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*["\']?(km|miles?|mi)["\']?\s*\)',
        formula, re.IGNORECASE,
    )
    if dist:
        units = dist.group(3).lower()
        R = "6371" if "km" in units else "3959"
        dax = (
            f"VAR _R = {R} "
            f"RETURN _R * ACOS(MAX(-1, MIN(1, "
            f"SIN(RADIANS(SELECTEDVALUE('{table_name}'[_lat1]))) * "
            f"SIN(RADIANS(SELECTEDVALUE('{table_name}'[_lat2]))) + "
            f"COS(RADIANS(SELECTEDVALUE('{table_name}'[_lat1]))) * "
            f"COS(RADIANS(SELECTEDVALUE('{table_name}'[_lat2]))) * "
            f"COS(RADIANS(SELECTEDVALUE('{table_name}'[_lon2]) - "
            f"SELECTEDVALUE('{table_name}'[_lon1]))))))"
        )
        return dax, False

    # BUFFER, AREA, INTERSECTS, COLLECT, GEOMETRY -> manual with comment
    for spatial_fn in ('BUFFER', 'AREA', 'INTERSECTS', 'COLLECT', 'GEOMETRY'):
        if re.match(rf'{spatial_fn}\s*\(', f_upper):
            return (f"/* SPATIAL_{spatial_fn}: no Power BI DAX equivalent "
                    f"-- requires custom visual */"), False

    return None, False


def _translate_complex_measures(measures, table_name, profile_col_names):
    """Translate complex Tableau formulas to DAX via heuristic + AI fallback.

    0. Run spatial translation first (MAKEPOINT, DISTANCE, etc.)
    1. Run heuristic translation on each needs_ai measure
    2. If heuristic produces a non-BLANK result, use it
    3. Remaining measures get sent to Claude API in a single batch
    4. Falls back to BLANK() if anything fails (never crashes)

    Mutates the measures list in-place and returns it.
    """
    needs_ai = [m for m in measures if m.get("needs_ai") and m.get("original_formula")]
    if not needs_ai:
        return measures

    # Phase 0: spatial pre-filter
    still_needs_translation = []
    for m in needs_ai:
        spatial_dax, is_map = _translate_spatial_formula(m["original_formula"], table_name)
        if spatial_dax is not None:
            m["dax"] = spatial_dax
            m["needs_ai"] = False
            m["translation_method"] = "spatial"
            if is_map:
                m["is_map_visual"] = True
            print(f"    [DAX-SPATIAL] {m['name']}: spatial translation applied")
        else:
            still_needs_translation.append(m)

    needs_ai = still_needs_translation

    # Phase 1: heuristic pre-filter
    still_needs_ai = []
    for m in needs_ai:
        heuristic_dax = _translate_formula_heuristic(
            m["original_formula"], table_name, profile_col_names
        )
        # If heuristic returned something different from the raw formula
        # and it doesn't look like untranslated Tableau syntax
        if (heuristic_dax
                and heuristic_dax != m["original_formula"]
                and "THEN" not in heuristic_dax
                and "ELSEIF" not in heuristic_dax
                and "END" not in heuristic_dax.split("(")[0]):
            m["dax"] = heuristic_dax
            m["needs_ai"] = False
            m["translation_method"] = "heuristic"
            print(f"    [DAX-HEURISTIC] {m['name']}: heuristic translation applied")
        else:
            still_needs_ai.append(m)

    if not still_needs_ai:
        return measures

    # Phase 2: AI translation via Claude (process all measures in batches of 20)
    print(f"    [DAX-AI] Sending {len(still_needs_ai)} complex measures to Claude for translation")
    try:
        import os
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("    [DAX-AI] No ANTHROPIC_API_KEY -- skipping AI translation")
            return measures

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        system_prompt = (
            "You are a DAX expert. Translate Tableau calculated field formulas to DAX.\n"
            "Return ONLY a JSON array: [{\"name\": \"field_name\", \"dax\": \"DAX_EXPRESSION\"}].\n"
            "Rules:\n"
            "- {FIXED [dim] : AGG([field])} -> CALCULATE(AGG(Table[field]), ALLEXCEPT(Table, Table[dim]))\n"
            "- {FIXED : AGG([field])} -> CALCULATE(AGG(Table[field]), ALL(Table))\n"
            "- IF condition THEN x ELSE y END -> IF(condition, x, y)\n"
            "- IIF(cond, x, y) -> IF(cond, x, y)\n"
            "- CASE WHEN -> SWITCH(TRUE(), ...)\n"
            "- ZN(expr) -> IF(ISBLANK(expr), 0, expr)\n"
            "- ISNULL(expr) -> ISBLANK(expr)\n"
            "- IFNULL(expr, val) -> IF(ISBLANK(expr), val, expr)\n"
            "- COUNTD -> DISTINCTCOUNT\n"
            "- STR(x) -> FORMAT(x, \"0\")\n"
            "- CONTAINS(str, sub) -> NOT(ISERROR(SEARCH(sub, str)))\n"
            "- DATEDIFF('unit', start, end) -> DATEDIFF(start, end, UNIT)\n"
            "- DATEPART('unit', date) -> YEAR/MONTH/DAY/QUARTER(date)\n"
            "- RUNNING_SUM -> use CALCULATE with window functions\n"
            "- RANK -> RANKX(ALL(Table), expression)\n"
            f"- Use table name '{table_name}' for all column references.\n"
            "- [FieldName] becomes 'TableName'[FieldName].\n"
            "- If you cannot translate, return BLANK() for that field.\n"
            "- Return ONLY the JSON array, no explanation."
        )

        total_applied = 0
        batch_size = 20
        for batch_start in range(0, len(still_needs_ai), batch_size):
            batch = still_needs_ai[batch_start:batch_start + batch_size]
            if batch_start > 0:
                print(f"    [DAX-AI] Batch {batch_start // batch_size + 1}: "
                      f"sending {len(batch)} more measures")

            payload = [
                {
                    "name": m["name"],
                    "formula": m["original_formula"],
                    "table_name": table_name,
                }
                for m in batch
            ]

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )

            response_text = response.content[0].text.strip()

            # Parse JSON from response (handle markdown code blocks)
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = [l for l in lines if not l.startswith("```")]
                response_text = "\n".join(lines)

            ai_results = json.loads(response_text)
            if not isinstance(ai_results, list):
                ai_results = []

            # Apply AI translations
            ai_lookup = {r["name"]: r["dax"] for r in ai_results if r.get("name") and r.get("dax")}
            applied = 0
            for m in batch:
                ai_dax = ai_lookup.get(m["name"], "")
                if ai_dax and ai_dax.strip() and ai_dax.strip() != "BLANK()":
                    m["dax"] = ai_dax
                    m["needs_ai"] = False
                    m["translation_method"] = "ai_claude"
                    applied += 1
                    print(f"    [DAX-AI] {m['name']}: AI translation applied")
                else:
                    m["translation_method"] = "ai_failed"
                    print(f"    [DAX-AI] {m['name']}: AI returned BLANK or no result, keeping BLANK()")

            total_applied += applied

        print(f"    [DAX-AI] Applied {total_applied}/{len(still_needs_ai)} AI translations")

    except ImportError:
        print("    [DAX-AI] anthropic package not installed -- skipping AI translation")
    except json.JSONDecodeError as e:
        print(f"    [DAX-AI] Failed to parse AI response as JSON: {e}")
    except Exception as e:
        print(f"    [DAX-AI] AI translation failed (non-fatal): {e}")

    return measures


# ------------------------------------------------------------------ #
#  Dashboard -> Page mapping                                           #
# ------------------------------------------------------------------ #

def _build_page_from_dashboard(dashboard, worksheets_by_name, profile_col_names,
                               col_types, table_name):
    """Build a PBI page (section) from a Tableau dashboard.

    Uses worksheets_used as the authoritative list of which worksheets belong
    on this dashboard page. Zones are only used for spatial layout (position).
    Each worksheet appears at most once per page (deduplication).

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
    ws_used = dashboard.get("worksheets_used", [])

    # Build zone position lookup (zone name -> scaled PBI position)
    pbi_positions = {}
    if zones and canvas.get("width") and canvas.get("height"):
        scaled = translate_positions(zones, canvas)
        for p in scaled:
            pbi_positions[p["name"]] = p

    # Build case-insensitive zone name lookup for position matching
    zone_positions_lower = {}
    for zname, pos in pbi_positions.items():
        zone_positions_lower[zname.lower().strip()] = pos

    visual_containers = []
    added_ws_names = set()  # Track worksheets already added to this page

    # Use worksheets_used as the authoritative source of which worksheets
    # belong on this page. This avoids the dangerous substring/index matching
    # that was causing duplicate and incorrect visual assignments.
    for ws_name in ws_used:
        # Skip if already added (deduplication)
        if ws_name in added_ws_names:
            continue

        # Look up the worksheet
        ws = worksheets_by_name.get(ws_name)
        if not ws:
            # Try case-insensitive match
            for k, v in worksheets_by_name.items():
                if k.lower().strip() == ws_name.lower().strip():
                    ws = v
                    break
        if not ws:
            # Worksheet not found in spec -- skip
            continue

        added_ws_names.add(ws_name)

        chart_type = translate_chart_type(ws.get("chart_type", "automatic"))

        # Skip UI decorations (zoom buttons, shape icons with no data)
        if chart_type == "skip":
            continue

        data_roles = _classify_fields_for_chart(
            ws, chart_type, profile_col_names, col_types
        )

        # Position: look up from zones by exact name, then case-insensitive
        pos = pbi_positions.get(ws_name)
        if not pos:
            pos = zone_positions_lower.get(ws_name.lower().strip())
        if not pos:
            pos = {}

        x = pos.get("x", 0)
        y = pos.get("y", 0)
        w = pos.get("width", 300)
        h = pos.get("height", 200)

        config = {
            "visualType": chart_type,
            "title": ws.get("name", ws_name),
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

        # Skip UI decorations (zoom buttons, shape icons with no data)
        if chart_type == "skip":
            continue

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
    data_profile = data_profile or {}
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

    # -- Translate complex measures (heuristic + AI) --
    measures = _translate_complex_measures(measures, table_name, profile_col_names)

    # -- Post-process: syntax validation + column name reconciliation --
    try:
        from core.dax_postprocess import postprocess_measures
        pp_stats = postprocess_measures(measures, table_name, profile_col_names)
        if pp_stats["syntax_fixed"] or pp_stats["columns_fixed"]:
            print(f"    [DAX-POSTPROCESS] {pp_stats['syntax_fixed']} syntax fixes, "
                  f"{pp_stats['columns_fixed']} column name fixes "
                  f"({pp_stats['total_processed']} measures processed)")
    except Exception as _pp_err:
        print(f"    [DAX-POSTPROCESS] Skipped (non-fatal): {_pp_err}")

    # -- Build pages: TAB-DRIVEN (from <windows>) or DASHBOARD-DRIVEN (legacy) --
    windows = tableau_spec.get("windows", [])
    visible_tabs = [w for w in windows if not w.get("hidden", False)]
    dashboards = tableau_spec.get("dashboards", [])
    dashboards_by_name = {d["name"]: d for d in dashboards}

    sections = []

    if visible_tabs:
        # TAB-DRIVEN: one PBI page per visible Tableau tab
        print(f"    [MAPPER] Tab-driven mode: {len(visible_tabs)} visible tabs")
        for tab in visible_tabs:
            tab_name = tab["name"]
            tab_type = tab["type"]

            if tab_type == "dashboard":
                db = dashboards_by_name.get(tab_name)
                if db:
                    section = _build_page_from_dashboard(
                        db, worksheets_by_name, profile_col_names,
                        col_types, table_name
                    )
                    if (not section.get("visualContainers")
                            and db.get("worksheets_used")):
                        ws_list = [
                            worksheets_by_name[n]
                            for n in db.get("worksheets_used", [])
                            if n in worksheets_by_name
                        ]
                        if ws_list:
                            fb = _build_page_for_orphan_worksheets(
                                ws_list, profile_col_names, col_types, table_name
                            )
                            if fb and fb.get("visualContainers"):
                                section["visualContainers"] = fb["visualContainers"]
                    sections.append(section)
                    print(f"    [MAPPER] Tab '{tab_name}' -> dashboard page "
                          f"({len(section.get('visualContainers', []))} visuals)")

            elif tab_type == "worksheet":
                ws = worksheets_by_name.get(tab_name)
                if ws:
                    chart_type = translate_chart_type(
                        ws.get("chart_type", "automatic")
                    )
                    data_roles = _classify_fields_for_chart(
                        ws, chart_type, profile_col_names, col_types
                    )
                    section = {
                        "displayName": tab_name,
                        "width": 1280,
                        "height": 720,
                        "visualContainers": [{
                            "x": 12, "y": 12, "width": 1256, "height": 696,
                            "config": {
                                "visualType": chart_type,
                                "title": tab_name,
                                "dataRoles": data_roles,
                                "worksheet_name": tab_name,
                            },
                        }],
                    }
                    sections.append(section)
                    print(f"    [MAPPER] Tab '{tab_name}' -> worksheet page "
                          f"(type={chart_type})")

            elif tab_type == "story":
                stories = tableau_spec.get("stories", [])
                story = next(
                    (s for s in stories if s["name"] == tab_name), None
                )
                if story and story.get("storypoints"):
                    for sp in story["storypoints"]:
                        sp_sheet = sp.get("sheet", "")
                        sp_caption = sp.get("caption", sp_sheet or tab_name)
                        if sp_sheet in dashboards_by_name:
                            db = dashboards_by_name[sp_sheet]
                            sec = _build_page_from_dashboard(
                                db, worksheets_by_name, profile_col_names,
                                col_types, table_name
                            )
                            sec["displayName"] = sp_caption
                            sections.append(sec)
                        elif sp_sheet in worksheets_by_name:
                            ws = worksheets_by_name[sp_sheet]
                            ct = translate_chart_type(
                                ws.get("chart_type", "automatic")
                            )
                            dr = _classify_fields_for_chart(
                                ws, ct, profile_col_names, col_types
                            )
                            sections.append({
                                "displayName": sp_caption,
                                "width": 1280, "height": 720,
                                "visualContainers": [{
                                    "x": 12, "y": 12,
                                    "width": 1256, "height": 696,
                                    "config": {
                                        "visualType": ct, "title": sp_caption,
                                        "dataRoles": dr,
                                        "worksheet_name": sp_sheet,
                                    },
                                }],
                            })
                    print(f"    [MAPPER] Tab '{tab_name}' -> story "
                          f"({len(story['storypoints'])} pages)")

    else:
        # LEGACY: dashboard-driven (no <windows> section)
        print(f"    [MAPPER] Legacy mode: no <windows> section")
        used_ws_names = set()
        for db in dashboards:
            section = _build_page_from_dashboard(
                db, worksheets_by_name, profile_col_names, col_types, table_name
            )
            if (not section.get("visualContainers")
                    and db.get("worksheets_used")):
                ws_list = [
                    worksheets_by_name[n]
                    for n in db.get("worksheets_used", [])
                    if n in worksheets_by_name
                ]
                if ws_list:
                    fb = _build_page_for_orphan_worksheets(
                        ws_list, profile_col_names, col_types, table_name
                    )
                    if fb and fb.get("visualContainers"):
                        section["visualContainers"] = fb["visualContainers"]
            sections.append(section)
            used_ws_names.update(db.get("worksheets_used", []))

        orphan_ws = [
            ws for ws in tableau_spec.get("worksheets", [])
            if ws["name"] not in used_ws_names
        ]
        if orphan_ws:
            orphan_section = _build_page_for_orphan_worksheets(
                orphan_ws, profile_col_names, col_types, table_name
            )
            if orphan_section:
                if not dashboards:
                    if len(orphan_ws) == 1:
                        orphan_section["displayName"] = orphan_ws[0].get(
                            "name", "Sheet"
                        )
                    else:
                        orphan_section["displayName"] = "Dashboard"
                sections.append(orphan_section)

    # If no dashboards and no worksheets, create a placeholder page
    # with at least one visual so the PBI file is never empty.
    if not sections:
        sections.append({
            "displayName": "Dashboard",
            "width": 1280,
            "height": 720,
            "visualContainers": [{
                "x": 12, "y": 12, "width": 1256, "height": 696,
                "config": {
                    "visualType": "tableEx",
                    "title": "Data",
                    "dataRoles": {"Values": list(profile_col_names)[:10] if profile_col_names else []},
                    "worksheet_name": "",
                },
            }],
        })
        print("    [MAPPER] No dashboards or worksheets -- created placeholder table visual")

    # Safety: strip out any sections with 0 visuals and rebuild them as tables
    for section in sections:
        if not section.get("visualContainers"):
            section["visualContainers"] = [{
                "x": 12, "y": 12, "width": 1256, "height": 696,
                "config": {
                    "visualType": "tableEx",
                    "title": section.get("displayName", "Data"),
                    "dataRoles": {"Values": list(profile_col_names)[:10] if profile_col_names else []},
                    "worksheet_name": "",
                },
            }]
            print(f"    [MAPPER] Page '{section.get('displayName', '?')}' had 0 visuals"
                  f" -- injected table visual as fallback")

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
        "measures_full": measures,
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
