"""
Visual Equivalency Engine -- maps Tableau visuals to Power BI equivalents.

Given a normalized VisualSpec (from tableau_extractor), returns the best
Power BI visual match with explicit gap documentation. Rule-based, no LLM.

Does NOT force pixel-perfect equivalence. Instead chooses the closest valid
PBI construct and records the gap.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


# ================================================================== #
#  Output types                                                         #
# ================================================================== #

class EquivalencyStatus(str, Enum):
    EXACT = "exact"
    CLOSE_EQUIVALENT = "close_equivalent"
    APPROXIMATION_REQUIRED = "approximation_required"
    UNSUPPORTED = "unsupported"


@dataclass
class FieldRole:
    """A data field role required by the PBI visual."""
    role: str           # Category, Values, Series, Rows, Columns, Legend, etc.
    field: str = ""     # Recommended field name (from Tableau spec)
    aggregation: str = ""
    required: bool = True
    notes: str = ""


@dataclass
class VisualEquivalency:
    """Complete equivalency result for one visual."""
    source_name: str
    source_mark_type: str

    chosen_powerbi_visual: str
    equivalency_status: EquivalencyStatus
    rationale: str

    required_field_roles: List[FieldRole] = field(default_factory=list)
    formatting_notes: List[str] = field(default_factory=list)
    interaction_notes: List[str] = field(default_factory=list)
    analyst_review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)


# ================================================================== #
#  Mapping rules                                                        #
# ================================================================== #

# Core mark-type rules. Each entry:
#   mark_type -> (pbi_visual, status, rationale)
# These are the baseline; refinement logic below may override.
_BASELINE_MAP = {
    "bar": (
        "clusteredBarChart",
        EquivalencyStatus.EXACT,
        "Tableau horizontal bar maps directly to PBI clustered bar chart."
    ),
    "stacked bar": (
        "stackedBarChart",
        EquivalencyStatus.EXACT,
        "Tableau stacked bar maps directly to PBI stacked bar chart."
    ),
    "line": (
        "lineChart",
        EquivalencyStatus.EXACT,
        "Tableau line mark maps directly to PBI line chart."
    ),
    "area": (
        "areaChart",
        EquivalencyStatus.EXACT,
        "Tableau area mark maps directly to PBI area chart."
    ),
    "circle": (
        "scatterChart",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau circle marks become PBI scatter chart. Circle sizing and "
        "jittering behave differently -- scatter uses X/Y positioning."
    ),
    "square": (
        "treemap",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau square marks become PBI treemap. Treemap uses hierarchical "
        "sizing rather than grid positioning."
    ),
    "text": (
        "tableEx",
        EquivalencyStatus.EXACT,
        "Tableau text table maps to PBI table visual."
    ),
    "polygon": (
        "map",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau polygon mark (filled map) maps to PBI map visual. "
        "PBI uses Bing Maps; custom polygons are not supported natively."
    ),
    "map": (
        "map",
        EquivalencyStatus.EXACT,
        "Tableau map mark maps to PBI map visual."
    ),
    "pie": (
        "pieChart",
        EquivalencyStatus.EXACT,
        "Tableau pie mark maps directly to PBI pie chart."
    ),
    "gantt": (
        "clusteredBarChart",
        EquivalencyStatus.APPROXIMATION_REQUIRED,
        "PBI has no native Gantt chart. A clustered bar chart with "
        "start/duration encoding approximates it. Consider the Gantt "
        "custom visual from AppSource for better fidelity."
    ),
    "shape": (
        "scatterChart",
        EquivalencyStatus.APPROXIMATION_REQUIRED,
        "Tableau shape marks have no direct PBI equivalent. Scatter chart "
        "is the closest native option. Custom shape encoding is not supported."
    ),
    "density": (
        "scatterChart",
        EquivalencyStatus.APPROXIMATION_REQUIRED,
        "Tableau density marks (heatmap scatter) become PBI scatter chart. "
        "True kernel density rendering is not available in PBI."
    ),
    "automatic": (
        "clusteredBarChart",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau 'Automatic' mark type. PBI equivalent depends on data shape. "
        "Defaulting to clustered bar; review may be needed."
    ),
}

# Visual types that PBI supports but Tableau represents differently
_SPECIAL_PATTERNS = {
    "combo": (
        "lineClusteredColumnComboChart",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau dual-axis / combo maps to PBI line+clustered column combo chart. "
        "PBI combo does not support independent Y-axis scaling."
    ),
    "donut": (
        "donutChart",
        EquivalencyStatus.EXACT,
        "Tableau donut maps directly to PBI donut chart."
    ),
    "waterfall": (
        "waterfallChart",
        EquivalencyStatus.EXACT,
        "Tableau waterfall maps directly to PBI waterfall chart."
    ),
    "funnel": (
        "funnelChart",
        EquivalencyStatus.EXACT,
        "Tableau funnel maps directly to PBI funnel chart."
    ),
    "histogram": (
        "clusteredBarChart",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "PBI has no native histogram. A clustered bar chart with binned "
        "dimension approximates it. Create a calculated column for bins."
    ),
    "box-plot": (
        "clusteredBarChart",
        EquivalencyStatus.APPROXIMATION_REQUIRED,
        "PBI has no native box plot. Consider the Box and Whisker custom "
        "visual from AppSource. A clustered bar with error bars is a "
        "distant approximation."
    ),
    "bullet": (
        "clusteredBarChart",
        EquivalencyStatus.APPROXIMATION_REQUIRED,
        "PBI has no native bullet chart. Consider the Bullet Chart custom "
        "visual from AppSource. A bar chart with reference lines approximates it."
    ),
    "highlight-table": (
        "matrix",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau highlight table maps to PBI matrix with conditional formatting. "
        "Apply background color rules to value cells for equivalent effect."
    ),
    "heatmap": (
        "matrix",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau heatmap maps to PBI matrix with conditional formatting. "
        "Color saturation on cell backgrounds provides similar effect."
    ),
    "crosstab": (
        "matrix",
        EquivalencyStatus.EXACT,
        "Tableau crosstab/pivot maps directly to PBI matrix visual."
    ),
    "dual-axis": (
        "lineClusteredColumnComboChart",
        EquivalencyStatus.CLOSE_EQUIVALENT,
        "Tableau dual-axis maps to PBI combo chart. Independent axis "
        "scaling is not supported in PBI -- both measures share the "
        "same Y-axis range."
    ),
}


# ================================================================== #
#  Semantic pattern detectors                                           #
# ================================================================== #

def _detect_column_vs_bar(vs) -> Optional[str]:
    """Distinguish horizontal bar from vertical column chart.

    Tableau convention:
    - Dimension on rows + measure on cols = horizontal bar
    - Dimension on cols + measure on rows = vertical column
    """
    rows = vs.encoding.shelf_rows if hasattr(vs, "encoding") else []
    cols = vs.encoding.shelf_cols if hasattr(vs, "encoding") else []

    has_dim_on_rows = any(c.role == "dimension" for c in rows)
    has_meas_on_cols = any(c.role == "measure" for c in cols)
    has_dim_on_cols = any(c.role == "dimension" for c in cols)
    has_meas_on_rows = any(c.role == "measure" for c in rows)

    if has_dim_on_cols and has_meas_on_rows:
        return "clusteredColumnChart"
    if has_dim_on_rows and has_meas_on_cols:
        return "clusteredBarChart"
    return None


def _detect_stacked(vs) -> bool:
    """Detect if a bar/column chart should be stacked.

    Stacking is indicated by a color encoding on a dimension field
    in addition to category + value encodings.
    """
    if not hasattr(vs, "encoding"):
        return False
    color = vs.encoding.color
    if color and color.role == "dimension":
        # Color on a dimension = stacked segments
        return True
    return False


def _detect_combo(vs) -> bool:
    """Detect dual-axis / combo chart pattern.

    Indicators:
    - Multiple measures on value shelf
    - mark_type contains "dual" or "combo"
    """
    mark = vs.mark_type.lower() if hasattr(vs, "mark_type") else ""
    if "dual" in mark or "combo" in mark:
        return True
    # Multiple measures on cols/rows suggests dual measures
    rows = vs.encoding.shelf_rows if hasattr(vs, "encoding") else []
    cols = vs.encoding.shelf_cols if hasattr(vs, "encoding") else []
    measure_count = (
        sum(1 for c in rows if c.role == "measure")
        + sum(1 for c in cols if c.role == "measure")
    )
    return measure_count >= 2


def _detect_kpi_card(vs) -> bool:
    """Detect KPI/card pattern."""
    if getattr(vs, "is_kpi_card", False):
        return True
    # Don't treat mark types that have their own special mapping as KPIs
    mark_raw = getattr(vs, "mark_type", "").lower()
    if mark_raw in _SPECIAL_PATTERNS or mark_raw.replace("-", " ") in _SPECIAL_PATTERNS:
        return False
    # No shelves + text mark + few measures
    if not vs.encoding.shelf_rows and not vs.encoding.shelf_cols:
        if vs.mark_type in ("text", "automatic"):
            return True
    return False


def _detect_slicer_pattern(vs) -> bool:
    """Detect if this visual acts as a slicer/filter control."""
    name_lower = vs.name.lower() if hasattr(vs, "name") else ""
    # Quick filters shown as visuals
    if any(kw in name_lower for kw in ("filter", "slicer", "selector", "picker")):
        return True
    return False


def _detect_pseudo_tab(vs) -> bool:
    """Detect horizontal tile/selector pattern used as pseudo-tabs.

    Breadcrumb patterns or single-dimension horizontal layouts with
    filter actions are pseudo-tabs.
    """
    if getattr(vs, "is_breadcrumb", False):
        return True
    name_lower = vs.name.lower() if hasattr(vs, "name") else ""
    if any(kw in name_lower for kw in ("bread crumb", "breadcrumb", "tab", "selector", "toggle")):
        return True
    return False


# ================================================================== #
#  Main engine                                                          #
# ================================================================== #

def evaluate(vs) -> VisualEquivalency:
    """Evaluate a single VisualSpec and return the best PBI equivalent.

    Args:
        vs: VisualSpec from tableau_extractor

    Returns:
        VisualEquivalency with chosen visual, status, rationale, and notes
    """
    name = getattr(vs, "name", "unnamed")
    mark = getattr(vs, "mark_type", "automatic").lower().replace(" ", "-")

    result = VisualEquivalency(
        source_name=name,
        source_mark_type=mark,
        chosen_powerbi_visual="",
        equivalency_status=EquivalencyStatus.EXACT,
        rationale="",
    )

    # -- Priority 1: Semantic pattern detection --

    if _detect_pseudo_tab(vs):
        result.chosen_powerbi_visual = "slicer"
        result.equivalency_status = EquivalencyStatus.CLOSE_EQUIVALENT
        result.rationale = (
            "Tableau pseudo-tab / breadcrumb pattern. PBI slicer set to "
            "horizontal tile mode provides equivalent tab-like selection "
            "behavior. Wire to cross-filter other visuals on the page."
        )
        result.required_field_roles = [
            FieldRole(role="Values", field=_primary_dimension(vs),
                      notes="The field users select from"),
        ]
        result.formatting_notes = [
            "Set slicer orientation to 'Horizontal'",
            "Style as tiles (Slicer settings > Options > Style > Tile)",
            "Position at top of page for tab-like appearance",
        ]
        result.interaction_notes = [
            "Enable cross-filter to all other visuals on the page",
            "Single-select mode recommended for tab behavior",
        ]
        return result

    if _detect_slicer_pattern(vs):
        result.chosen_powerbi_visual = "slicer"
        result.equivalency_status = EquivalencyStatus.EXACT
        result.rationale = "Visual acts as a filter control. PBI slicer is the direct equivalent."
        result.required_field_roles = [
            FieldRole(role="Values", field=_primary_dimension(vs)),
        ]
        return result

    if _detect_kpi_card(vs):
        result.chosen_powerbi_visual = "card"
        # If explicitly flagged as KPI, it's exact. If inferred from
        # empty shelves + automatic mark, it's approximate and needs review.
        if getattr(vs, "is_kpi_card", False):
            result.equivalency_status = EquivalencyStatus.EXACT
        else:
            result.equivalency_status = EquivalencyStatus.CLOSE_EQUIVALENT
            result.analyst_review_required = True
            result.review_reasons = [
                "Inferred as KPI card from empty shelves -- verify this is "
                "not a misconfigured chart"
            ]
        result.rationale = (
            "Single-value display. PBI card visual shows one aggregated number."
        )
        measure = _primary_measure(vs)
        result.required_field_roles = [
            FieldRole(role="Values", field=measure,
                      aggregation=_primary_measure_agg(vs),
                      notes="The KPI value to display"),
        ]
        result.formatting_notes = [
            "Set display units and decimal places to match source",
            "Consider adding a callout value or target line if available",
        ]
        return result

    if getattr(vs, "is_map", False):
        result.chosen_powerbi_visual = "map"
        result.equivalency_status = EquivalencyStatus.CLOSE_EQUIVALENT
        result.rationale = (
            "Geographic visual. PBI map uses Bing Maps. Custom polygons and "
            "Tableau background images are not supported natively in PBI."
        )
        result.required_field_roles = [
            FieldRole(role="Location", field=_geo_field(vs),
                      notes="State, Country, City, or Lat/Long"),
        ]
        color_field = getattr(vs.encoding, "color", None)
        if color_field:
            result.required_field_roles.append(
                FieldRole(role="Color saturation", field=color_field.name,
                          required=False)
            )
        result.formatting_notes = [
            "PBI filled map (filledMap) may be more appropriate than bubble map",
            "Verify geocoding matches Bing Maps expectations",
        ]
        result.analyst_review_required = True
        result.review_reasons = ["Geographic visuals need manual geocoding verification"]
        return result

    if getattr(vs, "is_text_table", False):
        result.chosen_powerbi_visual = "matrix"
        result.equivalency_status = EquivalencyStatus.EXACT
        result.rationale = (
            "Text table with row and column dimensions. PBI matrix visual "
            "handles row/column grouping with subtotals."
        )
        result.required_field_roles = _matrix_roles(vs)
        result.formatting_notes = [
            "Enable stepped layout for hierarchical rows",
            "Add conditional formatting to replicate Tableau highlight table effect",
        ]
        return result

    # -- Priority 2: Combo chart detection --

    if _detect_combo(vs):
        pbi, status, rationale = _SPECIAL_PATTERNS.get(
            "combo", _SPECIAL_PATTERNS["dual-axis"]
        )
        result.chosen_powerbi_visual = pbi
        result.equivalency_status = status
        result.rationale = rationale
        result.required_field_roles = _combo_roles(vs)
        result.formatting_notes = [
            "Assign first measure to column, second to line",
            "PBI does not support independent Y-axis scaling",
        ]
        result.interaction_notes = [
            "Tooltip shows both measures at the hovered category",
        ]
        if status != EquivalencyStatus.EXACT:
            result.analyst_review_required = True
            result.review_reasons = ["Dual-axis scaling differs between platforms"]
        return result

    # -- Priority 3: Bar/column refinement --

    if mark in ("bar", "automatic"):
        is_stacked = _detect_stacked(vs)
        orientation = _detect_column_vs_bar(vs)

        if is_stacked:
            if orientation == "clusteredColumnChart":
                result.chosen_powerbi_visual = "stackedColumnChart"
            else:
                result.chosen_powerbi_visual = "stackedBarChart"
            result.equivalency_status = EquivalencyStatus.EXACT
            result.rationale = (
                "Stacked variant detected from color-on-dimension encoding. "
                "PBI stacked chart is a direct equivalent."
            )
        elif orientation:
            result.chosen_powerbi_visual = orientation
            result.equivalency_status = EquivalencyStatus.EXACT
            result.rationale = (
                f"Bar/column orientation inferred from shelf layout. "
                f"{'Horizontal bar' if 'Bar' in orientation else 'Vertical column'} "
                f"maps directly to PBI {orientation}."
            )
        else:
            # Fallback
            result.chosen_powerbi_visual = "clusteredBarChart"
            result.equivalency_status = EquivalencyStatus.CLOSE_EQUIVALENT
            result.rationale = (
                "Mark type is 'automatic' or 'bar' but shelf layout is ambiguous. "
                "Defaulting to clustered bar chart."
            )
            result.analyst_review_required = True
            result.review_reasons = ["Ambiguous orientation -- verify horizontal vs vertical"]

        result.required_field_roles = _bar_roles(vs, is_stacked)
        return result

    # -- Priority 4: Check special patterns --

    special_key = mark if mark in _SPECIAL_PATTERNS else mark.replace("-", " ")
    if special_key in _SPECIAL_PATTERNS:
        pbi, status, rationale = _SPECIAL_PATTERNS[special_key]
        result.chosen_powerbi_visual = pbi
        result.equivalency_status = status
        result.rationale = rationale
        result.required_field_roles = _generic_roles(vs)
        if status in (EquivalencyStatus.APPROXIMATION_REQUIRED, EquivalencyStatus.UNSUPPORTED):
            result.analyst_review_required = True
            result.review_reasons = [f"'{mark}' requires manual verification in PBI"]
        return result

    # -- Priority 5: Baseline map --

    baseline_key = mark if mark in _BASELINE_MAP else mark.replace("-", " ")
    if baseline_key in _BASELINE_MAP:
        pbi, status, rationale = _BASELINE_MAP[baseline_key]
        result.chosen_powerbi_visual = pbi
        result.equivalency_status = status
        result.rationale = rationale
        result.required_field_roles = _generic_roles(vs)
        if status in (EquivalencyStatus.APPROXIMATION_REQUIRED, EquivalencyStatus.UNSUPPORTED):
            result.analyst_review_required = True
            result.review_reasons = [f"'{mark}' has no exact PBI equivalent"]
        return result

    # -- Fallback: unknown --

    result.chosen_powerbi_visual = "clusteredBarChart"
    result.equivalency_status = EquivalencyStatus.APPROXIMATION_REQUIRED
    result.rationale = (
        f"Unknown mark type '{mark}'. Defaulting to clustered bar chart. "
        f"Manual review is required to select the correct PBI visual."
    )
    result.analyst_review_required = True
    result.review_reasons = [f"Unknown Tableau mark type: {mark}"]
    result.required_field_roles = _generic_roles(vs)
    return result


def evaluate_batch(visuals: list) -> List[VisualEquivalency]:
    """Evaluate a list of VisualSpecs."""
    return [evaluate(vs) for vs in visuals]


# ================================================================== #
#  Field role helpers                                                   #
# ================================================================== #

def _primary_dimension(vs) -> str:
    """Get the primary dimension field name from a VisualSpec."""
    if hasattr(vs, "encoding"):
        for shelf in (vs.encoding.shelf_rows, vs.encoding.shelf_cols):
            for col in shelf:
                if col.role == "dimension":
                    return col.name
    if hasattr(vs, "columns_used"):
        for col in vs.columns_used:
            if col.role == "dimension":
                return col.name
    return ""


def _primary_measure(vs) -> str:
    """Get the primary measure field name from a VisualSpec."""
    if hasattr(vs, "encoding"):
        for shelf in (vs.encoding.shelf_cols, vs.encoding.shelf_rows):
            for col in shelf:
                if col.role == "measure":
                    return col.name
    if hasattr(vs, "columns_used"):
        for col in vs.columns_used:
            if col.role == "measure":
                return col.name
    return ""


def _primary_measure_agg(vs) -> str:
    """Get the aggregation of the primary measure."""
    if hasattr(vs, "encoding"):
        for shelf in (vs.encoding.shelf_cols, vs.encoding.shelf_rows):
            for col in shelf:
                if col.role == "measure":
                    return col.aggregation
    return "Sum"


def _geo_field(vs) -> str:
    """Get a geographic field from the VisualSpec."""
    geo_hints = {"state", "country", "city", "region", "zip", "postal",
                 "latitude", "longitude", "lat", "long", "geo"}
    if hasattr(vs, "columns_used"):
        for col in vs.columns_used:
            if any(h in col.name.lower() for h in geo_hints):
                return col.name
    return _primary_dimension(vs)


def _bar_roles(vs, is_stacked: bool) -> List[FieldRole]:
    """Build field roles for bar/column charts."""
    roles = []
    dim = _primary_dimension(vs)
    meas = _primary_measure(vs)
    agg = _primary_measure_agg(vs)

    if dim:
        roles.append(FieldRole(role="Category", field=dim))
    if meas:
        roles.append(FieldRole(role="Values", field=meas, aggregation=agg))

    if is_stacked and hasattr(vs, "encoding") and vs.encoding.color:
        roles.append(FieldRole(
            role="Legend",
            field=vs.encoding.color.name,
            notes="Color dimension becomes the stacking series"
        ))

    return roles


def _matrix_roles(vs) -> List[FieldRole]:
    """Build field roles for matrix/table visuals."""
    roles = []
    if hasattr(vs, "encoding"):
        for col in vs.encoding.shelf_rows:
            roles.append(FieldRole(role="Rows", field=col.name))
        for col in vs.encoding.shelf_cols:
            if col.role == "dimension":
                roles.append(FieldRole(role="Columns", field=col.name))
            else:
                roles.append(FieldRole(role="Values", field=col.name,
                                       aggregation=col.aggregation))
    return roles or [FieldRole(role="Values", field="(configure manually)")]


def _combo_roles(vs) -> List[FieldRole]:
    """Build field roles for combo charts."""
    roles = []
    dim = _primary_dimension(vs)
    if dim:
        roles.append(FieldRole(role="Shared axis", field=dim))

    measures_found = 0
    if hasattr(vs, "encoding"):
        for shelf in (vs.encoding.shelf_rows, vs.encoding.shelf_cols):
            for col in shelf:
                if col.role == "measure":
                    role_name = "Column values" if measures_found == 0 else "Line values"
                    roles.append(FieldRole(role=role_name, field=col.name,
                                           aggregation=col.aggregation))
                    measures_found += 1

    if measures_found < 2:
        roles.append(FieldRole(
            role="Line values", field="(second measure needed)",
            required=True, notes="Combo chart requires at least 2 measures"
        ))

    return roles


def _generic_roles(vs) -> List[FieldRole]:
    """Build generic field roles from whatever is on the shelves."""
    roles = []
    dim = _primary_dimension(vs)
    meas = _primary_measure(vs)
    if dim:
        roles.append(FieldRole(role="Category", field=dim))
    if meas:
        roles.append(FieldRole(role="Values", field=meas,
                                aggregation=_primary_measure_agg(vs)))
    if hasattr(vs, "encoding") and vs.encoding.color:
        roles.append(FieldRole(role="Legend", field=vs.encoding.color.name,
                                required=False))
    return roles or [FieldRole(role="Values", field="(configure manually)")]
