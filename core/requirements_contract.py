"""
Requirements Contract -- structured intermediate spec that the pipeline must obey.

Runs BEFORE any DAX generation or Power BI report generation.
Converts user instructions + Tableau metadata into a deterministic,
auditable contract. Downstream generators consume this and must not ignore it.

No LLM calls. Purely rule-based and deterministic.
"""

import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum


# ================================================================== #
#  Enums                                                                #
# ================================================================== #

class VisualType(str, Enum):
    BAR = "clusteredBarChart"
    STACKED_BAR = "stackedBarChart"
    COLUMN = "clusteredColumnChart"
    STACKED_COLUMN = "stackedColumnChart"
    LINE = "lineChart"
    AREA = "areaChart"
    PIE = "pieChart"
    DONUT = "donutChart"
    SCATTER = "scatterChart"
    TREEMAP = "treemap"
    MAP = "map"
    FILLED_MAP = "filledMap"
    TABLE = "tableEx"
    MATRIX = "matrix"
    CARD = "card"
    MULTI_ROW_CARD = "multiRowCard"
    KPI = "kpi"
    GAUGE = "gauge"
    WATERFALL = "waterfallChart"
    FUNNEL = "funnelChart"
    COMBO = "lineClusteredColumnComboChart"
    SLICER = "slicer"
    TEXT_BOX = "textbox"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    EXACT = "exact"
    HIGH = "high_confidence"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown_manual_review"


class StackDirection(str, Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    GRID = "grid"
    NONE = "none"


class Priority(str, Enum):
    """Source priority -- higher number = higher priority.

    Ordering:
    1. USER_EXPLICIT -- user said it in the request
    2. DASHBOARD_SPEC -- from an explicit dashboard specification
    3. TABLEAU_DERIVED -- extracted from Tableau structure
    4. DEFAULT -- safe default
    5. HEURISTIC -- inferred guess
    """
    USER_EXPLICIT = "user_explicit"
    DASHBOARD_SPEC = "dashboard_spec"
    TABLEAU_DERIVED = "tableau_derived"
    DEFAULT = "default"
    HEURISTIC = "heuristic"


# Priority rank: higher number = stronger authority
PRIORITY_RANK = {
    Priority.USER_EXPLICIT: 50,
    Priority.DASHBOARD_SPEC: 40,
    Priority.TABLEAU_DERIVED: 30,
    Priority.DEFAULT: 20,
    Priority.HEURISTIC: 10,
}


class Requirement(str, Enum):
    """How important this element is to the output."""
    MUST_HAVE = "must_have"
    SHOULD_HAVE = "should_have"
    NICE_TO_HAVE = "nice_to_have"
    UNSUPPORTED = "unsupported"
    MANUAL_REVIEW = "manual_review"


# ================================================================== #
#  Contract schema                                                      #
# ================================================================== #

@dataclass
class AxisSpec:
    field: str
    aggregation: str = ""       # Sum, Avg, Count, None
    sort_direction: str = ""    # ASC, DESC
    sort_by: str = ""           # field name to sort by


@dataclass
class LegendSpec:
    field: str
    title: str = ""
    position: str = ""          # right, bottom, none


@dataclass
class FilterContractSpec:
    """A filter/slicer that must appear in the output."""
    field: str
    display_as: str = "slicer"   # slicer, dropdown, date_range, slider
    position: str = "top"        # top, right, left, inline
    title: str = ""
    is_quick_filter: bool = True
    page: str = ""               # which page this belongs to, empty = all
    priority: Priority = Priority.DEFAULT
    requirement: Requirement = Requirement.SHOULD_HAVE


@dataclass
class NavigationElement:
    """Tab-like or breadcrumb navigation."""
    element_type: str = "tab_selector"  # tab_selector, breadcrumb, bookmark
    selector_field: str = ""            # field used for tab selection
    selector_values: List[str] = field(default_factory=list)
    position: str = "top"
    notes: str = ""


@dataclass
class VisualContract:
    """Contract for a single visual that the generator must produce."""
    visual_id: str
    page: str                           # which page this visual belongs to
    title: str = ""
    visual_type: VisualType = VisualType.UNKNOWN
    confidence: Confidence = Confidence.HIGH
    priority: Priority = Priority.DEFAULT
    requirement: Requirement = Requirement.SHOULD_HAVE

    # Data bindings
    category_axis: Optional[AxisSpec] = None
    value_axis: Optional[AxisSpec] = None
    secondary_axis: Optional[AxisSpec] = None
    detail_fields: List[str] = field(default_factory=list)
    tooltip_fields: List[str] = field(default_factory=list)

    # Visual properties
    legend: Optional[LegendSpec] = None
    color_field: str = ""
    size_field: str = ""

    # Layout
    position_hint: str = ""             # "row1_full", "row2_left", "row2_right", etc.
    stacking_order: int = 0             # vertical order (0 = top)
    width_pct: float = 100.0            # percentage of canvas width
    height_pct: float = 0.0            # percentage of canvas height (0 = auto)

    # Source
    source_worksheet: str = ""          # original Tableau worksheet name
    source_mark_type: str = ""          # original Tableau mark type

    notes: List[str] = field(default_factory=list)


@dataclass
class LayoutConstraint:
    """A layout rule the generator must follow."""
    constraint_type: str    # "single_page", "vertical_stack", "kpi_row_top",
                            # "sidebar_filters_right", "max_visuals_per_page",
                            # "canvas_size", "proportional_from_source"
    value: str = ""
    source: str = ""        # "user_request", "tableau_structure", "default"
    notes: str = ""
    priority: Priority = Priority.DEFAULT
    requirement: Requirement = Requirement.SHOULD_HAVE


@dataclass
class FormattingConstraint:
    """A formatting rule."""
    constraint_type: str    # "theme", "font", "title_visible", "grid_lines", etc.
    value: str = ""
    source: str = ""
    priority: Priority = Priority.DEFAULT
    requirement: Requirement = Requirement.SHOULD_HAVE


@dataclass
class InteractivityRule:
    """An interactivity behavior the output must have."""
    rule_type: str          # "cross_filter", "drill_through", "bookmark_nav",
                            # "slicer_sync", "tooltip_page"
    description: str = ""
    source: str = ""
    priority: Priority = Priority.DEFAULT
    requirement: Requirement = Requirement.SHOULD_HAVE


@dataclass
class MappingAssumption:
    """A mapping decision made by the parser that may need review."""
    assumption: str
    reason: str = ""
    confidence: Confidence = Confidence.HIGH


@dataclass
class ManualReviewItem:
    """Something ambiguous that a human should verify."""
    item: str
    reason: str
    severity: str = "warning"   # warning, error, info


@dataclass
class UnsupportedItem:
    """A Tableau feature that cannot be reproduced in PBI."""
    item: str
    tableau_feature: str = ""
    pbi_alternative: str = ""
    severity: str = "warning"


@dataclass
class RequirementsContract:
    """The master contract that downstream generators must obey."""

    # Identity
    dashboard_title: str = ""
    contract_hash: str = ""             # SHA256 of inputs for auditability

    # Pages
    page_count: int = 1
    page_names: List[str] = field(default_factory=list)

    # Filters at the top / global level
    top_filters: List[FilterContractSpec] = field(default_factory=list)

    # Navigation
    navigation_elements: List[NavigationElement] = field(default_factory=list)

    # Visuals (the core -- each one is a binding contract)
    visuals: List[VisualContract] = field(default_factory=list)

    # Constraints
    layout_constraints: List[LayoutConstraint] = field(default_factory=list)
    formatting_constraints: List[FormattingConstraint] = field(default_factory=list)
    interactivity_rules: List[InteractivityRule] = field(default_factory=list)

    # Transparency
    mapping_assumptions: List[MappingAssumption] = field(default_factory=list)
    unsupported_items: List[UnsupportedItem] = field(default_factory=list)
    manual_review_items: List[ManualReviewItem] = field(default_factory=list)

    # Metadata
    source_type: str = ""               # "tableau_migration", "fresh_design", "user_only"
    source_file: str = ""
    parser_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ================================================================== #
#  Parser: user request text analysis                                   #
# ================================================================== #

_WORD_TO_NUM = {
    "one": 1, "single": 1, "a": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_page_count(text: str) -> Optional[int]:
    """Extract explicit page count from user text."""
    if not text:
        return None
    lower = text.lower()

    # "single page", "one page", "1 page"
    m = re.search(r'(\d+)\s*[-\s]?pages?', lower)
    if m:
        return int(m.group(1))

    for word, num in _WORD_TO_NUM.items():
        if re.search(rf'\b{word}\b\s*[-\s]?pages?', lower):
            return num

    if "single page" in lower or "one page" in lower:
        return 1

    return None


def _parse_page_names(text: str) -> List[str]:
    """Extract explicit page names from user text.

    Matches patterns like:
    - 'page called "Overview"'
    - 'page named "Sales"'
    - pages: Overview, Sales, Details
    """
    names = []

    # Quoted names after "called" or "named"
    for m in re.finditer(r'(?:called|named)\s+["\']([^"\']+)["\']', text, re.IGNORECASE):
        names.append(m.group(1))

    # "page: Name" or "pages: Name1, Name2"
    m = re.search(r'pages?\s*:\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m and not names:
        parts = [p.strip().strip('"\'') for p in m.group(1).split(",")]
        names.extend(p for p in parts if p)

    return names


def _parse_stacking(text: str) -> Optional[StackDirection]:
    """Detect stacking intent from user text."""
    lower = text.lower()
    if re.search(r'stack\w*\s+\w*\s*vertically|vertical\w*\s+stack', lower):
        return StackDirection.VERTICAL
    if re.search(r'stack\w*\s+\w*\s*horizontally|side\s+by\s+side', lower):
        return StackDirection.HORIZONTAL
    if "grid" in lower:
        return StackDirection.GRID
    return None


def _parse_visual_mentions(text: str) -> List[dict]:
    """Extract explicitly mentioned visuals from user text.

    Patterns: "bar chart showing X", "line chart of Y", "KPI card for Z",
    "three monthly visuals", "slicer for Region"
    """
    visuals = []
    lower = text.lower()

    type_patterns = [
        (r'bar\s*chart', VisualType.BAR),
        (r'column\s*chart', VisualType.COLUMN),
        (r'line\s*chart', VisualType.LINE),
        (r'area\s*chart', VisualType.AREA),
        (r'pie\s*chart', VisualType.PIE),
        (r'donut\s*chart', VisualType.DONUT),
        (r'scatter\s*(?:plot|chart)', VisualType.SCATTER),
        (r'treemap', VisualType.TREEMAP),
        (r'(?:geo\s*)?map', VisualType.MAP),
        (r'table', VisualType.TABLE),
        (r'matrix', VisualType.MATRIX),
        (r'card|kpi', VisualType.CARD),
        (r'gauge', VisualType.GAUGE),
        (r'waterfall', VisualType.WATERFALL),
        (r'funnel', VisualType.FUNNEL),
        (r'combo\s*chart', VisualType.COMBO),
        (r'slicer', VisualType.SLICER),
    ]

    for pattern, vtype in type_patterns:
        for m in re.finditer(pattern, lower):
            visuals.append({"type": vtype, "position": m.start()})

    # "N visuals" or "N charts" or "N bar charts" -- count of visuals
    m = re.search(r'(\d+|three|four|five|six|seven|eight)\s+(?:\w+\s+)*?(?:visuals?|charts?)', lower)
    if m:
        count_str = m.group(1)
        count = _WORD_TO_NUM.get(count_str, None)
        if count is None:
            try:
                count = int(count_str)
            except ValueError:
                count = 0
        if count > len(visuals):
            for i in range(count - len(visuals)):
                visuals.append({"type": VisualType.UNKNOWN, "position": m.start() + i})

    return visuals


def _parse_filter_mentions(text: str) -> List[dict]:
    """Extract slicer/filter mentions from user text.

    Patterns: "3 slicers in the top row", "slicer for Region",
    "top slicers", "filter by Category"
    """
    filters = []
    lower = text.lower()

    # "N slicers in the top row" -- generic count-based
    m = re.search(r'(\d+)\s+slicers?\s+(?:in\s+)?(?:the\s+)?(?:top|header)\s*(?:row)?', lower)
    if m:
        count = int(m.group(1))
        for i in range(count):
            filters.append({"field": f"Slicer {i+1}", "position": "top", "generic": True})

    # "slicer(s) for X"
    for m in re.finditer(r'slicers?\s+(?:for|on|by)\s+(\w[\w\s]*?)(?:\s*,|\s+and\s+|$)', lower):
        filters.append({"field": m.group(1).strip(), "position": "top"})

    # "top slicers" without specific field (only if no count detected)
    if "top slicer" in lower and not filters:
        filters.append({"field": "", "position": "top", "generic": True})

    # "filter by X"
    for m in re.finditer(r'filter\s+(?:by|on)\s+(\w[\w\s]*?)(?:\s*,|\s+and\s+|$)', lower):
        f = m.group(1).strip()
        if f not in {fi.get("field") for fi in filters}:
            filters.append({"field": f, "position": "top"})

    return filters


def _parse_tab_mentions(text: str) -> bool:
    """Detect if user mentions tab-like navigation."""
    lower = text.lower()
    return any(p in lower for p in [
        "tab-like", "tab selector", "category selector", "tab navigation",
        "navigation tab", "toggle", "show/hide",
    ])


def _parse_descriptive_visuals(text: str) -> List[dict]:
    """Parse descriptive visual specifications from user text.

    Handles patterns like:
    - "1) TV MTD as stacked column with MonthYear on X-axis and legend by transaction subtype"
    - "2) TC MTD as stacked column with MonthYear on X-axis and legend by transaction subtype"
    - "3) UU MTD as clustered column with MonthYear on X-axis"
    """
    visuals = []

    # Pattern: N) TITLE as TYPE with FIELD on X-axis [and legend by FIELD]
    pattern = re.compile(
        r'(\d+)\)\s+'
        r'(\w[\w\s]*?)\s+'
        r'as\s+(stacked\s+column|clustered\s+column|stacked\s+bar|clustered\s+bar'
        r'|line\s+chart|bar\s+chart|column\s+chart|area\s+chart|pie\s+chart'
        r'|donut\s+chart|scatter|table|matrix|card|gauge|combo\s+chart'
        r'|waterfall|treemap|funnel)'
        r'(?:\s+with\s+(\w[\w\s]*?)\s+on\s+(?:the\s+)?(?:x[- ]?axis|category\s+axis|axis))?'
        r'(?:\s+and\s+legend\s+by\s+(\w[\w\s]*?))?'
        r'(?:\s*[,.]|$)',
        re.IGNORECASE
    )

    for m in pattern.finditer(text):
        idx = int(m.group(1))
        title = m.group(2).strip()
        chart_type_raw = m.group(3).strip().lower()
        x_field = m.group(4).strip() if m.group(4) else ""
        legend_field = m.group(5).strip() if m.group(5) else ""

        # Map descriptive type to VisualType
        type_map = {
            "stacked column": VisualType.STACKED_COLUMN,
            "clustered column": VisualType.COLUMN,
            "stacked bar": VisualType.STACKED_BAR,
            "clustered bar": VisualType.BAR,
            "line chart": VisualType.LINE,
            "bar chart": VisualType.BAR,
            "column chart": VisualType.COLUMN,
            "area chart": VisualType.AREA,
            "pie chart": VisualType.PIE,
            "donut chart": VisualType.DONUT,
            "scatter": VisualType.SCATTER,
            "table": VisualType.TABLE,
            "matrix": VisualType.MATRIX,
            "card": VisualType.CARD,
            "gauge": VisualType.GAUGE,
            "combo chart": VisualType.COMBO,
            "waterfall": VisualType.WATERFALL,
            "treemap": VisualType.TREEMAP,
            "funnel": VisualType.FUNNEL,
        }

        vtype = type_map.get(chart_type_raw, VisualType.UNKNOWN)

        visuals.append({
            "index": idx,
            "title": title,
            "type": vtype,
            "x_field": x_field,
            "legend_field": legend_field,
        })

    return visuals


def _parse_refresh_text(text: str) -> Optional[dict]:
    """Detect refresh text / last-updated text in user request."""
    lower = text.lower()
    patterns = [
        r'(?:top[- ]?right|top|header)\s+(?:last\s+)?refresh\s+text',
        r'last\s+refresh(?:ed)?\s+(?:text|timestamp|date)',
        r'refresh\s+(?:text|timestamp|date)\s+(?:in|at)\s+(?:the\s+)?(?:top|header)',
    ]
    for p in patterns:
        m = re.search(p, lower)
        if m:
            return {"position": "top-right", "text": "Last Refresh"}
    return None


def _parse_section_labels(text: str) -> List[dict]:
    """Extract section labels from user text.

    Matches: 'section label "Total Monthly"', 'header "Overview"'
    """
    labels = []
    for m in re.finditer(r'section\s+label\s+["\']([^"\']+)["\']', text, re.IGNORECASE):
        labels.append({"title": m.group(1), "type": "section_label"})
    for m in re.finditer(r'section\s+header\s+["\']([^"\']+)["\']', text, re.IGNORECASE):
        labels.append({"title": m.group(1), "type": "section_header"})
    return labels


def _parse_interactivity(text: str) -> List[dict]:
    """Parse interactivity rules from user text."""
    rules = []
    lower = text.lower()
    if any(p in lower for p in ["cross-filter", "cross filter", "crossfilter", "enable cross"]):
        rules.append({"rule_type": "cross_filter", "description": "Cross-filtering enabled between all visuals"})
    if any(p in lower for p in ["drill through", "drill-through", "drillthrough"]):
        rules.append({"rule_type": "drill_through", "description": "Drill-through enabled"})
    if any(p in lower for p in ["tooltip page", "custom tooltip"]):
        rules.append({"rule_type": "tooltip_page", "description": "Custom tooltip page"})
    return rules


def _parse_formatting(text: str) -> List[dict]:
    """Parse formatting constraints from user text."""
    constraints = []
    lower = text.lower()
    if any(p in lower for p in ["consistent legend color", "consistent color", "same legend color",
                                 "consistent legend", "shared legend color"]):
        constraints.append({
            "constraint_type": "legend_color_consistency",
            "value": "Consistent legend colors across all visuals with shared dimensions",
        })
    if "dark theme" in lower or "dark mode" in lower:
        constraints.append({"constraint_type": "theme", "value": "dark"})
    if "light theme" in lower or "light mode" in lower:
        constraints.append({"constraint_type": "theme", "value": "light"})
    return constraints


# ================================================================== #
#  Parser: Tableau structure analysis                                    #
# ================================================================== #

def _build_visuals_from_tableau(tableau_wb, page_name: str) -> Tuple[
    List[VisualContract], List[FilterContractSpec], List[NavigationElement],
    List[MappingAssumption], List[UnsupportedItem]
]:
    """Build visual contracts from a TableauWorkbook spec.

    Returns (visuals, filters, nav_elements, assumptions, unsupported)
    """
    visuals = []
    filters = []
    nav_elements = []
    assumptions = []
    unsupported = []

    # Process dashboards -> pages
    for db in tableau_wb.dashboards:
        # Worksheet zones -> visuals
        for idx, wz in enumerate(db.worksheet_zones):
            # Find the matching VisualSpec
            vs = None
            for ws in tableau_wb.worksheets:
                if ws.name == wz.name:
                    vs = ws
                    break

            vc = VisualContract(
                visual_id=f"v{idx + 1}",
                page=page_name or db.name,
                source_worksheet=wz.name,
                stacking_order=idx,
            )

            if vs:
                vc.title = vs.name
                vc.visual_type = VisualType(vs.mark_type_pbi) if vs.mark_type_pbi in [e.value for e in VisualType] else VisualType.UNKNOWN
                vc.confidence = Confidence(vs.confidence.value)
                vc.source_mark_type = vs.mark_type

                # Axes from shelf encodings
                if vs.encoding.shelf_rows:
                    first_row = vs.encoding.shelf_rows[0]
                    if first_row.role == "dimension":
                        vc.category_axis = AxisSpec(
                            field=first_row.name,
                            aggregation=first_row.aggregation,
                        )
                    else:
                        vc.value_axis = AxisSpec(
                            field=first_row.name,
                            aggregation=first_row.aggregation,
                        )

                if vs.encoding.shelf_cols:
                    first_col = vs.encoding.shelf_cols[0]
                    if first_col.role == "measure" and not vc.value_axis:
                        vc.value_axis = AxisSpec(
                            field=first_col.name,
                            aggregation=first_col.aggregation,
                        )
                    elif first_col.role == "dimension" and not vc.category_axis:
                        vc.category_axis = AxisSpec(
                            field=first_col.name,
                            aggregation=first_col.aggregation,
                        )

                # Color -> legend
                if vs.encoding.color:
                    vc.color_field = vs.encoding.color.name
                    vc.legend = LegendSpec(
                        field=vs.encoding.color.name,
                        title=vs.legend_title or vs.encoding.color.name,
                    )

                # Size
                if vs.encoding.size:
                    vc.size_field = vs.encoding.size.name

                # Tooltips
                vc.tooltip_fields = [t.name for t in vs.encoding.tooltip]

                # Sorts
                if vs.sorts:
                    sort = vs.sorts[0]
                    if vc.category_axis:
                        vc.category_axis.sort_direction = sort.direction
                        vc.category_axis.sort_by = sort.measure

                # KPI / breadcrumb notes
                if vs.is_kpi_card:
                    vc.notes.append("Detected as KPI card from Tableau structure")
                if vs.is_breadcrumb:
                    vc.notes.append("Tableau breadcrumb pattern -- shows current filter context")
                if vs.is_map:
                    vc.notes.append("Geographic visual -- requires lat/long or geo fields")

                # Mark type confidence notes
                if vs.confidence.value == "approximate":
                    assumptions.append(MappingAssumption(
                        assumption=f"'{vs.name}' mapped {vs.mark_type} -> {vs.mark_type_pbi}",
                        reason="No exact PBI equivalent",
                        confidence=Confidence.APPROXIMATE,
                    ))

            # Layout from zone coordinates
            if wz.pbi_w > 0:
                vc.width_pct = round(wz.pbi_w / 1280 * 100, 1)
                vc.height_pct = round(wz.pbi_h / 720 * 100, 1)

            visuals.append(vc)

        # Filter zones -> slicers
        for fz in db.filter_zones:
            field_name = fz.param
            # Clean Tableau field ref
            clean = re.sub(r'\[.*?\]\.', '', field_name)
            clean = re.sub(r'\[(.*?)\]', r'\1', clean)
            # Parse encoded name
            parts = clean.split(":")
            if len(parts) >= 2:
                clean = parts[1]

            mode_map = {
                "checkdropdown": "dropdown",
                "compact": "dropdown",
                "slider": "slider",
                "type_in": "text_input",
            }

            filters.append(FilterContractSpec(
                field=clean,
                display_as="slicer",
                position="top" if fz.y < 30000 else "right",
                title=clean,
                is_quick_filter=True,
                page=page_name or db.name,
            ))

        # Param zones -> slicers
        for pz in db.param_zones:
            param_ref = pz.param
            clean = re.sub(r'\[Parameters\]\.\[(.*?)\]', r'\1', param_ref)
            filters.append(FilterContractSpec(
                field=clean,
                display_as="slicer",
                position="top",
                title=clean,
                page=page_name or db.name,
            ))

        # Tab detection from hidden zones
        if db.has_tab_selector:
            nav_elements.append(NavigationElement(
                element_type="tab_selector",
                notes=f"Dashboard '{db.name}' has hidden zones suggesting tab-like navigation",
                position="top",
            ))

    # Unsupported features
    for cf in tableau_wb.calculated_fields:
        if cf.is_table_calc:
            unsupported.append(UnsupportedItem(
                item=f"Table calculation: {cf.name}",
                tableau_feature="Table calculations (RUNNING_, WINDOW_, INDEX, RANK)",
                pbi_alternative="DAX window functions or CALCULATE with FILTER",
                severity="warning",
            ))
        if cf.is_lod:
            assumptions.append(MappingAssumption(
                assumption=f"LOD expression '{cf.name}' converted to DAX CALCULATE with ALLEXCEPT/VALUES",
                reason="LOD expressions have no direct DAX equivalent -- semantics may differ",
                confidence=Confidence.APPROXIMATE,
            ))

    return visuals, filters, nav_elements, assumptions, unsupported


# ================================================================== #
#  Main parser                                                          #
# ================================================================== #

def build_contract(
    user_request: str,
    tableau_wb=None,
    source_file: str = "",
    data_columns: List[str] = None,
) -> RequirementsContract:
    """Build a RequirementsContract from user request + optional Tableau structure.

    This is the central function. It is deterministic -- same inputs always
    produce same outputs. No LLM calls.

    Args:
        user_request: natural language from the user
        tableau_wb: TableauWorkbook from tableau_extractor (optional)
        source_file: name of source file
        data_columns: list of column names from the data profile

    Returns:
        RequirementsContract
    """
    contract = RequirementsContract(
        source_file=source_file,
    )

    # Compute hash for auditability
    hash_input = f"{user_request}|{source_file}|{bool(tableau_wb)}"
    contract.contract_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # ---- Parse user request ----

    # Page count: user intent takes absolute priority
    user_page_count = _parse_page_count(user_request)
    user_page_names = _parse_page_names(user_request)
    user_stacking = _parse_stacking(user_request)
    user_visuals = _parse_visual_mentions(user_request)
    user_descriptive_visuals = _parse_descriptive_visuals(user_request)
    user_filters = _parse_filter_mentions(user_request)
    user_wants_tabs = _parse_tab_mentions(user_request)
    user_refresh = _parse_refresh_text(user_request)
    user_section_labels = _parse_section_labels(user_request)
    user_interactivity = _parse_interactivity(user_request)
    user_formatting = _parse_formatting(user_request)

    # ---- Tableau structure (if available) ----

    tableau_visuals = []
    tableau_filters = []
    tableau_nav = []
    tableau_assumptions = []
    tableau_unsupported = []

    if tableau_wb:
        contract.source_type = "tableau_migration"

        # Determine default page name
        default_page = ""
        if user_page_names:
            default_page = user_page_names[0]
        elif tableau_wb.dashboards:
            default_page = tableau_wb.dashboards[0].name

        tableau_visuals, tableau_filters, tableau_nav, tableau_assumptions, tableau_unsupported = \
            _build_visuals_from_tableau(tableau_wb, default_page)

        # Dashboard title from Tableau
        if not contract.dashboard_title and tableau_wb.dashboards:
            contract.dashboard_title = tableau_wb.dashboards[0].name
    else:
        contract.source_type = "fresh_design"

    # ---- Merge user intent with Tableau structure ----

    # Title: user > Tableau > default
    title_match = re.search(r'(?:titled?|called|named)\s+["\']([^"\']+)["\']', user_request, re.IGNORECASE)
    if title_match:
        contract.dashboard_title = title_match.group(1)
    elif not contract.dashboard_title:
        contract.dashboard_title = "Dashboard"

    # Page count: user explicit > Tableau dashboards > 1
    if user_page_count is not None:
        contract.page_count = user_page_count
        contract.layout_constraints.append(LayoutConstraint(
            constraint_type="page_count_fixed",
            value=str(user_page_count),
            source="user_request",
            notes=f"User explicitly requested {user_page_count} page(s)",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
        ))
    elif tableau_wb and tableau_wb.dashboards:
        contract.page_count = len(tableau_wb.dashboards)
    else:
        contract.page_count = 1

    # Page names
    if user_page_names:
        contract.page_names = user_page_names[:contract.page_count]
    elif tableau_wb and tableau_wb.dashboards:
        contract.page_names = [db.name for db in tableau_wb.dashboards[:contract.page_count]]

    # Pad page names if needed
    while len(contract.page_names) < contract.page_count:
        contract.page_names.append(f"Page {len(contract.page_names) + 1}")

    # Trim if user said fewer pages than Tableau has
    if len(contract.page_names) > contract.page_count:
        contract.page_names = contract.page_names[:contract.page_count]

    # If user specified single page but Tableau has multiple dashboards, record assumption
    if user_page_count == 1 and tableau_wb and len(tableau_wb.dashboards) > 1:
        contract.mapping_assumptions.append(MappingAssumption(
            assumption=f"Merging {len(tableau_wb.dashboards)} Tableau dashboards into 1 page per user request",
            reason="User explicitly requested single page",
            confidence=Confidence.HIGH,
        ))

    # ---- Visuals ----

    page_name = contract.page_names[0] if contract.page_names else "Page 1"

    if tableau_visuals:
        # Assign all visuals to the correct page(s)
        if contract.page_count == 1 and len(contract.page_names) == 1:
            for v in tableau_visuals:
                v.page = page_name
        contract.visuals = tableau_visuals

    elif user_descriptive_visuals:
        # Descriptive visuals take priority (they have titles, types, axes)
        chart_idx = 0
        for dv in user_descriptive_visuals:
            chart_idx += 1
            contract.visuals.append(VisualContract(
                visual_id=f"chart_{chart_idx}",
                page=page_name,
                title=dv["title"],
                visual_type=dv["type"],
                stacking_order=chart_idx - 1,
                confidence=Confidence.EXACT,
                priority=Priority.USER_EXPLICIT,
                requirement=Requirement.MUST_HAVE,
                category_axis=AxisSpec(field=dv["x_field"]) if dv.get("x_field") else None,
                value_axis=AxisSpec(field=dv["title"], aggregation="Sum") if dv.get("title") else None,
                color_field=dv.get("legend_field", ""),
                notes=[f"Parsed from descriptive spec: '{dv['title']} as {dv['type']}'"],
            ))

    elif user_visuals:
        # Generic visual mentions (bar chart, line chart, etc.)
        for idx, uv in enumerate(user_visuals):
            contract.visuals.append(VisualContract(
                visual_id=f"v{idx + 1}",
                page=page_name,
                visual_type=uv["type"],
                stacking_order=idx,
                confidence=Confidence.APPROXIMATE,
                priority=Priority.USER_EXPLICIT,
                requirement=Requirement.SHOULD_HAVE,
                notes=["Inferred from user request text"],
            ))

    # ---- Refresh text ----

    if user_refresh:
        contract.visuals.append(VisualContract(
            visual_id="refresh_text",
            page=page_name,
            title="Last Refresh",
            visual_type=VisualType.TEXT_BOX,
            confidence=Confidence.EXACT,
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
            position_hint=user_refresh.get("position", "top-right"),
            notes=["Refresh timestamp text box"],
        ))
        contract.layout_constraints.append(LayoutConstraint(
            constraint_type="refresh_text",
            value=user_refresh.get("position", "top-right"),
            source="user_request",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
        ))

    # ---- Section labels ----

    for sl in user_section_labels:
        contract.visuals.append(VisualContract(
            visual_id=f"section_{sl['title'].lower().replace(' ', '_')}",
            page=page_name,
            title=sl["title"],
            visual_type=VisualType.TEXT_BOX,
            confidence=Confidence.EXACT,
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
            notes=[f"Section label: {sl['title']}"],
        ))

    # ---- Stacking / layout ----

    if user_stacking:
        contract.layout_constraints.append(LayoutConstraint(
            constraint_type="stacking_direction",
            value=user_stacking.value,
            source="user_request",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
        ))
        # Apply stacking order to visuals
        if user_stacking == StackDirection.VERTICAL:
            for idx, v in enumerate(contract.visuals):
                v.stacking_order = idx
                v.width_pct = 100.0
                v.position_hint = f"row{idx + 1}_full"

    elif tableau_wb:
        # Infer from Tableau layout
        for db in tableau_wb.dashboards:
            if db.has_vertical_stack:
                contract.layout_constraints.append(LayoutConstraint(
                    constraint_type="stacking_direction",
                    value="vertical",
                    source="tableau_structure",
                    notes=f"Dashboard '{db.name}' uses vertical stacking",
                    priority=Priority.TABLEAU_DERIVED,
                    requirement=Requirement.SHOULD_HAVE,
                ))
            if db.layout_pattern:
                contract.layout_constraints.append(LayoutConstraint(
                    constraint_type="layout_pattern",
                    value=db.layout_pattern,
                    source="tableau_structure",
                    priority=Priority.TABLEAU_DERIVED,
                    requirement=Requirement.SHOULD_HAVE,
                ))

    # Canvas size
    contract.layout_constraints.append(LayoutConstraint(
        constraint_type="canvas_size",
        value="1280x720",
        source="default",
        priority=Priority.DEFAULT,
        requirement=Requirement.MUST_HAVE,
    ))

    # ---- Filters / slicers ----

    # Merge user-requested filters with Tableau-detected filters
    seen_fields = set()
    all_filters = []

    for uf in user_filters:
        f = FilterContractSpec(
            field=uf["field"],
            position=uf.get("position", "top"),
            page=contract.page_names[0] if contract.page_names else "",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
        )
        all_filters.append(f)
        if uf["field"]:
            seen_fields.add(uf["field"].lower())

    for tf in tableau_filters:
        if tf.field.lower() not in seen_fields:
            tf.priority = Priority.TABLEAU_DERIVED
            tf.requirement = Requirement.SHOULD_HAVE
            all_filters.append(tf)
            seen_fields.add(tf.field.lower())

    contract.top_filters = all_filters

    # ---- Navigation ----

    if user_wants_tabs:
        has_tab_from_tableau = any(
            ne.element_type == "tab_selector" for ne in tableau_nav
        )
        if not has_tab_from_tableau:
            contract.navigation_elements.append(NavigationElement(
                element_type="tab_selector",
                notes="User requested tab-like navigation",
                position="top",
            ))
    contract.navigation_elements.extend(tableau_nav)

    # ---- Interactivity rules ----

    for ir in user_interactivity:
        contract.interactivity_rules.append(InteractivityRule(
            rule_type=ir["rule_type"],
            description=ir["description"],
            source="user_request",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.MUST_HAVE,
        ))

    # ---- Formatting constraints ----

    for fc in user_formatting:
        contract.formatting_constraints.append(FormattingConstraint(
            constraint_type=fc["constraint_type"],
            value=fc["value"],
            source="user_request",
            priority=Priority.USER_EXPLICIT,
            requirement=Requirement.SHOULD_HAVE,
        ))

    # ---- Assumptions, unsupported, manual review ----

    contract.mapping_assumptions.extend(tableau_assumptions)
    contract.unsupported_items.extend(tableau_unsupported)

    # Check for ambiguities
    if not contract.visuals and not user_visuals:
        contract.manual_review_items.append(ManualReviewItem(
            item="No visuals specified",
            reason="Neither user request nor Tableau structure provided visual definitions. "
                   "The LLM will infer visuals from the data profile.",
            severity="warning",
        ))

    if user_page_count is None and not tableau_wb:
        contract.manual_review_items.append(ManualReviewItem(
            item="Page count not specified",
            reason="User did not specify how many pages. Defaulting to 1.",
            severity="info",
        ))

    # Ambiguous visual types
    for v in contract.visuals:
        if v.visual_type == VisualType.UNKNOWN:
            contract.manual_review_items.append(ManualReviewItem(
                item=f"Visual '{v.visual_id}' ({v.source_worksheet or 'unnamed'}) has unknown type",
                reason="Could not determine chart type from Tableau mark or user request",
                severity="warning",
            ))

    return contract


# ================================================================== #
#  Validator                                                            #
# ================================================================== #

def validate_contract(contract: RequirementsContract) -> List[str]:
    """Validate a RequirementsContract for internal consistency.

    Returns a list of error strings. Empty list = valid.
    """
    errors = []

    # Page count matches page names
    if contract.page_count != len(contract.page_names):
        errors.append(
            f"page_count ({contract.page_count}) != len(page_names) ({len(contract.page_names)})"
        )

    # Every visual references a valid page
    valid_pages = set(contract.page_names)
    for v in contract.visuals:
        if v.page and v.page not in valid_pages:
            errors.append(
                f"Visual '{v.visual_id}' references page '{v.page}' "
                f"which is not in page_names: {contract.page_names}"
            )

    # No duplicate visual IDs
    ids = [v.visual_id for v in contract.visuals]
    dupes = [vid for vid in ids if ids.count(vid) > 1]
    if dupes:
        errors.append(f"Duplicate visual IDs: {set(dupes)}")

    # Stacking order should be unique per page
    for page_name in contract.page_names:
        page_visuals = [v for v in contract.visuals if v.page == page_name]
        orders = [v.stacking_order for v in page_visuals]
        if len(orders) != len(set(orders)):
            errors.append(f"Page '{page_name}' has duplicate stacking_order values")

    # At least one page
    if contract.page_count < 1:
        errors.append("page_count must be >= 1")

    # Priority rule: check for conflicting constraints where lower-priority
    # would override higher-priority
    by_type = {}
    for lc in contract.layout_constraints:
        if lc.constraint_type not in by_type:
            by_type[lc.constraint_type] = []
        by_type[lc.constraint_type].append(lc)

    for ctype, constraints in by_type.items():
        if len(constraints) > 1:
            # Multiple constraints of same type -- verify priority ordering
            sorted_by_priority = sorted(
                constraints,
                key=lambda c: PRIORITY_RANK.get(c.priority, 0),
                reverse=True,
            )
            winner = sorted_by_priority[0]
            for other in sorted_by_priority[1:]:
                if other.value != winner.value:
                    rank_w = PRIORITY_RANK.get(winner.priority, 0)
                    rank_o = PRIORITY_RANK.get(other.priority, 0)
                    if rank_o > rank_w:
                        errors.append(
                            f"Priority violation: {ctype} has "
                            f"'{other.value}' (priority={other.priority.value}) "
                            f"overriding '{winner.value}' (priority={winner.priority.value})"
                        )

    return errors


# ================================================================== #
#  Contract enforcement (post-LLM validation)                           #
# ================================================================== #

def enforce_contract(contract: RequirementsContract, llm_spec: dict) -> Tuple[dict, List[str]]:
    """Validate an LLM-generated dashboard spec against the contract.

    Returns (corrected_spec, violations) where violations is a list of
    strings describing what was wrong and how it was fixed.
    """
    violations = []
    spec = llm_spec.copy()

    # Enforce page count
    pages = spec.get("pages", [])
    page_count_constraint = next(
        (lc for lc in contract.layout_constraints if lc.constraint_type == "page_count_fixed"),
        None
    )

    if page_count_constraint:
        required = int(page_count_constraint.value)
        if len(pages) != required:
            violations.append(
                f"Contract requires {required} page(s) but LLM generated {len(pages)}. "
                f"Trimming/padding to match."
            )
            # Trim excess pages
            while len(pages) > required:
                pages.pop()
            # Pad missing pages
            while len(pages) < required:
                pages.append({
                    "name": contract.page_names[len(pages)] if len(pages) < len(contract.page_names) else f"Page {len(pages) + 1}",
                    "visuals": [],
                    "slicers": [],
                })
            spec["pages"] = pages

    # Enforce page names
    for i, page in enumerate(spec.get("pages", [])):
        if i < len(contract.page_names):
            expected_name = contract.page_names[i]
            actual_name = page.get("name", "")
            if actual_name != expected_name:
                violations.append(
                    f"Page {i} name '{actual_name}' changed to '{expected_name}' per contract"
                )
                page["name"] = expected_name

    # Enforce dashboard title
    if contract.dashboard_title:
        actual_title = spec.get("dashboard_title", "")
        if actual_title != contract.dashboard_title:
            violations.append(
                f"Title '{actual_title}' changed to '{contract.dashboard_title}' per contract"
            )
            spec["dashboard_title"] = contract.dashboard_title

    # Enforce MUST_HAVE visuals exist
    must_have_visuals = [
        v for v in contract.visuals if v.requirement == Requirement.MUST_HAVE
    ]
    spec_visual_titles = set()
    for page in spec.get("pages", []):
        for vis in page.get("visuals", []):
            spec_visual_titles.add(vis.get("title", "").lower())
            spec_visual_titles.add(vis.get("id", "").lower())

    for mhv in must_have_visuals:
        found = (
            mhv.title.lower() in spec_visual_titles
            or mhv.visual_id.lower() in spec_visual_titles
        )
        if not found:
            violations.append(
                f"MUST_HAVE visual '{mhv.title or mhv.visual_id}' "
                f"(type={mhv.visual_type}) missing from LLM output. "
                f"Priority: {mhv.priority.value}. "
                f"This visual was explicitly requested and must not be omitted."
            )

    # Enforce MUST_HAVE slicers exist
    must_have_slicers = [
        f for f in contract.top_filters if f.requirement == Requirement.MUST_HAVE
    ]
    spec_slicers = set()
    for page in spec.get("pages", []):
        for s in page.get("slicers", []):
            spec_slicers.add(s.get("field", "").lower())

    for mhs in must_have_slicers:
        if mhs.field and mhs.field.lower() not in spec_slicers:
            violations.append(
                f"MUST_HAVE slicer for '{mhs.field}' missing from LLM output. "
                f"Priority: {mhs.priority.value}."
            )

    # Enforce MUST_HAVE interactivity rules
    must_have_rules = [
        ir for ir in contract.interactivity_rules
        if ir.requirement == Requirement.MUST_HAVE
    ]
    for mhr in must_have_rules:
        violations.append(
            f"MUST_HAVE interactivity rule '{mhr.rule_type}' -- verify manually "
            f"in Power BI Desktop."
        )

    return spec, violations


# ================================================================== #
#  Prompt formatter                                                     #
# ================================================================== #

def format_contract_for_prompt(contract: RequirementsContract) -> str:
    """Format the contract as a prompt section the LLM must follow.

    This is injected into the system or user message to constrain LLM output.
    """
    lines = []
    lines.append("=== REQUIREMENTS CONTRACT (BINDING -- DO NOT DEVIATE) ===")
    lines.append("")
    lines.append(f"TITLE: \"{contract.dashboard_title}\"")
    lines.append(f"PAGES: exactly {contract.page_count}")
    lines.append(f"PAGE NAMES: {contract.page_names}")
    lines.append("")

    # Layout constraints
    if contract.layout_constraints:
        lines.append("LAYOUT CONSTRAINTS:")
        for lc in contract.layout_constraints:
            lines.append(f"  - {lc.constraint_type}: {lc.value} (source: {lc.source})")
        lines.append("")

    # Filters
    if contract.top_filters:
        lines.append("REQUIRED SLICERS/FILTERS:")
        for f in contract.top_filters:
            lines.append(f"  - field=\"{f.field}\" display_as={f.display_as} position={f.position}")
        lines.append("")

    # Navigation
    if contract.navigation_elements:
        lines.append("NAVIGATION ELEMENTS:")
        for ne in contract.navigation_elements:
            lines.append(f"  - {ne.element_type}: {ne.notes}")
        lines.append("")

    # Visuals
    if contract.visuals:
        lines.append(f"REQUIRED VISUALS ({len(contract.visuals)} total):")
        for v in contract.visuals:
            parts = [f"  [{v.visual_id}] page=\"{v.page}\""]
            parts.append(f"type={v.visual_type.value}")
            if v.title:
                parts.append(f"title=\"{v.title}\"")
            if v.category_axis:
                parts.append(f"category={v.category_axis.field}")
            if v.value_axis:
                parts.append(f"value={v.value_axis.field}({v.value_axis.aggregation})")
            if v.color_field:
                parts.append(f"color={v.color_field}")
            parts.append(f"order={v.stacking_order}")
            parts.append(f"width={v.width_pct}%")
            parts.append(f"[{v.requirement.value}]")
            if v.confidence != Confidence.EXACT:
                parts.append(f"confidence={v.confidence.value}")
            lines.append(" ".join(parts))
        lines.append("")

    # Unsupported
    if contract.unsupported_items:
        lines.append("UNSUPPORTED ITEMS (include in migration_warnings):")
        for u in contract.unsupported_items:
            lines.append(f"  - {u.item}: {u.pbi_alternative}")
        lines.append("")

    # Manual review
    if contract.manual_review_items:
        lines.append("AMBIGUITIES (use best judgment, flag in executive_summary):")
        for mr in contract.manual_review_items:
            lines.append(f"  - [{mr.severity}] {mr.item}: {mr.reason}")
        lines.append("")

    lines.append("=== END CONTRACT ===")
    return "\n".join(lines)
