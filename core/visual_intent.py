"""
Visual Intent Model -- structured extraction of user + Tableau intent.

Bridges the gap between raw Tableau parsing and prompt construction by
producing structured intent objects that downstream prompts can rely on.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# -- Tableau -> PBI visual type mapping with fidelity metadata --

TABLEAU_TO_PBI_MAP: Dict[str, dict] = {
    "bar": {"pbi_type": "clusteredBarChart", "fidelity": "exact"},
    "stacked-bar": {"pbi_type": "stackedBarChart", "fidelity": "exact"},
    "line": {"pbi_type": "lineChart", "fidelity": "exact"},
    "area": {"pbi_type": "areaChart", "fidelity": "exact"},
    "map": {"pbi_type": "map", "fidelity": "exact"},
    "filled-map": {"pbi_type": "filledMap", "fidelity": "exact"},
    "text": {"pbi_type": "tableEx", "fidelity": "exact"},
    "text-table": {"pbi_type": "tableEx", "fidelity": "exact"},
    "crosstab": {"pbi_type": "matrix", "fidelity": "exact"},
    "scatter": {"pbi_type": "scatterChart", "fidelity": "exact"},
    "pie": {"pbi_type": "pieChart", "fidelity": "exact"},
    "treemap": {"pbi_type": "treemap", "fidelity": "exact"},
    "heatmap": {
        "pbi_type": "matrix",
        "fidelity": "approximate",
        "note": "PBI matrix with conditional formatting approximates a heatmap.",
    },
    "highlight-table": {
        "pbi_type": "matrix",
        "fidelity": "approximate",
        "note": "Highlight table maps to PBI matrix with conditional formatting.",
    },
    "dual-axis": {
        "pbi_type": "lineClusteredColumnComboChart",
        "fidelity": "approximate",
        "note": "Dual-axis becomes a combo chart. Independent axis scaling not supported.",
    },
    "combo": {"pbi_type": "lineClusteredColumnComboChart", "fidelity": "exact"},
    "histogram": {
        "pbi_type": "clusteredBarChart",
        "fidelity": "approximate",
        "note": "PBI has no native histogram. Binned bar chart is the closest equivalent.",
    },
    "box-plot": {
        "pbi_type": "clusteredBarChart",
        "fidelity": "lossy",
        "note": "PBI has no native box plot. Consider the Box and Whisker custom visual.",
    },
    "gantt": {
        "pbi_type": "clusteredBarChart",
        "fidelity": "lossy",
        "note": "PBI has no native Gantt. Consider the Gantt custom visual from AppSource.",
    },
    "bullet": {
        "pbi_type": "clusteredBarChart",
        "fidelity": "lossy",
        "note": "PBI has no native bullet chart. Consider the Bullet Chart custom visual.",
    },
    "waterfall": {"pbi_type": "waterfallChart", "fidelity": "exact"},
    "funnel": {"pbi_type": "funnelChart", "fidelity": "exact"},
    "donut": {"pbi_type": "donutChart", "fidelity": "exact"},
    "circle": {"pbi_type": "scatterChart", "fidelity": "approximate",
               "note": "Circle marks become scatter plot in PBI."},
    "shape": {"pbi_type": "scatterChart", "fidelity": "approximate",
              "note": "Shape marks become scatter plot in PBI."},
    "square": {"pbi_type": "treemap", "fidelity": "approximate",
               "note": "Square marks mapped to treemap."},
    "automatic": {"pbi_type": "clusteredBarChart", "fidelity": "inferred",
                  "note": "Chart type was 'Automatic' in Tableau -- defaulted to bar chart."},
}


@dataclass
class VisualIntent:
    """Structured intent for a single visual (derived from one Tableau worksheet)."""
    source_name: str
    chart_type_tableau: str
    chart_type_pbi: str
    fidelity: str  # "exact", "approximate", "lossy", "inferred"
    fidelity_note: str = ""
    rows_fields: List[str] = field(default_factory=list)
    cols_fields: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    measures: List[str] = field(default_factory=list)
    filters: List[dict] = field(default_factory=list)
    is_slicer: bool = False


@dataclass
class ZoneIntent:
    """Spatial layout intent for a single zone in a dashboard."""
    name: str
    zone_type: str  # worksheet, filter, blank, text, etc.
    x: Optional[int] = None
    y: Optional[int] = None
    w: Optional[int] = None
    h: Optional[int] = None


@dataclass
class PageIntent:
    """Structured intent for a single PBI page (derived from one Tableau dashboard)."""
    source_name: str
    display_name: str
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    visuals: List[VisualIntent] = field(default_factory=list)
    zones: List[ZoneIntent] = field(default_factory=list)
    slicers: List[dict] = field(default_factory=list)


@dataclass
class MigrationIntent:
    """Full migration intent combining user request + Tableau structure."""
    user_request: str
    requested_page_count: Optional[int] = None
    pages: List[PageIntent] = field(default_factory=list)
    orphan_worksheets: List[VisualIntent] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    calculated_fields: List[dict] = field(default_factory=list)
    parameters: List[dict] = field(default_factory=list)


def _parse_int(val) -> Optional[int]:
    """Safely parse an int from a string or int value."""
    if val is None or val == "":
        return None
    try:
        return int(str(val).replace("px", ""))
    except (ValueError, TypeError):
        return None


def _parse_page_count_from_request(user_request: str) -> Optional[int]:
    """Extract explicit page count from user request text.

    Matches patterns like:
    - "3 pages", "three pages", "a single page"
    - "multi-page", "one page"
    """
    if not user_request:
        return None

    text = user_request.lower()

    # "N pages" or "N-page"
    m = re.search(r'(\d+)\s*[-\s]?pages?', text)
    if m:
        return int(m.group(1))

    word_map = {
        "one": 1, "single": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    m = re.search(r'(one|single|two|three|four|five|six|seven|eight|nine|ten)\s*[-\s]?pages?', text)
    if m:
        return word_map.get(m.group(1), None)

    if "multi-page" in text or "multiple pages" in text:
        return None  # Indicates multi-page but no specific count

    return None


def extract_migration_intent(tableau_spec: dict, user_request: str) -> MigrationIntent:
    """Build a structured MigrationIntent from parsed Tableau spec + user request.

    This is the central function that maps Tableau structure to PBI intent,
    detecting chart types, layout, slicers, and surfacing unsupported features.
    """
    intent = MigrationIntent(
        user_request=user_request,
        requested_page_count=_parse_page_count_from_request(user_request),
        calculated_fields=tableau_spec.get("calculated_fields", []),
        parameters=tableau_spec.get("parameters", []),
    )

    # Build a lookup of worksheet name -> VisualIntent
    ws_lookup: Dict[str, VisualIntent] = {}
    for ws in tableau_spec.get("worksheets", []):
        name = ws.get("name", "")
        chart_type = ws.get("chart_type", "automatic")
        mapping = TABLEAU_TO_PBI_MAP.get(chart_type, TABLEAU_TO_PBI_MAP["automatic"])

        vi = VisualIntent(
            source_name=name,
            chart_type_tableau=chart_type,
            chart_type_pbi=mapping["pbi_type"],
            fidelity=mapping["fidelity"],
            fidelity_note=mapping.get("note", ""),
            rows_fields=ws.get("rows_fields", []),
            cols_fields=ws.get("cols_fields", []),
            dimensions=ws.get("dimensions", []),
            measures=ws.get("measures", []),
            filters=ws.get("filters", []),
        )

        if vi.fidelity in ("lossy", "approximate") and vi.fidelity_note:
            intent.warnings.append(
                f"Worksheet '{name}': {vi.fidelity_note}"
            )

        ws_lookup[name] = vi

    # Build pages from dashboards
    assigned_worksheets = set()

    for db in tableau_spec.get("dashboards", []):
        db_name = db.get("name", "Dashboard")
        size = db.get("size", {})

        page = PageIntent(
            source_name=db_name,
            display_name=db_name,
            source_width=_parse_int(size.get("width")),
            source_height=_parse_int(size.get("height")),
        )

        # Process zones
        for zone_data in db.get("zones", []):
            zone_name = zone_data if isinstance(zone_data, str) else zone_data.get("name", "")
            zone_type = "" if isinstance(zone_data, str) else zone_data.get("type", "")

            zi = ZoneIntent(
                name=zone_name,
                zone_type=zone_type,
                x=_parse_int(zone_data.get("x") if isinstance(zone_data, dict) else None),
                y=_parse_int(zone_data.get("y") if isinstance(zone_data, dict) else None),
                w=_parse_int(zone_data.get("w") if isinstance(zone_data, dict) else None),
                h=_parse_int(zone_data.get("h") if isinstance(zone_data, dict) else None),
            )
            page.zones.append(zi)

            # If this zone references a worksheet, add it as a visual
            if zone_name in ws_lookup:
                vi = ws_lookup[zone_name]
                page.visuals.append(vi)
                assigned_worksheets.add(zone_name)

                # Check if any filters on this worksheet should be slicers
                for f in vi.filters:
                    page.slicers.append({
                        "field": f.get("field", ""),
                        "type": "dropdown" if f.get("type") == "categorical" else "date_range",
                    })

        intent.pages.append(page)

    # Worksheets not assigned to any dashboard -> orphans
    for name, vi in ws_lookup.items():
        if name not in assigned_worksheets:
            intent.orphan_worksheets.append(vi)

    # If there are orphan worksheets and no dashboards, create a page per orphan
    if intent.orphan_worksheets and not intent.pages:
        for vi in intent.orphan_worksheets:
            page = PageIntent(
                source_name=vi.source_name,
                display_name=vi.source_name,
                visuals=[vi],
            )
            intent.pages.append(page)
        intent.orphan_worksheets = []

    # Respect explicit page count from user if it differs
    if intent.requested_page_count and intent.requested_page_count != len(intent.pages):
        intent.warnings.append(
            f"User requested {intent.requested_page_count} pages but Tableau "
            f"has {len(intent.pages)} dashboards. User preference takes priority."
        )

    return intent


def format_intent_for_prompt(intent: MigrationIntent, table_name: str) -> str:
    """Format MigrationIntent as a structured prompt section for Claude.

    Produces a clear, machine-parseable representation that the LLM can
    use to generate a faithful Power BI dashboard spec.
    """
    parts = []

    # User request at the top (highest prompt weight)
    parts.append(
        f"USER REQUEST:\n{intent.user_request}\n"
    )

    # Explicit page count instruction
    if intent.requested_page_count:
        parts.append(
            f"PAGE COUNT: The user explicitly requested {intent.requested_page_count} pages. "
            f"You MUST produce exactly {intent.requested_page_count} pages."
        )
    elif intent.pages:
        parts.append(
            f"PAGE COUNT: The Tableau workbook has {len(intent.pages)} dashboards. "
            f"Create {len(intent.pages)} pages to match, unless the user says otherwise."
        )

    # Page-by-page layout
    for pidx, page in enumerate(intent.pages):
        lines = [f"\nPAGE {pidx + 1}: \"{page.display_name}\""]

        if page.source_width and page.source_height:
            lines.append(
                f"  Source size: {page.source_width}x{page.source_height} "
                f"-> map proportionally to PBI 1280x720"
            )

        for vidx, vi in enumerate(page.visuals):
            line = (
                f"  VISUAL {vidx + 1}: \"{vi.source_name}\" "
                f"[Tableau {vi.chart_type_tableau} -> PBI {vi.chart_type_pbi}]"
            )
            if vi.fidelity != "exact":
                line += f" (fidelity: {vi.fidelity})"
            lines.append(line)

            if vi.rows_fields:
                lines.append(f"    rows: {vi.rows_fields}")
            if vi.cols_fields:
                lines.append(f"    cols: {vi.cols_fields}")
            if vi.filters:
                filter_names = [f.get("field", "?") for f in vi.filters]
                lines.append(f"    filters: {filter_names}")

        # Zones with spatial layout
        spatial_zones = [z for z in page.zones if z.x is not None]
        if spatial_zones:
            lines.append("  LAYOUT ZONES (source coordinates):")
            for z in spatial_zones:
                lines.append(
                    f"    \"{z.name}\": x={z.x}, y={z.y}, w={z.w}, h={z.h}"
                )

        # Slicers
        if page.slicers:
            slicer_fields = list({s["field"] for s in page.slicers})
            lines.append(f"  SLICERS: {slicer_fields}")

        parts.append("\n".join(lines))

    # Orphan worksheets (not on any dashboard)
    if intent.orphan_worksheets:
        orphan_lines = ["\nORPHAN WORKSHEETS (not placed on any Tableau dashboard):"]
        for vi in intent.orphan_worksheets:
            orphan_lines.append(
                f"  - \"{vi.source_name}\": {vi.chart_type_tableau} -> PBI {vi.chart_type_pbi}"
            )
        orphan_lines.append(
            "  Place these on additional pages or merge into existing pages as appropriate."
        )
        parts.append("\n".join(orphan_lines))

    # Calculated fields
    if intent.calculated_fields:
        calc_lines = ["\nCALCULATED FIELDS (convert to DAX):"]
        for cf in intent.calculated_fields[:30]:
            if isinstance(cf, dict):
                name = cf.get("name", "?")
                formula = cf.get("formula", "?")
                calc_lines.append(f"  - \"{name}\": {formula}")
        parts.append("\n".join(calc_lines))

    # Warnings about lossy mappings
    if intent.warnings:
        warn_lines = ["\nMIGRATION WARNINGS (communicate these to the user):"]
        for w in intent.warnings:
            warn_lines.append(f"  - {w}")
        parts.append("\n".join(warn_lines))

    return "\n\n".join(parts)
