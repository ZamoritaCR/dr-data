"""
Comprehensive Tableau Workbook Extractor.

Parses TWB XML to extract BOTH the data/semantic side AND the visual/layout side.
Produces a normalized TableauWorkbook spec that downstream stages can use to
faithfully reconstruct the dashboard in Power BI.

Covers:
- Worksheets with chart types, shelf encodings, axes, legends, filters
- Dashboards with zone trees, spatial layout, stacking, alignment
- Calculated fields, parameters, data sources with typed columns
- KPI/card detection, text blocks, tab-like selector patterns
- Confidence scoring per visual mapping
"""

import re
import defusedxml.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from core.utils import remove_tableau_brackets
from enum import Enum


# ================================================================== #
#  Confidence levels                                                    #
# ================================================================== #

class Confidence(str, Enum):
    EXACT = "exact"
    HIGH = "high_confidence"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown_manual_review"


# ================================================================== #
#  Normalized spec model                                                #
# ================================================================== #

@dataclass
class ColumnRef:
    """A reference to a data column as used in a visual."""
    name: str                       # Clean column name (e.g., "Sales")
    raw_ref: str = ""               # Original Tableau ref (e.g., "[sum:Sales:qk]")
    datatype: str = "string"        # string, integer, real, date, boolean
    role: str = "dimension"         # dimension, measure
    aggregation: str = ""           # Sum, Avg, CountD, Year, Month, None, Attribute
    tableau_type: str = ""          # nominal, ordinal, quantitative


@dataclass
class EncodingSpec:
    """How data maps to visual properties."""
    shelf_rows: List[ColumnRef] = field(default_factory=list)
    shelf_cols: List[ColumnRef] = field(default_factory=list)
    color: Optional[ColumnRef] = None
    size: Optional[ColumnRef] = None
    label: Optional[ColumnRef] = None
    detail: List[ColumnRef] = field(default_factory=list)
    tooltip: List[ColumnRef] = field(default_factory=list)
    path: Optional[ColumnRef] = None
    rows_expression: str = ""       # Raw shelf expression
    cols_expression: str = ""


@dataclass
class FilterSpec:
    """A filter applied to a worksheet or dashboard."""
    field: str
    raw_ref: str = ""
    filter_type: str = "categorical"  # categorical, quantitative, date, relative_date
    mode: str = ""                    # checkdropdown, compact, slider, etc.
    is_quick_filter: bool = False
    title: str = ""
    values: List[str] = field(default_factory=list)


@dataclass
class SortSpec:
    """Sort configuration."""
    dimension: str
    measure: str = ""
    direction: str = "DESC"
    shelf: str = ""


@dataclass
class VisualSpec:
    """Normalized spec for a single Tableau worksheet/visual."""
    name: str
    mark_type: str = "automatic"      # bar, line, area, circle, square, polygon, text, map, pie, gantt
    mark_type_pbi: str = ""           # Mapped PBI visual type
    confidence: Confidence = Confidence.HIGH

    # Encodings
    encoding: EncodingSpec = field(default_factory=EncodingSpec)

    # Filters
    filters: List[FilterSpec] = field(default_factory=list)
    slices: List[str] = field(default_factory=list)   # Filter shelf columns

    # Sort
    sorts: List[SortSpec] = field(default_factory=list)

    # Data context
    datasource: str = ""
    columns_used: List[ColumnRef] = field(default_factory=list)

    # Style hints
    title: str = ""
    title_visible: bool = True
    legend_title: str = ""
    legend_field: str = ""

    # Classification
    is_kpi_card: bool = False
    is_text_table: bool = False
    is_map: bool = False
    is_breadcrumb: bool = False     # Tableau "bread crumb" navigation pattern

    # Notes
    notes: List[str] = field(default_factory=list)


@dataclass
class ZoneSpec:
    """A single zone in a dashboard layout."""
    zone_id: str = ""
    zone_type: str = ""          # ws, filter, paramctrl, text, title, color, empty, layout-flow, layout-basic
    name: str = ""               # Worksheet name or empty
    param: str = ""              # Filter/param field ref, or flow direction (vert/horz)

    # Position in 100000-unit coordinate space
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    # Proportional position on PBI 1280x720 canvas
    pbi_x: int = 0
    pbi_y: int = 0
    pbi_w: int = 0
    pbi_h: int = 0

    # Layout properties
    is_fixed: bool = False
    fixed_size: int = 0
    is_hidden: bool = False
    show_title: bool = True
    layout_strategy: str = ""    # distribute-evenly, etc.
    flow_direction: str = ""     # vert, horz (for layout-flow)

    # Nested children
    children: List["ZoneSpec"] = field(default_factory=list)

    # Filter/param specifics
    filter_mode: str = ""        # checkdropdown, compact, slider, type_in


@dataclass
class DashboardSpec:
    """Normalized spec for a Tableau dashboard (= one PBI page)."""
    name: str
    width: int = 800
    height: int = 600
    sizing_mode: str = "automatic"    # fixed, automatic, range

    # Zone tree (root zone)
    root_zone: Optional[ZoneSpec] = None

    # Flattened lists for quick access
    worksheet_zones: List[ZoneSpec] = field(default_factory=list)
    filter_zones: List[ZoneSpec] = field(default_factory=list)
    param_zones: List[ZoneSpec] = field(default_factory=list)
    text_zones: List[ZoneSpec] = field(default_factory=list)
    title_zone: Optional[ZoneSpec] = None

    # Layout analysis
    has_vertical_stack: bool = False
    has_horizontal_split: bool = False
    has_tab_selector: bool = False
    has_kpi_row: bool = False
    layout_pattern: str = ""         # "full-width-stacked", "sidebar-filters", "grid", etc.

    # Notes
    notes: List[str] = field(default_factory=list)


@dataclass
class ParameterSpec:
    """A Tableau parameter."""
    name: str
    datatype: str = ""
    domain_type: str = ""       # range, list, all
    current_value: str = ""
    min_value: str = ""
    max_value: str = ""
    step: str = ""
    allowable_values: List[str] = field(default_factory=list)


@dataclass
class CalculatedFieldSpec:
    """A Tableau calculated field."""
    name: str
    formula: str
    datasource: str = ""
    return_type: str = ""       # string, integer, real, date, boolean
    is_table_calc: bool = False
    is_lod: bool = False        # FIXED/INCLUDE/EXCLUDE
    referenced_fields: List[str] = field(default_factory=list)


@dataclass
class DatasourceSpec:
    """A Tableau data source with typed columns."""
    name: str
    caption: str = ""
    connection_type: str = ""
    tables: List[str] = field(default_factory=list)
    columns: List[ColumnRef] = field(default_factory=list)
    server: str = ""
    database: str = ""
    filename: str = ""


@dataclass
class TableauWorkbook:
    """Top-level normalized spec for an entire Tableau workbook."""
    version: str = ""
    worksheets: List[VisualSpec] = field(default_factory=list)
    dashboards: List[DashboardSpec] = field(default_factory=list)
    datasources: List[DatasourceSpec] = field(default_factory=list)
    calculated_fields: List[CalculatedFieldSpec] = field(default_factory=list)
    parameters: List[ParameterSpec] = field(default_factory=list)
    relationships: List[dict] = field(default_factory=list)

    # Summary
    total_visuals: int = 0
    total_filters: int = 0
    total_measures: int = 0
    total_dimensions: int = 0

    # Extraction metadata
    extraction_notes: List[str] = field(default_factory=list)


# ================================================================== #
#  Mark type -> PBI mapping with confidence                             #
# ================================================================== #

_MARK_TO_PBI = {
    "automatic": ("clusteredBarChart", Confidence.APPROXIMATE),
    "bar": ("clusteredBarChart", Confidence.EXACT),
    "stacked bar": ("stackedBarChart", Confidence.EXACT),
    "line": ("lineChart", Confidence.EXACT),
    "area": ("areaChart", Confidence.EXACT),
    "circle": ("scatterChart", Confidence.HIGH),
    "square": ("treemap", Confidence.APPROXIMATE),
    "text": ("tableEx", Confidence.EXACT),
    "polygon": ("map", Confidence.HIGH),
    "map": ("map", Confidence.EXACT),
    "pie": ("pieChart", Confidence.EXACT),
    "gantt": ("clusteredBarChart", Confidence.APPROXIMATE),
    "shape": ("scatterChart", Confidence.APPROXIMATE),
    "density": ("scatterChart", Confidence.APPROXIMATE),
}


# ================================================================== #
#  Field reference parsing                                              #
# ================================================================== #

_FIELD_REF_RE = re.compile(r'\[([^\]]+)\]')
_DERIVATION_RE = re.compile(
    r'^(sum|avg|cnt|ctd|min|max|attr|none|yr|qr|mn|dy|hr|wk|md|tmn|tyr):'
    r'(.+?):(nk|ok|qk)$', re.IGNORECASE
)
_AGG_MAP = {
    "sum": "Sum", "avg": "Average", "cnt": "Count", "ctd": "CountD",
    "min": "Min", "max": "Max", "attr": "Attribute", "none": "None",
    "yr": "Year", "qr": "Quarter", "mn": "Month", "dy": "Day",
    "hr": "Hour", "wk": "Week", "md": "MonthDay", "tmn": "TruncMonth",
    "tyr": "TruncYear",
}
_TYPE_MAP = {"nk": "nominal", "ok": "ordinal", "qk": "quantitative"}


def parse_column_instance(name_attr: str, column_attr: str = "",
                          derivation_attr: str = "", type_attr: str = "") -> ColumnRef:
    """Parse a Tableau column-instance element into a ColumnRef."""
    # Try parsing the encoded name like [sum:Sales:qk]
    clean_name = remove_tableau_brackets(name_attr)
    m = _DERIVATION_RE.match(clean_name)

    if m:
        agg_code, field_name, type_code = m.groups()
        return ColumnRef(
            name=field_name,
            raw_ref=name_attr,
            aggregation=_AGG_MAP.get(agg_code.lower(), agg_code),
            tableau_type=_TYPE_MAP.get(type_code, type_code),
            role="measure" if type_code == "qk" else "dimension",
        )

    # Fallback: use column attribute and derivation
    col_name = remove_tableau_brackets(column_attr) if column_attr else clean_name
    return ColumnRef(
        name=col_name,
        raw_ref=name_attr,
        aggregation=derivation_attr or "None",
        tableau_type=type_attr or "",
        role="measure" if derivation_attr in ("Sum", "Avg", "Count", "CountD", "Min", "Max")
             else "dimension",
    )


def parse_shelf_expression(text: str) -> List[ColumnRef]:
    """Parse a rows/cols shelf expression into ColumnRefs.

    Handles patterns like:
    - [datasource].[none:Category:nk]
    - ([ds].[field1] / [ds].[field2])
    - [ds].[sum:Sales:qk]
    """
    if not text:
        return []

    refs = []
    # Find all [datasource].[field] patterns
    pairs = re.findall(r'\[([^\]]+)\]\.\[([^\]]+)\]', text)
    for ds, field_ref in pairs:
        ref = parse_column_instance(field_ref)
        refs.append(ref)

    return refs


# ================================================================== #
#  Main extractor                                                       #
# ================================================================== #

def extract_workbook(twb_path_or_bytes) -> TableauWorkbook:
    """Extract a complete normalized spec from a TWB file.

    Args:
        twb_path_or_bytes: path to .twb file, or bytes/string of TWB XML

    Returns:
        TableauWorkbook with all extracted information
    """
    if isinstance(twb_path_or_bytes, bytes):
        root = ET.fromstring(twb_path_or_bytes)
    elif isinstance(twb_path_or_bytes, str) and twb_path_or_bytes.strip().startswith("<?xml"):
        root = ET.fromstring(twb_path_or_bytes)
    else:
        tree = ET.parse(twb_path_or_bytes)
        root = tree.getroot()

    wb = TableauWorkbook(version=root.get("version", ""))

    # 1. Datasources
    wb.datasources = _extract_datasources(root)

    # 2. Calculated fields + parameters (from datasources)
    wb.calculated_fields, wb.parameters = _extract_calcs_and_params(root)

    # 3. Worksheets
    wb.worksheets = _extract_worksheets(root)

    # 4. Dashboards with layout
    wb.dashboards = _extract_dashboards(root, wb.worksheets)

    # 5. Summary stats
    wb.total_visuals = len(wb.worksheets)
    wb.total_filters = sum(len(v.filters) for v in wb.worksheets)
    all_cols = []
    for ds in wb.datasources:
        all_cols.extend(ds.columns)
    wb.total_measures = sum(1 for c in all_cols if c.role == "measure")
    wb.total_dimensions = sum(1 for c in all_cols if c.role == "dimension")

    return wb


# ================================================================== #
#  Datasource extraction                                                #
# ================================================================== #

def _extract_datasources(root) -> List[DatasourceSpec]:
    """Extract all non-parameter datasources with typed columns."""
    datasources = []
    seen_names = set()

    for ds_el in root.iter("datasource"):
        ds_name = ds_el.get("name", "") or ds_el.get("caption", "")
        if not ds_name or ds_name.startswith("Parameters"):
            continue
        if ds_name in seen_names:
            continue
        seen_names.add(ds_name)

        ds = DatasourceSpec(
            name=ds_name,
            caption=ds_el.get("caption", ds_name),
        )

        # Connection info
        conn = ds_el.find(".//connection")
        if conn is not None:
            ds.connection_type = conn.get("class", "")
            ds.server = conn.get("server", "")
            ds.database = conn.get("dbname", "")
            ds.filename = conn.get("filename", "")
            for rel in conn.iter("relation"):
                t = rel.get("table", rel.get("name", ""))
                if t:
                    ds.tables.append(t)

        # Columns
        for col_el in ds_el.findall("column"):
            col_name = remove_tableau_brackets(col_el.get("caption", "") or col_el.get("name", ""))
            if not col_name or col_name.startswith(":"):
                continue
            ds.columns.append(ColumnRef(
                name=col_name,
                raw_ref=col_el.get("name", ""),
                datatype=col_el.get("datatype", "string"),
                role=col_el.get("role", "dimension"),
                tableau_type=col_el.get("type", ""),
            ))

        datasources.append(ds)

    return datasources


# ================================================================== #
#  Calculated fields + parameters                                       #
# ================================================================== #

def _extract_calcs_and_params(root) -> Tuple[List[CalculatedFieldSpec], List[ParameterSpec]]:
    """Extract calculated fields and parameters from datasource definitions."""
    calcs = []
    params = []

    for ds_el in root.iter("datasource"):
        ds_name = ds_el.get("name", "") or ds_el.get("caption", "")

        if ds_name.startswith("Parameters"):
            for col_el in ds_el.iter("column"):
                p = ParameterSpec(
                    name=col_el.get("caption", col_el.get("name", "")),
                    datatype=col_el.get("datatype", ""),
                    domain_type=col_el.get("param-domain-type", ""),
                    current_value=col_el.get("value", ""),
                )
                # Range info
                range_el = col_el.find("range")
                if range_el is not None:
                    p.min_value = range_el.get("min", "")
                    p.max_value = range_el.get("max", "")
                    p.step = range_el.get("granularity", "")
                # Allowable values
                for member in col_el.iter("member"):
                    p.allowable_values.append(member.get("value", ""))
                params.append(p)
            continue

        for col_el in ds_el.iter("column"):
            calc_el = col_el.find("calculation")
            if calc_el is None:
                continue
            formula = calc_el.get("formula", "")
            if not formula:
                continue

            name = col_el.get("caption", col_el.get("name", ""))
            cf = CalculatedFieldSpec(
                name=name,
                formula=formula,
                datasource=ds_name,
                return_type=col_el.get("datatype", ""),
                is_lod=bool(re.search(r'\{(FIXED|INCLUDE|EXCLUDE)', formula, re.IGNORECASE)),
                is_table_calc=bool(re.search(
                    r'(RUNNING_|WINDOW_|INDEX\(\)|FIRST\(\)|LAST\(\)|LOOKUP\(|RANK\()',
                    formula, re.IGNORECASE
                )),
            )
            # Extract referenced fields
            cf.referenced_fields = [
                ref for ref in _FIELD_REF_RE.findall(formula)
                if not ref.startswith("Parameters.")
            ]
            calcs.append(cf)

    return calcs, params


# ================================================================== #
#  Worksheet extraction                                                 #
# ================================================================== #

def _extract_worksheets(root) -> List[VisualSpec]:
    """Extract all worksheets with mark types, encodings, filters, sorts."""
    visuals = []

    for ws_el in root.iter("worksheet"):
        ws_name = ws_el.get("name", "")
        vs = VisualSpec(name=ws_name)

        # -- Mark type --
        mark_el = ws_el.find(".//pane/mark")
        if mark_el is None:
            mark_el = ws_el.find(".//mark")
        if mark_el is not None:
            raw_mark = mark_el.get("class", "Automatic")
            vs.mark_type = raw_mark.lower().replace(" ", "-")
        else:
            vs.mark_type = "automatic"

        # Map to PBI
        pbi_type, confidence = _MARK_TO_PBI.get(
            vs.mark_type.replace("-", " "),
            ("clusteredBarChart", Confidence.UNKNOWN)
        )
        vs.mark_type_pbi = pbi_type
        vs.confidence = confidence

        # -- Shelf expressions (rows/cols) --
        table_el = ws_el.find(".//table")
        if table_el is not None:
            rows_el = table_el.find("rows")
            cols_el = table_el.find("cols")
            if rows_el is not None and rows_el.text:
                vs.encoding.rows_expression = rows_el.text.strip()
                vs.encoding.shelf_rows = parse_shelf_expression(rows_el.text)
            if cols_el is not None and cols_el.text:
                vs.encoding.cols_expression = cols_el.text.strip()
                vs.encoding.shelf_cols = parse_shelf_expression(cols_el.text)

        # -- Datasource dependencies (typed columns used by this worksheet) --
        for dep in ws_el.iter("datasource-dependencies"):
            ds_name = dep.get("datasource", "")
            if ds_name and not vs.datasource:
                vs.datasource = ds_name

            for col_el in dep.findall("column"):
                cr = ColumnRef(
                    name=remove_tableau_brackets(col_el.get("caption", "") or col_el.get("name", "")),
                    raw_ref=col_el.get("name", ""),
                    datatype=col_el.get("datatype", "string"),
                    role=col_el.get("role", "dimension"),
                    tableau_type=col_el.get("type", ""),
                )
                vs.columns_used.append(cr)

            for ci in dep.findall("column-instance"):
                ref = parse_column_instance(
                    ci.get("name", ""),
                    ci.get("column", ""),
                    ci.get("derivation", ""),
                    ci.get("type", ""),
                )
                # Avoid duplicates
                if ref.name not in {c.name for c in vs.columns_used}:
                    vs.columns_used.append(ref)

        # -- Pane encodings (color, size, tooltip, detail, lod) --
        for pane in ws_el.iter("pane"):
            encodings_el = pane.find("encodings")
            if encodings_el is None:
                continue
            for enc in encodings_el:
                col_attr = enc.get("column", "")
                if not col_attr:
                    continue
                ref = parse_column_instance(remove_tableau_brackets(col_attr.split(".")[-1]))

                tag = enc.tag.lower()
                if tag == "color":
                    vs.encoding.color = ref
                    vs.legend_field = ref.name
                elif tag == "size":
                    vs.encoding.size = ref
                elif tag == "text" or tag == "label":
                    vs.encoding.label = ref
                elif tag == "detail" or tag == "lod":
                    vs.encoding.detail.append(ref)
                elif tag == "tooltip":
                    vs.encoding.tooltip.append(ref)
                elif tag == "path":
                    vs.encoding.path = ref

        # -- Slices (filter shelf) --
        for slices_el in ws_el.iter("slices"):
            for col_el in slices_el.findall("column"):
                if col_el.text:
                    vs.slices.append(col_el.text.strip())

        # -- Filters --
        for filt_el in ws_el.iter("filter"):
            col = filt_el.get("column", "")
            if not col:
                continue
            # Extract clean field name
            field_match = re.search(r'\[([^\]]+)\]$', col)
            field_name = field_match.group(1) if field_match else col
            # Parse encoded name
            parsed = parse_column_instance(field_name)

            vs.filters.append(FilterSpec(
                field=parsed.name,
                raw_ref=col,
                filter_type=filt_el.get("class", "categorical"),
            ))

        # -- Sorts --
        for sort_el in ws_el.iter("shelf-sort-v2"):
            dim_ref = sort_el.get("dimension-to-sort", "")
            meas_ref = sort_el.get("measure-to-sort-by", "")
            vs.sorts.append(SortSpec(
                dimension=remove_tableau_brackets(dim_ref.split(".")[-1]) if dim_ref else "",
                measure=remove_tableau_brackets(meas_ref.split(".")[-1]) if meas_ref else "",
                direction=sort_el.get("direction", "DESC"),
                shelf=sort_el.get("shelf", ""),
            ))

        # -- Style hints (titles, legend titles) --
        for rule in ws_el.iter("style-rule"):
            element = rule.get("element", "")
            if element == "quick-filter":
                for fmt in rule.findall("format"):
                    if fmt.get("attr") == "title":
                        field_ref = fmt.get("field", "")
                        title_val = fmt.get("value", "")
                        # Match to existing filter
                        for f in vs.filters:
                            if f.raw_ref and field_ref and f.raw_ref in field_ref:
                                f.title = title_val
                                f.is_quick_filter = True
            elif element == "legend-title-text":
                for fmt in rule.findall("format"):
                    if fmt.get("attr") == "color":
                        vs.legend_title = fmt.get("value", "")

        # -- Classification --
        _classify_visual(vs)

        visuals.append(vs)

    return visuals


def _classify_visual(vs: VisualSpec):
    """Classify a visual as KPI card, text table, map, breadcrumb, etc."""
    name_lower = vs.name.lower()

    # Map detection
    if vs.mark_type in ("polygon", "map"):
        vs.is_map = True
        vs.mark_type_pbi = "map"
        vs.confidence = Confidence.HIGH
        return  # Maps are fully classified

    # Text table / crosstab
    if vs.mark_type == "text" and vs.encoding.shelf_rows and vs.encoding.shelf_cols:
        vs.is_text_table = True
        vs.mark_type_pbi = "matrix" if len(vs.encoding.shelf_rows) > 1 else "tableEx"
        vs.confidence = Confidence.EXACT

    # KPI card: single measure, no rows/cols, mark=text or automatic
    # Skip if already classified as map or text table
    if vs.is_map or vs.is_text_table:
        return
    if (not vs.encoding.shelf_rows and not vs.encoding.shelf_cols
            and len([c for c in vs.columns_used if c.role == "measure"]) <= 2):
        vs.is_kpi_card = True
        vs.mark_type_pbi = "card"
        vs.confidence = Confidence.HIGH
        return

    # Breadcrumb pattern: name contains "bread crumb" or formula uses
    # {EXCLUDE ... :COUNTD} pattern
    if "bread" in name_lower and "crumb" in name_lower:
        vs.is_breadcrumb = True
        vs.mark_type_pbi = "card"
        vs.confidence = Confidence.HIGH
        vs.notes.append("Tableau breadcrumb navigation pattern -- "
                        "map to PBI card showing current filter context")

    # Refine bar chart: check if horizontal
    if vs.mark_type in ("automatic", "bar"):
        if (vs.encoding.shelf_rows
                and any(c.role == "dimension" for c in vs.encoding.shelf_rows)
                and vs.encoding.shelf_cols
                and any(c.role == "measure" for c in vs.encoding.shelf_cols)):
            vs.mark_type_pbi = "clusteredBarChart"
            vs.confidence = Confidence.EXACT
        elif (vs.encoding.shelf_cols
              and any(c.role == "dimension" for c in vs.encoding.shelf_cols)
              and vs.encoding.shelf_rows
              and any(c.role == "measure" for c in vs.encoding.shelf_rows)):
            vs.mark_type_pbi = "clusteredColumnChart"
            vs.confidence = Confidence.EXACT


# ================================================================== #
#  Dashboard extraction + layout analysis                               #
# ================================================================== #

def _extract_dashboards(root, worksheets: List[VisualSpec]) -> List[DashboardSpec]:
    """Extract dashboards with full zone tree and layout analysis."""
    ws_lookup = {vs.name: vs for vs in worksheets}
    dashboards = []

    for db_el in root.iter("dashboard"):
        db_name = db_el.get("name", "")
        db = DashboardSpec(name=db_name)

        # Size
        size_el = db_el.find("size")
        if size_el is not None:
            db.width = _int(size_el.get("maxwidth", size_el.get("minwidth", "800")))
            db.height = _int(size_el.get("maxheight", size_el.get("minheight", "600")))
            db.sizing_mode = size_el.get("sizing-mode", "automatic")

        # Parse zone tree
        zones_el = db_el.find("zones")
        if zones_el is not None:
            for child in zones_el:
                if child.tag == "zone":
                    db.root_zone = _parse_zone_tree(child, db.width, db.height)
                    break

        # Flatten zones for quick access
        if db.root_zone:
            _flatten_zones(db.root_zone, db)

        # Layout analysis
        _analyze_layout(db, ws_lookup)

        dashboards.append(db)

    return dashboards


def _parse_zone_tree(zone_el, canvas_w: int, canvas_h: int) -> ZoneSpec:
    """Recursively parse a zone element into a ZoneSpec tree."""
    attrs = zone_el.attrib

    zone_type = attrs.get("type-v2", attrs.get("type", ""))
    name = attrs.get("name", "")

    # If no explicit type but has a name that matches a worksheet -> ws type
    if not zone_type and name:
        zone_type = "ws"

    zs = ZoneSpec(
        zone_id=attrs.get("id", ""),
        zone_type=zone_type,
        name=name,
        param=attrs.get("param", ""),
        x=_int(attrs.get("x", "0")),
        y=_int(attrs.get("y", "0")),
        w=_int(attrs.get("w", "0")),
        h=_int(attrs.get("h", "0")),
        is_fixed=attrs.get("is-fixed", "") == "true",
        fixed_size=_int(attrs.get("fixed-size", "0")),
        is_hidden=attrs.get("hidden-by-user", "") == "true",
        show_title=attrs.get("show-title", "true") != "false",
        layout_strategy=attrs.get("layout-strategy-id", ""),
        filter_mode=attrs.get("mode", ""),
    )

    # Flow direction for layout-flow zones
    if zone_type == "layout-flow":
        zs.flow_direction = attrs.get("param", "")

    # Map to PBI canvas (100000 units -> actual pixels)
    if canvas_w > 0 and canvas_h > 0:
        zs.pbi_x = round(zs.x / 100000 * 1280)
        zs.pbi_y = round(zs.y / 100000 * 720)
        zs.pbi_w = round(zs.w / 100000 * 1280)
        zs.pbi_h = round(zs.h / 100000 * 720)

    # Recurse children
    for child in zone_el:
        if child.tag == "zone":
            zs.children.append(_parse_zone_tree(child, canvas_w, canvas_h))

    return zs


def _flatten_zones(zone: ZoneSpec, db: DashboardSpec):
    """Flatten zone tree into categorized lists on the DashboardSpec."""
    zt = zone.zone_type

    if zt == "ws" or (not zt and zone.name):
        if not zone.is_hidden:
            db.worksheet_zones.append(zone)
    elif zt == "filter":
        db.filter_zones.append(zone)
    elif zt == "paramctrl":
        db.param_zones.append(zone)
    elif zt == "text":
        db.text_zones.append(zone)
    elif zt == "title":
        db.title_zone = zone

    for child in zone.children:
        _flatten_zones(child, db)


def _analyze_layout(db: DashboardSpec, ws_lookup: Dict[str, VisualSpec]):
    """Analyze the zone tree to detect layout patterns."""
    if not db.root_zone:
        return

    # Detect vertical/horizontal stacking
    flows = []
    _collect_flows(db.root_zone, flows)

    vert_flows = [f for f in flows if f.flow_direction == "vert"]
    horz_flows = [f for f in flows if f.flow_direction == "horz"]

    db.has_vertical_stack = len(vert_flows) > 0
    db.has_horizontal_split = len(horz_flows) > 0

    # Detect KPI row: horizontal flow with small height containing worksheet zones
    for flow in horz_flows:
        ws_children = [c for c in flow.children if c.zone_type in ("ws", "") and c.name]
        if len(ws_children) >= 2:
            # Check if these are small (KPI-sized)
            avg_h = sum(c.h for c in ws_children) / len(ws_children) if ws_children else 0
            if avg_h < 20000:  # Less than 20% of canvas height
                db.has_kpi_row = True
                for c in ws_children:
                    if c.name in ws_lookup:
                        vs = ws_lookup[c.name]
                        # Don't override maps or other already-classified visuals
                        if not vs.is_map and not vs.is_text_table:
                            vs.is_kpi_card = True
                            vs.mark_type_pbi = "card"

    # Detect sidebar filter pattern: filters stacked vertically on one side
    if db.filter_zones:
        filter_xs = [f.x for f in db.filter_zones]
        if filter_xs:
            # All filters on same x -> sidebar
            if max(filter_xs) - min(filter_xs) < 5000:
                avg_filter_x = sum(filter_xs) / len(filter_xs)
                if avg_filter_x > 75000:
                    db.layout_pattern = "right-sidebar-filters"
                elif avg_filter_x < 25000:
                    db.layout_pattern = "left-sidebar-filters"

    # Detect tab selector pattern: hidden zones with show/hide actions
    hidden_zones = []
    _collect_hidden(db.root_zone, hidden_zones)
    if hidden_zones:
        db.has_tab_selector = True
        db.notes.append(
            f"{len(hidden_zones)} hidden zone(s) detected -- "
            f"likely tab-like selector pattern"
        )

    # Overall pattern
    if not db.layout_pattern:
        if db.has_vertical_stack and not db.has_horizontal_split:
            db.layout_pattern = "full-width-stacked"
        elif db.has_vertical_stack and db.has_horizontal_split:
            db.layout_pattern = "grid"
        elif db.has_horizontal_split:
            db.layout_pattern = "side-by-side"
        else:
            db.layout_pattern = "single"


def _collect_flows(zone: ZoneSpec, flows: list):
    """Collect all layout-flow zones."""
    if zone.zone_type == "layout-flow":
        flows.append(zone)
    for child in zone.children:
        _collect_flows(child, flows)


def _collect_hidden(zone: ZoneSpec, hidden: list):
    """Collect all hidden zones."""
    if zone.is_hidden:
        hidden.append(zone)
    for child in zone.children:
        _collect_hidden(child, hidden)


# ================================================================== #
#  Serialization                                                        #
# ================================================================== #

def workbook_to_dict(wb: TableauWorkbook) -> dict:
    """Convert a TableauWorkbook to a JSON-serializable dict."""
    import dataclasses

    def _convert(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                d[f.name] = _convert(val)
            return d
        elif isinstance(obj, list):
            return [_convert(item) for item in obj]
        elif isinstance(obj, Enum):
            return obj.value
        elif obj is None:
            return None
        else:
            return obj

    return _convert(wb)


# ================================================================== #
#  Utility                                                              #
# ================================================================== #

def _int(val) -> int:
    try:
        return int(str(val).replace("px", ""))
    except (ValueError, TypeError):
        return 0
