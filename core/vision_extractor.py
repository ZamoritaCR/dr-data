"""
Vision Extractor -- GPT-4o Vision screenshot to tableau_spec pipeline.

Takes a PNG/JPG screenshot of any BI dashboard and returns a dict that
matches the structure produced by enhanced_tableau_parser.parse_twb().
This allows the existing direct_mapper, synthetic_data, and pbip_generator
pipelines to consume screenshot-derived specs without modification.

Required downstream keys:
  spec["type"]               -> "tableau_workbook" (triggers Tableau pipeline)
  spec["version"]            -> str
  spec["datasources"]        -> list[{name, caption, connection_type, columns: [{name, datatype, role}]}]
  spec["worksheets"]         -> list[{name, chart_type, mark_type, rows, cols,
                                      rows_fields, cols_fields, dimensions, measures,
                                      filters, color_field, size_field, label_fields,
                                      tooltip_fields, sort_field, design}]
  spec["dashboards"]         -> list[{name, size, canvas, worksheets_used, zones: [{name, layout:{x,y,w,h}}]}]
  spec["calculated_fields"]  -> list[{name, formula, datasource}]
  spec["parameters"]         -> list
  spec["filters"]            -> list
  spec["relationships"]      -> list
  spec["has_hyper"]          -> bool
  spec["design"]             -> {color_palettes, global_fonts}
"""

import os
import base64
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are a BI dashboard analyst. Your job is to analyze a screenshot of a \
dashboard and extract its structure into a precise JSON format.
You must return ONLY valid JSON. No markdown. No explanation. No backticks.
Be specific. Use the visual evidence in the image. Do not guess wildly.
If you cannot determine a value with confidence, use a sensible default."""

_USER_PROMPT = """\
Analyze this dashboard screenshot and return a JSON object with this exact structure:

{
  "workbook_name": "string - infer from title visible in image or use Dashboard",
  "datasources": [
    {
      "name": "string - infer from data visible or use Primary",
      "columns": [
        {
          "name": "string - exact field name from axis labels, titles, or tooltips visible in image",
          "data_type": "string | double | integer | date | boolean",
          "role": "dimension | measure"
        }
      ]
    }
  ],
  "worksheets": [
    {
      "name": "string - visual title if visible, or Visual_1, Visual_2 etc.",
      "mark_type": "bar | line | circle | map | text | pie | area | automatic",
      "chart_type_inferred": "clusteredColumnChart | lineChart | scatterChart | map | card | tableEx | pieChart | areaChart | donutChart",
      "rows_fields": ["field names on Y axis or rows"],
      "cols_fields": ["field names on X axis or columns"],
      "color_field": "field name used for color/legend or empty string",
      "filters": [],
      "position": {
        "x": 0,
        "y": 0,
        "w": 400,
        "h": 300
      }
    }
  ],
  "dashboards": [
    {
      "name": "string - dashboard title from image or Main Dashboard",
      "worksheets_used": ["list of worksheet names above"],
      "canvas_width": 1200,
      "canvas_height": 800
    }
  ],
  "color_palette": ["#hex1", "#hex2", "#hex3"],
  "font_family": "Segoe UI"
}

Rules for extraction:
- Identify EVERY distinct visual/chart in the screenshot as a separate worksheet entry
- For each visual, read the axis labels to determine field names
- Estimate position coordinates based on where each visual appears in the image \
(top-left = 0,0, full width = 1200, full height = 800)
- Extract color palette from the dominant colors visible in the charts
- chart_type_inferred must be a Power BI visual type, not a Tableau type
- If you see KPI cards or big number displays, use mark_type: "text" and chart_type_inferred: "card"
- If you see a map, use mark_type: "map"
- If you see a table/grid of data, use chart_type_inferred: "tableEx"
- For fields you can read from axis labels: use those exact names
- For fields you cannot read: infer from context (Sales, Revenue, Date, Category, Region, etc.)
- role: "measure" for numeric fields, "dimension" for categorical/date fields
- Return ONLY the JSON object. No other text."""


def extract_dashboard_spec_from_image(image_path: str) -> dict:
    """Extract a tableau_spec-shaped dict from a dashboard screenshot.

    Calls GPT-4o Vision to analyze the image and returns a dict compatible
    with the existing direct_mapper and synthetic_data pipelines.

    Args:
        image_path: path to a PNG or JPG file.

    Returns:
        dict matching the enhanced_tableau_parser spec structure.
        On failure, returns a minimal valid spec (never crashes).
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("[VISION] openai package not installed")
        return _empty_spec()

    api_key = _get_openai_key()
    if not api_key:
        logger.error("[VISION] No OPENAI_API_KEY available")
        return _empty_spec()

    try:
        client = OpenAI(api_key=api_key)

        image_data = Path(image_path).read_bytes()
        b64_image = base64.b64encode(image_data).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4000,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if GPT wraps in ```json ... ```
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            raw = "\n".join(lines)

        vision_result = json.loads(raw)
        spec = _convert_to_tableau_spec(vision_result)

        ws_count = len(spec.get("worksheets", []))
        ds_count = len(spec.get("datasources", []))
        logger.info(
            f"[VISION] Extracted {ws_count} visuals, "
            f"{ds_count} datasources from screenshot"
        )
        return spec

    except json.JSONDecodeError as e:
        logger.error(f"[VISION] Failed to parse GPT-4o response as JSON: {e}")
        return _empty_spec()
    except Exception as e:
        logger.error(f"[VISION] Extraction failed: {e}")
        return _empty_spec()


def _get_openai_key() -> str:
    """Load OpenAI API key from settings or environment."""
    try:
        from config.settings import OPENAI_API_KEY
        if OPENAI_API_KEY:
            return OPENAI_API_KEY
    except ImportError:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def _convert_to_tableau_spec(vision: dict) -> dict:
    """Convert the GPT-4o vision output to the canonical tableau_spec structure.

    Maps the simplified vision JSON to the full structure that
    direct_mapper.build_pbip_config_from_tableau() expects.
    """
    workbook_name = vision.get("workbook_name", "Imported Dashboard")

    # -- Datasources --
    datasources = []
    for ds in vision.get("datasources", []):
        columns = []
        for col in ds.get("columns", []):
            columns.append({
                "name": col.get("name", ""),
                "caption": col.get("name", ""),
                "datatype": col.get("data_type", "string"),
                "role": col.get("role", "dimension"),
                "type": "quantitative" if col.get("role") == "measure" else "nominal",
            })
        datasources.append({
            "name": ds.get("name", "Primary"),
            "caption": ds.get("name", "Primary"),
            "connection_type": "unknown",
            "server": "",
            "dbname": "",
            "filename": "",
            "tables": [],
            "columns": columns,
        })

    # If no datasources were returned, build one from all worksheet fields
    if not datasources:
        all_columns = _collect_columns_from_worksheets(
            vision.get("worksheets", [])
        )
        if all_columns:
            datasources = [{
                "name": "Primary",
                "caption": "Primary",
                "connection_type": "unknown",
                "server": "",
                "dbname": "",
                "filename": "",
                "tables": [],
                "columns": all_columns,
            }]

    # -- Worksheets --
    worksheets = []
    for ws in vision.get("worksheets", []):
        name = ws.get("name", f"Visual_{len(worksheets) + 1}")
        mark_type = ws.get("mark_type", "automatic")
        chart_type = ws.get("chart_type_inferred", "")

        # Map chart_type_inferred back to a Tableau mark_type for the parser
        # so design_translator.translate_chart_type() produces the right PBI type
        effective_mark = _pbi_to_mark_type(chart_type) if chart_type else mark_type

        rows_fields = ws.get("rows_fields", [])
        cols_fields = ws.get("cols_fields", [])
        color_field = ws.get("color_field", "")
        pos = ws.get("position", {})

        # Build dimensions/measures from field roles
        dimensions = []
        measures = []
        for f in cols_fields:
            if _looks_like_measure(f):
                measures.append(f)
            else:
                dimensions.append(f)
        for f in rows_fields:
            if _looks_like_measure(f):
                measures.append(f)
            else:
                dimensions.append(f)

        worksheets.append({
            "name": name,
            "chart_type": effective_mark,
            "mark_type": effective_mark,
            "marks": [mark_type] if mark_type else [],
            "rows": ", ".join(rows_fields),
            "cols": ", ".join(cols_fields),
            "rows_fields": rows_fields,
            "cols_fields": cols_fields,
            "dimensions": dimensions,
            "measures": measures,
            "filters": ws.get("filters", []),
            "color_field": color_field,
            "size_field": "",
            "label_fields": [],
            "tooltip_fields": [],
            "sort_field": "",
            "design": {
                "mark_colors": [],
                "background_color": "",
                "title_font": {},
                "axis_config": [],
                "mark_style": {},
                "border": {},
            },
            # Stash position for dashboard zone building
            "_position": {
                "x": pos.get("x", 0),
                "y": pos.get("y", 0),
                "w": pos.get("w", 400),
                "h": pos.get("h", 300),
            },
        })

    # -- Dashboards --
    dashboards = []
    for db in vision.get("dashboards", []):
        canvas_w = db.get("canvas_width", 1200)
        canvas_h = db.get("canvas_height", 800)
        ws_used = db.get("worksheets_used", [])

        # If worksheets_used is empty, use all worksheets
        if not ws_used:
            ws_used = [w["name"] for w in worksheets]

        zones = []
        for w in worksheets:
            if w["name"] in ws_used:
                p = w.get("_position", {})
                zones.append({
                    "name": w["name"],
                    "type": "worksheet",
                    "layout": {
                        "x": p.get("x", 0),
                        "y": p.get("y", 0),
                        "w": p.get("w", 400),
                        "h": p.get("h", 300),
                    },
                })

        dashboards.append({
            "name": db.get("name", workbook_name),
            "size": {"width": str(canvas_w), "height": str(canvas_h)},
            "canvas": {"width": canvas_w, "height": canvas_h},
            "worksheets_used": ws_used,
            "zones": zones,
        })

    # If no dashboards returned, create one from all worksheets
    if not dashboards and worksheets:
        ws_names = [w["name"] for w in worksheets]
        zones = []
        for w in worksheets:
            p = w.get("_position", {})
            zones.append({
                "name": w["name"],
                "type": "worksheet",
                "layout": {
                    "x": p.get("x", 0),
                    "y": p.get("y", 0),
                    "w": p.get("w", 400),
                    "h": p.get("h", 300),
                },
            })
        dashboards.append({
            "name": workbook_name,
            "size": {"width": "1200", "height": "800"},
            "canvas": {"width": 1200, "height": 800},
            "worksheets_used": ws_names,
            "zones": zones,
        })

    # -- Color palette -> design structure --
    palette_colors = vision.get("color_palette", [])
    font_family = vision.get("font_family", "Segoe UI")
    design = {
        "color_palettes": [],
        "global_fonts": {},
    }
    if palette_colors:
        design["color_palettes"] = [{
            "name": "extracted",
            "type": "regular",
            "colors": palette_colors,
        }]
    if font_family:
        design["global_fonts"] = {"font-family": font_family}

    # -- Assemble final spec --
    spec = {
        "type": "tableau_workbook",
        "version": "vision_extracted",
        "datasources": datasources,
        "worksheets": worksheets,
        "dashboards": dashboards,
        "calculated_fields": [],
        "parameters": [],
        "filters": [],
        "relationships": [],
        "has_hyper": False,
        "design": design,
        "source": "vision_screenshot",
    }

    return spec


def _collect_columns_from_worksheets(worksheets: list) -> list:
    """Build a column list from all fields mentioned across worksheets."""
    seen = set()
    columns = []
    for ws in worksheets:
        for f in ws.get("cols_fields", []) + ws.get("rows_fields", []):
            if f and f not in seen:
                seen.add(f)
                role = "measure" if _looks_like_measure(f) else "dimension"
                columns.append({
                    "name": f,
                    "caption": f,
                    "datatype": "double" if role == "measure" else "string",
                    "role": role,
                    "type": "quantitative" if role == "measure" else "nominal",
                })
        cf = ws.get("color_field", "")
        if cf and cf not in seen:
            seen.add(cf)
            columns.append({
                "name": cf,
                "caption": cf,
                "datatype": "string",
                "role": "dimension",
                "type": "nominal",
            })
    return columns


def _looks_like_measure(field_name: str) -> bool:
    """Heuristic: does this field name look like a numeric measure?"""
    measure_keywords = {
        "sales", "revenue", "profit", "cost", "price", "amount", "total",
        "count", "sum", "avg", "average", "quantity", "margin", "rate",
        "percent", "ratio", "value", "budget", "spend", "income",
        "volume", "units", "weight", "score", "index",
    }
    lower = field_name.lower().replace("_", " ").replace("-", " ")
    return any(kw in lower for kw in measure_keywords)


def _pbi_to_mark_type(pbi_type: str) -> str:
    """Map a PBI visual type back to a Tableau-style mark type.

    The parser chart_type values are what design_translator.translate_chart_type()
    expects as input. This maps the GPT output back to that domain.
    """
    mapping = {
        "clusteredColumnChart": "bar",
        "clusteredBarChart": "bar",
        "stackedBarChart": "bar",
        "lineChart": "line",
        "areaChart": "area",
        "scatterChart": "circle",
        "map": "map",
        "filledMap": "polygon",
        "card": "ban",
        "cardVisual": "ban",
        "kpi": "ban",
        "tableEx": "text",
        "matrix": "text",
        "pieChart": "pie",
        "donutChart": "pie",
        "treemap": "square",
        "waterfallChart": "bar",
        "funnelChart": "bar",
        "lineClusteredColumnComboChart": "bar",
    }
    return mapping.get(pbi_type, "automatic")


def _empty_spec() -> dict:
    """Minimal valid spec returned on total failure."""
    return {
        "type": "tableau_workbook",
        "version": "vision_extracted",
        "datasources": [],
        "worksheets": [],
        "dashboards": [],
        "calculated_fields": [],
        "parameters": [],
        "filters": [],
        "relationships": [],
        "has_hyper": False,
        "design": {"color_palettes": [], "global_fonts": {}},
        "source": "vision_screenshot",
    }
