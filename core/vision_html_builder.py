"""
Vision HTML Builder -- Deterministic HTML dashboard from vision-extracted spec.

Produces a standalone HTML file that replicates the layout, chart types,
colors, and field names from a screenshot-derived tableau_spec.
Uses absolute positioning to match the source screenshot's visual arrangement.
Charts are rendered with Plotly.js using synthetic data shaped to match
the extracted field structure.
"""

import json
import os
import random
import math
from datetime import datetime, timedelta


def build_replica_html(tableau_spec, output_dir="./output", title=None):
    """Build an HTML dashboard replicating a vision-extracted spec.

    Args:
        tableau_spec: dict from vision_extractor (source == "vision_screenshot")
        output_dir: directory to write the HTML file
        title: override title (uses spec workbook_name if None)

    Returns:
        str: absolute path to the generated HTML file, or None on failure.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        workbook_name = title or tableau_spec.get(
            "workbook_name", "Dashboard Replica"
        )
        worksheets = tableau_spec.get("worksheets", [])
        dashboards = tableau_spec.get("dashboards", [])
        datasources = tableau_spec.get("datasources", [])
        design = tableau_spec.get("design", {})

        # Extract color palette
        palette = _extract_palette(design)
        bg_color = _extract_bg_color(design)
        font_family = _extract_font(design)
        title_color = "#333333" if _is_light(bg_color) else "#FFFFFF"

        # Dashboard dimensions from first dashboard or default
        canvas_w = 1200
        canvas_h = 800
        if dashboards:
            canvas = dashboards[0].get("canvas", {})
            canvas_w = canvas.get("width", 1200)
            canvas_h = canvas.get("height", 800)

        # Build all field names for synthetic data
        all_fields = _collect_all_fields(worksheets, datasources)

        # Generate synthetic data
        synth_data = _generate_synthetic_data(all_fields, worksheets)

        # Build visual containers HTML and Plotly JS
        containers_html = []
        plotly_calls = []
        chart_idx = 0

        for ws in worksheets:
            pos = ws.get("_position", {})
            x = pos.get("x", 0)
            y = pos.get("y", 0)
            w = pos.get("w", 300)
            h = pos.get("h", 200)

            ws_name = ws.get("name", f"Visual {chart_idx + 1}")
            chart_type = ws.get("chart_type", ws.get("mark_type", "automatic"))
            rows_fields = ws.get("rows_fields", [])
            cols_fields = ws.get("cols_fields", [])
            color_field = ws.get("color_field", "")

            chart_color = palette[chart_idx % len(palette)] if palette else "#4E79A7"

            if chart_type in ("ban", "kpi", "text") and not rows_fields and not cols_fields:
                # KPI Card -- pure HTML, no Plotly
                value_field = ""
                for f in (rows_fields + cols_fields):
                    if f:
                        value_field = f
                        break
                if not value_field:
                    # Try to get a measure from the worksheet
                    for m in ws.get("measures", []):
                        value_field = m
                        break
                synth_val = _generate_kpi_value(value_field)
                containers_html.append(
                    _build_kpi_card(
                        x, y, w, h, ws_name, synth_val,
                        chart_color, bg_color, title_color
                    )
                )
            else:
                # Chart visual -- Plotly
                div_id = f"chart_{chart_idx}"
                containers_html.append(
                    _build_chart_container(
                        x, y, w, h, ws_name, div_id, title_color
                    )
                )
                plotly_call = _build_plotly_call(
                    div_id, chart_type, ws_name,
                    cols_fields, rows_fields, color_field,
                    synth_data, palette, chart_color,
                    w, h
                )
                plotly_calls.append(plotly_call)

            chart_idx += 1

        # Assemble full HTML
        html = _assemble_html(
            workbook_name, canvas_w, canvas_h,
            bg_color, font_family, title_color,
            palette, containers_html, plotly_calls,
            synth_data
        )

        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in workbook_name
        ).strip().replace(" ", "_") or "Replica"
        filepath = os.path.join(output_dir, f"{safe_name}_replica.html")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return os.path.abspath(filepath)

    except Exception as e:
        print(f"[VISION-HTML] Build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ------------------------------------------------------------------ #
#  Color / Design helpers                                              #
# ------------------------------------------------------------------ #

_DEFAULT_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]


def _extract_palette(design):
    palettes = design.get("color_palettes", [])
    for p in palettes:
        colors = p.get("colors", [])
        if len(colors) >= 2:
            return colors
    return list(_DEFAULT_PALETTE)


def _extract_bg_color(design):
    fonts = design.get("global_fonts", {})
    return "#FFFFFF"


def _extract_font(design):
    fonts = design.get("global_fonts", {})
    return fonts.get("font-family", "Inter, Segoe UI, sans-serif")


def _is_light(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return True
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r * 299 + g * 587 + b * 114) / 1000 > 128


# ------------------------------------------------------------------ #
#  Field collection                                                    #
# ------------------------------------------------------------------ #

def _collect_all_fields(worksheets, datasources):
    """Collect all unique field names with their roles."""
    fields = {}
    for ds in datasources:
        for col in ds.get("columns", []):
            name = col.get("name", "")
            if name:
                fields[name] = col.get("role", "dimension")
    for ws in worksheets:
        for f in ws.get("cols_fields", []) + ws.get("rows_fields", []):
            if f and f not in fields:
                fields[f] = "dimension"
        for m in ws.get("measures", []):
            if m:
                fields[m] = "measure"
        cf = ws.get("color_field", "")
        if cf and cf not in fields:
            fields[cf] = "dimension"
    return fields


# ------------------------------------------------------------------ #
#  Synthetic data                                                      #
# ------------------------------------------------------------------ #

_CATEGORY_TEMPLATES = {
    "region": ["North", "South", "East", "West", "Central"],
    "country": ["USA", "Canada", "UK", "Germany", "France", "Japan", "Brazil"],
    "state": ["California", "Texas", "New York", "Florida", "Illinois", "Ohio"],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Dallas"],
    "category": ["Technology", "Office Supplies", "Furniture"],
    "product": ["Phones", "Chairs", "Binders", "Tables", "Storage", "Art"],
    "segment": ["Consumer", "Corporate", "Home Office"],
    "status": ["Active", "Inactive", "Pending", "Closed"],
    "type": ["Type A", "Type B", "Type C", "Type D"],
    "department": ["Sales", "Marketing", "Engineering", "Support", "Finance"],
    "channel": ["Online", "Retail", "Wholesale", "Direct"],
    "priority": ["High", "Medium", "Low"],
}


def _generate_synthetic_data(all_fields, worksheets, n_rows=50):
    """Generate a synthetic dataset matching the extracted fields."""
    data = {}
    random.seed(42)

    for field_name, role in all_fields.items():
        lower = field_name.lower().replace("_", " ").replace("-", " ")

        if role == "measure" or any(kw in lower for kw in [
            "sales", "revenue", "profit", "cost", "amount", "total",
            "count", "price", "quantity", "value", "budget", "score"
        ]):
            # Numeric measure
            base = random.randint(50, 5000)
            data[field_name] = [
                round(base + random.gauss(0, base * 0.3))
                for _ in range(n_rows)
            ]
        elif any(kw in lower for kw in [
            "date", "time", "year", "month", "quarter", "period"
        ]):
            # Date field
            start = datetime(2023, 1, 1)
            data[field_name] = [
                (start + timedelta(days=i * 7)).strftime("%Y-%m-%d")
                for i in range(n_rows)
            ]
        else:
            # Categorical dimension -- find matching template
            categories = None
            for key, vals in _CATEGORY_TEMPLATES.items():
                if key in lower:
                    categories = vals
                    break
            if not categories:
                categories = [f"{field_name} {chr(65 + i)}" for i in range(5)]
            data[field_name] = [
                random.choice(categories) for _ in range(n_rows)
            ]

    return data


def _generate_kpi_value(field_name):
    """Generate a plausible KPI value for a field."""
    random.seed(hash(field_name) % 2**32)
    lower = (field_name or "").lower()
    if any(kw in lower for kw in ["percent", "rate", "ratio"]):
        return f"{random.uniform(20, 95):.1f}%"
    elif any(kw in lower for kw in ["count", "number", "total", "quantity"]):
        return f"{random.randint(100, 50000):,}"
    elif any(kw in lower for kw in ["revenue", "sales", "profit", "cost", "amount"]):
        val = random.randint(10000, 5000000)
        if val >= 1000000:
            return f"${val / 1000000:.1f}M"
        return f"${val / 1000:.0f}K"
    else:
        return f"{random.randint(50, 9999):,}"


# ------------------------------------------------------------------ #
#  HTML building                                                       #
# ------------------------------------------------------------------ #

def _build_kpi_card(x, y, w, h, title, value, accent_color, bg_color, text_color):
    return (
        f'<div style="position:absolute;left:{x}px;top:{y}px;width:{w}px;height:{h}px;'
        f'background:#fff;border-radius:8px;border:1px solid #e0e0e0;'
        f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.08);overflow:hidden;">'
        f'<div style="font-size:{max(24, min(42, h // 3))}px;font-weight:700;'
        f'color:{accent_color};">{value}</div>'
        f'<div style="font-size:12px;color:#666;margin-top:4px;text-align:center;'
        f'padding:0 8px;">{title}</div>'
        f'</div>'
    )


def _build_chart_container(x, y, w, h, title, div_id, title_color):
    return (
        f'<div class="visual-container" style="left:{x}px;top:{y}px;'
        f'width:{w}px;height:{h}px;">'
        f'<div class="visual-title">{title}</div>'
        f'<div id="{div_id}" class="chart-div"></div>'
        f'</div>'
    )


def _build_plotly_call(div_id, chart_type, title, cols_fields, rows_fields,
                       color_field, synth_data, palette, accent_color, w, h):
    """Build a Plotly.newPlot() JS call for one chart."""

    x_field = cols_fields[0] if cols_fields else ""
    y_field = rows_fields[0] if rows_fields else ""

    # Fallback: if only one axis has data, use it appropriately
    if not x_field and y_field:
        x_field = y_field
        y_field = ""
    if not y_field and x_field:
        # Find a measure to pair with
        for fname, vals in synth_data.items():
            if fname != x_field and isinstance(vals[0], (int, float)):
                y_field = fname
                break

    x_data = synth_data.get(x_field, list(range(20)))
    y_data = synth_data.get(y_field, [random.randint(100, 1000) for _ in range(20)])

    # Aggregate if categorical x + numeric y
    if x_field and y_field and x_data and not isinstance(x_data[0], (int, float)):
        agg = {}
        for xv, yv in zip(x_data, y_data):
            xv_str = str(xv)
            if isinstance(yv, (int, float)):
                agg[xv_str] = agg.get(xv_str, 0) + yv
        x_data = list(agg.keys())
        y_data = list(agg.values())

    x_json = json.dumps(x_data[:30])
    y_json = json.dumps(y_data[:30])
    chart_h = h - 40  # account for title bar

    # Map chart type to Plotly trace + layout
    if chart_type in ("line", "lineChart"):
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{x:{x_json}, y:{y_json}, type:"scatter", mode:"lines+markers", '
            f'line:{{color:"{accent_color}", width:2}}, '
            f'marker:{{color:"{accent_color}", size:5}}}}], '
            f'{{margin:{{l:50,r:20,t:10,b:40}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", plot_bgcolor:"transparent", '
            f'xaxis:{{title:"{x_field}", gridcolor:"#f0f0f0"}}, '
            f'yaxis:{{title:"{y_field}", gridcolor:"#f0f0f0"}}, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    elif chart_type in ("pie", "pieChart", "donutChart"):
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{labels:{x_json}, values:{y_json}, type:"pie", '
            f'hole:{0.4 if "donut" in chart_type else 0}, '
            f'marker:{{colors:{json.dumps(palette[:len(x_data)])}}}}}], '
            f'{{margin:{{l:20,r:20,t:10,b:20}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", showlegend:true, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    elif chart_type in ("circle", "scatterChart", "scatter"):
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{x:{x_json}, y:{y_json}, type:"scatter", mode:"markers", '
            f'marker:{{color:"{accent_color}", size:8, opacity:0.7}}}}], '
            f'{{margin:{{l:50,r:20,t:10,b:40}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", plot_bgcolor:"transparent", '
            f'xaxis:{{title:"{x_field}", gridcolor:"#f0f0f0"}}, '
            f'yaxis:{{title:"{y_field}", gridcolor:"#f0f0f0"}}, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    elif chart_type in ("area", "areaChart"):
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{x:{x_json}, y:{y_json}, type:"scatter", fill:"tozeroy", '
            f'line:{{color:"{accent_color}"}}, fillcolor:"{accent_color}22"}}], '
            f'{{margin:{{l:50,r:20,t:10,b:40}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", plot_bgcolor:"transparent", '
            f'xaxis:{{title:"{x_field}", gridcolor:"#f0f0f0"}}, '
            f'yaxis:{{title:"{y_field}", gridcolor:"#f0f0f0"}}, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    elif chart_type in ("map", "polygon", "filledMap"):
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{type:"scattergeo", mode:"markers", '
            f'lat:[40,34,41,29,33,39,42,37,47,35], '
            f'lon:[-74,-118,-87,-95,-112,-77,-71,-122,-122,-106], '
            f'marker:{{size:10, color:"{accent_color}", opacity:0.8}}, '
            f'text:{x_json}}}], '
            f'{{margin:{{l:0,r:0,t:0,b:0}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", '
            f'geo:{{scope:"usa", bgcolor:"transparent", '
            f'lakecolor:"#f0f0f0", landcolor:"#fafafa"}}, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    elif chart_type in ("text", "tableEx", "matrix"):
        # Table -- show first few columns
        header_vals = json.dumps([x_field or "Item", y_field or "Value"])
        cell_x = json.dumps(x_data[:15])
        cell_y = json.dumps(y_data[:15])
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{type:"table", '
            f'header:{{values:{header_vals}, '
            f'fill:{{color:"{accent_color}"}}, font:{{color:"white",size:12}}, '
            f'align:"left"}}, '
            f'cells:{{values:[{cell_x},{cell_y}], '
            f'fill:{{color:["#fafafa","#ffffff"]}}, '
            f'align:"left", font:{{size:11}}}}}}], '
            f'{{margin:{{l:4,r:4,t:4,b:4}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent"}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )

    else:
        # Default: vertical bar chart
        return (
            f'Plotly.newPlot("{div_id}", '
            f'[{{x:{x_json}, y:{y_json}, type:"bar", '
            f'marker:{{color:"{accent_color}"}}}}], '
            f'{{margin:{{l:50,r:20,t:10,b:60}}, height:{chart_h}, '
            f'paper_bgcolor:"transparent", plot_bgcolor:"transparent", '
            f'xaxis:{{title:"{x_field}", gridcolor:"#f0f0f0", tickangle:-30}}, '
            f'yaxis:{{title:"{y_field}", gridcolor:"#f0f0f0"}}, '
            f'font:{{size:11}}}}, '
            f'{{responsive:true, displayModeBar:false}});'
        )


def _assemble_html(workbook_name, canvas_w, canvas_h, bg_color,
                    font_family, title_color, palette,
                    containers_html, plotly_calls, synth_data):
    """Assemble the final HTML document."""

    containers_str = "\n    ".join(containers_html)
    plotly_str = "\n    ".join(plotly_calls)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{workbook_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: {bg_color};
    font-family: {font_family};
    padding: 24px;
    color: #333;
  }}
  .dashboard-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid {palette[0] if palette else "#4E79A7"};
  }}
  .dashboard-title {{
    font-size: 22px;
    font-weight: 700;
    color: {title_color};
  }}
  .dashboard-meta {{
    font-size: 12px;
    color: #999;
  }}
  .dashboard-container {{
    position: relative;
    width: {canvas_w}px;
    height: {canvas_h}px;
    margin: 0 auto;
  }}
  .visual-container {{
    position: absolute;
    background: #ffffff;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    overflow: hidden;
  }}
  .visual-title {{
    font-size: 13px;
    font-weight: 600;
    padding: 8px 12px;
    color: {title_color};
    border-bottom: 1px solid #f0f0f0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .chart-div {{
    width: 100%;
    height: calc(100% - 34px);
  }}
  .footer {{
    text-align: center;
    font-size: 11px;
    color: #aaa;
    margin-top: 24px;
    padding-top: 12px;
    border-top: 1px solid #e0e0e0;
  }}
  @media print {{
    body {{ padding: 12px; }}
    .visual-container {{ box-shadow: none; border: 1px solid #e0e0e0; }}
  }}
</style>
</head>
<body>

<div class="dashboard-header">
  <div class="dashboard-title">{workbook_name}</div>
  <div class="dashboard-meta">Replicated from screenshot | Built by Dr. Data</div>
</div>

<div class="dashboard-container">
    {containers_str}
</div>

<div class="footer">
  Built by Dr. Data | The Art of the Possible | Replicated from dashboard screenshot
</div>

<script>
    {plotly_str}
</script>

</body>
</html>'''
