"""
Interactive Dashboard -- generates self-contained HTML with Plotly.js charts
and DataTables.js data grid, Western Union branded.
"""
import os
import json
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np


# Professional palette
_BG = "#0F172A"
_CARD = "#1E293B"
_ALT = "#334155"
_BORDER = "#475569"
_ACCENT = "#6366F1"
_SERIES = ["#6366F1", "#22D3EE", "#10B981", "#F59E0B", "#F43F5E",
           "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#3B82F6",
           "#84CC16", "#EF4444"]
_WHITE = "#F1F5F9"
_GRAY = "#94A3B8"


def _hex_to_rgb(hex_color):
    """Convert #RRGGBB to 'R,G,B' string for rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


class InteractiveDashboard:

    def _detect_column_types(self, df):
        date_cols, numeric_cols, categorical_cols = [], [], []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    pd.to_datetime(df[col], errors="raise")
                    date_cols.append(col)
                except Exception:
                    if df[col].nunique() < 15:
                        categorical_cols.append(col)
        return {"date_cols": date_cols, "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols}

    def _pick_charts(self, df, types):
        charts = []
        date_cols = types["date_cols"]
        num_cols = types["numeric_cols"]
        cat_cols = types["categorical_cols"]

        # 1. Time series if date column exists
        if date_cols and num_cols:
            dc = date_cols[0]
            vals = num_cols[:3]
            traces = []
            for i, v in enumerate(vals):
                traces.append({
                    "x": "__DATE__",
                    "y": v,
                    "type": "scatter",
                    "mode": "lines",
                    "name": v,
                    "line": {"color": _SERIES[i % len(_SERIES)]},
                })
            charts.append({"title": f"Trends over {dc}", "traces": traces,
                           "date_col": dc, "kind": "timeseries"})

        # 2. Bar chart for top categorical
        if cat_cols and num_cols:
            cc = cat_cols[0]
            nc = num_cols[0]
            charts.append({"title": f"{nc} by {cc}", "cat_col": cc,
                           "num_col": nc, "kind": "bar"})

        # 3. Histogram for first numeric
        if num_cols:
            charts.append({"title": f"Distribution of {num_cols[0]}",
                           "num_col": num_cols[0], "kind": "histogram"})

        # 4. Second bar or second histogram
        if len(cat_cols) > 1 and num_cols:
            charts.append({"title": f"{num_cols[0]} by {cat_cols[1]}",
                           "cat_col": cat_cols[1], "num_col": num_cols[0],
                           "kind": "bar"})
        elif len(num_cols) > 1:
            charts.append({"title": f"Distribution of {num_cols[1]}",
                           "num_col": num_cols[1], "kind": "histogram"})

        return charts[:4]

    def _build_chart_js(self, df, chart, idx):
        layout_base = (
            f'{{paper_bgcolor:"{_CARD}",plot_bgcolor:"{_CARD}",'
            f'font:{{color:"{_WHITE}",family:"Inter,system-ui,sans-serif",size:12}},'
            f'margin:{{l:56,r:24,t:48,b:48}},'
            f'xaxis:{{gridcolor:"#1F2937",linecolor:"#374151",tickfont:{{color:"{_GRAY}"}}}},'
            f'yaxis:{{gridcolor:"#1F2937",linecolor:"#374151",tickfont:{{color:"{_GRAY}"}}}},'
            f'colorway:{json.dumps(_SERIES)},'
            f'bargap:0.2,bargroupgap:0.08,'
            f'hoverlabel:{{bgcolor:"#1F2937",font:{{color:"{_WHITE}",size:12}},bordercolor:"#4B5563"}},'
            f'hovermode:"closest",title:{{text:"{chart["title"]}",font:{{size:14,color:"{_WHITE}"}}}}}}'
        )
        cfg = '{responsive:true,displayModeBar:true,modeBarButtonsToRemove:["lasso2d","select2d"]}'
        div_id = f"chart{idx}"

        if chart["kind"] == "timeseries":
            dc = chart["date_col"]
            tmp = df.dropna(subset=[dc]).sort_values(dc)
            x_vals = tmp[dc].astype(str).tolist()
            trace_strs = []
            for ti, t in enumerate(chart["traces"]):
                y_vals = tmp[t["y"]].tolist()
                y_json = json.dumps(y_vals)
                color = _SERIES[ti % len(_SERIES)]
                trace_strs.append(
                    f'{{x:{json.dumps(x_vals)},y:{y_json},'
                    f'type:"scatter",mode:"lines+markers",name:"{t["y"]}",'
                    f'line:{{color:"{color}",width:2.5,shape:"spline"}},'
                    f'marker:{{size:4,color:"{color}"}},'
                    f'fill:"tozeroy",fillcolor:"rgba({_hex_to_rgb(color)},0.08)"}}'
                )
            data_str = "[" + ",".join(trace_strs) + "]"
            return div_id, f'Plotly.newPlot("{div_id}",{data_str},{layout_base},{cfg});'

        if chart["kind"] == "bar":
            cc, nc = chart["cat_col"], chart["num_col"]
            grp = df.groupby(cc, dropna=True)[nc].sum().sort_values(ascending=False).head(12)
            x_json = json.dumps(grp.index.tolist())
            y_json = json.dumps(grp.values.tolist())
            bar_colors = json.dumps([_SERIES[j % len(_SERIES)] for j in range(len(grp))])
            data_str = f'[{{x:{x_json},y:{y_json},type:"bar",marker:{{color:{bar_colors},line:{{color:"{_BG}",width:1}},opacity:0.9}}}}]'
            return div_id, f'Plotly.newPlot("{div_id}",{data_str},{layout_base},{cfg});'

        if chart["kind"] == "histogram":
            nc = chart["num_col"]
            vals = df[nc].dropna().tolist()
            hist_color = _SERIES[idx % len(_SERIES)]
            data_str = f'[{{x:{json.dumps(vals)},type:"histogram",marker:{{color:"{hist_color}",line:{{color:"{_BG}",width:1}},opacity:0.85}}}}]'
            return div_id, f'Plotly.newPlot("{div_id}",{data_str},{layout_base},{cfg});'

        return div_id, ""

    def generate(self, df, title, output_path=None):
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), f"{title}.html")
        try:
            print(f"[InteractiveDashboard] Generating: {title}")
            types = self._detect_column_types(df)
            charts = self._pick_charts(df, types)
            rows, cols = df.shape
            num_cols = types["numeric_cols"]

            # Stats
            stats_cards = [
                ("Total Rows", f"{rows:,}"),
                ("Columns", str(cols)),
                ("Numeric Fields", str(len(num_cols))),
                ("Null Cells", f"{int(df.isna().sum().sum()):,}"),
            ]
            if num_cols:
                top = num_cols[0]
                stats_cards.append((f"Avg {top[:18]}", f"{df[top].mean():,.2f}"))
                stats_cards.append((f"Max {top[:18]}", f"{df[top].max():,.2f}"))

            # Build chart divs + JS
            chart_divs = []
            chart_scripts = []
            for i, ch in enumerate(charts):
                div_id, js = self._build_chart_js(df, ch, i)
                chart_divs.append(f'<div class="chart-cell"><div id="{div_id}"></div></div>')
                chart_scripts.append(js)

            # Table data (cap 10000 rows)
            tbl_df = df.head(10000).copy()
            for c in tbl_df.columns:
                if pd.api.types.is_datetime64_any_dtype(tbl_df[c]):
                    tbl_df[c] = tbl_df[c].astype(str)
            tbl_df = tbl_df.fillna("")
            col_headers = "".join(f"<th>{c}</th>" for c in tbl_df.columns)
            tbl_data_json = json.dumps(tbl_df.values.tolist(), default=str).replace("</script>", "<\\/script>")
            col_defs_json = json.dumps([{"title": str(c)} for c in tbl_df.columns])

            # Stats cards HTML
            stats_html = "".join(
                f'<div class="stat-card"><div class="stat-val">{v}</div>'
                f'<div class="stat-lbl">{l}</div></div>'
                for l, v in stats_cards
            )

            date_str = datetime.now().strftime("%B %d, %Y")

            html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:{_BG};color:{_WHITE};font-family:'Inter',system-ui,-apple-system,sans-serif;padding:0;line-height:1.5}}

/* Header */
.top-bar{{background:{_CARD};border-bottom:1px solid {_BORDER};padding:20px 36px;display:flex;
  align-items:center;justify-content:space-between;position:relative}}
.top-bar::before{{content:"";position:absolute;top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,#6366F1,#22D3EE,#10B981,#F59E0B)}}
.top-bar h1{{font-size:22px;font-weight:700;color:{_WHITE};letter-spacing:-0.5px}}
.top-bar .date{{font-size:12px;color:{_GRAY}}}
.wu-mark{{background:linear-gradient(135deg,{_ACCENT},#8B5CF6);color:#fff;font-weight:700;
  font-size:11px;padding:5px 14px;border-radius:6px;letter-spacing:.5px}}

/* Layout */
.container{{max-width:1480px;margin:0 auto;padding:24px 36px}}
.section{{background:{_CARD};border:1px solid {_BORDER};border-radius:14px;padding:24px 28px;
  margin-bottom:24px;transition:border-color .2s}}
.section:hover{{border-color:{_ACCENT}}}
.section h2{{font-size:15px;font-weight:700;color:{_WHITE};margin-bottom:16px;
  padding-bottom:10px;border-bottom:1px solid rgba(71,85,105,0.5);
  display:flex;align-items:center;gap:8px}}
.section h2::before{{content:"";display:inline-block;width:4px;height:18px;
  border-radius:2px;background:{_ACCENT}}}

/* KPI Cards */
.stats-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}}
.stat-card{{background:{_ALT};border:1px solid {_BORDER};border-radius:12px;
  padding:18px 22px;text-align:center;position:relative;overflow:hidden;transition:all .25s}}
.stat-card::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}}
.stat-card:nth-child(1)::before{{background:linear-gradient(90deg,#6366F1,#818CF8)}}
.stat-card:nth-child(2)::before{{background:linear-gradient(90deg,#22D3EE,#67E8F9)}}
.stat-card:nth-child(3)::before{{background:linear-gradient(90deg,#10B981,#34D399)}}
.stat-card:nth-child(4)::before{{background:linear-gradient(90deg,#F59E0B,#FBBF24)}}
.stat-card:nth-child(5)::before{{background:linear-gradient(90deg,#F43F5E,#FB7185)}}
.stat-card:nth-child(6)::before{{background:linear-gradient(90deg,#8B5CF6,#A78BFA)}}
.stat-card:hover{{border-color:{_ACCENT};transform:translateY(-2px);
  box-shadow:0 8px 25px rgba(99,102,241,0.12)}}
.stat-val{{font-size:26px;font-weight:800;line-height:1.1}}
.stat-card:nth-child(1) .stat-val{{color:#6366F1}}
.stat-card:nth-child(2) .stat-val{{color:#22D3EE}}
.stat-card:nth-child(3) .stat-val{{color:#10B981}}
.stat-card:nth-child(4) .stat-val{{color:#F59E0B}}
.stat-card:nth-child(5) .stat-val{{color:#F43F5E}}
.stat-card:nth-child(6) .stat-val{{color:#8B5CF6}}
.stat-lbl{{font-size:11px;color:{_GRAY};text-transform:uppercase;letter-spacing:.8px;
  margin-top:6px;font-weight:500}}

/* Charts */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.chart-cell{{background:{_ALT};border:1px solid {_BORDER};border-radius:12px;padding:14px;
  min-height:340px;transition:all .25s}}
.chart-cell:hover{{border-color:{_ACCENT};box-shadow:0 4px 20px rgba(99,102,241,0.08)}}
@media(max-width:900px){{.chart-grid{{grid-template-columns:1fr}}.stats-row{{grid-template-columns:1fr 1fr}}}}

/* DataTables */
table.dataTable{{border-collapse:collapse!important;width:100%!important}}
table.dataTable thead th{{background:{_ACCENT}!important;color:#fff!important;
  font-weight:600!important;border-bottom:none!important;padding:12px 14px!important;font-size:12px}}
table.dataTable tbody td{{background:{_CARD}!important;color:{_WHITE}!important;
  border-bottom:1px solid rgba(71,85,105,0.3)!important;padding:10px 14px!important;font-size:13px}}
table.dataTable tbody tr:nth-child(even) td{{background:rgba(30,41,59,0.8)!important}}
table.dataTable tbody tr:hover td{{background:rgba(99,102,241,0.1)!important}}
table.dataTable tbody tr.selected td{{background:{_ACCENT}!important;color:#fff!important}}
.dataTables_wrapper,.dataTables_wrapper *{{color:{_WHITE}!important}}
.dataTables_wrapper .dataTables_filter input,
.dataTables_wrapper .dataTables_length select{{background:{_BG}!important;color:{_WHITE}!important;
  border:1px solid {_BORDER}!important;border-radius:8px!important;padding:6px 10px!important}}
.dataTables_wrapper .dataTables_filter input:focus{{border-color:{_ACCENT}!important;
  box-shadow:0 0 0 3px rgba(99,102,241,0.15)!important;outline:none}}
.dataTables_wrapper .dataTables_paginate .paginate_button{{color:{_WHITE}!important;
  background:{_ALT}!important;border:1px solid {_BORDER}!important;border-radius:6px!important}}
.dataTables_wrapper .dataTables_paginate .paginate_button.current{{background:{_ACCENT}!important;
  color:#fff!important;font-weight:600!important;border-color:{_ACCENT}!important}}
.dataTables_wrapper .dataTables_paginate .paginate_button:hover{{background:{_ACCENT}!important;
  color:#fff!important;border-color:{_ACCENT}!important}}
.dt-buttons .dt-button{{background:{_ALT}!important;color:{_WHITE}!important;
  border:1px solid {_BORDER}!important;border-radius:8px!important;padding:8px 16px!important;
  font-size:12px!important;font-family:inherit!important;transition:all .2s}}
.dt-buttons .dt-button:hover{{background:{_ACCENT}!important;color:#fff!important;
  border-color:{_ACCENT}!important}}
.dataTables_info{{font-size:12px;color:{_GRAY}!important}}
footer{{text-align:center;padding:20px;font-size:11px;color:{_GRAY};
  border-top:1px solid {_BORDER};margin-top:32px}}
</style>
</head><body>
<div class="top-bar">
  <div><h1>{title}</h1><span class="date">{date_str}</span></div>
  <span class="wu-mark">DR. DATA</span>
</div>
<div class="container">
  <div class="section"><h2>Executive Summary</h2><div class="stats-row">{stats_html}</div></div>
  <div class="section"><h2>Analysis</h2><div class="chart-grid">{"".join(chart_divs)}</div></div>
  <div class="section"><h2>Data Explorer</h2>
    <table id="dtable" class="display nowrap" style="width:100%">
      <thead><tr>{col_headers}</tr></thead>
    </table>
  </div>
</div>
<footer>Built by Dr. Data | The Art of the Possible | Powered by Claude Opus 4</footer>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script>
var _tblData={tbl_data_json};
var _tblCols={col_defs_json};
$(document).ready(function(){{
  $('#dtable').DataTable({{
    data:_tblData,columns:_tblCols,pageLength:25,
    dom:'Bfrtip',buttons:['copy','csv','excel'],
    order:[],scrollX:true,
    language:{{search:"Search:",lengthMenu:"Show _MENU_ rows"}}
  }});
}});
{"".join(chart_scripts)}
</script>
</body></html>"""

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[InteractiveDashboard] Saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"[InteractiveDashboard] Failed: {e}")
            return None
