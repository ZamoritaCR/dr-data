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


# WU palette
_BG = "#1a1a1a"
_CARD = "#2d2d2d"
_ALT = "#3a3a3a"
_BORDER = "#4a4a4a"
_YELLOW = "#FFDE00"
_SERIES = ["#FFDE00", "#FFE600", "#FFB800", "#FF9500", "#FFFFFF"]
_WHITE = "#FFFFFF"
_GRAY = "#B0B0B0"


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
            f'font:{{color:"{_WHITE}",family:"Arial,Helvetica,sans-serif"}},'
            f'margin:{{l:50,r:20,t:40,b:50}},'
            f'xaxis:{{gridcolor:"{_BORDER}",linecolor:"{_BORDER}"}},'
            f'yaxis:{{gridcolor:"{_BORDER}",linecolor:"{_BORDER}"}},'
            f'hovermode:"closest",title:{{text:"{chart["title"]}",font:{{size:14}}}}}}'
        )
        cfg = '{responsive:true,displayModeBar:true,modeBarButtonsToRemove:["lasso2d","select2d"]}'
        div_id = f"chart{idx}"

        if chart["kind"] == "timeseries":
            dc = chart["date_col"]
            tmp = df.dropna(subset=[dc]).sort_values(dc)
            x_vals = tmp[dc].astype(str).tolist()
            trace_strs = []
            for t in chart["traces"]:
                y_vals = tmp[t["y"]].tolist()
                y_json = json.dumps(y_vals)
                trace_strs.append(
                    f'{{x:{json.dumps(x_vals)},y:{y_json},'
                    f'type:"scatter",mode:"lines",name:"{t["y"]}",'
                    f'line:{{color:"{t["line"]["color"]}"}}}}'
                )
            data_str = "[" + ",".join(trace_strs) + "]"
            return div_id, f'Plotly.newPlot("{div_id}",{data_str},{layout_base},{cfg});'

        if chart["kind"] == "bar":
            cc, nc = chart["cat_col"], chart["num_col"]
            grp = df.groupby(cc, dropna=True)[nc].sum().sort_values(ascending=False).head(12)
            x_json = json.dumps(grp.index.tolist())
            y_json = json.dumps(grp.values.tolist())
            data_str = f'[{{x:{x_json},y:{y_json},type:"bar",marker:{{color:"{_YELLOW}"}}}}]'
            return div_id, f'Plotly.newPlot("{div_id}",{data_str},{layout_base},{cfg});'

        if chart["kind"] == "histogram":
            nc = chart["num_col"]
            vals = df[nc].dropna().tolist()
            data_str = f'[{{x:{json.dumps(vals)},type:"histogram",marker:{{color:"{_YELLOW}",line:{{color:"{_BORDER}",width:1}}}}}}]'
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
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:{_BG};color:{_WHITE};font-family:Arial,Helvetica,sans-serif;padding:0}}
.top-bar{{background:{_CARD};border-bottom:2px solid {_YELLOW};padding:18px 32px;display:flex;
  align-items:center;justify-content:space-between}}
.top-bar h1{{font-size:22px;font-weight:700;color:{_WHITE}}}
.top-bar .date{{font-size:12px;color:{_GRAY}}}
.wu-mark{{background:{_YELLOW};color:{_BG};font-weight:900;font-size:11px;padding:4px 10px;
  border-radius:3px;letter-spacing:1px}}
.container{{max-width:1400px;margin:0 auto;padding:20px 24px}}
.section{{background:{_CARD};border:1px solid {_BORDER};border-radius:10px;padding:20px 24px;
  margin-bottom:20px}}
.section h2{{font-size:15px;font-weight:600;color:{_YELLOW};margin-bottom:14px;
  padding-bottom:8px;border-bottom:1px solid {_BORDER}}}
.stats-row{{display:flex;gap:12px;flex-wrap:wrap}}
.stat-card{{flex:1;min-width:140px;background:{_ALT};border:1px solid {_BORDER};border-radius:8px;
  padding:14px 18px;text-align:center}}
.stat-val{{font-size:22px;font-weight:700;color:{_YELLOW}}}
.stat-lbl{{font-size:10px;color:{_GRAY};text-transform:uppercase;letter-spacing:.8px;margin-top:4px}}
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.chart-cell{{background:{_ALT};border:1px solid {_BORDER};border-radius:8px;padding:10px;
  min-height:320px}}
@media(max-width:900px){{.chart-grid{{grid-template-columns:1fr}}}}

/* DataTables dark overrides */
table.dataTable{{border-collapse:collapse!important;width:100%!important}}
table.dataTable thead th{{background:{_YELLOW}!important;color:{_BG}!important;
  font-weight:700!important;border-bottom:2px solid {_BORDER}!important;padding:10px 12px!important}}
table.dataTable tbody td{{background:{_CARD}!important;color:{_WHITE}!important;
  border-bottom:1px solid {_BORDER}!important;padding:8px 12px!important;font-size:13px}}
table.dataTable tbody tr:nth-child(even) td{{background:{_ALT}!important}}
table.dataTable tbody tr:hover td{{background:#4a4a4a!important}}
table.dataTable tbody tr.selected td{{background:{_YELLOW}!important;color:{_BG}!important}}
.dataTables_wrapper,.dataTables_wrapper *{{color:{_WHITE}!important}}
.dataTables_wrapper .dataTables_filter input,
.dataTables_wrapper .dataTables_length select{{background:{_ALT}!important;color:{_WHITE}!important;
  border:1px solid {_BORDER}!important;border-radius:4px!important;padding:4px 8px!important}}
.dataTables_wrapper .dataTables_paginate .paginate_button{{color:{_WHITE}!important;
  background:{_ALT}!important;border:1px solid {_BORDER}!important;border-radius:4px!important}}
.dataTables_wrapper .dataTables_paginate .paginate_button.current{{background:{_YELLOW}!important;
  color:{_BG}!important;font-weight:700!important}}
.dataTables_wrapper .dataTables_paginate .paginate_button:hover{{background:{_YELLOW}!important;
  color:{_BG}!important}}
.dt-buttons .dt-button{{background:{_ALT}!important;color:{_WHITE}!important;
  border:1px solid {_BORDER}!important;border-radius:4px!important;padding:6px 14px!important;
  font-size:12px!important}}
.dt-buttons .dt-button:hover{{background:{_YELLOW}!important;color:{_BG}!important}}
.dataTables_info{{font-size:12px;color:{_GRAY}!important}}
footer{{text-align:center;padding:16px;font-size:11px;color:{_GRAY};border-top:1px solid {_BORDER};
  margin-top:20px}}
</style>
</head><body>
<div class="top-bar">
  <div><h1>{title}</h1></div>
  <div style="display:flex;align-items:center;gap:16px">
    <span class="date">{date_str}</span>
    <span class="wu-mark">WESTERN UNION</span>
  </div>
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
<footer>Generated by Dr. Data -- Dashboard Intelligence Platform</footer>
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
