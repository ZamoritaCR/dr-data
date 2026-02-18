"""
Interactive HTML Dashboard Builder.

Produces a SINGLE HTML file with:
- All data embedded as JSON
- Filters that ACTUALLY filter every chart on the page
- KPI cards that update when you filter
- Plotly.js charts that re-render on filter change
- Dark professional theme
- Works offline (except Plotly CDN)
- Print-friendly

Drop this in: core/html_dashboard.py

Usage:
    from core.html_dashboard import HTMLDashboardBuilder
    builder = HTMLDashboardBuilder()
    path = builder.build(df, output_dir="./output", title="Sales Dashboard")
    # Open the .html file in any browser -- everything is interactive
"""
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime


class HTMLDashboardBuilder:

    def build(self, df, output_dir="./output", title=None, subtitle=None,
              dashboard_spec=None, data_path=None):
        """
        Build a fully interactive HTML dashboard from a DataFrame.

        Args:
            df: pandas DataFrame
            output_dir: where to save the HTML file
            title: dashboard title
            subtitle: subtitle text
            dashboard_spec: optional dict with custom chart config
            data_path: original file path for reference

        Returns:
            str: absolute path to the generated HTML file
        """
        os.makedirs(output_dir, exist_ok=True)

        if title is None:
            if data_path:
                base = os.path.basename(data_path).rsplit(".", 1)[0]
                title = base.replace("_", " ").replace("-", " ").title() + " Dashboard"
            else:
                title = "Dashboard"

        if subtitle is None:
            subtitle = ""

        # Analyze columns
        analysis = self._analyze(df)

        # Sample data if too large (keep it fast in browser)
        if len(df) > 50000:
            df_embed = df.sample(50000, random_state=42)
        else:
            df_embed = df.copy()

        # Convert dates to strings for JSON
        for col in df_embed.columns:
            if pd.api.types.is_datetime64_any_dtype(df_embed[col]):
                df_embed[col] = df_embed[col].dt.strftime("%Y-%m-%d")

        # Embed data as JSON
        data_json = df_embed.to_json(orient="records", date_format="iso")

        # Build chart configs
        charts = self._plan_charts(df, analysis, dashboard_spec)

        # Build filter configs
        filters = self._plan_filters(df, analysis)

        # KPI configs
        kpis = self._plan_kpis(df, analysis)

        # Assemble HTML
        html = self._render(
            title=title,
            subtitle=subtitle,
            data_json=data_json,
            row_count=len(df),
            col_count=len(df.columns),
            charts_json=json.dumps(charts),
            filters_json=json.dumps(filters),
            kpis_json=json.dumps(kpis),
            columns=list(df.columns),
            timestamp=datetime.now().strftime("%B %d, %Y at %H:%M"),
        )

        safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
        safe = safe.strip().replace(" ", "_") or "Dashboard"
        filepath = os.path.join(output_dir, f"{safe}.html")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return os.path.abspath(filepath)

    def _analyze(self, df):
        """Classify every column."""
        result = {"numeric": [], "categorical": [], "date": [], "high_card": [], "id_like": []}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it looks like an ID
                if df[col].nunique() > len(df) * 0.9:
                    result["id_like"].append(col)
                else:
                    result["numeric"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                result["date"].append(col)
            else:
                # Try parsing as date
                try:
                    sample = df[col].dropna().head(30)
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.6:
                        result["date"].append(col)
                        continue
                except Exception:
                    pass

                nunique = df[col].nunique()
                if nunique <= 30:
                    result["categorical"].append(col)
                elif nunique > len(df) * 0.8:
                    result["id_like"].append(col)
                else:
                    result["high_card"].append(col)

        return result

    def _plan_kpis(self, df, analysis):
        """Plan KPI cards."""
        kpis = []
        for col in analysis["numeric"][:4]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            kpis.append({
                "column": col,
                "label": col.replace("_", " ").replace("-", " ").title(),
                "agg": "sum",
            })
        return kpis

    def _plan_filters(self, df, analysis):
        """Plan interactive filters."""
        filters = []
        for col in analysis["categorical"][:5]:
            vals = sorted(df[col].dropna().unique().tolist())[:100]
            filters.append({"column": col, "values": [str(v) for v in vals]})
        return filters

    def _plan_charts(self, df, analysis, spec=None):
        """Plan which charts to build."""
        charts = []
        num = analysis["numeric"]
        cat = analysis["categorical"]
        date = analysis["date"]

        # 1. Time series
        if date and num:
            charts.append({
                "type": "line", "x": date[0], "y": num[0],
                "title": f"{num[0].replace('_',' ').title()} Over Time",
                "width": "full", "agg": "sum"
            })

        # 2. Bar: first category by first metric
        if cat and num:
            charts.append({
                "type": "bar", "x": cat[0], "y": num[0],
                "title": f"{num[0].replace('_',' ').title()} by {cat[0].replace('_',' ').title()}",
                "width": "half", "agg": "sum", "sort": "desc", "top_n": 15
            })

        # 3. Horizontal bar: second category or second metric
        if len(cat) > 1 and num:
            charts.append({
                "type": "hbar", "x": cat[1], "y": num[0],
                "title": f"{num[0].replace('_',' ').title()} by {cat[1].replace('_',' ').title()}",
                "width": "half", "agg": "sum", "sort": "desc", "top_n": 12
            })
        elif cat and len(num) > 1:
            charts.append({
                "type": "hbar", "x": cat[0], "y": num[1],
                "title": f"{num[1].replace('_',' ').title()} by {cat[0].replace('_',' ').title()}",
                "width": "half", "agg": "sum", "sort": "desc", "top_n": 12
            })

        # 4. Donut for low-cardinality
        low_card = [c for c in cat if df[c].nunique() <= 8]
        if low_card and num:
            charts.append({
                "type": "donut", "x": low_card[0], "y": num[0],
                "title": f"Distribution by {low_card[0].replace('_',' ').title()}",
                "width": "half", "agg": "sum"
            })

        # 5. Scatter if 2+ numeric
        if len(num) >= 2:
            charts.append({
                "type": "scatter", "x": num[0], "y": num[1],
                "title": f"{num[1].replace('_',' ').title()} vs {num[0].replace('_',' ').title()}",
                "width": "half"
            })

        # 6. Second time series
        if date and len(num) > 1:
            charts.append({
                "type": "area", "x": date[0], "y": num[1],
                "title": f"{num[1].replace('_',' ').title()} Trend",
                "width": "full", "agg": "sum"
            })

        # 7. More bars if dimensions available
        if len(cat) > 2 and num:
            charts.append({
                "type": "bar", "x": cat[2], "y": num[0],
                "title": f"{num[0].replace('_',' ').title()} by {cat[2].replace('_',' ').title()}",
                "width": "half", "agg": "sum", "sort": "desc", "top_n": 15
            })

        # 8. Second donut
        if len(low_card) > 1 and num:
            charts.append({
                "type": "donut", "x": low_card[1], "y": num[0],
                "title": f"Distribution by {low_card[1].replace('_',' ').title()}",
                "width": "half", "agg": "sum"
            })

        return charts

    def _render(self, title, subtitle, data_json, row_count, col_count,
                charts_json, filters_json, kpis_json, columns, timestamp):
        """Render the complete HTML with embedded JS that makes everything interactive."""

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --bg:#0D0D0D;--surface:#1A1A1A;--elevated:#262626;--border:#333333;
  --text:#FFFFFF;--text2:#B0B0B0;--muted:#808080;
  --accent:#FFE600;--accent2:#FFDE00;--green:#238636;--amber:#d29922;--red:#da3633;--purple:#a371f7;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:14px;}}
.hdr{{background:linear-gradient(135deg,#0D0D0D,#1A1A1A);border-bottom:1px solid var(--border);padding:20px 32px;position:relative;display:flex;align-items:center;justify-content:space-between;}}
.hdr::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent);}}
.hdr h1{{font-size:20px;font-weight:600;letter-spacing:-0.3px;}}
.hdr .meta{{display:flex;gap:20px;margin-top:6px;font-size:12px;color:var(--text2);flex-wrap:wrap;}}
.hdr .wu-badge{{background:var(--accent);color:#000;font-weight:900;font-size:11px;padding:5px 12px;border-radius:3px;letter-spacing:1px;white-space:nowrap;}}
.wrap{{max-width:1440px;margin:0 auto;padding:20px 32px;}}

/* KPIs */
.kpi-row{{display:flex;gap:14px;margin-bottom:20px;flex-wrap:wrap;}}
.kpi{{flex:1;min-width:160px;background:linear-gradient(135deg,var(--surface),var(--elevated));border:1px solid var(--border);border-radius:10px;padding:18px 22px;text-align:center;transition:border-color .2s;}}
.kpi:hover{{border-color:var(--accent);}}
.kpi .v{{font-size:26px;font-weight:700;line-height:1.2;}}
.kpi .l{{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;margin-top:4px;}}
.kpi .s{{font-size:11px;color:var(--muted);margin-top:2px;}}

/* Stats */
.stats{{display:flex;gap:16px;margin-bottom:16px;padding:10px 16px;background:var(--surface);border:1px solid var(--border);border-radius:8px;font-size:12px;color:var(--text2);flex-wrap:wrap;}}
.stats strong{{color:var(--text);}}

/* Filters */
.fbar{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;align-items:flex-end;}}
.fgrp label{{display:block;font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;}}
.fgrp select{{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:7px 28px 7px 10px;font-size:13px;min-width:140px;appearance:none;cursor:pointer;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%238b949e' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
background-repeat:no-repeat;background-position:right 8px center;}}
.fgrp select:hover{{border-color:var(--accent);}}
.fgrp .reset{{background:var(--elevated);color:var(--text2);border:1px solid var(--border);border-radius:6px;padding:7px 14px;font-size:12px;cursor:pointer;}}
.fgrp .reset:hover{{border-color:var(--red);color:var(--red);}}

/* Charts */
.cgrid{{display:flex;flex-wrap:wrap;gap:16px;}}
.cbox{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;min-height:340px;transition:border-color .2s;}}
.cbox:hover{{border-color:var(--accent);}}

.footer{{border-top:1px solid var(--border);padding:16px 32px;text-align:center;font-size:11px;color:var(--muted);margin-top:32px;}}
.filtered-badge{{display:inline-block;background:var(--accent);color:#000;font-size:10px;font-weight:600;padding:2px 8px;border-radius:10px;margin-left:8px;}}

@media(max-width:900px){{.cgrid>div{{width:100%!important;}}.kpi-row{{flex-direction:column;}}.wrap{{padding:12px;}}}}
@media print{{body{{background:#fff;color:#1a1a1a;}}.hdr{{background:#fff;border-bottom:3px solid #FFE600;}}.hdr::before{{background:#FFE600;}}.hdr h1{{color:#1a1a1a;}}.hdr .wu-badge{{background:#FFE600;-webkit-print-color-adjust:exact;print-color-adjust:exact;}}.kpi,.cbox{{background:#fff;border-color:#ddd;}}.kpi .v{{color:#1a1a1a;}}.stats{{background:#f5f5f5;}}footer{{color:#666;}}}}
</style>
</head>
<body>

<div class="hdr">
  <div>
    <h1>{title}<span id="filterBadge"></span></h1>
    <div class="meta">
      <span>{subtitle}</span>
      <span id="rowCount">{row_count:,} records</span>
      <span>{col_count} fields</span>
      <span>Built {timestamp}</span>
    </div>
  </div>
  <span class="wu-badge">WESTERN UNION</span>
</div>

<div class="wrap">
  <div class="kpi-row" id="kpiRow"></div>
  <div class="stats" id="statsBar"></div>
  <div class="fbar" id="filterBar"></div>
  <div class="cgrid" id="chartGrid"></div>
</div>

<div class="footer">Built by Dr. Data -- Western Union Analytics</div>

<script>
// ============================================================
// ALL DATA EMBEDDED -- Everything runs client-side from here
// ============================================================
var RAW_DATA = {data_json};
var CHARTS = {charts_json};
var FILTERS = {filters_json};
var KPIS = {kpis_json};
var TOTAL_ROWS = {row_count};
var COLORS = ['#FFE600','#FFDE00','#E6CF00','#D4A017','#F5C842','#FFB800','#C9A200','#FFF176','#FFD54F','#E0B400','#B8960F','#FFC107'];
var PC = {{responsive:true, displayModeBar:false}};
var BL = {{
  paper_bgcolor:'#1A1A1A', plot_bgcolor:'#1A1A1A',
  font:{{color:'#FFFFFF',family:'Inter,system-ui,sans-serif',size:12}},
  margin:{{l:50,r:20,t:40,b:40}},
  xaxis:{{gridcolor:'#333333',linecolor:'#333333',zerolinecolor:'#333333'}},
  yaxis:{{gridcolor:'#333333',linecolor:'#333333',zerolinecolor:'#333333'}},
  colorway:COLORS,
  legend:{{font:{{color:'#B0B0B0',size:11}},bgcolor:'transparent'}},
  hoverlabel:{{bgcolor:'#262626',font:{{color:'#FFFFFF',size:12}},bordercolor:'#333333'}}
}};
function ML(o){{ var b=JSON.parse(JSON.stringify(BL)); for(var k in o) b[k]=o[k]; return b; }}

// Current filter state
var activeFilters = {{}};

// ============================================================
// FILTER ENGINE -- Filters actually filter everything
// ============================================================
function getFilteredData() {{
  var d = RAW_DATA;
  for (var col in activeFilters) {{
    var val = activeFilters[col];
    if (val && val !== '__ALL__') {{
      d = d.filter(function(row) {{ return String(row[col]) === val; }});
    }}
  }}
  return d;
}}

function onFilterChange() {{
  // Read all filter selects
  activeFilters = {{}};
  var selects = document.querySelectorAll('.fgrp select');
  selects.forEach(function(s) {{
    var col = s.dataset.column;
    var val = s.value;
    if (val && val !== '__ALL__') activeFilters[col] = val;
  }});

  var filtered = getFilteredData();
  var isFiltered = Object.keys(activeFilters).length > 0;

  // Update badge
  var badge = document.getElementById('filterBadge');
  badge.innerHTML = isFiltered ? '<span class="filtered-badge">' + filtered.length.toLocaleString() + ' of ' + TOTAL_ROWS.toLocaleString() + ' rows</span>' : '';

  // Update row count
  document.getElementById('rowCount').textContent = filtered.length.toLocaleString() + ' records';

  // Rebuild KPIs
  renderKPIs(filtered);

  // Rebuild all charts
  renderCharts(filtered);
}}

function resetFilters() {{
  var selects = document.querySelectorAll('.fgrp select');
  selects.forEach(function(s) {{ s.value = '__ALL__'; }});
  activeFilters = {{}};
  onFilterChange();
}}

// ============================================================
// KPI RENDERING
// ============================================================
function formatNum(v) {{
  if (Math.abs(v) >= 1000000) return '$' + (v/1000000).toFixed(1) + 'M';
  if (Math.abs(v) >= 10000) return '$' + (v/1000).toFixed(1) + 'K';
  if (Math.abs(v) >= 100) return v.toLocaleString(undefined, {{maximumFractionDigits:0}});
  if (Math.abs(v) < 1 && Math.abs(v) > 0) return (v*100).toFixed(1) + '%';
  return v.toLocaleString(undefined, {{maximumFractionDigits:2}});
}}

function renderKPIs(data) {{
  var html = '';
  var kColors = ['#FFE600','#FFDE00','#F5C842','#FFB800'];
  KPIS.forEach(function(kpi, i) {{
    var vals = data.map(function(r){{ return parseFloat(r[kpi.column]); }}).filter(function(v){{ return !isNaN(v); }});
    if (vals.length === 0) return;
    var total = vals.reduce(function(a,b){{ return a+b; }}, 0);
    var avg = total / vals.length;
    var color = kColors[i % kColors.length];
    html += '<div class="kpi"><div class="v" style="color:'+color+'">'+formatNum(total)+'</div>';
    html += '<div class="l">'+kpi.label+'</div>';
    html += '<div class="s">Avg: '+formatNum(avg)+'</div></div>';
  }});
  document.getElementById('kpiRow').innerHTML = html;
}}

// ============================================================
// CHART RENDERING -- All charts rebuild on filter
// ============================================================
function aggregate(data, xCol, yCol, agg, sortDir, topN) {{
  var groups = {{}};
  data.forEach(function(r) {{
    var key = String(r[xCol] || 'Unknown');
    var val = parseFloat(r[yCol]);
    if (isNaN(val)) return;
    if (!groups[key]) groups[key] = [];
    groups[key].push(val);
  }});

  var result = [];
  for (var key in groups) {{
    var vals = groups[key];
    var v;
    if (agg === 'avg' || agg === 'mean') v = vals.reduce(function(a,b){{return a+b;}},0) / vals.length;
    else if (agg === 'count') v = vals.length;
    else if (agg === 'min') v = Math.min.apply(null, vals);
    else if (agg === 'max') v = Math.max.apply(null, vals);
    else v = vals.reduce(function(a,b){{return a+b;}},0); // sum
    result.push({{x: key, y: Math.round(v*100)/100}});
  }}

  if (sortDir === 'desc') result.sort(function(a,b){{return b.y - a.y;}});
  else if (sortDir === 'asc') result.sort(function(a,b){{return a.y - b.y;}});

  if (topN) result = result.slice(0, topN);
  return result;
}}

function timeAggregate(data, dateCol, yCol, agg) {{
  // Parse dates and bin by appropriate period
  var withDates = data.map(function(r) {{
    var d = new Date(r[dateCol]);
    var val = parseFloat(r[yCol]);
    return {{date: d, val: val, valid: !isNaN(d.getTime()) && !isNaN(val)}};
  }}).filter(function(r){{ return r.valid; }});

  if (withDates.length === 0) return {{x:[], y:[]}};

  withDates.sort(function(a,b){{ return a.date - b.date; }});
  var rangeMs = withDates[withDates.length-1].date - withDates[0].date;
  var rangeDays = rangeMs / 86400000;

  // Determine period
  var keyFn;
  if (rangeDays > 730) keyFn = function(d){{ return d.getFullYear() + ' Q' + (Math.floor(d.getMonth()/3)+1); }};
  else if (rangeDays > 180) keyFn = function(d){{ return d.getFullYear() + '-' + String(d.getMonth()+1).padStart(2,'0'); }};
  else if (rangeDays > 30) keyFn = function(d) {{
    var onejan = new Date(d.getFullYear(), 0, 1);
    var week = Math.ceil(((d - onejan) / 86400000 + onejan.getDay() + 1) / 7);
    return d.getFullYear() + '-W' + String(week).padStart(2,'0');
  }};
  else keyFn = function(d){{ return d.toISOString().slice(0,10); }};

  var groups = {{}};
  var order = [];
  withDates.forEach(function(r) {{
    var key = keyFn(r.date);
    if (!groups[key]) {{ groups[key] = []; order.push(key); }}
    groups[key].push(r.val);
  }});

  var x = [], y = [];
  order.forEach(function(key) {{
    x.push(key);
    var vals = groups[key];
    var v = vals.reduce(function(a,b){{return a+b;}},0);
    if (agg === 'avg') v = v / vals.length;
    y.push(Math.round(v*100)/100);
  }});

  return {{x:x, y:y}};
}}

function renderCharts(data) {{
  var grid = document.getElementById('chartGrid');
  grid.innerHTML = '';

  CHARTS.forEach(function(chart, i) {{
    var divId = 'chart_' + i;
    var w = chart.width === 'full' ? '100%' : 'calc(50% - 8px)';
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'width:'+w+';min-width:300px;';
    wrapper.innerHTML = '<div class="cbox" id="'+divId+'"></div>';
    grid.appendChild(wrapper);

    var type = chart.type;
    var xCol = chart.x;
    var yCol = chart.y;
    var agg = chart.agg || 'sum';
    var title = chart.title || '';

    if (type === 'line' || type === 'area') {{
      var ta = timeAggregate(data, xCol, yCol, agg);
      var fill = type === 'area' ? 'tozeroy' : 'none';
      var fillColor = type === 'area' ? 'rgba(255,230,0,0.10)' : 'rgba(255,230,0,0.06)';
      var lineColor = type === 'area' ? '#FFDE00' : '#FFE600';
      Plotly.newPlot(divId, [{{
        x:ta.x, y:ta.y, type:'scatter', mode:'lines+markers',
        line:{{color:lineColor, width:2.5, shape:'spline'}},
        marker:{{size:4, color:lineColor}},
        fill: type === 'line' ? 'tozeroy' : fill,
        fillcolor: type === 'line' ? 'rgba(255,230,0,0.06)' : fillColor,
        hovertemplate:'%{{x}}<br>%{{y:,.0f}}<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14}}}}}}), PC);

    }} else if (type === 'bar') {{
      var ag = aggregate(data, xCol, yCol, agg, chart.sort, chart.top_n);
      Plotly.newPlot(divId, [{{
        x:ag.map(function(r){{return r.x;}}),
        y:ag.map(function(r){{return r.y;}}),
        type:'bar', marker:{{color:'#FFE600'}},
        hovertemplate:'%{{x}}<br>%{{y:,.0f}}<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14}}}}}}), PC);

    }} else if (type === 'hbar') {{
      var ag = aggregate(data, xCol, yCol, agg, chart.sort, chart.top_n);
      ag.reverse();
      Plotly.newPlot(divId, [{{
        y:ag.map(function(r){{return r.x;}}),
        x:ag.map(function(r){{return r.y;}}),
        type:'bar', orientation:'h', marker:{{color:'#FFDE00'}},
        hovertemplate:'%{{y}}: %{{x:,.0f}}<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14}}}},yaxis:{{automargin:true}}}}), PC);

    }} else if (type === 'donut') {{
      var ag = aggregate(data, xCol, yCol, agg, 'desc', 8);
      Plotly.newPlot(divId, [{{
        labels:ag.map(function(r){{return r.x;}}),
        values:ag.map(function(r){{return r.y;}}),
        type:'pie', hole:0.5,
        textinfo:'percent+label',
        textfont:{{color:'#FFFFFF',size:11}},
        marker:{{colors:COLORS, line:{{color:'#0D0D0D',width:2}}}},
        hovertemplate:'%{{label}}<br>%{{value:,.0f}} (%{{percent}})<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14}}}},showlegend:true}}), PC);

    }} else if (type === 'scatter') {{
      var pts = data.map(function(r){{
        return {{x:parseFloat(r[xCol]), y:parseFloat(r[yCol])}};
      }}).filter(function(r){{ return !isNaN(r.x) && !isNaN(r.y); }}).slice(0, 2000);
      Plotly.newPlot(divId, [{{
        x:pts.map(function(r){{return r.x;}}),
        y:pts.map(function(r){{return r.y;}}),
        type:'scatter', mode:'markers',
        marker:{{color:'#FFE600',size:5,opacity:0.6}},
        hovertemplate:xCol+': %{{x:,.1f}}<br>'+yCol+': %{{y:,.1f}}<extra></extra>'
      }}], ML({{
        title:{{text:title,font:{{size:14}}}},
        xaxis:{{title:xCol.replace(/_/g,' ')}},
        yaxis:{{title:yCol.replace(/_/g,' ')}}
      }}), PC);
    }}
  }});
}}

// ============================================================
// BUILD UI
// ============================================================
function buildFilters() {{
  var bar = document.getElementById('filterBar');
  if (FILTERS.length === 0) {{ bar.style.display = 'none'; return; }}

  var html = '';
  FILTERS.forEach(function(f) {{
    var label = f.column.replace(/_/g,' ').replace(/\\b\\w/g, function(c){{return c.toUpperCase();}});
    html += '<div class="fgrp"><label>'+label+'</label>';
    html += '<select data-column="'+f.column+'" onchange="onFilterChange()">';
    html += '<option value="__ALL__">All</option>';
    f.values.forEach(function(v) {{
      html += '<option value="'+v+'">'+v+'</option>';
    }});
    html += '</select></div>';
  }});
  html += '<div class="fgrp"><button class="reset" onclick="resetFilters()">Reset Filters</button></div>';
  bar.innerHTML = html;
}}

function buildStats() {{
  var d = RAW_DATA;
  var cols = Object.keys(d[0] || {{}});
  var numCols = cols.filter(function(c){{ return !isNaN(parseFloat(d[0][c])); }});
  var html = '<strong>' + d.length.toLocaleString() + '</strong> rows';
  html += ' &middot; <strong>' + cols.length + '</strong> columns';
  html += ' &middot; <strong>' + numCols.length + '</strong> measures';
  html += ' &middot; <strong>' + (cols.length - numCols.length) + '</strong> dimensions';
  document.getElementById('statsBar').innerHTML = html;
}}

// ============================================================
// INIT
// ============================================================
buildFilters();
buildStats();
renderKPIs(RAW_DATA);
renderCharts(RAW_DATA);
</script>
</body>
</html>'''


# ================================================================
# STANDALONE TEST
# ================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        ext = sys.argv[1].rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(sys.argv[1])
        elif ext in ("xlsx", "xls"):
            from app.file_handler import load_excel_smart
            df, _ = load_excel_smart(sys.argv[1])
        else:
            df = pd.read_csv(sys.argv[1])
        title = sys.argv[2] if len(sys.argv) > 2 else None
        path = sys.argv[1]
    else:
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "Order_Date": pd.date_range("2022-01-01", periods=n, freq="D"),
            "Region": np.random.choice(["North", "South", "East", "West"], n),
            "Category": np.random.choice(["Electronics", "Furniture", "Clothing", "Food"], n),
            "Segment": np.random.choice(["Consumer", "Corporate", "Home Office"], n),
            "Sales": np.random.uniform(50, 5000, n).round(2),
            "Profit": np.random.uniform(-500, 2000, n).round(2),
            "Quantity": np.random.randint(1, 50, n),
            "Discount": np.random.uniform(0, 0.4, n).round(2),
        })
        title = "Sample Dashboard"
        path = "sample"

    builder = HTMLDashboardBuilder()
    out = builder.build(df, output_dir="./output", title=title, data_path=path)
    print(f"Dashboard: {out}")
    print(f"Size: {os.path.getsize(out)/1024:.0f} KB")

    try:
        import webbrowser
        webbrowser.open(f"file:///{os.path.abspath(out).replace(os.sep, '/')}")
    except Exception:
        pass
