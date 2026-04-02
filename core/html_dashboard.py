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
        data_json = df_embed.to_json(orient="records", date_format="iso").replace("</script>", "<\\/script>")

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
  --bg:#0F172A;--surface:#1E293B;--elevated:#334155;--border:#475569;
  --text:#F1F5F9;--text2:#94A3B8;--muted:#64748B;
  --accent:#6366F1;--accent2:#818CF8;--green:#10B981;--amber:#F59E0B;--red:#EF4444;--purple:#8B5CF6;--cyan:#22D3EE;--pink:#EC4899;
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.5;}}

/* Header */
.hdr{{background:linear-gradient(135deg,#0F172A 0%,#1E293B 100%);border-bottom:1px solid var(--border);padding:24px 36px;position:relative;display:flex;align-items:center;justify-content:space-between;}}
.hdr::before{{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#6366F1,#22D3EE,#10B981,#F59E0B);}}
.hdr h1{{font-size:22px;font-weight:700;letter-spacing:-0.5px;background:linear-gradient(135deg,#F1F5F9,#94A3B8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}}
.hdr .meta{{display:flex;gap:20px;margin-top:6px;font-size:12px;color:var(--text2);flex-wrap:wrap;}}
.hdr .wu-badge{{background:linear-gradient(135deg,var(--accent),var(--purple));color:#fff;font-weight:700;font-size:11px;padding:5px 14px;border-radius:6px;letter-spacing:.5px;white-space:nowrap;}}
.wrap{{max-width:1480px;margin:0 auto;padding:24px 36px;}}

/* KPI Cards */
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px;}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;position:relative;overflow:hidden;transition:all .25s ease;}}
.kpi::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0;}}
.kpi:nth-child(1)::before{{background:linear-gradient(90deg,#6366F1,#818CF8);}}
.kpi:nth-child(2)::before{{background:linear-gradient(90deg,#22D3EE,#67E8F9);}}
.kpi:nth-child(3)::before{{background:linear-gradient(90deg,#10B981,#34D399);}}
.kpi:nth-child(4)::before{{background:linear-gradient(90deg,#F59E0B,#FBBF24);}}
.kpi:hover{{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 8px 25px rgba(99,102,241,0.15);}}
.kpi .v{{font-size:28px;font-weight:800;line-height:1.1;margin-bottom:2px;}}
.kpi .l{{font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:.8px;font-weight:500;}}
.kpi .s{{font-size:12px;color:var(--muted);margin-top:4px;}}

/* Stats bar */
.stats{{display:flex;gap:20px;margin-bottom:20px;padding:12px 20px;background:var(--surface);border:1px solid var(--border);border-radius:10px;font-size:12px;color:var(--text2);flex-wrap:wrap;align-items:center;}}
.stats strong{{color:var(--text);font-weight:600;}}

/* Filters */
.fbar{{display:flex;gap:14px;margin-bottom:24px;flex-wrap:wrap;align-items:flex-end;padding:16px 20px;background:var(--surface);border:1px solid var(--border);border-radius:12px;}}
.fgrp label{{display:block;font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;font-weight:600;}}
.fgrp select{{background:#0F172A;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:8px 32px 8px 12px;font-size:13px;min-width:160px;appearance:none;cursor:pointer;font-family:inherit;transition:border-color .2s;
background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%2394A3B8' d='M2 4l4 4 4-4'/%3E%3C/svg%3E");
background-repeat:no-repeat;background-position:right 10px center;}}
.fgrp select:hover,.fgrp select:focus{{border-color:var(--accent);outline:none;box-shadow:0 0 0 3px rgba(99,102,241,0.15);}}
.fgrp .reset{{background:transparent;color:var(--red);border:1px solid var(--red);border-radius:8px;padding:8px 16px;font-size:12px;cursor:pointer;font-family:inherit;font-weight:500;transition:all .2s;}}
.fgrp .reset:hover{{background:var(--red);color:#fff;}}

/* Chart grid */
.cgrid{{display:flex;flex-wrap:wrap;gap:20px;}}
.cbox{{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:18px;min-height:360px;transition:all .25s ease;position:relative;}}
.cbox:hover{{border-color:var(--accent);box-shadow:0 4px 20px rgba(99,102,241,0.08);}}
.cbox .insight{{font-size:12px;color:var(--text2);padding:10px 4px 0;line-height:1.6;border-top:1px solid rgba(71,85,105,0.5);margin-top:10px;}}
.cbox .insight strong{{color:var(--cyan);font-weight:600;}}

/* Footer */
.footer{{border-top:1px solid var(--border);padding:20px 36px;text-align:center;font-size:11px;color:var(--muted);margin-top:40px;}}

/* Filter badge */
.filtered-badge{{display:inline-block;background:linear-gradient(135deg,var(--accent),var(--purple));color:#fff;font-size:10px;font-weight:600;padding:3px 10px;border-radius:12px;margin-left:10px;}}

/* Responsive */
@media(max-width:900px){{.cgrid>div{{width:100%!important;}}.kpi-row{{grid-template-columns:1fr 1fr;}}.wrap{{padding:16px;}}.fbar{{flex-direction:column;}}}}

/* Print */
@media print{{
  body{{background:#fff;color:#1e293b;}}
  .hdr{{background:#fff;border-bottom:3px solid #6366F1;}}
  .hdr::before{{background:linear-gradient(90deg,#6366F1,#22D3EE,#10B981);}}
  .hdr h1{{-webkit-text-fill-color:#1e293b;}}
  .kpi,.cbox{{background:#fff;border-color:#e2e8f0;}}
  .kpi .v{{color:#1e293b;}}
  .stats,.fbar{{background:#f8fafc;border-color:#e2e8f0;}}
  .footer{{color:#94a3b8;}}
}}
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
  <div class="wu-badge">DR. DATA</div>
</div>

<div class="wrap">
  <div class="kpi-row" id="kpiRow"></div>
  <div class="stats" id="statsBar"></div>
  <div class="fbar" id="filterBar"></div>
  <div class="cgrid" id="chartGrid"></div>
</div>

<div class="footer">Built by Dr. Data | The Art of the Possible | Powered by Claude Opus 4</div>

<script>
// ============================================================
// ALL DATA EMBEDDED -- Everything runs client-side from here
// ============================================================
var RAW_DATA = {data_json};
var CHARTS = {charts_json};
var FILTERS = {filters_json};
var KPIS = {kpis_json};
var TOTAL_ROWS = {row_count};
var COLORS = ['#6366F1','#22D3EE','#F59E0B','#10B981','#F43F5E','#8B5CF6','#EC4899','#14B8A6','#F97316','#3B82F6','#84CC16','#EF4444'];
var PC = {{responsive:true, displayModeBar:false}};
var BL = {{
  paper_bgcolor:'#111827', plot_bgcolor:'#111827',
  font:{{color:'#E5E7EB',family:'Inter,system-ui,sans-serif',size:12}},
  margin:{{l:56,r:24,t:48,b:48}},
  xaxis:{{gridcolor:'#1F2937',linecolor:'#374151',zerolinecolor:'#374151',tickfont:{{color:'#9CA3AF'}}}},
  yaxis:{{gridcolor:'#1F2937',linecolor:'#374151',zerolinecolor:'#374151',tickfont:{{color:'#9CA3AF'}}}},
  colorway:COLORS,
  legend:{{font:{{color:'#D1D5DB',size:11}},bgcolor:'transparent',orientation:'h',y:-0.15}},
  hoverlabel:{{bgcolor:'#1F2937',font:{{color:'#F9FAFB',size:12}},bordercolor:'#4B5563'}},
  bargap:0.2, bargroupgap:0.08
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
  var kColors = ['#6366F1','#22D3EE','#10B981','#F59E0B','#F43F5E','#8B5CF6'];
  KPIS.forEach(function(kpi, i) {{
    var vals = data.map(function(r){{ return parseFloat(r[kpi.column]); }}).filter(function(v){{ return !isNaN(v); }});
    if (vals.length === 0) return;
    var total = vals.reduce(function(a,b){{ return a+b; }}, 0);
    var avg = total / vals.length;
    var min = Math.min.apply(null, vals);
    var max = Math.max.apply(null, vals);
    var color = kColors[i % kColors.length];
    html += '<div class="kpi">';
    html += '<div class="v" style="color:'+color+'">'+formatNum(total)+'</div>';
    html += '<div class="l">'+kpi.label+'</div>';
    html += '<div class="s">Avg: '+formatNum(avg)+' | Range: '+formatNum(min)+' - '+formatNum(max)+'</div>';
    html += '</div>';
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

function chartInsight(chart, data) {{
  var type = chart.type, xCol = chart.x, yCol = chart.y;
  var agg = chart.agg || 'sum';
  var yLabel = (yCol||'').replace(/_/g,' ');
  var xLabel = (xCol||'').replace(/_/g,' ');

  if (type === 'line' || type === 'area') {{
    var ta = timeAggregate(data, xCol, yCol, agg);
    if (ta.y.length < 2) return '';
    var total = ta.y.reduce(function(a,b){{return a+b;}},0);
    var maxIdx = ta.y.indexOf(Math.max.apply(null, ta.y));
    var first = ta.y[0], last = ta.y[ta.y.length-1];
    var pctChange = first !== 0 ? Math.round((last - first) / Math.abs(first) * 100) : 0;
    var trend = pctChange > 5 ? 'upward' : pctChange < -5 ? 'downward' : 'stable';
    return '<strong>Talking point:</strong> Total ' + yLabel + ' of ' + formatNum(total) +
      '. Peaked at ' + ta.x[maxIdx] + ' (' + formatNum(ta.y[maxIdx]) + '). ' +
      'Overall <strong>' + trend + '</strong> trend (' + (pctChange >= 0 ? '+' : '') + pctChange + '%).';
  }}

  if (type === 'bar' || type === 'hbar') {{
    var ag = aggregate(data, xCol, yCol, agg, 'desc', chart.top_n || 15);
    if (ag.length === 0) return '';
    var total = ag.reduce(function(a,r){{return a + r.y;}}, 0);
    var topPct = total > 0 ? Math.round(ag[0].y / total * 100) : 0;
    var msg = '<strong>Talking point:</strong> Top ' + xLabel + ' is <strong>' + ag[0].x + '</strong> at ' + formatNum(ag[0].y) + ' (' + topPct + '% of total).';
    if (ag.length >= 3) {{
      var top3 = ag.slice(0,3).reduce(function(a,r){{return a + r.y;}}, 0);
      var top3Pct = total > 0 ? Math.round(top3 / total * 100) : 0;
      msg += ' Top 3 account for ' + top3Pct + '% of all ' + yLabel + '.';
    }}
    return msg;
  }}

  if (type === 'donut') {{
    var ag = aggregate(data, xCol, yCol, agg, 'desc', 8);
    if (ag.length === 0) return '';
    var total = ag.reduce(function(a,r){{return a + r.y;}}, 0);
    var topPct = total > 0 ? Math.round(ag[0].y / total * 100) : 0;
    return '<strong>Talking point:</strong> <strong>' + ag[0].x + '</strong> dominates at ' + topPct + '% of ' + yLabel + '. ' + ag.length + ' segments shown.';
  }}

  if (type === 'scatter') {{
    var pts = data.map(function(r){{return {{x:parseFloat(r[xCol]),y:parseFloat(r[yCol])}};
    }}).filter(function(r){{return !isNaN(r.x) && !isNaN(r.y);}});
    if (pts.length < 5) return '';
    var mx = pts.reduce(function(a,r){{return a+r.x;}},0)/pts.length;
    var my = pts.reduce(function(a,r){{return a+r.y;}},0)/pts.length;
    var num=0, dx2=0, dy2=0;
    pts.forEach(function(r){{ num+=(r.x-mx)*(r.y-my); dx2+=(r.x-mx)*(r.x-mx); dy2+=(r.y-my)*(r.y-my); }});
    var corr = (dx2>0&&dy2>0) ? num/Math.sqrt(dx2*dy2) : 0;
    var strength = Math.abs(corr) > 0.7 ? 'strong' : Math.abs(corr) > 0.3 ? 'moderate' : 'weak';
    var dir = corr > 0.1 ? 'positive' : corr < -0.1 ? 'negative' : 'no clear';
    return '<strong>Talking point:</strong> ' + pts.length + ' data points. Shows a <strong>' + strength + ' ' + dir + '</strong> relationship (r=' + corr.toFixed(2) + ') between ' + xLabel + ' and ' + yLabel + '.';
  }}

  return '';
}}

function renderCharts(data) {{
  var grid = document.getElementById('chartGrid');
  grid.innerHTML = '';

  CHARTS.forEach(function(chart, i) {{
    var divId = 'chart_' + i;
    var insId = 'insight_' + i;
    var w = chart.width === 'full' ? '100%' : 'calc(50% - 8px)';
    var wrapper = document.createElement('div');
    wrapper.style.cssText = 'width:'+w+';min-width:300px;';
    wrapper.innerHTML = '<div class="cbox"><div id="'+divId+'"></div><div class="insight" id="'+insId+'"></div></div>';
    grid.appendChild(wrapper);

    var type = chart.type;
    var xCol = chart.x;
    var yCol = chart.y;
    var agg = chart.agg || 'sum';
    var title = chart.title || '';

    var ci = i % COLORS.length;
    var chartColor = COLORS[ci];
    var chartColor2 = COLORS[(ci+1) % COLORS.length];

    if (type === 'line' || type === 'area') {{
      var ta = timeAggregate(data, xCol, yCol, agg);
      var fillAlpha = type === 'area' ? '0.15' : '0.05';
      var fc = chartColor.replace('#','');
      var r=parseInt(fc.substr(0,2),16), g=parseInt(fc.substr(2,2),16), b=parseInt(fc.substr(4,2),16);
      var fillRgba = 'rgba('+r+','+g+','+b+','+fillAlpha+')';
      Plotly.newPlot(divId, [{{
        x:ta.x, y:ta.y, type:'scatter', mode:'lines+markers',
        line:{{color:chartColor, width:3, shape:'spline'}},
        marker:{{size:5, color:chartColor, line:{{color:'#0F172A',width:1}}}},
        fill:'tozeroy', fillcolor:fillRgba,
        hovertemplate:'%{{x}}<br><b>%{{y:,.0f}}</b><extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14,color:'#E5E7EB'}}}},
        xaxis:{{gridcolor:'#1F2937',showgrid:true}},
        yaxis:{{gridcolor:'#1F2937',showgrid:true}}
      }}), PC);

    }} else if (type === 'bar') {{
      var ag = aggregate(data, xCol, yCol, agg, chart.sort, chart.top_n);
      // Gradient-like coloring: each bar a shade from the palette
      var barColors = ag.map(function(_,idx){{ return COLORS[idx % COLORS.length]; }});
      Plotly.newPlot(divId, [{{
        x:ag.map(function(r){{return r.x;}}),
        y:ag.map(function(r){{return r.y;}}),
        type:'bar',
        marker:{{color:barColors, line:{{color:'#0F172A',width:1}}, opacity:0.9}},
        hovertemplate:'<b>%{{x}}</b><br>%{{y:,.0f}}<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14,color:'#E5E7EB'}}}},
        xaxis:{{tickangle:ag.length>8?-45:0}}
      }}), PC);

    }} else if (type === 'hbar') {{
      var ag = aggregate(data, xCol, yCol, agg, chart.sort, chart.top_n);
      ag.reverse();
      var hColors = ag.map(function(_,idx){{ return COLORS[idx % COLORS.length]; }});
      Plotly.newPlot(divId, [{{
        y:ag.map(function(r){{return r.x;}}),
        x:ag.map(function(r){{return r.y;}}),
        type:'bar', orientation:'h',
        marker:{{color:hColors, line:{{color:'#0F172A',width:1}}, opacity:0.9}},
        hovertemplate:'<b>%{{y}}</b>: %{{x:,.0f}}<extra></extra>'
      }}], ML({{title:{{text:title,font:{{size:14,color:'#E5E7EB'}}}},yaxis:{{automargin:true}}}}), PC);

    }} else if (type === 'donut') {{
      var ag = aggregate(data, xCol, yCol, agg, 'desc', 8);
      Plotly.newPlot(divId, [{{
        labels:ag.map(function(r){{return r.x;}}),
        values:ag.map(function(r){{return r.y;}}),
        type:'pie', hole:0.55,
        textinfo:'percent+label',
        textposition:'inside',
        textfont:{{color:'#F1F5F9',size:11,family:'Inter'}},
        marker:{{colors:COLORS.slice(0,ag.length), line:{{color:'#0F172A',width:2}}}},
        hovertemplate:'<b>%{{label}}</b><br>%{{value:,.0f}} (%{{percent}})<extra></extra>',
        pull:ag.map(function(_,idx){{return idx===0?0.04:0;}})
      }}], ML({{title:{{text:title,font:{{size:14,color:'#E5E7EB'}}}},showlegend:true,
        legend:{{font:{{color:'#D1D5DB',size:11}},orientation:'v',x:1.02,y:0.5}}
      }}), PC);

    }} else if (type === 'scatter') {{
      var pts = data.map(function(r){{
        return {{x:parseFloat(r[xCol]), y:parseFloat(r[yCol])}};
      }}).filter(function(r){{ return !isNaN(r.x) && !isNaN(r.y); }}).slice(0, 2000);
      Plotly.newPlot(divId, [{{
        x:pts.map(function(r){{return r.x;}}),
        y:pts.map(function(r){{return r.y;}}),
        type:'scatter', mode:'markers',
        marker:{{color:chartColor,size:6,opacity:0.7,
          line:{{color:'#0F172A',width:0.5}}}},
        hovertemplate:xCol+': <b>%{{x:,.1f}}</b><br>'+yCol+': <b>%{{y:,.1f}}</b><extra></extra>'
      }}], ML({{
        title:{{text:title,font:{{size:14,color:'#E5E7EB'}}}},
        xaxis:{{title:{{text:xCol.replace(/_/g,' '),font:{{color:'#94A3B8',size:12}}}}}},
        yaxis:{{title:{{text:yCol.replace(/_/g,' '),font:{{color:'#94A3B8',size:12}}}}}}
      }}), PC);
    }}

    // Talking point below each chart
    var ins = chartInsight(chart, data);
    if (ins) document.getElementById(insId).innerHTML = ins;
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
