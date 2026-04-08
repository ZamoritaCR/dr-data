"""
Tableau-to-Power BI Formula Translation Audit Report Generator.

Produces a self-contained HTML report with:
- DQ score and dimension breakdown
- KPI scorecard
- Warnings
- Interactive formula translation table (search, filter, expand, copy DAX)
- Summary metadata
"""
import html as _html
from datetime import datetime


# ---------------------------------------------------------------------------
#  Tier Classification
# ---------------------------------------------------------------------------

def _classify_tier(measure, calc_detail=None):
    """Classify a measure into a translation tier.

    Returns (tier, confidence, method_label, deductions_text).
    """
    name = measure.get("name", "")
    dax = measure.get("dax", "")
    needs_ai = measure.get("needs_ai", False)
    method = measure.get("translation_method", "")
    original = measure.get("original_formula", "")

    # BLOCKED: no usable DAX
    if not dax or dax.strip() == "BLANK()":
        reason = "No DAX mapping available"
        if original:
            reason = f"No DAX equivalent for: {original[:60]}"
        return "BLOCKED", 0, "NO_MAPPING", f"-100 no mapping"

    # AUTO: simple aggregation, deterministic (no AI, no heuristic)
    if not needs_ai and method != "heuristic" and method != "ai_claude":
        return "AUTO", 100, "FUNCTION_MAP", ""

    # GOOD: heuristic translation
    if method == "heuristic":
        conf = 85
        deductions = "-15 heuristic approximation"
        # Check calc_validator results if available
        if calc_detail and calc_detail.get("match") is False:
            delta = calc_detail.get("delta_pct", 0)
            conf = max(30, 85 - int(delta))
            deductions = f"-15 heuristic -{int(delta)} value mismatch ({delta:.1f}%)"
            return "REVIEW", conf, "HEURISTIC", deductions
        return "GOOD", conf, "HEURISTIC", deductions

    # AI translation
    if method == "ai_claude":
        conf = 70
        deductions = "-30 AI translation"
        if calc_detail and calc_detail.get("match") is True:
            conf = 90
            deductions = "-10 AI (verified match)"
            return "GOOD", conf, "AI_VERIFIED", deductions
        if calc_detail and calc_detail.get("match") is False:
            delta = calc_detail.get("delta_pct", 0)
            conf = max(20, 70 - int(delta))
            deductions = f"-30 AI -{int(delta)} value mismatch ({delta:.1f}%)"
            return "REVIEW", conf, "AI_UNVERIFIED", deductions
        return "REVIEW", conf, "AI_UNVERIFIED", deductions

    # AI failed
    if method == "ai_failed":
        return "BLOCKED", 0, "AI_FAILED", "-100 AI translation failed"

    # Default: simple measure with no original formula (implicit)
    if not original:
        return "AUTO", 100, "IMPLICIT_AGG", ""

    return "REVIEW", 50, "UNKNOWN", "-50 unknown translation path"


# ---------------------------------------------------------------------------
#  DQ Score Computation
# ---------------------------------------------------------------------------

def _compute_dq(fields):
    """Compute DQ score from classified fields.

    Returns dict with score, reason, dimensions list.
    """
    total = len(fields)
    if total == 0:
        return {"score": 100, "reason": "No fields to evaluate",
                "dimensions": []}

    blocked = [f for f in fields if f["tier"] == "BLOCKED"]
    review = [f for f in fields if f["tier"] in ("REVIEW", "MANUAL")]
    low_conf = [f for f in fields if 0 < f["confidence"] < 50]
    auto_good = [f for f in fields if f["tier"] in ("AUTO", "GOOD")]

    dimensions = []

    # Blocked fields: -20 per blocked field (max -60)
    blocked_deduction = min(len(blocked) * 20, 60)
    dimensions.append({
        "name": "Blocked Fields",
        "description": f"{len(blocked)} field(s) with no DAX equivalent",
        "deduction": -blocked_deduction if blocked else 0,
        "status": "FAIL" if blocked else "PASS",
    })

    # Review fields: -5 per review field (max -25)
    review_deduction = min(len(review) * 5, 25)
    dimensions.append({
        "name": "Review Required",
        "description": f"{len(review)} field(s) need manual review",
        "deduction": -review_deduction if review else 0,
        "status": "WARN" if review else "PASS",
    })

    # Low confidence: -3 per low-confidence field (max -15)
    low_deduction = min(len(low_conf) * 3, 15)
    dimensions.append({
        "name": "Low Confidence",
        "description": f"{len(low_conf)} field(s) below 50% confidence",
        "deduction": -low_deduction if low_conf else 0,
        "status": "WARN" if low_conf else "PASS",
    })

    # Coverage ratio
    coverage = len(auto_good) / total * 100 if total else 100
    coverage_ok = coverage >= 80
    coverage_deduction = 0 if coverage_ok else int((80 - coverage) / 2)
    dimensions.append({
        "name": "Translation Coverage",
        "description": f"{len(auto_good)}/{total} fields auto-translated ({coverage:.0f}%)",
        "deduction": -coverage_deduction,
        "status": "PASS" if coverage_ok else "WARN",
    })

    score = max(0, 100 - blocked_deduction - review_deduction
                - low_deduction - coverage_deduction)

    # Build reason string
    parts = []
    if blocked:
        parts.append(f"{len(blocked)} blocked field(s) require manual DAX rewrite")
    if review:
        parts.append(f"{len(review)} field(s) need review before deployment")
    if not parts:
        parts.append("All fields translated successfully")

    return {
        "score": score,
        "reason": "; ".join(parts),
        "dimensions": dimensions,
    }


# ---------------------------------------------------------------------------
#  Warnings Builder
# ---------------------------------------------------------------------------

def _build_warnings(fields):
    """Build warning list from classified fields."""
    warnings = []
    blocked = [f for f in fields if f["tier"] == "BLOCKED"]
    review = [f for f in fields if f["tier"] == "REVIEW"]

    if blocked:
        names = ", ".join(f["name"] for f in blocked[:5])
        extra = f" (+{len(blocked)-5} more)" if len(blocked) > 5 else ""
        warnings.append({
            "level": "ERROR",
            "title": f"{len(blocked)} BLOCKED field(s)",
            "detail": f"{names}{extra} -- requires manual DAX rewrite",
        })

    if review:
        names = ", ".join(f["name"] for f in review[:5])
        extra = f" (+{len(review)-5} more)" if len(review) > 5 else ""
        warnings.append({
            "level": "WARN",
            "title": f"{len(review)} REVIEW field(s)",
            "detail": f"{names}{extra} -- verify DAX output before deploying",
        })

    # Check for duplicate field names
    name_counts = {}
    for f in fields:
        name_counts[f["name"]] = name_counts.get(f["name"], 0) + 1
    dupes = {k: v for k, v in name_counts.items() if v > 1}
    if dupes:
        dupe_names = ", ".join(f"{k} (x{v})" for k, v in list(dupes.items())[:3])
        warnings.append({
            "level": "ERROR",
            "title": f"{len(dupes)} duplicate field name(s)",
            "detail": f"{dupe_names} -- will cause errors in Power BI",
        })

    return warnings


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_audit_data(measures, tableau_spec, calc_audit_result=None,
                     workbook_name="", tableau_version=""):
    """Build the full audit data structure from pipeline outputs.

    Args:
        measures: list of measure dicts from direct_mapper
        tableau_spec: dict from enhanced_tableau_parser
        calc_audit_result: optional dict from calc_validator.validate_calculations()
        workbook_name: name of the source workbook
        tableau_version: Tableau version string

    Returns:
        dict with keys: fields, dq, warnings, meta
    """
    # Build calc detail lookup
    calc_lookup = {}
    if calc_audit_result:
        for d in calc_audit_result.get("details", []):
            calc_lookup[d["name"].lower()] = d

    # Classify each measure
    fields = []
    for m in measures:
        calc_detail = calc_lookup.get(m["name"].lower())
        tier, confidence, method, deductions = _classify_tier(m, calc_detail)

        # Determine source worksheet
        source = ""
        for ws in tableau_spec.get("worksheets", []):
            for shelf_key in ("rows_fields", "cols_fields"):
                for ref in ws.get(shelf_key, []):
                    if m["name"].lower() in ref.lower():
                        source = ws.get("name", "")
                        break
                if source:
                    break
            if source:
                break
        if not source:
            for cf in tableau_spec.get("calculated_fields", []):
                if cf.get("name", "").lower() == m["name"].lower():
                    source = cf.get("datasource", "Calculated")
                    break

        fields.append({
            "name": m["name"],
            "source": source or "Data",
            "formula": m.get("original_formula", m.get("dax", "")),
            "dax": m.get("dax", ""),
            "confidence": confidence,
            "tier": tier,
            "method": method,
            "deductions": deductions,
        })

    # Compute DQ score
    dq = _compute_dq(fields)

    # Build warnings
    warnings = _build_warnings(fields)

    # Build metadata
    auto_good = sum(1 for f in fields if f["tier"] in ("AUTO", "GOOD"))
    review_count = sum(1 for f in fields if f["tier"] in ("REVIEW", "MANUAL"))
    blocked_count = sum(1 for f in fields if f["tier"] == "BLOCKED")
    avg_conf = (sum(f["confidence"] for f in fields) / len(fields)
                if fields else 0)

    meta = {
        "workbook_name": workbook_name or tableau_spec.get("workbook_name", "Workbook"),
        "tableau_version": tableau_version or tableau_spec.get("version", ""),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": len(fields),
        "auto_good": auto_good,
        "review": review_count,
        "blocked": blocked_count,
        "avg_confidence": int(avg_conf),
    }

    return {
        "fields": fields,
        "dq": dq,
        "warnings": warnings,
        "meta": meta,
    }


def generate_audit_report(fields, dq, warnings, meta):
    """Generate a self-contained HTML audit report.

    Args:
        fields: list of field dicts with name, source, formula, dax,
                confidence, tier, method, deductions
        dq: dict with score, reason, dimensions
        warnings: list of warning dicts with level, title, detail
        meta: dict with workbook_name, tableau_version, timestamp,
              total, auto_good, review, blocked, avg_confidence

    Returns:
        str: complete HTML document
    """
    e = _html.escape

    # DQ score color
    score = dq.get("score", 0)
    if score >= 90:
        score_color = "#00ff88"
    elif score >= 70:
        score_color = "#ffd700"
    elif score >= 50:
        score_color = "#ff8800"
    else:
        score_color = "#ff4466"

    # Build field rows
    field_rows_html = ""
    for i, f in enumerate(fields):
        tier = f["tier"]
        tier_colors = {
            "AUTO": "#00ff88", "GOOD": "#4488ff", "REVIEW": "#ffd700",
            "MANUAL": "#ff8800", "BLOCKED": "#ff4466",
        }
        tc = tier_colors.get(tier, "#888899")

        formula_display = e(f["formula"][:60]) if f["formula"] else "--"
        dax_display = e(f["dax"][:60]) if f["dax"] else "--"

        # Expansion content
        expand_content = ""
        if f["formula"]:
            expand_content += (
                f'<div class="expand-label">Tableau Formula:</div>'
                f'<pre class="expand-code">{e(f["formula"])}</pre>'
            )
        if f["dax"]:
            expand_content += (
                f'<div class="expand-label">DAX Output:</div>'
                f'<pre class="expand-code dax-code">{e(f["dax"])}</pre>'
                f'<button class="copy-btn" onclick="copyDax({_js_str(f["dax"])}, this)">Copy DAX</button>'
            )
        if tier == "BLOCKED" and f["deductions"]:
            expand_content += (
                f'<div class="expand-reason blocked-reason">{e(f["deductions"])}</div>'
            )
        elif tier == "REVIEW" and f["deductions"]:
            expand_content += (
                f'<div class="expand-reason review-reason">{e(f["deductions"])}</div>'
            )

        field_rows_html += f'''
        <tr class="formula-row" data-tier="{tier}">
            <td class="col-num">{i+1}</td>
            <td class="col-name">{e(f["name"])}</td>
            <td class="col-source">{e(f["source"])}</td>
            <td class="col-formula mono">{formula_display}</td>
            <td class="col-dax mono">{dax_display}</td>
            <td class="col-conf">{f["confidence"]}%</td>
            <td><span class="tier-badge" style="background:{tc}20;color:{tc};border:1px solid {tc}40">{tier}</span></td>
            <td class="col-method">{e(f["method"])}</td>
            <td class="col-expand"><span class="chevron" onclick="toggleExpand(this)">&#9654;</span></td>
        </tr>
        <tr class="expand-row" style="display:none">
            <td colspan="9">
                <div class="expand-content">{expand_content}</div>
            </td>
        </tr>'''

    # DQ dimensions table
    dq_dims_html = ""
    for dim in dq.get("dimensions", []):
        status = dim.get("status", "PASS")
        if status == "PASS":
            badge = '<span class="status-pass">PASS</span>'
        elif status == "WARN":
            badge = f'<span class="status-warn">WARN {dim["deduction"]}</span>'
        else:
            badge = f'<span class="status-fail">FAIL {dim["deduction"]}</span>'
        dq_dims_html += f'''
            <tr>
                <td>{e(dim["name"])}</td>
                <td>{e(dim["description"])}</td>
                <td style="text-align:center">{dim["deduction"]}</td>
                <td style="text-align:center">{badge}</td>
            </tr>'''

    # Warnings HTML
    warnings_html = ""
    if warnings:
        warning_items = ""
        for w in warnings:
            level = w.get("level", "WARN")
            border_color = "#ff4466" if level == "ERROR" else "#ffd700"
            icon = "!!" if level == "ERROR" else "!"
            warning_items += f'''
                <div class="warning-item" style="border-left:4px solid {border_color}">
                    <span class="warning-icon" style="color:{border_color}">[{icon}]</span>
                    <div>
                        <div class="warning-title">{e(w["title"])}</div>
                        <div class="warning-detail">{e(w["detail"])}</div>
                    </div>
                </div>'''

        collapse_attr = ""
        collapse_btn = ""
        if len(warnings) > 3:
            collapse_attr = ' id="warnings-list"'
            collapse_btn = '<button class="collapse-btn" onclick="toggleWarnings()">Show all warnings</button>'

        warnings_html = f'''
        <div class="section">
            <h2 class="section-title">Warnings</h2>
            <div{collapse_attr}>{warning_items}</div>
            {collapse_btn}
        </div>'''

    # Summary table
    summary_rows = [
        ("Workbook", e(meta.get("workbook_name", "")), ""),
        ("Generated", e(meta.get("timestamp", "")), ""),
        ("Total Fields", str(meta.get("total", 0)), ""),
        ("Verified (AUTO/GOOD)", str(meta.get("auto_good", 0)), "#00ff88"),
        ("Review (REVIEW/MANUAL)", str(meta.get("review", 0)), "#ffd700"),
        ("Blocked", str(meta.get("blocked", 0)), "#ff4466"),
        ("Avg Confidence", f'{meta.get("avg_confidence", 0)}%', ""),
        ("DQ Score", f'{score}/100', score_color),
    ]
    summary_html = ""
    for label, value, color in summary_rows:
        style = f' style="color:{color}"' if color else ""
        summary_html += f'<tr><td class="sum-label">{label}</td><td class="sum-value"{style}>{value}</td></tr>'

    # Tableau version line
    version_line = ""
    if meta.get("tableau_version"):
        version_line = f' | Tableau {e(meta["tableau_version"])}'

    total = meta.get("total", 0)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dr. Data Audit Report - {e(meta.get("workbook_name", ""))}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
    background: #0a0a12;
    color: #e8e8f0;
    font-family: 'Inter', system-ui, sans-serif;
    line-height: 1.5;
    padding: 32px;
    max-width: 1400px;
    margin: 0 auto;
}}

.mono {{ font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px; }}

/* -- Header -- */
.header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 32px;
    padding-bottom: 24px;
    border-bottom: 1px solid #1e1e32;
}}
.header-left h1 {{
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}}
.header-left .subtitle {{
    font-size: 16px;
    color: #888899;
    margin-bottom: 2px;
}}
.header-left .meta-line {{
    font-size: 12px;
    color: #555566;
}}
.dq-score-display {{
    text-align: right;
}}
.dq-score-number {{
    font-size: 56px;
    font-weight: 700;
    line-height: 1;
}}
.dq-score-label {{
    font-size: 13px;
    color: #888899;
    margin-top: 4px;
}}

/* -- KPI Cards -- */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}}
.kpi-card {{
    background: #12121e;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 20px 16px;
    text-align: center;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
}}
.kpi-label {{
    font-size: 12px;
    color: #888899;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* -- Sections -- */
.section {{
    margin-bottom: 32px;
}}
.section-title {{
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    color: #e8e8f0;
}}

/* -- DQ Breakdown -- */
.dq-banner {{
    background: #12121e;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 16px;
}}
.dq-banner-score {{
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 6px;
}}
.dq-banner-reason {{
    font-size: 14px;
    color: #888899;
}}
.dq-table {{
    width: 100%;
    border-collapse: collapse;
    background: #12121e;
    border-radius: 8px;
    overflow: hidden;
}}
.dq-table th {{
    text-align: left;
    padding: 10px 14px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888899;
    border-bottom: 1px solid #1e1e32;
    background: #0e0e1a;
}}
.dq-table td {{
    padding: 10px 14px;
    font-size: 13px;
    border-bottom: 1px solid #1e1e32;
}}
.status-pass {{
    background: #00ff8820;
    color: #00ff88;
    border: 1px solid #00ff8840;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}}
.status-warn {{
    background: #ffd70020;
    color: #ffd700;
    border: 1px solid #ffd70040;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}}
.status-fail {{
    background: #ff446620;
    color: #ff4466;
    border: 1px solid #ff446640;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}}

/* -- Warnings -- */
.warning-item {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: #12121e;
    border-radius: 6px;
    margin-bottom: 8px;
}}
.warning-icon {{
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 1px;
}}
.warning-title {{
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 2px;
}}
.warning-detail {{
    font-size: 13px;
    color: #888899;
}}
.collapse-btn {{
    background: none;
    border: 1px solid #1e1e32;
    color: #4488ff;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    margin-top: 8px;
}}
.collapse-btn:hover {{ border-color: #4488ff; }}

/* -- Formula Table Controls -- */
.table-controls {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}}
.search-box {{
    background: #12121e;
    border: 1px solid #1e1e32;
    color: #e8e8f0;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 13px;
    width: 260px;
    font-family: 'Inter', system-ui, sans-serif;
}}
.search-box:focus {{
    outline: none;
    border-color: #4488ff;
}}
.search-box::placeholder {{ color: #555566; }}
.filter-btn {{
    background: #12121e;
    border: 1px solid #1e1e32;
    color: #888899;
    padding: 6px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    font-weight: 500;
    font-family: 'Inter', system-ui, sans-serif;
    transition: all 0.15s;
}}
.filter-btn:hover {{ border-color: #4488ff; color: #e8e8f0; }}
.filter-btn.active {{ background: #4488ff20; border-color: #4488ff; color: #4488ff; }}
.count-display {{
    font-size: 12px;
    color: #888899;
    margin-left: auto;
}}

/* -- Formula Table -- */
.formula-table-wrap {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #1e1e32;
}}
.formula-table {{
    width: 100%;
    border-collapse: collapse;
    background: #12121e;
    font-size: 13px;
}}
.formula-table th {{
    text-align: left;
    padding: 10px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888899;
    background: #0e0e1a;
    border-bottom: 1px solid #1e1e32;
    position: sticky;
    top: 0;
    z-index: 1;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}}
.formula-table th:hover {{ color: #e8e8f0; }}
.formula-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid #1e1e3210;
    vertical-align: middle;
}}
.formula-table tr.formula-row:hover {{
    background: #1a1a2e;
}}
.col-num {{ width: 40px; color: #555566; text-align: center; }}
.col-name {{ min-width: 140px; font-weight: 500; }}
.col-source {{ min-width: 80px; color: #888899; }}
.col-formula {{ min-width: 180px; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #aaaacc; }}
.col-dax {{ min-width: 180px; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #aaaacc; }}
.col-conf {{ width: 60px; text-align: center; }}
.col-method {{ min-width: 100px; color: #888899; font-size: 11px; }}
.col-expand {{ width: 32px; text-align: center; }}

.tier-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
}}
.chevron {{
    cursor: pointer;
    display: inline-block;
    transition: transform 0.2s;
    color: #555566;
    font-size: 12px;
}}
.chevron.open {{ transform: rotate(90deg); color: #4488ff; }}

/* -- Expand Row -- */
.expand-row td {{
    padding: 0;
    background: #0e0e1a;
}}
.expand-content {{
    padding: 16px 24px;
}}
.expand-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #888899;
    margin-bottom: 4px;
    margin-top: 12px;
}}
.expand-label:first-child {{ margin-top: 0; }}
.expand-code {{
    background: #0a0a12;
    border: 1px solid #1e1e32;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.6;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
    color: #e8e8f0;
}}
.dax-code {{ color: #4488ff; }}
.copy-btn {{
    background: #12121e;
    border: 1px solid #1e1e32;
    color: #888899;
    padding: 4px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    margin-top: 8px;
    font-family: 'Inter', system-ui, sans-serif;
    transition: all 0.15s;
}}
.copy-btn:hover {{ border-color: #4488ff; color: #e8e8f0; }}
.expand-reason {{
    margin-top: 10px;
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
}}
.blocked-reason {{
    background: #ff446615;
    color: #ff4466;
    border: 1px solid #ff446630;
}}
.review-reason {{
    background: #ffd70015;
    color: #ffd700;
    border: 1px solid #ffd70030;
}}

/* -- Summary -- */
.summary-table {{
    width: 100%;
    max-width: 400px;
    border-collapse: collapse;
    background: #12121e;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1e1e32;
}}
.summary-table td {{
    padding: 8px 16px;
    font-size: 13px;
    border-bottom: 1px solid #1e1e3210;
}}
.sum-label {{ color: #888899; width: 180px; }}
.sum-value {{ font-weight: 600; }}

/* -- Footer -- */
.footer {{
    text-align: center;
    padding: 24px 0;
    margin-top: 32px;
    border-top: 1px solid #1e1e32;
    font-size: 12px;
    color: #555566;
}}

/* -- Responsive -- */
@media (max-width: 900px) {{
    .kpi-row {{ grid-template-columns: repeat(3, 1fr); }}
    .header {{ flex-direction: column; gap: 16px; }}
    .dq-score-display {{ text-align: left; }}
    body {{ padding: 16px; }}
}}
</style>
</head>
<body>

<!-- SECTION 1: Header -->
<div class="header">
    <div class="header-left">
        <h1>Dr. Data</h1>
        <div class="subtitle">{e(meta.get("workbook_name", ""))}</div>
        <div class="meta-line">{e(meta.get("timestamp", ""))}{version_line}</div>
    </div>
    <div class="dq-score-display">
        <div class="dq-score-number" style="color:{score_color}">{score}</div>
        <div class="dq-score-label">DQ Score</div>
    </div>
</div>

<!-- SECTION 2: KPI Scorecard -->
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-value">{meta.get("total", 0)}</div>
        <div class="kpi-label">Total Fields</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#00ff88">{meta.get("auto_good", 0)}</div>
        <div class="kpi-label">Mapped</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:#ff4466">{meta.get("blocked", 0)}</div>
        <div class="kpi-label">Blocked</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value">{meta.get("avg_confidence", 0)}%</div>
        <div class="kpi-label">Avg Confidence</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-value" style="color:{score_color}">{score}</div>
        <div class="kpi-label">DQ Score</div>
    </div>
</div>

<!-- SECTION 3: DQ Score Breakdown -->
<div class="section">
    <h2 class="section-title">DQ Score Breakdown</h2>
    <div class="dq-banner">
        <div class="dq-banner-score" style="color:{score_color}">DQ Score: {score}/100</div>
        <div class="dq-banner-reason">{e(dq.get("reason", ""))}</div>
    </div>
    <table class="dq-table">
        <thead><tr>
            <th>Dimension</th><th>Description</th><th>Deduction</th><th>Status</th>
        </tr></thead>
        <tbody>{dq_dims_html}</tbody>
    </table>
</div>

<!-- SECTION 4: Warnings -->
{warnings_html}

<!-- SECTION 5: Formula Translation Table -->
<div class="section">
    <h2 class="section-title">Formula Translation Table</h2>
    <div class="table-controls">
        <input type="text" class="search-box" id="search" placeholder="Search fields, formulas, DAX..." oninput="filterTable()">
        <button class="filter-btn active" onclick="setFilter('ALL', this)">ALL</button>
        <button class="filter-btn" onclick="setFilter('AUTO', this)">AUTO</button>
        <button class="filter-btn" onclick="setFilter('GOOD', this)">GOOD</button>
        <button class="filter-btn" onclick="setFilter('REVIEW', this)">REVIEW</button>
        <button class="filter-btn" onclick="setFilter('MANUAL', this)">MANUAL</button>
        <button class="filter-btn" onclick="setFilter('BLOCKED', this)">BLOCKED</button>
        <span class="count-display" id="count-display">Showing {total} of {total} fields</span>
    </div>
    <div class="formula-table-wrap">
        <table class="formula-table">
            <thead><tr>
                <th>#</th>
                <th onclick="sortTable('name')">Field</th>
                <th>Source</th>
                <th>Tableau Formula</th>
                <th>DAX Output</th>
                <th onclick="sortTable('conf')">Conf</th>
                <th onclick="sortTable('tier')">Tier</th>
                <th>Method</th>
                <th></th>
            </tr></thead>
            <tbody id="formula-tbody">{field_rows_html}</tbody>
        </table>
    </div>
</div>

<!-- SECTION 6: Summary -->
<div class="section">
    <h2 class="section-title">Summary</h2>
    <table class="summary-table">
        <tbody>{summary_html}</tbody>
    </table>
</div>

<!-- Footer -->
<div class="footer">
    Dr. Data -- The Art of The Possible -- theartofthepossible.io
</div>

<script>
var activeTier = 'ALL';
var totalFields = {total};

function filterTable() {{
    var query = document.getElementById('search').value.toLowerCase();
    var rows = document.querySelectorAll('.formula-row');
    rows.forEach(function(row) {{
        var text = row.textContent.toLowerCase();
        var tier = row.getAttribute('data-tier');
        var matchesSearch = !query || text.indexOf(query) !== -1;
        var matchesTier = activeTier === 'ALL' || tier === activeTier;
        row.style.display = (matchesSearch && matchesTier) ? '' : 'none';
        // Also hide expand row if parent is hidden
        var expandRow = row.nextElementSibling;
        if (expandRow && expandRow.classList.contains('expand-row')) {{
            if (row.style.display === 'none') expandRow.style.display = 'none';
        }}
    }});
    updateCount();
}}

function setFilter(tier, btn) {{
    activeTier = tier;
    document.querySelectorAll('.filter-btn').forEach(function(b) {{
        b.classList.remove('active');
    }});
    btn.classList.add('active');
    filterTable();
}}

function updateCount() {{
    var visible = 0;
    document.querySelectorAll('.formula-row').forEach(function(row) {{
        if (row.style.display !== 'none') visible++;
    }});
    document.getElementById('count-display').textContent =
        'Showing ' + visible + ' of ' + totalFields + ' fields';
}}

function toggleExpand(chevron) {{
    chevron.classList.toggle('open');
    var formulaRow = chevron.closest('tr');
    var expandRow = formulaRow.nextElementSibling;
    if (expandRow && expandRow.classList.contains('expand-row')) {{
        expandRow.style.display = expandRow.style.display === 'none' ? '' : 'none';
    }}
}}

function copyDax(dax, btn) {{
    navigator.clipboard.writeText(dax).then(function() {{
        btn.textContent = 'Copied';
        btn.style.color = '#00ff88';
        btn.style.borderColor = '#00ff88';
        setTimeout(function() {{
            btn.textContent = 'Copy DAX';
            btn.style.color = '';
            btn.style.borderColor = '';
        }}, 2000);
    }});
}}

function sortTable(key) {{
    var tbody = document.getElementById('formula-tbody');
    var pairs = [];
    var rows = tbody.querySelectorAll('tr');
    for (var i = 0; i < rows.length; i += 2) {{
        pairs.push([rows[i], rows[i + 1]]);
    }}
    pairs.sort(function(a, b) {{
        if (key === 'conf') {{
            var aVal = parseInt(a[0].querySelector('.col-conf').textContent) || 0;
            var bVal = parseInt(b[0].querySelector('.col-conf').textContent) || 0;
            return bVal - aVal;
        }} else if (key === 'tier') {{
            var order = {{'AUTO': 0, 'GOOD': 1, 'REVIEW': 2, 'MANUAL': 3, 'BLOCKED': 4}};
            var aT = a[0].getAttribute('data-tier');
            var bT = b[0].getAttribute('data-tier');
            return (order[aT] || 5) - (order[bT] || 5);
        }} else {{
            var aName = a[0].querySelector('.col-name').textContent.toLowerCase();
            var bName = b[0].querySelector('.col-name').textContent.toLowerCase();
            return aName < bName ? -1 : aName > bName ? 1 : 0;
        }}
    }});
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
    pairs.forEach(function(pair) {{
        tbody.appendChild(pair[0]);
        tbody.appendChild(pair[1]);
    }});
}}

function toggleWarnings() {{
    var list = document.getElementById('warnings-list');
    if (!list) return;
    var items = list.querySelectorAll('.warning-item');
    var btn = list.parentElement.querySelector('.collapse-btn');
    if (items.length <= 3) return;
    var collapsed = items[3].style.display === 'none';
    for (var i = 3; i < items.length; i++) {{
        items[i].style.display = collapsed ? '' : 'none';
    }}
    btn.textContent = collapsed ? 'Show fewer warnings' : 'Show all warnings';
}}

// Initialize: collapse warnings beyond 3 if needed
(function() {{
    var list = document.getElementById('warnings-list');
    if (!list) return;
    var items = list.querySelectorAll('.warning-item');
    for (var i = 3; i < items.length; i++) {{
        items[i].style.display = 'none';
    }}
}})();
</script>

</body>
</html>'''


def _js_str(s):
    """Escape a Python string for safe use in a JS function argument."""
    escaped = s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'
