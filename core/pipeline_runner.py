"""
Dr. Data V2 -- Pipeline Runner

Executes individual pipeline stages, producing structured output for
the human-gated state machine. Each run_*() method does work and returns
a dict that the chat layer presents to the analyst.

This module contains ZERO Streamlit imports and ZERO chat/LLM orchestration.
It calls core modules directly.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("drdata-v2.pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_parse(twbx_path: str) -> dict:
    """Stage 0: Parse Tableau workbook structure.

    Returns:
        {
            "tableau_spec": dict,      # full parsed spec
            "worksheets": [{"name", "mark_type", "fields"}...],
            "dashboards": [{"name", "zones"}...],
            "calc_fields": [{"name", "formula", "datatype"}...],
            "datasources": [{"name", "tables"}...],
            "palette": list,           # extracted colors
            "summary": str,            # human-readable
        }
    """
    from core.enhanced_tableau_parser import parse_twb, get_xml_root

    spec = parse_twb(twbx_path)

    worksheets = []
    for ws in spec.get("worksheets", []):
        worksheets.append({
            "name": ws.get("name", ""),
            "mark_type": ws.get("mark_type", "automatic"),
            "fields": ws.get("fields", [])[:20],
        })

    dashboards = []
    for db in spec.get("dashboards", []):
        dashboards.append({
            "name": db.get("name", ""),
            "zones": [z.get("name", "") for z in db.get("zones", [])],
        })

    calcs = []
    for cf in spec.get("calculated_fields", []):
        calcs.append({
            "name": cf.get("name", ""),
            "formula": cf.get("formula", "")[:200],
            "datatype": cf.get("datatype", ""),
        })

    datasources = []
    for ds in spec.get("datasources", []):
        datasources.append({
            "name": ds.get("name", ""),
            "tables": ds.get("tables", [])[:10],
        })

    # Extract colors
    palette = []
    try:
        from core.color_extractor import extract_all_colors, build_unified_palette
        xml_root = get_xml_root(twbx_path)
        if xml_root is not None:
            extracted = extract_all_colors(xml_root)
            palette = build_unified_palette(extracted, max_colors=12)
    except Exception as e:
        logger.warning(f"Color extraction (non-fatal): {e}")

    ws_count = len(worksheets)
    db_count = len(dashboards)
    cf_count = len(calcs)
    ds_count = len(datasources)

    summary_lines = [
        f"Parsed {Path(twbx_path).name}:",
        f"  {ws_count} worksheets, {db_count} dashboards",
        f"  {cf_count} calculated fields, {ds_count} datasources",
    ]
    if palette:
        summary_lines.append(f"  {len(palette)} Tableau colors extracted")
    if worksheets:
        summary_lines.append("")
        summary_lines.append("Worksheets:")
        for ws in worksheets[:10]:
            fields_str = ", ".join(ws["fields"][:5])
            summary_lines.append(f"  - {ws['name']} ({ws['mark_type']}): {fields_str}")
    if calcs:
        summary_lines.append("")
        summary_lines.append(f"Calculated Fields ({cf_count}):")
        for cf in calcs[:8]:
            summary_lines.append(f"  - {cf['name']}: {cf['formula'][:80]}")
        if cf_count > 8:
            summary_lines.append(f"  ... and {cf_count - 8} more")

    return {
        "tableau_spec": spec,
        "worksheets": worksheets,
        "dashboards": dashboards,
        "calc_fields": calcs,
        "datasources": datasources,
        "palette": palette,
        "summary": "\n".join(summary_lines),
    }


def run_field_mapping(tableau_spec: dict, data_profile: dict,
                      table_name: str = "Data") -> dict:
    """Stage 2: Map Tableau fields to Power BI columns.

    Returns:
        {
            "config": dict,           # from build_pbip_config_from_tableau
            "dashboard_spec": dict,
            "field_map": [{"tableau_field", "pbi_column", "type", "role"}...],
            "summary": str,
        }
    """
    from core.direct_mapper import build_pbip_config_from_tableau

    config, dspec = build_pbip_config_from_tableau(
        tableau_spec, data_profile, table_name
    )

    # Extract field mapping for presentation
    field_map = []
    tmdl = config.get("tmdl_model", {})
    tables = tmdl.get("tables", [])
    if tables:
        for col in tables[0].get("columns", []):
            field_map.append({
                "pbi_column": col.get("name", ""),
                "type": col.get("dataType", ""),
                "role": "measure" if col.get("summarizeBy") else "dimension",
            })

    summary_lines = [
        f"Field Mapping ({len(field_map)} columns):",
    ]
    for fm in field_map[:15]:
        summary_lines.append(f"  {fm['pbi_column']} ({fm['type']}, {fm['role']})")
    if len(field_map) > 15:
        summary_lines.append(f"  ... and {len(field_map) - 15} more")

    measures = []
    if tables:
        measures = tables[0].get("measures", [])
    if measures:
        summary_lines.append(f"\nDAX Measures ({len(measures)}):")
        for m in measures[:10]:
            summary_lines.append(f"  {m.get('name','')}: {m.get('expression','')[:80]}")
        if len(measures) > 10:
            summary_lines.append(f"  ... and {len(measures) - 10} more")

    return {
        "config": config,
        "dashboard_spec": dspec,
        "field_map": field_map,
        "measures": measures,
        "summary": "\n".join(summary_lines),
    }


def run_formula_translation(calc_fields: list, tableau_spec: dict,
                            dataframe: pd.DataFrame = None) -> dict:
    """Stage 3: Translate Tableau calculated fields to DAX.

    Returns translation results with QA validation and optional multi-brain
    consensus for REVIEW/BLOCKED formulas.
    """
    from core.formula_transpiler import TableauFormulaTranspiler

    transpiler = TableauFormulaTranspiler()
    translations = []
    review_needed = []

    for cf in calc_fields:
        name = cf.get("name", "")
        formula = cf.get("formula", "")
        if not formula:
            continue

        result = transpiler.transpile(formula)
        confidence = result.get("confidence", 0)
        # Tier based on confidence
        if confidence >= 0.9:
            tier = "AUTO"
        elif confidence >= 0.7:
            tier = "GOOD"
        elif confidence >= 0.4:
            tier = "REVIEW"
        else:
            tier = "BLOCKED"

        entry = {
            "name": name,
            "tableau_formula": formula[:200],
            "dax": result.get("dax", ""),
            "tier": tier,
            "confidence": confidence,
            "notes": "; ".join(result.get("warnings", [])),
        }
        translations.append(entry)
        if tier in ("REVIEW", "BLOCKED"):
            review_needed.append(entry)

    # QA: flag formulas with warnings
    qa_notes = []
    for t in translations:
        if t["notes"]:
            qa_notes.append(f"{t['name']}: {t['notes'][:100]}")

    auto_count = sum(1 for t in translations if t["tier"] == "AUTO")
    good_count = sum(1 for t in translations if t["tier"] == "GOOD")
    review_count = sum(1 for t in translations if t["tier"] == "REVIEW")
    blocked_count = sum(1 for t in translations if t["tier"] == "BLOCKED")

    summary_lines = [
        f"Formula Translation ({len(translations)} calculated fields):",
        f"  AUTO (deterministic): {auto_count}",
        f"  GOOD (high confidence): {good_count}",
        f"  REVIEW (needs verification): {review_count}",
        f"  BLOCKED (manual required): {blocked_count}",
    ]
    if review_needed:
        summary_lines.append("\nFormulas needing review:")
        for r in review_needed[:5]:
            summary_lines.append(f"  {r['name']}:")
            summary_lines.append(f"    Tableau: {r['tableau_formula'][:100]}")
            summary_lines.append(f"    DAX:     {r['dax'][:100] or '[no translation]'}")
            summary_lines.append(f"    Notes:   {r['notes']}")

    return {
        "translations": translations,
        "review_needed": review_needed,
        "auto_count": auto_count,
        "good_count": good_count,
        "review_count": review_count,
        "blocked_count": blocked_count,
        "qa_notes": qa_notes,
        "summary": "\n".join(summary_lines),
    }


def run_visual_mapping(tableau_spec: dict, config: dict,
                       palette: list = None) -> dict:
    """Stage 4: Map Tableau visuals to Power BI chart types.

    Returns visual mapping with fidelity notes.
    """
    sections = config.get("report_layout", {}).get("sections", [])

    visuals = []
    for sec in sections:
        page_name = sec.get("displayName", "Page")
        for vc in sec.get("visualContainers", []):
            vis_config = vc.get("config", {}).get("singleVisual", {})
            visuals.append({
                "page": page_name,
                "type": vis_config.get("visualType", "unknown"),
                "title": vis_config.get("title", {}).get("text", ""),
                "position": vc.get("position", {}),
            })

    total = len(visuals)
    pages = len(sections)
    types_used = list(set(v["type"] for v in visuals))

    summary_lines = [
        f"Visual Mapping ({total} visuals across {pages} pages):",
        f"  Chart types: {', '.join(types_used)}",
    ]
    if palette:
        summary_lines.append(f"  Palette: {', '.join(palette[:6])}")

    for sec in sections:
        page_name = sec.get("displayName", "Page")
        vcs = sec.get("visualContainers", [])
        summary_lines.append(f"\n  {page_name} ({len(vcs)} visuals):")
        for vc in vcs[:6]:
            vc_cfg = vc.get("config", {}).get("singleVisual", {})
            vtype = vc_cfg.get("visualType", "?")
            vtitle = vc_cfg.get("title", {}).get("text", "")
            summary_lines.append(f"    - {vtype}: {vtitle}")

    return {
        "visuals": visuals,
        "pages": pages,
        "total_visuals": total,
        "types_used": types_used,
        "summary": "\n".join(summary_lines),
    }


def run_generate(config: dict, data_profile: dict, dashboard_spec: dict,
                 dataframe: pd.DataFrame = None,
                 palette: list = None,
                 session_id: str = "") -> dict:
    """Stage 7: Generate PBIP project files and validate.

    Returns:
        {
            "pbip_path": str,
            "zip_path": str,
            "file_count": int,
            "preflight": {"passed", "failed", "all_passed"},
            "audit_html": str or None,
            "summary": str,
        }
    """
    # Inject palette into dashboard spec
    if palette and dashboard_spec:
        dashboard_spec["_unified_palette"] = palette

    out_dir = str(OUTPUT_DIR / f"v2_{session_id or 'gen'}_{int(time.time())}")

    from generators.pbip_generator import PBIPGenerator
    gen = PBIPGenerator(out_dir)
    gen_result = gen.generate(
        config, data_profile, dashboard_spec,
        data_file_path=None,
        dataframe=dataframe,
    )
    pbip_path = gen_result["path"]
    file_count = gen_result.get("file_count", 0)

    # Preflight
    from core.preflight_validator import validate as preflight
    pf = preflight(pbip_path)

    healer_fixes = 0
    if not pf.all_passed:
        try:
            from core.pbip_healer import heal
            fixes = heal(pbip_path, pf)
            healer_fixes = len(fixes)
            pf = preflight(pbip_path)
        except Exception as e:
            logger.warning(f"Healer error: {e}")

    # Audit report
    audit_html = None
    try:
        from core.audit_report import build_audit_data, generate_audit_report
        measures = config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", [])
        tableau_spec = dashboard_spec.get("_tableau_spec", {})
        audit_data = build_audit_data(measures, tableau_spec)
        audit_html_str = generate_audit_report(**audit_data)
        audit_path = Path(pbip_path).parent / "translation_audit.html"
        audit_path.write_text(audit_html_str, encoding="utf-8")
        audit_html = str(audit_path)
    except Exception as e:
        logger.warning(f"Audit report (non-fatal): {e}")

    # ZIP
    zip_path = None
    try:
        zip_base = Path(pbip_path).parent / Path(pbip_path).name
        zip_path = shutil.make_archive(str(zip_base), "zip", pbip_path)
    except Exception as e:
        logger.warning(f"ZIP creation (non-fatal): {e}")

    summary_lines = [
        f"PBIP Project Generated:",
        f"  {file_count} files at {pbip_path}",
        f"  Preflight: {pf.pass_count} passed, {pf.fail_count} failed",
    ]
    if healer_fixes:
        summary_lines.append(f"  Auto-healed: {healer_fixes} issues")
    if pf.all_passed:
        summary_lines.append("  All preflight checks passed.")
    else:
        summary_lines.append("  WARNING: Some preflight checks failed.")
        for r in pf.failed[:5]:
            summary_lines.append(f"    - {r}")
    if zip_path:
        summary_lines.append(f"  ZIP: {zip_path}")

    return {
        "pbip_path": pbip_path,
        "zip_path": zip_path,
        "file_count": file_count,
        "preflight": {
            "passed": pf.pass_count,
            "failed": pf.fail_count,
            "all_passed": pf.all_passed,
        },
        "audit_html": audit_html,
        "summary": "\n".join(summary_lines),
    }


def run_synthetic_data(tableau_spec: dict, num_rows: int = 2000) -> dict:
    """Generate synthetic data from Tableau spec.

    Returns {"dataframe": pd.DataFrame, "profile": dict, "summary": str}
    """
    from core.synthetic_data import generate_from_tableau_spec

    df, csv_path, schema = generate_from_tableau_spec(tableau_spec, num_rows=num_rows)

    # Build profile
    profile = {
        "table_name": "Data",
        "row_count": len(df),
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "semantic_type": (
                    "measure" if pd.api.types.is_numeric_dtype(df[c])
                    and not pd.api.types.is_datetime64_any_dtype(df[c])
                    else "dimension"
                ),
                "unique_count": int(df[c].nunique()),
            }
            for c in df.columns
        ],
    }

    summary_lines = [
        f"Synthetic data generated: {len(df)} rows, {len(df.columns)} columns",
        "Columns:",
    ]
    for col_info in profile["columns"][:15]:
        summary_lines.append(
            f"  {col_info['name']} ({col_info['dtype']}, {col_info['semantic_type']}, "
            f"{col_info['unique_count']} unique)"
        )

    return {
        "dataframe": df,
        "profile": profile,
        "csv_path": csv_path,
        "summary": "\n".join(summary_lines),
    }


def dispatch_multi_brain(formula_prompt: str) -> dict:
    """Fire a formula translation prompt to multiple brains for consensus.

    Returns {"results": {model: response}, "consensus": str, "summary": str}
    """
    import requests

    OLLAMA_URL = "http://localhost:11434/api/generate"
    results = {}

    # Local brains (free)
    for model in ["deepseek-coder-v2", "qwen2.5-coder", "phi4"]:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": formula_prompt, "stream": False},
                timeout=60,
            )
            if resp.status_code == 200:
                results[model] = resp.json().get("response", "")
        except Exception as e:
            results[model] = f"[unavailable: {e}]"

    # Claude (paid)
    try:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": formula_prompt}],
            )
            results["claude-sonnet"] = msg.content[0].text
    except Exception as e:
        results["claude-sonnet"] = f"[unavailable: {e}]"

    # GPT-4o (paid)
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": formula_prompt}],
                max_tokens=1024,
            )
            results["gpt-4o"] = resp.choices[0].message.content
    except Exception as e:
        results["gpt-4o"] = f"[unavailable: {e}]"

    # Build summary
    summary_lines = [f"Multi-brain consensus ({len(results)} brains consulted):"]
    for model, response in results.items():
        preview = response[:120].replace("\n", " ")
        summary_lines.append(f"  {model}: {preview}")

    return {
        "results": results,
        "summary": "\n".join(summary_lines),
    }
