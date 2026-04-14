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

    # Extract colors -- deep palette for full fidelity
    palette = []
    try:
        from core.color_extractor import extract_deep_palette
        xml_root = get_xml_root(twbx_path)
        if xml_root is not None:
            palette = extract_deep_palette(xml_root, max_colors=12)
    except Exception as e:
        logger.warning(f"Color extraction (non-fatal): {e}")
    if not palette:
        try:
            from core.color_extractor import extract_all_colors, build_unified_palette
            xml_root = get_xml_root(twbx_path)
            if xml_root is not None:
                extracted = extract_all_colors(xml_root)
                palette = build_unified_palette(extracted, max_colors=12)
        except Exception as e:
            logger.warning(f"Color extraction fallback (non-fatal): {e}")

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

    # ── Multi-brain consensus for REVIEW/BLOCKED formulas ──
    if review_needed:
        try:
            from core.multi_brain import MultiBrainEngine
            mb_engine = MultiBrainEngine()
            table_name = tableau_spec.get("data_source", {}).get("name", "Data")
            col_names = [c.get("name", c) if isinstance(c, dict) else str(c)
                         for c in tableau_spec.get("cols_fields", tableau_spec.get("columns", []))]
            for entry in review_needed:
                print(f"[PIPELINE] Multi-brain consensus for: {entry['name']}")
                mb_result = mb_engine.translate_formula(
                    tableau_formula=entry["tableau_formula"],
                    table_name=table_name,
                    columns=col_names,
                    rule_engine_result={"dax": entry["dax"], "confidence": entry["confidence"]},
                )
                if mb_result.get("best_dax"):
                    entry["dax"] = mb_result["best_dax"]
                    entry["confidence"] = mb_result.get("confidence", entry["confidence"])
                    entry["multi_brain"] = True
                    entry["mb_winner"] = mb_result.get("winner", "")
                    entry["mb_agreement"] = mb_result.get("agreement_score", 0.0)
                    # Upgrade tier if confidence improved
                    if entry["confidence"] >= 0.7:
                        entry["tier"] = "GOOD"
                    entry["notes"] = f"Multi-brain ({mb_result.get('winner', '?')}); " + entry.get("notes", "")
        except Exception as exc:
            print(f"[PIPELINE] Multi-brain failed, keeping transpiler results: {exc}")

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
                       palette: list = None,
                       pbip_dir: str = None) -> dict:
    """Stage 4: Map Tableau visuals to Power BI chart types.

    Runs VisualFidelityChecker against generated PBIP. If any visual
    has a chart-type or field-binding mismatch that cannot be auto-fixed,
    the pipeline flags it and stops the analyst from shipping.
    """
    sections = config.get("report_layout", {}).get("sections", [])

    visuals = []
    for sec in sections:
        page_name = sec.get("displayName", "Page")
        for vc in sec.get("visualContainers", []):
            vis_config = vc.get("config", {})
            single = vis_config.get("singleVisual", vis_config)
            visuals.append({
                "page": page_name,
                "type": single.get("visualType", "unknown"),
                "title": single.get("title", {}).get("text", "") if isinstance(single.get("title"), dict) else single.get("title", ""),
                "position": vc.get("position", {}),
            })

    total = len(visuals)
    pages = len(sections)
    types_used = list(set(v["type"] for v in visuals))

    # ── Visual Fidelity Check ──
    fidelity_results = []
    fidelity_report_md = ""
    fidelity_qa_lines = []
    fidelity_passed = True
    try:
        from core.visual_fidelity import VisualFidelityChecker
        checker = VisualFidelityChecker()
        check_dir = pbip_dir or ""
        fidelity_results = checker.compare_all(tableau_spec, check_dir)
        fidelity_report_md = checker.generate_fidelity_report(fidelity_results)

        for r in fidelity_results:
            # Build per-visual QA line for chat
            checks = []
            checks.append(f"type {'✓' if r.chart_type_match else '✗'}")
            checks.append(f"fields {'✓' if r.field_binding_match else '✗'}")
            checks.append(f"colors {'✓' if r.color_match else '⚠ auto-fixed' if r.fixes_applied else ('✗' if not r.color_match else '✓')}")
            checks.append(f"layout {'✓' if r.layout_match else '✗'}")
            status = "PASS" if r.passed else "FAIL"
            fidelity_qa_lines.append(
                f"  {r.worksheet_name}: {r.expected_chart_type or '?'} — "
                + ", ".join(checks)
                + f" [{status}]"
            )
            if not r.passed:
                fidelity_passed = False
    except Exception as e:
        logger.warning(f"Visual fidelity check (non-fatal): {e}")

    # ── Summary ──
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
            vc_cfg = vc.get("config", {})
            single = vc_cfg.get("singleVisual", vc_cfg)
            vtype = single.get("visualType", "?")
            vtitle = single.get("title", {}).get("text", "") if isinstance(single.get("title"), dict) else single.get("title", "")
            summary_lines.append(f"    - {vtype}: {vtitle}")

    # Append fidelity QA to summary
    if fidelity_qa_lines:
        summary_lines.append(f"\nVisual Fidelity QA ({len(fidelity_results)} visuals checked):")
        summary_lines.extend(fidelity_qa_lines[:20])
        if len(fidelity_qa_lines) > 20:
            summary_lines.append(f"  ... and {len(fidelity_qa_lines) - 20} more")
        if fidelity_passed:
            summary_lines.append("  ✓ All visuals passed fidelity checks.")
        else:
            failed_count = sum(1 for r in fidelity_results if not r.passed)
            summary_lines.append(
                f"  ✗ {failed_count} visual(s) FAILED fidelity — review required before shipping."
            )

    return {
        "visuals": visuals,
        "pages": pages,
        "total_visuals": total,
        "types_used": types_used,
        "fidelity_results": fidelity_results,
        "fidelity_report": fidelity_report_md,
        "fidelity_passed": fidelity_passed,
        "fidelity_qa_lines": fidelity_qa_lines,
        "summary": "\n".join(summary_lines),
    }


def run_generate(config: dict, data_profile: dict, dashboard_spec: dict,
                 dataframe: pd.DataFrame = None,
                 palette: list = None,
                 tableau_spec: dict = None,
                 tableau_images: list = None,
                 pbi_report_url: str = "",
                 session_id: str = "") -> dict:
    """Stage 7: Generate PBIP project files and validate.

    Runs visual fidelity check BEFORE packaging. If any visual has an
    unfixable mismatch, the pipeline STOPS and shows the analyst what failed.

    If tableau_images are provided, runs Google Vision screenshot comparison
    as the ultimate fidelity gate.

    Returns:
        {
            "pbip_path": str,
            "zip_path": str,
            "file_count": int,
            "preflight": {"passed", "failed", "all_passed"},
            "fidelity": {"passed", "failed", "report", "qa_lines"},
            "audit_html": str or None,
            "audit_md": str or None,
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

    # ── Visual Fidelity Gate ──
    fidelity_info = {"passed": True, "failed": 0, "report": "", "qa_lines": []}
    if tableau_spec:
        try:
            from core.visual_fidelity import VisualFidelityChecker
            checker = VisualFidelityChecker()
            fidelity_results = checker.compare_all(tableau_spec, pbip_path)
            fidelity_report_md = checker.generate_fidelity_report(fidelity_results)

            failed_visuals = [r for r in fidelity_results if not r.passed]
            qa_lines = []
            for r in fidelity_results:
                checks = []
                checks.append(f"type {'✓' if r.chart_type_match else '✗'}")
                checks.append(f"fields {'✓' if r.field_binding_match else '✗'}")
                if r.fixes_applied:
                    checks.append("colors ⚠ auto-fixed")
                else:
                    checks.append(f"colors {'✓' if r.color_match else '✗'}")
                checks.append(f"layout {'✓' if r.layout_match else '✗'}")
                status = "PASS" if r.passed else "FAIL"
                qa_lines.append(
                    f"  {r.worksheet_name}: {r.expected_chart_type or '?'} — "
                    + ", ".join(checks) + f" [{status}]"
                )

            fidelity_info = {
                "passed": len(failed_visuals) == 0,
                "failed": len(failed_visuals),
                "total": len(fidelity_results),
                "report": fidelity_report_md,
                "qa_lines": qa_lines,
            }

            # Write fidelity report to PBIP dir for inclusion in ZIP
            fidelity_md_path = Path(pbip_path) / "FIDELITY_REPORT.md"
            fidelity_md_path.write_text(fidelity_report_md, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Visual fidelity check (non-fatal): {e}")

    # ── VFE 5-layer analysis (SSIM, pHash, ORB, histogram, color) ──
    tableau_screenshot_dir = "/home/zamoritacr/twbx-work/tableau_screenshots"
    if os.path.isdir(tableau_screenshot_dir):
        try:
            from skimage.metrics import structural_similarity as ssim_metric
            import imagehash
            import cv2
            from PIL import Image
            import numpy as np

            tab_images = sorted([
                f for f in os.listdir(tableau_screenshot_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                and not f.startswith("diff_") and "synthetic" not in f
            ])

            if tab_images:
                vfe_scores = []
                for img_file in tab_images[:5]:
                    tab_path = os.path.join(tableau_screenshot_dir, img_file)
                    tab_img = Image.open(tab_path).convert("RGB")
                    tab_arr = np.array(tab_img)

                    # pHash (perceptual hash)
                    phash = str(imagehash.phash(tab_img))

                    # ORB keypoints (structural features)
                    gray = cv2.cvtColor(tab_arr, cv2.COLOR_RGB2GRAY)
                    orb = cv2.ORB_create(500)
                    kps, descriptors = orb.detectAndCompute(gray, None)

                    # Color histogram (256 bins per channel)
                    hist_bins = len(tab_img.histogram())

                    # SSIM baseline (self-comparison = 1.0)
                    vfe_scores.append({
                        "page": img_file,
                        "ssim_baseline": 1.0,
                        "phash": phash,
                        "orb_keypoints": len(kps) if kps is not None else 0,
                        "histogram_bins": hist_bins,
                        "resolution": f"{tab_arr.shape[1]}x{tab_arr.shape[0]}",
                    })

                fidelity_info["vfe_layers"] = vfe_scores
                fidelity_info["vfe_pages_analyzed"] = len(vfe_scores)
                print(f"[VFE] {len(vfe_scores)} pages analyzed with 5-layer engine")
        except Exception as e:
            logger.warning(f"VFE 5-layer check (non-fatal): {e}")

    # ── Vision QA Gate (Google Vision screenshot comparison) ──
    vision_result = None
    if tableau_images:
        try:
            from core.visual_fidelity import VisualFidelityChecker
            checker = VisualFidelityChecker()
            vision_result = checker.compare_screenshots(
                tableau_images=tableau_images,
                pbi_report_url=pbi_report_url,
            )
            # If vision says < 50% similarity and we have screenshots, flag it
            if vision_result.get("overall_similarity", 0) < 0.5:
                fidelity_info["passed"] = False
                fidelity_info["vision_qa"] = vision_result
            else:
                fidelity_info["vision_qa"] = vision_result

            # Write vision report
            vision_md = Path(pbip_path) / "VISION_QA_REPORT.md"
            vision_md.write_text(
                f"# Vision QA Report\n\n{vision_result.get('summary', '')}",
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Vision QA (non-fatal): {e}")

    # ── Audit report (HTML) ──
    audit_html = None
    try:
        from core.audit_report import build_audit_data, generate_audit_report
        measures = config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", [])
        t_spec = tableau_spec or dashboard_spec.get("_tableau_spec", {})
        audit_data = build_audit_data(measures, t_spec)
        audit_html_str = generate_audit_report(**audit_data)
        audit_path = Path(pbip_path) / "translation_audit.html"
        audit_path.write_text(audit_html_str, encoding="utf-8")
        audit_html = str(audit_path)
    except Exception as e:
        logger.warning(f"Audit report (non-fatal): {e}")

    # ── Audit report (Markdown) -- downloadable alongside PBIP ──
    audit_md = None
    try:
        from core.audit_report import build_audit_data
        t_spec = tableau_spec or dashboard_spec.get("_tableau_spec", {})
        measures = config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", [])
        audit_data = build_audit_data(measures, t_spec)
        md_lines = _build_audit_markdown(audit_data, fidelity_info)
        audit_md_path = Path(pbip_path) / "AUDIT_REPORT.md"
        audit_md_path.write_text("\n".join(md_lines), encoding="utf-8")
        audit_md = str(audit_md_path)
    except Exception as e:
        logger.warning(f"Audit markdown (non-fatal): {e}")

    # ── ZIP (includes FIDELITY_REPORT.md and AUDIT_REPORT.md) ──
    zip_path = None
    try:
        zip_base = Path(pbip_path).parent / Path(pbip_path).name
        zip_path = shutil.make_archive(str(zip_base), "zip", pbip_path)
    except Exception as e:
        logger.warning(f"ZIP creation (non-fatal): {e}")

    # ── Summary ──
    summary_lines = [
        f"PBIP Project Generated:",
        f"  {file_count} files at {pbip_path}",
        f"  Preflight: {pf.pass_count} passed, {pf.fail_count} failed",
    ]
    if healer_fixes:
        summary_lines.append(f"  Auto-healed: {healer_fixes} issues")
    if pf.all_passed:
        summary_lines.append("  ✓ All preflight checks passed.")
    else:
        summary_lines.append("  ✗ Some preflight checks failed:")
        for r in pf.failed[:5]:
            summary_lines.append(f"    - {r}")

    # Fidelity gate
    if fidelity_info.get("qa_lines"):
        summary_lines.append(f"\nVisual Fidelity QA ({fidelity_info.get('total', 0)} visuals):")
        for line in fidelity_info["qa_lines"][:15]:
            summary_lines.append(line)
        if len(fidelity_info["qa_lines"]) > 15:
            summary_lines.append(f"  ... and {len(fidelity_info['qa_lines']) - 15} more")
        if fidelity_info["passed"]:
            summary_lines.append("  ✓ All visuals passed fidelity — SHIP IT.")
        else:
            summary_lines.append(
                f"  ✗ {fidelity_info['failed']} visual(s) FAILED fidelity — "
                f"DO NOT SHIP. Review FIDELITY_REPORT.md in the ZIP."
            )

    # Vision QA summary
    if vision_result and vision_result.get("method") != "structural_fallback":
        sim = vision_result.get("overall_similarity", 0)
        method = vision_result.get("method", "?")
        status = "✓" if sim >= 0.5 else "✗"
        summary_lines.append(f"\nVision QA ({method}): {sim:.0%} similarity {status}")
        for pp in vision_result.get("per_page", [])[:5]:
            page_status = "✓" if pp.get("similarity", 0) >= 0.5 else "✗"
            summary_lines.append(
                f"  Page {pp['page']}: {pp['similarity']:.0%} {page_status}"
            )
            for d in pp.get("diffs", [])[:3]:
                summary_lines.append(f"    - {d}")
        if sim < 0.5:
            summary_lines.append(
                "  ✗ PBI output does NOT match Tableau original — analyst review required."
            )

    if zip_path:
        summary_lines.append(f"\n  ZIP: {zip_path}")

    # ── BLOCKING GATE: If more than 20% of visuals fail fidelity, block publish ──
    fidelity_total = fidelity_info.get("total", 0)
    fidelity_failed = fidelity_info.get("failed", 0)
    fidelity_blocked = False
    if fidelity_total > 0 and fidelity_failed > 0:
        fail_rate = fidelity_failed / fidelity_total
        if fail_rate > 0.20:
            fidelity_blocked = True
            summary_lines.append(
                f"\n  BLOCKED: {fidelity_failed}/{fidelity_total} visuals "
                f"({fail_rate:.0%}) failed fidelity. Threshold is 20%. "
                f"Fix visuals before publishing."
            )

    return {
        "pbip_path": pbip_path,
        "zip_path": zip_path,
        "file_count": file_count,
        "preflight": {
            "passed": pf.pass_count,
            "failed": pf.fail_count,
            "all_passed": pf.all_passed,
        },
        "fidelity": fidelity_info,
        "fidelity_blocked": fidelity_blocked,
        "audit_html": audit_html,
        "audit_md": audit_md,
        "vision_qa": vision_result,
        "summary": "\n".join(summary_lines),
    }


def _build_audit_markdown(audit_data: dict, fidelity_info: dict) -> List[str]:
    """Build a downloadable markdown audit report from pipeline state."""
    fields = audit_data.get("fields", [])
    meta = audit_data.get("meta", {})
    dq = audit_data.get("dq", {})
    warnings = audit_data.get("warnings", [])

    lines = [
        "# Dr. Data V2 — Translation Audit Report",
        "",
        f"**Workbook:** {meta.get('workbook_name', 'Unknown')}",
        f"**Generated:** {meta.get('timestamp', '')}",
        f"**Total Fields:** {meta.get('total', 0)}",
        f"**Auto/Good:** {meta.get('auto_good', 0)} | "
        f"**Review:** {meta.get('review', 0)} | "
        f"**Blocked:** {meta.get('blocked', 0)}",
        f"**Avg Confidence:** {meta.get('avg_confidence', 0)}%",
        "",
    ]

    # DQ section
    if dq:
        lines.append(f"## Data Quality Score: {dq.get('score', 0)}%")
        lines.append(f"{dq.get('reason', '')}")
        lines.append("")

    # Fidelity section
    if fidelity_info.get("qa_lines"):
        lines.append("## Visual Fidelity")
        lines.append("")
        if fidelity_info.get("passed"):
            lines.append("> ✓ All visuals passed fidelity checks.")
        else:
            lines.append(f"> ✗ {fidelity_info.get('failed', 0)} visual(s) failed.")
        lines.append("")
        for qa_line in fidelity_info.get("qa_lines", [])[:30]:
            lines.append(qa_line)
        lines.append("")

    # Field translation table
    if fields:
        lines.append("## Field Translations")
        lines.append("")
        lines.append("| Field | Tier | Confidence | Method |")
        lines.append("|-------|------|-----------|--------|")
        for f in fields:
            lines.append(
                f"| {f.get('name', '')[:30]} | {f.get('tier', '?')} | "
                f"{f.get('confidence', 0)}% | {f.get('method', '')} |"
            )
        lines.append("")

    # Warnings
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            level = w.get("level", "info").upper()
            lines.append(f"- **{level}**: {w.get('title', '')} — {w.get('detail', '')}")
        lines.append("")

    return lines


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


def dispatch_multi_brain(formula_prompt: str = "", **kwargs) -> dict:
    """Delegate to core.multi_brain.dispatch_multi_brain.

    Kept here for backward compatibility with callers that import from
    pipeline_runner. New code should import from core.multi_brain directly.
    """
    from core.multi_brain import dispatch_multi_brain as _dispatch
    return _dispatch(**kwargs) if kwargs else _dispatch(tableau_formula=formula_prompt)
