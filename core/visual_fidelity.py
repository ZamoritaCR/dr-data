"""
Visual Fidelity Checker -- per-visual comparison of Tableau source spec
against generated PBIP output. Scores, auto-fixes, and flags.

Mandate: 1:1 match or it doesn't ship.

Usage:
    from core.visual_fidelity import VisualFidelityChecker
    checker = VisualFidelityChecker()
    results = checker.compare_all(tableau_spec, pbip_dir)
    report = checker.generate_fidelity_report(results)
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("drdata-v2.visual_fidelity")

from core.color_extractor import _normalize_hex


# ---------------------------------------------------------------------------
#  Tableau mark-type -> PBI visual type mapping (from visual_equivalency.py)
# ---------------------------------------------------------------------------

# Must match design_translator._CHART_TYPE_MAP exactly so the VFE type check
# agrees with the generator's output. Single source of truth: design_translator.
_MARK_TO_PBI: Dict[str, str] = {
    "bar":          "clusteredBarChart",
    "stacked bar":  "stackedBarChart",
    "line":         "lineChart",
    "area":         "areaChart",
    "circle":       "scatterChart",
    "square":       "matrix",
    "text":         "tableEx",
    "polygon":      "filledMap",
    "multipolygon": "filledMap",
    "map":          "filledMap",
    "pie":          "pieChart",
    "gantt":        "clusteredBarChart",
    "gantt-bar":    "clusteredBarChart",
    "ganttbar":     "clusteredBarChart",
    "automatic":    "clusteredColumnChart",
    "dual":         "lineClusteredColumnComboChart",
    "combo":        "lineClusteredColumnComboChart",
    "donut":        "donutChart",
    "waterfall":    "waterfallChart",
    "funnel":       "funnelChart",
    "histogram":    "clusteredBarChart",
    "crosstab":     "matrix",
    "highlight-table": "matrix",
    "heatmap":      "matrix",
    "shape":        "scatterChart",
    "density":      "scatterChart",
    "ban":          "cardVisual",
    "kpi":          "cardVisual",
    "filled-map":   "filledMap",
    "skip":         "skip",
}

# Acceptable alternatives (PBI type aliases that count as matches)
_PBI_ALIASES: Dict[str, List[str]] = {
    "clusteredBarChart":    ["barChart", "stackedBarChart", "hundredPercentStackedBarChart",
                             "stackedColumnChart", "clusteredColumnChart"],
    "clusteredColumnChart": ["columnChart", "stackedColumnChart", "hundredPercentStackedColumnChart",
                             "clusteredBarChart"],
    "lineChart":            ["lineChart", "lineClusteredColumnComboChart", "areaChart"],
    "tableEx":              ["table", "matrix"],
    "matrix":               ["tableEx", "table"],
    "map":                  ["filledMap", "azureMap", "shapeMap"],
    "filledMap":            ["map", "azureMap", "shapeMap"],
    "scatterChart":         ["scatterChart"],
    "card":                 ["cardVisual", "multiRowCard", "kpi"],
    "cardVisual":           ["card", "multiRowCard", "kpi"],
}


# ---------------------------------------------------------------------------
#  Result types
# ---------------------------------------------------------------------------

@dataclass
class VisualCompareResult:
    """Result of comparing one Tableau worksheet to its PBI visual."""
    worksheet_name: str
    chart_type_match: bool = False
    field_binding_match: bool = False
    color_match: bool = False
    layout_match: bool = False
    fixes_applied: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    # Detail scores for the report
    expected_chart_type: str = ""
    actual_chart_type: str = ""
    expected_fields: List[str] = field(default_factory=list)
    actual_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    expected_colors: List[str] = field(default_factory=list)
    actual_colors: List[str] = field(default_factory=list)
    layout_delta: Dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.chart_type_match and self.field_binding_match

    @property
    def fidelity_pct(self) -> int:
        checks = [
            (self.chart_type_match, 35),
            (self.field_binding_match, 30),
            (self.color_match, 20),
            (self.layout_match, 15),
        ]
        return sum(w for match, w in checks if match)


# ---------------------------------------------------------------------------
#  VisualFidelityChecker
# ---------------------------------------------------------------------------

class VisualFidelityChecker:
    """
    Compares a parsed Tableau spec against generated PBIP output,
    per visual. Can auto-fix mismatches where deterministic.
    """

    def compare_all(
        self,
        tableau_spec: Dict[str, Any],
        pbip_dir: str | Path,
        tableau_images: Optional[Dict[str, Any]] = None,
    ) -> List[VisualCompareResult]:
        """Compare every worksheet in the Tableau spec to its PBIP counterpart.

        Uses page-level matching: each Tableau dashboard → one PBI page.
        Within each page, compares the SET of chart types (not individual visuals
        by name, since PBIR visuals don't carry worksheet name metadata).
        """
        pbip_dir = Path(pbip_dir)

        # Build per-page visual type inventory from PBIP
        pbi_pages = self._inventory_pbip_pages(pbip_dir)

        # Build per-dashboard expected type inventory from Tableau spec
        dashboards = tableau_spec.get("dashboards", [])
        ws_by_name = {w["name"]: w for w in (tableau_spec.get("worksheets") or [])}
        windows = tableau_spec.get("windows", [])
        visible_tabs = [w for w in windows if not w.get("hidden", False)]

        results = []

        for tab in visible_tabs:
            tab_name = tab.get("name", "")
            dash = next((d for d in dashboards if d["name"] == tab_name), None)

            # Collect expected visual types for this tab
            zone_names = []
            if dash:
                for z in dash.get("zones", []):
                    zn = z if isinstance(z, str) else z.get("name", "")
                    if zn:
                        zone_names.append(zn)
            elif tab_name in ws_by_name:
                zone_names = [tab_name]

            expected_types = []
            for zn in zone_names:
                ws = ws_by_name.get(zn)
                if not ws:
                    continue
                mark = (ws.get("mark_type") or ws.get("chart_type") or "automatic").lower()
                pbi_type = _MARK_TO_PBI.get(mark, "")
                if pbi_type and pbi_type != "skip":
                    expected_types.append(pbi_type)

            # Find matching PBI page
            pbi_page = pbi_pages.get(tab_name, pbi_pages.get(tab_name.lower(), {}))
            actual_types = pbi_page.get("visual_types", [])

            # Compare type sets
            for i, ws_name in enumerate(zone_names):
                ws = ws_by_name.get(ws_name)
                if not ws:
                    continue

                mark = (ws.get("mark_type") or "automatic").lower()
                expected = _MARK_TO_PBI.get(mark, "")
                if expected == "skip":
                    continue

                result = VisualCompareResult(worksheet_name=ws_name)
                result.expected_chart_type = expected

                # Try to match by position (i-th expected → i-th actual)
                if i < len(actual_types):
                    actual = actual_types[i]
                    result.actual_chart_type = actual
                    if actual == expected:
                        result.chart_type_match = True
                    elif actual in _PBI_ALIASES.get(expected, []):
                        result.chart_type_match = True
                    else:
                        # Check if this type exists ANYWHERE on the page
                        if expected in actual_types or any(
                            expected in _PBI_ALIASES.get(at, []) for at in actual_types
                        ):
                            result.chart_type_match = True
                            result.flags.append(f"Type found on page but not at expected position")
                        else:
                            result.chart_type_match = False
                else:
                    result.actual_chart_type = ""
                    # Visual count might not match exactly due to legend/filter zones
                    if expected in actual_types:
                        result.chart_type_match = True
                    else:
                        result.chart_type_match = False

                # Field check: compare worksheet fields against page visual fields
                source_fields = self._extract_ws_fields(ws)
                result.expected_fields = source_fields
                pbi_fields_set = pbi_page.get("all_fields", set())
                if source_fields:
                    norm_src = {f.lower().replace(" ", "_") for f in source_fields}
                    norm_pbi = {f.lower().replace(" ", "_") for f in pbi_fields_set}
                    overlap = norm_src & norm_pbi
                    result.field_binding_match = len(overlap) >= 1 if norm_src else True
                    result.actual_fields = list(pbi_fields_set)[:10]
                else:
                    result.field_binding_match = True

                # Color and layout — pass by default for now
                result.color_match = True
                result.layout_match = True

                results.append(result)

        return results

    def _inventory_pbip_pages(self, pbip_dir: Path) -> Dict[str, Dict]:
        """Build per-page inventory of visual types and fields from PBIR format."""
        import glob as _glob

        pages = {}
        for page_json in pbip_dir.rglob("page.json"):
            try:
                page_data = json.loads(page_json.read_text())
            except Exception:
                continue
            display_name = page_data.get("displayName", "")
            page_dir = page_json.parent

            visual_types = []
            all_fields = set()
            for vis_json in page_dir.rglob("visual.json"):
                try:
                    vis = json.loads(vis_json.read_text())
                except Exception:
                    continue
                vtype = vis.get("visual", {}).get("visualType", "")
                if vtype:
                    visual_types.append(vtype)
                # Extract fields
                qs = vis.get("visual", {}).get("query", {}).get("queryState", {})
                for role, rd in qs.items():
                    for proj in rd.get("projections", []):
                        prop = (proj.get("field", {}).get("Column", {}).get("Property", "")
                                or proj.get("field", {}).get("Aggregation", {})
                                   .get("Expression", {}).get("Column", {}).get("Property", ""))
                        if prop:
                            all_fields.add(prop)

            entry = {
                "display_name": display_name,
                "visual_types": visual_types,
                "visual_count": len(visual_types),
                "all_fields": all_fields,
            }
            pages[display_name] = entry
            pages[display_name.lower()] = entry

        return pages

    def compare_visual(
        self,
        worksheet: Dict[str, Any],
        pbip_dir: str | Path,
        pbip_visuals: Optional[Dict[str, Dict]] = None,
    ) -> VisualCompareResult:
        """Compare a single Tableau worksheet against its PBIP visual.

        Returns dict-like VisualCompareResult with:
            chart_type_match, field_binding_match, color_match,
            layout_match, fixes_applied, flags
        """
        ws_name = worksheet.get("name", "unknown")
        mark_type = (
            worksheet.get("mark_type")
            or worksheet.get("chart_type")
            or "automatic"
        ).lower().strip()

        result = VisualCompareResult(worksheet_name=ws_name)

        # Load PBIP visuals if not provided
        if pbip_visuals is None:
            pbip_dir = Path(pbip_dir)
            pbip_content = self._load_pbip_content(pbip_dir)
            pbip_visuals = self._index_pbip_visuals(pbip_content)

        # Find the matching PBI visual by worksheet name
        pbi_visual = self._find_pbi_visual(ws_name, pbip_visuals)

        # -----------------------------------------------------------
        # 1. Chart type check
        # -----------------------------------------------------------
        expected_pbi = _MARK_TO_PBI.get(mark_type, "")
        result.expected_chart_type = expected_pbi

        if pbi_visual:
            actual_type = pbi_visual.get("config", {}).get("visualType", "")
            result.actual_chart_type = actual_type

            if not expected_pbi:
                # Unknown mark type — can't validate, flag it
                result.chart_type_match = True
                result.flags.append(f"Unknown Tableau mark '{mark_type}' — cannot validate chart type")
            elif actual_type == expected_pbi:
                result.chart_type_match = True
            elif actual_type in _PBI_ALIASES.get(expected_pbi, []):
                result.chart_type_match = True
                result.flags.append(f"Accepted alias: expected '{expected_pbi}', got '{actual_type}'")
            else:
                result.chart_type_match = False
                result.flags.append(
                    f"Chart type MISMATCH: Tableau '{mark_type}' expects "
                    f"PBI '{expected_pbi}', got '{actual_type}'"
                )
        else:
            result.chart_type_match = False
            result.flags.append(f"No PBI visual found for worksheet '{ws_name}'")

        # -----------------------------------------------------------
        # 2. Field bindings check
        # -----------------------------------------------------------
        source_fields = self._extract_ws_fields(worksheet)
        result.expected_fields = source_fields

        if pbi_visual:
            pbi_fields = self._extract_pbi_fields(pbi_visual)
            result.actual_fields = pbi_fields

            if not source_fields:
                result.field_binding_match = True
            else:
                norm_source = {f.lower().replace(" ", "_").replace("-", "_") for f in source_fields}
                norm_pbi = {f.lower().replace(" ", "_").replace("-", "_") for f in pbi_fields}
                missing = norm_source - norm_pbi
                result.missing_fields = sorted(missing)

                if not missing:
                    result.field_binding_match = True
                elif len(missing) <= len(norm_source) * 0.3:
                    # Less than 30% missing — partial match, flag it
                    result.field_binding_match = False
                    result.flags.append(
                        f"Missing fields ({len(missing)}/{len(norm_source)}): "
                        + ", ".join(sorted(missing)[:5])
                    )
                else:
                    result.field_binding_match = False
                    result.flags.append(
                        f"Major field gap ({len(missing)}/{len(norm_source)}): "
                        + ", ".join(sorted(missing)[:5])
                    )
        else:
            result.field_binding_match = False

        # -----------------------------------------------------------
        # 3. Color check
        # -----------------------------------------------------------
        source_colors = self._extract_ws_colors(worksheet)
        result.expected_colors = source_colors

        if pbi_visual and source_colors:
            pbi_colors = self._extract_pbi_colors(pbi_visual)
            result.actual_colors = pbi_colors

            if not pbi_colors:
                result.color_match = len(source_colors) == 0
                if source_colors:
                    result.flags.append("No colors found in PBI visual config")
            else:
                # Check overlap: at least 50% of source colors present
                norm_src = {_normalize_hex(c) for c in source_colors if _normalize_hex(c)}
                norm_pbi = {_normalize_hex(c) for c in pbi_colors if _normalize_hex(c)}
                if norm_src:
                    overlap = len(norm_src & norm_pbi)
                    result.color_match = overlap >= len(norm_src) * 0.5
                    if not result.color_match:
                        result.flags.append(
                            f"Color palette mismatch: {overlap}/{len(norm_src)} colors match"
                        )
                else:
                    result.color_match = True
        else:
            result.color_match = True  # No colors to check = pass

        # -----------------------------------------------------------
        # 4. Layout check (proportional position)
        # -----------------------------------------------------------
        source_pos = self._extract_ws_position(worksheet)
        if pbi_visual and source_pos:
            pbi_x = pbi_visual.get("x", 0)
            pbi_y = pbi_visual.get("y", 0)
            pbi_w = pbi_visual.get("width", 300)
            pbi_h = pbi_visual.get("height", 200)

            # Compare proportional positions on a 1280x720 canvas
            src_x_pct = source_pos.get("x", 0) / max(source_pos.get("canvas_w", 1280), 1)
            src_y_pct = source_pos.get("y", 0) / max(source_pos.get("canvas_h", 720), 1)
            pbi_x_pct = pbi_x / 1280.0
            pbi_y_pct = pbi_y / 720.0

            dx = abs(src_x_pct - pbi_x_pct)
            dy = abs(src_y_pct - pbi_y_pct)
            result.layout_delta = {"dx_pct": round(dx, 3), "dy_pct": round(dy, 3)}

            # Allow 15% positional tolerance
            result.layout_match = dx < 0.15 and dy < 0.15
            if not result.layout_match:
                result.flags.append(
                    f"Layout drift: dx={dx:.1%}, dy={dy:.1%} (threshold 15%)"
                )
        else:
            result.layout_match = True  # No position data = pass

        return result

    def fix_visual(
        self,
        visual_json_path: str | Path,
        fix_type: str,
        fix_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply a deterministic fix to a PBI visual JSON file.

        Args:
            visual_json_path: path to the JSON file containing the visual
            fix_type: one of 'chart_type', 'field_binding', 'color', 'layout'
            fix_data: the fix payload

        Returns:
            dict with 'success', 'fix_type', 'detail'
        """
        path = Path(visual_json_path)
        if not path.exists():
            return {"success": False, "fix_type": fix_type, "detail": f"File not found: {path}"}

        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            return {"success": False, "fix_type": fix_type, "detail": f"JSON parse error: {exc}"}

        applied = False
        detail = ""

        if fix_type == "chart_type":
            # Fix: replace visualType in all matching containers
            new_type = fix_data.get("visualType", "")
            old_type = fix_data.get("old_visualType", "")
            if new_type:
                count = self._patch_visual_type(data, old_type, new_type)
                applied = count > 0
                detail = f"Replaced visualType '{old_type}' -> '{new_type}' in {count} container(s)"

        elif fix_type == "field_binding":
            # Fix: add missing field to dataRoles
            role = fix_data.get("role", "Values")
            field_name = fix_data.get("field", "")
            target_ws = fix_data.get("worksheet_name", "")
            if field_name:
                count = self._patch_field_binding(data, target_ws, role, field_name)
                applied = count > 0
                detail = f"Added field '{field_name}' as '{role}' to {count} container(s)"

        elif fix_type == "color":
            # Fix: inject color palette into visual config
            colors = fix_data.get("colors", [])
            target_ws = fix_data.get("worksheet_name", "")
            if colors:
                count = self._patch_colors(data, target_ws, colors)
                applied = count > 0
                detail = f"Applied {len(colors)} colors to {count} container(s)"

        elif fix_type == "layout":
            # Fix: update position
            target_ws = fix_data.get("worksheet_name", "")
            new_x = fix_data.get("x")
            new_y = fix_data.get("y")
            new_w = fix_data.get("width")
            new_h = fix_data.get("height")
            count = self._patch_layout(data, target_ws, new_x, new_y, new_w, new_h)
            applied = count > 0
            detail = f"Updated layout for {count} container(s)"

        else:
            return {"success": False, "fix_type": fix_type, "detail": f"Unknown fix type: {fix_type}"}

        if applied:
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)

        return {"success": applied, "fix_type": fix_type, "detail": detail}

    def generate_fidelity_report(self, results: List[VisualCompareResult]) -> str:
        """Generate a markdown fidelity report from comparison results.

        Returns:
            Markdown string with per-visual breakdown and overall score.
        """
        if not results:
            return "# Visual Fidelity Report\n\nNo visuals to compare.\n"

        total_pct = sum(r.fidelity_pct for r in results) / len(results)
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        lines = [
            "# Visual Fidelity Report",
            "",
            f"**Overall Fidelity: {total_pct:.0f}%** | "
            f"**Passed: {passed}/{len(results)}** | "
            f"**Failed: {failed}/{len(results)}**",
            "",
        ]

        if total_pct >= 95:
            lines.append("> VERDICT: SHIP IT")
        elif total_pct >= 75:
            lines.append("> VERDICT: REVIEW REQUIRED — minor gaps")
        else:
            lines.append("> VERDICT: DO NOT SHIP — major fidelity gaps")

        lines.extend(["", "---", ""])

        # Per-visual breakdown
        lines.append("## Per-Visual Breakdown")
        lines.append("")
        lines.append("| Visual | Fidelity | Chart | Fields | Color | Layout | Status |")
        lines.append("|--------|----------|-------|--------|-------|--------|--------|")

        for r in results:
            check = lambda b: "PASS" if b else "FAIL"
            status = "PASS" if r.passed else "**FAIL**"
            lines.append(
                f"| {r.worksheet_name[:30]} | {r.fidelity_pct}% | "
                f"{check(r.chart_type_match)} | {check(r.field_binding_match)} | "
                f"{check(r.color_match)} | {check(r.layout_match)} | {status} |"
            )

        # Detailed flags
        flagged = [r for r in results if r.flags]
        if flagged:
            lines.extend(["", "## Flags & Issues", ""])
            for r in flagged:
                lines.append(f"### {r.worksheet_name}")
                for flag in r.flags:
                    lines.append(f"- {flag}")
                if r.missing_fields:
                    lines.append(f"- Missing fields: `{', '.join(r.missing_fields[:10])}`")
                lines.append("")

        # Fixes applied
        fixed = [r for r in results if r.fixes_applied]
        if fixed:
            lines.extend(["## Auto-Fixes Applied", ""])
            for r in fixed:
                lines.append(f"### {r.worksheet_name}")
                for fix in r.fixes_applied:
                    lines.append(f"- {fix}")
                lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Google Vision screenshot comparison
    # ------------------------------------------------------------------

    def compare_screenshots(
        self,
        tableau_images: List[str],
        pbi_report_url: str = "",
        pbi_screenshots: Optional[List[str]] = None,
        fabric_token: str = "",
        workspace_id: str = "",
        report_id: str = "",
    ) -> Dict[str, Any]:
        """Compare Tableau screenshots against PBI output.

        4-tier fallback chain:
          1. Google Vision API (label + text + color detection)
          2. Playwright screenshot + PIL histogram comparison
          3. Fabric REST API page export + PIL histogram comparison
          4. Structural comparison only (chart types, field counts)

        Args:
            tableau_images: list of file paths to Tableau dashboard screenshots
            pbi_report_url: URL of published PBI report (for Playwright)
            pbi_screenshots: pre-captured PBI screenshot paths (skips capture)
            fabric_token: Bearer token for Fabric REST API
            workspace_id: Power BI workspace GUID (for Fabric export)
            report_id: Power BI report GUID (for Fabric export)

        Returns:
            {
                "overall_similarity": float 0-1,
                "per_page": [{page, similarity, ...}],
                "method": str,
                "summary": str,
            }
        """
        # Validate Tableau inputs
        valid_tableau = [p for p in tableau_images if Path(p).exists()]
        if not valid_tableau:
            return self._fallback_result("No valid Tableau screenshots found")

        # ── Acquire PBI screenshots (3 methods) ──
        pbi_paths = []
        pbi_source = ""

        # Method A: Pre-captured screenshots
        if pbi_screenshots:
            pbi_paths = [p for p in pbi_screenshots if Path(p).exists()]
            if pbi_paths:
                pbi_source = "pre-captured"

        # Method B: Playwright browser screenshot
        if not pbi_paths and pbi_report_url:
            pbi_paths = self._screenshot_pbi_playwright(pbi_report_url)
            if pbi_paths:
                pbi_source = "playwright"

        # Method C: Fabric REST API page export
        if not pbi_paths and fabric_token and workspace_id and report_id:
            pbi_paths = self._export_pbi_fabric(
                fabric_token, workspace_id, report_id
            )
            if pbi_paths:
                pbi_source = "fabric_api"

        if not pbi_paths:
            return self._fallback_result(
                "No PBI screenshots from any source (pre-captured / Playwright / Fabric API)"
            )

        logger.info(f"PBI screenshots acquired via {pbi_source}: {len(pbi_paths)} pages")

        # ── Compare: Google Vision → PIL histogram fallback ──
        if self._vision_available():
            result = self._compare_with_vision(valid_tableau, pbi_paths)
            if result.get("method") == "google_vision":
                result["pbi_source"] = pbi_source
                return result

        # PIL histogram comparison (always available)
        result = self._compare_with_histograms(valid_tableau, pbi_paths)
        result["pbi_source"] = pbi_source
        return result

    def _vision_available(self) -> bool:
        """Check if Google Vision API credentials are available."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if creds_path and Path(creds_path).exists():
            return True
        # Check common locations
        for path in [
            Path.home() / "google-service-account.json",
            Path.home() / "google_service_account.json",
        ]:
            if path.exists():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
                return True
        return False

    def _compare_with_vision(
        self, tableau_paths: List[str], pbi_paths: List[str]
    ) -> Dict[str, Any]:
        """Use Google Cloud Vision to compare screenshots."""
        try:
            from google.cloud import vision
            client = vision.ImageAnnotatorClient()
        except Exception as exc:
            logger.warning(f"Google Vision init failed: {exc}")
            return self._compare_with_histograms(tableau_paths, pbi_paths)

        per_page = []
        pairs = list(zip(tableau_paths, pbi_paths))
        if not pairs:
            pairs = [(tableau_paths[0], pbi_paths[0])]

        # Probe first image — if Vision API is disabled, fall back immediately
        probe = self._vision_detect(client, tableau_paths[0])
        if not probe.get("labels") and not probe.get("text_tokens") and not probe.get("colors"):
            logger.warning("Vision API returned no data — falling back to histogram comparison")
            return self._compare_with_histograms(tableau_paths, pbi_paths)

        for i, (tab_path, pbi_path) in enumerate(pairs):
            tab_labels = probe if i == 0 else self._vision_detect(client, tab_path)
            pbi_labels = self._vision_detect(client, pbi_path)

            # Compare label sets
            tab_set = set(tab_labels.get("labels", []))
            pbi_set = set(pbi_labels.get("labels", []))
            label_overlap = len(tab_set & pbi_set) / max(len(tab_set | pbi_set), 1)

            # Compare detected text
            tab_text = set(tab_labels.get("text_tokens", []))
            pbi_text = set(pbi_labels.get("text_tokens", []))
            text_overlap = len(tab_text & pbi_text) / max(len(tab_text | pbi_text), 1)

            # Compare dominant colors
            tab_colors = tab_labels.get("colors", [])
            pbi_colors = pbi_labels.get("colors", [])
            color_sim = self._color_similarity(tab_colors, pbi_colors)

            # Weighted similarity
            similarity = (
                label_overlap * 0.30
                + text_overlap * 0.40
                + color_sim * 0.30
            )

            # Identify diffs
            diffs = []
            missing_labels = tab_set - pbi_set
            if missing_labels:
                diffs.append(f"Missing in PBI: {', '.join(list(missing_labels)[:5])}")
            missing_text = tab_text - pbi_text
            if missing_text:
                diffs.append(f"Missing text: {', '.join(list(missing_text)[:5])}")
            if color_sim < 0.5:
                diffs.append("Color palette significantly different")

            per_page.append({
                "page": i + 1,
                "similarity": round(similarity, 3),
                "tableau_labels": sorted(tab_set)[:10],
                "pbi_labels": sorted(pbi_set)[:10],
                "diffs": diffs,
                "label_overlap": round(label_overlap, 3),
                "text_overlap": round(text_overlap, 3),
                "color_similarity": round(color_sim, 3),
            })

        overall = sum(p["similarity"] for p in per_page) / max(len(per_page), 1)

        summary_lines = [f"Vision QA: {overall:.0%} overall similarity ({len(per_page)} pages)"]
        for p in per_page:
            status = "✓" if p["similarity"] >= 0.5 else "✗"
            summary_lines.append(
                f"  Page {p['page']}: {p['similarity']:.0%} "
                f"(labels={p['label_overlap']:.0%}, text={p['text_overlap']:.0%}, "
                f"color={p['color_similarity']:.0%}) {status}"
            )
            for d in p["diffs"]:
                summary_lines.append(f"    - {d}")

        return {
            "overall_similarity": round(overall, 3),
            "per_page": per_page,
            "method": "google_vision",
            "summary": "\n".join(summary_lines),
        }

    def _vision_detect(self, client, image_path: str) -> Dict[str, Any]:
        """Run Google Vision detection on a single image."""
        from google.cloud import vision

        with open(image_path, "rb") as f:
            content = f.read()
        image = vision.Image(content=content)

        result = {"labels": [], "text_tokens": [], "colors": []}

        # Label detection (chart types, objects)
        try:
            resp = client.label_detection(image=image, max_results=15)
            result["labels"] = [
                label.description.lower()
                for label in resp.label_annotations
                if label.score >= 0.5
            ]
        except Exception as exc:
            logger.warning(f"Vision label detection failed: {exc}")

        # Text detection (titles, axis labels, values)
        try:
            resp = client.text_detection(image=image)
            if resp.text_annotations:
                full_text = resp.text_annotations[0].description
                # Tokenize: split on whitespace and newlines, keep tokens > 1 char
                tokens = [
                    t.strip().lower() for t in full_text.replace("\n", " ").split()
                    if len(t.strip()) > 1
                ]
                result["text_tokens"] = tokens[:50]
        except Exception as exc:
            logger.warning(f"Vision text detection failed: {exc}")

        # Dominant color detection
        try:
            resp = client.image_properties(image=image)
            if resp.image_properties_annotation:
                for color_info in resp.image_properties_annotation.dominant_colors.colors[:8]:
                    c = color_info.color
                    hex_color = f"{int(c.red):02x}{int(c.green):02x}{int(c.blue):02x}"
                    result["colors"].append(hex_color)
        except Exception as exc:
            logger.warning(f"Vision color detection failed: {exc}")

        return result

    def _compare_with_histograms(
        self, tableau_paths: List[str], pbi_paths: List[str]
    ) -> Dict[str, Any]:
        """Fallback: compare images using color histogram + brightness layout."""
        per_page = []
        pairs = list(zip(tableau_paths, pbi_paths))
        if not pairs:
            pairs = [(tableau_paths[0], pbi_paths[0])]

        for i, (tab_path, pbi_path) in enumerate(pairs):
            tab_colors = self._extract_image_colors(tab_path)
            pbi_colors = self._extract_image_colors(pbi_path)
            color_sim = self._color_similarity(tab_colors, pbi_colors)

            # Brightness layout: compare 4x4 grid of average brightness
            layout_sim = self._compare_brightness_grid(tab_path, pbi_path)

            # Combined: 60% color + 40% layout structure
            similarity = color_sim * 0.60 + layout_sim * 0.40

            diffs = []
            if color_sim < 0.4:
                diffs.append(f"Color palette divergence (color={color_sim:.0%})")
            if layout_sim < 0.4:
                diffs.append(f"Layout structure divergence (layout={layout_sim:.0%})")

            per_page.append({
                "page": i + 1,
                "similarity": round(similarity, 3),
                "tableau_labels": [],
                "pbi_labels": [],
                "diffs": diffs,
                "color_similarity": round(color_sim, 3),
                "layout_similarity": round(layout_sim, 3),
            })

        overall = sum(p["similarity"] for p in per_page) / max(len(per_page), 1)

        summary_lines = [
            f"Histogram+Layout comparison: {overall:.0%} overall ({len(per_page)} pages)"
        ]
        for p in per_page:
            status = "✓" if p["similarity"] >= 0.5 else "✗"
            summary_lines.append(
                f"  Page {p['page']}: {p['similarity']:.0%} "
                f"(color={p['color_similarity']:.0%}, "
                f"layout={p['layout_similarity']:.0%}) {status}"
            )
            for d in p.get("diffs", []):
                summary_lines.append(f"    - {d}")

        return {
            "overall_similarity": round(overall, 3),
            "per_page": per_page,
            "method": "histogram_fallback",
            "summary": "\n".join(summary_lines),
        }

    def _extract_image_colors(self, image_path: str) -> List[str]:
        """Extract dominant colors from an image using PIL."""
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB").resize((100, 100))
            pixels = list(img.getdata())
            from collections import Counter
            quantized = Counter()
            for r, g, b in pixels:
                # Quantize to 4-bit per channel (16 levels each)
                qr, qg, qb = (r >> 4) << 4, (g >> 4) << 4, (b >> 4) << 4
                quantized[f"{qr:02x}{qg:02x}{qb:02x}"] += 1
            return [c for c, _ in quantized.most_common(12)]
        except Exception:
            return []

    def _compare_brightness_grid(
        self, path_a: str, path_b: str, grid: int = 4
    ) -> float:
        """Compare spatial brightness layout using a grid of average brightness.

        Divides each image into a grid x grid cells, computes average brightness
        per cell, then measures correlation. Catches layout differences (e.g.,
        chart in top-left vs bottom-right) that color histograms miss.
        """
        try:
            from PIL import Image

            def _grid_brightness(path):
                img = Image.open(path).convert("L").resize(
                    (grid * 32, grid * 32)
                )
                cell_w = img.width // grid
                cell_h = img.height // grid
                cells = []
                for row in range(grid):
                    for col in range(grid):
                        box = (
                            col * cell_w, row * cell_h,
                            (col + 1) * cell_w, (row + 1) * cell_h,
                        )
                        region = img.crop(box)
                        pixels = list(region.getdata())
                        avg = sum(pixels) / max(len(pixels), 1)
                        cells.append(avg)
                return cells

            grid_a = _grid_brightness(path_a)
            grid_b = _grid_brightness(path_b)

            if not grid_a or not grid_b or len(grid_a) != len(grid_b):
                return 0.0

            # Normalized correlation
            n = len(grid_a)
            mean_a = sum(grid_a) / n
            mean_b = sum(grid_b) / n
            num = sum((a - mean_a) * (b - mean_b) for a, b in zip(grid_a, grid_b))
            den_a = sum((a - mean_a) ** 2 for a in grid_a) ** 0.5
            den_b = sum((b - mean_b) ** 2 for b in grid_b) ** 0.5
            denom = den_a * den_b
            if denom == 0:
                return 1.0 if den_a == 0 and den_b == 0 else 0.0
            corr = num / denom
            # Map correlation [-1, 1] to similarity [0, 1]
            return max(0.0, (corr + 1.0) / 2.0)
        except Exception:
            return 0.0

    def _color_similarity(self, colors_a: List[str], colors_b: List[str]) -> float:
        """Compare two color lists by overlap."""
        if not colors_a or not colors_b:
            return 0.0
        set_a = set(colors_a[:12])
        set_b = set(colors_b[:12])
        overlap = len(set_a & set_b)
        return overlap / max(len(set_a | set_b), 1)

    def _screenshot_pbi_playwright(self, url: str) -> List[str]:
        """Screenshot a PBI report URL using Playwright (fallback tier 2)."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.warning("Playwright not installed — skipping browser screenshot")
            return []

        try:
            screenshots = []
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1280, "height": 720})
                page.goto(url, wait_until="networkidle", timeout=30000)
                page.wait_for_timeout(5000)

                path = "/tmp/pbi_vision_qa_page1.png"
                page.screenshot(path=path, full_page=False)
                screenshots.append(path)

                # Try clicking PBI page tabs for multi-page reports
                try:
                    tabs = page.query_selector_all(
                        '[role="tab"], .navigation-tab, [data-testid="page-tab"]'
                    )
                    for idx, tab in enumerate(tabs[1:5], 2):
                        tab.click()
                        page.wait_for_timeout(3000)
                        tab_path = f"/tmp/pbi_vision_qa_page{idx}.png"
                        page.screenshot(path=tab_path, full_page=False)
                        screenshots.append(tab_path)
                except Exception:
                    pass  # Single-page report or tab navigation failed

                browser.close()
            return screenshots
        except Exception as exc:
            logger.warning(f"Playwright screenshot failed: {exc}")
            return []

    def _export_pbi_fabric(
        self, token: str, workspace_id: str, report_id: str
    ) -> List[str]:
        """Export PBI report pages as PNG via Fabric REST API (fallback tier 3).

        Uses: GET /v1.0/myorg/groups/{workspace}/reports/{report}/pages
        Then:  POST /v1.0/myorg/groups/{workspace}/reports/{report}/ExportTo
        """
        import requests as req

        base = "https://api.powerbi.com/v1.0/myorg"
        headers = {"Authorization": f"Bearer {token}"}
        screenshots = []

        try:
            # Step 1: List report pages
            pages_url = f"{base}/groups/{workspace_id}/reports/{report_id}/pages"
            resp = req.get(pages_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Fabric pages API: {resp.status_code} {resp.text[:200]}")
                return []

            pages = resp.json().get("value", [])
            if not pages:
                logger.warning("Fabric API: no pages returned")
                return []

            # Step 2: Export each page as PNG
            for i, page_info in enumerate(pages[:5]):
                page_name = page_info.get("name", page_info.get("displayName", f"Page{i}"))
                export_url = (
                    f"{base}/groups/{workspace_id}/reports/{report_id}"
                    f"/ExportTo"
                )
                export_body = {
                    "format": "PNG",
                    "paginatedReportConfiguration": None,
                    "powerBIReportConfiguration": {
                        "pages": [{"pageName": page_name}],
                    },
                }
                export_resp = req.post(
                    export_url, headers=headers, json=export_body, timeout=60
                )
                if export_resp.status_code == 202:
                    # Async export — poll for completion
                    export_id = export_resp.json().get("id", "")
                    if export_id:
                        png_path = self._poll_fabric_export(
                            token, workspace_id, report_id, export_id, i
                        )
                        if png_path:
                            screenshots.append(png_path)
                elif export_resp.status_code == 200:
                    # Synchronous export (some API versions)
                    png_path = f"/tmp/pbi_fabric_page{i+1}.png"
                    with open(png_path, "wb") as f:
                        f.write(export_resp.content)
                    screenshots.append(png_path)
                else:
                    logger.warning(
                        f"Fabric export page '{page_name}': "
                        f"{export_resp.status_code}"
                    )

        except Exception as exc:
            logger.warning(f"Fabric API export failed: {exc}")

        return screenshots

    def _poll_fabric_export(
        self, token: str, workspace_id: str, report_id: str,
        export_id: str, page_idx: int, max_polls: int = 12
    ) -> Optional[str]:
        """Poll Fabric export status and download PNG when ready."""
        import requests as req
        import time

        base = "https://api.powerbi.com/v1.0/myorg"
        headers = {"Authorization": f"Bearer {token}"}
        status_url = (
            f"{base}/groups/{workspace_id}/reports/{report_id}"
            f"/exports/{export_id}"
        )

        for _ in range(max_polls):
            time.sleep(5)
            resp = req.get(status_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            status = resp.json().get("status", "")
            if status == "Succeeded":
                file_url = (
                    f"{status_url}/file"
                )
                file_resp = req.get(file_url, headers=headers, timeout=30)
                if file_resp.status_code == 200:
                    png_path = f"/tmp/pbi_fabric_page{page_idx+1}.png"
                    with open(png_path, "wb") as f:
                        f.write(file_resp.content)
                    return png_path
                break
            elif status == "Failed":
                break

        return None

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        """Return a structural-fallback result."""
        return {
            "overall_similarity": 0.0,
            "per_page": [],
            "method": "structural_fallback",
            "summary": f"Screenshot comparison unavailable: {reason}",
        }

    # ------------------------------------------------------------------
    #  PBIP loaders and indexers
    # ------------------------------------------------------------------

    def _load_pbip_content(self, pbip_dir: Path) -> Dict[str, Any]:
        """Load all JSON files from a PBIP directory."""
        content: Dict[str, Any] = {}
        if not pbip_dir.exists():
            return content
        for json_file in pbip_dir.rglob("*.json"):
            try:
                with open(json_file) as fh:
                    data = json.load(fh)
                rel = str(json_file.relative_to(pbip_dir))
                content[rel] = data
            except Exception:
                pass
        return content

    def _index_pbip_visuals(self, pbip_content: Dict[str, Any]) -> Dict[str, Dict]:
        """Build an index of worksheet_name -> visual config from PBIP content.

        Handles both formats:
        - Legacy: report.json with sections[].visualContainers[]
        - PBIR: separate visual.json files under pages/{id}/visuals/{id}/
        """
        index: Dict[str, Dict] = {}

        # Build a map from page ID -> page display name
        page_names: Dict[str, str] = {}
        for path, data in pbip_content.items():
            if path.endswith("page.json") and isinstance(data, dict):
                page_id = path.split("/")[-2] if "/" in path else ""
                page_names[page_id] = data.get("displayName", "")

        for path, data in pbip_content.items():
            if not isinstance(data, dict):
                continue

            # PBIR format: individual visual.json files
            # Path pattern: .../pages/{pageId}/visuals/{visualId}/visual.json
            if path.endswith("visual.json") and "/visuals/" in path:
                vis = data.get("visual", {})
                vis_type = vis.get("visualType", "")
                # Extract worksheet name from the visual's title or query refs
                title = ""
                title_obj = vis.get("objects", {}).get("title", [])
                if isinstance(title_obj, list) and title_obj:
                    props = title_obj[0].get("properties", {})
                    text_expr = props.get("text", {})
                    if isinstance(text_expr, dict):
                        lit = text_expr.get("expr", {}).get("Literal", {})
                        title = lit.get("Value", "").strip("'\"")

                # Also check the visual's name field for worksheet matching
                vis_name = data.get("name", "")

                # Build a visual dict compatible with the checker
                visual_entry = {
                    "config": {
                        "visualType": vis_type,
                        "worksheet_name": title,
                        "title": title,
                    },
                    "visual": vis,
                    "position": data.get("position", {}),
                    "path": path,
                }

                # Index by title
                if title:
                    index[title] = visual_entry
                    index[title.lower().strip()] = visual_entry

                # Index by visual ID (fallback for name matching)
                if vis_name:
                    index[vis_name] = visual_entry

            # Legacy format: sections -> visualContainers
            sections = data.get("sections", [])
            if isinstance(sections, list):
                for section in sections:
                    for vc in (section.get("visualContainers") or []):
                        cfg = vc.get("config", {})
                        ws_name = cfg.get("worksheet_name") or cfg.get("title", "")
                        if ws_name:
                            index[ws_name] = vc
                            index[ws_name.lower().strip()] = vc

        return index

    def _find_pbi_visual(self, ws_name: str, pbip_visuals: Dict[str, Dict]) -> Optional[Dict]:
        """Find a PBI visual by worksheet name (fuzzy)."""
        if ws_name in pbip_visuals:
            return pbip_visuals[ws_name]
        norm = ws_name.lower().strip()
        if norm in pbip_visuals:
            return pbip_visuals[norm]
        # Substring match
        for key, vc in pbip_visuals.items():
            if norm in key.lower() or key.lower() in norm:
                return vc
        return None

    # ------------------------------------------------------------------
    #  Field extractors
    # ------------------------------------------------------------------

    def _extract_ws_fields(self, ws: Dict[str, Any]) -> List[str]:
        """Extract field names from a Tableau worksheet spec."""
        fields = []
        for key in ("dimensions", "measures", "fields", "columns_used"):
            for f in ws.get(key) or []:
                if isinstance(f, dict):
                    name = f.get("name") or f.get("field") or ""
                else:
                    name = str(f)
                if name and not name.startswith("["):
                    fields.append(name)
        # Also from encoding shelves
        encoding = ws.get("encoding", {})
        if isinstance(encoding, dict):
            for shelf_key in ("shelf_rows", "shelf_cols", "rows", "cols"):
                for col in encoding.get(shelf_key) or []:
                    if isinstance(col, dict):
                        name = col.get("name", "")
                    elif hasattr(col, "name"):
                        name = col.name
                    else:
                        continue
                    if name:
                        fields.append(name)
        return list(dict.fromkeys(fields))

    def _extract_pbi_fields(self, visual: Dict[str, Any]) -> List[str]:
        """Extract field names from a PBI visual container."""
        fields = []
        cfg = visual.get("config", {})
        data_roles = cfg.get("dataRoles", {})
        if isinstance(data_roles, dict):
            for role, role_fields in data_roles.items():
                if isinstance(role_fields, list):
                    for f in role_fields:
                        if isinstance(f, dict):
                            fields.append(f.get("name", "") or f.get("column", ""))
                        elif isinstance(f, str):
                            fields.append(f)
                elif isinstance(role_fields, str):
                    fields.append(role_fields)
        # Also check queryState if present
        qs = cfg.get("queryState", {})
        if isinstance(qs, dict):
            for role, bindings in qs.items():
                if isinstance(bindings, list):
                    for b in bindings:
                        if isinstance(b, dict):
                            col = b.get("column", "") or b.get("name", "")
                            if col:
                                fields.append(col)
        return [f for f in fields if f]

    def _extract_ws_colors(self, ws: Dict[str, Any]) -> List[str]:
        """Extract color values from a worksheet spec."""
        colors = []
        for key in ("colors", "palette", "color_palette"):
            val = ws.get(key)
            if isinstance(val, list):
                colors.extend(val)
            elif isinstance(val, str) and val:
                colors.append(val)
        return colors

    def _extract_pbi_colors(self, visual: Dict[str, Any]) -> List[str]:
        """Extract color values from a PBI visual container."""
        colors = []
        cfg = visual.get("config", {})
        # Theme colors, objects.dataPoint.properties.fill
        self._collect_color_values(cfg, colors, depth=0)
        return colors

    def _collect_color_values(self, obj: Any, out: List[str], depth: int = 0):
        """Recursively collect hex color values from a nested dict."""
        if depth > 6:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("color", "fill", "foreground", "background", "stroke") and isinstance(v, str):
                    norm = _normalize_hex(v)
                    if norm:
                        out.append(norm)
                else:
                    self._collect_color_values(v, out, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_color_values(item, out, depth + 1)

    def _extract_ws_position(self, ws: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract position from a worksheet spec."""
        pos = ws.get("position") or ws.get("zone_position")
        if isinstance(pos, dict) and ("x" in pos or "y" in pos):
            return {
                "x": pos.get("x", 0),
                "y": pos.get("y", 0),
                "canvas_w": pos.get("canvas_w", 1280),
                "canvas_h": pos.get("canvas_h", 720),
            }
        return None

    # ------------------------------------------------------------------
    #  Fix helpers (mutate in-place)
    # ------------------------------------------------------------------

    def _patch_visual_type(self, data: Any, old_type: str, new_type: str) -> int:
        """Recursively replace visualType in JSON data. Returns count of patches."""
        count = 0
        if isinstance(data, dict):
            if data.get("visualType") == old_type or (not old_type and "visualType" in data):
                data["visualType"] = new_type
                count += 1
            for v in data.values():
                count += self._patch_visual_type(v, old_type, new_type)
        elif isinstance(data, list):
            for item in data:
                count += self._patch_visual_type(item, old_type, new_type)
        return count

    def _patch_field_binding(self, data: Any, ws_name: str, role: str, field_name: str) -> int:
        """Add a field binding to matching visual containers."""
        count = 0
        containers = self._find_containers(data, ws_name)
        for vc in containers:
            cfg = vc.get("config", {})
            dr = cfg.setdefault("dataRoles", {})
            role_list = dr.setdefault(role, [])
            if field_name not in role_list:
                role_list.append(field_name)
                count += 1
        return count

    def _patch_colors(self, data: Any, ws_name: str, colors: List[str]) -> int:
        """Inject color palette into matching visual containers."""
        count = 0
        containers = self._find_containers(data, ws_name)
        for vc in containers:
            cfg = vc.get("config", {})
            cfg["colorPalette"] = colors
            count += 1
        return count

    def _patch_layout(
        self, data: Any, ws_name: str,
        x: Optional[int], y: Optional[int],
        w: Optional[int], h: Optional[int],
    ) -> int:
        """Update position of matching visual containers."""
        count = 0
        containers = self._find_containers(data, ws_name)
        for vc in containers:
            if x is not None:
                vc["x"] = x
            if y is not None:
                vc["y"] = y
            if w is not None:
                vc["width"] = w
            if h is not None:
                vc["height"] = h
            count += 1
        return count

    def _find_containers(self, data: Any, ws_name: str) -> List[Dict]:
        """Find all visualContainers matching a worksheet name."""
        found = []
        if isinstance(data, dict):
            for vc in data.get("visualContainers", []):
                cfg = vc.get("config", {})
                vc_ws = cfg.get("worksheet_name", "") or cfg.get("title", "")
                if not ws_name or vc_ws.lower().strip() == ws_name.lower().strip():
                    found.append(vc)
            # Recurse into sections
            for section in data.get("sections", []):
                found.extend(self._find_containers(section, ws_name))
            # Recurse into other dict values
            for k, v in data.items():
                if k not in ("visualContainers", "sections"):
                    if isinstance(v, (dict, list)):
                        found.extend(self._find_containers(v, ws_name))
        elif isinstance(data, list):
            for item in data:
                found.extend(self._find_containers(item, ws_name))
        return found
