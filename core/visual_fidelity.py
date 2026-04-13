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

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.color_extractor import _normalize_hex


# ---------------------------------------------------------------------------
#  Tableau mark-type -> PBI visual type mapping (from visual_equivalency.py)
# ---------------------------------------------------------------------------

_MARK_TO_PBI: Dict[str, str] = {
    "bar":          "clusteredBarChart",
    "stacked bar":  "stackedBarChart",
    "line":         "lineChart",
    "area":         "areaChart",
    "circle":       "scatterChart",
    "square":       "treemap",
    "text":         "tableEx",
    "polygon":      "map",
    "map":          "map",
    "pie":          "pieChart",
    "gantt":        "clusteredBarChart",
    "ganttbar":     "clusteredBarChart",
    "automatic":    "clusteredBarChart",
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
    "ban":          "card",
    "filled-map":   "filledMap",
}

# Acceptable alternatives (PBI type aliases that count as matches)
_PBI_ALIASES: Dict[str, List[str]] = {
    "clusteredBarChart":    ["barChart", "stackedBarChart", "hundredPercentStackedBarChart",
                             "stackedColumnChart", "clusteredColumnChart"],
    "clusteredColumnChart": ["columnChart", "stackedColumnChart", "hundredPercentStackedColumnChart"],
    "lineChart":            ["lineChart", "lineClusteredColumnComboChart"],
    "tableEx":              ["table", "matrix"],
    "matrix":               ["tableEx", "table"],
    "map":                  ["filledMap", "azureMap", "shapeMap"],
    "filledMap":            ["map", "azureMap", "shapeMap"],
    "scatterChart":         ["scatterChart"],
    "card":                 ["multiRowCard", "kpi"],
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

        Args:
            tableau_spec: parsed spec from enhanced_tableau_parser.parse_twb()
            pbip_dir: path to generated PBIP output directory
            tableau_images: optional dict of worksheet_name -> image path (unused for now)

        Returns:
            List of VisualCompareResult, one per worksheet.
        """
        pbip_dir = Path(pbip_dir)
        pbip_content = self._load_pbip_content(pbip_dir)
        pbip_visuals = self._index_pbip_visuals(pbip_content)

        worksheets = tableau_spec.get("worksheets") or []
        results = []

        for ws in worksheets:
            result = self.compare_visual(ws, pbip_dir, pbip_visuals=pbip_visuals)
            results.append(result)

        return results

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
        """Build an index of worksheet_name -> visual container from PBIP content."""
        index: Dict[str, Dict] = {}

        for path, data in pbip_content.items():
            if not isinstance(data, dict):
                continue
            # Look for sections -> visualContainers
            sections = data.get("sections", [])
            if isinstance(sections, list):
                for section in sections:
                    for vc in (section.get("visualContainers") or []):
                        cfg = vc.get("config", {})
                        ws_name = cfg.get("worksheet_name") or cfg.get("title", "")
                        if ws_name:
                            index[ws_name] = vc
                            index[ws_name.lower().strip()] = vc

            # Also check top-level visualContainers
            for vc in (data.get("visualContainers") or []):
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
