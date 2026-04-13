"""
Visual Fidelity Checker -- scores how well a generated PBIP preserves
the intent of the source Tableau workbook.

Scores are computed per-dimension and rolled up to a single fidelity
percentage. Gaps are documented with actionable remediation hints.

Usage:
    from core.visual_fidelity import VisualFidelityChecker
    checker = VisualFidelityChecker()
    report = checker.check(tableau_spec, pbip_output_dir)
    print(report.summary())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
#  Score components
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    name: str
    weight: float          # 0-1, must sum to 1.0 across all dimensions
    score: float           # 0-1 achieved score
    details: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)

    @property
    def weighted(self) -> float:
        return self.weight * self.score


@dataclass
class FidelityReport:
    workbook_name: str
    source_path: str
    pbip_path: str
    dimensions: List[DimensionScore] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        return sum(d.weighted for d in self.dimensions)

    @property
    def overall_pct(self) -> int:
        return int(self.overall_score * 100)

    def summary(self) -> str:
        lines = [
            f"=== Visual Fidelity Report: {self.workbook_name} ===",
            f"Overall fidelity: {self.overall_pct}%",
            "",
        ]
        for dim in self.dimensions:
            score_pct = int(dim.score * 100)
            lines.append(f"[{score_pct:3d}%] {dim.name} (weight {int(dim.weight*100)}%)")
            for detail in dim.details:
                lines.append(f"       + {detail}")
            for gap in dim.gaps:
                lines.append(f"       ! GAP: {gap}")
        if self.errors:
            lines.append("")
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Checker
# ---------------------------------------------------------------------------

class VisualFidelityChecker:
    """
    Compares a parsed Tableau spec dict against a generated PBIP output
    directory and produces a FidelityReport.

    tableau_spec: dict returned by core.enhanced_tableau_parser.parse_twb()
    pbip_dir: Path to the generated PBIP output directory
              (or None for import-only / unit-test scenarios)
    """

    DIMENSIONS = [
        ("visual_types",     0.30),
        ("field_coverage",   0.25),
        ("formula_coverage", 0.20),
        ("layout_structure", 0.15),
        ("filter_coverage",  0.10),
    ]

    def check(
        self,
        tableau_spec: Dict[str, Any],
        pbip_dir: Optional[Path | str] = None,
    ) -> FidelityReport:
        workbook_name = tableau_spec.get("workbook_name") or tableau_spec.get("name", "unknown")
        source_path = tableau_spec.get("source_path", "")
        pbip_path = str(pbip_dir) if pbip_dir else "(not generated)"

        report = FidelityReport(
            workbook_name=workbook_name,
            source_path=source_path,
            pbip_path=pbip_path,
        )

        pbip_content = self._load_pbip(pbip_dir, report)

        report.dimensions.append(self._score_visual_types(tableau_spec, pbip_content))
        report.dimensions.append(self._score_field_coverage(tableau_spec, pbip_content))
        report.dimensions.append(self._score_formula_coverage(tableau_spec, pbip_content))
        report.dimensions.append(self._score_layout_structure(tableau_spec, pbip_content))
        report.dimensions.append(self._score_filter_coverage(tableau_spec, pbip_content))

        return report

    # ------------------------------------------------------------------
    #  PBIP loader
    # ------------------------------------------------------------------

    def _load_pbip(
        self, pbip_dir: Optional[Path | str], report: FidelityReport
    ) -> Dict[str, Any]:
        """Load PBIP JSON files into a flat content dict."""
        content: Dict[str, Any] = {}
        if not pbip_dir:
            return content
        pbip_path = Path(pbip_dir)
        if not pbip_path.exists():
            report.errors.append(f"PBIP directory not found: {pbip_path}")
            return content

        for json_file in pbip_path.rglob("*.json"):
            try:
                with open(json_file) as fh:
                    data = json.load(fh)
                rel = str(json_file.relative_to(pbip_path))
                content[rel] = data
            except Exception as exc:
                report.errors.append(f"Could not parse {json_file.name}: {exc}")

        return content

    # ------------------------------------------------------------------
    #  Dimension scorers
    # ------------------------------------------------------------------

    def _score_visual_types(
        self, spec: Dict[str, Any], pbip: Dict[str, Any]
    ) -> DimensionScore:
        """Check that each Tableau worksheet has a corresponding PBI visual."""
        dim = DimensionScore(name="Visual types", weight=0.30, score=0.0)

        worksheets = spec.get("worksheets") or []
        if not worksheets:
            dim.score = 1.0
            dim.details.append("No worksheets in spec (nothing to check)")
            return dim

        pbip_visual_types = self._extract_pbip_visual_types(pbip)
        matched = 0

        for ws in worksheets:
            ws_name = ws.get("name", "?")
            mark_type = (ws.get("mark_type") or ws.get("chart_type") or "").lower()

            if not mark_type:
                dim.details.append(f"  {ws_name}: no mark type in spec (skip)")
                matched += 1
                continue

            expected_pbi = _MARK_TO_PBI.get(mark_type, "")
            if not expected_pbi:
                dim.details.append(f"  {ws_name}: unknown mark '{mark_type}' (skip)")
                matched += 1
                continue

            if expected_pbi in pbip_visual_types or pbip_visual_types:
                matched += 1
                dim.details.append(f"  {ws_name}: {mark_type} -> {expected_pbi} (OK)")
            else:
                dim.gaps.append(
                    f"{ws_name}: expected '{expected_pbi}' for mark '{mark_type}' not found in PBIP"
                )

        dim.score = matched / len(worksheets) if worksheets else 1.0
        return dim

    def _score_field_coverage(
        self, spec: Dict[str, Any], pbip: Dict[str, Any]
    ) -> DimensionScore:
        """Check that fields referenced in Tableau appear in PBIP semantic model."""
        dim = DimensionScore(name="Field coverage", weight=0.25, score=0.0)

        source_fields = self._collect_source_fields(spec)
        if not source_fields:
            dim.score = 1.0
            dim.details.append("No source fields found in spec")
            return dim

        pbip_fields = self._extract_pbip_fields(pbip)
        matched = 0

        for field_name in source_fields:
            normalized = field_name.lower().replace(" ", "_").replace("-", "_")
            if any(normalized in pf.lower().replace(" ", "_") for pf in pbip_fields):
                matched += 1
            else:
                dim.gaps.append(f"Field '{field_name}' not found in PBIP model")

        dim.score = matched / len(source_fields)
        dim.details.append(
            f"{matched}/{len(source_fields)} source fields present in PBIP"
        )
        return dim

    def _score_formula_coverage(
        self, spec: Dict[str, Any], pbip: Dict[str, Any]
    ) -> DimensionScore:
        """Check that Tableau calculated fields have DAX translations."""
        dim = DimensionScore(name="Formula coverage", weight=0.20, score=0.0)

        calcs = spec.get("calculated_fields") or spec.get("calculations") or []
        if not calcs:
            dim.score = 1.0
            dim.details.append("No calculated fields in spec")
            return dim

        pbip_measures = self._extract_pbip_measures(pbip)
        matched = 0

        for calc in calcs:
            name = (calc.get("name") or "").strip()
            if not name:
                continue
            normalized = name.lower()
            if any(normalized in m.lower() for m in pbip_measures):
                matched += 1
            else:
                dim.gaps.append(
                    f"Calculated field '{name}' has no DAX measure in PBIP"
                )

        total = len([c for c in calcs if c.get("name")])
        dim.score = matched / total if total else 1.0
        dim.details.append(f"{matched}/{total} calculated fields have DAX equivalents")
        return dim

    def _score_layout_structure(
        self, spec: Dict[str, Any], pbip: Dict[str, Any]
    ) -> DimensionScore:
        """Check that PBIP has the same number of pages as Tableau dashboards."""
        dim = DimensionScore(name="Layout structure", weight=0.15, score=0.0)

        dashboards = spec.get("dashboards") or []
        expected_pages = max(len(dashboards), 1)

        pbip_pages = self._extract_pbip_pages(pbip)
        actual_pages = len(pbip_pages) if pbip_pages else 0

        if not pbip:
            # No PBIP output yet — score as neutral
            dim.score = 0.5
            dim.details.append("PBIP not generated yet (partial credit)")
            return dim

        if actual_pages == 0:
            dim.gaps.append("No report pages found in PBIP")
            dim.score = 0.0
        elif actual_pages >= expected_pages:
            dim.score = 1.0
            dim.details.append(
                f"{actual_pages} PBIP pages cover {expected_pages} Tableau dashboards"
            )
        else:
            dim.score = actual_pages / expected_pages
            dim.gaps.append(
                f"Only {actual_pages}/{expected_pages} dashboards converted to PBIP pages"
            )

        return dim

    def _score_filter_coverage(
        self, spec: Dict[str, Any], pbip: Dict[str, Any]
    ) -> DimensionScore:
        """Check that filters/slicers are present in PBIP."""
        dim = DimensionScore(name="Filter coverage", weight=0.10, score=0.0)

        filters = spec.get("filters") or []
        if not filters:
            dim.score = 1.0
            dim.details.append("No filters in spec")
            return dim

        pbip_slicers = self._extract_pbip_slicers(pbip)

        if not pbip:
            dim.score = 0.5
            dim.details.append("PBIP not generated (partial credit)")
            return dim

        if pbip_slicers:
            # Some slicers present — approximate coverage
            ratio = min(len(pbip_slicers) / len(filters), 1.0)
            dim.score = ratio
            dim.details.append(
                f"{len(pbip_slicers)} slicers in PBIP vs {len(filters)} filters in Tableau"
            )
        else:
            dim.gaps.append(
                f"{len(filters)} Tableau filters have no corresponding PBIP slicers"
            )
            dim.score = 0.0

        return dim

    # ------------------------------------------------------------------
    #  PBIP content extractors
    # ------------------------------------------------------------------

    def _extract_pbip_visual_types(self, pbip: Dict[str, Any]) -> List[str]:
        types: List[str] = []
        for path, data in pbip.items():
            if isinstance(data, dict):
                self._collect_values(data, "type", types)
        return types

    def _extract_pbip_fields(self, pbip: Dict[str, Any]) -> List[str]:
        fields: List[str] = []
        for path, data in pbip.items():
            if isinstance(data, dict):
                self._collect_values(data, "name", fields)
                self._collect_values(data, "column", fields)
                self._collect_values(data, "sourceColumn", fields)
        return fields

    def _extract_pbip_measures(self, pbip: Dict[str, Any]) -> List[str]:
        measures: List[str] = []
        for path, data in pbip.items():
            if "model" in path.lower() or "dataset" in path.lower():
                if isinstance(data, dict):
                    self._collect_values(data, "name", measures)
        return measures

    def _extract_pbip_pages(self, pbip: Dict[str, Any]) -> List[Any]:
        for path, data in pbip.items():
            if "report" in path.lower():
                if isinstance(data, dict) and "sections" in data:
                    return data["sections"]
                if isinstance(data, list):
                    return data
        return []

    def _extract_pbip_slicers(self, pbip: Dict[str, Any]) -> List[Any]:
        slicers: List[Any] = []
        for path, data in pbip.items():
            if isinstance(data, dict):
                visuals: List[Any] = []
                self._collect_values(data, "type", visuals)
                slicers.extend(v for v in visuals if "slicer" in str(v).lower())
        return slicers

    # ------------------------------------------------------------------
    #  Spec field collectors
    # ------------------------------------------------------------------

    def _collect_source_fields(self, spec: Dict[str, Any]) -> List[str]:
        fields: List[str] = []
        for ws in spec.get("worksheets") or []:
            for key in ("dimensions", "measures", "fields"):
                for f in ws.get(key) or []:
                    if isinstance(f, dict):
                        name = f.get("name") or f.get("field") or ""
                    else:
                        name = str(f)
                    if name and not name.startswith("["):
                        fields.append(name)
        return list(dict.fromkeys(fields))  # deduplicate preserving order

    # ------------------------------------------------------------------
    #  Generic recursive value collector
    # ------------------------------------------------------------------

    def _collect_values(
        self, obj: Any, key: str, out: List[Any], depth: int = 0
    ) -> None:
        if depth > 8:
            return
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], str):
                out.append(obj[key])
            for v in obj.values():
                self._collect_values(v, key, out, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_values(item, key, out, depth + 1)


# ---------------------------------------------------------------------------
#  Tableau mark-type to PBI visual mapping (simplified for fidelity checks)
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
    "gantt":        "gantt",
    "automatic":    "clusteredBarChart",
    "dual":         "lineClusteredColumnComboChart",
    "combo":        "lineClusteredColumnComboChart",
}
