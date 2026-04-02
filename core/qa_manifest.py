"""
QA Manifest -- mandatory audit artifact for every Power BI generation run.

Produces a machine-readable JSON manifest and a human-readable Markdown report
with enough detail for an analyst to verify the generated output.

Collects from:
- RequirementsContract (intent, constraints, assumptions, ambiguities)
- VisualEquivalency results (mapping decisions, rationale, review flags)
- Build context (field audit, measures, pages, files)
- Dashboard spec (DAX, pages, visuals)
- Layout assembly (positions, stacking)
"""

import json
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# ================================================================== #
#  Manifest schema                                                      #
# ================================================================== #

@dataclass
class SourceAsset:
    """A source file or structure used in generation."""
    asset_type: str         # "tableau_workbook", "data_file", "synthetic_data"
    filename: str
    details: Dict = field(default_factory=dict)
    # details might include: worksheet_count, dashboard_count, row_count, col_count


@dataclass
class FieldUsage:
    """How a source field was used in the output."""
    field_name: str
    source: str = ""        # datasource or table name
    role: str = ""          # "dimension", "measure", "filter", "slicer", "unused"
    used_in: List[str] = field(default_factory=list)  # visual IDs
    notes: str = ""


@dataclass
class MeasureAudit:
    """Audit record for a generated DAX measure."""
    name: str
    dax: str = ""
    source_formula: str = ""    # original Tableau formula if migration
    status: str = ""            # "created", "skipped", "failed_validation"
    reason: str = ""


@dataclass
class VisualMappingAudit:
    """Audit record for a visual mapping decision."""
    visual_id: str
    source_worksheet: str = ""
    source_mark_type: str = ""
    chosen_pbi_visual: str = ""
    equivalency_status: str = ""    # exact, close_equivalent, approximation_required
    rationale: str = ""
    analyst_review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    field_roles: List[Dict] = field(default_factory=list)


@dataclass
class QAChecklistItem:
    """A specific thing an analyst should verify."""
    check_id: str
    category: str           # "visual", "data", "dax", "layout", "interaction", "slicer"
    description: str
    severity: str = "must_verify"   # must_verify, should_verify, informational
    auto_result: str = ""   # "pass", "warning", "fail", "" (not checked)
    notes: str = ""


@dataclass
class QAManifest:
    """Complete audit manifest for a generation run."""

    # Identity
    generation_id: str = ""
    timestamp: str = ""
    generator_version: str = "1.0"

    # Source assets
    source_assets: List[SourceAsset] = field(default_factory=list)

    # Fields
    source_fields_used: List[FieldUsage] = field(default_factory=list)

    # Contract
    mapping_assumptions: List[Dict] = field(default_factory=list)
    unsupported_features: List[Dict] = field(default_factory=list)
    approximations_used: List[Dict] = field(default_factory=list)
    layout_constraints_applied: List[Dict] = field(default_factory=list)
    unresolved_ambiguities: List[Dict] = field(default_factory=list)

    # Generated output
    generated_measures: List[MeasureAudit] = field(default_factory=list)
    visual_mappings: List[VisualMappingAudit] = field(default_factory=list)

    # Field audit
    fields_valid: int = 0
    fields_auto_fixed: int = 0
    fields_removed: int = 0

    # Output summary
    output_pages: int = 0
    output_visuals: int = 0
    output_measures: int = 0
    output_slicers: int = 0
    output_file_path: str = ""
    output_file_count: int = 0

    # QA checklist
    qa_checklist: List[QAChecklistItem] = field(default_factory=list)

    # Contract enforcement
    contract_violations_corrected: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ================================================================== #
#  Builder                                                              #
# ================================================================== #

def build_manifest(
    contract=None,
    equivalency_results: list = None,
    build_context: dict = None,
    dashboard_spec: dict = None,
    field_audit: dict = None,
    contract_violations: list = None,
    output_path: str = "",
) -> QAManifest:
    """Build a complete QA manifest from all pipeline outputs.

    All parameters are optional -- the manifest captures whatever is available.
    """
    manifest = QAManifest(
        generation_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    bc = build_context or {}
    spec = dashboard_spec or {}

    # -- Source assets --
    if bc.get("tableau_file"):
        manifest.source_assets.append(SourceAsset(
            asset_type="tableau_workbook",
            filename=bc["tableau_file"],
            details={
                "worksheets": bc.get("tableau_worksheets", []),
                "calculated_fields_found": len(bc.get("tableau_calcs_found", [])),
                "calculated_fields_converted": len(bc.get("tableau_calcs_converted", [])),
                "calculated_fields_failed": len(bc.get("tableau_calcs_failed", [])),
            },
        ))

    if bc.get("data_file"):
        asset_type = "synthetic_data" if "synthetic" in bc["data_file"].lower() else "data_file"
        manifest.source_assets.append(SourceAsset(
            asset_type=asset_type,
            filename=bc["data_file"],
            details={
                "row_count": bc.get("row_count", 0),
                "col_count": bc.get("col_count", 0),
                "numeric_columns": bc.get("numeric_count", 0),
                "categorical_columns": bc.get("categorical_count", 0),
                "table_name": bc.get("table_name", ""),
            },
        ))

    # -- Fields used --
    if spec.get("pages"):
        fields_seen = {}
        for page in spec["pages"]:
            for visual in page.get("visuals", []):
                vid = visual.get("id", "?")
                for role_name, fields in visual.get("data_roles", {}).items():
                    if isinstance(fields, list):
                        for f in fields:
                            if f not in fields_seen:
                                fields_seen[f] = FieldUsage(
                                    field_name=f,
                                    source=bc.get("table_name", ""),
                                    role=role_name,
                                )
                            fields_seen[f].used_in.append(vid)
        manifest.source_fields_used = list(fields_seen.values())

    # -- Contract data --
    if contract:
        manifest.mapping_assumptions = [
            {"assumption": a.assumption, "reason": a.reason,
             "confidence": a.confidence.value if hasattr(a.confidence, "value") else str(a.confidence)}
            for a in getattr(contract, "mapping_assumptions", [])
        ]
        manifest.unsupported_features = [
            {"item": u.item, "tableau_feature": u.tableau_feature,
             "pbi_alternative": u.pbi_alternative, "severity": u.severity}
            for u in getattr(contract, "unsupported_items", [])
        ]
        manifest.unresolved_ambiguities = [
            {"item": m.item, "reason": m.reason, "severity": m.severity}
            for m in getattr(contract, "manual_review_items", [])
        ]
        manifest.layout_constraints_applied = [
            {"type": lc.constraint_type, "value": lc.value, "source": lc.source}
            for lc in getattr(contract, "layout_constraints", [])
        ]

    # -- Visual mappings --
    if equivalency_results:
        for eq in equivalency_results:
            manifest.visual_mappings.append(VisualMappingAudit(
                visual_id=getattr(eq, "source_name", ""),
                source_worksheet=getattr(eq, "source_name", ""),
                source_mark_type=getattr(eq, "source_mark_type", ""),
                chosen_pbi_visual=getattr(eq, "chosen_powerbi_visual", ""),
                equivalency_status=getattr(eq, "equivalency_status", "").value
                    if hasattr(getattr(eq, "equivalency_status", ""), "value")
                    else str(getattr(eq, "equivalency_status", "")),
                rationale=getattr(eq, "rationale", ""),
                analyst_review_required=getattr(eq, "analyst_review_required", False),
                review_reasons=getattr(eq, "review_reasons", []),
                field_roles=[
                    {"role": r.role, "field": r.field, "aggregation": r.aggregation}
                    for r in getattr(eq, "required_field_roles", [])
                ],
            ))
            # Track approximations
            status = getattr(eq, "equivalency_status", None)
            if status and hasattr(status, "value") and status.value in ("approximation_required", "close_equivalent"):
                manifest.approximations_used.append({
                    "visual": getattr(eq, "source_name", ""),
                    "source_type": getattr(eq, "source_mark_type", ""),
                    "pbi_type": getattr(eq, "chosen_powerbi_visual", ""),
                    "rationale": getattr(eq, "rationale", ""),
                })

    # -- Measures --
    if spec.get("measures"):
        created = set(bc.get("measures_created", []))
        skipped = set(bc.get("measures_skipped", []))
        tableau_calcs = {
            cf.get("name", ""): cf.get("formula", "")
            for cf in (spec.get("tableau_calcs", []) or [])
            if isinstance(cf, dict)
        }

        for m in spec["measures"]:
            name = m.get("name", "?")
            status = "created" if name in created else (
                "skipped" if name in skipped else "requested"
            )
            manifest.generated_measures.append(MeasureAudit(
                name=name,
                dax=m.get("dax", m.get("expression", "")),
                source_formula=tableau_calcs.get(name, ""),
                status=status,
            ))

    # -- Field audit --
    fa = field_audit or {}
    manifest.fields_valid = fa.get("valid", 0)
    manifest.fields_auto_fixed = fa.get("fixed", 0)
    manifest.fields_removed = fa.get("removed", 0)

    # -- Output summary --
    manifest.output_pages = len(spec.get("pages", []))
    manifest.output_visuals = sum(
        len(p.get("visuals", [])) for p in spec.get("pages", [])
    )
    manifest.output_measures = len(spec.get("measures", []))
    manifest.output_slicers = sum(
        len(p.get("slicers", [])) for p in spec.get("pages", [])
    )
    manifest.output_file_path = output_path
    manifest.output_file_count = bc.get("file_count", 0) if bc else 0

    # -- Contract enforcement --
    manifest.contract_violations_corrected = contract_violations or []

    # -- QA checklist --
    manifest.qa_checklist = _generate_checklist(manifest)

    return manifest


# ================================================================== #
#  Checklist generator                                                  #
# ================================================================== #

def _generate_checklist(manifest: QAManifest) -> List[QAChecklistItem]:
    """Generate a QA checklist from the manifest data."""
    checks = []
    check_idx = [0]

    def _add(category, description, severity="must_verify", auto_result="", notes=""):
        check_idx[0] += 1
        checks.append(QAChecklistItem(
            check_id=f"QA-{check_idx[0]:03d}",
            category=category,
            description=description,
            severity=severity,
            auto_result=auto_result,
            notes=notes,
        ))

    # Data checks
    for asset in manifest.source_assets:
        if asset.asset_type == "synthetic_data":
            _add("data",
                 f"Synthetic data was used ({asset.filename}). Replace with real "
                 f"data before production use.",
                 severity="must_verify", auto_result="warning")
        elif asset.asset_type == "data_file":
            _add("data",
                 f"Verify data file '{asset.filename}' is the correct source "
                 f"({asset.details.get('row_count', '?')} rows).",
                 severity="should_verify", auto_result="pass")

    # Field audit checks
    if manifest.fields_removed > 0:
        _add("data",
             f"{manifest.fields_removed} field reference(s) were removed because "
             f"they did not match any column in the dataset. Verify no critical "
             f"fields were lost.",
             severity="must_verify", auto_result="warning")

    if manifest.fields_auto_fixed > 0:
        _add("data",
             f"{manifest.fields_auto_fixed} field reference(s) were auto-corrected "
             f"(fuzzy match). Verify the corrections are semantically correct.",
             severity="should_verify", auto_result="warning")

    # Measure checks
    skipped = [m for m in manifest.generated_measures if m.status == "skipped"]
    if skipped:
        names = ", ".join(m.name for m in skipped[:5])
        _add("dax",
             f"{len(skipped)} DAX measure(s) failed validation and were skipped: "
             f"{names}{'...' if len(skipped) > 5 else ''}. "
             f"These may need manual creation in Power BI.",
             severity="must_verify", auto_result="fail")

    for m in manifest.generated_measures:
        if m.status == "created" and m.dax:
            _add("dax",
                 f"Verify DAX measure '{m.name}': {m.dax[:80]}{'...' if len(m.dax) > 80 else ''}",
                 severity="should_verify", auto_result="pass",
                 notes=f"Source: {m.source_formula[:60]}" if m.source_formula else "")

    # Visual mapping checks
    for vm in manifest.visual_mappings:
        if vm.analyst_review_required:
            _add("visual",
                 f"Visual '{vm.visual_id}' requires analyst review: "
                 f"{vm.source_mark_type} -> {vm.chosen_pbi_visual} "
                 f"({vm.equivalency_status})",
                 severity="must_verify", auto_result="warning",
                 notes="; ".join(vm.review_reasons))

    # Approximation checks
    for approx in manifest.approximations_used:
        _add("visual",
             f"Approximation: '{approx['visual']}' uses {approx['pbi_type']} "
             f"instead of exact equivalent for Tableau {approx['source_type']}.",
             severity="should_verify", auto_result="warning",
             notes=approx.get("rationale", "")[:100])

    # Unsupported feature checks
    for uf in manifest.unsupported_features:
        _add("visual",
             f"Unsupported: {uf['item']}. PBI alternative: {uf.get('pbi_alternative', 'none')}.",
             severity="must_verify", auto_result="fail")

    # Ambiguity checks
    for amb in manifest.unresolved_ambiguities:
        _add("layout",
             f"Ambiguity: {amb['item']} -- {amb['reason']}",
             severity="must_verify" if amb.get("severity") == "warning" else "informational",
             auto_result="warning")

    # Layout checks
    _add("layout",
         f"Verify page count: {manifest.output_pages} page(s) generated.",
         severity="should_verify", auto_result="pass")

    _add("layout",
         f"Verify visual count: {manifest.output_visuals} visual(s) generated.",
         severity="should_verify", auto_result="pass")

    # Contract violation checks
    for cv in manifest.contract_violations_corrected:
        _add("layout",
             f"Contract violation corrected: {cv}",
             severity="informational", auto_result="warning")

    # General checks
    _add("interaction",
         "Open the .pbip in Power BI Desktop and verify all slicers filter correctly.",
         severity="must_verify")

    _add("interaction",
         "Click through each page and verify visuals render data.",
         severity="must_verify")

    _add("data",
         "Verify the M expression in the TMDL model points to the correct data source path.",
         severity="must_verify")

    return checks


# ================================================================== #
#  Markdown report                                                      #
# ================================================================== #

def manifest_to_markdown(manifest: QAManifest) -> str:
    """Render the manifest as a human-readable Markdown report."""
    lines = []

    lines.append(f"# QA Audit Report")
    lines.append(f"")
    lines.append(f"**Generated:** {manifest.timestamp}")
    lines.append(f"**Run ID:** {manifest.generation_id}")
    lines.append(f"**Generator:** v{manifest.generator_version}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Pages | {manifest.output_pages} |")
    lines.append(f"| Visuals | {manifest.output_visuals} |")
    lines.append(f"| DAX Measures | {manifest.output_measures} |")
    lines.append(f"| Slicers | {manifest.output_slicers} |")
    lines.append(f"| Fields valid | {manifest.fields_valid} |")
    lines.append(f"| Fields auto-fixed | {manifest.fields_auto_fixed} |")
    lines.append(f"| Fields removed | {manifest.fields_removed} |")
    lines.append(f"| Files generated | {manifest.output_file_count} |")
    lines.append("")

    # Source assets
    if manifest.source_assets:
        lines.append("## Source Assets")
        lines.append("")
        for sa in manifest.source_assets:
            lines.append(f"### {sa.asset_type}: {sa.filename}")
            for k, v in sa.details.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

    # Visual mappings
    if manifest.visual_mappings:
        lines.append("## Visual Mapping Decisions")
        lines.append("")
        lines.append("| Source | Tableau Type | PBI Visual | Status | Review? |")
        lines.append("|--------|-------------|------------|--------|---------|")
        for vm in manifest.visual_mappings:
            review = "YES" if vm.analyst_review_required else "no"
            lines.append(
                f"| {vm.source_worksheet} | {vm.source_mark_type} | "
                f"{vm.chosen_pbi_visual} | {vm.equivalency_status} | {review} |"
            )
        lines.append("")

        # Detail for review-required visuals
        review_needed = [vm for vm in manifest.visual_mappings if vm.analyst_review_required]
        if review_needed:
            lines.append("### Visuals Requiring Review")
            lines.append("")
            for vm in review_needed:
                lines.append(f"**{vm.visual_id}** ({vm.source_mark_type} -> {vm.chosen_pbi_visual})")
                lines.append(f"- Status: {vm.equivalency_status}")
                lines.append(f"- Rationale: {vm.rationale}")
                for r in vm.review_reasons:
                    lines.append(f"- Review: {r}")
                lines.append("")

    # DAX Measures
    if manifest.generated_measures:
        lines.append("## Generated DAX Measures")
        lines.append("")
        for m in manifest.generated_measures:
            status_icon = {"created": "[OK]", "skipped": "[SKIP]", "failed_validation": "[FAIL]"}.get(
                m.status, "[?]"
            )
            lines.append(f"### {status_icon} {m.name}")
            if m.dax:
                lines.append(f"```dax")
                lines.append(m.dax)
                lines.append(f"```")
            if m.source_formula:
                lines.append(f"Source (Tableau): `{m.source_formula}`")
            if m.status == "skipped":
                lines.append(f"**Skipped:** {m.reason}")
            lines.append("")

    # Mapping assumptions
    if manifest.mapping_assumptions:
        lines.append("## Mapping Assumptions")
        lines.append("")
        for a in manifest.mapping_assumptions:
            lines.append(f"- **{a['assumption']}** ({a.get('confidence', '?')})")
            if a.get("reason"):
                lines.append(f"  Reason: {a['reason']}")
        lines.append("")

    # Unsupported features
    if manifest.unsupported_features:
        lines.append("## Unsupported Features")
        lines.append("")
        for uf in manifest.unsupported_features:
            lines.append(f"- **{uf['item']}**")
            if uf.get("pbi_alternative"):
                lines.append(f"  PBI alternative: {uf['pbi_alternative']}")
        lines.append("")

    # Approximations
    if manifest.approximations_used:
        lines.append("## Approximations Used")
        lines.append("")
        for approx in manifest.approximations_used:
            lines.append(
                f"- **{approx['visual']}**: Tableau {approx['source_type']} "
                f"-> PBI {approx['pbi_type']}"
            )
            if approx.get("rationale"):
                lines.append(f"  {approx['rationale'][:120]}")
        lines.append("")

    # Layout constraints
    if manifest.layout_constraints_applied:
        lines.append("## Layout Constraints Applied")
        lines.append("")
        for lc in manifest.layout_constraints_applied:
            lines.append(f"- {lc['type']}: {lc['value']} (source: {lc.get('source', '?')})")
        lines.append("")

    # Ambiguities
    if manifest.unresolved_ambiguities:
        lines.append("## Unresolved Ambiguities")
        lines.append("")
        for amb in manifest.unresolved_ambiguities:
            lines.append(f"- **{amb['item']}**: {amb['reason']}")
        lines.append("")

    # Contract violations
    if manifest.contract_violations_corrected:
        lines.append("## Contract Violations Corrected")
        lines.append("")
        for cv in manifest.contract_violations_corrected:
            lines.append(f"- {cv}")
        lines.append("")

    # QA Checklist
    lines.append("## QA Checklist")
    lines.append("")
    lines.append("| # | Category | Check | Severity | Auto | Notes |")
    lines.append("|---|----------|-------|----------|------|-------|")

    for item in manifest.qa_checklist:
        auto = {"pass": "PASS", "warning": "WARN", "fail": "FAIL"}.get(
            item.auto_result, "--"
        )
        notes = item.notes[:60] + "..." if len(item.notes) > 60 else item.notes
        lines.append(
            f"| {item.check_id} | {item.category} | {item.description[:80]} | "
            f"{item.severity} | {auto} | {notes} |"
        )
    lines.append("")

    # Counts
    must = sum(1 for c in manifest.qa_checklist if c.severity == "must_verify")
    should = sum(1 for c in manifest.qa_checklist if c.severity == "should_verify")
    fails = sum(1 for c in manifest.qa_checklist if c.auto_result == "fail")
    warns = sum(1 for c in manifest.qa_checklist if c.auto_result == "warning")

    lines.append(f"**Checklist:** {must} must-verify, {should} should-verify, "
                 f"{fails} auto-fail, {warns} auto-warning")
    lines.append("")

    return "\n".join(lines)


# ================================================================== #
#  File writers                                                         #
# ================================================================== #

def save_manifest(manifest: QAManifest, output_dir: str) -> dict:
    """Save manifest as both JSON and Markdown.

    Returns dict with paths: {"json": path, "markdown": path}
    """
    os.makedirs(output_dir, exist_ok=True)
    run_id = manifest.generation_id

    json_path = os.path.join(output_dir, f"qa_manifest_{run_id}.json")
    md_path = os.path.join(output_dir, f"qa_report_{run_id}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(manifest_to_markdown(manifest))

    return {"json": json_path, "markdown": md_path}


# ================================================================== #
#  Completeness validator                                               #
# ================================================================== #

def validate_manifest(manifest: QAManifest) -> List[str]:
    """Check the manifest has minimum required content.

    Returns list of issues. Empty = complete.
    """
    issues = []

    if not manifest.generation_id:
        issues.append("Missing generation_id")
    if not manifest.timestamp:
        issues.append("Missing timestamp")
    if not manifest.source_assets:
        issues.append("No source assets recorded")
    if manifest.output_pages == 0 and manifest.output_visuals == 0:
        issues.append("No output pages or visuals recorded")
    if not manifest.qa_checklist:
        issues.append("QA checklist is empty")

    return issues
