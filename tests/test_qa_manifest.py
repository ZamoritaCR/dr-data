"""
Tests for core/qa_manifest.py -- QA audit manifest.
"""

import sys
import json
import tempfile
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.qa_manifest import (
    build_manifest,
    manifest_to_markdown,
    save_manifest,
    validate_manifest,
    QAManifest,
    SourceAsset,
    MeasureAudit,
    VisualMappingAudit,
    QAChecklistItem,
)
from core.requirements_contract import (
    RequirementsContract,
    VisualContract,
    VisualType,
    FilterContractSpec,
    LayoutConstraint,
    MappingAssumption,
    UnsupportedItem,
    ManualReviewItem,
    Confidence,
)
from core.visual_equivalency import (
    VisualEquivalency,
    EquivalencyStatus,
    FieldRole,
)


# ================================================================== #
#  Fixtures                                                             #
# ================================================================== #

def _build_context():
    return {
        "data_file": "superstore.csv",
        "row_count": 9994,
        "col_count": 21,
        "numeric_count": 7,
        "categorical_count": 14,
        "table_name": "Superstore",
        "page_names": ["Overview"],
        "visuals_per_page": [4],
        "measures_created": ["Total Sales", "Total Profit"],
        "measures_skipped": ["Complex Calc"],
        "fields_valid": 12,
        "fields_fixed": 2,
        "fields_removed": 1,
        "tableau_file": "US - Superstore.twbx",
        "tableau_worksheets": ["Sales Chart", "Profit Trend", "KPI Sales"],
        "tableau_calcs_found": ["Profit Ratio", "YoY Growth", "Complex Calc"],
        "tableau_calcs_converted": ["Profit Ratio", "YoY Growth"],
        "tableau_calcs_failed": ["Complex Calc"],
        "file_count": 35,
    }


def _dashboard_spec():
    return {
        "dashboard_title": "Sales Overview",
        "pages": [
            {
                "name": "Overview",
                "visuals": [
                    {"id": "v1", "type": "clusteredBarChart", "title": "Sales",
                     "data_roles": {"category": ["Category"], "values": ["Total Sales"]}},
                    {"id": "v2", "type": "lineChart", "title": "Trend",
                     "data_roles": {"category": ["Order Date"], "values": ["Total Sales"]}},
                ],
                "slicers": [{"field": "Region", "type": "dropdown"}],
            }
        ],
        "measures": [
            {"name": "Total Sales", "dax": "SUM(Superstore[Sales])"},
            {"name": "Total Profit", "dax": "SUM(Superstore[Profit])"},
            {"name": "Complex Calc", "dax": ""},
        ],
        "tableau_calcs": [
            {"name": "Profit Ratio", "formula": "SUM([Profit])/SUM([Sales])"},
        ],
    }


def _contract():
    return RequirementsContract(
        dashboard_title="Sales Overview",
        page_count=1,
        page_names=["Overview"],
        mapping_assumptions=[
            MappingAssumption(
                assumption="LOD expression converted to CALCULATE",
                reason="No direct DAX equivalent",
                confidence=Confidence.APPROXIMATE,
            ),
        ],
        unsupported_items=[
            UnsupportedItem(
                item="Table calculation: Running Total",
                tableau_feature="RUNNING_SUM",
                pbi_alternative="DAX window functions",
                severity="warning",
            ),
        ],
        manual_review_items=[
            ManualReviewItem(
                item="Ambiguous chart type for Worksheet 3",
                reason="Mark type was Automatic",
                severity="warning",
            ),
        ],
        layout_constraints=[
            LayoutConstraint(constraint_type="page_count_fixed", value="1", source="user_request"),
            LayoutConstraint(constraint_type="stacking_direction", value="vertical", source="user_request"),
        ],
    )


def _equivalencies():
    return [
        VisualEquivalency(
            source_name="Sales Chart",
            source_mark_type="bar",
            chosen_powerbi_visual="clusteredBarChart",
            equivalency_status=EquivalencyStatus.EXACT,
            rationale="Direct mapping.",
            required_field_roles=[
                FieldRole(role="Category", field="Category"),
                FieldRole(role="Values", field="Sales", aggregation="Sum"),
            ],
        ),
        VisualEquivalency(
            source_name="Geo View",
            source_mark_type="polygon",
            chosen_powerbi_visual="map",
            equivalency_status=EquivalencyStatus.CLOSE_EQUIVALENT,
            rationale="PBI uses Bing Maps.",
            analyst_review_required=True,
            review_reasons=["Geocoding verification needed"],
            required_field_roles=[
                FieldRole(role="Location", field="State"),
            ],
        ),
    ]


# ================================================================== #
#  Manifest building                                                    #
# ================================================================== #

class TestBuildManifest:

    def test_basic_build(self):
        m = build_manifest(
            contract=_contract(),
            equivalency_results=_equivalencies(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
            field_audit={"valid": 12, "fixed": 2, "removed": 1},
        )
        assert m.generation_id != ""
        assert m.timestamp != ""

    def test_source_assets_captured(self):
        m = build_manifest(build_context=_build_context())
        assert len(m.source_assets) == 2  # tableau + data
        types = {sa.asset_type for sa in m.source_assets}
        assert "tableau_workbook" in types
        assert "data_file" in types

    def test_synthetic_data_flagged(self):
        bc = _build_context()
        bc["data_file"] = "synthetic_tableau_data.csv"
        m = build_manifest(build_context=bc)
        asset = next(sa for sa in m.source_assets if "synthetic" in sa.asset_type)
        assert asset.asset_type == "synthetic_data"

    def test_fields_used_extracted(self):
        m = build_manifest(
            dashboard_spec=_dashboard_spec(),
            build_context=_build_context(),
        )
        field_names = {f.field_name for f in m.source_fields_used}
        assert "Category" in field_names
        assert "Total Sales" in field_names

    def test_field_usage_tracks_visuals(self):
        m = build_manifest(
            dashboard_spec=_dashboard_spec(),
            build_context=_build_context(),
        )
        sales = next(f for f in m.source_fields_used if f.field_name == "Total Sales")
        assert len(sales.used_in) >= 2  # used in v1 and v2

    def test_measures_captured(self):
        m = build_manifest(
            dashboard_spec=_dashboard_spec(),
            build_context=_build_context(),
        )
        assert len(m.generated_measures) == 3
        created = [mm for mm in m.generated_measures if mm.status == "created"]
        skipped = [mm for mm in m.generated_measures if mm.status == "skipped"]
        assert len(created) == 2
        assert len(skipped) == 1

    def test_contract_data_captured(self):
        m = build_manifest(contract=_contract())
        assert len(m.mapping_assumptions) == 1
        assert len(m.unsupported_features) == 1
        assert len(m.unresolved_ambiguities) == 1
        assert len(m.layout_constraints_applied) == 2

    def test_visual_mappings_captured(self):
        m = build_manifest(equivalency_results=_equivalencies())
        assert len(m.visual_mappings) == 2
        exact = [vm for vm in m.visual_mappings if vm.equivalency_status == "exact"]
        assert len(exact) == 1
        review = [vm for vm in m.visual_mappings if vm.analyst_review_required]
        assert len(review) == 1

    def test_approximations_tracked(self):
        m = build_manifest(equivalency_results=_equivalencies())
        assert len(m.approximations_used) == 1
        assert m.approximations_used[0]["visual"] == "Geo View"

    def test_field_audit_captured(self):
        m = build_manifest(field_audit={"valid": 10, "fixed": 3, "removed": 2})
        assert m.fields_valid == 10
        assert m.fields_auto_fixed == 3
        assert m.fields_removed == 2

    def test_contract_violations_captured(self):
        m = build_manifest(contract_violations=["Fixed page count from 2 to 1"])
        assert len(m.contract_violations_corrected) == 1

    def test_output_summary(self):
        m = build_manifest(
            dashboard_spec=_dashboard_spec(),
            build_context=_build_context(),
        )
        assert m.output_pages == 1
        assert m.output_visuals == 2
        assert m.output_measures == 3
        assert m.output_slicers == 1


# ================================================================== #
#  QA Checklist                                                         #
# ================================================================== #

class TestChecklist:

    def test_checklist_generated(self):
        m = build_manifest(
            contract=_contract(),
            equivalency_results=_equivalencies(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
            field_audit={"valid": 12, "fixed": 2, "removed": 1},
        )
        assert len(m.qa_checklist) > 0

    def test_skipped_measures_flagged(self):
        m = build_manifest(
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
        )
        fail_checks = [c for c in m.qa_checklist if c.auto_result == "fail"]
        assert any("skipped" in c.description.lower() for c in fail_checks)

    def test_removed_fields_flagged(self):
        m = build_manifest(field_audit={"valid": 10, "fixed": 0, "removed": 3})
        warn_checks = [c for c in m.qa_checklist if "removed" in c.description.lower()]
        assert len(warn_checks) >= 1

    def test_review_visuals_flagged(self):
        m = build_manifest(equivalency_results=_equivalencies())
        review_checks = [c for c in m.qa_checklist
                         if c.category == "visual" and "review" in c.description.lower()]
        assert len(review_checks) >= 1

    def test_unsupported_flagged(self):
        m = build_manifest(contract=_contract())
        unsup_checks = [c for c in m.qa_checklist if "unsupported" in c.description.lower()]
        assert len(unsup_checks) >= 1

    def test_mandatory_interaction_checks(self):
        m = build_manifest()
        interaction = [c for c in m.qa_checklist if c.category == "interaction"]
        assert len(interaction) >= 2

    def test_data_source_check_present(self):
        m = build_manifest()
        data_checks = [c for c in m.qa_checklist
                       if "m expression" in c.description.lower()
                       or "data source" in c.description.lower()]
        assert len(data_checks) >= 1

    def test_synthetic_data_must_verify(self):
        bc = _build_context()
        bc["data_file"] = "synthetic_tableau_data.csv"
        m = build_manifest(build_context=bc)
        synth_checks = [c for c in m.qa_checklist
                        if "synthetic" in c.description.lower()]
        assert len(synth_checks) >= 1
        assert synth_checks[0].severity == "must_verify"

    def test_check_ids_unique(self):
        m = build_manifest(
            contract=_contract(),
            equivalency_results=_equivalencies(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
            field_audit={"valid": 12, "fixed": 2, "removed": 1},
        )
        ids = [c.check_id for c in m.qa_checklist]
        assert len(ids) == len(set(ids))


# ================================================================== #
#  Markdown report                                                      #
# ================================================================== #

class TestMarkdown:

    def test_markdown_not_empty(self):
        m = build_manifest(
            contract=_contract(),
            equivalency_results=_equivalencies(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
        )
        md = manifest_to_markdown(m)
        assert len(md) > 500

    def test_markdown_has_sections(self):
        m = build_manifest(
            contract=_contract(),
            equivalency_results=_equivalencies(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
        )
        md = manifest_to_markdown(m)
        assert "## Summary" in md
        assert "## Source Assets" in md
        assert "## Visual Mapping Decisions" in md
        assert "## Generated DAX Measures" in md
        assert "## QA Checklist" in md

    def test_markdown_has_table(self):
        m = build_manifest(equivalency_results=_equivalencies())
        md = manifest_to_markdown(m)
        assert "| Source |" in md
        assert "clusteredBarChart" in md

    def test_markdown_has_dax_blocks(self):
        m = build_manifest(
            dashboard_spec=_dashboard_spec(),
            build_context=_build_context(),
        )
        md = manifest_to_markdown(m)
        assert "```dax" in md
        assert "SUM(Superstore[Sales])" in md

    def test_markdown_checklist_table(self):
        m = build_manifest(
            contract=_contract(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
            field_audit={"valid": 12, "fixed": 2, "removed": 1},
        )
        md = manifest_to_markdown(m)
        assert "QA-" in md
        assert "must_verify" in md


# ================================================================== #
#  Serialization                                                        #
# ================================================================== #

class TestSerialization:

    def test_to_json(self):
        m = build_manifest(build_context=_build_context())
        j = m.to_json()
        parsed = json.loads(j)
        assert "source_assets" in parsed
        assert "qa_checklist" in parsed

    def test_to_dict(self):
        m = build_manifest(build_context=_build_context())
        d = m.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["source_assets"], list)


# ================================================================== #
#  File saving                                                          #
# ================================================================== #

class TestFileSaving:

    def test_save_creates_files(self):
        m = build_manifest(
            contract=_contract(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_manifest(m, tmp)
            assert os.path.exists(paths["json"])
            assert os.path.exists(paths["markdown"])
            # JSON is valid
            with open(paths["json"]) as f:
                parsed = json.load(f)
            assert "qa_checklist" in parsed
            # Markdown is non-empty
            with open(paths["markdown"]) as f:
                md = f.read()
            assert len(md) > 100


# ================================================================== #
#  Completeness validation                                              #
# ================================================================== #

class TestValidation:

    def test_complete_manifest_passes(self):
        m = build_manifest(
            contract=_contract(),
            build_context=_build_context(),
            dashboard_spec=_dashboard_spec(),
        )
        issues = validate_manifest(m)
        assert issues == []

    def test_empty_manifest_fails(self):
        m = QAManifest()
        issues = validate_manifest(m)
        assert len(issues) >= 3  # missing id, timestamp, source_assets

    def test_no_output_fails(self):
        m = build_manifest()  # no build_context, no spec
        issues = validate_manifest(m)
        assert any("output" in i.lower() for i in issues)


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
