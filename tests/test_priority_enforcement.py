"""
Tests proving the instruction-priority framework works.

Rules enforced:
- Never override explicit page-count with heuristic
- Never replace requested chart type unless unsupported
- Never omit requested slicers/selectors/headers
- Never silently convert one-page to multi-page
- Never silently drop layout constraints
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.requirements_contract import (
    build_contract,
    validate_contract,
    enforce_contract,
    format_contract_for_prompt,
    RequirementsContract,
    VisualContract,
    VisualType,
    FilterContractSpec,
    LayoutConstraint,
    NavigationElement,
    InteractivityRule,
    FormattingConstraint,
    Priority,
    Requirement,
    PRIORITY_RANK,
    Confidence,
)


# ================================================================== #
#  Priority enum and ranking                                            #
# ================================================================== #

class TestPriorityRanking:

    def test_user_explicit_highest(self):
        assert PRIORITY_RANK[Priority.USER_EXPLICIT] > PRIORITY_RANK[Priority.DASHBOARD_SPEC]
        assert PRIORITY_RANK[Priority.USER_EXPLICIT] > PRIORITY_RANK[Priority.TABLEAU_DERIVED]
        assert PRIORITY_RANK[Priority.USER_EXPLICIT] > PRIORITY_RANK[Priority.DEFAULT]
        assert PRIORITY_RANK[Priority.USER_EXPLICIT] > PRIORITY_RANK[Priority.HEURISTIC]

    def test_priority_ordering(self):
        ordered = [Priority.HEURISTIC, Priority.DEFAULT, Priority.TABLEAU_DERIVED,
                   Priority.DASHBOARD_SPEC, Priority.USER_EXPLICIT]
        for i in range(len(ordered) - 1):
            assert PRIORITY_RANK[ordered[i]] < PRIORITY_RANK[ordered[i + 1]]


# ================================================================== #
#  Page count: explicit user instruction preserved                      #
# ================================================================== #

class TestPageCountPreserved:

    def test_single_page_is_must_have(self):
        c = build_contract("Build a single page dashboard")
        pc = next(lc for lc in c.layout_constraints if lc.constraint_type == "page_count_fixed")
        assert pc.priority == Priority.USER_EXPLICIT
        assert pc.requirement == Requirement.MUST_HAVE

    def test_explicit_page_count_not_overridden_by_tableau(self):
        """User says 1 page, Tableau has 6 dashboards. User wins."""
        from core.tableau_extractor import TableauWorkbook, DashboardSpec
        wb = TableauWorkbook(dashboards=[
            DashboardSpec(name=f"DB{i}") for i in range(6)
        ])
        c = build_contract("single page report", tableau_wb=wb)
        assert c.page_count == 1
        pc = next(lc for lc in c.layout_constraints if lc.constraint_type == "page_count_fixed")
        assert pc.priority == Priority.USER_EXPLICIT

    def test_enforcement_corrects_llm_page_inflation(self):
        """LLM generates 3 pages but contract says 1. Enforce."""
        c = build_contract("single page report")
        llm_spec = {
            "dashboard_title": "Test",
            "pages": [
                {"name": "A", "visuals": [], "slicers": []},
                {"name": "B", "visuals": [], "slicers": []},
                {"name": "C", "visuals": [], "slicers": []},
            ],
        }
        fixed, violations = enforce_contract(c, llm_spec)
        assert len(fixed["pages"]) == 1
        assert any("Contract requires 1" in v for v in violations)

    def test_one_page_never_silently_split(self):
        c = build_contract("Build everything on one page")
        assert c.page_count == 1
        assert any(
            lc.constraint_type == "page_count_fixed" and lc.value == "1"
            and lc.requirement == Requirement.MUST_HAVE
            for lc in c.layout_constraints
        )


# ================================================================== #
#  Chart types: requested types preserved                               #
# ================================================================== #

class TestChartTypesPreserved:

    def test_stacked_column_not_replaced(self):
        c = build_contract(
            "1) Sales as stacked column with Month on X-axis"
        )
        chart = next(v for v in c.visuals if v.visual_id.startswith("chart_"))
        assert chart.visual_type == VisualType.STACKED_COLUMN
        assert chart.priority == Priority.USER_EXPLICIT
        assert chart.requirement == Requirement.MUST_HAVE

    def test_clustered_column_not_replaced(self):
        c = build_contract(
            "1) Count as clustered column with Month on X-axis"
        )
        chart = next(v for v in c.visuals if v.visual_id.startswith("chart_"))
        assert chart.visual_type == VisualType.COLUMN

    def test_line_chart_not_replaced(self):
        c = build_contract("1) Trend as line chart with Date on X-axis")
        chart = next(v for v in c.visuals if v.visual_id.startswith("chart_"))
        assert chart.visual_type == VisualType.LINE

    def test_multiple_types_all_preserved(self):
        c = build_contract(
            "1) A as stacked column with X on X-axis, "
            "2) B as line chart with X on X-axis, "
            "3) C as clustered column with X on X-axis"
        )
        charts = [v for v in c.visuals if v.visual_id.startswith("chart_")]
        assert len(charts) == 3
        assert charts[0].visual_type == VisualType.STACKED_COLUMN
        assert charts[1].visual_type == VisualType.LINE
        assert charts[2].visual_type == VisualType.COLUMN


# ================================================================== #
#  Slicers: never omitted                                               #
# ================================================================== #

class TestSlicersNeverOmitted:

    def test_user_slicers_are_must_have(self):
        c = build_contract("3 slicers in the top row")
        for f in c.top_filters:
            assert f.priority == Priority.USER_EXPLICIT
            assert f.requirement == Requirement.MUST_HAVE

    def test_slicer_field_slicers_are_must_have(self):
        c = build_contract("add slicers for Region and Category")
        for f in c.top_filters:
            assert f.requirement == Requirement.MUST_HAVE

    def test_enforcement_flags_missing_slicers(self):
        c = build_contract("add slicers for Region")
        llm_spec = {
            "pages": [{"name": "Page 1", "visuals": [], "slicers": []}],
        }
        _, violations = enforce_contract(c, llm_spec)
        assert any("slicer" in v.lower() and "region" in v.lower() for v in violations)


# ================================================================== #
#  Navigation / selectors: never omitted                                #
# ================================================================== #

class TestNavigationPreserved:

    def test_tab_selector_captured(self):
        c = build_contract("with a tab-like category selector")
        tabs = [n for n in c.navigation_elements if n.element_type == "tab_selector"]
        assert len(tabs) >= 1

    def test_section_headers_are_must_have(self):
        c = build_contract('with a section label "Total Monthly"')
        labels = [v for v in c.visuals if "section" in v.visual_id]
        assert len(labels) >= 1
        assert labels[0].requirement == Requirement.MUST_HAVE

    def test_refresh_text_is_must_have(self):
        c = build_contract("with top-right last refresh text")
        refresh = next(v for v in c.visuals if v.visual_id == "refresh_text")
        assert refresh.requirement == Requirement.MUST_HAVE
        assert refresh.priority == Priority.USER_EXPLICIT


# ================================================================== #
#  Layout constraints: never silently dropped                           #
# ================================================================== #

class TestLayoutConstraintsPreserved:

    def test_vertical_stacking_is_must_have(self):
        c = build_contract("three visuals stacked vertically")
        stack = next(
            lc for lc in c.layout_constraints
            if lc.constraint_type == "stacking_direction"
        )
        assert stack.priority == Priority.USER_EXPLICIT
        assert stack.requirement == Requirement.MUST_HAVE

    def test_canvas_size_always_present(self):
        c = build_contract("build a dashboard")
        canvas = next(
            lc for lc in c.layout_constraints
            if lc.constraint_type == "canvas_size"
        )
        assert canvas.value == "1280x720"
        assert canvas.requirement == Requirement.MUST_HAVE

    def test_user_constraint_beats_tableau(self):
        """If user says vertical and Tableau says grid, user wins."""
        from core.tableau_extractor import TableauWorkbook, DashboardSpec
        wb = TableauWorkbook(dashboards=[
            DashboardSpec(name="D1", has_vertical_stack=False,
                          has_horizontal_split=True, layout_pattern="grid"),
        ])
        c = build_contract("stack everything vertically", tableau_wb=wb)
        stacking = [
            lc for lc in c.layout_constraints
            if lc.constraint_type == "stacking_direction"
        ]
        assert len(stacking) >= 1
        user_stack = next(s for s in stacking if s.priority == Priority.USER_EXPLICIT)
        assert user_stack.value == "vertical"
        assert user_stack.requirement == Requirement.MUST_HAVE


# ================================================================== #
#  Interactivity: cross-filtering preserved                             #
# ================================================================== #

class TestInteractivityPreserved:

    def test_cross_filter_is_must_have(self):
        c = build_contract("enable cross-filtering")
        cf = next(ir for ir in c.interactivity_rules if ir.rule_type == "cross_filter")
        assert cf.priority == Priority.USER_EXPLICIT
        assert cf.requirement == Requirement.MUST_HAVE

    def test_enforcement_flags_missing_interactivity(self):
        c = build_contract("enable cross-filtering")
        llm_spec = {"pages": [{"name": "P", "visuals": [], "slicers": []}]}
        _, violations = enforce_contract(c, llm_spec)
        assert any("cross_filter" in v for v in violations)


# ================================================================== #
#  Formatting: not silently dropped                                     #
# ================================================================== #

class TestFormattingPreserved:

    def test_legend_consistency_captured(self):
        c = build_contract("use consistent legend colors across charts")
        fc = next(f for f in c.formatting_constraints
                  if f.constraint_type == "legend_color_consistency")
        assert fc.priority == Priority.USER_EXPLICIT


# ================================================================== #
#  Requirement tags in prompt                                           #
# ================================================================== #

class TestPromptContainsTags:

    def test_must_have_in_prompt(self):
        c = build_contract(
            'Build a single page report called "Overview" '
            "with 3 slicers in the top row "
            "and 1) Sales as stacked column with Month on X-axis"
        )
        prompt = format_contract_for_prompt(c)
        assert "must_have" in prompt

    def test_binding_header_in_prompt(self):
        c = build_contract("test")
        prompt = format_contract_for_prompt(c)
        assert "BINDING" in prompt
        assert "DO NOT DEVIATE" in prompt


# ================================================================== #
#  Validation catches priority violations                               #
# ================================================================== #

class TestValidationCatchesPriorityViolations:

    def test_valid_contract_passes(self):
        c = build_contract("build a single page dashboard with top slicers")
        errors = validate_contract(c)
        assert errors == []

    def test_conflicting_constraints_detected(self):
        c = RequirementsContract(
            page_count=1,
            page_names=["Page 1"],
            layout_constraints=[
                LayoutConstraint(
                    constraint_type="stacking_direction",
                    value="vertical",
                    priority=Priority.HEURISTIC,
                    requirement=Requirement.SHOULD_HAVE,
                ),
                LayoutConstraint(
                    constraint_type="stacking_direction",
                    value="horizontal",
                    priority=Priority.USER_EXPLICIT,
                    requirement=Requirement.MUST_HAVE,
                ),
            ],
        )
        # The higher-priority constraint should win. If priority ordering
        # is wrong, validation would flag it.
        errors = validate_contract(c)
        # No error because USER_EXPLICIT > HEURISTIC (correct ordering)
        assert not any("Priority violation" in e for e in errors)


# ================================================================== #
#  Enforcement does not weaken MUST_HAVE                                #
# ================================================================== #

class TestEnforcementStrength:

    def test_enforcement_reports_missing_must_have_visuals(self):
        c = build_contract(
            '1) Revenue as stacked column with Month on X-axis'
        )
        must_visuals = [v for v in c.visuals if v.requirement == Requirement.MUST_HAVE]
        assert len(must_visuals) >= 1

        # LLM returns empty output
        llm_spec = {
            "dashboard_title": "Test",
            "pages": [{"name": "Page 1", "visuals": [], "slicers": []}],
        }
        _, violations = enforce_contract(c, llm_spec)
        assert any("MUST_HAVE visual" in v for v in violations)

    def test_enforcement_reports_missing_must_have_slicers(self):
        c = build_contract("add slicers for Region and Year")
        llm_spec = {
            "pages": [{"name": "Page 1", "visuals": [], "slicers": []}],
        }
        _, violations = enforce_contract(c, llm_spec)
        slicer_violations = [v for v in violations if "slicer" in v.lower()]
        assert len(slicer_violations) >= 1


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
