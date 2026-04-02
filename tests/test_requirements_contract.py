"""
Tests for core/requirements_contract.py -- Requirements Contract stage.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.requirements_contract import (
    RequirementsContract,
    VisualContract,
    VisualType,
    Confidence,
    StackDirection,
    FilterContractSpec,
    NavigationElement,
    LayoutConstraint,
    ManualReviewItem,
    MappingAssumption,
    AxisSpec,
    build_contract,
    validate_contract,
    enforce_contract,
    format_contract_for_prompt,
    _parse_page_count,
    _parse_page_names,
    _parse_stacking,
    _parse_visual_mentions,
    _parse_filter_mentions,
    _parse_tab_mentions,
)


# ================================================================== #
#  User text parsing                                                    #
# ================================================================== #

class TestPageCountParsing:

    def test_single_page(self):
        assert _parse_page_count("build a single page dashboard") == 1

    def test_numeric(self):
        assert _parse_page_count("I want 3 pages") == 3

    def test_word(self):
        assert _parse_page_count("create a two-page report") == 2

    def test_one_page(self):
        assert _parse_page_count("one page report") == 1

    def test_no_mention(self):
        assert _parse_page_count("build me a dashboard") is None

    def test_five_pages(self):
        assert _parse_page_count("five pages please") == 5


class TestPageNameParsing:

    def test_quoted_called(self):
        names = _parse_page_names('page called "Overview"')
        assert names == ["Overview"]

    def test_quoted_named(self):
        names = _parse_page_names("page named 'Sales Dashboard'")
        assert names == ["Sales Dashboard"]

    def test_no_names(self):
        assert _parse_page_names("build a dashboard") == []


class TestStackingParsing:

    def test_vertical(self):
        assert _parse_stacking("three visuals stacked vertically") == StackDirection.VERTICAL

    def test_horizontal(self):
        assert _parse_stacking("charts side by side") == StackDirection.HORIZONTAL

    def test_none(self):
        assert _parse_stacking("build a dashboard") is None


class TestVisualMentions:

    def test_bar_chart(self):
        visuals = _parse_visual_mentions("add a bar chart showing sales")
        assert any(v["type"] == VisualType.BAR for v in visuals)

    def test_multiple_types(self):
        visuals = _parse_visual_mentions("I want a line chart and a pie chart")
        types = {v["type"] for v in visuals}
        assert VisualType.LINE in types
        assert VisualType.PIE in types

    def test_count_of_visuals(self):
        visuals = _parse_visual_mentions("three monthly visuals stacked vertically")
        assert len(visuals) >= 3

    def test_kpi(self):
        visuals = _parse_visual_mentions("add a KPI card for revenue")
        assert any(v["type"] == VisualType.CARD for v in visuals)


class TestFilterMentions:

    def test_slicer_for_field(self):
        filters = _parse_filter_mentions("add slicers for Region and Category")
        fields = [f["field"].lower() for f in filters]
        assert "region" in fields

    def test_top_slicers(self):
        filters = _parse_filter_mentions("with top slicers")
        assert len(filters) >= 1
        assert filters[0]["position"] == "top"

    def test_filter_by(self):
        filters = _parse_filter_mentions("filter by Date")
        assert any(f["field"].lower() == "date" for f in filters)


class TestTabMentions:

    def test_tab_like(self):
        assert _parse_tab_mentions("with a tab-like category selector") is True

    def test_tab_selector(self):
        assert _parse_tab_mentions("add a tab selector at the top") is True

    def test_no_tabs(self):
        assert _parse_tab_mentions("build a dashboard") is False


# ================================================================== #
#  Contract building (no Tableau)                                       #
# ================================================================== #

class TestBuildContractFreshDesign:

    def test_basic_contract(self):
        c = build_contract("build me a dashboard")
        assert c.page_count == 1
        assert len(c.page_names) == 1
        assert c.source_type == "fresh_design"
        assert c.contract_hash  # non-empty

    def test_single_page_enforced(self):
        c = build_contract("single page report called 'Overview'")
        assert c.page_count == 1
        assert c.page_names == ["Overview"]
        assert any(lc.constraint_type == "page_count_fixed" for lc in c.layout_constraints)

    def test_multi_page(self):
        c = build_contract("create a 3 page report")
        assert c.page_count == 3
        assert len(c.page_names) == 3

    def test_stacking_captured(self):
        c = build_contract("three visuals stacked vertically")
        assert any(
            lc.constraint_type == "stacking_direction" and lc.value == "vertical"
            for lc in c.layout_constraints
        )

    def test_filters_captured(self):
        c = build_contract("add slicers for Region")
        assert any(f.field.lower() == "region" for f in c.top_filters)

    def test_tabs_captured(self):
        c = build_contract("with a tab-like selector")
        assert len(c.navigation_elements) >= 1
        assert c.navigation_elements[0].element_type == "tab_selector"

    def test_title_from_user(self):
        c = build_contract('dashboard titled "My Report"')
        assert c.dashboard_title == "My Report"

    def test_ambiguity_no_visuals(self):
        c = build_contract("build something")
        assert any(
            "No visuals specified" in mr.item
            for mr in c.manual_review_items
        )

    def test_deterministic(self):
        """Same input must produce same output."""
        c1 = build_contract("build a 2 page dashboard with slicers for Region")
        c2 = build_contract("build a 2 page dashboard with slicers for Region")
        assert c1.contract_hash == c2.contract_hash
        assert c1.page_count == c2.page_count
        assert c1.page_names == c2.page_names
        assert len(c1.visuals) == len(c2.visuals)
        assert len(c1.top_filters) == len(c2.top_filters)


# ================================================================== #
#  Contract building (with Tableau)                                     #
# ================================================================== #

class TestBuildContractWithTableau:

    def _make_tableau_wb(self):
        """Minimal TableauWorkbook-like object for testing."""
        from core.tableau_extractor import (
            TableauWorkbook, VisualSpec, DashboardSpec, ZoneSpec,
            EncodingSpec, ColumnRef, CalculatedFieldSpec,
            Confidence as TConf,
        )

        ws1 = VisualSpec(
            name="Sales Chart",
            mark_type="bar",
            mark_type_pbi="clusteredBarChart",
            confidence=TConf.EXACT,
            encoding=EncodingSpec(
                shelf_rows=[ColumnRef(name="Category", role="dimension")],
                shelf_cols=[ColumnRef(name="Sales", role="measure", aggregation="Sum")],
                color=ColumnRef(name="Profit", role="measure"),
            ),
            legend_title="Profit",
        )
        ws2 = VisualSpec(
            name="Trend Line",
            mark_type="line",
            mark_type_pbi="lineChart",
            confidence=TConf.EXACT,
            encoding=EncodingSpec(
                shelf_rows=[ColumnRef(name="Sales", role="measure", aggregation="Sum")],
                shelf_cols=[ColumnRef(name="Order Date", role="dimension", aggregation="Month")],
            ),
        )
        ws3 = VisualSpec(
            name="KPI Sales",
            mark_type="text",
            mark_type_pbi="card",
            confidence=TConf.HIGH,
            is_kpi_card=True,
        )

        wz1 = ZoneSpec(name="Sales Chart", zone_type="ws", pbi_x=0, pbi_y=100, pbi_w=640, pbi_h=400)
        wz2 = ZoneSpec(name="Trend Line", zone_type="ws", pbi_x=640, pbi_y=100, pbi_w=640, pbi_h=400)
        wz3 = ZoneSpec(name="KPI Sales", zone_type="ws", pbi_x=0, pbi_y=0, pbi_w=1280, pbi_h=80)
        fz = ZoneSpec(name="Sales Chart", zone_type="filter", param="[none:Region:nk]", y=5000,
                      filter_mode="checkdropdown")

        db = DashboardSpec(
            name="Overview",
            width=1200,
            height=800,
            worksheet_zones=[wz1, wz2, wz3],
            filter_zones=[fz],
            has_vertical_stack=True,
            layout_pattern="grid",
        )

        cf = CalculatedFieldSpec(
            name="Running Total",
            formula="RUNNING_SUM(SUM([Sales]))",
            is_table_calc=True,
        )

        wb = TableauWorkbook(
            worksheets=[ws1, ws2, ws3],
            dashboards=[db],
            calculated_fields=[cf],
        )
        return wb

    def test_tableau_visuals_extracted(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert to power bi", tableau_wb=wb)
        assert len(c.visuals) == 3
        assert c.visuals[0].source_worksheet == "Sales Chart"

    def test_tableau_filters_extracted(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert to power bi", tableau_wb=wb)
        assert len(c.top_filters) >= 1
        assert any("Region" in f.field for f in c.top_filters)

    def test_page_name_from_dashboard(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert to power bi", tableau_wb=wb)
        assert c.page_names[0] == "Overview"

    def test_user_page_overrides_tableau(self):
        wb = self._make_tableau_wb()
        c = build_contract("single page called 'Dash Overall Transactions'", tableau_wb=wb)
        assert c.page_count == 1
        assert c.page_names == ["Dash Overall Transactions"]
        # All visuals should be on that page
        for v in c.visuals:
            assert v.page == "Dash Overall Transactions"

    def test_table_calc_unsupported(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert", tableau_wb=wb)
        assert any("Running Total" in u.item for u in c.unsupported_items)

    def test_axes_captured(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert", tableau_wb=wb)
        sales_chart = next(v for v in c.visuals if v.source_worksheet == "Sales Chart")
        assert sales_chart.category_axis is not None
        assert sales_chart.category_axis.field == "Category"
        assert sales_chart.value_axis is not None
        assert sales_chart.value_axis.field == "Sales"

    def test_legend_captured(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert", tableau_wb=wb)
        sales_chart = next(v for v in c.visuals if v.source_worksheet == "Sales Chart")
        assert sales_chart.legend is not None
        assert sales_chart.legend.field == "Profit"

    def test_kpi_detected(self):
        wb = self._make_tableau_wb()
        c = build_contract("convert", tableau_wb=wb)
        kpi = next(v for v in c.visuals if v.source_worksheet == "KPI Sales")
        assert kpi.visual_type == VisualType.CARD


# ================================================================== #
#  Validation                                                           #
# ================================================================== #

class TestValidation:

    def test_valid_contract(self):
        c = build_contract("single page dashboard")
        errors = validate_contract(c)
        assert errors == []

    def test_page_count_mismatch(self):
        c = RequirementsContract(page_count=2, page_names=["Page 1"])
        errors = validate_contract(c)
        assert any("page_count" in e for e in errors)

    def test_duplicate_visual_ids(self):
        c = RequirementsContract(
            page_count=1,
            page_names=["Page 1"],
            visuals=[
                VisualContract(visual_id="v1", page="Page 1"),
                VisualContract(visual_id="v1", page="Page 1"),
            ],
        )
        errors = validate_contract(c)
        assert any("Duplicate" in e for e in errors)

    def test_visual_references_invalid_page(self):
        c = RequirementsContract(
            page_count=1,
            page_names=["Page 1"],
            visuals=[VisualContract(visual_id="v1", page="Nonexistent")],
        )
        errors = validate_contract(c)
        assert any("Nonexistent" in e for e in errors)

    def test_zero_pages_invalid(self):
        c = RequirementsContract(page_count=0, page_names=[])
        errors = validate_contract(c)
        assert any("page_count" in e for e in errors)


# ================================================================== #
#  Enforcement                                                          #
# ================================================================== #

class TestEnforcement:

    def test_page_count_enforced(self):
        c = build_contract("single page report")
        llm_spec = {
            "dashboard_title": "Wrong Title",
            "pages": [
                {"name": "Page A", "visuals": []},
                {"name": "Page B", "visuals": []},
            ],
        }
        fixed, violations = enforce_contract(c, llm_spec)
        assert len(fixed["pages"]) == 1
        assert len(violations) >= 1

    def test_title_enforced(self):
        c = build_contract('dashboard titled "My Report"')
        llm_spec = {"dashboard_title": "Something Else", "pages": []}
        fixed, violations = enforce_contract(c, llm_spec)
        assert fixed["dashboard_title"] == "My Report"

    def test_page_names_enforced(self):
        c = build_contract("single page called 'Sales Overview'")
        llm_spec = {"pages": [{"name": "WrongName", "visuals": []}]}
        fixed, violations = enforce_contract(c, llm_spec)
        assert fixed["pages"][0]["name"] == "Sales Overview"


# ================================================================== #
#  Prompt formatting                                                    #
# ================================================================== #

class TestPromptFormatting:

    def test_contains_binding_header(self):
        c = build_contract("single page dashboard")
        prompt = format_contract_for_prompt(c)
        assert "REQUIREMENTS CONTRACT" in prompt
        assert "BINDING" in prompt

    def test_contains_page_count(self):
        c = build_contract("3 page report")
        prompt = format_contract_for_prompt(c)
        assert "exactly 3" in prompt

    def test_contains_visuals(self):
        c = build_contract("build a bar chart")
        prompt = format_contract_for_prompt(c)
        assert "REQUIRED VISUALS" in prompt or "AMBIGUITIES" in prompt


# ================================================================== #
#  Serialization                                                        #
# ================================================================== #

class TestSerialization:

    def test_to_dict(self):
        c = build_contract("test")
        d = c.to_dict()
        assert isinstance(d, dict)
        assert "page_count" in d
        assert "visuals" in d

    def test_to_json(self):
        c = build_contract("test")
        j = c.to_json()
        parsed = json.loads(j)
        assert parsed["page_count"] == 1

    def test_roundtrip_deterministic(self):
        c1 = build_contract("build a 2 page dashboard with top slicers")
        c2 = build_contract("build a 2 page dashboard with top slicers")
        assert c1.to_json() == c2.to_json()


# ================================================================== #
#  REGRESSION TEST: Specific scenario                                   #
# ================================================================== #

class TestRegressionDashOverall:
    """
    Scenario: The output must be a single report page called
    "Dash Overall Transactions" with top slicers, tab-like category
    selector, and three monthly visuals stacked vertically.
    """

    USER_REQUEST = (
        'Build a single page Power BI report called "Dash Overall Transactions" '
        'with top slicers, a tab-like category selector, '
        'and three monthly visuals stacked vertically'
    )

    def test_single_page(self):
        c = build_contract(self.USER_REQUEST)
        assert c.page_count == 1
        assert any(
            lc.constraint_type == "page_count_fixed" and lc.value == "1"
            for lc in c.layout_constraints
        )

    def test_page_name(self):
        c = build_contract(self.USER_REQUEST)
        assert c.page_names == ["Dash Overall Transactions"]

    def test_top_slicers(self):
        c = build_contract(self.USER_REQUEST)
        assert len(c.top_filters) >= 1
        assert any(f.position == "top" for f in c.top_filters)

    def test_tab_selector(self):
        c = build_contract(self.USER_REQUEST)
        assert any(
            ne.element_type == "tab_selector"
            for ne in c.navigation_elements
        )

    def test_three_visuals(self):
        c = build_contract(self.USER_REQUEST)
        assert len(c.visuals) >= 3

    def test_vertical_stacking(self):
        c = build_contract(self.USER_REQUEST)
        assert any(
            lc.constraint_type == "stacking_direction" and lc.value == "vertical"
            for lc in c.layout_constraints
        )
        # Visuals should have sequential stacking order and full width
        for v in c.visuals:
            assert v.width_pct == 100.0

    def test_all_visuals_on_same_page(self):
        c = build_contract(self.USER_REQUEST)
        for v in c.visuals:
            assert v.page == "Dash Overall Transactions"

    def test_contract_validates(self):
        c = build_contract(self.USER_REQUEST)
        errors = validate_contract(c)
        assert errors == [], f"Validation errors: {errors}"

    def test_enforcement_preserves_structure(self):
        c = build_contract(self.USER_REQUEST)
        llm_spec = {
            "dashboard_title": "Wrong",
            "pages": [
                {"name": "Wrong Page", "visuals": []},
                {"name": "Extra Page", "visuals": []},
            ],
        }
        fixed, violations = enforce_contract(c, llm_spec)
        assert len(fixed["pages"]) == 1
        assert fixed["pages"][0]["name"] == "Dash Overall Transactions"
        assert fixed["dashboard_title"] == "Dash Overall Transactions"

    def test_prompt_contains_all_requirements(self):
        c = build_contract(self.USER_REQUEST)
        prompt = format_contract_for_prompt(c)
        assert "Dash Overall Transactions" in prompt
        assert "exactly 1" in prompt
        assert "stacking_direction: vertical" in prompt
        assert "tab_selector" in prompt

    def test_deterministic_across_runs(self):
        c1 = build_contract(self.USER_REQUEST)
        c2 = build_contract(self.USER_REQUEST)
        assert c1.to_json() == c2.to_json()


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
