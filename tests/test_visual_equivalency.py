"""
Tests for core/visual_equivalency.py -- Visual Equivalency Engine.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.visual_equivalency import (
    evaluate,
    evaluate_batch,
    EquivalencyStatus,
    VisualEquivalency,
)
from core.tableau_extractor import (
    VisualSpec,
    EncodingSpec,
    ColumnRef,
    Confidence,
)


# ================================================================== #
#  Helpers                                                              #
# ================================================================== #

def _vs(name="Test", mark="bar", rows=None, cols=None, color=None,
        is_kpi=False, is_map=False, is_text_table=False, is_breadcrumb=False,
        columns_used=None):
    """Build a minimal VisualSpec for testing."""
    enc = EncodingSpec(
        shelf_rows=rows or [],
        shelf_cols=cols or [],
        color=color,
    )
    return VisualSpec(
        name=name,
        mark_type=mark,
        encoding=enc,
        is_kpi_card=is_kpi,
        is_map=is_map,
        is_text_table=is_text_table,
        is_breadcrumb=is_breadcrumb,
        columns_used=columns_used or [],
    )


def _dim(name):
    return ColumnRef(name=name, role="dimension")


def _meas(name, agg="Sum"):
    return ColumnRef(name=name, role="measure", aggregation=agg)


# ================================================================== #
#  Core visual type mappings                                            #
# ================================================================== #

class TestBarChart:

    def test_horizontal_bar(self):
        vs = _vs("Sales by Category", "bar",
                  rows=[_dim("Category")], cols=[_meas("Sales")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "clusteredBarChart"
        assert eq.equivalency_status == EquivalencyStatus.EXACT
        assert not eq.analyst_review_required

    def test_vertical_column(self):
        vs = _vs("Sales by Month", "bar",
                  rows=[_meas("Sales")], cols=[_dim("Month")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "clusteredColumnChart"
        assert eq.equivalency_status == EquivalencyStatus.EXACT

    def test_stacked_bar(self):
        vs = _vs("Sales by Cat/Seg", "bar",
                  rows=[_dim("Category")], cols=[_meas("Sales")],
                  color=_dim("Segment"))
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "stackedBarChart"
        assert eq.equivalency_status == EquivalencyStatus.EXACT
        assert any(r.role == "Legend" for r in eq.required_field_roles)

    def test_stacked_column(self):
        vs = _vs("Stacked", "bar",
                  rows=[_meas("Sales")], cols=[_dim("Month")],
                  color=_dim("Category"))
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "stackedColumnChart"
        assert eq.equivalency_status == EquivalencyStatus.EXACT


class TestLineChart:

    def test_line(self):
        vs = _vs("Trend", "line",
                  rows=[_meas("Sales")], cols=[_dim("Date")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "lineChart"
        assert eq.equivalency_status == EquivalencyStatus.EXACT


class TestComboChart:

    def test_dual_axis(self):
        vs = _vs("Dual", "dual-axis",
                  rows=[_meas("Sales"), _meas("Profit")], cols=[_dim("Month")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "lineClusteredColumnComboChart"
        assert eq.equivalency_status == EquivalencyStatus.CLOSE_EQUIVALENT
        assert eq.analyst_review_required

    def test_two_measures_detected_as_combo(self):
        vs = _vs("Multi", "automatic",
                  rows=[_meas("Sales"), _meas("Profit")], cols=[_dim("Month")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "lineClusteredColumnComboChart"


class TestCard:

    def test_kpi_card(self):
        vs = _vs("Total Sales", "text", is_kpi=True)
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "card"
        assert eq.equivalency_status == EquivalencyStatus.EXACT

    def test_auto_kpi_detection(self):
        """Empty shelves + text mark = KPI card."""
        vs = _vs("Revenue", "text",
                  columns_used=[_meas("Revenue")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "card"


class TestTable:

    def test_text_table(self):
        vs = _vs("Detail", "text", is_text_table=True,
                  rows=[_dim("Product"), _dim("Region")],
                  cols=[_meas("Sales"), _meas("Profit")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "matrix"
        assert eq.equivalency_status == EquivalencyStatus.EXACT
        assert any(r.role == "Rows" for r in eq.required_field_roles)
        assert any(r.role == "Values" for r in eq.required_field_roles)


class TestMatrix:

    def test_highlight_table(self):
        vs = _vs("Highlight", "highlight-table")
        # Not detected as text_table since no flag
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "matrix"
        assert eq.equivalency_status == EquivalencyStatus.CLOSE_EQUIVALENT


class TestSlicer:

    def test_filter_visual(self):
        vs = _vs("Region Filter", "automatic",
                  rows=[_dim("Region")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "slicer"
        assert eq.equivalency_status == EquivalencyStatus.EXACT

    def test_selector_visual(self):
        vs = _vs("Category Selector", "automatic",
                  cols=[_dim("Category")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "slicer"


class TestPseudoTab:

    def test_breadcrumb(self):
        vs = _vs("Ship Mode Bread Crumb", "text", is_breadcrumb=True)
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "slicer"
        assert eq.equivalency_status == EquivalencyStatus.CLOSE_EQUIVALENT
        assert "Horizontal" in eq.formatting_notes[0]
        assert "tile" in eq.formatting_notes[1].lower()

    def test_tab_selector(self):
        vs = _vs("Category Tab Selector", "automatic",
                  cols=[_dim("Category")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "slicer"

    def test_pseudo_tab_interaction_notes(self):
        vs = _vs("Toggle View", "text", is_breadcrumb=True)
        eq = evaluate(vs)
        assert any("cross-filter" in n.lower() for n in eq.interaction_notes)
        assert any("single-select" in n.lower() for n in eq.interaction_notes)


class TestMap:

    def test_polygon_map(self):
        vs = _vs("Geo View", "polygon", is_map=True,
                  columns_used=[_dim("State")])
        eq = evaluate(vs)
        assert eq.chosen_powerbi_visual == "map"
        assert eq.equivalency_status == EquivalencyStatus.CLOSE_EQUIVALENT
        assert eq.analyst_review_required
        assert any(r.role == "Location" for r in eq.required_field_roles)


class TestUnsupported:

    def test_gantt(self):
        vs = _vs("Timeline", "gantt")
        eq = evaluate(vs)
        assert eq.equivalency_status == EquivalencyStatus.APPROXIMATION_REQUIRED
        assert eq.analyst_review_required
        assert "AppSource" in eq.rationale

    def test_box_plot(self):
        vs = _vs("Distribution", "box-plot")
        eq = evaluate(vs)
        assert eq.equivalency_status == EquivalencyStatus.APPROXIMATION_REQUIRED
        assert eq.analyst_review_required

    def test_unknown_mark(self):
        vs = _vs("Mystery", "sparkline")
        eq = evaluate(vs)
        assert eq.equivalency_status == EquivalencyStatus.APPROXIMATION_REQUIRED
        assert eq.analyst_review_required
        assert "Unknown" in eq.rationale


class TestAutomatic:

    def test_automatic_ambiguous(self):
        """Automatic with no clear shelf orientation needs review."""
        vs = _vs("Ambiguous", "automatic")
        eq = evaluate(vs)
        assert eq.analyst_review_required or eq.equivalency_status != EquivalencyStatus.EXACT


# ================================================================== #
#  Batch evaluation                                                     #
# ================================================================== #

class TestBatch:

    def test_batch(self):
        visuals = [
            _vs("A", "bar", rows=[_dim("X")], cols=[_meas("Y")]),
            _vs("B", "line", rows=[_meas("Y")], cols=[_dim("Date")]),
            _vs("C", "text", is_kpi=True),
        ]
        results = evaluate_batch(visuals)
        assert len(results) == 3
        assert results[0].chosen_powerbi_visual == "clusteredBarChart"
        assert results[1].chosen_powerbi_visual == "lineChart"
        assert results[2].chosen_powerbi_visual == "card"


# ================================================================== #
#  Output structure validation                                          #
# ================================================================== #

class TestOutputStructure:

    def test_all_fields_present(self):
        vs = _vs("Test", "bar", rows=[_dim("X")], cols=[_meas("Y")])
        eq = evaluate(vs)
        assert eq.source_name == "Test"
        assert eq.source_mark_type == "bar"
        assert eq.chosen_powerbi_visual != ""
        assert isinstance(eq.equivalency_status, EquivalencyStatus)
        assert eq.rationale != ""
        assert isinstance(eq.required_field_roles, list)
        assert isinstance(eq.formatting_notes, list)
        assert isinstance(eq.interaction_notes, list)
        assert isinstance(eq.analyst_review_required, bool)

    def test_field_roles_have_role_name(self):
        vs = _vs("Test", "bar",
                  rows=[_dim("Category")], cols=[_meas("Sales")],
                  color=_dim("Segment"))
        eq = evaluate(vs)
        for role in eq.required_field_roles:
            assert role.role != ""


# ================================================================== #
#  REGRESSION: Layout with slicers + tabs + stacked visuals            #
# ================================================================== #

class TestRegressionLayout:
    """
    Layout:
    - Top-row slicers (Region, Year)
    - Horizontal category selector (pseudo-tab)
    - Three vertically stacked monthly visuals (bar, line, table)
    """

    def _build_layout(self):
        return [
            # Top-row slicers
            _vs("Region Filter", "automatic", rows=[_dim("Region")]),
            _vs("Year Selector", "automatic", cols=[_dim("Year")]),
            # Horizontal category selector (pseudo-tab)
            _vs("Category Tab Selector", "automatic", cols=[_dim("Category")]),
            # Three monthly visuals stacked vertically
            _vs("Monthly Sales", "bar",
                rows=[_dim("Month")], cols=[_meas("Sales")]),
            _vs("Monthly Trend", "line",
                rows=[_meas("Profit")], cols=[_dim("Month")]),
            _vs("Monthly Detail", "text", is_text_table=True,
                rows=[_dim("Month"), _dim("Product")],
                cols=[_meas("Sales"), _meas("Profit")]),
        ]

    def test_slicers_detected(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        slicer_results = [r for r in results if r.chosen_powerbi_visual == "slicer"]
        assert len(slicer_results) >= 2, (
            f"Expected at least 2 slicers, got {len(slicer_results)}: "
            f"{[r.source_name for r in slicer_results]}"
        )

    def test_tab_selector_detected(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        tab_result = next(r for r in results if r.source_name == "Category Tab Selector")
        assert tab_result.chosen_powerbi_visual == "slicer"
        assert tab_result.equivalency_status == EquivalencyStatus.CLOSE_EQUIVALENT
        assert any("Horizontal" in n for n in tab_result.formatting_notes)
        assert any("tile" in n.lower() for n in tab_result.formatting_notes)

    def test_monthly_bar(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        bar = next(r for r in results if r.source_name == "Monthly Sales")
        assert bar.chosen_powerbi_visual == "clusteredBarChart"
        assert bar.equivalency_status == EquivalencyStatus.EXACT

    def test_monthly_line(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        line = next(r for r in results if r.source_name == "Monthly Trend")
        assert line.chosen_powerbi_visual == "lineChart"
        assert line.equivalency_status == EquivalencyStatus.EXACT

    def test_monthly_table(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        table = next(r for r in results if r.source_name == "Monthly Detail")
        assert table.chosen_powerbi_visual == "matrix"
        assert table.equivalency_status == EquivalencyStatus.EXACT

    def test_no_unsupported_in_layout(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        unsupported = [r for r in results
                       if r.equivalency_status == EquivalencyStatus.UNSUPPORTED]
        assert len(unsupported) == 0

    def test_field_roles_populated_for_all(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        for r in results:
            assert len(r.required_field_roles) >= 1, (
                f"{r.source_name} has no field roles"
            )

    def test_review_only_where_needed(self):
        visuals = self._build_layout()
        results = evaluate_batch(visuals)
        review_names = [r.source_name for r in results if r.analyst_review_required]
        # Slicers, bar, line, table, tab should not need review
        assert "Monthly Sales" not in review_names
        assert "Monthly Trend" not in review_names
        assert "Monthly Detail" not in review_names


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
