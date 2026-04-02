"""
Golden regression test for the Tableau -> Power BI pipeline.

Expected output: A single Power BI page named "Dash Overall Transactions" with:
- top-right last refresh text
- 3 slicers in the top row
- horizontal tab-like selector for category
- section label "Total Monthly"
- 3 monthly visuals stacked vertically:
  1) TV MTD as stacked column with MonthYear on X-axis, legend by transaction subtype
  2) TC MTD as stacked column with MonthYear on X-axis, legend by transaction subtype
  3) UU MTD as clustered column with MonthYear on X-axis
- cross-filtering enabled
- consistent legend colors across stacked charts

Measures:
- TV MTD = total month-to-date amount
- TC MTD = total month-to-date row count
- UU MTD = total month-to-date distinct account count

These tests are written to FAIL first, then code is fixed to make them pass.
Do NOT weaken tests to fit current code.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.requirements_contract import (
    build_contract,
    validate_contract,
    RequirementsContract,
    VisualType,
)
from core.layout_assembler import (
    assemble,
    CANVAS_W,
    CANVAS_H,
    MARGIN,
    SLICER_H,
)
from core.qa_manifest import build_manifest


# The exact user request that drives the golden scenario.
GOLDEN_REQUEST = (
    'Build a single page Power BI report called "Dash Overall Transactions" with: '
    "top-right last refresh text, "
    "3 slicers in the top row, "
    "a horizontal tab-like selector for category, "
    'a section label "Total Monthly", '
    "3 monthly visuals stacked vertically: "
    "1) TV MTD as stacked column with MonthYear on X-axis and legend by transaction subtype, "
    "2) TC MTD as stacked column with MonthYear on X-axis and legend by transaction subtype, "
    "3) UU MTD as clustered column with MonthYear on X-axis. "
    "Enable cross-filtering. Use consistent legend colors across the stacked charts."
)


def _build():
    return build_contract(GOLDEN_REQUEST)


# ================================================================== #
#  Page structure                                                       #
# ================================================================== #

class TestGoldenPageStructure:

    def test_single_page(self):
        c = _build()
        assert c.page_count == 1

    def test_page_name(self):
        c = _build()
        assert c.page_names == ["Dash Overall Transactions"]

    def test_title(self):
        c = _build()
        assert c.dashboard_title == "Dash Overall Transactions"

    def test_contract_validates(self):
        c = _build()
        errors = validate_contract(c)
        assert errors == [], f"Validation errors: {errors}"


# ================================================================== #
#  Slicers                                                              #
# ================================================================== #

class TestGoldenSlicers:

    def test_three_slicers(self):
        c = _build()
        top_slicers = [f for f in c.top_filters if f.position == "top"]
        assert len(top_slicers) == 3, (
            f"Expected 3 top slicers, got {len(top_slicers)}: "
            f"{[f.field for f in top_slicers]}"
        )

    def test_slicers_in_layout(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        slicer_visuals = [v for v in p.visuals
                          if v.visual_type == "slicer" and "slicer_" in v.visual_id]
        assert len(slicer_visuals) == 3

    def test_slicers_same_row(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        slicers = [v for v in p.visuals
                   if v.visual_type == "slicer" and "slicer_" in v.visual_id]
        if len(slicers) >= 2:
            ys = {s.y for s in slicers}
            assert len(ys) == 1, f"Slicers at different y positions: {ys}"


# ================================================================== #
#  Tab selector                                                         #
# ================================================================== #

class TestGoldenTabSelector:

    def test_tab_selector_present(self):
        c = _build()
        tabs = [n for n in c.navigation_elements if n.element_type == "tab_selector"]
        assert len(tabs) >= 1

    def test_tab_in_layout(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        navs = [v for v in p.visuals if "nav_" in v.visual_id]
        assert len(navs) >= 1


# ================================================================== #
#  Refresh text                                                         #
# ================================================================== #

class TestGoldenRefreshText:

    def test_refresh_text_in_contract(self):
        c = _build()
        # Must have a visual or layout element for "last refresh" text
        has_refresh = (
            any(v.visual_type == VisualType.TEXT_BOX and "refresh" in v.title.lower()
                for v in c.visuals)
            or any("refresh" in lc.constraint_type.lower() or "refresh" in lc.value.lower()
                   for lc in c.layout_constraints)
        )
        assert has_refresh, "No refresh text element found in contract"

    def test_refresh_text_in_layout(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        refresh = [v for v in p.visuals
                   if "refresh" in v.visual_id.lower() or "refresh" in v.title.lower()]
        assert len(refresh) >= 1, "No refresh text visual in layout"


# ================================================================== #
#  Section label                                                        #
# ================================================================== #

class TestGoldenSectionLabel:

    def test_section_label_in_contract(self):
        c = _build()
        has_label = any(
            v.visual_type == VisualType.TEXT_BOX and "total monthly" in v.title.lower()
            for v in c.visuals
        )
        assert has_label, "No section label 'Total Monthly' in contract visuals"

    def test_section_label_in_layout(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        labels = [v for v in p.visuals
                  if "total monthly" in v.title.lower() or "section" in v.visual_id.lower()]
        assert len(labels) >= 1


# ================================================================== #
#  Monthly visuals                                                      #
# ================================================================== #

class TestGoldenMonthlyVisuals:

    def test_three_content_visuals(self):
        c = _build()
        charts = [v for v in c.visuals
                  if v.visual_type not in (VisualType.SLICER, VisualType.TEXT_BOX, VisualType.UNKNOWN)]
        assert len(charts) == 3, (
            f"Expected 3 chart visuals, got {len(charts)}: "
            f"{[(v.visual_id, v.visual_type, v.title) for v in charts]}"
        )

    def test_tv_mtd_stacked_column(self):
        c = _build()
        tv = next((v for v in c.visuals if "tv mtd" in v.title.lower()), None)
        assert tv is not None, "No visual with title containing 'TV MTD'"
        assert tv.visual_type == VisualType.STACKED_BAR or tv.visual_type.value == "stackedColumnChart", (
            f"TV MTD should be stacked column, got {tv.visual_type}"
        )

    def test_tc_mtd_stacked_column(self):
        c = _build()
        tc = next((v for v in c.visuals if "tc mtd" in v.title.lower()), None)
        assert tc is not None, "No visual with title containing 'TC MTD'"
        assert tc.visual_type == VisualType.STACKED_BAR or tc.visual_type.value == "stackedColumnChart", (
            f"TC MTD should be stacked column, got {tc.visual_type}"
        )

    def test_uu_mtd_clustered_column(self):
        c = _build()
        uu = next((v for v in c.visuals if "uu mtd" in v.title.lower()), None)
        assert uu is not None, "No visual with title containing 'UU MTD'"
        assert tv_is_column(uu), f"UU MTD should be clustered column, got {uu.visual_type}"

    def test_visuals_have_monthyear_axis(self):
        c = _build()
        for v in c.visuals:
            if v.title and "mtd" in v.title.lower():
                assert v.category_axis is not None, f"{v.title} missing category axis"
                assert "monthyear" in v.category_axis.field.lower().replace(" ", "").replace("_", ""), (
                    f"{v.title} axis should be MonthYear, got '{v.category_axis.field}'"
                )

    def test_stacked_visuals_have_legend(self):
        c = _build()
        for v in c.visuals:
            if v.title and "mtd" in v.title.lower() and v.title.lower().startswith(("tv", "tc")):
                assert v.color_field, (
                    f"{v.title} should have a legend/color field (transaction subtype)"
                )

    def test_vertical_stacking(self):
        c = _build()
        assert any(
            lc.constraint_type == "stacking_direction" and lc.value == "vertical"
            for lc in c.layout_constraints
        )

    def test_visuals_stacked_in_layout(self):
        c = _build()
        layout = assemble(c)
        p = layout.pages[0]
        charts = sorted(
            [v for v in p.visuals if v.visual_id.startswith("chart_")],
            key=lambda v: v.y,
        )
        if len(charts) >= 3:
            assert charts[0].y < charts[1].y < charts[2].y
            # All full width
            for ch in charts:
                assert ch.width == CANVAS_W - 2 * MARGIN


# ================================================================== #
#  Interactivity                                                        #
# ================================================================== #

class TestGoldenInteractivity:

    def test_cross_filtering_enabled(self):
        c = _build()
        has_cross_filter = any(
            ir.rule_type == "cross_filter" for ir in c.interactivity_rules
        )
        assert has_cross_filter, "Cross-filtering rule not found in contract"

    def test_consistent_legend_colors(self):
        c = _build()
        has_color_rule = any(
            "legend" in fc.constraint_type.lower() or "color" in fc.constraint_type.lower()
            or "consistent" in fc.value.lower()
            for fc in c.formatting_constraints
        )
        assert has_color_rule, "Consistent legend color constraint not found"


# ================================================================== #
#  DAX measures                                                         #
# ================================================================== #

class TestGoldenMeasures:

    def test_tv_mtd_measure_in_contract(self):
        c = _build()
        tv = next((v for v in c.visuals if "tv mtd" in v.title.lower()), None)
        assert tv is not None
        # The value axis should reference TV MTD
        assert tv.value_axis is not None, "TV MTD must have a value axis"

    def test_tc_mtd_measure_in_contract(self):
        c = _build()
        tc = next((v for v in c.visuals if "tc mtd" in v.title.lower()), None)
        assert tc is not None
        assert tc.value_axis is not None

    def test_uu_mtd_measure_in_contract(self):
        c = _build()
        uu = next((v for v in c.visuals if "uu mtd" in v.title.lower()), None)
        assert uu is not None
        assert uu.value_axis is not None


# ================================================================== #
#  QA manifest                                                          #
# ================================================================== #

class TestGoldenManifest:

    def test_manifest_builds(self):
        c = _build()
        m = build_manifest(contract=c)
        assert m.generation_id != ""

    def test_manifest_has_checklist(self):
        c = _build()
        m = build_manifest(contract=c)
        assert len(m.qa_checklist) > 0

    def test_manifest_captures_layout_constraints(self):
        c = _build()
        m = build_manifest(contract=c)
        assert len(m.layout_constraints_applied) >= 2


# ================================================================== #
#  Helper                                                               #
# ================================================================== #

def tv_is_column(v):
    """Check if a visual type represents a clustered column chart."""
    if hasattr(v, "visual_type"):
        vt = v.visual_type
        if hasattr(vt, "value"):
            return vt.value in ("clusteredColumnChart", "clusteredBarChart")
        return str(vt) in ("clusteredColumnChart", "clusteredBarChart",
                           "VisualType.BAR", "VisualType.COLUMN")
    return False


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
