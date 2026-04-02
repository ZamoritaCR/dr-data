"""
Tests for core/layout_assembler.py -- Page assembly and layout constraints.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.layout_assembler import (
    assemble,
    AssembledLayout,
    AssembledPage,
    PlacedVisual,
    CANVAS_W,
    CANVAS_H,
    MARGIN,
    GAP,
    SLICER_H,
    NAV_H,
    HEADER_H,
    CARD_H,
)
from core.requirements_contract import (
    RequirementsContract,
    VisualContract,
    VisualType,
    Confidence,
    FilterContractSpec,
    NavigationElement,
    LayoutConstraint,
    AxisSpec,
    build_contract,
)


# ================================================================== #
#  Helpers                                                              #
# ================================================================== #

def _vc(vid, page, vtype=VisualType.BAR, title="", order=0,
        cat_field="", val_field="", color_field="", width_pct=100.0):
    cat = AxisSpec(field=cat_field) if cat_field else None
    val = AxisSpec(field=val_field, aggregation="Sum") if val_field else None
    return VisualContract(
        visual_id=vid, page=page, visual_type=vtype, title=title,
        stacking_order=order, category_axis=cat, value_axis=val,
        color_field=color_field, width_pct=width_pct,
    )


def _contract(
    title="Dashboard",
    page_count=1,
    page_names=None,
    visuals=None,
    filters=None,
    nav=None,
    constraints=None,
):
    return RequirementsContract(
        dashboard_title=title,
        page_count=page_count,
        page_names=page_names or ["Page 1"],
        visuals=visuals or [],
        top_filters=filters or [],
        navigation_elements=nav or [],
        layout_constraints=constraints or [
            LayoutConstraint(constraint_type="canvas_size", value="1280x720"),
        ],
    )


# ================================================================== #
#  Single page enforcement                                              #
# ================================================================== #

class TestSinglePage:

    def test_single_page_produces_one_page(self):
        c = _contract(page_count=1, page_names=["Overview"])
        layout = assemble(c)
        assert len(layout.pages) == 1
        assert layout.pages[0].display_name == "Overview"

    def test_single_page_not_split(self):
        """Even with many visuals, single page stays single."""
        visuals = [
            _vc(f"v{i}", "Overview", order=i) for i in range(10)
        ]
        c = _contract(
            page_count=1, page_names=["Overview"],
            visuals=visuals,
            constraints=[
                LayoutConstraint(constraint_type="page_count_fixed", value="1"),
                LayoutConstraint(constraint_type="stacking_direction", value="vertical"),
            ],
        )
        layout = assemble(c)
        assert len(layout.pages) == 1
        assert len([v for v in layout.pages[0].visuals
                     if v.visual_type != "textbox"]) == 10

    def test_named_page_title(self):
        c = _contract(title="My Dashboard", page_count=1, page_names=["Sales"])
        layout = assemble(c)
        assert layout.pages[0].display_name == "Sales"
        # Title card should contain dashboard title
        title_visuals = [v for v in layout.pages[0].visuals if v.visual_id == "title_card"]
        assert len(title_visuals) == 1
        assert title_visuals[0].title == "My Dashboard"


# ================================================================== #
#  Multi-page                                                           #
# ================================================================== #

class TestMultiPage:

    def test_two_pages(self):
        visuals = [
            _vc("v1", "Sales", order=0),
            _vc("v2", "Details", order=0),
        ]
        c = _contract(
            page_count=2, page_names=["Sales", "Details"],
            visuals=visuals,
        )
        layout = assemble(c)
        assert len(layout.pages) == 2
        assert layout.pages[0].display_name == "Sales"
        assert layout.pages[1].display_name == "Details"

    def test_visuals_go_to_correct_page(self):
        visuals = [
            _vc("v1", "Sales", title="Sales Chart"),
            _vc("v2", "Details", title="Detail Table"),
        ]
        c = _contract(
            page_count=2, page_names=["Sales", "Details"],
            visuals=visuals,
        )
        layout = assemble(c)
        p1_titles = [v.title for v in layout.pages[0].visuals if v.visual_type != "textbox"]
        p2_titles = [v.title for v in layout.pages[1].visuals if v.visual_type != "textbox"]
        assert "Sales Chart" in p1_titles
        assert "Detail Table" in p2_titles


# ================================================================== #
#  Vertical stacking                                                    #
# ================================================================== #

class TestVerticalStack:

    def test_three_visuals_stacked(self):
        visuals = [
            _vc("v1", "Page 1", title="Chart A", order=0),
            _vc("v2", "Page 1", title="Chart B", order=1),
            _vc("v3", "Page 1", title="Chart C", order=2),
        ]
        c = _contract(
            visuals=visuals,
            constraints=[
                LayoutConstraint(constraint_type="stacking_direction", value="vertical"),
            ],
        )
        layout = assemble(c)
        content = [v for v in layout.pages[0].visuals
                   if v.visual_type != "textbox"]
        assert len(content) == 3

        # All full width
        for v in content:
            assert v.width == CANVAS_W - 2 * MARGIN

        # Vertically ordered: y increases
        ys = [v.y for v in content]
        assert ys == sorted(ys)
        assert ys[0] < ys[1] < ys[2]

    def test_equal_height_distribution(self):
        visuals = [
            _vc("v1", "Page 1", order=0),
            _vc("v2", "Page 1", order=1),
        ]
        c = _contract(
            visuals=visuals,
            constraints=[
                LayoutConstraint(constraint_type="stacking_direction", value="vertical"),
            ],
        )
        layout = assemble(c)
        content = [v for v in layout.pages[0].visuals
                   if v.visual_type != "textbox"]
        heights = [v.height for v in content]
        # Should be roughly equal (within rounding)
        assert abs(heights[0] - heights[1]) <= 1

    def test_no_overlap(self):
        visuals = [
            _vc("v1", "Page 1", order=0),
            _vc("v2", "Page 1", order=1),
            _vc("v3", "Page 1", order=2),
        ]
        c = _contract(
            visuals=visuals,
            constraints=[
                LayoutConstraint(constraint_type="stacking_direction", value="vertical"),
            ],
        )
        layout = assemble(c)
        all_v = layout.pages[0].visuals
        for i in range(len(all_v)):
            for j in range(i + 1, len(all_v)):
                a, b = all_v[i], all_v[j]
                # Check no vertical overlap (same column)
                if a.x < b.x + b.width and a.x + a.width > b.x:
                    assert a.y + a.height <= b.y or b.y + b.height <= a.y, \
                        f"{a.visual_id} overlaps {b.visual_id}"


# ================================================================== #
#  Top-row slicers                                                      #
# ================================================================== #

class TestTopSlicers:

    def test_slicers_placed_at_top(self):
        c = _contract(
            filters=[
                FilterContractSpec(field="Region", position="top", page="Page 1"),
                FilterContractSpec(field="Year", position="top", page="Page 1"),
            ],
        )
        layout = assemble(c)
        slicers = [v for v in layout.pages[0].visuals if v.visual_type == "slicer"]
        assert len(slicers) == 2
        # Both at same y (one row)
        assert slicers[0].y == slicers[1].y
        # Same height
        assert slicers[0].height == SLICER_H

    def test_slicers_equally_wide(self):
        c = _contract(
            filters=[
                FilterContractSpec(field="A", position="top", page="Page 1"),
                FilterContractSpec(field="B", position="top", page="Page 1"),
                FilterContractSpec(field="C", position="top", page="Page 1"),
            ],
        )
        layout = assemble(c)
        slicers = [v for v in layout.pages[0].visuals if v.visual_type == "slicer"]
        widths = [s.width for s in slicers]
        # All same width
        assert len(set(widths)) == 1
        # Fill the row
        total_w = sum(widths) + (len(slicers) - 1) * GAP
        assert abs(total_w - (CANVAS_W - 2 * MARGIN)) <= len(slicers)  # rounding tolerance

    def test_slicers_above_content(self):
        c = _contract(
            visuals=[_vc("v1", "Page 1", order=0)],
            filters=[
                FilterContractSpec(field="Region", position="top", page="Page 1"),
            ],
        )
        layout = assemble(c)
        slicer = next(v for v in layout.pages[0].visuals if v.visual_type == "slicer")
        content = next(v for v in layout.pages[0].visuals
                       if v.visual_id == "v1")
        assert slicer.y + slicer.height < content.y


# ================================================================== #
#  Navigation / tab selector placement                                  #
# ================================================================== #

class TestNavPlacement:

    def test_nav_below_slicers(self):
        c = _contract(
            filters=[
                FilterContractSpec(field="Region", position="top", page="Page 1"),
            ],
            nav=[
                NavigationElement(element_type="tab_selector", selector_field="Category"),
            ],
        )
        layout = assemble(c)
        slicer = next(v for v in layout.pages[0].visuals
                      if "slicer_" in v.visual_id)
        nav = next(v for v in layout.pages[0].visuals
                   if "nav_" in v.visual_id)
        assert nav.y > slicer.y + slicer.height - 1

    def test_nav_full_width(self):
        c = _contract(
            nav=[NavigationElement(element_type="tab_selector")],
        )
        layout = assemble(c)
        nav = next(v for v in layout.pages[0].visuals
                   if "nav_" in v.visual_id)
        assert nav.width == CANVAS_W - 2 * MARGIN


# ================================================================== #
#  Title / header card                                                  #
# ================================================================== #

class TestTitleCard:

    def test_title_at_top(self):
        c = _contract(title="My Dashboard")
        layout = assemble(c)
        title = next(v for v in layout.pages[0].visuals
                     if v.visual_id == "title_card")
        assert title.y == MARGIN
        assert title.visual_type == "textbox"

    def test_title_above_slicers(self):
        c = _contract(
            title="Dashboard",
            filters=[FilterContractSpec(field="X", position="top", page="Page 1")],
        )
        layout = assemble(c)
        title = next(v for v in layout.pages[0].visuals
                     if v.visual_id == "title_card")
        slicer = next(v for v in layout.pages[0].visuals
                      if v.visual_type == "slicer")
        assert title.y + title.height < slicer.y

    def test_no_title_when_empty(self):
        c = _contract(title="")
        layout = assemble(c)
        titles = [v for v in layout.pages[0].visuals
                  if v.visual_id == "title_card"]
        assert len(titles) == 0


# ================================================================== #
#  KPI card placement                                                   #
# ================================================================== #

class TestKPICards:

    def test_kpi_cards_in_row(self):
        visuals = [
            _vc("kpi1", "Page 1", vtype=VisualType.CARD, title="Revenue", order=0),
            _vc("kpi2", "Page 1", vtype=VisualType.CARD, title="Profit", order=1),
            _vc("v1", "Page 1", vtype=VisualType.BAR, title="Chart", order=2),
        ]
        c = _contract(visuals=visuals)
        layout = assemble(c)

        kpis = [v for v in layout.pages[0].visuals
                if v.visual_id.startswith("kpi")]
        charts = [v for v in layout.pages[0].visuals
                  if v.visual_id == "v1"]

        assert len(kpis) == 2
        # KPIs same y (one row)
        assert kpis[0].y == kpis[1].y
        # KPI height is CARD_H
        assert kpis[0].height == CARD_H
        # Chart below KPIs
        assert charts[0].y > kpis[0].y + kpis[0].height

    def test_kpi_equal_width(self):
        visuals = [
            _vc("k1", "Page 1", vtype=VisualType.CARD, order=0),
            _vc("k2", "Page 1", vtype=VisualType.CARD, order=1),
            _vc("k3", "Page 1", vtype=VisualType.CARD, order=2),
        ]
        c = _contract(visuals=visuals)
        layout = assemble(c)
        kpis = [v for v in layout.pages[0].visuals
                if v.visual_id.startswith("k")]
        widths = [k.width for k in kpis]
        assert len(set(widths)) == 1


# ================================================================== #
#  Canvas bounds                                                        #
# ================================================================== #

class TestCanvasBounds:

    def test_nothing_exceeds_canvas(self):
        visuals = [
            _vc(f"v{i}", "Page 1", order=i) for i in range(5)
        ]
        c = _contract(
            visuals=visuals,
            filters=[
                FilterContractSpec(field="A", position="top", page="Page 1"),
                FilterContractSpec(field="B", position="top", page="Page 1"),
            ],
            nav=[NavigationElement(element_type="tab_selector")],
        )
        layout = assemble(c)
        for v in layout.pages[0].visuals:
            assert v.x >= 0, f"{v.visual_id} x={v.x}"
            assert v.y >= 0, f"{v.visual_id} y={v.y}"
            assert v.x + v.width <= CANVAS_W + 1, \
                f"{v.visual_id} right edge {v.x + v.width} > {CANVAS_W}"


# ================================================================== #
#  Output format                                                        #
# ================================================================== #

class TestOutputFormat:

    def test_to_report_layout(self):
        c = _contract(
            visuals=[_vc("v1", "Page 1", title="Test")],
        )
        layout = assemble(c)
        rl = layout.to_report_layout()

        assert "sections" in rl
        assert len(rl["sections"]) == 1
        section = rl["sections"][0]
        assert "displayName" in section
        assert "visualContainers" in section
        assert len(section["visualContainers"]) >= 1

        vc = section["visualContainers"][-1]  # last one = the visual
        assert "x" in vc
        assert "y" in vc
        assert "width" in vc
        assert "height" in vc
        assert "config" in vc
        assert "visualType" in vc["config"]

    def test_section_has_display_option(self):
        c = _contract()
        layout = assemble(c)
        section = layout.to_report_layout()["sections"][0]
        assert section["displayOption"] == 1


# ================================================================== #
#  REGRESSION: Contract scenario                                        #
# ================================================================== #

class TestRegressionFullLayout:
    """
    Contract:
    - One page called "Dash Overall Transactions"
    - Top slicers (Region, Year)
    - Tab-like selector (Category)
    - Three visuals stacked vertically
    """

    def _build(self):
        return _contract(
            title="Dash Overall Transactions",
            page_count=1,
            page_names=["Dash Overall Transactions"],
            visuals=[
                _vc("v1", "Dash Overall Transactions", vtype=VisualType.BAR,
                    title="Monthly Sales", order=0, cat_field="Month", val_field="Sales"),
                _vc("v2", "Dash Overall Transactions", vtype=VisualType.LINE,
                    title="Monthly Trend", order=1, cat_field="Month", val_field="Profit"),
                _vc("v3", "Dash Overall Transactions", vtype=VisualType.TABLE,
                    title="Monthly Detail", order=2, cat_field="Month", val_field="Amount"),
            ],
            filters=[
                FilterContractSpec(field="Region", position="top",
                                   page="Dash Overall Transactions"),
                FilterContractSpec(field="Year", position="top",
                                   page="Dash Overall Transactions"),
            ],
            nav=[
                NavigationElement(element_type="tab_selector",
                                  selector_field="Category"),
            ],
            constraints=[
                LayoutConstraint(constraint_type="page_count_fixed", value="1",
                                 source="user_request"),
                LayoutConstraint(constraint_type="stacking_direction", value="vertical",
                                 source="user_request"),
                LayoutConstraint(constraint_type="canvas_size", value="1280x720"),
            ],
        )

    def test_one_page(self):
        layout = assemble(self._build())
        assert len(layout.pages) == 1

    def test_page_name(self):
        layout = assemble(self._build())
        assert layout.pages[0].display_name == "Dash Overall Transactions"

    def test_title_card_present(self):
        layout = assemble(self._build())
        titles = [v for v in layout.pages[0].visuals if v.visual_id == "title_card"]
        assert len(titles) == 1
        assert titles[0].title == "Dash Overall Transactions"

    def test_two_slicers_in_top_row(self):
        layout = assemble(self._build())
        slicers = [v for v in layout.pages[0].visuals
                   if v.visual_type == "slicer" and "slicer_" in v.visual_id]
        assert len(slicers) == 2
        assert slicers[0].y == slicers[1].y

    def test_tab_selector_present(self):
        layout = assemble(self._build())
        navs = [v for v in layout.pages[0].visuals if "nav_" in v.visual_id]
        assert len(navs) == 1
        assert navs[0].visual_type == "slicer"

    def test_ordering_title_slicers_nav_content(self):
        layout = assemble(self._build())
        p = layout.pages[0]
        title = next(v for v in p.visuals if v.visual_id == "title_card")
        slicers = [v for v in p.visuals if "slicer_" in v.visual_id]
        nav = next(v for v in p.visuals if "nav_" in v.visual_id)
        content = [v for v in p.visuals if v.visual_id in ("v1", "v2", "v3")]

        # Title above slicers
        assert title.y < slicers[0].y
        # Slicers above nav
        assert slicers[0].y + slicers[0].height <= nav.y + 1
        # Nav above content
        for c in content:
            assert nav.y + nav.height <= c.y + 1

    def test_three_content_visuals_stacked(self):
        layout = assemble(self._build())
        content = sorted(
            [v for v in layout.pages[0].visuals if v.visual_id in ("v1", "v2", "v3")],
            key=lambda v: v.y,
        )
        assert len(content) == 3
        # All same width
        assert content[0].width == content[1].width == content[2].width
        # Vertically ordered
        assert content[0].y < content[1].y < content[2].y
        # Full width
        assert content[0].width == CANVAS_W - 2 * MARGIN

    def test_no_overflow(self):
        layout = assemble(self._build())
        for v in layout.pages[0].visuals:
            assert v.x >= 0
            assert v.y >= 0
            assert v.x + v.width <= CANVAS_W + 1

    def test_report_layout_format(self):
        layout = assemble(self._build())
        rl = layout.to_report_layout()
        assert len(rl["sections"]) == 1
        section = rl["sections"][0]
        assert section["displayName"] == "Dash Overall Transactions"
        # At least: title + 2 slicers + 1 nav + 3 visuals = 7
        assert len(section["visualContainers"]) >= 7


# ================================================================== #
#  Run                                                                  #
# ================================================================== #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
