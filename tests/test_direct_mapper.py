"""
Tests for core.direct_mapper -- Direct Tableau-to-PBIP deterministic mapping.

Verifies that the mapper produces the correct PBI config structure
from Tableau specs without any AI calls.
"""

import json
import pytest

from core.direct_mapper import (
    _clean_field_name,
    _clean_shelf_fields,
    _resolve_field_against_profile,
    _classify_fields_for_chart,
    _build_measures_from_spec,
    build_pbip_config_from_tableau,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_tableau_spec():
    """Minimal Tableau spec with one dashboard and two worksheets."""
    return {
        "type": "tableau_workbook",
        "version": "2024.1",
        "worksheets": [
            {
                "name": "Sales by Region",
                "chart_type": "bar",
                "mark_type": "bar",
                "rows": "[none:Region:nk]",
                "cols": "SUM([Sales])",
                "rows_fields": ["none:Region:nk"],
                "cols_fields": ["sum:Sales:qk"],
                "dimensions": [],
                "measures": [],
                "filters": [],
                "color_field": "",
                "size_field": "",
                "label_fields": [],
                "tooltip_fields": [],
                "sort_field": "",
                "design": {
                    "mark_colors": [{"color": "#FF0000", "value": ""}],
                    "background_color": "",
                    "title_font": {},
                    "axis_config": [],
                    "mark_style": {},
                    "border": {},
                },
            },
            {
                "name": "Trend Over Time",
                "chart_type": "line",
                "mark_type": "line",
                "rows": "SUM([Revenue])",
                "cols": "[tqr:Date:qk]",
                "rows_fields": ["sum:Revenue:qk"],
                "cols_fields": ["tqr:Date:qk"],
                "dimensions": [],
                "measures": [],
                "filters": [],
                "color_field": "[none:Category:nk]",
                "size_field": "",
                "label_fields": [],
                "tooltip_fields": [],
                "sort_field": "",
                "design": {
                    "mark_colors": [],
                    "background_color": "",
                    "title_font": {},
                    "axis_config": [],
                    "mark_style": {},
                    "border": {},
                },
            },
        ],
        "dashboards": [
            {
                "name": "Sales Dashboard",
                "size": {"width": "1200", "height": "800"},
                "canvas": {"width": 1200, "height": 800},
                "worksheets_used": ["Sales by Region", "Trend Over Time"],
                "zones": [
                    {
                        "name": "Sales by Region",
                        "type": "worksheet",
                        "layout": {"x": 0, "y": 0, "w": 600, "h": 400},
                    },
                    {
                        "name": "Trend Over Time",
                        "type": "worksheet",
                        "layout": {"x": 600, "y": 0, "w": 600, "h": 400},
                    },
                ],
            }
        ],
        "calculated_fields": [
            {
                "name": "Total Revenue",
                "formula": "SUM([Revenue])",
                "datasource": "test",
            },
            {
                "name": "Profit Margin",
                "formula": "SUM([Revenue]) - SUM([Cost])",
                "datasource": "test",
            },
        ],
        "design": {
            "color_palettes": [
                {
                    "name": "test_palette",
                    "type": "regular",
                    "colors": ["#FF0000", "#00FF00", "#0000FF"],
                }
            ],
            "global_fonts": {},
        },
        "datasources": [],
        "parameters": [],
        "filters": [],
        "relationships": [],
        "has_hyper": False,
    }


@pytest.fixture
def sample_data_profile():
    """Minimal data profile matching the Tableau spec."""
    return {
        "table_name": "TestData",
        "columns": [
            {"name": "Region", "dtype": "object", "semantic_type": "dimension", "unique_count": 4},
            {"name": "Date", "dtype": "datetime64[ns]", "semantic_type": "date", "unique_count": 50},
            {"name": "Sales", "dtype": "float64", "semantic_type": "measure", "unique_count": 80},
            {"name": "Revenue", "dtype": "float64", "semantic_type": "measure", "unique_count": 90},
            {"name": "Cost", "dtype": "float64", "semantic_type": "measure", "unique_count": 85},
            {"name": "Category", "dtype": "object", "semantic_type": "dimension", "unique_count": 5},
        ],
    }


# ------------------------------------------------------------------ #
#  Field name cleaning tests                                           #
# ------------------------------------------------------------------ #

class TestCleanFieldName:

    def test_plain_name(self):
        assert _clean_field_name("Region") == "Region"

    def test_brackets(self):
        assert _clean_field_name("[Region]") == "Region"

    def test_shelf_prefix_none(self):
        assert _clean_field_name("none:Region:nk") == "Region"

    def test_shelf_prefix_sum(self):
        assert _clean_field_name("sum:Sales:qk") == "Sales"

    def test_shelf_prefix_yr(self):
        assert _clean_field_name("yr:Date:ok") == "Date"

    def test_shelf_prefix_tqr(self):
        assert _clean_field_name("tqr:Date:qk") == "Date"

    def test_federated_skip(self):
        assert _clean_field_name("federated.abc123") == ""

    def test_colon_prefix_skip(self):
        assert _clean_field_name(":Measure Names") == ""

    def test_quoted_name(self):
        assert _clean_field_name('"Region"') == "Region"

    def test_empty(self):
        assert _clean_field_name("") == ""

    def test_whitespace_only(self):
        assert _clean_field_name("   ") == ""


class TestCleanShelfFields:

    def test_basic(self):
        result = _clean_shelf_fields(["none:Region:nk", "sum:Sales:qk"])
        assert result == ["Region", "Sales"]

    def test_dedup(self):
        result = _clean_shelf_fields(["Region", "Region"])
        assert result == ["Region"]

    def test_skip_federated(self):
        result = _clean_shelf_fields(["federated.abc", "Region"])
        assert result == ["Region"]


class TestResolveFieldAgainstProfile:

    def test_exact_match(self):
        cols = {"Region", "Sales", "Date"}
        assert _resolve_field_against_profile("Region", cols) == "Region"

    def test_case_insensitive(self):
        cols = {"Region", "Sales"}
        assert _resolve_field_against_profile("region", cols) == "Region"

    def test_partial_match(self):
        cols = {"Customer Type", "Sales"}
        result = _resolve_field_against_profile("Customer", cols)
        assert result == "Customer Type"

    def test_no_match(self):
        cols = {"Region", "Sales"}
        assert _resolve_field_against_profile("Unknown", cols) == "Unknown"


# ------------------------------------------------------------------ #
#  Field classification tests                                          #
# ------------------------------------------------------------------ #

class TestClassifyFieldsForChart:

    def test_bar_chart(self):
        ws = {
            "rows_fields": ["none:Region:nk"],
            "cols_fields": ["sum:Sales:qk"],
            "color_field": "",
        }
        profile_cols = {"Region", "Sales"}
        col_types = {"Region": "dimension", "Sales": "measure"}
        result = _classify_fields_for_chart(ws, "clusteredBarChart", profile_cols, col_types)
        assert "Region" in result["category"]
        assert "Sales" in result["values"]

    def test_line_chart_with_series(self):
        ws = {
            "rows_fields": ["sum:Revenue:qk"],
            "cols_fields": ["tqr:Date:qk"],
            "color_field": "[none:Category:nk]",
        }
        profile_cols = {"Date", "Revenue", "Category"}
        col_types = {"Date": "date", "Revenue": "measure", "Category": "dimension"}
        result = _classify_fields_for_chart(ws, "lineChart", profile_cols, col_types)
        assert "Date" in result["category"]
        assert "Revenue" in result["values"]
        assert "Category" in result["series"]

    def test_card(self):
        ws = {
            "rows_fields": [],
            "cols_fields": ["sum:Sales:qk"],
            "color_field": "",
        }
        profile_cols = {"Sales"}
        col_types = {"Sales": "measure"}
        result = _classify_fields_for_chart(ws, "card", profile_cols, col_types)
        assert "Sales" in result["values"]
        assert result["category"] == []

    def test_table(self):
        ws = {
            "rows_fields": ["none:Region:nk"],
            "cols_fields": ["sum:Sales:qk"],
            "color_field": "",
        }
        profile_cols = {"Region", "Sales"}
        col_types = {"Region": "dimension", "Sales": "measure"}
        result = _classify_fields_for_chart(ws, "tableEx", profile_cols, col_types)
        # Tables put all fields in values
        assert "Region" in result["values"]
        assert "Sales" in result["values"]


# ------------------------------------------------------------------ #
#  Measure generation tests                                            #
# ------------------------------------------------------------------ #

class TestBuildMeasures:

    def test_simple_sum(self):
        spec = {
            "calculated_fields": [
                {"name": "Total Revenue", "formula": "SUM([Revenue])", "datasource": "ds"},
            ],
            "worksheets": [],
        }
        measures = _build_measures_from_spec(spec, "Data", {"Revenue", "Sales"})
        assert any(m["name"] == "Total Revenue" for m in measures)
        match = [m for m in measures if m["name"] == "Total Revenue"][0]
        assert "SUM" in match["dax"]
        assert match["needs_ai"] is False

    def test_simple_countd(self):
        spec = {
            "calculated_fields": [
                {"name": "Unique Customers", "formula": "COUNTD([CustomerID])", "datasource": "ds"},
            ],
            "worksheets": [],
        }
        measures = _build_measures_from_spec(spec, "Data", {"CustomerID"})
        match = [m for m in measures if m["name"] == "Unique Customers"][0]
        assert "DISTINCTCOUNT" in match["dax"]
        assert match["needs_ai"] is False

    def test_complex_formula_flagged(self):
        spec = {
            "calculated_fields": [
                {
                    "name": "Profit Margin",
                    "formula": "SUM([Revenue]) - SUM([Cost])",
                    "datasource": "ds",
                },
            ],
            "worksheets": [],
        }
        measures = _build_measures_from_spec(spec, "Data", {"Revenue", "Cost"})
        match = [m for m in measures if m["name"] == "Profit Margin"][0]
        assert match["needs_ai"] is True

    def test_implicit_measures_from_shelves(self):
        spec = {
            "calculated_fields": [],
            "worksheets": [
                {
                    "rows": "[none:Region:nk]",
                    "cols": "SUM([Sales])",
                    "rows_fields": ["none:Region:nk"],
                    "cols_fields": ["sum:Sales:qk"],
                    "filters": [],
                },
            ],
        }
        measures = _build_measures_from_spec(spec, "Data", {"Region", "Sales"})
        # Should create a SUM measure for Sales
        assert any("Sales" in m["name"] for m in measures)


# ------------------------------------------------------------------ #
#  Main function (build_pbip_config_from_tableau) tests                #
# ------------------------------------------------------------------ #

class TestBuildPbipConfig:

    def test_returns_tuple(self, sample_tableau_spec, sample_data_profile):
        result = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_config_structure(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        assert "report_layout" in config
        assert "tmdl_model" in config
        assert "sections" in config["report_layout"]
        assert "tables" in config["tmdl_model"]

    def test_page_count_matches_dashboards(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        sections = config["report_layout"]["sections"]
        # 1 dashboard page (no orphan worksheets since both are used)
        assert len(sections) == 1

    def test_visual_count_matches_zones(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        section = config["report_layout"]["sections"][0]
        # 2 zones in the dashboard
        assert len(section["visualContainers"]) == 2

    def test_visual_types_from_marks(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        section = config["report_layout"]["sections"][0]
        types = set()
        for vc in section["visualContainers"]:
            cfg = vc["config"]
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            types.add(cfg["visualType"])
        # bar -> clusteredBarChart, line -> lineChart
        assert "clusteredBarChart" in types
        assert "lineChart" in types

    def test_positions_are_scaled(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        section = config["report_layout"]["sections"][0]
        for vc in section["visualContainers"]:
            assert "x" in vc
            assert "y" in vc
            assert "width" in vc
            assert "height" in vc
            # Positions should be within PBI canvas (1280x720)
            assert 0 <= vc["x"] <= 1280
            assert 0 <= vc["y"] <= 720
            assert vc["width"] > 0
            assert vc["height"] > 0

    def test_measures_generated(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        tables = config["tmdl_model"]["tables"]
        assert len(tables) >= 1
        measures = tables[0].get("measures", [])
        assert len(measures) > 0
        for m in measures:
            assert "name" in m
            assert "dax" in m
            assert "format" in m

    def test_dashboard_spec_structure(self, sample_tableau_spec, sample_data_profile):
        _, dash_spec = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        assert "dashboard_title" in dash_spec
        assert "pages" in dash_spec
        assert "measures" in dash_spec
        assert "design" in dash_spec
        assert "worksheet_designs" in dash_spec
        assert "worksheet_chart_types" in dash_spec
        assert "source" in dash_spec
        assert dash_spec["source"] == "direct_tableau_mapper"

    def test_dashboard_title(self, sample_tableau_spec, sample_data_profile):
        _, dash_spec = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        assert dash_spec["dashboard_title"] == "Sales Dashboard"

    def test_color_theme_extracted(self, sample_tableau_spec, sample_data_profile):
        _, dash_spec = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        theme = dash_spec.get("theme", {})
        colors = theme.get("dataColors", [])
        assert len(colors) > 0

    def test_visual_data_roles(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        section = config["report_layout"]["sections"][0]
        for vc in section["visualContainers"]:
            cfg = vc["config"]
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            roles = cfg.get("dataRoles", {})
            assert isinstance(roles, dict)
            # At least one role should have fields
            all_fields = roles.get("category", []) + roles.get("values", [])
            assert len(all_fields) > 0

    def test_page_dimensions(self, sample_tableau_spec, sample_data_profile):
        config, _ = build_pbip_config_from_tableau(
            sample_tableau_spec, sample_data_profile, "TestData"
        )
        for section in config["report_layout"]["sections"]:
            assert section["width"] == 1280
            assert section["height"] == 720


class TestOrphanWorksheets:

    def test_orphans_get_own_page(self, sample_data_profile):
        """Worksheets not in any dashboard get their own page."""
        spec = {
            "worksheets": [
                {
                    "name": "Orphan 1",
                    "chart_type": "bar",
                    "rows_fields": ["none:Region:nk"],
                    "cols_fields": ["sum:Sales:qk"],
                    "color_field": "",
                    "filters": [],
                    "design": {},
                },
            ],
            "dashboards": [
                {
                    "name": "Dashboard",
                    "canvas": {"width": 1000, "height": 800},
                    "worksheets_used": [],
                    "zones": [],
                }
            ],
            "calculated_fields": [],
            "design": {},
        }
        config, _ = build_pbip_config_from_tableau(spec, sample_data_profile, "TestData")
        sections = config["report_layout"]["sections"]
        # 1 dashboard page + 1 orphan page
        assert len(sections) == 2
        orphan_section = sections[1]
        assert orphan_section["displayName"] == "Additional Worksheets"
        assert len(orphan_section["visualContainers"]) == 1


class TestNoDashboards:

    def test_no_dashboards_all_orphan(self, sample_data_profile):
        """When there are no dashboards, all worksheets go on one orphan page."""
        spec = {
            "worksheets": [
                {
                    "name": "WS1",
                    "chart_type": "line",
                    "rows_fields": ["sum:Revenue:qk"],
                    "cols_fields": ["tqr:Date:qk"],
                    "color_field": "",
                    "filters": [],
                    "design": {},
                },
            ],
            "dashboards": [],
            "calculated_fields": [],
            "design": {},
        }
        config, _ = build_pbip_config_from_tableau(spec, sample_data_profile, "TestData")
        sections = config["report_layout"]["sections"]
        assert len(sections) == 1
        assert sections[0]["displayName"] == "WS1"


class TestEmptySpec:

    def test_empty_worksheets_and_dashboards(self, sample_data_profile):
        spec = {
            "worksheets": [],
            "dashboards": [],
            "calculated_fields": [],
            "design": {},
        }
        config, dash_spec = build_pbip_config_from_tableau(spec, sample_data_profile, "TestData")
        # Should still produce at least one empty page
        assert len(config["report_layout"]["sections"]) >= 1
        assert config["tmdl_model"]["tables"][0]["name"] == "TestData"
