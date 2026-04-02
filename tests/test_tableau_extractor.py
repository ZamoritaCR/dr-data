"""
Tests for core/tableau_extractor.py -- comprehensive Tableau extraction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.tableau_extractor import (
    extract_workbook,
    workbook_to_dict,
    parse_column_instance,
    parse_shelf_expression,
    Confidence,
    ColumnRef,
)


# ------------------------------------------------------------------ #
#  Column reference parsing                                            #
# ------------------------------------------------------------------ #

class TestColumnParsing:

    def test_sum_measure(self):
        ref = parse_column_instance("sum:Sales:qk")
        assert ref.name == "Sales"
        assert ref.aggregation == "Sum"
        assert ref.role == "measure"
        assert ref.tableau_type == "quantitative"

    def test_none_dimension(self):
        ref = parse_column_instance("none:Category:nk")
        assert ref.name == "Category"
        assert ref.aggregation == "None"
        assert ref.role == "dimension"
        assert ref.tableau_type == "nominal"

    def test_year_date(self):
        ref = parse_column_instance("yr:Order Date:ok")
        assert ref.name == "Order Date"
        assert ref.aggregation == "Year"
        assert ref.tableau_type == "ordinal"

    def test_countd(self):
        ref = parse_column_instance("ctd:Region:qk")
        assert ref.name == "Region"
        assert ref.aggregation == "CountD"
        assert ref.role == "measure"

    def test_attribute(self):
        ref = parse_column_instance("attr:MyField:nk")
        assert ref.name == "MyField"
        assert ref.aggregation == "Attribute"

    def test_fallback_plain_name(self):
        ref = parse_column_instance("SomeField", column_attr="[SomeField]",
                                     derivation_attr="Sum", type_attr="quantitative")
        assert ref.name == "SomeField"
        assert ref.aggregation == "Sum"


class TestShelfParsing:

    def test_simple_expression(self):
        refs = parse_shelf_expression("[DS].[none:Category:nk]")
        assert len(refs) == 1
        assert refs[0].name == "Category"

    def test_compound_expression(self):
        refs = parse_shelf_expression(
            "([DS].[none:Cat:nk] / [DS].[none:SubCat:nk])"
        )
        assert len(refs) == 2
        assert refs[0].name == "Cat"
        assert refs[1].name == "SubCat"

    def test_mixed_roles(self):
        refs = parse_shelf_expression(
            "([DS].[none:Segment:nk] * [DS].[sum:Sales:qk])"
        )
        assert len(refs) == 2
        assert refs[0].role == "dimension"
        assert refs[1].role == "measure"

    def test_empty_expression(self):
        assert parse_shelf_expression("") == []
        assert parse_shelf_expression(None) == []


# ------------------------------------------------------------------ #
#  TWB extraction from XML strings                                     #
# ------------------------------------------------------------------ #

_MINIMAL_TWB = """<?xml version='1.0'?>
<workbook version='2024.1'>
  <datasources>
    <datasource name='Sample'>
      <column name='[Category]' caption='Category' datatype='string' role='dimension' type='nominal'/>
      <column name='[Sales]' datatype='real' role='measure' type='quantitative'/>
      <column name='[Profit]' datatype='real' role='measure' type='quantitative'/>
      <column name='[Order Date]' datatype='date' role='dimension' type='ordinal'/>
      <column name='[Region]' datatype='string' role='dimension' type='nominal'/>
      <column name='[Calc1]' caption='Profit Ratio' datatype='real' role='measure'>
        <calculation class='tableau' formula='SUM([Profit])/SUM([Sales])'/>
      </column>
    </datasource>
    <datasource name='Parameters'>
      <column name='[Top N]' caption='Top N' datatype='integer' param-domain-type='range' value='10'>
        <range min='1' max='50' granularity='1'/>
      </column>
    </datasource>
  </datasources>
  <worksheets>
    <worksheet name='Sales by Category'>
      <table>
        <view>
          <datasource-dependencies datasource='Sample'>
            <column name='[Category]' datatype='string' role='dimension' type='nominal'/>
            <column name='[Sales]' datatype='real' role='measure' type='quantitative'/>
            <column-instance column='[Category]' derivation='None' name='[none:Category:nk]' type='nominal'/>
            <column-instance column='[Sales]' derivation='Sum' name='[sum:Sales:qk]' type='quantitative'/>
          </datasource-dependencies>
          <shelf-sorts>
            <shelf-sort-v2 dimension-to-sort='[Sample].[none:Category:nk]'
                           measure-to-sort-by='[Sample].[sum:Sales:qk]'
                           direction='DESC' shelf='rows'/>
          </shelf-sorts>
          <slices>
            <column>[Sample].[none:Region:nk]</column>
          </slices>
        </view>
        <style>
          <style-rule element='quick-filter'>
            <format attr='title' field='[Sample].[none:Region:nk]' value='Region'/>
          </style-rule>
        </style>
        <panes>
          <pane>
            <mark class='Bar'/>
            <encodings>
              <color column='[Sample].[sum:Profit:qk]'/>
            </encodings>
          </pane>
        </panes>
        <rows>[Sample].[none:Category:nk]</rows>
        <cols>[Sample].[sum:Sales:qk]</cols>
      </table>
    </worksheet>
    <worksheet name='KPI Card'>
      <table>
        <view>
          <datasource-dependencies datasource='Sample'>
            <column name='[Sales]' datatype='real' role='measure' type='quantitative'/>
            <column-instance column='[Sales]' derivation='Sum' name='[sum:Sales:qk]' type='quantitative'/>
          </datasource-dependencies>
        </view>
        <panes>
          <pane>
            <mark class='Text'/>
          </pane>
        </panes>
        <rows></rows>
        <cols></cols>
      </table>
    </worksheet>
    <worksheet name='Geo Map'>
      <table>
        <panes>
          <pane>
            <mark class='Polygon'/>
          </pane>
        </panes>
        <rows></rows>
        <cols></cols>
      </table>
    </worksheet>
  </worksheets>
  <dashboards>
    <dashboard name='Overview'>
      <size maxwidth='1200' maxheight='800' sizing-mode='fixed'/>
      <zones>
        <zone id='1' type-v2='layout-basic' x='0' y='0' w='100000' h='100000'>
          <zone id='2' type-v2='layout-flow' param='vert' x='1000' y='1000' w='98000' h='98000'>
            <zone id='3' type-v2='title' x='1000' y='1000' w='98000' h='5000'/>
            <zone id='4' type-v2='layout-flow' param='horz' x='1000' y='6000' w='98000' h='12000'
                  layout-strategy-id='distribute-evenly'>
              <zone name='KPI Card' id='5' x='1000' y='6000' w='49000' h='12000'/>
              <zone name='Geo Map' id='6' x='50000' y='6000' w='49000' h='12000' show-title='false'/>
            </zone>
            <zone name='Sales by Category' id='7' x='1000' y='18000' w='80000' h='80000'/>
            <zone name='Sales by Category' id='8' type-v2='filter' x='81000' y='18000' w='18000' h='20000'
                  param='[Sample].[none:Region:nk]' mode='checkdropdown'/>
          </zone>
        </zone>
      </zones>
    </dashboard>
  </dashboards>
</workbook>"""


class TestWorkbookExtraction:

    def _extract(self):
        return extract_workbook(_MINIMAL_TWB)

    def test_version(self):
        wb = self._extract()
        assert wb.version == "2024.1"

    def test_datasources(self):
        wb = self._extract()
        assert len(wb.datasources) == 1
        ds = wb.datasources[0]
        assert ds.name == "Sample"
        assert len(ds.columns) >= 5

    def test_calculated_fields(self):
        wb = self._extract()
        assert len(wb.calculated_fields) == 1
        cf = wb.calculated_fields[0]
        assert cf.name == "Profit Ratio"
        assert "SUM([Profit])" in cf.formula
        assert "Profit" in cf.referenced_fields

    def test_parameters(self):
        wb = self._extract()
        assert len(wb.parameters) == 1
        p = wb.parameters[0]
        assert p.name == "Top N"
        assert p.domain_type == "range"
        assert p.min_value == "1"
        assert p.max_value == "50"

    def test_worksheet_count(self):
        wb = self._extract()
        assert len(wb.worksheets) == 3

    def test_bar_chart_detection(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert vs.mark_type == "bar"
        assert vs.mark_type_pbi == "clusteredBarChart"
        assert vs.confidence == Confidence.EXACT

    def test_shelf_encoding(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert len(vs.encoding.shelf_rows) >= 1
        assert vs.encoding.shelf_rows[0].name == "Category"
        assert len(vs.encoding.shelf_cols) >= 1
        assert vs.encoding.shelf_cols[0].name == "Sales"

    def test_color_encoding(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert vs.encoding.color is not None
        assert vs.encoding.color.name == "Profit"

    def test_sort_extraction(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert len(vs.sorts) == 1
        assert vs.sorts[0].direction == "DESC"

    def test_slice_extraction(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert len(vs.slices) >= 1

    def test_kpi_card_detection(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "KPI Card")
        assert vs.is_kpi_card is True
        assert vs.mark_type_pbi == "card"

    def test_map_detection(self):
        wb = self._extract()
        vs = next(v for v in wb.worksheets if v.name == "Geo Map")
        assert vs.is_map is True
        assert vs.mark_type_pbi == "map"


class TestDashboardExtraction:

    def _extract(self):
        return extract_workbook(_MINIMAL_TWB)

    def test_dashboard_count(self):
        wb = self._extract()
        assert len(wb.dashboards) == 1

    def test_dashboard_size(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert db.width == 1200
        assert db.height == 800
        assert db.sizing_mode == "fixed"

    def test_worksheet_zones(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert len(db.worksheet_zones) >= 2
        ws_names = [z.name for z in db.worksheet_zones]
        assert "Sales by Category" in ws_names
        assert "KPI Card" in ws_names

    def test_filter_zones(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert len(db.filter_zones) >= 1
        fz = db.filter_zones[0]
        assert fz.filter_mode == "checkdropdown"
        assert "Region" in fz.param

    def test_title_zone(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert db.title_zone is not None

    def test_pbi_coordinates(self):
        wb = self._extract()
        db = wb.dashboards[0]
        for wz in db.worksheet_zones:
            assert wz.pbi_w > 0
            assert wz.pbi_h > 0
            assert wz.pbi_x + wz.pbi_w <= 1300  # within canvas

    def test_layout_pattern(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert db.has_vertical_stack is True

    def test_kpi_row_detection(self):
        wb = self._extract()
        db = wb.dashboards[0]
        assert db.has_kpi_row is True


class TestConfidenceScoring:

    def test_exact_for_bar(self):
        wb = extract_workbook(_MINIMAL_TWB)
        vs = next(v for v in wb.worksheets if v.name == "Sales by Category")
        assert vs.confidence == Confidence.EXACT

    def test_high_for_map(self):
        wb = extract_workbook(_MINIMAL_TWB)
        vs = next(v for v in wb.worksheets if v.name == "Geo Map")
        assert vs.confidence == Confidence.HIGH

    def test_kpi_high_confidence(self):
        wb = extract_workbook(_MINIMAL_TWB)
        vs = next(v for v in wb.worksheets if v.name == "KPI Card")
        assert vs.confidence == Confidence.HIGH


class TestSerialization:

    def test_to_dict(self):
        wb = extract_workbook(_MINIMAL_TWB)
        d = workbook_to_dict(wb)
        assert isinstance(d, dict)
        assert "worksheets" in d
        assert "dashboards" in d
        assert len(d["worksheets"]) == 3

    def test_confidence_serialized_as_string(self):
        wb = extract_workbook(_MINIMAL_TWB)
        d = workbook_to_dict(wb)
        for ws in d["worksheets"]:
            assert isinstance(ws["confidence"], str)
            assert ws["confidence"] in ("exact", "high_confidence", "approximate",
                                        "unknown_manual_review")


class TestSummaryStats:

    def test_totals(self):
        wb = extract_workbook(_MINIMAL_TWB)
        assert wb.total_visuals == 3
        assert wb.total_measures >= 2
        assert wb.total_dimensions >= 3


# ------------------------------------------------------------------ #
#  Run                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
