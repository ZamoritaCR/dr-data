"""
End-to-end pipeline tests for the Tableau-to-Power-BI conversion.

Tests the full chain:
  Stage 1: enhanced_tableau_parser.parse_twb() -- Parse .twb XML -> tableau_spec
  Stage 2: synthetic_data.extract_schema_from_tableau() -- Extract column schema
  Stage 3: synthetic_data.generate_from_tableau_spec() -- Generate synthetic DataFrame
  Stage 4: visual_intent.extract_migration_intent() -- Build structured intent
  Stage 5: visual_intent.format_intent_for_prompt() -- Format for LLM prompt
  Stage 6: Validate all column names are clean and consistent through pipeline

Note: Stages 4-6 (Claude/GPT API calls and PBIP file writing) are not tested here
because they require API keys. Those are integration tests, not unit tests.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.enhanced_tableau_parser import parse_twb
from core.synthetic_data import (
    extract_schema_from_tableau,
    generate_from_tableau_spec,
    generate_synthetic_dataframe,
    _clean_field_name,
)
from core.visual_intent import extract_migration_intent, format_intent_for_prompt


# ------------------------------------------------------------------ #
#  Test TWB XML fixtures                                               #
# ------------------------------------------------------------------ #

SIMPLE_TWB = """<?xml version='1.0'?>
<workbook version='2024.1'>
  <datasources>
    <datasource name='federated.abc123' caption='Sales Data'>
      <connection class='federated'>
        <relation table='[Sales]'/>
      </connection>
      <column name='[Region]' caption='Region' datatype='string' role='dimension'/>
      <column name='[Sales]' caption='Sales' datatype='real' role='measure'/>
      <column name='[Profit]' caption='Profit' datatype='real' role='measure'/>
      <column name='[Order Date]' caption='Order Date' datatype='date' role='dimension'/>
      <column name='[Category]' caption='Category' datatype='string' role='dimension'/>
      <column name='[Customer Name]' caption='Customer Name' datatype='string' role='dimension'/>
      <column name='[Discount]' caption='Discount' datatype='real' role='measure'/>
      <column name='[Quantity]' caption='Quantity' datatype='integer' role='measure'/>
      <column name='[:Measure Names]' datatype='string' role='dimension'/>
      <column name='[Number of Records]' caption='Number of Records' datatype='integer' role='measure'/>
      <column name='[Calculation_abc]' caption='Profit Ratio' datatype='real' role='measure'>
        <calculation formula='SUM([Profit])/SUM([Sales])'/>
      </column>
    </datasource>
  </datasources>
  <worksheets>
    <worksheet name='Sales Over Time'>
      <table>
        <pane><mark class='Line'/></pane>
        <rows>[federated.abc123].[sum:Sales:qk]</rows>
        <cols>[federated.abc123].[yr:Order Date:ok]</cols>
      </table>
      <filter class='categorical' column='[federated.abc123].[none:Region:nk]'/>
    </worksheet>
    <worksheet name='Category Breakdown'>
      <table>
        <pane><mark class='Bar'/></pane>
        <rows>[federated.abc123].[sum:Profit:qk]</rows>
        <cols>[federated.abc123].[none:Category:nk]</cols>
      </table>
    </worksheet>
    <worksheet name='Quarterly Trend'>
      <table>
        <pane><mark class='automatic'/></pane>
        <rows>[federated.abc123].[sum:Sales:qk]</rows>
        <cols>[federated.abc123].[tqr:Order Date:qk]</cols>
      </table>
    </worksheet>
    <worksheet name='Monthly Detail'>
      <table>
        <pane><mark class='Bar'/></pane>
        <rows>[federated.abc123].[avg:Discount:qk]</rows>
        <cols>[federated.abc123].[mn:Order Date:ok]</cols>
      </table>
    </worksheet>
    <worksheet name='Map View'>
      <table>
        <pane><mark class='map'/></pane>
        <rows>[federated.abc123].[Latitude (generated)]</rows>
        <cols>[federated.abc123].[Longitude (generated)]</cols>
      </table>
    </worksheet>
    <worksheet name='Quoted Fields'>
      <table>
        <pane><mark class='Bar'/></pane>
        <rows>[federated.abc123].[none:"Customer Name":nk]</rows>
        <cols>[federated.abc123].[sum:"Sales":qk]</cols>
      </table>
    </worksheet>
  </worksheets>
  <dashboards>
    <dashboard name='Executive Overview'>
      <size maxwidth='1200' maxheight='900'/>
      <zones>
        <zone name='Sales Over Time' type-v2='worksheet'
              x='10' y='10' w='580' h='400'/>
        <zone name='Category Breakdown' type-v2='worksheet'
              x='600' y='10' w='580' h='400'/>
        <zone name='Quarterly Trend' type-v2='worksheet'
              x='10' y='420' w='580' h='400'/>
        <zone name='Monthly Detail' type-v2='worksheet'
              x='600' y='420' w='580' h='400'/>
      </zones>
    </dashboard>
  </dashboards>
</workbook>"""

# TWB with edge cases: empty shelves, action filters, generated fields
EDGE_CASE_TWB = """<?xml version='1.0'?>
<workbook version='2024.1'>
  <datasources>
    <datasource name='federated.xyz' caption='Test Data'>
      <column name='[Amount]' datatype='real' role='measure'/>
      <column name='[Status]' datatype='string' role='dimension'/>
    </datasource>
  </datasources>
  <worksheets>
    <worksheet name='Empty Shelf'>
      <table>
        <pane><mark class='Bar'/></pane>
        <rows></rows>
        <cols></cols>
      </table>
    </worksheet>
    <worksheet name='Action Filters'>
      <table>
        <pane><mark class='Bar'/></pane>
        <rows>[federated.xyz].[sum:Amount:qk]</rows>
        <cols>[federated.xyz].[none:Status:nk]</cols>
      </table>
      <filter class='categorical' column='[federated.xyz].[Action (Region,Sub-Region)]'/>
      <filter class='categorical' column='[federated.xyz].[Tooltip (Info)]'/>
    </worksheet>
  </worksheets>
</workbook>"""


def _write_twb(content):
    """Write TWB XML to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".twb", delete=False)
    f.write(content)
    f.close()
    return f.name


# ------------------------------------------------------------------ #
#  Stage 1: Parser tests                                               #
# ------------------------------------------------------------------ #

class TestParserStage:

    def test_parse_returns_all_sections(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            assert spec["type"] == "tableau_workbook"
            assert len(spec["worksheets"]) == 6
            assert len(spec["dashboards"]) == 1
            assert len(spec["datasources"]) >= 1
            assert not spec.get("parse_error")
        finally:
            os.unlink(path)

    def test_chart_types_extracted(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            ws_map = {ws["name"]: ws for ws in spec["worksheets"]}
            assert ws_map["Sales Over Time"]["chart_type"] == "line"
            assert ws_map["Category Breakdown"]["chart_type"] == "bar"
            assert ws_map["Quarterly Trend"]["chart_type"] == "automatic"
            assert ws_map["Map View"]["chart_type"] == "map"
        finally:
            os.unlink(path)

    def test_filters_are_dicts(self):
        """Enhanced parser must produce filter dicts, not strings."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            ws = spec["worksheets"][0]  # Sales Over Time
            assert len(ws["filters"]) >= 1
            for f in ws["filters"]:
                assert isinstance(f, dict), f"Filter should be dict, got {type(f)}: {f}"
                assert "field" in f
                assert "type" in f
        finally:
            os.unlink(path)

    def test_calculated_fields_extracted(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            assert len(spec["calculated_fields"]) >= 1
            cf = spec["calculated_fields"][0]
            assert "formula" in cf
            assert "SUM" in cf["formula"]
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Stage 2: Schema extraction                                          #
# ------------------------------------------------------------------ #

class TestSchemaExtraction:

    def test_columns_are_clean(self):
        """No column name should be a single quote, empty, or contain raw Tableau refs."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            assert len(schema) > 0

            for col in schema:
                name = col["name"]
                # No empty names
                assert name, f"Empty column name in schema"
                # No single-char punctuation
                assert len(name) >= 2, f"Column name too short: {name!r}"
                # No raw Tableau refs
                assert not name.startswith("federated."), f"Raw federated ref: {name!r}"
                assert ":" not in name or len(name.split(":")) == 1, \
                    f"Unresolved Tableau prefix in column: {name!r}"
                # No surrounding quotes
                assert not (name.startswith('"') and name.endswith('"')), \
                    f"Unstripped quotes in column: {name!r}"
                # Has at least one alphanumeric char
                assert re.search(r'[a-zA-Z0-9]', name), \
                    f"No alphanumeric chars in column: {name!r}"
        finally:
            os.unlink(path)

    def test_no_internal_columns(self):
        """Tableau internal columns should be filtered out."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            names = [c["name"] for c in schema]

            # These should NOT appear
            assert "Number of Records" not in names
            assert "Measure Names" not in names
            assert ":Measure Names" not in names

            # Internal prefixes should not appear
            for name in names:
                assert not name.startswith("Calculation_"), f"Raw calc ID leaked: {name}"
                assert not name.startswith("Action ("), f"Action filter leaked: {name}"
        finally:
            os.unlink(path)

    def test_generated_fields_filtered(self):
        """Latitude/Longitude (generated) should be filtered out."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            names = [c["name"] for c in schema]
            assert "Latitude (generated)" not in names
            assert "Longitude (generated)" not in names
        finally:
            os.unlink(path)

    def test_shelf_prefixes_resolved(self):
        """Tableau shelf prefixes (sum:, none:, yr:, tqr:, mn:) should be stripped."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            names = [c["name"] for c in schema]

            # These prefixed forms should NOT appear
            for name in names:
                assert not re.match(r'^(sum|avg|none|yr|mn|tqr|qr|tmn|twk)\:', name, re.I), \
                    f"Unresolved shelf prefix in column: {name!r}"

            # The actual field names should appear
            assert "Sales" in names
            assert "Region" in names
            assert "Category" in names
            assert "Order Date" in names
        finally:
            os.unlink(path)

    def test_quoted_fields_cleaned(self):
        """Fields with quoted names in shelf expressions should be unquoted."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            names = [c["name"] for c in schema]

            # Customer Name appears as none:"Customer Name":nk in the shelf
            assert "Customer Name" in names
            # No double-quoted variants
            for name in names:
                assert name != '"Customer Name"', f"Quotes not stripped: {name!r}"
                assert name != '"', f"Single quote as column name"
        finally:
            os.unlink(path)

    def test_measures_detected(self):
        """Aggregated shelf fields should be tagged as measures."""
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            col_map = {c["name"]: c for c in schema}

            assert col_map["Sales"]["role"] == "measure"
            assert col_map["Profit"]["role"] == "measure"
            assert col_map["Region"]["role"] == "dimension"
            assert col_map["Category"]["role"] == "dimension"
        finally:
            os.unlink(path)

    def test_edge_case_empty_shelves(self):
        """Worksheets with empty shelves should not crash."""
        path = _write_twb(EDGE_CASE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            # Should still get columns from datasource definition
            names = [c["name"] for c in schema]
            assert "Amount" in names
            assert "Status" in names
        finally:
            os.unlink(path)

    def test_edge_case_action_filters_excluded(self):
        """Action filters should be excluded from schema."""
        path = _write_twb(EDGE_CASE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            names = [c["name"] for c in schema]
            for name in names:
                assert not name.startswith("Action ("), f"Action filter in schema: {name}"
                assert not name.startswith("Tooltip ("), f"Tooltip filter in schema: {name}"
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Stage 3: Synthetic data generation                                  #
# ------------------------------------------------------------------ #

class TestSyntheticDataGeneration:

    def test_generates_correct_columns(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            df, csv_path, schema = generate_from_tableau_spec(spec, num_rows=50)

            # DataFrame should have same columns as schema
            schema_names = {c["name"] for c in schema}
            df_names = set(df.columns)
            assert schema_names == df_names, \
                f"Mismatch: schema={schema_names} vs df={df_names}"

            assert len(df) == 50
        finally:
            os.unlink(path)

    def test_no_garbage_column_names(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            df, _, _ = generate_from_tableau_spec(spec, num_rows=20)

            for col in df.columns:
                assert col, "Empty column name"
                assert len(col) >= 2, f"Column name too short: {col!r}"
                assert col != '"', f"Single double-quote as column name"
                assert re.search(r'[a-zA-Z0-9]', col), f"No alnum in: {col!r}"
        finally:
            os.unlink(path)

    def test_date_columns_use_relative_dates(self):
        """Date columns should use relative dates (not hardcoded to old range)."""
        from datetime import datetime
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            df, _, _ = generate_from_tableau_spec(spec, num_rows=100)

            # Find date column
            date_cols = [c for c in df.columns if "date" in c.lower()]
            assert len(date_cols) >= 1

            for dc in date_cols:
                dates = pd.to_datetime(df[dc])
                max_date = dates.max()
                # Should be within 2 years of now, not hardcoded to 2023-2025
                now = datetime.now()
                assert max_date.year >= now.year - 2, \
                    f"Date column {dc} has max year {max_date.year}, expected >= {now.year - 2}"
        finally:
            os.unlink(path)

    def test_measure_columns_are_numeric(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            schema = extract_schema_from_tableau(spec)
            df, _, _ = generate_from_tableau_spec(spec, num_rows=50)

            for col_info in schema:
                if col_info["role"] == "measure":
                    name = col_info["name"]
                    assert pd.api.types.is_numeric_dtype(df[name]), \
                        f"Measure column {name!r} is not numeric: {df[name].dtype}"
        finally:
            os.unlink(path)

    def test_csv_output_roundtrip(self):
        """Generated CSV should be loadable and have the same columns."""
        path = _write_twb(SIMPLE_TWB)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                spec = parse_twb(path)
                df, csv_path, schema = generate_from_tableau_spec(
                    spec, num_rows=50, output_dir=tmpdir
                )
                assert csv_path
                assert os.path.exists(csv_path)

                # Reload and verify
                df2 = pd.read_csv(csv_path)
                assert list(df2.columns) == list(df.columns)
                assert len(df2) == len(df)
        finally:
            os.unlink(path)

    def test_fallback_on_empty_spec(self):
        """Empty spec should produce a fallback DataFrame, not crash."""
        empty_spec = {
            "type": "tableau_workbook",
            "datasources": [],
            "worksheets": [],
            "dashboards": [],
            "calculated_fields": [],
            "parameters": [],
        }
        df, _, schema = generate_from_tableau_spec(empty_spec, num_rows=10)
        assert len(df) == 10
        assert len(df.columns) >= 1


# ------------------------------------------------------------------ #
#  Stage 4: Visual intent extraction                                   #
# ------------------------------------------------------------------ #

class TestVisualIntentPipeline:

    def test_intent_from_parsed_spec(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            intent = extract_migration_intent(spec, "convert to power bi")

            # Should have 1 dashboard page + orphan worksheets
            assert len(intent.pages) >= 1
            # Executive Overview has 4 worksheets
            page = intent.pages[0]
            assert page.display_name == "Executive Overview"
            assert len(page.visuals) == 4

            # Map View and Quoted Fields should be orphans (not on dashboard)
            assert len(intent.orphan_worksheets) >= 1
        finally:
            os.unlink(path)

    def test_prompt_contains_all_visuals(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            intent = extract_migration_intent(spec, "convert to power bi")
            prompt = format_intent_for_prompt(intent, "SalesData")

            assert "Sales Over Time" in prompt
            assert "Category Breakdown" in prompt
            assert "lineChart" in prompt
            assert "clusteredBarChart" in prompt
        finally:
            os.unlink(path)

    def test_filters_become_slicers(self):
        path = _write_twb(SIMPLE_TWB)
        try:
            spec = parse_twb(path)
            intent = extract_migration_intent(spec, "convert")
            # Sales Over Time has a Region filter
            page = intent.pages[0]
            slicer_fields = [s["field"] for s in page.slicers]
            # The filter field comes from the parser as something like
            # "none:Region:nk" -> field = "none:Region:nk" or "Region"
            # At least one slicer should be present
            assert len(page.slicers) >= 1
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Column name sanitization (unit tests)                               #
# ------------------------------------------------------------------ #

class TestCleanFieldName:

    def test_strips_double_quotes(self):
        assert _clean_field_name('"Region"') == "Region"

    def test_strips_single_quotes(self):
        assert _clean_field_name("'Region'") == "Region"

    def test_empty_after_strip(self):
        assert _clean_field_name('""') == ""

    def test_single_quote_char(self):
        assert _clean_field_name('"') == ""

    def test_punctuation_only(self):
        assert _clean_field_name("...") == ""
        assert _clean_field_name(":::") == ""
        assert _clean_field_name("   ") == ""

    def test_preserves_normal_names(self):
        assert _clean_field_name("Sales Amount") == "Sales Amount"
        assert _clean_field_name("Order Date") == "Order Date"

    def test_preserves_names_with_special_chars(self):
        assert _clean_field_name("Sub-Category") == "Sub-Category"
        assert _clean_field_name("Ship Mode") == "Ship Mode"

    def test_strips_whitespace(self):
        assert _clean_field_name("  Region  ") == "Region"


# ------------------------------------------------------------------ #
#  Real TWBX file (if available)                                       #
# ------------------------------------------------------------------ #

TWBX_PATH = PROJECT_ROOT / "output" / "Digital_Ads_Sales_Performance_Dashboard_pbip" / "DigitalAds-Sales-Data.twbx"


@pytest.mark.skipif(not TWBX_PATH.exists(), reason="TWBX test file not available")
class TestRealTwbxFile:

    def test_parse_real_twbx(self):
        spec = parse_twb(str(TWBX_PATH))
        assert spec["type"] == "tableau_workbook"
        assert not spec.get("parse_error")
        assert len(spec["worksheets"]) > 0

    def test_schema_from_real_twbx(self):
        spec = parse_twb(str(TWBX_PATH))
        schema = extract_schema_from_tableau(spec)
        assert len(schema) > 0

        for col in schema:
            name = col["name"]
            assert name, "Empty column"
            assert len(name) >= 2, f"Short: {name!r}"
            assert name != '"'
            assert re.search(r'[a-zA-Z0-9]', name)
            # No Tableau internal prefixes
            assert not re.match(r'^(sum|avg|none|yr|mn|tqr|qr|tmn|twk)\:', name, re.I)
            assert not name.endswith("(generated)")
            assert not name.startswith("Action (")

    def test_synthetic_data_from_real_twbx(self):
        spec = parse_twb(str(TWBX_PATH))
        df, _, schema = generate_from_tableau_spec(spec, num_rows=100)

        assert len(df) == 100
        assert len(df.columns) == len(schema)

        for col in df.columns:
            assert col and len(col) >= 2
            assert col != '"'
            assert re.search(r'[a-zA-Z0-9]', col)

    def test_intent_from_real_twbx(self):
        spec = parse_twb(str(TWBX_PATH))
        intent = extract_migration_intent(spec, "convert to power bi")
        assert len(intent.pages) >= 1 or len(intent.orphan_worksheets) >= 1


# ------------------------------------------------------------------ #
#  Run with pytest or standalone                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
