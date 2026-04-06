"""
Tests for the Calculation Validation Framework.

Verifies that Tableau formula evaluator, DAX measure evaluator,
and comparison engine produce correct results and handle edge cases
gracefully (nulls, empty strings, division by zero, unparseable formulas).
"""
import math
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from core.calc_validator import (
    evaluate_tableau_formula,
    evaluate_dax_measure,
    validate_calculations,
    extract_dax_measures_from_config,
    write_calculation_audit,
    _resolve_column,
    _strip_brackets,
    _split_args,
    _match_tableau_to_dax,
    _make_serializable,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small DataFrame for testing aggregation and string functions."""
    return pd.DataFrame({
        "Sales": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Quantity": [1, 2, 3, 4, 5],
        "Region": ["East", "West", "East", "West", "East"],
        "Category": ["A", "B", "A", "B", "C"],
        "Date": pd.to_datetime([
            "2024-01-15", "2024-02-20", "2024-03-10",
            "2024-04-05", "2024-05-25",
        ]),
        "Name": ["  Alice  ", "Bob", "Charlie", "Diana", "Eve"],
        "Profit": [10.5, -5.0, 30.0, 0.0, 15.5],
    })


@pytest.fixture
def df_with_nulls():
    """DataFrame with null values for null-handling tests."""
    return pd.DataFrame({
        "Sales": [100.0, None, 300.0, None, 500.0],
        "Region": ["East", None, "East", "West", None],
        "Value": [1.0, 2.0, np.nan, 4.0, 5.0],
    })


@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame({"Sales": pd.Series(dtype="float64")})


# ---------------------------------------------------------------------------
#  Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_strip_brackets(self):
        assert _strip_brackets("[Sales]") == "Sales"
        assert _strip_brackets("Sales") == "Sales"
        assert _strip_brackets("[My Field]") == "My Field"
        assert _strip_brackets("") == ""

    def test_resolve_column_case_insensitive(self, sample_df):
        assert _resolve_column(sample_df, "sales") == "Sales"
        assert _resolve_column(sample_df, "SALES") == "Sales"
        assert _resolve_column(sample_df, "Sales") == "Sales"
        assert _resolve_column(sample_df, "nonexistent") is None

    def test_split_args_simple(self):
        assert _split_args("a, b, c") == ["a", "b", "c"]

    def test_split_args_nested(self):
        result = _split_args("SUM([Sales]), ROUND(AVG([Qty]), 2)")
        assert len(result) == 2
        assert "SUM([Sales])" in result[0]
        assert "ROUND" in result[1]


# ---------------------------------------------------------------------------
#  Tableau Formula Evaluator
# ---------------------------------------------------------------------------

class TestTableauAggregations:
    def test_sum(self, sample_df):
        result = evaluate_tableau_formula("SUM([Sales])", sample_df)
        assert result == pytest.approx(1500.0)

    def test_avg(self, sample_df):
        result = evaluate_tableau_formula("AVG([Sales])", sample_df)
        assert result == pytest.approx(300.0)

    def test_count(self, sample_df):
        result = evaluate_tableau_formula("COUNT([Sales])", sample_df)
        assert result == pytest.approx(5.0)

    def test_countd(self, sample_df):
        result = evaluate_tableau_formula("COUNTD([Region])", sample_df)
        assert result == pytest.approx(2.0)

    def test_min(self, sample_df):
        result = evaluate_tableau_formula("MIN([Sales])", sample_df)
        assert result == pytest.approx(100.0)

    def test_max(self, sample_df):
        result = evaluate_tableau_formula("MAX([Sales])", sample_df)
        assert result == pytest.approx(500.0)

    def test_median(self, sample_df):
        result = evaluate_tableau_formula("MEDIAN([Sales])", sample_df)
        assert result == pytest.approx(300.0)

    def test_case_insensitive_function(self, sample_df):
        result = evaluate_tableau_formula("sum([Sales])", sample_df)
        assert result == pytest.approx(1500.0)

    def test_unknown_column_returns_none(self, sample_df):
        result = evaluate_tableau_formula("SUM([NonExistent])", sample_df)
        assert result is None


class TestTableauMath:
    def test_abs(self, sample_df):
        result = evaluate_tableau_formula("ABS(-42)", sample_df)
        assert result == pytest.approx(42.0)

    def test_round(self, sample_df):
        result = evaluate_tableau_formula("ROUND(3.14159, 2)", sample_df)
        assert result == pytest.approx(3.14)

    def test_ceiling(self, sample_df):
        result = evaluate_tableau_formula("CEILING(3.2)", sample_df)
        assert result == pytest.approx(4.0)

    def test_floor(self, sample_df):
        result = evaluate_tableau_formula("FLOOR(3.8)", sample_df)
        assert result == pytest.approx(3.0)

    def test_power(self, sample_df):
        result = evaluate_tableau_formula("POWER(2, 10)", sample_df)
        assert result == pytest.approx(1024.0)

    def test_sqrt(self, sample_df):
        result = evaluate_tableau_formula("SQRT(144)", sample_df)
        assert result == pytest.approx(12.0)

    def test_ln(self, sample_df):
        result = evaluate_tableau_formula("LN(1)", sample_df)
        assert result == pytest.approx(0.0)

    def test_log(self, sample_df):
        result = evaluate_tableau_formula("LOG(100, 10)", sample_df)
        assert result == pytest.approx(2.0)


class TestTableauString:
    def test_len(self, sample_df):
        result = evaluate_tableau_formula("LEN([Region])", sample_df)
        # "East"=4, "West"=4, "East"=4, "West"=4, "East"=4 => sum = 20
        assert result == pytest.approx(20.0)

    def test_upper(self, sample_df):
        result = evaluate_tableau_formula("UPPER([Region])", sample_df)
        assert isinstance(result, list)
        assert result[0] == "EAST"

    def test_lower(self, sample_df):
        result = evaluate_tableau_formula("LOWER([Region])", sample_df)
        assert isinstance(result, list)
        assert result[0] == "east"

    def test_trim(self, sample_df):
        result = evaluate_tableau_formula("TRIM([Name])", sample_df)
        assert isinstance(result, list)
        assert result[0] == "Alice"

    def test_left(self, sample_df):
        result = evaluate_tableau_formula("LEFT([Region], 2)", sample_df)
        assert isinstance(result, list)
        assert result[0] == "Ea"

    def test_right(self, sample_df):
        result = evaluate_tableau_formula("RIGHT([Region], 2)", sample_df)
        assert isinstance(result, list)
        assert result[0] == "st"

    def test_mid(self, sample_df):
        result = evaluate_tableau_formula("MID([Region], 2, 2)", sample_df)
        assert isinstance(result, list)
        assert result[0] == "as"

    def test_contains(self, sample_df):
        result = evaluate_tableau_formula("CONTAINS([Region], 'East')", sample_df)
        assert result == pytest.approx(3.0)  # 3 rows contain "East"


class TestTableauDate:
    def test_year(self, sample_df):
        result = evaluate_tableau_formula("YEAR([Date])", sample_df)
        assert result == pytest.approx(2024.0)

    def test_month(self, sample_df):
        # Mode of months: 1,2,3,4,5 -- all unique, mode returns first
        result = evaluate_tableau_formula("MONTH([Date])", sample_df)
        assert result is not None

    def test_day(self, sample_df):
        result = evaluate_tableau_formula("DAY([Date])", sample_df)
        assert result is not None

    def test_datepart_year(self, sample_df):
        result = evaluate_tableau_formula("DATEPART('year', [Date])", sample_df)
        assert result == pytest.approx(2024.0)

    def test_datepart_quarter(self, sample_df):
        result = evaluate_tableau_formula("DATEPART('quarter', [Date])", sample_df)
        assert result is not None

    def test_datetrunc_returns_none(self, sample_df):
        # DATETRUNC returns date series, not numeric -- should return None
        result = evaluate_tableau_formula("DATETRUNC('month', [Date])", sample_df)
        assert result is None

    def test_datediff_returns_none(self, sample_df):
        result = evaluate_tableau_formula(
            "DATEDIFF('day', [Date], TODAY())", sample_df
        )
        assert result is None


class TestTableauLogic:
    def test_zn_with_nulls(self, df_with_nulls):
        result = evaluate_tableau_formula("ZN([Sales])", df_with_nulls)
        # 100 + 0 + 300 + 0 + 500 = 900
        assert result == pytest.approx(900.0)

    def test_ifnull(self, df_with_nulls):
        result = evaluate_tableau_formula("IFNULL([Value], 0)", df_with_nulls)
        # 1 + 2 + 0 + 4 + 5 = 12
        assert result == pytest.approx(12.0)

    def test_isnull(self, df_with_nulls):
        result = evaluate_tableau_formula("ISNULL([Sales])", df_with_nulls)
        # 2 nulls
        assert result == pytest.approx(2.0)

    def test_if_then_else(self, sample_df):
        result = evaluate_tableau_formula(
            "IF [Region] = 'East' THEN SUM([Sales]) ELSE 0 END",
            sample_df,
        )
        # Evaluates THEN branch: SUM of all Sales = 1500
        assert result == pytest.approx(1500.0)

    def test_iif_returns_none(self, sample_df):
        result = evaluate_tableau_formula(
            "IIF([Sales] > 200, 1, 0)", sample_df
        )
        # IIF is too complex for general parsing
        assert result is None

    def test_case_returns_none(self, sample_df):
        result = evaluate_tableau_formula(
            "CASE [Region] WHEN 'East' THEN 1 WHEN 'West' THEN 2 END",
            sample_df,
        )
        assert result is None


class TestTableauLOD:
    def test_fixed_sum(self, sample_df):
        # { FIXED [Region] : SUM([Sales]) }
        # East: 100+300+500=900, West: 200+400=600
        # Total: 900+600=1500
        result = evaluate_tableau_formula(
            "{ FIXED [Region] : SUM([Sales]) }",
            sample_df,
        )
        assert result == pytest.approx(1500.0)

    def test_fixed_avg(self, sample_df):
        # { FIXED [Region] : AVG([Sales]) }
        # East: avg=300, West: avg=300
        # Total: 300+300=600
        result = evaluate_tableau_formula(
            "{ FIXED [Region] : AVG([Sales]) }",
            sample_df,
        )
        assert result == pytest.approx(600.0)

    def test_fixed_countd(self, sample_df):
        # { FIXED [Region] : COUNTD([Category]) }
        # East: A,A,C -> 2 unique, West: B,B -> 1 unique
        # Total: 2+1=3
        result = evaluate_tableau_formula(
            "{ FIXED [Region] : COUNTD([Category]) }",
            sample_df,
        )
        assert result == pytest.approx(3.0)

    def test_fixed_unknown_column(self, sample_df):
        result = evaluate_tableau_formula(
            "{ FIXED [Region] : SUM([NoSuchField]) }",
            sample_df,
        )
        assert result is None

    def test_include_returns_none(self, sample_df):
        result = evaluate_tableau_formula(
            "{ INCLUDE [Category] : SUM([Sales]) }",
            sample_df,
        )
        assert result is None

    def test_exclude_returns_none(self, sample_df):
        result = evaluate_tableau_formula(
            "{ EXCLUDE [Category] : SUM([Sales]) }",
            sample_df,
        )
        assert result is None


# ---------------------------------------------------------------------------
#  DAX Measure Evaluator
# ---------------------------------------------------------------------------

class TestDAXAggregations:
    def test_sum(self, sample_df):
        result = evaluate_dax_measure("SUM(Table[Sales])", sample_df)
        assert result == pytest.approx(1500.0)

    def test_average(self, sample_df):
        result = evaluate_dax_measure("AVERAGE(Table[Sales])", sample_df)
        assert result == pytest.approx(300.0)

    def test_count(self, sample_df):
        result = evaluate_dax_measure("COUNT(Table[Sales])", sample_df)
        assert result == pytest.approx(5.0)

    def test_countrows(self, sample_df):
        result = evaluate_dax_measure("COUNTROWS(Table)", sample_df)
        assert result == pytest.approx(5.0)

    def test_distinctcount(self, sample_df):
        result = evaluate_dax_measure("DISTINCTCOUNT(Table[Region])", sample_df)
        assert result == pytest.approx(2.0)

    def test_min(self, sample_df):
        result = evaluate_dax_measure("MIN(Table[Sales])", sample_df)
        assert result == pytest.approx(100.0)

    def test_max(self, sample_df):
        result = evaluate_dax_measure("MAX(Table[Sales])", sample_df)
        assert result == pytest.approx(500.0)

    def test_bracket_only_ref(self, sample_df):
        result = evaluate_dax_measure("SUM([Sales])", sample_df)
        assert result == pytest.approx(1500.0)


class TestDAXMath:
    def test_abs(self, sample_df):
        result = evaluate_dax_measure("ABS(-42)", sample_df)
        assert result == pytest.approx(42.0)

    def test_round(self, sample_df):
        result = evaluate_dax_measure("ROUND(3.14159, 2)", sample_df)
        assert result == pytest.approx(3.14)

    def test_divide(self, sample_df):
        result = evaluate_dax_measure(
            "DIVIDE(SUM(Table[Sales]), COUNT(Table[Sales]))", sample_df
        )
        assert result == pytest.approx(300.0)

    def test_divide_by_zero(self, sample_df):
        result = evaluate_dax_measure("DIVIDE(100, 0)", sample_df)
        assert result == pytest.approx(0.0)

    def test_divide_by_zero_with_alt(self, sample_df):
        result = evaluate_dax_measure("DIVIDE(100, 0, -1)", sample_df)
        assert result == pytest.approx(-1.0)


class TestDAXLogic:
    def test_if(self, sample_df):
        result = evaluate_dax_measure(
            "IF(SUM(Table[Sales]) > 0, SUM(Table[Sales]), 0)", sample_df
        )
        # Evaluates THEN branch
        assert result == pytest.approx(1500.0)

    def test_blank(self, sample_df):
        result = evaluate_dax_measure("BLANK()", sample_df)
        assert result == pytest.approx(0.0)

    def test_switch_returns_none(self, sample_df):
        result = evaluate_dax_measure(
            "SWITCH(TRUE(), [Sales] > 100, 1, 0)", sample_df
        )
        assert result is None

    def test_var_return_returns_none(self, sample_df):
        result = evaluate_dax_measure(
            "VAR x = SUM(Table[Sales]) RETURN x / 2", sample_df
        )
        assert result is None


class TestDAXCalculate:
    def test_simple_calculate(self, sample_df):
        result = evaluate_dax_measure(
            "CALCULATE(SUM(Table[Sales]))", sample_df
        )
        assert result == pytest.approx(1500.0)

    def test_calculate_with_filter_ignores_filter(self, sample_df):
        result = evaluate_dax_measure(
            "CALCULATE(SUM(Table[Sales]), Table[Region] = \"East\")", sample_df
        )
        # Filters are ignored -- returns total SUM
        assert result == pytest.approx(1500.0)


class TestDAXTimeIntelligence:
    def test_totalytd_returns_none(self, sample_df):
        result = evaluate_dax_measure(
            "TOTALYTD(SUM(Table[Sales]), Table[Date])", sample_df
        )
        assert result is None

    def test_sameperiodlastyear_returns_none(self, sample_df):
        result = evaluate_dax_measure(
            "CALCULATE(SUM(Table[Sales]), SAMEPERIODLASTYEAR(Table[Date]))",
            sample_df,
        )
        assert result is None


class TestDAXDate:
    def test_year(self, sample_df):
        result = evaluate_dax_measure("YEAR(Table[Date])", sample_df)
        assert result == pytest.approx(2024.0)


# ---------------------------------------------------------------------------
#  Comparison Engine
# ---------------------------------------------------------------------------

class TestValidateCalculations:
    def test_matching_calcs(self, sample_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total Sales", "formula": "SUM([Sales])"},
                {"name": "Avg Sales", "formula": "AVG([Sales])"},
            ]
        }
        dax_measures = {
            "Total Sales": "SUM(Data[Sales])",
            "Avg Sales": "AVERAGE(Data[Sales])",
        }
        result = validate_calculations(tableau_spec, dax_measures, sample_df)

        assert result["total_calcs"] == 2
        assert result["validated"] == 2
        assert result["mismatched"] == 0
        assert len(result["details"]) == 2
        assert all(d["match"] is True for d in result["details"])

    def test_mismatched_calcs(self, sample_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total Sales", "formula": "SUM([Sales])"},
            ]
        }
        # Deliberately wrong: using COUNT instead of SUM
        dax_measures = {
            "Total Sales": "COUNT(Data[Sales])",
        }
        result = validate_calculations(tableau_spec, dax_measures, sample_df)

        assert result["total_calcs"] == 1
        assert result["mismatched"] == 1
        assert result["details"][0]["match"] is False
        assert result["details"][0]["delta_pct"] > 0

    def test_skipped_tableau(self, sample_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Complex", "formula": "CASE [X] WHEN 'A' THEN 1 WHEN 'B' THEN 2 END"},
            ]
        }
        dax_measures = {
            "Complex": "SUM(Data[Sales])",
        }
        result = validate_calculations(tableau_spec, dax_measures, sample_df)

        assert result["skipped_tableau"] == 1

    def test_skipped_dax(self, sample_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total Sales", "formula": "SUM([Sales])"},
            ]
        }
        # No matching DAX measure
        dax_measures = {}
        result = validate_calculations(tableau_spec, dax_measures, sample_df)

        assert result["skipped_dax"] == 1

    def test_empty_df(self, empty_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total Sales", "formula": "SUM([Sales])"},
            ]
        }
        dax_measures = {"Total Sales": "SUM(Data[Sales])"}
        result = validate_calculations(tableau_spec, dax_measures, empty_df)
        assert result["skipped_tableau"] == 1

    def test_no_calculated_fields(self, sample_df):
        tableau_spec = {"calculated_fields": []}
        dax_measures = {}
        result = validate_calculations(tableau_spec, dax_measures, sample_df)
        assert result["total_calcs"] == 0
        assert result["validated"] == 0

    def test_measure_name_suffix_matching(self, sample_df):
        """Measures renamed with ' Measure' suffix should still match."""
        tableau_spec = {
            "calculated_fields": [
                {"name": "Sales", "formula": "SUM([Sales])"},
            ]
        }
        dax_measures = {
            "Sales Measure": "SUM(Data[Sales])",
        }
        result = validate_calculations(tableau_spec, dax_measures, sample_df)
        assert result["validated"] == 1

    def test_null_dataframe(self):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total", "formula": "SUM([Sales])"},
            ]
        }
        dax_measures = {"Total": "SUM(Data[Sales])"}
        result = validate_calculations(tableau_spec, dax_measures, None)
        assert result["skipped_tableau"] == 1


class TestMatchTableauToDax:
    def test_exact_match(self):
        dax = {"Total Sales": "SUM(T[S])"}
        name, formula = _match_tableau_to_dax("Total Sales", dax)
        assert name == "Total Sales"

    def test_case_insensitive(self):
        dax = {"total sales": "SUM(T[S])"}
        name, formula = _match_tableau_to_dax("Total Sales", dax)
        assert name == "total sales"

    def test_measure_suffix(self):
        dax = {"Sales Measure": "SUM(T[S])"}
        name, formula = _match_tableau_to_dax("Sales", dax)
        assert name == "Sales Measure"

    def test_no_match(self):
        dax = {"Revenue": "SUM(T[S])"}
        name, formula = _match_tableau_to_dax("Total Sales", dax)
        assert name is None


# ---------------------------------------------------------------------------
#  Config extraction
# ---------------------------------------------------------------------------

class TestExtractDaxMeasures:
    def test_extract_from_config(self):
        config = {
            "tmdl_model": {
                "tables": [
                    {
                        "name": "Data",
                        "measures": [
                            {"name": "Total Sales", "dax": "SUM(Data[Sales])"},
                            {"name": "Avg Qty", "expression": "AVERAGE(Data[Quantity])"},
                        ],
                    }
                ]
            }
        }
        measures = extract_dax_measures_from_config(config)
        assert "Total Sales" in measures
        assert measures["Total Sales"] == "SUM(Data[Sales])"
        assert "Avg Qty" in measures
        assert measures["Avg Qty"] == "AVERAGE(Data[Quantity])"

    def test_empty_config(self):
        measures = extract_dax_measures_from_config({})
        assert measures == {}


# ---------------------------------------------------------------------------
#  Audit file writing
# ---------------------------------------------------------------------------

class TestWriteAudit:
    def test_write_and_read(self, sample_df):
        tableau_spec = {
            "calculated_fields": [
                {"name": "Total Sales", "formula": "SUM([Sales])"},
            ]
        }
        dax_measures = {"Total Sales": "SUM(Data[Sales])"}
        audit = validate_calculations(tableau_spec, dax_measures, sample_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_calculation_audit(audit, tmpdir)
            assert os.path.isfile(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data["validated"] == 1
            assert data["details"][0]["match"] is True


# ---------------------------------------------------------------------------
#  Serialization edge cases
# ---------------------------------------------------------------------------

class TestMakeSerializable:
    def test_numpy_int(self):
        assert _make_serializable(np.int64(42)) == 42

    def test_numpy_float(self):
        assert _make_serializable(np.float64(3.14)) == pytest.approx(3.14)

    def test_nan_becomes_none(self):
        assert _make_serializable(float("nan")) is None

    def test_inf_becomes_none(self):
        assert _make_serializable(float("inf")) is None

    def test_nested_dict(self):
        data = {"a": np.int64(1), "b": [np.float64(2.0)]}
        result = _make_serializable(data)
        assert result == {"a": 1, "b": [2.0]}


# ---------------------------------------------------------------------------
#  Edge cases: unparseable formulas
# ---------------------------------------------------------------------------

class TestUnparseableFormulas:
    def test_empty_formula(self, sample_df):
        assert evaluate_tableau_formula("", sample_df) is None
        assert evaluate_tableau_formula(None, sample_df) is None

    def test_gibberish(self, sample_df):
        assert evaluate_tableau_formula("!@#$%^&*", sample_df) is None

    def test_deeply_nested_never_crashes(self, sample_df):
        formula = "IF THEN ELSE END " * 10
        result = evaluate_tableau_formula(formula, sample_df)
        # Should return None, not raise
        assert result is None

    def test_dax_empty(self, sample_df):
        assert evaluate_dax_measure("", sample_df) is None
        assert evaluate_dax_measure(None, sample_df) is None

    def test_dax_gibberish(self, sample_df):
        assert evaluate_dax_measure("!@#$%^&*", sample_df) is None

    def test_bare_field_numeric(self, sample_df):
        result = evaluate_tableau_formula("[Sales]", sample_df)
        assert result == pytest.approx(1500.0)

    def test_bare_field_string(self, sample_df):
        result = evaluate_tableau_formula("[Region]", sample_df)
        assert result is None  # Non-numeric

    def test_numeric_literal(self, sample_df):
        assert evaluate_tableau_formula("42", sample_df) == pytest.approx(42.0)

    def test_dax_numeric_literal(self, sample_df):
        assert evaluate_dax_measure("42", sample_df) == pytest.approx(42.0)

    def test_dax_leading_equals(self, sample_df):
        result = evaluate_dax_measure("= SUM(T[Sales])", sample_df)
        assert result == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
#  Division by zero
# ---------------------------------------------------------------------------

class TestDivisionByZero:
    def test_dax_divide_zero(self, sample_df):
        result = evaluate_dax_measure("DIVIDE(SUM(Data[Sales]), 0)", sample_df)
        assert result == pytest.approx(0.0)

    def test_dax_divide_zero_with_alt(self, sample_df):
        result = evaluate_dax_measure("DIVIDE(SUM(Data[Sales]), 0, -999)", sample_df)
        assert result == pytest.approx(-999.0)
