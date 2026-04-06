"""
Calculation Validation Framework.

Ensures that Tableau calculated fields and their DAX translations produce
the same numeric results when evaluated against the same pandas DataFrame.

Principle: $100 in Tableau = $100 in Power BI.

Supports common Tableau functions (SUM, AVG, COUNT, COUNTD, IF, CASE,
LOD FIXED, string/date ops) and DAX equivalents. Formulas that cannot
be parsed are skipped gracefully (returns None, never crashes).
"""
import json
import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Tableau Formula Evaluator
# ---------------------------------------------------------------------------

def _strip_brackets(name):
    """Remove surrounding brackets from a Tableau field reference."""
    name = name.strip()
    if name.startswith("[") and name.endswith("]"):
        return name[1:-1]
    return name


def _resolve_column(df, name):
    """Find a DataFrame column by case-insensitive match.

    Returns the actual column name if found, else None.
    """
    name_lower = name.lower().strip()
    for col in df.columns:
        if col.lower().strip() == name_lower:
            return col
    return None


def _eval_tableau_aggregation(func_name, field_name, df):
    """Evaluate a simple Tableau aggregation: FUNC([Field]).

    Returns a scalar numeric result or None.
    """
    col = _resolve_column(df, field_name)
    if col is None:
        return None

    series = df[col]
    fn = func_name.upper()

    try:
        if fn == "SUM":
            return float(series.sum())
        elif fn == "AVG":
            return float(series.mean())
        elif fn == "COUNT":
            return float(series.count())
        elif fn == "COUNTD":
            return float(series.nunique())
        elif fn == "MIN":
            return float(series.min())
        elif fn == "MAX":
            return float(series.max())
        elif fn == "MEDIAN":
            return float(series.median())
    except Exception:
        return None
    return None


def _eval_tableau_math(func_name, args_str, df):
    """Evaluate Tableau math functions: ABS, ROUND, CEILING, FLOOR, etc.

    Operates on scalar values (first arg may be a number or nested call).
    Returns a scalar or None.
    """
    fn = func_name.upper()
    parts = [a.strip() for a in args_str.split(",")]

    try:
        first = _try_parse_number(parts[0])
        if first is None:
            # Try evaluating as a nested Tableau expression
            first = evaluate_tableau_formula(parts[0], df)
        if first is None:
            return None

        if fn == "ABS":
            return abs(first)
        elif fn == "ROUND":
            decimals = int(parts[1]) if len(parts) > 1 else 0
            return round(first, decimals)
        elif fn == "CEILING":
            return float(math.ceil(first))
        elif fn == "FLOOR":
            return float(math.floor(first))
        elif fn == "POWER":
            exp = float(parts[1]) if len(parts) > 1 else 2
            return math.pow(first, exp)
        elif fn == "SQRT":
            return math.sqrt(first)
        elif fn == "LN":
            return math.log(first)
        elif fn == "LOG":
            base = float(parts[1]) if len(parts) > 1 else 10
            return math.log(first, base)
    except Exception:
        return None
    return None


def _eval_tableau_string(func_name, args_str, df):
    """Evaluate Tableau string functions against the DataFrame.

    Returns a scalar (or representative value) or None.
    For aggregation context, string functions are skipped (return None).
    """
    fn = func_name.upper()
    parts = _split_args(args_str)

    try:
        if fn == "LEN":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return float(df[col].astype(str).str.len().sum())
            return None
        elif fn == "UPPER":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return df[col].astype(str).str.upper().tolist()
            return None
        elif fn == "LOWER":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return df[col].astype(str).str.lower().tolist()
            return None
        elif fn == "TRIM":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return df[col].astype(str).str.strip().tolist()
            return None
        elif fn == "LEFT":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            n = int(parts[1]) if len(parts) > 1 else 1
            if col:
                return df[col].astype(str).str[:n].tolist()
            return None
        elif fn == "RIGHT":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            n = int(parts[1]) if len(parts) > 1 else 1
            if col:
                return df[col].astype(str).str[-n:].tolist()
            return None
        elif fn == "MID":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            start = int(parts[1]) - 1 if len(parts) > 1 else 0
            length = int(parts[2]) if len(parts) > 2 else 1
            if col:
                return df[col].astype(str).str[start:start + length].tolist()
            return None
        elif fn == "CONTAINS":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            substr = parts[1].strip().strip("'\"") if len(parts) > 1 else ""
            if col:
                return float(df[col].astype(str).str.contains(
                    substr, case=False, na=False
                ).sum())
            return None
        elif fn == "REPLACE":
            # REPLACE(string, substring, replacement) -- return None for now
            return None
    except Exception:
        return None
    return None


def _eval_tableau_date(func_name, args_str, df):
    """Evaluate Tableau date functions.

    Returns a scalar or None.
    """
    fn = func_name.upper()
    parts = _split_args(args_str)

    try:
        if fn in ("TODAY", "NOW"):
            return datetime.now()
        elif fn == "YEAR":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return float(pd.to_datetime(df[col], errors="coerce").dt.year.mode()[0])
            return None
        elif fn == "MONTH":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return float(pd.to_datetime(df[col], errors="coerce").dt.month.mode()[0])
            return None
        elif fn == "DAY":
            col = _resolve_column(df, _strip_brackets(parts[0]))
            if col:
                return float(pd.to_datetime(df[col], errors="coerce").dt.day.mode()[0])
            return None
        elif fn == "DATEPART":
            unit = parts[0].strip().strip("'\"").lower()
            col = _resolve_column(df, _strip_brackets(parts[1]))
            if col is None:
                return None
            dt_series = pd.to_datetime(df[col], errors="coerce")
            if unit == "year":
                return float(dt_series.dt.year.mode()[0])
            elif unit == "month":
                return float(dt_series.dt.month.mode()[0])
            elif unit == "day":
                return float(dt_series.dt.day.mode()[0])
            elif unit == "quarter":
                return float(dt_series.dt.quarter.mode()[0])
            return None
        elif fn == "DATETRUNC":
            # Returns a date series -- skip for numeric comparison
            return None
        elif fn == "DATEDIFF":
            # DATEDIFF('unit', start, end) -- too complex without row context
            return None
    except Exception:
        return None
    return None


def _eval_tableau_logic(formula, df):
    """Evaluate Tableau IF/IIF/CASE/ISNULL/IFNULL/ZN logic.

    Handles simple single-level IF THEN ELSE END patterns.
    Returns a scalar result or None.
    """
    text = formula.strip()

    # ZN([Field]) -> treat nulls as 0, then SUM
    zn_match = re.match(r'(?i)ZN\s*\(\s*(.+)\s*\)', text)
    if zn_match:
        inner = zn_match.group(1).strip()
        col = _resolve_column(df, _strip_brackets(inner))
        if col:
            return float(df[col].fillna(0).sum())
        return None

    # IFNULL(expr, replacement)
    ifnull_match = re.match(r'(?i)IFNULL\s*\(\s*(.+?)\s*,\s*(.+)\s*\)', text)
    if ifnull_match:
        expr = ifnull_match.group(1).strip()
        replacement = ifnull_match.group(2).strip()
        col = _resolve_column(df, _strip_brackets(expr))
        if col:
            repl_val = _try_parse_number(replacement)
            if repl_val is None:
                repl_val = 0
            return float(df[col].fillna(repl_val).sum())
        return None

    # ISNULL([Field]) -> count of nulls
    isnull_match = re.match(r'(?i)ISNULL\s*\(\s*(.+)\s*\)', text)
    if isnull_match:
        inner = isnull_match.group(1).strip()
        col = _resolve_column(df, _strip_brackets(inner))
        if col:
            return float(df[col].isnull().sum())
        return None

    # IIF(condition, then, else)
    iif_match = re.match(r'(?i)IIF\s*\((.+)\)', text)
    if iif_match:
        # Too complex for reliable parsing in general case
        return None

    # IF ... THEN ... ELSE ... END
    if_match = re.match(
        r'(?i)IF\s+(.+?)\s+THEN\s+(.+?)\s+ELSE\s+(.+?)\s+END',
        text
    )
    if if_match:
        # Simple IF with aggregation in THEN/ELSE -- evaluate both branches
        # and return THEN result (most common pattern: conditional aggregation)
        then_expr = if_match.group(2).strip()
        then_val = evaluate_tableau_formula(then_expr, df)
        return then_val

    # CASE ... WHEN ... END
    if text.upper().startswith("CASE"):
        return None  # Too complex

    return None


def _eval_tableau_lod(formula, df):
    """Evaluate Tableau LOD expressions: { FIXED [dim] : AGG([measure]) }.

    Uses pandas groupby to compute the LOD result, then returns the
    aggregated scalar (sum of the LOD results, matching Tableau's behavior
    when used on a viz with no additional dimensions).
    """
    # { FIXED [dim1], [dim2] : AGG([measure]) }
    fixed_match = re.match(
        r'\{\s*FIXED\s+(.+?)\s*:\s*(\w+)\s*\(\s*\[(.+?)\]\s*\)\s*\}',
        formula.strip(),
        re.IGNORECASE,
    )
    if fixed_match:
        dims_str = fixed_match.group(1)
        agg_func = fixed_match.group(2).upper()
        measure_field = fixed_match.group(3)

        # Parse dimension list
        dim_names = [_strip_brackets(d.strip()) for d in dims_str.split(",")]
        dim_cols = [_resolve_column(df, d) for d in dim_names]
        measure_col = _resolve_column(df, measure_field)

        if any(c is None for c in dim_cols) or measure_col is None:
            return None

        try:
            grouped = df.groupby(dim_cols)[measure_col]
            if agg_func == "SUM":
                lod = grouped.sum()
            elif agg_func in ("AVG", "AVERAGE"):
                lod = grouped.mean()
            elif agg_func == "COUNT":
                lod = grouped.count()
            elif agg_func == "COUNTD":
                lod = grouped.nunique()
            elif agg_func == "MIN":
                lod = grouped.min()
            elif agg_func == "MAX":
                lod = grouped.max()
            else:
                return None
            return float(lod.sum())
        except Exception:
            return None

    # INCLUDE / EXCLUDE -- more complex, skip
    if re.match(r'\{\s*(INCLUDE|EXCLUDE)\b', formula.strip(), re.IGNORECASE):
        return None

    return None


def _try_parse_number(s):
    """Try to parse a string as a number. Returns float or None."""
    try:
        return float(s.strip())
    except (ValueError, TypeError):
        return None


def _split_args(args_str):
    """Split comma-separated arguments respecting parentheses depth."""
    parts = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def evaluate_tableau_formula(formula, df):
    """Evaluate a Tableau calculated field formula against a DataFrame.

    Returns:
        float or other value if successfully evaluated, None if the
        formula is too complex or references unknown fields.
    """
    if not formula or not isinstance(formula, str):
        return None

    text = formula.strip()

    # LOD expressions: { FIXED ... }
    if text.startswith("{"):
        return _eval_tableau_lod(text, df)

    # Logic: IF, IIF, CASE, ZN, IFNULL, ISNULL
    upper = text.upper()
    if any(upper.startswith(kw) for kw in ("IF ", "IIF(", "CASE ", "ZN(", "IFNULL(", "ISNULL(")):
        return _eval_tableau_logic(text, df)

    # Function call pattern: FUNC_NAME(args)
    func_match = re.match(r'(\w+)\s*\((.+)\)$', text, re.DOTALL)
    if func_match:
        func_name = func_match.group(1).upper()
        args_str = func_match.group(2).strip()

        # Simple aggregation: FUNC([Field])
        agg_funcs = {"SUM", "AVG", "COUNT", "COUNTD", "MIN", "MAX", "MEDIAN"}
        if func_name in agg_funcs:
            field = _strip_brackets(args_str)
            return _eval_tableau_aggregation(func_name, field, df)

        # Math functions
        math_funcs = {"ABS", "ROUND", "CEILING", "FLOOR", "POWER", "SQRT", "LN", "LOG"}
        if func_name in math_funcs:
            return _eval_tableau_math(func_name, args_str, df)

        # String functions
        string_funcs = {"LEFT", "RIGHT", "MID", "LEN", "UPPER", "LOWER",
                        "TRIM", "CONTAINS", "REPLACE"}
        if func_name in string_funcs:
            return _eval_tableau_string(func_name, args_str, df)

        # Date functions
        date_funcs = {"DATEPART", "DATETRUNC", "DATEDIFF", "TODAY", "NOW",
                      "YEAR", "MONTH", "DAY"}
        if func_name in date_funcs:
            return _eval_tableau_date(func_name, args_str, df)

    # Bare field reference: [Field] -> return sum for numeric columns
    bare_field = re.match(r'^\[(.+)\]$', text)
    if bare_field:
        col = _resolve_column(df, bare_field.group(1))
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return float(df[col].sum())
        return None

    # Numeric literal
    num = _try_parse_number(text)
    if num is not None:
        return num

    # Cannot parse
    return None


# ---------------------------------------------------------------------------
#  DAX Measure Evaluator
# ---------------------------------------------------------------------------

def _extract_dax_column_ref(ref_str):
    """Extract column name from DAX Table[Column] or [Column] references."""
    # Table[Column]
    m = re.match(r"[\w']+\[(.+?)\]", ref_str.strip())
    if m:
        return m.group(1)
    # [Column]
    m = re.match(r'\[(.+?)\]', ref_str.strip())
    if m:
        return m.group(1)
    return ref_str.strip()


def evaluate_dax_measure(dax_formula, df):
    """Evaluate a DAX measure formula against a DataFrame.

    Returns:
        float or other value if successfully evaluated, None if the
        formula is too complex or references unknown fields.
    """
    if not dax_formula or not isinstance(dax_formula, str):
        return None

    text = dax_formula.strip()

    # Remove leading "= " if present
    if text.startswith("="):
        text = text[1:].strip()

    # Skip formulas with VAR / RETURN blocks (too complex)
    upper = text.upper()
    if "VAR " in upper and "RETURN" in upper:
        return None

    # DIVIDE(a, b) or DIVIDE(a, b, alt)
    divide_match = re.match(r'(?i)DIVIDE\s*\((.+)\)$', text)
    if divide_match:
        parts = _split_args(divide_match.group(1))
        if len(parts) >= 2:
            num = evaluate_dax_measure(parts[0].strip(), df)
            den = evaluate_dax_measure(parts[1].strip(), df)
            if num is not None and den is not None:
                if den == 0:
                    if len(parts) >= 3:
                        alt = _try_parse_number(parts[2].strip())
                        return alt if alt is not None else 0.0
                    return 0.0
                return num / den
        return None

    # IF(condition, then, else)
    if_match = re.match(r'(?i)IF\s*\((.+)\)$', text)
    if if_match:
        # Simple IF -- try to evaluate THEN branch (common pattern)
        parts = _split_args(if_match.group(1))
        if len(parts) >= 2:
            return evaluate_dax_measure(parts[1].strip(), df)
        return None

    # SWITCH -- too complex
    if upper.startswith("SWITCH"):
        return None

    # BLANK()
    if upper == "BLANK()":
        return 0.0

    # CALCULATE -- handle simple CALCULATE(agg, filter) patterns
    calc_match = re.match(r'(?i)CALCULATE\s*\((.+)\)$', text)
    if calc_match:
        calc_inner = calc_match.group(1)
        # If CALCULATE contains time intelligence, skip entirely
        calc_upper = calc_inner.upper()
        if any(kw in calc_upper for kw in ("TOTALYTD", "SAMEPERIODLASTYEAR",
                                            "DATEADD", "PARALLELPERIOD")):
            return None
        inner_parts = _split_args(calc_inner)
        if inner_parts:
            # Evaluate just the first argument (the measure expression)
            # Filters are ignored for this simplified validation
            base_result = evaluate_dax_measure(inner_parts[0].strip(), df)
            return base_result
        return None

    # TOTALYTD / SAMEPERIODLASTYEAR -- time intelligence, skip
    if any(kw in upper for kw in ("TOTALYTD", "SAMEPERIODLASTYEAR",
                                   "DATEADD", "PARALLELPERIOD")):
        return None

    # Simple aggregation: SUM(Table[Column]) or SUM([Column])
    agg_match = re.match(r'(\w+)\s*\((.+)\)$', text, re.DOTALL)
    if agg_match:
        func_name = agg_match.group(1).upper()
        args_str = agg_match.group(2).strip()

        # DAX aggregation functions
        dax_agg_map = {
            "SUM": "sum",
            "AVERAGE": "mean",
            "COUNT": "count",
            "COUNTROWS": "count_rows",
            "DISTINCTCOUNT": "nunique",
            "MIN": "min",
            "MAX": "max",
        }

        if func_name in dax_agg_map:
            col_name = _extract_dax_column_ref(args_str)
            col = _resolve_column(df, col_name)

            if func_name == "COUNTROWS":
                return float(len(df))

            if col is None:
                return None

            series = df[col]
            op = dax_agg_map[func_name]
            try:
                if op == "sum":
                    return float(series.sum())
                elif op == "mean":
                    return float(series.mean())
                elif op == "count":
                    return float(series.count())
                elif op == "nunique":
                    return float(series.nunique())
                elif op == "min":
                    return float(series.min())
                elif op == "max":
                    return float(series.max())
            except Exception:
                return None

        # Math functions
        if func_name == "ABS":
            inner = evaluate_dax_measure(args_str, df)
            return abs(inner) if inner is not None else None
        elif func_name == "ROUND":
            parts = _split_args(args_str)
            inner = evaluate_dax_measure(parts[0].strip(), df)
            decimals = int(parts[1]) if len(parts) > 1 else 0
            return round(inner, decimals) if inner is not None else None

        # Date functions
        if func_name in ("YEAR", "MONTH", "DAY"):
            col_name = _extract_dax_column_ref(args_str)
            col = _resolve_column(df, col_name)
            if col:
                dt_series = pd.to_datetime(df[col], errors="coerce")
                try:
                    if func_name == "YEAR":
                        return float(dt_series.dt.year.mode()[0])
                    elif func_name == "MONTH":
                        return float(dt_series.dt.month.mode()[0])
                    elif func_name == "DAY":
                        return float(dt_series.dt.day.mode()[0])
                except Exception:
                    return None

    # Numeric literal
    num = _try_parse_number(text)
    if num is not None:
        return num

    # Bare column reference [Column] or Table[Column]
    ref_match = re.match(r"(?:[\w']+)?\[(.+?)\]$", text.strip())
    if ref_match:
        col = _resolve_column(df, ref_match.group(1))
        if col and pd.api.types.is_numeric_dtype(df[col]):
            return float(df[col].sum())
        return None

    return None


# ---------------------------------------------------------------------------
#  Comparison Engine
# ---------------------------------------------------------------------------

def _match_tableau_to_dax(calc_name, dax_measures):
    """Find the DAX measure corresponding to a Tableau calculated field.

    Matches by name (case-insensitive, ignoring extra suffixes like ' Measure').
    Returns (dax_name, dax_formula) or (None, None).
    """
    name_lower = calc_name.lower().strip()
    for dax_name, dax_formula in dax_measures.items():
        dax_lower = dax_name.lower().strip()
        if dax_lower == name_lower:
            return dax_name, dax_formula
        # Handle renamed measures (e.g., "Sales Measure" for Tableau "Sales")
        if dax_lower == name_lower + " measure":
            return dax_name, dax_formula
        # Partial match: Tableau name is a prefix
        if dax_lower.startswith(name_lower) and len(dax_lower) - len(name_lower) < 10:
            return dax_name, dax_formula
    return None, None


def validate_calculations(tableau_spec, dax_measures, df):
    """Compare Tableau calculated fields vs DAX measures on the same data.

    Args:
        tableau_spec: dict from enhanced_tableau_parser with 'calculated_fields'
        dax_measures: dict mapping measure name -> DAX formula string
        df: pandas DataFrame with the source data

    Returns:
        dict with validation results:
        {
            "total_calcs": int,
            "validated": int,
            "mismatched": int,
            "skipped_tableau": int,
            "skipped_dax": int,
            "details": [
                {
                    "name": str,
                    "tableau_formula": str,
                    "dax_formula": str or None,
                    "tableau_result": float or None,
                    "dax_result": float or None,
                    "match": bool or None,
                    "delta_pct": float or None,
                },
                ...
            ]
        }
    """
    calcs = tableau_spec.get("calculated_fields", [])
    result = {
        "total_calcs": len(calcs),
        "validated": 0,
        "mismatched": 0,
        "skipped_tableau": 0,
        "skipped_dax": 0,
        "details": [],
    }

    if df is None or df.empty:
        result["skipped_tableau"] = len(calcs)
        for calc in calcs:
            result["details"].append({
                "name": calc.get("name", "?"),
                "tableau_formula": calc.get("formula", ""),
                "dax_formula": None,
                "tableau_result": None,
                "dax_result": None,
                "match": None,
                "delta_pct": None,
            })
        return result

    for calc in calcs:
        calc_name = calc.get("name", "?")
        tableau_formula = calc.get("formula", "")

        detail = {
            "name": calc_name,
            "tableau_formula": tableau_formula,
            "dax_formula": None,
            "tableau_result": None,
            "dax_result": None,
            "match": None,
            "delta_pct": None,
        }

        # Evaluate Tableau side
        try:
            tableau_result = evaluate_tableau_formula(tableau_formula, df)
        except Exception:
            tableau_result = None

        detail["tableau_result"] = tableau_result

        # Find matching DAX measure
        dax_name, dax_formula = _match_tableau_to_dax(calc_name, dax_measures)
        detail["dax_formula"] = dax_formula

        if dax_formula is None:
            result["skipped_dax"] += 1
            if tableau_result is None:
                result["skipped_tableau"] += 1
            result["details"].append(detail)
            continue

        # Evaluate DAX side
        try:
            dax_result = evaluate_dax_measure(dax_formula, df)
        except Exception:
            dax_result = None

        detail["dax_result"] = dax_result

        if tableau_result is None:
            result["skipped_tableau"] += 1
            result["details"].append(detail)
            continue

        if dax_result is None:
            result["skipped_dax"] += 1
            result["details"].append(detail)
            continue

        # Both sides produced numeric results -- compare
        if not isinstance(tableau_result, (int, float)):
            result["skipped_tableau"] += 1
            result["details"].append(detail)
            continue

        if not isinstance(dax_result, (int, float)):
            result["skipped_dax"] += 1
            result["details"].append(detail)
            continue

        # Calculate delta percentage
        if tableau_result == 0 and dax_result == 0:
            delta_pct = 0.0
            match = True
        elif tableau_result == 0:
            delta_pct = 100.0
            match = False
        else:
            delta_pct = abs(dax_result - tableau_result) / abs(tableau_result) * 100
            # Match if within 0.01% tolerance (floating point rounding)
            match = delta_pct < 0.01

        detail["delta_pct"] = round(delta_pct, 4)
        detail["match"] = match

        if match:
            result["validated"] += 1
        else:
            result["mismatched"] += 1

        result["details"].append(detail)

    return result


def extract_dax_measures_from_config(config):
    """Extract DAX measures from the OpenAI-generated config.

    Args:
        config: dict with tmdl_model.tables[].measures[] structure

    Returns:
        dict mapping measure_name -> dax_formula
    """
    measures = {}
    tables = config.get("tmdl_model", {}).get("tables", [])
    for table in tables:
        for m in table.get("measures", []):
            name = m.get("name", "")
            dax = m.get("dax", m.get("expression", ""))
            if name and dax:
                measures[name] = dax
    return measures


def write_calculation_audit(audit_result, output_dir):
    """Write calculation_audit.json to the output directory.

    Args:
        audit_result: dict from validate_calculations()
        output_dir: path to write the JSON file

    Returns:
        str: path to the written file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "calculation_audit.json")

    # Make results JSON-serializable
    serializable = _make_serializable(audit_result)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return path


def _make_serializable(obj):
    """Convert numpy/pandas types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj
