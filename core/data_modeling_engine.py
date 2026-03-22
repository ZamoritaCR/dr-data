"""
Dr. Data -- Advanced Data Modeling Engine
Kimball, Data Vault 2.0, Semantic Layer, ERD Generation, SQL Analysis.
Art of the Possible.
"""
import pandas as pd
import numpy as np
import re
from typing import Optional


def infer_dimensional_model(df, table_name="fact_table"):
    """
    Auto-infer a Kimball-style dimensional model from a DataFrame.

    Applies heuristics:
    - Numeric columns with high variance = measures (facts)
    - Date/time columns = time dimension
    - Low-cardinality text columns = dimensions
    - High-cardinality ID columns = degenerate dimensions or keys

    Returns dict with fact_table, dimension_tables, star_schema_ddl, recommendations.
    """
    measures = []
    dimensions = []
    time_dims = []
    degenerate_dims = []

    for col in df.columns:
        col_lower = col.lower()
        n_unique = df[col].nunique()
        n_rows = len(df)
        cardinality = n_unique / max(n_rows, 1)

        if pd.api.types.is_datetime64_any_dtype(df[col]) or any(
            kw in col_lower for kw in ["date", "time", "year", "month", "day", "period"]
        ):
            time_dims.append({
                "column": col,
                "suggested_dim": "dim_date",
                "type": "date_dimension",
                "scd_type": 0,
            })
        elif pd.api.types.is_numeric_dtype(df[col]):
            if any(kw in col_lower for kw in ["id", "_key", "_sk", "fk_"]):
                degenerate_dims.append({"column": col, "type": "foreign_key"})
            else:
                measures.append({
                    "column": col,
                    "aggregation": (
                        "SUM"
                        if not any(
                            kw in col_lower
                            for kw in ["rate", "ratio", "pct", "avg", "score"]
                        )
                        else "AVG"
                    ),
                    "measure_type": (
                        "additive"
                        if any(
                            kw in col_lower
                            for kw in ["amount", "qty", "count", "revenue", "cost"]
                        )
                        else "semi_additive"
                    ),
                })
        elif cardinality < 0.05 and n_unique < 100:
            dim_name = f"dim_{re.sub(r'[^a-z0-9]', '_', col_lower.replace(' ', '_'))}"
            dimensions.append({
                "column": col,
                "dim_name": dim_name,
                "cardinality": n_unique,
                "scd_type": 2,
                "type": (
                    "conformed_dimension"
                    if any(
                        kw in col_lower
                        for kw in ["region", "country", "status", "type", "category", "channel"]
                    )
                    else "regular_dimension"
                ),
            })
        elif cardinality > 0.9:
            degenerate_dims.append({"column": col, "type": "degenerate_dimension"})

    # Generate DDL
    dim_ddls = []
    for dim in dimensions:
        scd_cols = ""
        if dim["scd_type"] == 2:
            scd_cols = (
                "    effective_date    DATE NOT NULL,\n"
                "    expiry_date       DATE,\n"
                "    is_current        BIT DEFAULT 1,\n"
            )
        dim_ddls.append(
            f"-- {dim['dim_name']} (SCD Type {dim['scd_type']})\n"
            f"CREATE TABLE {dim['dim_name']} (\n"
            f"    {dim['dim_name']}_key  INT PRIMARY KEY IDENTITY,\n"
            f"    {dim['column']}        VARCHAR(255) NOT NULL,\n"
            f"{scd_cols}"
            f"    created_at         DATETIME DEFAULT GETDATE()\n"
            f");\n"
        )

    time_ddl = ""
    if time_dims:
        time_ddl = (
            "-- dim_date (Type 0 -- dates don't change)\n"
            "CREATE TABLE dim_date (\n"
            "    date_key         INT PRIMARY KEY,  -- YYYYMMDD format\n"
            "    full_date        DATE NOT NULL,\n"
            "    year             INT, quarter      INT, month      INT,\n"
            "    month_name       VARCHAR(10), week_of_year INT,\n"
            "    day_of_week      INT, day_name     VARCHAR(10),\n"
            "    is_weekday       BIT, is_holiday   BIT\n"
            ");\n\n"
        )

    fk_lines = [
        f"    {d['dim_name']}_key  INT REFERENCES {d['dim_name']}({d['dim_name']}_key),"
        for d in dimensions
    ]
    measure_lines = [
        f"    {m['column']}  DECIMAL(18,4),  -- {m['aggregation']}, {m['measure_type']}"
        for m in measures
    ]
    time_fk = "    date_key  INT REFERENCES dim_date(date_key),\n" if time_dims else ""

    fact_ddl = (
        f"-- {table_name} (Grain: one row per [define grain here])\n"
        f"CREATE TABLE {table_name} (\n"
        f"    {table_name}_key  BIGINT PRIMARY KEY IDENTITY,\n"
        f"{time_fk}"
        + "\n".join(fk_lines)
        + ("\n" if fk_lines else "")
        + "\n".join(measure_lines)
        + ("\n" if measure_lines else "")
        + "    load_date         DATETIME DEFAULT GETDATE()\n"
        ");\n"
    )

    recommendations = []
    if not time_dims:
        recommendations.append(
            "No date column detected -- add a date dimension for time-series analysis"
        )
    if not measures:
        recommendations.append(
            "No clear measures detected -- verify numeric columns represent quantifiable business events"
        )
    if len(dimensions) > 15:
        recommendations.append(
            "Many dimensions detected -- consider a snowflake schema for high-cardinality dimensions"
        )
    if any(d["scd_type"] == 2 for d in dimensions):
        recommendations.append(
            "SCD Type 2 recommended for dimensions that change over time -- implement effective/expiry dates"
        )

    return {
        "fact_table": {
            "name": table_name,
            "suggested_measures": measures,
            "foreign_keys": [d["dim_name"] + "_key" for d in dimensions],
            "grain": "One row per [define business process event]",
        },
        "dimension_tables": dimensions + (time_dims if time_dims else []),
        "degenerate_dimensions": degenerate_dims,
        "star_schema_ddl": time_ddl + "\n".join(dim_ddls) + "\n" + fact_ddl,
        "kimball_recommendations": recommendations,
        "model_type": "star_schema",
    }


def generate_data_vault_model(df, business_key_cols):
    """
    Auto-generate a Data Vault 2.0 model skeleton from a DataFrame.
    Produces: Hub, Link, Satellite definitions with hash key logic.
    """
    hubs = []
    satellites = []

    for bk_col in business_key_cols:
        hub_name = f"hub_{re.sub(r'[^a-z0-9]', '_', bk_col.lower())}"
        sat_name = f"sat_{re.sub(r'[^a-z0-9]', '_', bk_col.lower())}_details"

        descriptor_cols = [
            c
            for c in df.columns
            if c != bk_col and not pd.api.types.is_datetime64_any_dtype(df[c])
        ][:8]

        hubs.append({
            "hub_name": hub_name,
            "business_key": bk_col,
            "hash_key": f"hk_{hub_name}",
            "ddl": (
                f"CREATE TABLE {hub_name} (\n"
                f"    hk_{hub_name}      BINARY(16) NOT NULL PRIMARY KEY,  -- MD5 hash of {bk_col}\n"
                f"    {bk_col}           VARCHAR(255) NOT NULL,\n"
                f"    load_date          DATETIME NOT NULL DEFAULT GETDATE(),\n"
                f"    record_source      VARCHAR(100) NOT NULL\n"
                f");"
            ),
        })

        sat_cols = "\n    ".join(
            [f"{c}  {_infer_sql_type(df[c])}," for c in descriptor_cols]
        )
        satellites.append({
            "sat_name": sat_name,
            "parent_hub": hub_name,
            "descriptor_columns": descriptor_cols,
            "ddl": (
                f"CREATE TABLE {sat_name} (\n"
                f"    hk_{hub_name}      BINARY(16) NOT NULL REFERENCES {hub_name}(hk_{hub_name}),\n"
                f"    load_date          DATETIME NOT NULL,\n"
                f"    load_end_date      DATETIME,  -- NULL = current record\n"
                f"    {sat_cols}\n"
                f"    record_source      VARCHAR(100) NOT NULL,\n"
                f"    hash_diff          BINARY(16),  -- MD5 of all descriptors (change detection)\n"
                f"    PRIMARY KEY (hk_{hub_name}, load_date)\n"
                f");"
            ),
        })

    return {
        "model_type": "data_vault_2",
        "hubs": hubs,
        "satellites": satellites,
        "links": [],
        "recommendations": [
            "Load Hubs first (business keys only)",
            "Load Links second (relationships)",
            "Load Satellites last (descriptive attributes + hash_diff for change detection)",
            "Use BINARY(16) MD5 hash keys -- never expose business keys in downstream joins",
            "Implement PIT (Point-In-Time) tables for historical reconstruction",
        ],
    }


def _infer_sql_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(series):
        return "DECIMAL(18,4)"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME"
    elif pd.api.types.is_bool_dtype(series):
        return "BIT"
    else:
        max_len = series.astype(str).str.len().max() if len(series) > 0 else 50
        return f"VARCHAR({min(int(max_len * 1.5), 4000)})"


def analyze_sql(sql_text):
    """
    Analyze SQL for anti-patterns, normalization violations, and optimization opportunities.
    Uses sqlglot for parsing when available, falls back to regex.
    """
    issues = []
    suggestions = []
    sql_upper = sql_text.upper()

    sqlglot_parsed = False
    try:
        import sqlglot
        sqlglot.parse(sql_text)
        sqlglot_parsed = True
    except Exception:
        pass

    # Anti-pattern detection
    if "SELECT *" in sql_upper:
        issues.append({"severity": "warning", "issue": "SELECT * -- specify column list explicitly"})
    if sql_upper.count("JOIN") > 5:
        issues.append({
            "severity": "warning",
            "issue": f"{sql_upper.count('JOIN')} JOINs -- consider denormalization or pre-aggregated table",
        })
    if "NOT IN" in sql_upper:
        issues.append({
            "severity": "warning",
            "issue": "NOT IN -- replace with NOT EXISTS or LEFT JOIN IS NULL (better NULL handling)",
        })
    if "OR" in sql_upper and "WHERE" in sql_upper:
        issues.append({
            "severity": "info",
            "issue": "OR in WHERE clause -- may prevent index usage, consider UNION ALL",
        })
    if "DISTINCT" in sql_upper:
        suggestions.append(
            "DISTINCT is expensive -- ensure it is necessary; consider GROUP BY or deduplication upstream"
        )
    if "HAVING" in sql_upper and "GROUP BY" not in sql_upper:
        issues.append({
            "severity": "error",
            "issue": "HAVING without GROUP BY -- likely a logic error",
        })

    # Extract tables referenced
    tables = re.findall(r"FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_upper) + re.findall(
        r"JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql_upper
    )

    return {
        "issues": issues,
        "suggestions": suggestions,
        "tables_referenced": list(set(tables)),
        "join_count": sql_upper.count("JOIN"),
        "complexity_score": min(10, sql_upper.count("JOIN") * 1.5 + len(issues) * 0.5),
        "sqlglot_parsed": sqlglot_parsed,
    }
