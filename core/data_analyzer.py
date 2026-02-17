"""
Data Analyzer - Produces a complete profile of a pandas DataFrame.
The profile is JSON-serializable and designed to feed into Claude for analysis.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np


class DataAnalyzer:
    """Analyze a DataFrame and produce a detailed column-by-column profile."""

    def analyze(self, df, table_name="Data"):
        """Generate a full profile dictionary.

        Args:
            df: pandas DataFrame to analyze.
            table_name: logical name for this table (typically from filename).

        Returns:
            dict: JSON-serializable profile.
        """
        print(f"[OK] Analyzing table: {table_name}")
        print(f"     {len(df)} rows x {len(df.columns)} columns")

        profile = {
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [],
            "suggested_relationships": [],
            "data_quality_score": 0,
        }

        total_quality = 0

        for col_name in df.columns:
            series = df[col_name]
            col_profile = self._profile_column(col_name, series, len(df))
            profile["columns"].append(col_profile)
            total_quality += col_profile["_quality"]

        # Remove internal quality scores from column output
        for col in profile["columns"]:
            del col["_quality"]

        # Overall data quality score
        if df.columns.size > 0:
            profile["data_quality_score"] = round(total_quality / len(df.columns), 1)
        else:
            profile["data_quality_score"] = 0

        # Suggested relationships
        profile["suggested_relationships"] = self._find_relationship_candidates(df)

        # Summary counts by semantic type
        type_counts = {}
        for col in profile["columns"]:
            st = col["semantic_type"]
            type_counts[st] = type_counts.get(st, 0) + 1
        profile["semantic_type_summary"] = type_counts

        print(f"     Quality score: {profile['data_quality_score']}/100")
        print(f"     Types: {type_counts}")
        return profile

    # ------------------------------------------------------------------ #
    #  Column profiling                                                    #
    # ------------------------------------------------------------------ #

    def _profile_column(self, name, series, total_rows):
        """Build a profile dict for a single column."""
        null_count = int(series.isna().sum())
        null_pct = round(null_count / total_rows * 100, 2) if total_rows > 0 else 0
        unique_count = int(series.nunique())

        col = {
            "name": name,
            "dtype": str(series.dtype),
            "semantic_type": self._classify_semantic_type(name, series),
            "null_count": null_count,
            "null_percentage": null_pct,
            "unique_count": unique_count,
        }

        sem = col["semantic_type"]

        if sem == "measure":
            col["stats"] = self._numeric_stats(series)
        elif sem == "date":
            col["stats"] = self._date_stats(series)
        elif sem == "dimension":
            col["stats"] = self._categorical_stats(series)

        # Quality score for this column (0-100)
        col["_quality"] = self._column_quality(null_pct, unique_count, total_rows, sem)

        return col

    # ------------------------------------------------------------------ #
    #  Semantic type classification                                        #
    # ------------------------------------------------------------------ #

    def _classify_semantic_type(self, name, series):
        """Determine if a column is a measure, dimension, or date."""
        dtype = series.dtype

        # Datetime types
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "date"

        # Check column name hints for dates
        date_hints = ["date", "time", "timestamp", "created", "updated", "year", "month", "day"]
        name_lower = name.lower()
        if any(hint in name_lower for hint in date_hints):
            # Try parsing as date
            sample = series.dropna().head(20)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    return "date"
                except (ValueError, TypeError):
                    pass

        # Numeric types
        if pd.api.types.is_numeric_dtype(dtype):
            # Columns with "id", "code", "zip", "postal", "number", "row" in name
            # are likely identifiers, not measures
            id_hints = ["id", "code", "zip", "postal", "row", "index", "fk", "key"]
            name_tokens = name_lower.replace(" ", "_").replace("-", "_").split("_")
            if any(hint in name_tokens for hint in id_hints):
                return "dimension"
            # Integer columns with very few unique values might be categorical
            if pd.api.types.is_integer_dtype(dtype):
                if series.nunique() <= 5:
                    return "dimension"
            return "measure"

        # Object / string types
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            # Check id/code hints first -- even if parseable as numeric
            id_hints_str = ["id", "code", "zip", "postal", "row", "index", "fk", "key"]
            name_tokens = name_lower.replace(" ", "_").replace("-", "_").split("_")
            if any(hint in name_tokens for hint in id_hints_str):
                return "dimension"
            # Could still be numeric stored as string
            sample = series.dropna().head(50)
            if len(sample) > 0:
                try:
                    pd.to_numeric(sample)
                    return "measure"
                except (ValueError, TypeError):
                    pass

        return "dimension"

    # ------------------------------------------------------------------ #
    #  Stats by type                                                       #
    # ------------------------------------------------------------------ #

    def _numeric_stats(self, series):
        """Stats for numeric columns."""
        clean = series.dropna()
        if len(clean) == 0:
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}

        # Ensure numeric
        clean = pd.to_numeric(clean, errors="coerce").dropna()
        if len(clean) == 0:
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}

        return {
            "min": self._safe_scalar(clean.min()),
            "max": self._safe_scalar(clean.max()),
            "mean": round(float(clean.mean()), 4),
            "median": round(float(clean.median()), 4),
            "std": round(float(clean.std()), 4) if len(clean) > 1 else 0,
        }

    def _date_stats(self, series):
        """Stats for date columns."""
        try:
            dates = pd.to_datetime(series, errors="coerce").dropna()
        except Exception:
            return {"min_date": None, "max_date": None, "date_range_days": None}

        if len(dates) == 0:
            return {"min_date": None, "max_date": None, "date_range_days": None}

        min_d = dates.min()
        max_d = dates.max()
        return {
            "min_date": str(min_d.date()),
            "max_date": str(max_d.date()),
            "date_range_days": int((max_d - min_d).days),
        }

    def _categorical_stats(self, series):
        """Stats for categorical/dimension columns."""
        counts = series.value_counts(dropna=True).head(10)
        top_values = [
            {"value": str(val), "count": int(cnt)}
            for val, cnt in counts.items()
        ]
        return {"top_values": top_values}

    # ------------------------------------------------------------------ #
    #  Quality scoring                                                     #
    # ------------------------------------------------------------------ #

    def _column_quality(self, null_pct, unique_count, total_rows, sem_type):
        """Score 0-100 for a single column's data quality."""
        score = 100.0

        # Penalize nulls
        if null_pct > 50:
            score -= 40
        elif null_pct > 20:
            score -= 25
        elif null_pct > 5:
            score -= 10
        elif null_pct > 0:
            score -= 5

        # Penalize zero-variance columns (all same value)
        if unique_count <= 1 and total_rows > 1:
            score -= 20

        # Penalize dimensions with too many unique values (possible free-text)
        if sem_type == "dimension" and total_rows > 0:
            uniqueness = unique_count / total_rows
            if uniqueness > 0.95:
                score -= 10  # Almost all unique -- might be an ID or free text

        return max(score, 0)

    # ------------------------------------------------------------------ #
    #  Relationship detection                                              #
    # ------------------------------------------------------------------ #

    def _find_relationship_candidates(self, df):
        """Find columns that might be foreign keys or join candidates."""
        candidates = []
        key_hints = ["id", "key", "code", "fk", "ref", "number", "num", "no"]

        for col in df.columns:
            name_lower = col.lower().replace(" ", "_")
            for hint in key_hints:
                if hint in name_lower.split("_"):
                    candidates.append({
                        "column": col,
                        "reason": f"Column name contains '{hint}'",
                        "unique_count": int(df[col].nunique()),
                        "is_unique": bool(df[col].nunique() == df[col].dropna().shape[0]),
                    })
                    break

        return candidates

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _safe_scalar(self, value):
        """Convert numpy scalars to native Python types for JSON serialization."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            val = float(value)
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 4)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value


# ------------------------------------------------------------------ #
#  CLI test                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # Add project root to path so we can import connectors
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    from connectors.file_connector import FileConnector

    # Find the first data file
    data_file = None
    for pattern in ["*.csv", "*.xlsx", "*.xls"]:
        matches = sorted(project_root.glob(pattern))
        if matches:
            data_file = matches[0]
            break

    if not data_file:
        print("[ERROR] No CSV or Excel file found in project root")
        sys.exit(1)

    print(f"=== Data Analysis: {data_file.name} ===")
    print("=" * 60)

    # Load
    connector = FileConnector()
    df, meta = connector.load(data_file)

    # Analyze
    print()
    table_name = data_file.stem  # filename without extension
    analyzer = DataAnalyzer()
    profile = analyzer.analyze(df, table_name=table_name)

    # Save
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "data_profile.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Profile saved to {output_path}")
    print(f"\n{'=' * 60}")
    print(f"PROFILE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Table: {profile['table_name']}")
    print(f"Rows:  {profile['row_count']}")
    print(f"Cols:  {profile['column_count']}")
    print(f"Quality: {profile['data_quality_score']}/100")
    print(f"Types: {profile['semantic_type_summary']}")
    print()

    # Print column details
    for col in profile["columns"]:
        sem = col["semantic_type"]
        tag = f"[{sem.upper():9s}]"
        null_info = f"nulls={col['null_pct']}%" if col.get("null_percentage", 0) > 0 else ""
        if not null_info:
            null_info = f"nulls={col['null_percentage']}%" if col.get("null_percentage", 0) > 0 else "complete"
        print(f"  {tag} {col['name']:30s} unique={col['unique_count']:<8d} {null_info}")
