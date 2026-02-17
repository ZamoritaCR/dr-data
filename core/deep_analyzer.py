"""
Deep Data Analyzer.

This is not pandas .describe(). This is a full statistical profiling
engine that identifies distributions, correlations, outliers, business
signals, and data quality issues.
"""
import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


class DeepAnalyzer:

    def profile(self, df):
        """Full deep profile of a DataFrame."""
        result = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "columns": [],
            "correlations": self._correlations(df),
            "data_quality": self._data_quality(df),
            "business_signals": self._business_signals(df),
            "quick_insights": []
        }

        for col in df.columns:
            col_profile = self._profile_column(df, col)
            result["columns"].append(col_profile)

        result["quick_insights"] = self._generate_insights(df, result)

        return result

    def _profile_column(self, df, col):
        """Deep profile of a single column."""
        series = df[col]
        profile = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 1),
            "unique_count": int(series.nunique()),
            "unique_pct": round(series.nunique() / max(len(series), 1) * 100, 1),
        }

        if pd.api.types.is_numeric_dtype(series):
            profile["semantic_type"] = "numeric"
            self._profile_numeric(series, profile)
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile["semantic_type"] = "datetime"
            self._profile_datetime(series, profile)
        elif pd.api.types.is_bool_dtype(series):
            profile["semantic_type"] = "boolean"
            vc = series.value_counts()
            profile["true_count"] = int(vc.get(True, 0))
            profile["false_count"] = int(vc.get(False, 0))
        else:
            profile["semantic_type"] = "categorical"
            if series.nunique() < 50 or series.nunique() / max(len(series), 1) < 0.05:
                self._profile_categorical(series, profile)
            else:
                profile["semantic_type"] = "text"
                profile["avg_length"] = round(series.dropna().astype(str).str.len().mean(), 1)
                profile["sample_values"] = series.dropna().head(5).tolist()

            # Try date detection
            if profile["semantic_type"] == "categorical":
                try:
                    parsed = pd.to_datetime(series.dropna().head(100), errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        profile["semantic_type"] = "likely_datetime"
                        profile["date_parse_note"] = "Column appears to contain dates stored as text"
                except Exception:
                    pass

        return profile

    def _profile_numeric(self, series, profile):
        """Deep numeric column profiling."""
        clean = series.dropna()
        if len(clean) == 0:
            return

        profile["mean"] = round(float(clean.mean()), 4)
        profile["median"] = round(float(clean.median()), 4)
        profile["std"] = round(float(clean.std()), 4)
        profile["min"] = round(float(clean.min()), 4)
        profile["max"] = round(float(clean.max()), 4)
        profile["q25"] = round(float(clean.quantile(0.25)), 4)
        profile["q75"] = round(float(clean.quantile(0.75)), 4)
        profile["sum"] = round(float(clean.sum()), 2)

        # Distribution shape
        try:
            skew = float(clean.skew())
            kurt = float(clean.kurtosis())
            profile["skewness"] = round(skew, 3)
            profile["kurtosis"] = round(kurt, 3)

            if abs(skew) < 0.5 and abs(kurt) < 1:
                profile["distribution"] = "approximately_normal"
            elif skew > 1:
                profile["distribution"] = "right_skewed"
            elif skew < -1:
                profile["distribution"] = "left_skewed"
            elif kurt > 3:
                profile["distribution"] = "heavy_tailed"
            else:
                profile["distribution"] = "moderate_skew"
        except Exception:
            profile["distribution"] = "unknown"

        # Outliers (IQR method)
        try:
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = clean[(clean < lower) | (clean > upper)]
            profile["outlier_count"] = int(len(outliers))
            profile["outlier_pct"] = round(len(outliers) / len(clean) * 100, 1)
            if len(outliers) > 0:
                profile["outlier_range"] = {
                    "below": round(float(lower), 2),
                    "above": round(float(upper), 2)
                }
        except Exception:
            profile["outlier_count"] = 0

        # Special value counts
        profile["zero_count"] = int((clean == 0).sum())
        profile["negative_count"] = int((clean < 0).sum())
        profile["negative_pct"] = round((clean < 0).mean() * 100, 1)

    def _profile_categorical(self, series, profile):
        """Deep categorical column profiling."""
        clean = series.dropna()
        if len(clean) == 0:
            return

        vc = clean.value_counts()
        top_n = min(10, len(vc))

        profile["top_values"] = [
            {"value": str(val), "count": int(cnt),
             "pct": round(cnt / len(clean) * 100, 1)}
            for val, cnt in vc.head(top_n).items()
        ]
        profile["mode"] = str(vc.index[0]) if len(vc) > 0 else None

        if len(vc) > 0:
            top_pct = vc.iloc[0] / len(clean)
            profile["top_value_concentration"] = round(top_pct * 100, 1)
            if top_pct > 0.9:
                profile["concentration_flag"] = "near_constant"
            elif top_pct > 0.5:
                profile["concentration_flag"] = "dominant_value"

    def _profile_datetime(self, series, profile):
        """Deep datetime column profiling."""
        clean = series.dropna()
        if len(clean) == 0:
            return

        profile["min_date"] = str(clean.min())
        profile["max_date"] = str(clean.max())
        profile["date_range_days"] = int((clean.max() - clean.min()).days)

        if len(clean) > 1:
            sorted_dates = clean.sort_values()
            diffs = sorted_dates.diff().dropna()
            median_gap = diffs.median()
            max_gap = diffs.max()
            if max_gap > median_gap * 5:
                profile["has_large_gaps"] = True
                profile["largest_gap_days"] = int(max_gap.days)

        try:
            if profile["date_range_days"] > 365:
                monthly = clean.dt.month.value_counts().sort_index()
                if monthly.std() / monthly.mean() > 0.3:
                    profile["seasonality_hint"] = "possible_monthly_pattern"
        except Exception:
            pass

    def _correlations(self, df):
        """Find strong correlations between numeric columns."""
        result = {"strong_positive": [], "strong_negative": []}
        numeric = df.select_dtypes(include="number")

        if len(numeric.columns) < 2:
            return result

        try:
            corr = numeric.corr()
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if pd.notna(val):
                        pair = (corr.columns[i], corr.columns[j], round(val, 3))
                        if val > 0.7:
                            result["strong_positive"].append(pair)
                        elif val < -0.7:
                            result["strong_negative"].append(pair)
        except Exception:
            pass

        return result

    def _data_quality(self, df):
        """Assess overall data quality."""
        total_cells = df.shape[0] * df.shape[1]
        total_nulls = int(df.isna().sum().sum())

        return {
            "total_cells": total_cells,
            "total_nulls": total_nulls,
            "total_null_pct": round(total_nulls / max(total_cells, 1) * 100, 1),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_row_pct": round(df.duplicated().mean() * 100, 1),
            "columns_with_nulls": [
                col for col in df.columns if df[col].isna().any()
            ],
            "constant_columns": [
                col for col in df.columns if df[col].nunique() <= 1
            ],
            "high_cardinality_text": [
                col for col in df.select_dtypes(include="object").columns
                if df[col].nunique() > 100
            ]
        }

    def _business_signals(self, df):
        """Identify likely business semantics."""
        signals = {
            "likely_kpis": [],
            "likely_dimensions": [],
            "likely_date_column": None,
            "likely_id_column": None,
            "suggested_calculations": [],
            "detected_hierarchies": []
        }

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = df.select_dtypes(include="datetime").columns.tolist()

        kpi_keywords = [
            "revenue", "sales", "profit", "cost", "price", "amount",
            "total", "quantity", "qty", "count", "margin", "income",
            "expense", "budget", "actual", "target", "score", "rate",
            "value", "payment", "balance", "discount"
        ]
        for col in numeric_cols:
            lower = col.lower().replace("_", " ").replace("-", " ")
            if any(kw in lower for kw in kpi_keywords):
                signals["likely_kpis"].append(col)
            elif df[col].nunique() > 20 and df[col].std() > 0:
                signals["likely_kpis"].append(col)

        dim_keywords = [
            "region", "country", "state", "city", "category",
            "segment", "department", "product", "customer", "type",
            "status", "channel", "source", "brand", "group", "class"
        ]
        for col in cat_cols:
            lower = col.lower().replace("_", " ").replace("-", " ")
            if any(kw in lower for kw in dim_keywords):
                signals["likely_dimensions"].append(col)
            elif 2 <= df[col].nunique() <= 50:
                signals["likely_dimensions"].append(col)

        if date_cols:
            signals["likely_date_column"] = date_cols[0]
        else:
            for col in cat_cols:
                try:
                    sample = df[col].dropna().head(20)
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        signals["likely_date_column"] = col
                        break
                except Exception:
                    pass

        for col in df.columns:
            lower = col.lower()
            if any(kw in lower for kw in ["id", "key", "code", "number", "no"]):
                if df[col].nunique() > len(df) * 0.8:
                    signals["likely_id_column"] = col
                    break

        col_lower = {c.lower().replace("_", ""): c for c in df.columns}
        if "sales" in col_lower and "profit" in col_lower:
            signals["suggested_calculations"].append({
                "name": "Profit Margin %",
                "formula": f"{col_lower['profit']} / {col_lower['sales']} * 100",
                "rationale": "Profitability ratio -- critical business metric"
            })
        if "quantity" in col_lower and "sales" in col_lower:
            signals["suggested_calculations"].append({
                "name": "Average Selling Price",
                "formula": f"{col_lower['sales']} / {col_lower['quantity']}",
                "rationale": "Unit economics indicator"
            })

        if len(cat_cols) >= 2:
            for i, col1 in enumerate(cat_cols):
                for col2 in cat_cols[i+1:]:
                    n1 = df[col1].nunique()
                    n2 = df[col2].nunique()
                    if n1 < n2 and n2 > n1 * 2:
                        try:
                            grouped = df.groupby(col2)[col1].nunique()
                            if grouped.max() == 1:
                                signals["detected_hierarchies"].append({
                                    "parent": col1,
                                    "child": col2,
                                    "parent_cardinality": int(n1),
                                    "child_cardinality": int(n2)
                                })
                        except Exception:
                            pass

        return signals

    def _generate_insights(self, df, profile):
        """Generate top insights as plain text."""
        insights = []

        insights.append(
            f"Dataset contains {len(df):,} rows and "
            f"{len(df.columns)} columns."
        )

        for col_p in profile["columns"]:
            if col_p.get("semantic_type") in ("datetime", "likely_datetime"):
                if "min_date" in col_p:
                    insights.append(
                        f"Time range: {col_p['min_date'][:10]} to "
                        f"{col_p['max_date'][:10]} "
                        f"({col_p.get('date_range_days', '?')} days)."
                    )
                break

        kpis = profile["business_signals"]["likely_kpis"][:3]
        for kpi in kpis:
            col_p = next((c for c in profile["columns"] if c["name"] == kpi), None)
            if col_p and "sum" in col_p:
                insights.append(
                    f"{kpi}: total {col_p['sum']:,.0f}, "
                    f"avg {col_p['mean']:,.2f}, "
                    f"range {col_p['min']:,.2f} to {col_p['max']:,.2f}."
                )

        dq = profile["data_quality"]
        if dq["total_null_pct"] > 5:
            insights.append(
                f"Data quality note: {dq['total_null_pct']}% null values "
                f"across {len(dq['columns_with_nulls'])} columns."
            )
        if dq["duplicate_row_pct"] > 1:
            insights.append(
                f"Warning: {dq['duplicate_rows']:,} duplicate rows "
                f"({dq['duplicate_row_pct']}%)."
            )

        strong = profile["correlations"]["strong_positive"]
        if strong:
            top = strong[0]
            insights.append(
                f"Strong correlation: {top[0]} and {top[1]} "
                f"(r={top[2]})."
            )

        return insights
