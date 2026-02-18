"""
Data Quality Engine -- enterprise DAMA-DMBOK scoring across six dimensions.

Scans DataFrames for Completeness, Accuracy, Consistency,
Timeliness, Uniqueness, and Validity.  Generates scorecards,
recommendations, and auto-remediation actions.
"""
import re
import json
import warnings
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from scipy import stats


# Placeholder patterns flagged as suspect
_SUSPECT_VALUES = {
    "n/a", "na", "null", "none", "test", "xxx", "tbd",
    "unknown", "-", "0", ".", "..", "---", "missing", "temp",
}

_POSITIVE_NAME_HINTS = {
    "amount", "price", "quantity", "count", "age", "balance",
    "revenue", "cost", "total", "salary", "weight", "height",
}


class DataQualityEngine:
    """Enterprise data quality scanner aligned with DAMA-DMBOK."""

    def __init__(self):
        self.scan_results = {}
        self.scan_timestamp = None
        self.thresholds = {
            "completeness_warn": 95,
            "completeness_fail": 80,
            "uniqueness_warn": 99,
            "uniqueness_fail": 95,
            "validity_warn": 95,
            "validity_fail": 85,
            "freshness_hours_warn": 24,
            "freshness_hours_fail": 72,
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def scan_table(self, df, table_name, metadata=None):
        """Run all six DAMA dimensions on *df*. Returns a result dict."""
        try:
            print(f"[DQ ENGINE] Scanning table: {table_name} "
                  f"({len(df):,} rows x {len(df.columns)} cols)")
            result = {
                "table_name": table_name,
                "scan_timestamp": datetime.now(timezone.utc).isoformat(),
                "row_count": len(df),
                "column_count": len(df.columns),
                "overall_score": 0.0,
                "dimensions": {
                    "completeness": self._score_completeness(df),
                    "accuracy": self._score_accuracy(df),
                    "consistency": self._score_consistency(df),
                    "timeliness": self._score_timeliness(df),
                    "uniqueness": self._score_uniqueness(df),
                    "validity": self._score_validity(df),
                },
                "column_details": self._profile_columns(df),
                "recommendations": [],
                "remediation_actions": [],
            }
            # Weighted overall score (skip None timeliness)
            weights = {
                "completeness": 0.25,
                "accuracy": 0.20,
                "consistency": 0.15,
                "timeliness": 0.10,
                "uniqueness": 0.15,
                "validity": 0.15,
            }
            total_w, total_s = 0.0, 0.0
            for dim, w in weights.items():
                score = result["dimensions"][dim].get("score")
                if score is not None:
                    total_w += w
                    total_s += score * w
            result["overall_score"] = round(total_s / total_w, 1) if total_w else 0.0
            self._generate_recommendations(result)
            self.scan_results[table_name] = result
            self.scan_timestamp = result["scan_timestamp"]
            print(f"[DQ ENGINE] {table_name} overall score: "
                  f"{result['overall_score']}")
            return result
        except Exception as e:
            print(f"[DQ ENGINE] Scan failed for {table_name}: {e}")
            import traceback; traceback.print_exc()
            return {"table_name": table_name, "error": str(e)}

    def scan_multiple_tables(self, tables_dict):
        """Scan several tables and run cross-table analysis."""
        try:
            results = {}
            for name, df in tables_dict.items():
                results[name] = self.scan_table(df, name)

            cross = self._cross_table_analysis(tables_dict)
            self._last_cross_results = cross
            scores = [
                r["overall_score"]
                for r in results.values()
                if isinstance(r.get("overall_score"), (int, float))
            ]
            return {
                "tables": results,
                "cross_table": cross,
                "overall_score": round(np.mean(scores), 1) if scores else 0,
                "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"[DQ ENGINE] Multi-table scan failed: {e}")
            return {"error": str(e)}

    def generate_scorecard_data(self):
        """Summary dict optimised for dashboard display."""
        try:
            all_scores = {}
            total_cols, total_rows = 0, 0
            crit = high = med = low = 0
            auto_fix = 0
            top_recs = []
            for name, res in self.scan_results.items():
                if "error" in res:
                    continue
                total_rows += res.get("row_count", 0)
                total_cols += res.get("column_count", 0)
                for r in res.get("recommendations", []):
                    p = r.get("priority", "LOW")
                    if p == "CRITICAL":
                        crit += 1
                    elif p == "HIGH":
                        high += 1
                    elif p == "MEDIUM":
                        med += 1
                    else:
                        low += 1
                    if r.get("auto_fixable"):
                        auto_fix += 1
                    top_recs.append(r)
                for dim, data in res.get("dimensions", {}).items():
                    s = data.get("score")
                    if s is not None:
                        all_scores.setdefault(dim, []).append(s)
            dim_avg = {d: round(np.mean(v), 1) for d, v in all_scores.items()}
            overall = round(np.mean(list(dim_avg.values())), 1) if dim_avg else 0
            top_recs.sort(key=lambda r: {"CRITICAL": 0, "HIGH": 1,
                                          "MEDIUM": 2, "LOW": 3}.get(
                                              r.get("priority", "LOW"), 4))
            return {
                "overall_score": overall,
                "dimension_scores": dim_avg,
                "tables_scanned": len(self.scan_results),
                "total_columns": total_cols,
                "total_rows": total_rows,
                "critical_issues": crit,
                "high_issues": high,
                "medium_issues": med,
                "low_issues": low,
                "top_recommendations": top_recs[:10],
                "auto_fixable_count": auto_fix,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Scorecard generation failed: {e}")
            return {}

    def auto_remediate(self, df, table_name, fix_types=None):
        """Apply automatic fixes. Returns dict with fixed_df and log."""
        try:
            fixes = fix_types or ["imputation", "dedup",
                                  "format_standardize", "remove_outliers"]
            changes = []
            df_fixed = df.copy()

            if "dedup" in fixes:
                before = len(df_fixed)
                df_fixed = df_fixed.drop_duplicates()
                removed = before - len(df_fixed)
                if removed:
                    changes.append({"action": "dedup", "column": "(all)",
                                    "rows_affected": removed})

            if "imputation" in fixes:
                for col in df_fixed.columns:
                    nulls = df_fixed[col].isna().sum()
                    if nulls == 0:
                        continue
                    if pd.api.types.is_numeric_dtype(df_fixed[col]):
                        med = df_fixed[col].median()
                        df_fixed[col] = df_fixed[col].fillna(med)
                        changes.append({"action": "imputation",
                                        "column": col,
                                        "rows_affected": int(nulls)})
                    elif df_fixed[col].dtype == object:
                        df_fixed[col] = df_fixed[col].fillna("Unknown")
                        changes.append({"action": "imputation",
                                        "column": col,
                                        "rows_affected": int(nulls)})

            if "format_standardize" in fixes:
                for col in df_fixed.select_dtypes(include="object").columns:
                    if df_fixed[col].nunique() <= 20:
                        original = df_fixed[col].copy()
                        df_fixed[col] = df_fixed[col].str.strip()
                        changed = (original != df_fixed[col]).sum()
                        if changed:
                            changes.append({"action": "format_standardize",
                                            "column": col,
                                            "rows_affected": int(changed)})

            if "remove_outliers" in fixes:
                for col in df_fixed.select_dtypes(include="number").columns:
                    q1 = df_fixed[col].quantile(0.25)
                    q3 = df_fixed[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr == 0:
                        continue
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    mask = (df_fixed[col] < lo) | (df_fixed[col] > hi)
                    clipped = mask.sum()
                    if clipped:
                        df_fixed[col] = df_fixed[col].clip(lo, hi)
                        changes.append({"action": "remove_outliers",
                                        "column": col,
                                        "rows_affected": int(clipped)})

            print(f"[DQ ENGINE] Auto-remediation on {table_name}: "
                  f"{len(changes)} change(s)")
            return {"fixed_df": df_fixed, "changes_made": changes}
        except Exception as e:
            print(f"[DQ ENGINE] Auto-remediation failed: {e}")
            return {"fixed_df": df, "changes_made": []}

    # ------------------------------------------------------------------ #
    #  Dimension Scorers                                                  #
    # ------------------------------------------------------------------ #

    def _score_completeness(self, df):
        try:
            col_scores = {}
            for col in df.columns:
                pct = round(df[col].notna().mean() * 100, 2)
                nc = int(df[col].isna().sum())
                status = ("PASS" if pct >= self.thresholds["completeness_warn"]
                          else "WARN" if pct >= self.thresholds["completeness_fail"]
                          else "FAIL")
                col_scores[col] = {"score": pct, "null_count": nc,
                                   "status": status}
            table_score = round(np.mean([v["score"] for v in col_scores.values()]), 2)
            status = ("PASS" if table_score >= self.thresholds["completeness_warn"]
                      else "WARN" if table_score >= self.thresholds["completeness_fail"]
                      else "FAIL")
            return {"score": table_score, "status": status, "columns": col_scores}
        except Exception as e:
            print(f"[DQ ENGINE] Completeness scoring failed: {e}")
            return {"score": None, "status": "ERROR", "columns": {}}

    def _score_accuracy(self, df):
        try:
            outliers = {}
            suspect_values = {}
            scored_cols = 0
            total_accuracy = 0.0

            for col in df.select_dtypes(include="number").columns:
                s = df[col].dropna()
                if len(s) < 10:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    iqr_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
                else:
                    iqr_mask = pd.Series(False, index=s.index)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    z = np.abs(stats.zscore(s, nan_policy="omit"))
                z_mask = z > 3
                combined = iqr_mask | z_mask
                count = int(combined.sum())
                pct = round(count / len(s) * 100, 2) if len(s) else 0
                examples = s[combined].head(5).tolist()
                outliers[col] = {"count": count, "pct": pct,
                                 "method": "IQR+zscore", "examples": examples}
                col_acc = 100 - pct
                total_accuracy += col_acc
                scored_cols += 1

            for col in df.select_dtypes(include="object").columns:
                s = df[col].dropna().astype(str).str.strip().str.lower()
                hits = s[s.isin(_SUSPECT_VALUES) | (s.str.len() <= 1)]
                if len(hits):
                    suspect_values[col] = {
                        "count": len(hits),
                        "values": list(hits.unique()[:10]),
                    }
                    col_acc = round((1 - len(hits) / len(s)) * 100, 2) if len(s) else 100
                    total_accuracy += col_acc
                    scored_cols += 1

            score = round(total_accuracy / scored_cols, 2) if scored_cols else 100.0
            status = ("PASS" if score >= 95 else "WARN" if score >= 85 else "FAIL")
            return {"score": score, "status": status,
                    "outliers": outliers, "suspect_values": suspect_values}
        except Exception as e:
            print(f"[DQ ENGINE] Accuracy scoring failed: {e}")
            return {"score": None, "status": "ERROR",
                    "outliers": {}, "suspect_values": {}}

    def _score_consistency(self, df):
        try:
            format_issues = {}
            cross_issues = []
            scored_cols = 0
            total_consistency = 0.0

            for col in df.select_dtypes(include="object").columns:
                s = df[col].dropna().astype(str)
                if len(s) < 5:
                    continue
                patterns = {
                    "all_upper": s.str.match(r"^[A-Z0-9\s\-]+$"),
                    "all_lower": s.str.match(r"^[a-z0-9\s\-]+$"),
                    "title_case": s.str.match(r"^[A-Z][a-z]"),
                    "email": s.str.match(r"^[^@]+@[^@]+\.[^@]+$"),
                    "numeric_string": s.str.match(r"^\d+\.?\d*$"),
                }
                best_name, best_pct = "mixed", 0.0
                for pname, mask in patterns.items():
                    pct = mask.mean() * 100 if mask.any() else 0
                    if pct > best_pct:
                        best_pct = pct
                        best_name = pname
                conforming = round(best_pct, 2)
                if conforming < 90:
                    non_conform = s[~patterns.get(best_name,
                                                  pd.Series(True, index=s.index))]
                    format_issues[col] = {
                        "dominant_pattern": best_name,
                        "conforming_pct": conforming,
                        "non_conforming_examples": list(
                            non_conform.head(5).values),
                    }
                total_consistency += conforming
                scored_cols += 1

            # Cross-column: start_date <= end_date
            cols_lower = {c.lower(): c for c in df.columns}
            for prefix in ("start", "begin", "open"):
                for suffix in ("end", "close", "finish"):
                    s_col = None
                    e_col = None
                    for cl, orig in cols_lower.items():
                        if prefix in cl and "date" in cl:
                            s_col = orig
                        if suffix in cl and "date" in cl:
                            e_col = orig
                    if s_col and e_col and s_col != e_col:
                        try:
                            sd = pd.to_datetime(df[s_col], errors="coerce")
                            ed = pd.to_datetime(df[e_col], errors="coerce")
                            bad = ((sd > ed) & sd.notna() & ed.notna()).sum()
                            if bad:
                                cross_issues.append(
                                    f"{s_col} > {e_col} in {bad} rows")
                        except Exception:
                            pass

            score = round(total_consistency / scored_cols, 2) if scored_cols else 100.0
            status = ("PASS" if score >= 95 else "WARN" if score >= 85 else "FAIL")
            return {"score": score, "status": status,
                    "format_issues": format_issues,
                    "cross_column_issues": cross_issues}
        except Exception as e:
            print(f"[DQ ENGINE] Consistency scoring failed: {e}")
            return {"score": None, "status": "ERROR",
                    "format_issues": {}, "cross_column_issues": []}

    def _score_timeliness(self, df):
        try:
            dt_cols = {}
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_cols[col] = df[col]
                elif df[col].dtype == object:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(df[col], errors="coerce",
                                                infer_datetime_format=True)
                    if parsed.notna().mean() > 0.5:
                        dt_cols[col] = parsed

            if not dt_cols:
                return {"score": None, "status": "N/A",
                        "datetime_columns": {},
                        "note": "No temporal columns detected"}

            now = pd.Timestamp.now(tz="UTC")
            col_details = {}
            staleness_scores = []
            for col, series in dt_cols.items():
                s = series.dropna()
                if len(s) == 0:
                    continue
                most_recent = s.max()
                if most_recent.tzinfo is None:
                    most_recent = most_recent.tz_localize("UTC")
                staleness_h = (now - most_recent).total_seconds() / 3600
                # Gap analysis
                sorted_s = s.sort_values()
                diffs = sorted_s.diff().dropna()
                median_gap = diffs.median()
                gaps = 0
                largest_gap = pd.Timedelta(0)
                if median_gap and median_gap.total_seconds() > 0:
                    big = diffs[diffs > 2 * median_gap]
                    gaps = len(big)
                    if len(big):
                        largest_gap = big.max()
                col_details[col] = {
                    "most_recent": str(most_recent)[:19],
                    "staleness_hours": round(staleness_h, 1),
                    "gaps_detected": gaps,
                    "largest_gap": str(largest_gap),
                }
                # Score: 100 if fresh, degrades with staleness
                if staleness_h <= self.thresholds["freshness_hours_warn"]:
                    staleness_scores.append(100.0)
                elif staleness_h <= self.thresholds["freshness_hours_fail"]:
                    staleness_scores.append(70.0)
                else:
                    staleness_scores.append(40.0)

            score = round(np.mean(staleness_scores), 2) if staleness_scores else None
            status = ("PASS" if score and score >= 95
                      else "WARN" if score and score >= 70
                      else "FAIL" if score else "N/A")
            return {"score": score, "status": status,
                    "datetime_columns": col_details}
        except Exception as e:
            print(f"[DQ ENGINE] Timeliness scoring failed: {e}")
            return {"score": None, "status": "ERROR", "datetime_columns": {}}

    def _score_uniqueness(self, df):
        try:
            col_uniq = {}
            potential_pks = []
            total = len(df)
            for col in df.columns:
                unique = df[col].nunique()
                dup = total - unique
                dup_pct = round(dup / total * 100, 2) if total else 0
                col_uniq[col] = {
                    "unique_count": int(unique),
                    "duplicate_count": int(dup),
                    "uniqueness_pct": round(unique / total * 100, 2) if total else 0,
                }
                if unique == total and df[col].isna().sum() == 0:
                    potential_pks.append(col)

            full_dups = int(df.duplicated().sum())
            full_dup_pct = round(full_dups / total * 100, 2) if total else 0
            score = round(100 - full_dup_pct, 2)
            status = ("PASS" if score >= self.thresholds["uniqueness_warn"]
                      else "WARN" if score >= self.thresholds["uniqueness_fail"]
                      else "FAIL")
            return {
                "score": score, "status": status,
                "full_row_duplicates": {"count": full_dups, "pct": full_dup_pct},
                "column_uniqueness": col_uniq,
                "potential_primary_keys": potential_pks,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Uniqueness scoring failed: {e}")
            return {"score": None, "status": "ERROR",
                    "full_row_duplicates": {}, "column_uniqueness": {},
                    "potential_primary_keys": []}

    def _score_validity(self, df):
        try:
            col_validity = {}
            scored = 0
            total_validity = 0.0

            for col in df.columns:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                rule = None
                violations = 0
                examples = []
                name_l = col.lower()

                if pd.api.types.is_numeric_dtype(df[col]):
                    if any(h in name_l for h in _POSITIVE_NAME_HINTS):
                        neg = s[s < 0]
                        violations = len(neg)
                        rule = "expected_positive"
                        examples = neg.head(5).tolist()

                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    future = s[s > pd.Timestamp.now()]
                    old = s[s < pd.Timestamp("1900-01-01")]
                    violations = len(future) + len(old)
                    rule = "date_range"
                    examples = list(future.head(3).astype(str)) + list(
                        old.head(2).astype(str))

                elif df[col].dtype == object:
                    str_s = s.astype(str)
                    # Whitespace-only
                    ws_only = str_s[str_s.str.strip() == ""]
                    violations += len(ws_only)

                    if "email" in name_l:
                        rule = "email_format"
                        bad = str_s[~str_s.str.match(
                            r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False)]
                        violations = len(bad)
                        examples = list(bad.head(5).values)
                    elif "phone" in name_l or "tel" in name_l:
                        rule = "phone_format"
                        digits = str_s.str.replace(r"\D", "", regex=True)
                        bad = str_s[digits.str.len() < 7]
                        violations = len(bad)
                        examples = list(bad.head(5).values)
                    elif "zip" in name_l or "postal" in name_l:
                        rule = "postal_format"
                        bad = str_s[~str_s.str.match(
                            r"^\d{4,10}$|^\d{5}-\d{4}$|^[A-Z]\d[A-Z]\s?\d[A-Z]\d$",
                            na=False)]
                        violations = len(bad)
                        examples = list(bad.head(5).values)
                    elif "url" in name_l or "website" in name_l:
                        rule = "url_format"
                        bad = str_s[~str_s.str.match(
                            r"^https?://", na=False)]
                        violations = len(bad)
                        examples = list(bad.head(5).values)
                    else:
                        rule = "whitespace_only"
                        examples = list(ws_only.head(5).values)

                if rule:
                    pct_valid = round((1 - violations / len(s)) * 100, 2)
                    col_validity[col] = {
                        "score": pct_valid, "rule": rule,
                        "violations": int(violations),
                        "examples": examples[:5],
                    }
                    total_validity += pct_valid
                    scored += 1

            score = round(total_validity / scored, 2) if scored else 100.0
            status = ("PASS" if score >= self.thresholds["validity_warn"]
                      else "WARN" if score >= self.thresholds["validity_fail"]
                      else "FAIL")
            return {"score": score, "status": status,
                    "column_validity": col_validity}
        except Exception as e:
            print(f"[DQ ENGINE] Validity scoring failed: {e}")
            return {"score": None, "status": "ERROR", "column_validity": {}}

    # ------------------------------------------------------------------ #
    #  Column Profiler                                                    #
    # ------------------------------------------------------------------ #

    def _profile_columns(self, df):
        try:
            profiles = []
            for col in df.columns:
                s = df[col]
                null_ct = int(s.isna().sum())
                unique_ct = int(s.nunique())
                total = len(df)
                prof = {
                    "name": col,
                    "dtype": str(s.dtype),
                    "null_count": null_ct,
                    "null_pct": round(null_ct / total * 100, 2) if total else 0,
                    "unique_count": unique_ct,
                    "unique_pct": round(unique_ct / total * 100, 2) if total else 0,
                    "sample_values": list(s.dropna().head(5).astype(str)),
                    "is_potential_pk": (unique_ct == total and null_ct == 0),
                    "is_potential_fk": bool(re.search(
                        r"(_id|_key|_code|Id$|Key$|Code$)", col)),
                    "stats": {},
                }
                if pd.api.types.is_numeric_dtype(s):
                    prof["stats"] = {
                        "min": float(s.min()) if s.notna().any() else None,
                        "max": float(s.max()) if s.notna().any() else None,
                        "mean": round(float(s.mean()), 4) if s.notna().any() else None,
                        "std": round(float(s.std()), 4) if s.notna().any() else None,
                    }
                elif s.dtype == object:
                    lengths = s.dropna().astype(str).str.len()
                    prof["stats"] = {
                        "min_length": int(lengths.min()) if len(lengths) else 0,
                        "max_length": int(lengths.max()) if len(lengths) else 0,
                        "avg_length": round(float(lengths.mean()), 1) if len(lengths) else 0,
                    }
                profiles.append(prof)
            return profiles
        except Exception as e:
            print(f"[DQ ENGINE] Column profiling failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  Recommendations                                                    #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(self, result):
        recs = []
        dims = result.get("dimensions", {})

        for dim_name, dim_data in dims.items():
            score = dim_data.get("score")
            if score is None:
                continue
            if score < 80:
                priority = "CRITICAL"
            elif score < 90:
                priority = "HIGH"
            elif score < 95:
                priority = "MEDIUM"
            else:
                continue

            finding = f"{dim_name.title()} score is {score}%"
            rec = impact = ""
            auto = False
            fix_type = None

            if dim_name == "completeness":
                worst = sorted(
                    dim_data.get("columns", {}).items(),
                    key=lambda x: x[1].get("score", 100))[:3]
                cols_str = ", ".join(f"{c} ({d['score']}%)" for c, d in worst)
                finding += f". Worst columns: {cols_str}"
                rec = "Investigate upstream ETL for dropped records. Consider imputation for non-critical fields."
                impact = "Incomplete data leads to biased analysis and incorrect aggregations."
                auto = True
                fix_type = "imputation"

            elif dim_name == "accuracy":
                outlier_cols = list(dim_data.get("outliers", {}).keys())[:3]
                suspect_cols = list(dim_data.get("suspect_values", {}).keys())[:3]
                if outlier_cols:
                    finding += f". Outlier columns: {', '.join(outlier_cols)}"
                if suspect_cols:
                    finding += f". Suspect value columns: {', '.join(suspect_cols)}"
                rec = "Review outlier values for data entry errors. Validate against authoritative sources."
                impact = "Inaccurate data produces misleading KPIs and erroneous business decisions."
                auto = True
                fix_type = "remove_outliers"

            elif dim_name == "consistency":
                issues = list(dim_data.get("format_issues", {}).keys())[:3]
                finding += f". Inconsistent columns: {', '.join(issues)}" if issues else ""
                rec = "Standardize formats at the source. Apply data validation rules in ETL pipeline."
                impact = "Inconsistent formats cause join failures and aggregation errors."
                auto = True
                fix_type = "format_standardize"

            elif dim_name == "uniqueness":
                dup_info = dim_data.get("full_row_duplicates", {})
                dup_ct = dup_info.get("count", 0)
                finding += f". {dup_ct} full-row duplicates detected"
                rec = "Implement deduplication in ETL. Add unique constraints at database level."
                impact = "Duplicate records inflate counts and skew averages."
                auto = True
                fix_type = "dedup"

            elif dim_name == "validity":
                bad_cols = [
                    c for c, d in dim_data.get("column_validity", {}).items()
                    if d.get("violations", 0) > 0
                ][:3]
                finding += f". Columns with violations: {', '.join(bad_cols)}" if bad_cols else ""
                rec = "Add domain-level validation rules. Enforce constraints at data entry point."
                impact = "Invalid data fails downstream business rules and reporting logic."

            elif dim_name == "timeliness":
                stale = [
                    f"{c} ({d['staleness_hours']:.0f}h)"
                    for c, d in dim_data.get("datetime_columns", {}).items()
                    if d.get("staleness_hours", 0) > self.thresholds["freshness_hours_warn"]
                ]
                finding += f". Stale columns: {', '.join(stale)}" if stale else ""
                rec = "Review data pipeline refresh schedules. Set up freshness SLA monitoring."
                impact = "Stale data leads to decisions based on outdated information."

            recs.append({
                "priority": priority,
                "dimension": dim_name,
                "finding": finding,
                "recommendation": rec,
                "impact": impact,
                "auto_fixable": auto,
                "fix_type": fix_type,
            })

        recs.sort(key=lambda r: {"CRITICAL": 0, "HIGH": 1,
                                  "MEDIUM": 2, "LOW": 3}[r["priority"]])
        result["recommendations"] = recs
        result["remediation_actions"] = [r for r in recs if r["auto_fixable"]]

    # ------------------------------------------------------------------ #
    #  Cross-Table Analysis                                               #
    # ------------------------------------------------------------------ #

    def _cross_table_analysis(self, tables_dict):
        try:
            orphan_fks = []
            detected_rels = []
            all_cols = {}
            for tname, df in tables_dict.items():
                for col in df.columns:
                    all_cols.setdefault(col, []).append(tname)

            # Detect potential joins by column name overlap
            seen_pairs = set()
            for col, tables in all_cols.items():
                if len(tables) >= 2:
                    for i, t1 in enumerate(tables):
                        for t2 in tables[i + 1:]:
                            pair = tuple(sorted([t1, t2]))
                            if (pair, col) in seen_pairs:
                                continue
                            seen_pairs.add((pair, col))
                            df1 = tables_dict[t1]
                            df2 = tables_dict[t2]
                            vals1 = set(df1[col].dropna().unique())
                            vals2 = set(df2[col].dropna().unique())
                            if not vals1 or not vals2:
                                continue
                            overlap = len(vals1 & vals2) / min(len(vals1), len(vals2))
                            if overlap > 0.3:
                                detected_rels.append({
                                    "from_table": t1, "from_col": col,
                                    "to_table": t2, "to_col": col,
                                    "confidence": round(overlap * 100, 1),
                                })
                                # Check orphans
                                orphans_1 = vals1 - vals2
                                orphans_2 = vals2 - vals1
                                if orphans_1 and len(orphans_1) < len(vals1) * 0.5:
                                    orphan_fks.append(
                                        f"{t1}.{col}: {len(orphans_1)} values "
                                        f"not in {t2}.{col}")
                                if orphans_2 and len(orphans_2) < len(vals2) * 0.5:
                                    orphan_fks.append(
                                        f"{t2}.{col}: {len(orphans_2)} values "
                                        f"not in {t1}.{col}")

            # Naming consistency
            all_names = []
            for df in tables_dict.values():
                all_names.extend(df.columns.tolist())
            snake = sum(1 for n in all_names if re.match(r"^[a-z_]+$", n))
            camel = sum(1 for n in all_names if re.match(r"^[a-z]+[A-Z]", n))
            naming_score = round(max(snake, camel) / len(all_names) * 100, 1) if all_names else 100

            schema_health = round(
                (naming_score + (100 if not orphan_fks else 60)) / 2, 1)

            return {
                "orphan_fks": orphan_fks,
                "naming_consistency": naming_score,
                "detected_relationships": detected_rels,
                "schema_health_score": schema_health,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Cross-table analysis failed: {e}")
            return {"orphan_fks": [], "naming_consistency": 0,
                    "detected_relationships": [], "schema_health_score": 0}

    # ------------------------------------------------------------------ #
    #  HTML Scorecard Export                                               #
    # ------------------------------------------------------------------ #

    def generate_html_scorecard(self):
        """Generate a self-contained HTML scorecard report. WU dark theme."""
        if not self.scan_results:
            return ""

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        sc = self.generate_scorecard_data()
        overall = sc.get("overall_score", 0)

        def _color(val, green=90, yellow=70):
            if val >= green:
                return "#238636"
            if val >= yellow:
                return "#FFE600"
            return "#da3633"

        def _svg_gauge(label, score, size=90):
            if score is None:
                score = 0
            r = 36
            circ = 2 * 3.14159 * r
            fill = circ * score / 100
            c = _color(score, 95, 80)
            return (
                f'<div style="text-align:center;margin:8px">'
                f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
                f'<circle cx="{size//2}" cy="{size//2}" r="{r}" '
                f'fill="none" stroke="#333" stroke-width="7"/>'
                f'<circle cx="{size//2}" cy="{size//2}" r="{r}" '
                f'fill="none" stroke="{c}" stroke-width="7" '
                f'stroke-dasharray="{fill:.1f} {circ:.1f}" '
                f'stroke-linecap="round" '
                f'transform="rotate(-90 {size//2} {size//2})"/>'
                f'<text x="{size//2}" y="{size//2+2}" text-anchor="middle" '
                f'font-size="16" font-weight="bold" fill="{c}" '
                f'font-family="Inter,system-ui,sans-serif">'
                f'{score:.0f}</text>'
                f'</svg>'
                f'<div style="font-size:11px;color:#aaa;margin-top:2px">'
                f'{label}</div></div>'
            )

        def _bar(val, width=120):
            if val is None:
                val = 0
            c = _color(val, 95, 80)
            w = max(0, min(100, val))
            return (
                f'<div style="background:#333;border-radius:4px;'
                f'width:{width}px;height:14px;display:inline-block;'
                f'vertical-align:middle">'
                f'<div style="background:{c};width:{w}%;height:100%;'
                f'border-radius:4px"></div></div>'
                f' <span style="font-size:12px;color:#ccc">{val:.0f}%</span>'
            )

        # ── Dimension gauges ──
        dim_names = [
            "completeness", "accuracy", "consistency",
            "timeliness", "uniqueness", "validity",
        ]
        avg_dims = {}
        for dim in dim_names:
            scores = []
            for res in self.scan_results.values():
                d = res.get("dimensions", {}).get(dim, {})
                s = d.get("score") if isinstance(d, dict) else None
                if s is not None:
                    scores.append(s)
            avg_dims[dim] = round(sum(scores) / len(scores), 1) if scores else 0

        gauges_html = "".join(
            _svg_gauge(d.title(), avg_dims[d]) for d in dim_names)

        # ── Table summary cards ──
        table_cards = ""
        for tname, result in self.scan_results.items():
            ts_score = result.get("overall_score", 0)
            tc = _color(ts_score)
            rows = result.get("row_count", "?")
            cols = result.get("column_count", "?")
            dims = result.get("dimensions", {})
            dim_bars = ""
            for d in dim_names:
                dd = dims.get(d, {})
                ds = dd.get("score", 0) if isinstance(dd, dict) else 0
                dim_bars += (
                    f'<div style="display:flex;align-items:center;'
                    f'margin:3px 0">'
                    f'<span style="width:100px;font-size:12px;color:#aaa">'
                    f'{d.title()}</span>{_bar(ds)}</div>'
                )
            table_cards += (
                f'<div style="background:#1A1A1A;border:1px solid #333;'
                f'border-radius:8px;padding:16px;margin:10px 0">'
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:center;margin-bottom:10px">'
                f'<span style="font-size:16px;font-weight:600;color:#fff">'
                f'{tname}</span>'
                f'<span style="font-size:24px;font-weight:bold;color:{tc}">'
                f'{ts_score:.1f}</span></div>'
                f'<div style="font-size:12px;color:#888;margin-bottom:8px">'
                f'{rows} rows | {cols} columns</div>'
                f'{dim_bars}</div>'
            )

        # ── Top Issues ──
        all_recs = []
        for tname, result in self.scan_results.items():
            for rec in result.get("recommendations", []):
                r = dict(rec)
                r["table"] = tname
                all_recs.append(r)
        all_recs.sort(key=lambda r: {
            "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3
        }.get(r.get("priority", "LOW"), 4))

        pri_colors = {
            "CRITICAL": "#da3633", "HIGH": "#d29922",
            "MEDIUM": "#FFE600", "LOW": "#238636",
        }
        issues_html = ""
        for rec in all_recs[:15]:
            pri = rec.get("priority", "LOW")
            pc = pri_colors.get(pri, "#888")
            fix_tag = (
                '<span style="background:#238636;color:#fff;'
                'padding:1px 6px;border-radius:3px;font-size:10px;'
                'margin-left:6px">AUTO-FIX</span>'
                if rec.get("auto_fixable") else ""
            )
            issues_html += (
                f'<div style="border-left:3px solid {pc};padding:8px 12px;'
                f'margin:6px 0;background:#1A1A1A;border-radius:0 6px 6px 0">'
                f'<span style="color:{pc};font-weight:bold;font-size:12px">'
                f'[{pri}]</span> '
                f'<span style="color:#ccc;font-size:12px">{rec.get("table","")}</span>'
                f'{fix_tag}'
                f'<div style="color:#fff;font-size:13px;margin-top:4px">'
                f'{rec.get("finding","")}</div>'
                f'<div style="color:#888;font-size:12px;margin-top:2px">'
                f'{rec.get("recommendation","")}</div></div>'
            )

        # ── Column-level heatmap ──
        heatmap_html = ""
        for tname, result in self.scan_results.items():
            dims = result.get("dimensions", {})
            comp_cols = dims.get("completeness", {}).get("columns", {})
            valid_cols = dims.get("validity", {}).get("column_rules", {})

            if not comp_cols:
                continue

            rows_html = ""
            for col_name, col_data in comp_cols.items():
                comp_s = col_data.get("score", 100)
                comp_c = _color(comp_s, 95, 80)
                # Check validity for this column
                v_data = valid_cols.get(col_name, {})
                v_score = v_data.get("valid_pct", 100) if isinstance(v_data, dict) else 100
                v_c = _color(v_score, 95, 80)

                rows_html += (
                    f'<tr>'
                    f'<td style="padding:4px 8px;color:#ccc;font-size:12px;'
                    f'border-bottom:1px solid #333">{col_name}</td>'
                    f'<td style="padding:4px 8px;text-align:center;'
                    f'background:{comp_c}22;color:{comp_c};font-size:12px;'
                    f'border-bottom:1px solid #333">{comp_s:.0f}%</td>'
                    f'<td style="padding:4px 8px;text-align:center;'
                    f'background:{v_c}22;color:{v_c};font-size:12px;'
                    f'border-bottom:1px solid #333">{v_score:.0f}%</td>'
                    f'</tr>'
                )

            heatmap_html += (
                f'<div style="margin:12px 0">'
                f'<div style="font-size:14px;color:#FFE600;margin-bottom:6px;'
                f'font-weight:600">{tname}</div>'
                f'<table style="width:100%;border-collapse:collapse">'
                f'<tr style="background:#262626">'
                f'<th style="padding:6px 8px;text-align:left;color:#888;'
                f'font-size:11px;font-weight:600">Column</th>'
                f'<th style="padding:6px 8px;text-align:center;color:#888;'
                f'font-size:11px;font-weight:600">Completeness</th>'
                f'<th style="padding:6px 8px;text-align:center;color:#888;'
                f'font-size:11px;font-weight:600">Validity</th>'
                f'</tr>{rows_html}</table></div>'
            )

        # ── Cross-table analysis ──
        cross_html = ""
        cross = getattr(self, "_last_cross_results", None)
        if cross:
            rels = cross.get("detected_relationships", [])
            orphans = cross.get("orphan_fks", [])
            naming = cross.get("naming_consistency", 0)

            if rels:
                cross_html += (
                    '<div style="font-size:14px;color:#FFE600;'
                    'margin:12px 0 6px;font-weight:600">'
                    'Detected Relationships</div>')
                for rel in rels:
                    cross_html += (
                        f'<div style="color:#ccc;font-size:12px;'
                        f'padding:3px 0">'
                        f'{rel["from_table"]}.{rel["from_col"]} '
                        f'&harr; {rel["to_table"]}.{rel["to_col"]} '
                        f'({rel["confidence"]}% confidence)</div>'
                    )

            if orphans:
                cross_html += (
                    '<div style="font-size:14px;color:#da3633;'
                    'margin:12px 0 6px;font-weight:600">'
                    'Orphan Foreign Keys</div>')
                for o in orphans:
                    cross_html += (
                        f'<div style="color:#ccc;font-size:12px;'
                        f'padding:3px 0">{o}</div>')

            cross_html += (
                f'<div style="color:#888;font-size:12px;margin-top:8px">'
                f'Naming Consistency: {naming}%</div>')

        # ── Assemble full HTML ──
        oc = _color(overall)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Quality Scorecard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0D0D0D;color:#fff;font-family:Inter,system-ui,-apple-system,sans-serif;
padding:24px;line-height:1.5}}
.container{{max-width:1100px;margin:0 auto}}
.header{{text-align:center;padding:30px 0;border-bottom:2px solid #FFE600}}
.header h1{{font-size:22px;color:#FFE600;font-weight:700;letter-spacing:0.5px}}
.header .ts{{font-size:12px;color:#666;margin-top:4px}}
.big-score{{font-size:72px;font-weight:800;margin:16px 0 4px}}
.section{{margin:28px 0}}
.section-title{{font-size:16px;font-weight:700;color:#FFE600;
border-left:3px solid #FFE600;padding-left:10px;margin-bottom:14px}}
.gauges{{display:flex;justify-content:center;flex-wrap:wrap;gap:12px}}
.footer{{text-align:center;color:#555;font-size:11px;margin-top:40px;
padding-top:16px;border-top:1px solid #333}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Data Quality Scorecard -- DAMA-DMBOK Assessment</h1>
<div class="ts">{ts}</div>
<div class="big-score" style="color:{oc}">{overall:.1f}</div>
<div style="color:#888;font-size:13px">Overall Quality Score / 100</div>
</div>

<div class="section">
<div class="section-title">DAMA Dimension Scores</div>
<div class="gauges">{gauges_html}</div>
</div>

<div class="section">
<div class="section-title">Table Summary</div>
{table_cards}
</div>

<div class="section">
<div class="section-title">Top Issues</div>
{issues_html if issues_html else '<div style="color:#888">No significant issues detected.</div>'}
</div>

<div class="section">
<div class="section-title">Column-Level Heatmap</div>
{heatmap_html if heatmap_html else '<div style="color:#888">No column-level data available.</div>'}
</div>

{"<div class='section'><div class='section-title'>Cross-Table Analysis</div>" + cross_html + "</div>" if cross_html else ""}

<div class="footer">
Generated by Dr. Data -- DAMA-DMBOK Data Quality Assessment Engine
</div>
</div>
</body>
</html>"""
        return html

    # ------------------------------------------------------------------ #
    #  Monte Carlo-Style Observability                                     #
    # ------------------------------------------------------------------ #

    def detect_schema_drift(self, df, table_name, baseline=None):
        """Compare current schema against a stored baseline."""
        try:
            # Resolve baseline
            if baseline is None:
                baselines = getattr(self, "baselines", {})
                baseline = baselines.get(table_name)
            if baseline is None and table_name in self.scan_results:
                # Fall back to previous scan column_details
                baseline = {
                    "column_details": self.scan_results[table_name].get(
                        "column_details", []),
                }

            if not baseline:
                return {
                    "has_drift": False, "new_columns": [],
                    "removed_columns": [], "type_changes": [],
                    "nullable_changes": [], "severity": "OK",
                    "message": "No baseline available for comparison",
                }

            base_cols = {
                c["name"]: c
                for c in baseline.get("column_details", [])
            }
            curr_cols = {}
            for col in df.columns:
                s = df[col]
                curr_cols[col] = {
                    "name": col,
                    "dtype": str(s.dtype),
                    "null_count": int(s.isna().sum()),
                }

            new_columns = [c for c in curr_cols if c not in base_cols]
            removed_columns = [c for c in base_cols if c not in curr_cols]

            type_changes = []
            nullable_changes = []
            for col in curr_cols:
                if col in base_cols:
                    old_dtype = base_cols[col].get("dtype", "")
                    new_dtype = curr_cols[col]["dtype"]
                    if old_dtype != new_dtype:
                        type_changes.append({
                            "column": col,
                            "old_type": old_dtype,
                            "new_type": new_dtype,
                        })
                    was_nullable = base_cols[col].get("null_count", 0) > 0
                    now_nullable = curr_cols[col]["null_count"] > 0
                    if was_nullable != now_nullable:
                        nullable_changes.append({
                            "column": col,
                            "was_nullable": was_nullable,
                            "now_nullable": now_nullable,
                        })

            has_drift = bool(
                new_columns or removed_columns
                or type_changes or nullable_changes
            )
            if removed_columns or type_changes:
                severity = "CRITICAL"
            elif new_columns:
                severity = "WARN"
            else:
                severity = "OK"

            return {
                "has_drift": has_drift,
                "new_columns": new_columns,
                "removed_columns": removed_columns,
                "type_changes": type_changes,
                "nullable_changes": nullable_changes,
                "severity": severity,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Schema drift detection failed: {e}")
            return {
                "has_drift": False, "new_columns": [],
                "removed_columns": [], "type_changes": [],
                "nullable_changes": [], "severity": "ERROR",
            }

    def detect_volume_anomaly(self, df, table_name, expected_range=None):
        """Detect row-count anomalies vs baseline or expected range."""
        try:
            current_rows = len(df)
            previous_rows = None
            change_pct = None

            baselines = getattr(self, "baselines", {})
            if table_name in baselines:
                previous_rows = baselines[table_name].get("row_count")
            elif table_name in self.scan_results:
                previous_rows = self.scan_results[table_name].get("row_count")

            is_anomaly = False
            severity = "OK"

            if expected_range:
                min_r, max_r = expected_range
                if current_rows < min_r or current_rows > max_r:
                    is_anomaly = True
                    severity = "CRITICAL"

            if previous_rows and previous_rows > 0:
                change_pct = round(
                    (current_rows - previous_rows) / previous_rows * 100, 1)
                abs_change = abs(change_pct)
                if abs_change > 50:
                    is_anomaly = True
                    severity = "CRITICAL"
                elif abs_change > 20:
                    is_anomaly = True
                    severity = "WARN" if severity == "OK" else severity

            return {
                "current_rows": current_rows,
                "previous_rows": previous_rows,
                "change_pct": change_pct,
                "is_anomaly": is_anomaly,
                "severity": severity,
                "expected_range": expected_range,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Volume anomaly detection failed: {e}")
            return {
                "current_rows": len(df), "previous_rows": None,
                "change_pct": None, "is_anomaly": False,
                "severity": "ERROR", "expected_range": expected_range,
            }

    def detect_distribution_drift(self, df, table_name):
        """Detect statistical distribution drift vs stored baseline."""
        try:
            baselines = getattr(self, "baselines", {})
            baseline = baselines.get(table_name)

            numeric_drift = {}
            categorical_drift = {}
            drifted_columns = []
            drift_scores = []

            for col in df.columns:
                s = df[col].dropna()
                if len(s) == 0:
                    continue

                if pd.api.types.is_numeric_dtype(s):
                    if (baseline
                            and col in baseline.get(
                                "column_distributions", {})):
                        base_dist = baseline["column_distributions"][col]
                        if base_dist.get("type") == "numeric":
                            # Synthetic comparison using stored stats
                            base_mean = base_dist.get("mean", 0)
                            base_std = base_dist.get("std", 1) or 1
                            # Generate synthetic baseline sample
                            np.random.seed(42)
                            synth = np.random.normal(
                                base_mean, base_std, min(len(s), 1000))
                            curr_sample = s.sample(
                                n=min(len(s), 1000),
                                random_state=42).values
                            ks_stat, p_val = stats.ks_2samp(
                                curr_sample, synth)
                            drifted = p_val < 0.05
                            numeric_drift[col] = {
                                "ks_statistic": round(float(ks_stat), 4),
                                "p_value": round(float(p_val), 6),
                                "drifted": drifted,
                            }
                            if drifted:
                                drifted_columns.append(col)
                            drift_scores.append(
                                100.0 if not drifted else max(0, p_val * 100))
                    else:
                        # No baseline -- report current stats only
                        numeric_drift[col] = {
                            "ks_statistic": None,
                            "p_value": None,
                            "drifted": False,
                            "note": "No baseline for comparison",
                        }
                        drift_scores.append(100.0)

                elif s.dtype == object:
                    curr_cats = set(s.unique())
                    if (baseline
                            and col in baseline.get(
                                "column_distributions", {})):
                        base_dist = baseline["column_distributions"][col]
                        if base_dist.get("type") == "categorical":
                            base_cats = set(
                                base_dist.get("top_values", {}).keys())
                            new_cats = list(curr_cats - base_cats)[:10]
                            removed_cats = list(base_cats - curr_cats)[:10]
                            base_dom = base_dist.get("dominant_category")
                            curr_dom = s.value_counts().index[0] if len(
                                s) > 0 else None
                            dom_changed = (
                                base_dom is not None
                                and curr_dom is not None
                                and str(base_dom) != str(curr_dom)
                            )
                            categorical_drift[col] = {
                                "new_categories": new_cats,
                                "removed_categories": removed_cats,
                                "dominant_changed": dom_changed,
                            }
                            if new_cats or removed_cats or dom_changed:
                                drifted_columns.append(col)
                                drift_scores.append(50.0)
                            else:
                                drift_scores.append(100.0)
                    else:
                        categorical_drift[col] = {
                            "new_categories": [],
                            "removed_categories": [],
                            "dominant_changed": False,
                            "note": "No baseline for comparison",
                        }
                        drift_scores.append(100.0)

            overall_drift_score = round(
                float(np.mean(drift_scores)), 1
            ) if drift_scores else 100.0

            return {
                "numeric_drift": numeric_drift,
                "categorical_drift": categorical_drift,
                "overall_drift_score": overall_drift_score,
                "drifted_columns": drifted_columns,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Distribution drift detection failed: {e}")
            return {
                "numeric_drift": {}, "categorical_drift": {},
                "overall_drift_score": 100.0, "drifted_columns": [],
            }

    def store_baseline(self, table_name, df=None):
        """Store current scan result as baseline snapshot for future drift detection."""
        try:
            if table_name not in self.scan_results:
                print(f"[DQ ENGINE] No scan results for {table_name} to baseline")
                return

            if not hasattr(self, "baselines"):
                self.baselines = {}

            result = self.scan_results[table_name]
            col_distributions = {}

            # Build column distribution snapshots
            details = result.get("column_details", [])
            for prof in details:
                col_name = prof["name"]
                st = prof.get("stats", {})
                if "mean" in st:
                    # Numeric column
                    col_distributions[col_name] = {
                        "type": "numeric",
                        "mean": st.get("mean"),
                        "std": st.get("std"),
                        "min": st.get("min"),
                        "max": st.get("max"),
                    }
                    # Add quartiles if df provided
                    if df is not None and col_name in df.columns:
                        s = df[col_name].dropna()
                        if len(s) > 0 and pd.api.types.is_numeric_dtype(s):
                            q = s.quantile([0.25, 0.5, 0.75]).to_dict()
                            col_distributions[col_name]["q25"] = round(
                                float(q.get(0.25, 0)), 4)
                            col_distributions[col_name]["q50"] = round(
                                float(q.get(0.5, 0)), 4)
                            col_distributions[col_name]["q75"] = round(
                                float(q.get(0.75, 0)), 4)
                elif "min_length" in st:
                    # Categorical column -- store top values
                    if df is not None and col_name in df.columns:
                        vc = df[col_name].dropna().value_counts().head(20)
                        dominant = vc.index[0] if len(vc) > 0 else None
                        col_distributions[col_name] = {
                            "type": "categorical",
                            "top_values": {
                                str(k): int(v) for k, v in vc.items()},
                            "dominant_category": str(dominant)
                            if dominant is not None else None,
                        }

            dim_scores = {}
            for k, v in result.get("dimensions", {}).items():
                dim_scores[k] = v.get("score") if isinstance(v, dict) else None

            self.baselines[table_name] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "row_count": result.get("row_count"),
                "column_details": result.get("column_details", []),
                "dimension_scores": dim_scores,
                "column_distributions": col_distributions,
            }
            print(f"[DQ ENGINE] Baseline stored for {table_name} "
                  f"({len(col_distributions)} columns profiled)")
        except Exception as e:
            print(f"[DQ ENGINE] Baseline storage failed: {e}")

    def generate_observability_report(self, df, table_name):
        """Run all three observability checks and return a combined report."""
        try:
            schema = self.detect_schema_drift(df, table_name)
            volume = self.detect_volume_anomaly(df, table_name)
            distribution = self.detect_distribution_drift(df, table_name)

            issues = []
            if schema.get("severity") == "CRITICAL":
                issues.append("Schema drift (CRITICAL)")
            elif schema.get("severity") == "WARN":
                issues.append("Schema drift (WARN)")
            if volume.get("severity") == "CRITICAL":
                issues.append("Volume anomaly (CRITICAL)")
            elif volume.get("severity") == "WARN":
                issues.append("Volume anomaly (WARN)")
            if distribution.get("drifted_columns"):
                issues.append(
                    f"Distribution drift in "
                    f"{len(distribution['drifted_columns'])} column(s)")

            if any("CRITICAL" in i for i in issues):
                health = "CRITICAL"
            elif issues:
                health = "WARN"
            else:
                health = "HEALTHY"

            return {
                "table_name": table_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "health_status": health,
                "issues": issues,
                "schema_drift": schema,
                "volume_anomaly": volume,
                "distribution_drift": distribution,
            }
        except Exception as e:
            print(f"[DQ ENGINE] Observability report failed: {e}")
            return {
                "table_name": table_name,
                "health_status": "ERROR",
                "issues": [str(e)],
                "schema_drift": {}, "volume_anomaly": {},
                "distribution_drift": {},
            }
