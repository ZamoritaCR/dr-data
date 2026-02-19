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
                        parsed = pd.to_datetime(df[col], errors="coerce")
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

    def generate_html_scorecard(self, catalog=None, rules_engine=None,
                                history=None, trust_scorer=None,
                                copdq_result=None, compliance=None,
                                stewardship=None, incidents=None):
        """Generate a comprehensive self-contained HTML report. WU dark theme."""
        if not self.scan_results:
            return ""
        try:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            sc = self.generate_scorecard_data()
            overall = sc.get("overall_score", 0)
            total_rows = sum(r.get("row_count", 0) for r in self.scan_results.values())
            n_tables = len(self.scan_results)

            # ── Helpers ──
            def _c(v, g=90, y=70):
                if v >= g: return "#27AE60"
                return "#FFE600" if v >= y else "#E74C3C"

            def _gauge(label, score, sz=90):
                s = score if score is not None else 0
                r, circ = 36, 2 * 3.14159 * 36
                fill, c = circ * s / 100, _c(s, 95, 80)
                h = sz // 2
                return (f'<div style="text-align:center;margin:8px">'
                    f'<svg width="{sz}" height="{sz}" viewBox="0 0 {sz} {sz}">'
                    f'<circle cx="{h}" cy="{h}" r="{r}" fill="none" stroke="#333" stroke-width="7"/>'
                    f'<circle cx="{h}" cy="{h}" r="{r}" fill="none" stroke="{c}" stroke-width="7" '
                    f'stroke-dasharray="{fill:.1f} {circ:.1f}" stroke-linecap="round" '
                    f'transform="rotate(-90 {h} {h})"/>'
                    f'<text x="{h}" y="{h+2}" text-anchor="middle" font-size="16" '
                    f'font-weight="bold" fill="{c}">{s:.0f}</text></svg>'
                    f'<div style="font-size:11px;color:#aaa;margin-top:2px">{label}</div></div>')

            def _bar(v, w=120):
                v = v if v is not None else 0
                c = _c(v, 95, 80)
                return (f'<div style="background:#333;border-radius:4px;width:{w}px;height:14px;'
                    f'display:inline-block;vertical-align:middle">'
                    f'<div style="background:{c};width:{max(0,min(100,v))}%;height:100%;'
                    f'border-radius:4px"></div></div>'
                    f' <span style="font-size:12px;color:#ccc">{v:.0f}%</span>')

            def _card(label, value, color="#FFE600"):
                return (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                    f'padding:14px;text-align:center;flex:1;min-width:130px">'
                    f'<div style="font-size:24px;font-weight:700;color:{color}">{value}</div>'
                    f'<div style="font-size:11px;color:#888;margin-top:4px">{label}</div></div>')

            _sec = lambda t: f'<div class="section"><div class="section-title">{t}</div>'
            _pri_c = {"CRITICAL": "#E74C3C", "HIGH": "#F39C12", "MEDIUM": "#FFE600", "LOW": "#27AE60"}

            # ── Dimension averages ──
            dims6 = ["completeness", "accuracy", "consistency", "timeliness", "uniqueness", "validity"]
            avg_d = {}
            for dm in dims6:
                ss = [d.get("score") for r in self.scan_results.values()
                      for d in [r.get("dimensions", {}).get(dm, {})] if isinstance(d, dict) and d.get("score") is not None]
                avg_d[dm] = round(sum(ss) / len(ss), 1) if ss else 0

            # ── Executive summary metrics ──
            crit_count = sum(1 for r in self.scan_results.values()
                             for rc in r.get("recommendations", []) if rc.get("priority") == "CRITICAL")
            comp_score = ""
            if compliance:
                cs = compliance.get_compliance_summary()
                comp_score = f"{cs.get('overall_score', 0):.0f}%"
            inc_open = ""
            if incidents:
                ist = incidents.get_dashboard_stats()
                inc_open = str(ist.get("open", 0))
            copdq_str = ""
            if copdq_result:
                ac = copdq_result.get("total_annual_cost", 0)
                copdq_str = f"${ac/1_000_000:.1f}M" if ac >= 1_000_000 else f"${ac/1_000:.0f}K" if ac >= 1_000 else f"${ac:.0f}"

            exec_cards = _card("Tables Scanned", n_tables)
            exec_cards += _card("Total Rows", f"{total_rows:,}")
            exec_cards += _card("Critical Issues", crit_count, "#E74C3C" if crit_count else "#27AE60")
            if copdq_str:
                exec_cards += _card("COPDQ Annual", copdq_str, "#E74C3C")
            if comp_score:
                exec_cards += _card("Compliance", comp_score)
            if inc_open:
                exec_cards += _card("Open Incidents", inc_open, "#F39C12" if int(inc_open) else "#27AE60")

            # ── Gauges ──
            gauges = "".join(_gauge(d.title(), avg_d[d]) for d in dims6)

            # ── Table details with heatmap ──
            table_html = ""
            for tname, result in self.scan_results.items():
                ts_s = result.get("overall_score", 0)
                tc = _c(ts_s)
                rw, cl = result.get("row_count", "?"), result.get("column_count", "?")
                dims = result.get("dimensions", {})
                db = ""
                for d in dims6:
                    dd = dims.get(d, {})
                    ds = dd.get("score", 0) if isinstance(dd, dict) else 0
                    db += (f'<div style="display:flex;align-items:center;margin:3px 0">'
                        f'<span style="width:100px;font-size:12px;color:#aaa">{d.title()}</span>{_bar(ds)}</div>')
                # Column heatmap
                comp_cols = dims.get("completeness", {}).get("columns", {})
                valid_cols = dims.get("validity", {}).get("column_validity", {})
                hm = ""
                if comp_cols:
                    hm_rows = ""
                    for cn, cd in list(comp_cols.items())[:30]:
                        cs = cd.get("score", 100)
                        cc = _c(cs, 95, 80)
                        vd = valid_cols.get(cn, {})
                        vs = vd.get("valid_pct", 100) if isinstance(vd, dict) else 100
                        vc = _c(vs, 95, 80)
                        hm_rows += (f'<tr><td style="padding:3px 8px;color:#ccc;font-size:11px;'
                            f'border-bottom:1px solid #333">{cn}</td>'
                            f'<td style="padding:3px 8px;text-align:center;background:{cc}22;'
                            f'color:{cc};font-size:11px;border-bottom:1px solid #333">{cs:.0f}%</td>'
                            f'<td style="padding:3px 8px;text-align:center;background:{vc}22;'
                            f'color:{vc};font-size:11px;border-bottom:1px solid #333">{vs:.0f}%</td></tr>')
                    hm = (f'<div style="margin-top:10px"><table style="width:100%;border-collapse:collapse">'
                        f'<tr style="background:#262626"><th style="padding:4px 8px;text-align:left;'
                        f'color:#888;font-size:10px">Column</th><th style="padding:4px 8px;text-align:center;'
                        f'color:#888;font-size:10px">Complete</th><th style="padding:4px 8px;text-align:center;'
                        f'color:#888;font-size:10px">Valid</th></tr>{hm_rows}</table></div>')
                table_html += (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                    f'padding:16px;margin:10px 0"><div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:10px"><span style="font-size:16px;font-weight:600;'
                    f'color:#fff">{tname}</span><span style="font-size:24px;font-weight:bold;color:{tc}">'
                    f'{ts_s:.1f}</span></div><div style="font-size:12px;color:#888;margin-bottom:8px">'
                    f'{rw} rows | {cl} columns</div>{db}{hm}</div>')

            # ── Trust scores ──
            trust_html = ""
            if trust_scorer:
                try:
                    all_ts = trust_scorer.score_all_tables(self.scan_results, catalog=catalog)
                    for tn, td in all_ts.items():
                        tscore = td.get("trust_score", 0)
                        cert = td.get("recommended_certification", "?")
                        cert_c = {"Certified": "#27AE60", "Warning": "#FFE600"}.get(cert, "#E74C3C")
                        trust_html += (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                            f'padding:12px;margin:6px 0;display:flex;justify-content:space-between;align-items:center">'
                            f'<span style="color:#fff;font-size:14px">{tn}</span>'
                            f'<span style="font-size:18px;font-weight:700;color:{_c(tscore)}">{tscore:.0f}</span>'
                            f'<span style="background:{cert_c};color:#000;padding:2px 8px;border-radius:4px;'
                            f'font-size:11px;font-weight:600">{cert}</span></div>')
                except Exception:
                    pass

            # ── COPDQ breakdown ──
            copdq_html = ""
            if copdq_result and copdq_result.get("cost_breakdown"):
                for tn, tc in copdq_result["cost_breakdown"].items():
                    total = tc.get("total", 0)
                    if total <= 0:
                        continue
                    costs = tc.get("costs", {})
                    bars = ""
                    for dim_name, dim_cost in costs.items():
                        ac = dim_cost.get("annual_cost", 0)
                        pct = (ac / total * 100) if total else 0
                        bars += (f'<div style="display:flex;align-items:center;margin:3px 0">'
                            f'<span style="width:100px;font-size:11px;color:#aaa">{dim_name.title()}</span>'
                            f'<div style="background:#333;border-radius:4px;width:200px;height:12px;margin-right:8px">'
                            f'<div style="background:#E74C3C;width:{min(pct,100):.0f}%;height:100%;border-radius:4px"></div></div>'
                            f'<span style="font-size:11px;color:#ccc">${ac:,.0f}</span></div>')
                    copdq_html += (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                        f'padding:12px;margin:6px 0"><div style="display:flex;justify-content:space-between;'
                        f'margin-bottom:8px"><span style="color:#fff;font-size:14px">{tn}</span>'
                        f'<span style="color:#E74C3C;font-weight:700">${total:,.0f}/yr</span></div>{bars}</div>')

            # ── Compliance ──
            comp_html = ""
            if compliance:
                try:
                    cs = compliance.get_compliance_summary()
                    for fw, fd in cs.get("by_framework", {}).items():
                        fs = fd.get("score", 0)
                        fc = _c(fs, 80, 60)
                        comp_html += (f'<div style="display:flex;align-items:center;margin:6px 0">'
                            f'<span style="width:100px;font-size:12px;color:#aaa">{fw}</span>'
                            f'{_bar(fs, 200)}'
                            f'<span style="font-size:11px;color:#888;margin-left:8px">'
                            f'{fd.get("compliant",0)} pass / {fd.get("non_compliant",0)} fail</span></div>')
                    gaps = cs.get("critical_gaps", [])
                    for g in gaps[:5]:
                        comp_html += (f'<div style="border-left:3px solid #E74C3C;padding:4px 10px;margin:4px 0;'
                            f'background:#1A1A1A;font-size:12px;color:#ccc">[{g.get("severity","HIGH")}] '
                            f'{g.get("framework","")}/{g.get("requirement_id","")}: '
                            f'{g.get("title","")} ({g.get("score",0):.0f}%)</div>')
                except Exception:
                    pass

            # ── Business Rules ──
            rules_html = ""
            if rules_engine:
                try:
                    for tn, rr in getattr(rules_engine, "evaluation_results", {}).items():
                        pr = rr.get("overall_pass_rate", 0)
                        failed = [r for r in rr.get("rule_results", []) if not r.get("passed")]
                        rules_html += (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                            f'padding:12px;margin:6px 0"><span style="color:#fff;font-size:14px">{tn}</span>'
                            f' <span style="color:{_c(pr, 90, 70)};font-weight:600">{pr:.0f}% pass</span>')
                        for fr in failed[:5]:
                            sev = fr.get("severity", "MEDIUM")
                            rules_html += (f'<div style="font-size:11px;color:#ccc;margin:3px 0 3px 12px">'
                                f'<span style="color:{_pri_c.get(sev,"#888")}">[{sev}]</span> '
                                f'{fr.get("rule_name","")}: {fr.get("message","")}</div>')
                        rules_html += '</div>'
                except Exception:
                    pass

            # ── Trending sparklines ──
            trend_html = ""
            if history:
                try:
                    for tn in list(self.scan_results.keys())[:5]:
                        ht = history.get_table_trend(tn)
                        entries = ht.get("entries", [])[-12:]
                        if len(entries) < 2:
                            continue
                        max_s = max(e.get("overall_score", 100) for e in entries) or 100
                        bars = ""
                        for e in entries:
                            s = e.get("overall_score", 0)
                            h = max(2, int(s / max_s * 40))
                            bars += (f'<div style="width:8px;height:{h}px;background:{_c(s)};'
                                f'border-radius:2px;margin:0 1px;display:inline-block;vertical-align:bottom"></div>')
                        trend_html += (f'<div style="background:#1A1A1A;border:1px solid #333;border-radius:8px;'
                            f'padding:10px;margin:4px 0;display:flex;align-items:end;gap:12px">'
                            f'<span style="color:#fff;font-size:13px;min-width:150px">{tn}</span>'
                            f'<div style="display:flex;align-items:end">{bars}</div></div>')
                except Exception:
                    pass

            # ── Stewardship ──
            stew_html = ""
            if stewardship:
                try:
                    sd = stewardship.get_dashboard_stats()
                    stew_html = (f'<div style="display:flex;gap:16px;flex-wrap:wrap">'
                        f'{_card("Open Issues", sd.get("total_open", 0), "#F39C12")}'
                        f'{_card("SLA Breaches", sd.get("sla_breaches", 0), "#E74C3C")}'
                        f'{_card("Unassigned", sd.get("unassigned", 0), "#FFE600")}'
                        f'{_card("Resolved", sd.get("total_resolved", 0), "#27AE60")}</div>')
                except Exception:
                    pass

            # ── Incidents ──
            inc_html = ""
            if incidents:
                try:
                    ist = incidents.get_dashboard_stats()
                    mttr = ist.get("mttr")
                    mttr_s = f"{mttr:.1f}h" if mttr else "N/A"
                    inc_html = (f'<div style="display:flex;gap:16px;flex-wrap:wrap">'
                        f'{_card("Open", ist.get("open", 0), "#F39C12")}'
                        f'{_card("Resolved", ist.get("resolved", 0), "#27AE60")}'
                        f'{_card("MTTR", mttr_s, "#FFE600")}'
                        f'{_card("Postmortem Pending", ist.get("postmortems_pending", 0), "#E74C3C")}</div>')
                except Exception:
                    pass

            # ── Top recommendations ──
            all_recs = []
            for tname, result in self.scan_results.items():
                for rec in result.get("recommendations", []):
                    r = dict(rec); r["table"] = tname; all_recs.append(r)
            all_recs.sort(key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(x.get("priority", "LOW"), 4))
            recs_html = ""
            for rec in all_recs[:20]:
                pri = rec.get("priority", "LOW")
                pc = _pri_c.get(pri, "#888")
                fix = (' <span style="background:#27AE60;color:#fff;padding:1px 6px;border-radius:3px;'
                    'font-size:10px">AUTO-FIX</span>' if rec.get("auto_fixable") else "")
                recs_html += (f'<div style="border-left:3px solid {pc};padding:6px 12px;margin:4px 0;'
                    f'background:#1A1A1A;border-radius:0 6px 6px 0">'
                    f'<span style="color:{pc};font-weight:bold;font-size:11px">[{pri}]</span> '
                    f'<span style="color:#ccc;font-size:11px">{rec.get("table","")}</span>{fix}'
                    f'<div style="color:#fff;font-size:12px;margin-top:2px">{rec.get("finding","")}</div>'
                    f'<div style="color:#888;font-size:11px">{rec.get("recommendation","")}</div></div>')

            # ── Cross-table ──
            cross_html = ""
            cross = getattr(self, "_last_cross_results", None)
            if cross:
                for rel in cross.get("detected_relationships", [])[:10]:
                    cross_html += (f'<div style="color:#ccc;font-size:12px;padding:2px 0">'
                        f'{rel["from_table"]}.{rel["from_col"]} &harr; '
                        f'{rel["to_table"]}.{rel["to_col"]} ({rel["confidence"]}%)</div>')
                for o in cross.get("orphan_fks", [])[:5]:
                    cross_html += f'<div style="color:#E74C3C;font-size:12px;padding:2px 0">Orphan FK: {o}</div>'

            # ── Build optional sections ──
            def _opt(title, content):
                return f'{_sec(title)}{content}</div>' if content else ""

            opt_sections = ""
            opt_sections += _opt("Trust Scores", trust_html)
            opt_sections += _opt("Cost of Poor Data Quality", copdq_html)
            opt_sections += _opt("Regulatory Compliance", comp_html)
            opt_sections += _opt("Business Rules", rules_html)
            opt_sections += _opt("Score Trending", trend_html)
            opt_sections += _opt("Stewardship", stew_html)
            opt_sections += _opt("Incidents", inc_html)
            opt_sections += _opt("Cross-Table Analysis", cross_html)

            oc = _c(overall)
            html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Enterprise Data Quality Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0D0D0D;color:#fff;font-family:Inter,system-ui,-apple-system,sans-serif;padding:24px;line-height:1.5}}
.container{{max-width:1100px;margin:0 auto}}
.header{{text-align:center;padding:30px 0;border-bottom:2px solid #FFE600}}
.header h1{{font-size:22px;color:#FFE600;font-weight:700;letter-spacing:0.5px}}
.badge{{background:#E74C3C;color:#fff;font-size:10px;padding:2px 8px;border-radius:3px;margin-left:8px;vertical-align:middle}}
.big-score{{font-size:72px;font-weight:800;margin:16px 0 4px}}
.section{{margin:28px 0}}
.section-title{{font-size:16px;font-weight:700;color:#FFE600;border-left:3px solid #FFE600;padding-left:10px;margin-bottom:14px}}
.gauges{{display:flex;justify-content:center;flex-wrap:wrap;gap:12px}}
.exec-cards{{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}}
.footer{{text-align:center;color:#555;font-size:11px;margin-top:40px;padding-top:16px;border-top:1px solid #333}}
@media print{{body{{background:#fff;color:#000}} .section-title{{color:#333;border-color:#333}} .footer{{color:#999}}}}
</style></head><body><div class="container">
<div class="header">
<h1>Enterprise Data Quality Report<span class="badge">CONFIDENTIAL</span></h1>
<div style="font-size:12px;color:#666;margin-top:4px">{ts}</div>
<div class="big-score" style="color:{oc}">{overall:.1f}</div>
<div style="color:#888;font-size:13px">Overall Quality Score / 100</div>
</div>
{_sec("Executive Summary")}<div class="exec-cards">{exec_cards}</div></div>
{_sec("DAMA Dimension Scores")}<div class="gauges">{gauges}</div></div>
{_sec("Table Details & Column Heatmap")}{table_html}</div>
{opt_sections}
{_sec("Top Recommendations")}{recs_html if recs_html else '<div style="color:#888">No issues detected.</div>'}</div>
<div class="footer">Generated by Dr. Data -- Enterprise Data Quality Assessment Engine | {ts} | CONFIDENTIAL</div>
</div></body></html>"""
            return html
        except Exception as e:
            print(f"[DQ_ENGINE] generate_html_scorecard failed: {e}")
            return ""

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
                            rng = np.random.default_rng(42)
                            synth = rng.normal(
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
