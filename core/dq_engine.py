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
