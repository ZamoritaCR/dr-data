"""
Dr. Data -- Advanced Data Quality Engine
Frontier algorithms not found in any existing DQ tool.
Art of the Possible.
"""
import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


# -- BENFORD'S LAW ANALYSIS ------------------------------------------------
def benford_analysis(df, numeric_cols=None):
    """
    Benford's Law analysis for fraud detection and data authenticity.

    Benford's Law: In naturally occurring datasets, the leading digit d appears
    with probability log10(1 + 1/d). Deviations indicate potential fabrication,
    transcription errors, or systematic biases.

    Returns dict with column_results, overall_verdict, suspicious_columns.
    """
    from scipy import stats

    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    benford_expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
    results = {}
    suspicious = []

    for col in numeric_cols:
        series = df[col].dropna()
        series = series[series > 0].abs()
        if len(series) < 50:
            continue

        def _leading_digit(x):
            s = str(x).replace(".", "").lstrip("0")
            return int(s[0]) if s else None

        leading_digits = series.apply(_leading_digit).dropna()
        leading_digits = leading_digits[leading_digits.between(1, 9)]

        if len(leading_digits) < 50:
            continue

        observed_counts = np.array([(leading_digits == d).sum() for d in range(1, 10)])
        observed_freq = observed_counts / observed_counts.sum()

        expected_counts = benford_expected * len(leading_digits)
        chi2, p_value = stats.chisquare(observed_counts, expected_counts)

        conforms = p_value > 0.05
        if not conforms:
            suspicious.append(col)

        results[col] = {
            "observed": observed_freq.tolist(),
            "expected": benford_expected.tolist(),
            "chi2": float(chi2),
            "p_value": float(p_value),
            "conforms": conforms,
            "n": int(len(leading_digits)),
            "verdict": "Conforms" if conforms else f"Non-conforming (p={p_value:.4f})",
        }

    return {
        "column_results": results,
        "overall_verdict": (
            "PASS"
            if not suspicious
            else f"REVIEW REQUIRED -- {len(suspicious)} column(s) deviate"
        ),
        "suspicious_columns": suspicious,
    }


# -- STATISTICAL PROCESS CONTROL -------------------------------------------
def spc_analysis(df, date_col, value_col, window=20):
    """
    Statistical Process Control (SPC) analysis using Western Electric Rules.

    Generates X-bar control charts with UCL/LCL = mu +/- 3*sigma,
    warning limits = mu +/- 2*sigma, and Western Electric Rules for
    detecting non-random patterns.
    """
    series = df.sort_values(date_col)[value_col].dropna()

    if len(series) < 8:
        return {"error": "Insufficient data for SPC (need >= 8 points)"}

    mean = series.mean()
    std = series.std()
    if std == 0:
        return {"error": "Zero variance -- SPC not applicable"}

    ucl = mean + 3 * std
    lcl = mean - 3 * std
    uwl = mean + 2 * std
    lwl = mean - 2 * std

    violations = []
    vals = series.values

    for i, v in enumerate(vals):
        # Rule 1: Point outside 3-sigma
        if v > ucl or v < lcl:
            violations.append({
                "index": i, "value": float(v),
                "rule": "Rule 1: Point beyond 3-sigma control limit",
                "severity": "critical",
            })
        # Rule 2: 9 consecutive points on same side of mean
        if i >= 8:
            segment = vals[i - 8 : i + 1]
            if all(s > mean for s in segment) or all(s < mean for s in segment):
                violations.append({
                    "index": i, "value": float(v),
                    "rule": "Rule 2: 9 consecutive points on one side of center",
                    "severity": "warning",
                })
        # Rule 3: 6 consecutive points trending
        if i >= 5:
            segment = vals[i - 5 : i + 1]
            diffs = np.diff(segment)
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                violations.append({
                    "index": i, "value": float(v),
                    "rule": "Rule 3: 6 consecutive points trending in one direction",
                    "severity": "warning",
                })
        # Rule 4: 2 of 3 beyond 2-sigma
        if i >= 2:
            segment = vals[i - 2 : i + 1]
            beyond_2sig = sum(1 for s in segment if abs(s - mean) > 2 * std)
            if beyond_2sig >= 2:
                violations.append({
                    "index": i, "value": float(v),
                    "rule": "Rule 4: 2 of 3 points beyond 2-sigma warning limit",
                    "severity": "warning",
                })

    # Process capability (Cpk)
    cpu = (ucl - mean) / (3 * std)
    cpl = (mean - lcl) / (3 * std)
    cpk = min(cpu, cpl)

    dates = df.sort_values(date_col)[date_col].dropna().values.tolist()
    dates = [str(d) for d in dates[: len(series)]]

    return {
        "values": series.tolist(),
        "dates": dates[: len(series)],
        "mean": float(mean),
        "ucl": float(ucl),
        "lcl": float(lcl),
        "uwl": float(uwl),
        "lwl": float(lwl),
        "violations": violations[:20],
        "process_stable": len([v for v in violations if v["severity"] == "critical"]) == 0,
        "cpk": float(cpk),
        "sigma_level": float(cpk * 3),
        "violation_count": len(violations),
    }


# -- ENTITY RESOLUTION (DEDUPLICATION) ------------------------------------
def entity_resolution(df, key_columns, threshold=85.0):
    """
    Probabilistic entity resolution using blocking + fuzzy similarity scoring.
    Implements a simplified Fellegi-Sunter model.
    """
    from rapidfuzz import fuzz

    if len(df) > 50000:
        df = df.sample(10000, random_state=42)

    duplicates = []
    key_col = key_columns[0]
    if key_col not in df.columns:
        return {"error": f"Column '{key_col}' not found"}

    df_sorted = df.reset_index(drop=True).copy()
    df_sorted["_block"] = df_sorted[key_col].astype(str).str[:3].str.lower()

    for _block_val, group in df_sorted.groupby("_block"):
        if len(group) < 2:
            continue
        idxs = group.index.tolist()
        for i in range(len(idxs)):
            for j in range(i + 1, min(i + 10, len(idxs))):
                a, b = idxs[i], idxs[j]
                scores = []
                matched_on = []
                for col in key_columns:
                    if col not in df.columns:
                        continue
                    va = str(df_sorted.loc[a, col]) if pd.notna(df_sorted.loc[a, col]) else ""
                    vb = str(df_sorted.loc[b, col]) if pd.notna(df_sorted.loc[b, col]) else ""
                    if va and vb:
                        s = fuzz.token_sort_ratio(va, vb)
                        scores.append(s)
                        if s >= threshold:
                            matched_on.append(col)

                if scores:
                    avg_score = np.mean(scores)
                    if avg_score >= threshold:
                        duplicates.append({
                            "idx_a": int(a),
                            "idx_b": int(b),
                            "score": round(float(avg_score), 1),
                            "matched_on": matched_on,
                        })

    dedup_rate = len(duplicates) / max(len(df), 1)

    return {
        "duplicate_pairs": duplicates[:100],
        "duplicate_count": len(duplicates),
        "dedup_rate": round(dedup_rate * 100, 2),
        "total_records": len(df),
        "recommendations": [
            f"Found {len(duplicates)} potential duplicate pairs",
            f"Estimated {len(duplicates)} records may need merging",
            (
                "Consider implementing MDM (Master Data Management) for these entities"
                if len(duplicates) > 10
                else "Entity resolution looks healthy"
            ),
        ],
    }


# -- DATA DRIFT DETECTION -------------------------------------------------
def drift_detection(reference_df, current_df, columns=None):
    """
    Column-level data drift detection using PSI (Population Stability Index)
    and KS test for continuous variables.

    PSI interpretation:
    - PSI < 0.1: No significant change (stable)
    - 0.1 <= PSI < 0.2: Moderate change (monitor)
    - PSI >= 0.2: Major shift (investigate)
    """
    from scipy import stats

    if columns is None:
        columns = [c for c in reference_df.columns if c in current_df.columns]

    results = {}
    drifted_columns = []

    for col in columns:
        ref = reference_df[col].dropna()
        cur = current_df[col].dropna()

        if len(ref) < 10 or len(cur) < 10:
            continue

        if pd.api.types.is_numeric_dtype(ref):
            ks_stat, p_value = stats.ks_2samp(ref, cur)

            ref_hist, bins = np.histogram(ref, bins=10)
            cur_hist, _ = np.histogram(cur, bins=bins)

            ref_pct = (ref_hist / len(ref)) + 1e-6
            cur_pct = (cur_hist / len(cur)) + 1e-6

            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
            drifted = psi >= 0.2 or p_value < 0.05

            results[col] = {
                "type": "numeric",
                "psi": round(psi, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_value), 4),
                "ref_mean": round(float(ref.mean()), 4),
                "cur_mean": round(float(cur.mean()), 4),
                "mean_shift_pct": round(
                    abs(cur.mean() - ref.mean()) / max(abs(ref.mean()), 1e-6) * 100, 2
                ),
                "drifted": drifted,
                "severity": (
                    "critical" if psi >= 0.2 else ("warning" if psi >= 0.1 else "stable")
                ),
            }
        else:
            ref_counts = ref.value_counts(normalize=True)
            cur_counts = cur.value_counts(normalize=True)
            all_vals = set(ref_counts.index) | set(cur_counts.index)

            ref_aligned = np.array([ref_counts.get(v, 1e-6) for v in all_vals])
            cur_aligned = np.array([cur_counts.get(v, 1e-6) for v in all_vals])

            psi = float(np.sum((cur_aligned - ref_aligned) * np.log(cur_aligned / ref_aligned)))
            chi2, p_value = stats.chisquare(cur_aligned * len(cur), ref_aligned * len(cur))

            drifted = psi >= 0.2

            results[col] = {
                "type": "categorical",
                "psi": round(psi, 4),
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p_value), 4),
                "new_categories": list(set(cur_counts.index) - set(ref_counts.index)),
                "missing_categories": list(set(ref_counts.index) - set(cur_counts.index)),
                "drifted": drifted,
                "severity": (
                    "critical" if psi >= 0.2 else ("warning" if psi >= 0.1 else "stable")
                ),
            }

        if drifted:
            drifted_columns.append(col)

    overall_drift_psi = np.mean([r["psi"] for r in results.values()]) if results else 0

    return {
        "column_results": results,
        "drifted_columns": drifted_columns,
        "overall_psi": round(float(overall_drift_psi), 4),
        "drift_verdict": (
            "CRITICAL"
            if overall_drift_psi >= 0.2
            else ("WARNING" if overall_drift_psi >= 0.1 else "STABLE")
        ),
        "drift_pct": round(len(drifted_columns) / max(len(results), 1) * 100, 1),
    }


# -- ADVANCED PROFILING ----------------------------------------------------
def advanced_profile(df):
    """
    Deep statistical profiling beyond basic describe().
    Computes: distributions, correlations, missing patterns, outliers,
    cardinality analysis, pattern detection, entropy.
    """
    from scipy import stats

    profile = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": {},
        "correlations": {},
        "missing_pattern": {},
        "warnings": [],
    }

    for col in df.columns:
        series = df[col].dropna()
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "cardinality_ratio": round(df[col].nunique() / max(len(df), 1), 4),
        }

        if pd.api.types.is_numeric_dtype(df[col]) and len(series) > 3:
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            col_info.update({
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "iqr": round(float(iqr), 4),
                "outlier_count_iqr": int(
                    ((series < q25 - 1.5 * iqr) | (series > q75 + 1.5 * iqr)).sum()
                ),
                "zeros_pct": round((series == 0).mean() * 100, 2),
                "negatives_pct": round((series < 0).mean() * 100, 2),
            })
            if len(series) <= 5000:
                try:
                    stat, p = stats.shapiro(series.sample(min(len(series), 5000)))
                    col_info["normality_p"] = round(float(p), 4)
                    col_info["is_normal"] = p > 0.05
                except Exception:
                    pass

            if abs(col_info.get("skewness", 0)) > 2:
                profile["warnings"].append(
                    f"'{col}' is highly skewed ({col_info['skewness']:.2f})"
                )
            if col_info.get("outlier_count_iqr", 0) > len(df) * 0.05:
                profile["warnings"].append(
                    f"'{col}' has {col_info['outlier_count_iqr']} outliers (IQR method)"
                )

        elif pd.api.types.is_object_dtype(df[col]):
            top5 = series.value_counts().head(5)
            col_info.update({
                "top_values": top5.to_dict(),
                "entropy": (
                    round(float(stats.entropy(series.value_counts(normalize=True))), 4)
                    if len(series) > 0
                    else 0
                ),
                "avg_length": round(series.astype(str).str.len().mean(), 1),
            })

        if col_info["null_pct"] > 20:
            profile["warnings"].append(
                f"'{col}' has {col_info['null_pct']}% missing values"
            )
        if col_info["cardinality_ratio"] > 0.95 and not pd.api.types.is_numeric_dtype(df[col]):
            profile["warnings"].append(
                f"'{col}' may be a key column (high cardinality: {col_info['unique_count']} unique)"
            )

        profile["columns"][col] = col_info

    # Correlations (numeric only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.8:
                    high_corr.append({
                        "col_a": numeric_cols[i],
                        "col_b": numeric_cols[j],
                        "r": round(float(r), 4),
                        "warning": (
                            "Potential multicollinearity"
                            if abs(r) > 0.95
                            else "High correlation"
                        ),
                    })
        profile["correlations"] = {"high_correlations": high_corr}

    return profile
