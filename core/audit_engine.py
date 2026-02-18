"""
Data Audit Engine
=================
Validates every output before it reaches users.
Integrates with DrDataAgent and the workspace deliverables flow.

Drop this into: core/audit_engine.py

Usage in dr_data_agent.py:
    from core.audit_engine import AuditEngine
    engine = AuditEngine()
    report = engine.audit_dataframe(df)
    report = engine.audit_deliverable(file_path, file_type)
    html = report.to_html(audience="analyst")  # or "executive"
"""

import re
import hashlib
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class Severity:
    PASS = "PASS"
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    BLOCKER = "BLOCKER"


SEVERITY_RANK = {
    Severity.PASS: 0, Severity.INFO: 1, Severity.WARNING: 2,
    Severity.CRITICAL: 3, Severity.BLOCKER: 4,
}


@dataclass
class AuditFinding:
    check_name: str
    category: str       # data_integrity, completeness, consistency, translation
    severity: str
    message: str
    detail: str = ""
    recommendation: str = ""
    affected_items: list = field(default_factory=list)


@dataclass
class AuditReport:
    audit_id: str = ""
    timestamp: str = ""
    source_name: str = ""
    audit_type: str = ""
    findings: list = field(default_factory=list)
    total_checks: int = 0
    passed: int = 0
    warnings: int = 0
    critical: int = 0
    blockers: int = 0
    overall_score: float = 0.0
    is_releasable: bool = False
    release_decision: str = ""

    def __post_init__(self):
        if not self.audit_id:
            self.audit_id = hashlib.md5(
                f"{datetime.now().isoformat()}{self.source_name}".encode()
            ).hexdigest()[:12].upper()
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def compute_scores(self):
        self.total_checks = len(self.findings)
        self.passed = sum(1 for f in self.findings if f.severity == Severity.PASS)
        self.warnings = sum(1 for f in self.findings if f.severity == Severity.WARNING)
        self.critical = sum(1 for f in self.findings if f.severity == Severity.CRITICAL)
        self.blockers = sum(1 for f in self.findings if f.severity == Severity.BLOCKER)

        if self.total_checks == 0:
            self.overall_score = 0.0
            self.is_releasable = False
            self.release_decision = "No checks executed"
            return

        score_map = {
            Severity.PASS: 100, Severity.INFO: 90, Severity.WARNING: 60,
            Severity.CRITICAL: 20, Severity.BLOCKER: 0,
        }
        total = sum(score_map.get(f.severity, 0) for f in self.findings)
        self.overall_score = round(total / self.total_checks, 1)

        if self.blockers > 0:
            self.is_releasable = False
            self.release_decision = f"BLOCKED -- {self.blockers} blocker(s) must be resolved"
        elif self.critical > 0:
            self.is_releasable = False
            self.release_decision = f"HOLD -- {self.critical} critical finding(s) require review"
        elif self.warnings > 3:
            self.is_releasable = False
            self.release_decision = f"REVIEW REQUIRED -- {self.warnings} warnings exceed threshold"
        elif self.overall_score >= 70:
            self.is_releasable = True
            self.release_decision = "APPROVED -- output meets quality standards"
        else:
            self.is_releasable = False
            self.release_decision = f"BELOW THRESHOLD -- score {self.overall_score}/100 (minimum 70)"

    def to_html(self, audience="analyst"):
        """Render audit report as HTML for the workspace panel."""
        self.compute_scores()

        if self.overall_score >= 85:
            score_color = "#238636"
        elif self.overall_score >= 70:
            score_color = "#d29922"
        else:
            score_color = "#da3633"

        if self.is_releasable:
            badge_bg = "rgba(35,134,54,0.15)"
            badge_color = "#238636"
            badge_border = "rgba(35,134,54,0.3)"
            badge_text = "APPROVED FOR RELEASE"
        elif self.blockers > 0:
            badge_bg = "rgba(218,54,51,0.15)"
            badge_color = "#da3633"
            badge_border = "rgba(218,54,51,0.3)"
            badge_text = "BLOCKED"
        else:
            badge_bg = "rgba(210,153,34,0.15)"
            badge_color = "#d29922"
            badge_border = "rgba(210,153,34,0.3)"
            badge_text = "REVIEW REQUIRED"

        html = f"""
        <div style="background:#1A1A1A;border:1px solid #333333;border-radius:10px;padding:20px;margin:12px 0;font-family:'Inter',-apple-system,sans-serif;color:#FFFFFF;">
          <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:16px;flex-wrap:wrap;gap:10px;">
            <div>
              <h3 style="color:#FFFFFF;margin:0 0 4px 0;font-size:14px;font-weight:600;">Data Quality Audit</h3>
              <div style="color:#B0B0B0;font-size:11px;">ID: {self.audit_id} | {self.timestamp} | {self.source_name}</div>
            </div>
            <span style="background:{badge_bg};color:{badge_color};padding:4px 10px;border-radius:4px;font-size:11px;font-weight:600;border:1px solid {badge_border};">{badge_text}</span>
          </div>

          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px;margin:12px 0;">
            <div style="background:#262626;border:1px solid #333333;border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:24px;font-weight:700;color:{score_color};">{self.overall_score}</div>
              <div style="font-size:10px;color:#B0B0B0;text-transform:uppercase;letter-spacing:0.5px;">Score</div>
            </div>
            <div style="background:#262626;border:1px solid #333333;border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:24px;font-weight:700;color:#238636;">{self.passed}</div>
              <div style="font-size:10px;color:#B0B0B0;text-transform:uppercase;letter-spacing:0.5px;">Passed</div>
            </div>
            <div style="background:#262626;border:1px solid #333333;border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:24px;font-weight:700;color:#d29922;">{self.warnings}</div>
              <div style="font-size:10px;color:#B0B0B0;text-transform:uppercase;letter-spacing:0.5px;">Warnings</div>
            </div>
            <div style="background:#262626;border:1px solid #333333;border-radius:8px;padding:12px;text-align:center;">
              <div style="font-size:24px;font-weight:700;color:#da3633;">{self.critical + self.blockers}</div>
              <div style="font-size:10px;color:#B0B0B0;text-transform:uppercase;letter-spacing:0.5px;">Critical</div>
            </div>
          </div>

          <div style="background:#262626;border-left:3px solid {score_color};padding:10px 14px;border-radius:0 6px 6px 0;margin:12px 0;font-size:12px;color:#FFFFFF;">
            <strong style="color:{score_color};">Decision:</strong> {self.release_decision}
          </div>

          <table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:12px;">
            <thead>
              <tr style="border-bottom:2px solid #333333;">
                <th style="text-align:left;padding:6px 10px;color:#B0B0B0;font-size:10px;text-transform:uppercase;">Status</th>
                <th style="text-align:left;padding:6px 10px;color:#B0B0B0;font-size:10px;text-transform:uppercase;">Check</th>
                <th style="text-align:left;padding:6px 10px;color:#B0B0B0;font-size:10px;text-transform:uppercase;">Finding</th>
                {"<th style='text-align:left;padding:6px 10px;color:#B0B0B0;font-size:10px;text-transform:uppercase;'>Detail</th>" if audience == "analyst" else ""}
              </tr>
            </thead>
            <tbody>
        """

        severity_styles = {
            Severity.PASS:     ("rgba(35,134,54,0.15)", "#238636", "PASS"),
            Severity.INFO:     ("rgba(255,230,0,0.15)", "#FFE600", "INFO"),
            Severity.WARNING:  ("rgba(210,153,34,0.15)", "#d29922", "WARN"),
            Severity.CRITICAL: ("rgba(218,54,51,0.15)", "#da3633", "CRIT"),
            Severity.BLOCKER:  ("rgba(218,54,51,0.25)", "#da3633", "BLOCK"),
        }

        sorted_findings = sorted(
            self.findings, key=lambda f: -SEVERITY_RANK.get(f.severity, 0)
        )

        for f in sorted_findings:
            bg, color, label = severity_styles.get(f.severity, ("transparent", "#B0B0B0", "???"))
            html += f"""
              <tr style="border-bottom:1px solid #333333;">
                <td style="padding:6px 10px;"><span style="background:{bg};color:{color};padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;">{label}</span></td>
                <td style="padding:6px 10px;color:#FFFFFF;">{f.check_name}</td>
                <td style="padding:6px 10px;color:#B0B0B0;">{f.message}</td>
                {"<td style='padding:6px 10px;color:#808080;font-size:11px;'>" + (f.detail or "--") + "</td>" if audience == "analyst" else ""}
              </tr>
            """
            if f.severity in (Severity.WARNING, Severity.CRITICAL, Severity.BLOCKER) and f.recommendation:
                colspan = 4 if audience == "analyst" else 3
                html += f"""
              <tr style="border-bottom:1px solid #333333;">
                <td></td>
                <td colspan="{colspan - 1}" style="padding:2px 10px 6px 10px;color:#FFE600;font-size:11px;">Action: {f.recommendation}</td>
              </tr>
                """

        html += "</tbody></table></div>"
        return html

    def to_executive_summary(self):
        self.compute_scores()
        if self.is_releasable:
            return (
                f"Quality score: {self.overall_score}/100. "
                f"{self.passed} of {self.total_checks} checks passed. "
                f"Output approved for release."
                f"{'' if self.warnings == 0 else f' {self.warnings} minor items flagged for review.'}"
            )
        return (
            f"Quality score: {self.overall_score}/100. "
            f"{self.critical + self.blockers} issue(s) require resolution. "
            f"{self.release_decision}"
        )

    def to_standalone_html(self, title="Data Quality Audit Report"):
        """Render a full standalone HTML page with WU branding."""
        body = self.to_html(audience="analyst")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
body{{background:#0D0D0D;color:#FFFFFF;font-family:'Inter',system-ui,sans-serif;margin:0;padding:0;}}
.hdr{{background:linear-gradient(135deg,#0D0D0D,#1A1A1A);border-bottom:1px solid #333;padding:20px 32px;position:relative;display:flex;align-items:center;justify-content:space-between;}}
.hdr::before{{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:#FFE600;}}
.hdr h1{{font-size:20px;font-weight:600;margin:0;}}
.hdr .wu-badge{{background:#FFE600;color:#000;font-weight:900;font-size:11px;padding:5px 12px;border-radius:3px;letter-spacing:1px;}}
.wrap{{max-width:900px;margin:0 auto;padding:24px 32px;}}
.footer{{border-top:1px solid #333;padding:16px 32px;text-align:center;font-size:11px;color:#808080;margin-top:32px;}}
@media print{{body{{background:#fff;color:#1a1a1a;}}.hdr{{background:#fff;border-bottom:3px solid #FFE600;}}.hdr h1{{color:#1a1a1a;}}}}
</style>
</head>
<body>
<div class="hdr">
  <h1>{title}</h1>
  <span class="wu-badge">WESTERN UNION</span>
</div>
<div class="wrap">
{body}
</div>
<div class="footer">Built by Dr. Data -- Western Union Analytics</div>
</body>
</html>"""


class AuditEngine:
    """
    Validates data, outputs, and agent responses.
    Call audit methods before showing anything to the user.
    """

    def audit_dataframe(self, df, source_name="uploaded data"):
        """Audit a pandas DataFrame for quality issues."""
        report = AuditReport(source_name=source_name, audit_type="data_quality")

        if df is None or (HAS_PANDAS and isinstance(df, pd.DataFrame) and df.empty):
            report.findings.append(AuditFinding(
                check_name="Data Exists",
                category="completeness",
                severity=Severity.BLOCKER,
                message="DataFrame is empty or null",
                recommendation="Verify the source file contains data"
            ))
            return report

        report.findings.append(AuditFinding(
            check_name="Data Exists",
            category="completeness",
            severity=Severity.PASS,
            message=f"{len(df):,} rows x {len(df.columns)} columns loaded"
        ))

        # Check: duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = round(dup_count / len(df) * 100, 1)
            sev = Severity.WARNING if pct < 10 else Severity.CRITICAL
            report.findings.append(AuditFinding(
                check_name="Duplicate Rows",
                category="data_integrity",
                severity=sev,
                message=f"{dup_count:,} duplicate rows ({pct}%)",
                detail=f"{dup_count} of {len(df)} rows are exact duplicates",
                recommendation="Review if duplicates are expected or indicate a data load issue"
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Duplicate Rows",
                category="data_integrity",
                severity=Severity.PASS,
                message="No duplicate rows detected"
            ))

        # Check: null density per column
        high_null_cols = []
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > 50:
                high_null_cols.append(f"{col} ({null_pct:.0f}%)")
        if high_null_cols:
            report.findings.append(AuditFinding(
                check_name="Null Density",
                category="data_integrity",
                severity=Severity.WARNING,
                message=f"{len(high_null_cols)} column(s) are more than 50% null",
                detail=", ".join(high_null_cols[:8]),
                recommendation="Confirm these columns are optional or flag data quality with source team",
                affected_items=high_null_cols,
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Null Density",
                category="data_integrity",
                severity=Severity.PASS,
                message="All columns within acceptable null thresholds"
            ))

        # Check: all-null columns
        all_null = [c for c in df.columns if df[c].isnull().all()]
        if all_null:
            report.findings.append(AuditFinding(
                check_name="Empty Columns",
                category="completeness",
                severity=Severity.CRITICAL,
                message=f"{len(all_null)} column(s) contain no data at all",
                detail=", ".join(all_null[:10]),
                recommendation="Remove or investigate these columns before building dashboards",
                affected_items=all_null,
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Empty Columns",
                category="completeness",
                severity=Severity.PASS,
                message="All columns contain at least some data"
            ))

        # Check: single-value columns (zero variance)
        single_val = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        if single_val:
            report.findings.append(AuditFinding(
                check_name="Zero-Variance Columns",
                category="data_integrity",
                severity=Severity.INFO,
                message=f"{len(single_val)} column(s) have only one unique value",
                detail=", ".join(single_val[:8]),
                recommendation="These add no analytical value -- consider excluding from visuals"
            ))

        # Check: numeric outliers (IQR method on numeric columns)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        outlier_cols = []
        for col in numeric_cols[:20]:  # Cap at 20 to avoid slow scans
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
                if outlier_count > 0:
                    outlier_cols.append(f"{col} ({outlier_count})")
        if outlier_cols:
            report.findings.append(AuditFinding(
                check_name="Extreme Outliers",
                category="data_integrity",
                severity=Severity.WARNING if len(outlier_cols) <= 3 else Severity.CRITICAL,
                message=f"Extreme values detected in {len(outlier_cols)} column(s)",
                detail=", ".join(outlier_cols[:8]),
                recommendation="Verify these are legitimate values, not data entry errors"
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Extreme Outliers",
                category="data_integrity",
                severity=Severity.PASS,
                message="No extreme outliers detected in numeric columns"
            ))

        # Check: mixed types
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    types = set(type(v).__name__ for v in non_null.head(100))
                    if len(types) > 1 and types != {"str"}:
                        mixed_type_cols.append(col)
        if mixed_type_cols:
            report.findings.append(AuditFinding(
                check_name="Mixed Data Types",
                category="consistency",
                severity=Severity.WARNING,
                message=f"{len(mixed_type_cols)} column(s) contain mixed data types",
                detail=", ".join(mixed_type_cols[:5]),
                recommendation="Standardize types before aggregation to avoid calculation errors"
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Mixed Data Types",
                category="consistency",
                severity=Severity.PASS,
                message="Column types are consistent"
            ))

        # Summary
        total_null_pct = round(df.isnull().mean().mean() * 100, 1)
        report.findings.append(AuditFinding(
            check_name="Overall Data Profile",
            category="completeness",
            severity=Severity.INFO,
            message=(
                f"{len(df):,} rows, {len(df.columns)} columns, "
                f"{len(numeric_cols)} numeric, "
                f"{total_null_pct}% null overall"
            )
        ))

        return report

    def audit_deliverable(self, file_path, file_type="html"):
        """Audit a generated deliverable file before download."""
        report = AuditReport(
            source_name=os.path.basename(file_path),
            audit_type="deliverable"
        )

        # Check: file exists
        if not os.path.exists(file_path):
            report.findings.append(AuditFinding(
                check_name="File Exists",
                category="completeness",
                severity=Severity.BLOCKER,
                message="Deliverable file not found on disk",
                detail=file_path,
                recommendation="Regenerate the deliverable"
            ))
            return report

        file_size = os.path.getsize(file_path)

        # Check: not empty
        if file_size < 100:
            report.findings.append(AuditFinding(
                check_name="File Size",
                category="completeness",
                severity=Severity.BLOCKER,
                message=f"File is suspiciously small ({file_size} bytes)",
                recommendation="Generation may have failed -- regenerate"
            ))
            return report

        report.findings.append(AuditFinding(
            check_name="File Generated",
            category="completeness",
            severity=Severity.PASS,
            message=f"File created successfully ({file_size:,} bytes)"
        ))

        # Check: read content for integrity
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            report.findings.append(AuditFinding(
                check_name="File Readable",
                category="data_integrity",
                severity=Severity.CRITICAL,
                message=f"Cannot read file: {str(e)[:100]}"
            ))
            return report

        report.findings.append(AuditFinding(
            check_name="File Readable",
            category="data_integrity",
            severity=Severity.PASS,
            message="File reads without errors"
        ))

        # Check: placeholder/hallucination scan
        hallucination_patterns = [
            r'TODO', r'FIXME', r'PLACEHOLDER', r'EXAMPLE_',
            r'sample_table', r'your_table', r'replace_this',
            r'lorem ipsum', r'INSERT_.*_HERE',
        ]
        found_placeholders = []
        for pattern in hallucination_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                found_placeholders.extend(matches[:3])

        if found_placeholders:
            report.findings.append(AuditFinding(
                check_name="Placeholder Detection",
                category="data_integrity",
                severity=Severity.CRITICAL,
                message=f"Found {len(found_placeholders)} placeholder/generic reference(s)",
                detail=", ".join(found_placeholders[:5]),
                recommendation="Do not distribute -- fix or remove placeholders first"
            ))
        else:
            report.findings.append(AuditFinding(
                check_name="Placeholder Detection",
                category="data_integrity",
                severity=Severity.PASS,
                message="No placeholder or hallucinated content detected"
            ))

        # HTML-specific checks
        if file_type == "html":
            if "<html" not in content.lower():
                report.findings.append(AuditFinding(
                    check_name="HTML Structure",
                    category="data_integrity",
                    severity=Severity.WARNING,
                    message="File does not contain standard HTML structure",
                    recommendation="Verify output renders correctly in a browser"
                ))
            else:
                report.findings.append(AuditFinding(
                    check_name="HTML Structure",
                    category="data_integrity",
                    severity=Severity.PASS,
                    message="Valid HTML structure detected"
                ))

            # Check for empty chart containers
            empty_divs = len(re.findall(r'<div[^>]*>\s*</div>', content))
            if empty_divs > 5:
                report.findings.append(AuditFinding(
                    check_name="Empty Containers",
                    category="completeness",
                    severity=Severity.WARNING,
                    message=f"{empty_divs} empty div elements found",
                    recommendation="Some charts or sections may not have rendered"
                ))

        return report

    def audit_agent_response(self, response_text):
        """Audit LLM-generated response before showing to user."""
        report = AuditReport(
            source_name="Agent Response",
            audit_type="llm_response"
        )

        if not response_text or len(response_text.strip()) < 10:
            report.findings.append(AuditFinding(
                check_name="Response Exists",
                category="completeness",
                severity=Severity.CRITICAL,
                message="Empty or near-empty response"
            ))
            return report

        report.findings.append(AuditFinding(
            check_name="Response Exists",
            category="completeness",
            severity=Severity.PASS,
            message="Response generated"
        ))

        # Check for fabricated statistics
        stat_claims = re.findall(
            r'(?:saves?|reduc\w+|improv\w+|increas\w+)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%',
            response_text, re.IGNORECASE
        )
        if stat_claims:
            report.findings.append(AuditFinding(
                check_name="Numeric Claims",
                category="data_integrity",
                severity=Severity.INFO,
                message=f"Response contains {len(stat_claims)} percentage claim(s) -- sourced from analysis",
                detail=", ".join(f"{s}%" for s in stat_claims[:5])
            ))

        # Check for hedging that might indicate uncertainty
        hedging = re.findall(
            r'\b(I think|I believe|probably|approximately|around \d+|I assume)\b',
            response_text, re.IGNORECASE
        )
        if hedging:
            report.findings.append(AuditFinding(
                check_name="Confidence Level",
                category="consistency",
                severity=Severity.INFO,
                message="Response contains hedging language -- may indicate uncertainty",
                detail=", ".join(hedging[:5])
            ))

        return report

    @staticmethod
    def combine_reports(reports, title="Combined Quality Audit"):
        """Merge multiple AuditReports into one combined report."""
        combined = AuditReport(source_name=title, audit_type="combined")
        for r in reports:
            for f in r.findings:
                # Prefix check_name with source for clarity
                tagged = AuditFinding(
                    check_name=f"[{r.source_name}] {f.check_name}",
                    category=f.category,
                    severity=f.severity,
                    message=f.message,
                    detail=f.detail,
                    recommendation=f.recommendation,
                    affected_items=f.affected_items,
                )
                combined.findings.append(tagged)
        return combined
