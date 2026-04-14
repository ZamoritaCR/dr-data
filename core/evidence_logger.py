"""Structured audit logging for Dr. Data V2 pipeline runs."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class EvidenceLogger:
    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.entries: List[Dict[str, Any]] = []

    def _summarize(self, value: Any) -> str:
        try:
            if value is None:
                text = ""
            elif isinstance(value, str):
                text = value
            else:
                text = json.dumps(value, indent=2, sort_keys=True, default=str)
        except Exception:
            text = repr(value)
        if len(text) > 500:
            return text[:500] + "... [truncated]"
        return text

    def log(
        self,
        phase: str,
        action: str,
        input_data=None,
        output_data=None,
        status: str = "ok",
        **kwargs,
    ):
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "action": action,
                "input_summary": self._summarize(input_data),
                "output_summary": self._summarize(output_data),
                "status": status,
                "metadata": kwargs or {},
            }
            self.entries.append(entry)
        except Exception:
            return

    def get_summary(self) -> dict:
        errors = sum(1 for e in self.entries if e.get("status") in ("error", "failed"))
        warnings = sum(1 for e in self.entries if e.get("status") == "warning")
        phases = {e.get("phase", "") for e in self.entries if e.get("phase")}
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "started_at": self.started_at,
            "phases_completed": len(phases),
            "actions_logged": len(self.entries),
            "errors": errors,
            "warnings": warnings,
        }

    def export_html(self, output_path: str) -> str:
        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            phase_sections = [
                ("parse", "Parse Evidence"),
                ("translate", "Translation Evidence"),
                ("build", "Build Evidence"),
                ("publish", "Publish Evidence"),
                ("qa", "QA Evidence"),
                ("validate", "Validation Evidence"),
                ("vfe", "VFE Evidence"),
                ("done", "Run Completion"),
                ("error", "Run Errors"),
            ]

            grouped = defaultdict(list)
            for entry in self.entries:
                grouped[entry.get("phase", "other")].append(entry)

            summary = self.get_summary()
            section_html = []
            for phase_key, title in phase_sections:
                entries = grouped.get(phase_key, [])
                if not entries:
                    continue
                cards = []
                for entry in entries:
                    metadata = self._summarize(entry.get("metadata", {}))
                    cards.append(
                        f"""
                        <div class="entry status-{html.escape(entry.get('status', 'ok'))}">
                          <div class="entry-head">
                            <span class="action">{html.escape(entry.get('action', ''))}</span>
                            <span class="timestamp">{html.escape(entry.get('timestamp', ''))}</span>
                          </div>
                          <div class="grid">
                            <div>
                              <div class="label">Input</div>
                              <pre>{html.escape(entry.get('input_summary', ''))}</pre>
                            </div>
                            <div>
                              <div class="label">Output</div>
                              <pre>{html.escape(entry.get('output_summary', ''))}</pre>
                            </div>
                          </div>
                          <div class="label">Metadata</div>
                          <pre>{html.escape(metadata)}</pre>
                        </div>
                        """
                    )
                section_html.append(
                    f"""
                    <details open>
                      <summary>{html.escape(title)} <span>{len(entries)} actions</span></summary>
                      <div class="section-body">
                        {''.join(cards)}
                      </div>
                    </details>
                    """
                )

            html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dr. Data V2 Evidence Report</title>
  <style>
    :root {{
      --bg: #0d0d22;
      --card: #171733;
      --border: rgba(255,255,255,0.1);
      --green: #00ff88;
      --gold: #ffd700;
      --red: #ff5876;
      --text: #f3f5ff;
      --muted: #98a0b3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Space Mono", ui-monospace, monospace;
      padding: 24px;
    }}
    h1, summary {{
      font-family: "Rajdhani", "Segoe UI", sans-serif;
      letter-spacing: 0.04em;
    }}
    h1 {{
      margin: 0 0 8px;
      color: var(--green);
      font-size: 34px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 20px;
      font-size: 13px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .summary-card, details {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .summary-card {{
      padding: 14px 16px;
    }}
    .summary-card .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }}
    .summary-card .value {{
      color: var(--gold);
      font-family: "JetBrains Mono", ui-monospace, monospace;
      font-size: 22px;
      margin-top: 8px;
    }}
    details {{
      margin-bottom: 14px;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-size: 18px;
      color: var(--gold);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    summary::-webkit-details-marker {{ display: none; }}
    .section-body {{
      padding: 0 16px 16px;
    }}
    .entry {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      margin-top: 12px;
      background: rgba(255,255,255,0.02);
    }}
    .status-error, .status-failed {{ border-color: rgba(255,88,118,0.4); }}
    .status-warning {{ border-color: rgba(255,215,0,0.35); }}
    .entry-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .action {{
      color: var(--green);
      font-weight: 700;
      font-family: "Rajdhani", sans-serif;
      text-transform: uppercase;
    }}
    .timestamp {{
      color: var(--muted);
      font-size: 11px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(0,0,0,0.25);
      border-radius: 8px;
      padding: 10px;
      border: 1px solid rgba(255,255,255,0.06);
      font-family: "JetBrains Mono", ui-monospace, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <h1>Dr. Data V2 Evidence Report</h1>
  <div class="sub">Job <strong>{html.escape(self.job_id)}</strong> · File <strong>{html.escape(self.filename)}</strong> · Started {html.escape(self.started_at)}</div>
  <div class="summary">
    <div class="summary-card"><div class="label">File Processed</div><div class="value">{html.escape(self.filename)}</div></div>
    <div class="summary-card"><div class="label">Phases Completed</div><div class="value">{summary['phases_completed']}</div></div>
    <div class="summary-card"><div class="label">Actions Logged</div><div class="value">{summary['actions_logged']}</div></div>
    <div class="summary-card"><div class="label">Errors</div><div class="value">{summary['errors']}</div></div>
    <div class="summary-card"><div class="label">Warnings</div><div class="value">{summary['warnings']}</div></div>
  </div>
  {''.join(section_html)}
</body>
</html>"""
            out.write_text(html_doc, encoding="utf-8")
            return str(out)
        except Exception:
            return ""
