"""
Trace Logger -- audit trail for every deliverable Dr. Data produces.

Records file loads, analysis steps, relationships, LLM calls, and
deliverable generation into a structured log that can be exported
as Markdown or branded HTML.
"""
import os
import json
from datetime import datetime


class TraceLogger:
    """Creates a detailed audit trail document for each session."""

    def __init__(self):
        self.entries = []

    def log(self, action, details):
        self.entries.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details,
        })

    def log_file_loaded(self, filename, rows, cols, dtypes_summary):
        self.log("file_loaded", {
            "filename": filename,
            "rows": rows,
            "columns": cols,
            "dtypes": dtypes_summary,
        })

    def log_analysis(self, insights_found, columns_analyzed, methods_used):
        self.log("analysis", {
            "insights_found": insights_found,
            "columns_analyzed": columns_analyzed,
            "methods_used": methods_used,
        })

    def log_relationships(self, relationships_list):
        self.log("relationships", {
            "count": len(relationships_list),
            "details": relationships_list,
        })

    def log_deliverable(self, deliverable_type, output_path, config):
        self.log("deliverable", {
            "type": deliverable_type,
            "output_path": output_path,
            "config": config,
        })

    def log_llm_call(self, engine, model, prompt_summary, response_summary):
        self.log("llm_call", {
            "engine": engine,
            "model": model,
            "prompt_summary": str(prompt_summary)[:200],
            "response_summary": str(response_summary)[:200],
        })

    def reset(self):
        self.entries = []

    # ------------------------------------------------------------------
    # Document generation
    # ------------------------------------------------------------------

    def generate_trace_doc(self, output_path, format="md"):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if format == "html":
            content = self._build_html()
        else:
            content = self._build_md()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    def _by_action(self, action):
        return [e for e in self.entries if e["action"] == action]

    def _qa_recommendations(self):
        tips = []
        for e in self._by_action("relationships"):
            for rel in e["details"].get("details", []):
                if isinstance(rel, dict):
                    tips.append(
                        f"Verify the join between {rel.get('left_table', '?')} "
                        f"and {rel.get('right_table', '?')} produced the "
                        f"expected row count (overlap {rel.get('overlap_pct', '?')}%)."
                    )
        for e in self._by_action("analysis"):
            for ins in e["details"].get("insights_found", []):
                if "outlier" in str(ins).lower():
                    tips.append(f"Check that outlier values are legitimate: {ins}")
        for e in self._by_action("file_loaded"):
            d = e["details"]
            tips.append(
                f"Confirm {d['filename']} loaded {d['rows']} rows and "
                f"{d['columns']} columns as expected."
            )
        if not tips:
            tips.append("No specific QA flags -- review deliverables manually.")
        return tips

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _build_md(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Dr. Data - Processing Trace Log",
            f"Generated: {now}", "",
        ]

        files = self._by_action("file_loaded")
        if files:
            lines.append("## Files Loaded")
            for e in files:
                d = e["details"]
                lines.append(
                    f"- **{d['filename']}** -- {d['rows']} rows, "
                    f"{d['columns']} columns ({e['timestamp']})"
                )
                if d.get("dtypes"):
                    lines.append(f"  - Types: {d['dtypes']}")
            lines.append("")

        analyses = self._by_action("analysis")
        if analyses:
            lines.append("## Analysis Performed")
            for e in analyses:
                d = e["details"]
                lines.append(f"- Methods: {', '.join(d.get('methods_used', []))}")
                lines.append(f"  - Columns: {', '.join(d.get('columns_analyzed', []))}")
                for ins in d.get("insights_found", []):
                    lines.append(f"  - Finding: {ins}")
            lines.append("")

        rels = self._by_action("relationships")
        if rels:
            lines.append("## Data Relationships")
            for e in rels:
                lines.append(f"- {e['details']['count']} relationship(s) detected")
                for rel in e["details"].get("details", []):
                    if isinstance(rel, dict):
                        lines.append(
                            f"  - {rel.get('left_table','?')}.{rel.get('left_col','?')} "
                            f"<-> {rel.get('right_table','?')}.{rel.get('right_col','?')} "
                            f"({rel.get('match_type','?')}, {rel.get('overlap_pct','?')}%)"
                        )
            lines.append("")

        delivs = self._by_action("deliverable")
        if delivs:
            lines.append("## Deliverables Produced")
            for e in delivs:
                d = e["details"]
                lines.append(f"- **{d['type']}** -> `{d['output_path']}` ({e['timestamp']})")
                cfg = d.get("config", {})
                if cfg:
                    for k, v in cfg.items():
                        lines.append(f"  - {k}: {v}")
            lines.append("")

        llm = self._by_action("llm_call")
        if llm:
            lines.append("## LLM Calls")
            for e in llm:
                d = e["details"]
                lines.append(
                    f"- [{d['engine']}/{d['model']}] {e['timestamp']}"
                )
                lines.append(f"  - Prompt: {d['prompt_summary']}")
                lines.append(f"  - Response: {d['response_summary']}")
            lines.append("")

        lines.append("## Recommendations for QA")
        for tip in self._qa_recommendations():
            lines.append(f"- {tip}")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML (WU branded)
    # ------------------------------------------------------------------

    def _build_html(self):
        md = self._build_md()
        # Convert markdown to simple HTML
        body_lines = []
        for line in md.split("\n"):
            if line.startswith("# "):
                body_lines.append(f"<h1>{_esc(line[2:])}</h1>")
            elif line.startswith("## "):
                body_lines.append(f"<h2>{_esc(line[3:])}</h2>")
            elif line.startswith("  - "):
                body_lines.append(f"<div style='margin-left:24px;color:#ccc;'>{_md_inline(line[4:])}</div>")
            elif line.startswith("- "):
                body_lines.append(f"<div style='margin-left:8px;padding:2px 0;'>{_md_inline(line[2:])}</div>")
            elif line.strip():
                body_lines.append(f"<p>{_md_inline(line)}</p>")

        body = "\n".join(body_lines)
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<style>"
            "body{background:#1a1a1a;color:#fff;font-family:Arial,sans-serif;"
            "max-width:800px;margin:40px auto;padding:0 20px;line-height:1.6;}"
            "h1{color:#FFDE00;border-bottom:2px solid #FFDE00;padding-bottom:8px;}"
            "h2{color:#FFDE00;margin-top:28px;}"
            "code{background:#2d2d2d;padding:2px 6px;border-radius:3px;color:#FFE600;}"
            "strong{color:#FFDE00;}"
            "</style></head><body>" + body + "</body></html>"
        )


def _esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _md_inline(text):
    """Minimal markdown inline: **bold** and `code`."""
    import re
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text
