#!/usr/bin/env python3
"""
Multi-brain consensus engine test.
Parses all TWBX files, extracts calculated fields,
runs each through MultiBrainEngine, saves results.
"""
import glob
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime

sys.path.insert(0, "/home/zamoritacr/taop-repos/dr-data")

from core.multi_brain import MultiBrainEngine
from core.tableau_extractor import extract_workbook

TWBX_DIR = "/home/zamoritacr/twbx-work"
OUT_FILE = "/tmp/multi_brain_test_results.md"

# How many fields to test per file (None = all)
LIMIT_PER_FILE = 5


def load_twbx(path: str):
    with zipfile.ZipFile(path) as z:
        twb_files = [n for n in z.namelist() if n.endswith(".twb")]
        if not twb_files:
            return None
        twb_data = z.read(twb_files[0])
        return extract_workbook(io.BytesIO(twb_data))


def derive_table_name(twbx_path: str) -> str:
    base = os.path.basename(twbx_path).replace(".twbx", "")
    # Normalize to PBI-style table name
    import re
    return re.sub(r"[^A-Za-z0-9 ]", " ", base).strip()


def main():
    engine = MultiBrainEngine()
    print(f"[TEST] Ollama models: {engine.ollama_models}")
    print(f"[TEST] Anthropic key: {'set' if os.getenv('ANTHROPIC_API_KEY') else 'not set'}")
    print(f"[TEST] OpenAI key:    {'set' if os.getenv('OPENAI_API_KEY') else 'not set'}")
    print()

    twbx_files = sorted(glob.glob(f"{TWBX_DIR}/*.twbx"))
    if not twbx_files:
        print("ERROR: No TWBX files found.")
        return

    all_results = {}  # file -> list of result dicts

    for twbx_path in twbx_files:
        fname = os.path.basename(twbx_path)
        print(f"\n{'='*60}")
        print(f"FILE: {fname}")
        print(f"{'='*60}")

        try:
            wb = load_twbx(twbx_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            all_results[fname] = [{"error": str(e)}]
            continue

        if wb is None:
            print("  No .twb inside zip, skipping.")
            continue

        # Filter to non-trivial fields
        calc_fields = [
            cf for cf in wb.calculated_fields
            if cf.formula and cf.formula.strip() not in ("1", "", "True", "False")
        ]

        if not calc_fields:
            print("  No calculated fields found.")
            all_results[fname] = []
            continue

        print(f"  Found {len(calc_fields)} non-trivial calculated fields")

        # Limit for testing
        test_fields = calc_fields[:LIMIT_PER_FILE]
        table_name = derive_table_name(twbx_path)

        # Build column list from workbook
        columns = [cf.name for cf in wb.calculated_fields]
        # Also try to get data source columns if available
        if hasattr(wb, "datasources") and wb.datasources:
            ds = wb.datasources[0]
            if hasattr(ds, "columns"):
                ds_cols = ds.columns
                if isinstance(ds_cols, dict):
                    columns = list(ds_cols.keys()) + columns
                elif isinstance(ds_cols, (list, tuple)):
                    columns = [str(c) for c in ds_cols] + columns
        columns = list(dict.fromkeys(columns))[:60]  # dedupe, limit

        file_results = []
        for i, cf in enumerate(test_fields, 1):
            print(f"\n  [{i}/{len(test_fields)}] Field: {cf.name!r}")
            formula_preview = cf.formula[:100].replace("\n", " ")
            print(f"    Formula: {formula_preview}...")

            t0 = time.time()
            result = engine.translate_formula(
                tableau_formula=cf.formula,
                table_name=table_name,
                columns=columns,
            )
            elapsed = round(time.time() - t0, 1)

            print(f"    Winner:    {result.get('winner')}")
            print(f"    Confidence: {result.get('confidence', 0):.0%}")
            print(f"    Agreement:  {result.get('agreement_score', 0):.0%}")
            print(f"    Time:       {elapsed}s")
            dax_preview = (result.get("best_dax") or "")[:100].replace("\n", " ")
            print(f"    DAX:        {dax_preview}...")

            file_results.append({
                "field_name": cf.name,
                "tableau_formula": cf.formula,
                "result": result,
                "elapsed_s": elapsed,
            })

        all_results[fname] = file_results

    # Write markdown report
    write_report(all_results, engine)
    print(f"\n\n[TEST] Results saved to {OUT_FILE}")


def write_report(all_results: dict, engine: MultiBrainEngine):
    lines = [
        f"# Multi-Brain Test Results",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Engine Configuration",
        f"- Ollama models: {', '.join(engine.ollama_models) or 'none'}",
        f"- Claude Opus: {'yes' if os.getenv('ANTHROPIC_API_KEY') else 'no (no API key)'}",
        f"- GPT-4o: {'yes' if os.getenv('OPENAI_API_KEY') else 'no (no API key)'}",
        f"- Limit per file: {LIMIT_PER_FILE}",
        f"",
    ]

    total_fields = sum(len(v) for v in all_results.values())
    total_errors = sum(1 for v in all_results.values() for r in v if r.get("error"))
    avg_conf = 0.0
    conf_count = 0
    for v in all_results.values():
        for r in v:
            c = r.get("result", {}).get("confidence", 0)
            if c:
                avg_conf += c
                conf_count += 1
    avg_conf = avg_conf / conf_count if conf_count else 0.0

    lines += [
        f"## Summary",
        f"- Files tested: {len(all_results)}",
        f"- Fields tested: {total_fields}",
        f"- Errors: {total_errors}",
        f"- Average confidence: {avg_conf:.0%}",
        f"",
    ]

    for fname, file_results in all_results.items():
        lines.append(f"---")
        lines.append(f"## {fname}")
        lines.append(f"")

        if not file_results:
            lines.append("_No calculated fields found._")
            lines.append("")
            continue

        if len(file_results) == 1 and file_results[0].get("error"):
            lines.append(f"**ERROR**: {file_results[0]['error']}")
            lines.append("")
            continue

        for r in file_results:
            if r.get("error"):
                lines.append(f"- **ERROR**: {r['error']}")
                continue

            res = r.get("result", {})
            field_name = r.get("field_name", "?")
            formula = r.get("tableau_formula", "")
            best_dax = res.get("best_dax", "")
            confidence = res.get("confidence", 0)
            agreement = res.get("agreement_score", 0)
            winner = res.get("winner", "?")
            reasoning = res.get("reasoning", "")
            disagreements = res.get("disagreements", [])
            per_brain = res.get("per_brain", {})
            elapsed = r.get("elapsed_s", 0)

            lines += [
                f"### `{field_name}`",
                f"",
                f"**Tableau Formula:**",
                f"```",
                formula[:500],
                f"```",
                f"",
                f"**Best DAX** (winner: `{winner}`, confidence: {confidence:.0%}, agreement: {agreement:.0%}, time: {elapsed}s):",
                f"```dax",
                best_dax[:1000] if best_dax else "/* no output */",
                f"```",
                f"",
            ]

            if reasoning:
                lines.append(f"**Reasoning:** {reasoning}")
                lines.append("")

            if disagreements:
                lines.append(f"**Disagreements:**")
                for d in disagreements[:3]:
                    lines.append(f"- `{d[:150]}`")
                lines.append("")

            if per_brain:
                lines.append(f"**Per-Brain Scores:**")
                lines.append(f"| Brain | Score | Issue |")
                lines.append(f"|-------|-------|-------|")
                for brain, info in per_brain.items():
                    score = info.get("score", "?")
                    issue = str(info.get("issue", ""))[:100]
                    lines.append(f"| {brain} | {score} | {issue} |")
                lines.append("")

            # Also show all_results for each brain
            all_brain_res = res.get("all_results", {})
            if all_brain_res:
                lines.append("<details><summary>All brain outputs</summary>")
                lines.append("")
                for brain, bres in all_brain_res.items():
                    dax = bres.get("dax", "")
                    conf = bres.get("confidence", 0)
                    notes = bres.get("notes", "")
                    err = bres.get("error", "")
                    lat = bres.get("_latency_s", "?")
                    lines.append(f"**{brain}** (conf: {conf:.0%}, latency: {lat}s)")
                    if err:
                        lines.append(f"ERROR: {err}")
                    else:
                        lines.append(f"```dax")
                        lines.append(dax[:400] if dax else "/* empty */")
                        lines.append(f"```")
                        if notes:
                            lines.append(f"Notes: {notes}")
                    lines.append("")
                lines.append("</details>")
                lines.append("")

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
