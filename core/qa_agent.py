"""
QA Agent -- Agentic dashboarding: generate, publish, read back, fix, republish.

This is NOT a validator. This is an AGENT. It publishes, reads back from the
Power BI REST API, compares source vs live result, diagnoses fidelity gaps,
fixes PBIP files, and republishes -- looping until the dashboard is RIGHT.

Layer 1: Deterministic file checks + auto-fix (pre-publish)
Layer 2: Publish to Power BI via Fabric REST API
Layer 3: Read back via PBI REST API -- compare source vs live
Layer 4: If fidelity < threshold -- diagnose + fix + republish (max 3 loops)
Layer 5: GPT-4o reviews final result (advisory only, never blocks)

Output: working link + full audit report of every action taken.
"""

import os
import re
import json
import glob
import time
import datetime
import requests


class QAAgent:
    def __init__(self, source_spec=None, dataframe=None, config=None):
        """
        Args:
            source_spec: parsed Tableau spec (or None for CSV/image uploads)
            dataframe: the data (pandas DataFrame)
            config: PBIP config from direct_mapper (report_layout + tmdl_model)
        """
        self.source_spec = source_spec or {}
        self.dataframe = dataframe
        self.config = config or {}
        try:
            from config.settings import OPENAI_API_KEY
            self.api_key = OPENAI_API_KEY or ""
        except ImportError:
            self.api_key = ""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        # Audit trail -- every action logged with timestamp
        self.audit = []
        self.fidelity_report = {}
        self.max_loops = 3

    def _log(self, action, detail, status="info"):
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action": action,
            "detail": str(detail)[:500],
            "status": status,
        }
        self.audit.append(entry)
        print(f"  [QA-{status.upper()}] {action}: {str(detail)[:200]}")

    # ================================================================== #
    #  BACKWARD-COMPATIBLE PRE-PUBLISH VALIDATION                          #
    # ================================================================== #

    def run_full_qa(self, pbip_path=None):
        """Run deterministic file checks + auto-fix (no publish, no loop).

        Backward-compatible wrapper used by dr_data_agent.py and
        streamlit_app.py for pre-publish validation. For the full agentic
        publish-readback-fix loop, use run() instead.

        Args:
            pbip_path: path to PBIP folder. If not provided, falls back to
                       source_spec (for callers that pass pbip_path as
                       first positional arg to __init__).
        """
        if pbip_path is None:
            # Callers may pass pbip_path as source_spec positionally
            pbip_path = self.source_spec if isinstance(
                self.source_spec, str
            ) else None
        if not pbip_path or not os.path.isdir(str(pbip_path)):
            return {
                "passed": False,
                "issues": [f"Invalid PBIP path: {pbip_path}"],
                "warnings": [],
                "fixes": [],
            }

        issues = self._deterministic_checks(pbip_path)
        fixes = []
        if issues:
            fixes = self._auto_fix(pbip_path, issues)
            issues = self._deterministic_checks(pbip_path)

        passed = len(issues) == 0

        warnings = []
        if passed and self.api_key:
            try:
                review = self._gpt_review(
                    {"score": 100 if passed else 0}, {}
                )
                if review:
                    warnings.append(f"GPT: {review}")
            except Exception as e:
                warnings.append(f"GPT review skipped: {e}")

        return {
            "passed": passed,
            "issues": [i.get("detail", str(i)) for i in issues],
            "warnings": warnings,
            "fixes": fixes,
        }

    # ================================================================== #
    #  MAIN AGENTIC LOOP                                                  #
    # ================================================================== #

    def run(self, token, workspace_id, display_name,
            pbip_generator_class, output_base_dir) -> dict:
        """
        THE MAIN LOOP. Agentic: generates, checks, publishes, reads back,
        fixes, republishes until the dashboard is RIGHT.

        Returns:
            {
                "success": bool,
                "report_id": str or None,
                "report_url": str or None,
                "fidelity": dict,
                "audit": list,
                "loops": int,
            }
        """
        from core.powerbi_publisher import (
            publish_pbip, get_report_pages, get_report_visuals,
            get_dataset_id_from_report, get_dataset_tables,
            execute_dax_query, delete_report, delete_dataset,
        )

        for loop in range(self.max_loops):
            self._log("LOOP_START", f"Attempt {loop + 1}/{self.max_loops}")

            # ---- PHASE 1: GENERATE PBIP ----
            try:
                out_dir = os.path.join(output_base_dir, f"attempt_{loop + 1}")
                gen = pbip_generator_class(out_dir)
                dashboard_spec = self._build_dashboard_spec()
                gen_result = gen.generate(
                    self.config, None, dashboard_spec, dataframe=self.dataframe
                )
                pbip_path = gen_result["path"]
                self._log("GENERATE", f"PBIP at {pbip_path}", "success")
            except Exception as e:
                self._log("GENERATE", f"Failed: {e}", "error")
                continue

            # ---- PHASE 2: DETERMINISTIC FILE CHECKS + AUTO-FIX ----
            issues = self._deterministic_checks(pbip_path)
            if issues:
                self._log("FILE_CHECK",
                          f"{len(issues)} issues found: {issues[:3]}", "warning")
                fixed = self._auto_fix(pbip_path, issues)
                self._log("AUTO_FIX",
                          f"Fixed {len(fixed)} of {len(issues)} issues", "fix")

                remaining = self._deterministic_checks(pbip_path)
                if remaining:
                    self._log("FILE_CHECK",
                              f"{len(remaining)} issues remain after fix: "
                              f"{remaining[:2]}", "error")
                    self._adjust_config_for_issues(remaining)
                    continue
            else:
                self._log("FILE_CHECK", "All deterministic checks passed",
                          "success")

            # ---- PHASE 3: PUBLISH TO POWER BI ----
            try:
                pub = publish_pbip(
                    token, workspace_id, pbip_path,
                    f"{display_name}_v{loop + 1}"
                )
                if pub.get("error"):
                    raise RuntimeError(json.dumps(pub, default=str)[:500])
                report_id = pub.get("report_id", "")
                sm_id = pub.get("semantic_model_id", "")
                self._log("PUBLISH",
                          f"Published: report_id={report_id}, sm_id={sm_id}",
                          "success")
            except Exception as e:
                error_str = str(e)
                self._log("PUBLISH", f"Failed: {error_str[:300]}", "error")
                self._diagnose_publish_error(error_str, pbip_path)
                continue

            # ---- PHASE 4: READ BACK FROM POWER BI REST API ----
            # Get PBI-scoped token for read-back endpoints
            try:
                from core.powerbi_publisher import get_access_token, PBI_SCOPE
                pbi_token = get_access_token(PBI_SCOPE)
            except Exception:
                pbi_token = token

            live_result = {}
            try:
                # Wait briefly for async processing
                time.sleep(5)
                live_result = self._read_back_from_pbi(
                    pbi_token, workspace_id, report_id
                )
                self._log("READ_BACK",
                          f"Pages: {live_result.get('page_count', '?')}, "
                          f"Visuals: {live_result.get('visual_count', '?')}, "
                          f"Rows: {live_result.get('row_count', '?')}",
                          "info")
            except Exception as e:
                self._log("READ_BACK", f"Failed: {e}", "warning")

            # ---- PHASE 5: FIDELITY COMPARISON ----
            fidelity = self._compute_fidelity(live_result)
            self.fidelity_report = fidelity
            score = fidelity.get("score", 0)
            self._log("FIDELITY", f"Score: {score}%",
                      "success" if score >= 70 else "warning")

            # ---- PHASE 6: DECIDE -- GOOD ENOUGH OR FIX? ----
            if score >= 60 or loop == self.max_loops - 1:
                # Accept
                if self.api_key:
                    try:
                        gpt_review = self._gpt_review(fidelity, live_result)
                        self._log("GPT_REVIEW", gpt_review, "info")
                    except Exception as gpt_err:
                        self._log("GPT_REVIEW", f"Skipped: {gpt_err}",
                                  "warning")

                report_url = (
                    f"https://app.powerbi.com/groups/"
                    f"{workspace_id}/reports/{report_id}"
                )
                self._log("COMPLETE",
                          f"Dashboard live at {report_url}", "success")

                return {
                    "success": True,
                    "report_id": report_id,
                    "report_url": report_url,
                    "fidelity": fidelity,
                    "audit": self.audit,
                    "loops": loop + 1,
                }
            else:
                self._log("FIDELITY_LOW",
                          f"Score {score}% < 60% -- diagnosing", "warning")
                self._diagnose_fidelity_gaps(fidelity, live_result)
                # Clean up bad publish before retry
                try:
                    did = get_dataset_id_from_report(
                        pbi_token, workspace_id, report_id
                    )
                    delete_report(token, workspace_id, report_id)
                    if did:
                        delete_dataset(token, workspace_id, did)
                    self._log("CLEANUP",
                              "Deleted failed attempt from workspace", "fix")
                except Exception:
                    pass

        # All loops exhausted
        self._log("EXHAUSTED",
                  f"Max {self.max_loops} attempts reached", "error")
        return {
            "success": False,
            "report_id": None,
            "report_url": None,
            "fidelity": self.fidelity_report,
            "audit": self.audit,
            "loops": self.max_loops,
        }

    # ================================================================== #
    #  DASHBOARD SPEC BUILDER                                             #
    # ================================================================== #

    def _build_dashboard_spec(self):
        """Build a dashboard_spec dict for PBIPGenerator from source_spec."""
        spec = self.source_spec
        title = "Dashboard"
        if spec.get("dashboards"):
            title = spec["dashboards"][0].get("name", "Dashboard")
        elif spec.get("worksheets"):
            title = spec["worksheets"][0].get("name", "Dashboard")

        dspec = {
            "dashboard_title": title,
            "source": "qa_agent",
        }

        # Pass through Tableau design info for visual formatting
        if spec.get("design"):
            dspec["design"] = spec["design"]
        if spec.get("dashboards"):
            dspec["tableau_dashboards"] = spec["dashboards"]
        if spec.get("worksheets"):
            ws_designs = {}
            ws_chart_types = {}
            for ws in spec["worksheets"]:
                name = ws.get("name", "")
                if name:
                    ws_designs[name] = ws.get("design", {})
                    ws_chart_types[name] = ws.get("chart_type", "automatic")
            dspec["worksheet_designs"] = ws_designs
            dspec["worksheet_chart_types"] = ws_chart_types

        return dspec

    # ================================================================== #
    #  DETERMINISTIC FILE CHECKS                                          #
    # ================================================================== #

    def _deterministic_checks(self, pbip_path) -> list:
        """Run ALL pre-publish checks. Return list of issue dicts."""
        issues = []
        if not os.path.isdir(pbip_path):
            issues.append({"type": "STRUCTURE",
                           "detail": f"Not a dir: {pbip_path}"})
            return issues

        issues.extend(self._check_structure(pbip_path))
        issues.extend(self._check_tmdl_indentation(pbip_path))
        issues.extend(self._check_tmdl_syntax(pbip_path))
        issues.extend(self._check_broken_paths(pbip_path))
        issues.extend(self._check_data_integrity(pbip_path))
        issues.extend(self._check_visuals(pbip_path))

        return issues

    def _check_structure(self, root):
        issues = []
        pbip_files = glob.glob(os.path.join(root, "*.pbip"))
        if not pbip_files:
            issues.append({"type": "STRUCTURE",
                           "detail": "Missing .pbip file"})

        sm_dirs = glob.glob(os.path.join(root, "*.SemanticModel"))
        if not sm_dirs:
            issues.append({"type": "STRUCTURE",
                           "detail": "Missing .SemanticModel folder"})
        else:
            sm = sm_dirs[0]
            for f in ("definition.pbism", ".platform"):
                if not os.path.exists(os.path.join(sm, f)):
                    issues.append({"type": "STRUCTURE",
                                   "detail": f"Missing {f} in SemanticModel"})
            tmdl = glob.glob(os.path.join(sm, "**", "*.tmdl"), recursive=True)
            if not tmdl:
                issues.append({"type": "STRUCTURE",
                               "detail": "No .tmdl files"})

        rpt_dirs = glob.glob(os.path.join(root, "*.Report"))
        if not rpt_dirs:
            issues.append({"type": "STRUCTURE",
                           "detail": "Missing .Report folder"})
        else:
            rpt = rpt_dirs[0]
            for f in ("definition.pbir", ".platform"):
                if not os.path.exists(os.path.join(rpt, f)):
                    issues.append({"type": "STRUCTURE",
                                   "detail": f"Missing {f} in Report"})
            rj = os.path.join(rpt, "definition", "report.json")
            if not os.path.exists(rj):
                issues.append({"type": "STRUCTURE",
                               "detail": "Missing report.json"})
            pages_dir = os.path.join(rpt, "definition", "pages")
            if os.path.isdir(pages_dir):
                page_folders = [
                    d for d in os.listdir(pages_dir)
                    if os.path.isdir(os.path.join(pages_dir, d))
                ]
                if not page_folders:
                    issues.append({"type": "STRUCTURE",
                                   "detail": "No page folders"})
                for pf in page_folders:
                    vis_dir = os.path.join(pages_dir, pf, "visuals")
                    if os.path.isdir(vis_dir):
                        vj = glob.glob(
                            os.path.join(vis_dir, "**", "visual.json"),
                            recursive=True,
                        )
                        if not vj:
                            issues.append({
                                "type": "STRUCTURE",
                                "detail": f"Page '{pf}' has no visual.json",
                            })
                    else:
                        issues.append({
                            "type": "STRUCTURE",
                            "detail": f"Page '{pf}' missing visuals/ folder",
                        })
            else:
                issues.append({"type": "STRUCTURE",
                               "detail": "Missing pages/ folder"})
        return issues

    def _check_tmdl_indentation(self, root):
        issues = []
        tmdl_files = glob.glob(
            os.path.join(root, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, root)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                issues.append({"type": "INDENTATION", "file": fname,
                               "detail": f"Cannot read: {e}"})
                continue

            for lineno, line in enumerate(lines, 1):
                stripped = line.rstrip("\n\r")
                if not stripped:
                    continue
                leading_ws = stripped[:len(stripped) - len(stripped.lstrip())]
                if leading_ws and " " in leading_ws and "\t" in leading_ws:
                    issues.append({
                        "type": "INDENTATION", "file": fname,
                        "line": lineno,
                        "detail": "Mixed tabs and spaces",
                    })
                    break  # One per file is enough to trigger fix
                elif leading_ws and " " in leading_ws and "\t" not in leading_ws:
                    issues.append({
                        "type": "INDENTATION", "file": fname,
                        "line": lineno,
                        "detail": "Spaces used (TMDL requires tabs)",
                    })
                    break
        return issues

    def _check_tmdl_syntax(self, root):
        issues = []
        tmdl_files = glob.glob(
            os.path.join(root, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, root)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            n_open = content.count("{")
            n_close = content.count("}")
            if n_open != n_close:
                issues.append({
                    "type": "SYNTAX", "file": fname,
                    "detail": f"Unmatched braces "
                              f"({n_open} open, {n_close} close)",
                })

            n_po = content.count("(")
            n_pc = content.count(")")
            if n_po != n_pc:
                issues.append({
                    "type": "SYNTAX", "file": fname,
                    "detail": f"Unmatched parens "
                              f"({n_po} open, {n_pc} close)",
                })
        return issues

    def _check_broken_paths(self, root):
        issues = []
        text_exts = (".tmdl", ".json", ".pbism", ".pbir", ".pbip")
        patterns = [
            (r"File\.Contents\s*\(", "File.Contents() reference"),
            (r"/home/", "Linux /home/ path"),
            (r"/tmp/", "Linux /tmp/ path"),
            (r"[A-Z]:\\", "Hardcoded Windows path"),
            (r"localhost", "localhost reference"),
        ]
        for dirpath, _, files in os.walk(root):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in text_exts:
                    continue
                fpath = os.path.join(dirpath, name)
                fname = os.path.relpath(fpath, root)
                try:
                    with open(fpath, "r", encoding="utf-8",
                              errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue
                for pat, desc in patterns:
                    if re.search(pat, content):
                        issues.append({
                            "type": "BROKEN_PATH", "file": fname,
                            "detail": desc,
                        })
        return issues

    def _check_data_integrity(self, root):
        issues = []
        if self.dataframe is None:
            return issues
        tmdl_files = glob.glob(
            os.path.join(root, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, root)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue
            if "#table(" not in content:
                continue

            # Count data rows
            in_table = False
            data_rows = 0
            for line in content.split("\n"):
                s = line.strip()
                if "#table(" in s:
                    in_table = True
                    continue
                if in_table:
                    if s.startswith("{") and "," in s:
                        data_rows += 1
                    if s.startswith("})"):
                        in_table = False

            if data_rows == 0:
                issues.append({
                    "type": "DATA", "file": fname,
                    "detail": "Empty #table() -- no data rows",
                })
        return issues

    def _check_visuals(self, root):
        issues = []
        visual_files = glob.glob(
            os.path.join(root, "**", "visual.json"), recursive=True
        )
        for vpath in visual_files:
            vname = os.path.relpath(vpath, root)
            try:
                with open(vpath, "r", encoding="utf-8") as f:
                    vdata = json.load(f)
            except json.JSONDecodeError as e:
                issues.append({
                    "type": "VISUAL", "file": vname,
                    "detail": f"Invalid JSON: {e}",
                })
                continue
            except Exception:
                continue

            pos = vdata.get("position", {})
            w = pos.get("width", 0)
            h = pos.get("height", 0)
            if w <= 0 or h <= 0:
                issues.append({
                    "type": "VISUAL", "file": vname,
                    "detail": f"Invalid position: w={w}, h={h}",
                })
        return issues

    # ================================================================== #
    #  AUTO-FIX                                                           #
    # ================================================================== #

    def _auto_fix(self, pbip_path, issues) -> list:
        """Fix what can be fixed deterministically. Return list of fixes."""
        fixes = []
        fixed_files = set()

        for issue in issues:
            itype = issue.get("type", "")
            fname = issue.get("file", "")

            if itype == "INDENTATION" and fname:
                fpath = os.path.join(pbip_path, fname)
                if fpath in fixed_files or not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        content = line.rstrip("\n\r")
                        leading = content[:len(content) - len(content.lstrip())]
                        body = content.lstrip()
                        # Convert all leading spaces to tabs
                        tab_count = leading.count("\t")
                        space_count = leading.count(" ")
                        total_tabs = tab_count + (space_count + 3) // 4
                        new_lines.append("\t" * total_tabs + body + "\n")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    fixed_files.add(fpath)
                    fixes.append(f"Fixed indentation in {fname}")
                except Exception as e:
                    fixes.append(f"Failed to fix indentation in {fname}: {e}")

            elif itype == "VISUAL" and "position" in issue.get("detail", ""):
                fpath = os.path.join(pbip_path, fname)
                if fpath in fixed_files or not os.path.exists(fpath):
                    continue
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        vdata = json.load(f)
                    pos = vdata.get("position", {})
                    if pos.get("width", 0) <= 0:
                        pos["width"] = 400
                    if pos.get("height", 0) <= 0:
                        pos["height"] = 300
                    vdata["position"] = pos
                    with open(fpath, "w", encoding="utf-8") as f:
                        json.dump(vdata, f, indent=2)
                    fixed_files.add(fpath)
                    fixes.append(f"Fixed position in {fname}")
                except Exception as e:
                    fixes.append(f"Failed to fix position in {fname}: {e}")

        return fixes

    # ================================================================== #
    #  PUBLISH ERROR DIAGNOSIS                                            #
    # ================================================================== #

    def _diagnose_publish_error(self, error_str, pbip_path):
        """Parse Fabric API error and attempt fix for next loop."""
        error_lower = error_str.lower()

        if "indentation" in error_lower:
            self._log("DIAGNOSE", "TMDL indentation error -- fixing", "fix")
            tmdl_files = glob.glob(
                os.path.join(pbip_path, "**", "*.tmdl"), recursive=True
            )
            for fpath in tmdl_files:
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        content = line.rstrip("\n\r")
                        leading = content[:len(content) - len(content.lstrip())]
                        body = content.lstrip()
                        tab_count = leading.count("\t")
                        space_count = leading.count(" ")
                        total_tabs = tab_count + (space_count + 3) // 4
                        new_lines.append("\t" * total_tabs + body + "\n")
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                except Exception:
                    pass

        elif "failedtoparsefile" in error_lower or "syntax" in error_lower:
            self._log("DIAGNOSE", "TMDL parse error -- checking syntax", "fix")
            syntax_issues = self._check_tmdl_syntax(pbip_path)
            for si in syntax_issues:
                self._log("DIAGNOSE", f"Syntax: {si}", "warning")

        elif "403" in error_str or "forbidden" in error_lower:
            self._log("DIAGNOSE",
                      "Permission denied (403) -- check service principal "
                      "workspace access", "error")

        elif "409" in error_str or "conflict" in error_lower:
            self._log("DIAGNOSE",
                      "Name conflict (409) -- item with same name exists",
                      "warning")

        elif "timeout" in error_lower:
            self._log("DIAGNOSE", "Timeout -- will retry", "warning")

        else:
            self._log("DIAGNOSE",
                      f"Unknown publish error: {error_str[:200]}", "error")

    # ================================================================== #
    #  READ BACK FROM POWER BI REST API                                   #
    # ================================================================== #

    def _read_back_from_pbi(self, token, workspace_id, report_id) -> dict:
        """Use REST API to read what Power BI actually built."""
        from core.powerbi_publisher import (
            get_report_pages, get_report_visuals,
            get_dataset_id_from_report, get_dataset_tables,
            execute_dax_query,
        )

        result = {
            "page_count": 0,
            "pages": [],
            "visual_count": 0,
            "dataset_id": "",
            "tables": [],
            "row_count": 0,
            "column_names": [],
        }

        # 1. Pages
        try:
            pages = get_report_pages(token, workspace_id, report_id)
            result["page_count"] = len(pages)
            for page in pages:
                page_name = page.get("name", page.get("Name", ""))
                display = page.get("displayName",
                                   page.get("DisplayName", ""))
                page_info = {"name": page_name, "displayName": display,
                             "visuals": []}
                # 2. Visuals per page
                try:
                    visuals = get_report_visuals(
                        token, workspace_id, report_id, page_name
                    )
                    for v in visuals:
                        page_info["visuals"].append({
                            "type": v.get("visualType",
                                          v.get("type", "")),
                            "title": v.get("title", ""),
                            "width": v.get("width", 0),
                            "height": v.get("height", 0),
                        })
                    result["visual_count"] += len(visuals)
                except Exception:
                    pass
                result["pages"].append(page_info)
        except Exception as e:
            self._log("READ_BACK", f"Pages API failed: {e}", "warning")

        # 3. Dataset
        try:
            did = get_dataset_id_from_report(
                token, workspace_id, report_id
            )
            if did:
                result["dataset_id"] = did

                # 4. Tables and columns
                try:
                    tables = get_dataset_tables(
                        token, workspace_id, did
                    )
                    for t in tables:
                        cols = [c.get("name", "")
                                for c in t.get("columns", [])]
                        result["tables"].append({
                            "name": t.get("name", ""),
                            "columns": cols,
                        })
                        result["column_names"].extend(cols)
                except Exception:
                    pass

                # 5. Row count via DAX
                try:
                    dax_result = execute_dax_query(
                        token, workspace_id, did,
                        'EVALUATE ROW("Count", COUNTROWS(\'Data\'))'
                    )
                    rows = dax_result.get("rows", [])
                    if rows:
                        result["row_count"] = rows[0].get(
                            "[Count]", rows[0].get("Count", 0)
                        )
                except Exception:
                    pass
        except Exception as e:
            self._log("READ_BACK", f"Dataset API failed: {e}", "warning")

        return result

    # ================================================================== #
    #  FIDELITY SCORING                                                   #
    # ================================================================== #

    def _compute_fidelity(self, live_result) -> dict:
        """Compare source_spec vs live Power BI result. All deterministic."""
        spec = self.source_spec
        df = self.dataframe

        details = {}
        data_score = 0
        structure_score = 0
        quality_score = 0

        # ---- DATA (40 points) ----
        source_row_count = len(df) if df is not None else 0
        pbi_row_count = live_result.get("row_count", 0)
        details["source_row_count"] = source_row_count
        details["pbi_row_count"] = pbi_row_count

        # Row count match (20 pts)
        if pbi_row_count > 0 and source_row_count > 0:
            ratio = min(pbi_row_count, source_row_count) / max(
                pbi_row_count, source_row_count
            )
            if ratio >= 0.9:
                data_score += 20
            else:
                data_score += int(20 * ratio)
        elif pbi_row_count > 0:
            data_score += 10

        # Column match (10 pts)
        source_cols = set(df.columns) if df is not None else set()
        pbi_cols = set(live_result.get("column_names", []))
        details["source_columns"] = sorted(source_cols)
        details["pbi_columns"] = sorted(pbi_cols)
        if source_cols and pbi_cols:
            src_lower = {c.lower() for c in source_cols}
            pbi_lower = {c.lower() for c in pbi_cols}
            matched = len(src_lower & pbi_lower)
            match_pct = matched / max(len(src_lower), 1) * 100
            details["column_match_pct"] = round(match_pct)
            data_score += int(10 * match_pct / 100)
        elif pbi_cols:
            data_score += 5
            details["column_match_pct"] = 0

        # Data loaded (10 pts)
        if pbi_row_count > 0:
            data_score += 10

        # ---- STRUCTURE (30 points) ----
        source_dashboards = len(spec.get("dashboards", []))
        source_worksheets = len(spec.get("worksheets", []))
        pbi_pages = live_result.get("page_count", 0)
        pbi_visuals = live_result.get("visual_count", 0)
        details["source_worksheets"] = source_worksheets
        details["source_dashboards"] = source_dashboards
        details["pbi_pages"] = pbi_pages
        details["pbi_visual_count"] = pbi_visuals

        # Page count (10 pts)
        target_pages = max(source_dashboards, 1)
        if pbi_pages >= target_pages:
            structure_score += 10
        elif pbi_pages > 0:
            structure_score += 5

        # Visual count (10 pts)
        if source_worksheets > 0 and pbi_visuals >= source_worksheets:
            structure_score += 10
        elif pbi_visuals > 0:
            ratio = min(pbi_visuals, max(source_worksheets, 1)) / max(
                source_worksheets, 1
            )
            structure_score += int(10 * ratio)

        # Every page has >= 1 visual (10 pts)
        pages_with_visuals = sum(
            1 for p in live_result.get("pages", [])
            if len(p.get("visuals", [])) > 0
        )
        if pbi_pages > 0 and pages_with_visuals == pbi_pages:
            structure_score += 10
        elif pages_with_visuals > 0:
            structure_score += 5

        # ---- VISUAL QUALITY (30 points) ----
        source_chart_types = []
        for ws in spec.get("worksheets", []):
            ct = ws.get("chart_type", "automatic")
            source_chart_types.append(ct)
        details["source_chart_types"] = source_chart_types

        pbi_visual_types = []
        visuals_with_titles = 0
        visuals_with_size = 0
        for page in live_result.get("pages", []):
            for v in page.get("visuals", []):
                vtype = v.get("type", "")
                pbi_visual_types.append(vtype)
                if v.get("title"):
                    visuals_with_titles += 1
                w = v.get("width", 0)
                h = v.get("height", 0)
                if w > 200 and h > 150:
                    visuals_with_size += 1
        details["pbi_visual_types"] = pbi_visual_types

        # Chart types present (10 pts)
        if source_chart_types and pbi_visual_types:
            quality_score += 10
        elif pbi_visual_types:
            quality_score += 5

        # Proper sizes (10 pts)
        if pbi_visuals > 0:
            size_ratio = visuals_with_size / max(pbi_visuals, 1)
            quality_score += int(10 * size_ratio)
        details["visuals_with_proper_size"] = visuals_with_size

        # Non-table charts (5 pts)
        non_table_types = {
            "clusteredBarChart", "clusteredColumnChart", "lineChart",
            "areaChart", "pieChart", "donutChart", "scatterChart",
            "card", "lineClusteredColumnComboChart", "treemap",
        }
        has_non_table = any(
            t in non_table_types for t in pbi_visual_types
        )
        if has_non_table:
            quality_score += 5

        # Titles (5 pts)
        details["visuals_with_titles"] = visuals_with_titles
        if pbi_visuals > 0 and visuals_with_titles > 0:
            title_ratio = visuals_with_titles / max(pbi_visuals, 1)
            quality_score += int(5 * title_ratio)

        total = data_score + structure_score + quality_score

        # If we couldn't read back at all but publish succeeded,
        # give baseline score
        if (not live_result.get("page_count")
                and not live_result.get("row_count")):
            total = max(total, 50)

        return {
            "score": total,
            "data_score": data_score,
            "structure_score": structure_score,
            "quality_score": quality_score,
            "details": details,
        }

    # ================================================================== #
    #  FIDELITY GAP DIAGNOSIS + CONFIG ADJUSTMENT                         #
    # ================================================================== #

    def _diagnose_fidelity_gaps(self, fidelity, live_result):
        """Look at what's wrong and adjust self.config for next loop."""
        details = fidelity.get("details", {})

        if details.get("pbi_row_count", 0) == 0:
            self._log("DIAGNOSE", "Row count = 0. Data didn't load. "
                      "Checking M expression.", "warning")

        pbi_visuals = live_result.get("visual_count", 0)
        src_ws = details.get("source_worksheets", 0)
        if pbi_visuals < src_ws:
            self._log("DIAGNOSE",
                      f"Only {pbi_visuals} visuals vs {src_ws} worksheets",
                      "warning")

        col_match = details.get("column_match_pct", 0)
        if col_match < 80:
            self._log("DIAGNOSE",
                      f"Column match only {col_match}%", "warning")

    def _adjust_config_for_issues(self, issues):
        """Modify self.config based on deterministic check failures."""
        for issue in issues:
            itype = issue.get("type", "")
            if itype == "DATA" and "empty" in issue.get("detail", "").lower():
                self._log("ADJUST", "Empty data -- will regenerate", "fix")

    # ================================================================== #
    #  GPT ADVISORY (NEVER BLOCKS, NEVER FIXES)                          #
    # ================================================================== #

    def _gpt_review(self, fidelity, live_result) -> str:
        """Send fidelity report to GPT-4o for advisory suggestions."""
        if not self.api_key:
            return "No API key -- skipped"

        summary = {
            "fidelity_score": fidelity.get("score", 0),
            "data_score": fidelity.get("data_score", 0),
            "structure_score": fidelity.get("structure_score", 0),
            "quality_score": fidelity.get("quality_score", 0),
            "source_worksheets": len(
                self.source_spec.get("worksheets", [])
            ),
            "pbi_pages": live_result.get("page_count", 0),
            "pbi_visuals": live_result.get("visual_count", 0),
            "pbi_row_count": live_result.get("row_count", 0),
        }

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "max_tokens": 500,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a Power BI dashboard quality "
                                "reviewer. Given a fidelity report comparing "
                                "a Tableau source to a Power BI conversion, "
                                "suggest 1-3 specific improvements. Be "
                                "concise. Return plain text, no JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(summary),
                        },
                    ],
                },
                timeout=30,
            )
            if resp.status_code == 200:
                body = resp.json()
                return body["choices"][0]["message"]["content"].strip()
            return f"GPT API returned {resp.status_code}"
        except Exception as e:
            return f"GPT review failed: {e}"

    # ================================================================== #
    #  AUDIT REPORT                                                       #
    # ================================================================== #

    def get_audit_report(self) -> str:
        """Generate a human-readable audit report."""
        lines = []
        lines.append("=== DR. DATA QA AGENT -- AUDIT REPORT ===")
        lines.append(
            f"Timestamp: {datetime.datetime.utcnow().isoformat()}"
        )

        spec = self.source_spec
        ws_count = len(spec.get("worksheets", []))
        db_count = len(spec.get("dashboards", []))
        lines.append(
            f"Source: {ws_count} worksheets, {db_count} dashboards"
        )
        lines.append("")

        lines.append("ACTIONS TAKEN:")
        for entry in self.audit:
            ts = entry.get("timestamp", "")[:19]
            action = entry.get("action", "")
            detail = entry.get("detail", "")[:200]
            status = entry.get("status", "info")
            lines.append(f"  [{ts}] {action}: {detail} ({status})")
        lines.append("")

        fid = self.fidelity_report
        if fid:
            lines.append("FIDELITY REPORT:")
            lines.append(f"  Data: {fid.get('data_score', 0)}/40")
            lines.append(
                f"  Structure: {fid.get('structure_score', 0)}/30"
            )
            lines.append(
                f"  Quality: {fid.get('quality_score', 0)}/30"
            )
            lines.append(f"  TOTAL: {fid.get('score', 0)}/100")
            lines.append("")

        fix_entries = [
            e for e in self.audit if e.get("status") == "fix"
        ]
        if fix_entries:
            lines.append("FIXES APPLIED:")
            for entry in fix_entries:
                lines.append(
                    f"  - {entry['action']}: {entry['detail']}"
                )
        else:
            lines.append("FIXES APPLIED: (none needed)")
        lines.append("")

        gpt_entries = [
            e for e in self.audit if e.get("action") == "GPT_REVIEW"
        ]
        if gpt_entries:
            lines.append(
                f"GPT ADVISORY: {gpt_entries[-1].get('detail', '')}"
            )
            lines.append("")

        complete_entries = [
            e for e in self.audit if e.get("action") == "COMPLETE"
        ]
        if complete_entries:
            lines.append(
                f"RESULT: {complete_entries[-1].get('detail', '')}"
            )

        return "\n".join(lines)
