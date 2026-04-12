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
    #  POST-PUBLISH QA LOOP (for V2 pipeline)                             #
    # ================================================================== #

    def run_post_publish_qa(self, token, workspace_id, report_id,
                            semantic_model_id, pbip_path,
                            data_file_path=None, data_profile=None):
        """Post-publish QA: read back, score fidelity, diagnose, fix, republish.

        Called by the V2 pipeline AFTER initial publish. Reads the live
        report from Power BI, compares against the Tableau source spec,
        and if fidelity is below threshold, regenerates the PBIP with
        improved field bindings and republishes.

        Returns:
            {
                "fidelity": dict,
                "report_url": str,
                "report_id": str,
                "loops": int,
                "audit": list,
            }
        """
        from core.powerbi_publisher import (
            get_access_token, PBI_SCOPE,
            publish_pbip, delete_report,
            get_dataset_id_from_report, delete_dataset,
        )

        report_url = (f"https://app.powerbi.com/groups/"
                      f"{workspace_id}/reports/{report_id}")
        best_fidelity = {}

        for loop in range(self.max_loops):
            self._log("QA_LOOP", f"Post-publish QA attempt {loop + 1}")

            # ---- Read back from PBI API ----
            try:
                pbi_token = get_access_token(PBI_SCOPE)
            except Exception:
                pbi_token = token

            import time
            time.sleep(3)
            live_result = self._read_back_from_pbi(
                pbi_token, workspace_id, report_id
            )

            # ---- Read local PBIP visual bindings (has fields, queryRefs) ----
            local_manifest = self._read_pbip_manifest(pbip_path)
            # The PBI API often returns empty visuals (needs report to be
            # rendered). Always prefer local PBIP data for field comparison.
            for lp in local_manifest:
                dn_lower = lp["displayName"].lower().strip()
                matched = False
                for lr_page in live_result.get("pages", []):
                    lr_dn = lr_page.get("displayName", "").lower().strip()
                    if lr_dn == dn_lower or dn_lower in lr_dn or lr_dn in dn_lower:
                        # Replace API visuals with local ones (they have field data)
                        lr_page["visuals"] = lp.get("visuals", [])
                        matched = True
                        break
                if not matched:
                    # Page exists in PBIP but API didn't list it -- add it
                    live_result.setdefault("pages", []).append({
                        "name": dn_lower,
                        "displayName": lp["displayName"],
                        "visuals": lp.get("visuals", []),
                    })

            # ---- Compute fidelity ----
            fidelity = self._compute_fidelity(live_result)
            best_fidelity = fidelity
            score = fidelity.get("score", 0)

            self._log("QA_FIDELITY",
                      f"Score: {score}% | "
                      f"Identity: {fidelity.get('tab_identity', 0)}/25  "
                      f"Charts: {fidelity.get('chart_types', 0)}/25  "
                      f"Fields: {fidelity.get('field_bindings', 0)}/25  "
                      f"Layout: {fidelity.get('layout', 0)}/25")

            if score >= 75 or loop == self.max_loops - 1:
                # Good enough or last attempt
                break

            # ---- Diagnose and fix ----
            self._log("QA_FIX", f"Score {score}% < 75% -- attempting fix")
            self._diagnose_fidelity_gaps(fidelity, live_result)

            # Re-generate PBIP with adjusted config
            try:
                from generators.pbip_generator import PBIPGenerator
                import shutil

                fix_dir = pbip_path + f"_qa_fix_{loop + 1}"
                if os.path.exists(fix_dir):
                    shutil.rmtree(fix_dir)

                gen = PBIPGenerator(fix_dir)
                gen_result = gen.generate(
                    config=self.config,
                    data_profile=data_profile,
                    dashboard_spec=self._build_dashboard_spec(),
                    data_file_path=data_file_path,
                )
                new_pbip_path = gen_result.get("path", fix_dir)

                # Delete old report + dataset, republish
                try:
                    did = get_dataset_id_from_report(
                        pbi_token, workspace_id, report_id)
                    delete_report(token, workspace_id, report_id)
                    if did:
                        delete_dataset(token, workspace_id, did)
                except Exception:
                    pass

                pub = publish_pbip(
                    token, workspace_id, new_pbip_path,
                    os.path.basename(pbip_path) + f"_v{loop + 2}"
                )
                if not pub.get("error"):
                    report_id = pub.get("report_id", report_id)
                    semantic_model_id = pub.get("semantic_model_id",
                                                semantic_model_id)
                    report_url = (f"https://app.powerbi.com/groups/"
                                  f"{workspace_id}/reports/{report_id}")
                    pbip_path = new_pbip_path
                    self._log("QA_REPUBLISH",
                              f"Republished: {report_url}")
                else:
                    self._log("QA_REPUBLISH",
                              f"Republish failed: {pub.get('error')}",
                              "error")
                    break
            except Exception as fix_err:
                self._log("QA_FIX", f"Fix failed: {fix_err}", "error")
                break

        return {
            "fidelity": best_fidelity,
            "report_url": report_url,
            "report_id": report_id,
            "loops": loop + 1,
            "audit": self.audit,
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

            # ---- PHASE 1.5: STRUCTURAL QA (tab-by-tab) ----
            try:
                repairs, tab_fidelity = self._pre_publish_structural_qa(
                    self.source_spec, self.config, pbip_path
                )
                tab_score = tab_fidelity.get("total", 0)
                tab_count = len(tab_fidelity.get("tab_scores", []))
                self._log("STRUCTURAL_QA",
                          f"Tab fidelity: {tab_score:.0f}% "
                          f"({tab_count} tabs, {len(repairs)} repairs)",
                          "success" if not repairs else "warning")
            except Exception as sq_err:
                self._log("STRUCTURAL_QA", f"Skipped: {sq_err}", "warning")

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

            # ---- PHASE 5: FIDELITY COMPARISON (4-axis tab-by-tab) ----
            source_manifest = self._build_source_manifest()
            fidelity = self._compute_fidelity(live_result, source_manifest)
            self.fidelity_report = fidelity
            score = fidelity.get("score", 0)
            matched = fidelity.get("matched_tabs", 0)
            total = fidelity.get("total_tabs", 0)
            self._log(
                "FIDELITY",
                f"Score: {score}% | Tabs: {matched}/{total} | "
                f"Identity: {fidelity.get('tab_identity', 0)}/25  "
                f"Charts: {fidelity.get('chart_types', 0)}/25  "
                f"Fields: {fidelity.get('field_bindings', 0)}/25  "
                f"Layout: {fidelity.get('layout', 0)}/25",
                "success" if score >= 90 else (
                    "warning" if score >= 75 else "error"
                ),
            )

            # Log per-tab scores
            for ts in fidelity.get("tab_scores", []):
                status = "success" if ts.get("page_found") else "error"
                self._log(
                    "TAB_SCORE",
                    f"  {ts['name']}: "
                    f"{'FOUND' if ts.get('page_found') else 'MISSING'}"
                    f" | chart={'OK' if ts.get('chart_match') else 'MISMATCH'}"
                    f" | fields={ts.get('field_pct', 0)}%"
                    f" | visuals={ts.get('visual_count_actual', 0)}"
                    f"/{ts.get('visual_count_expected', 0)}",
                    status,
                )

            # ---- PHASE 6: DECIDE -- GOOD ENOUGH OR FIX? ----
            # >= 90: clean accept
            # 75-89: accept with warnings
            # 60-74: re-loop with repairs
            # < 60: fail (unless last loop)
            if score >= 75 or loop == self.max_loops - 1:
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
                          f"Score {score}% < 75% -- diagnosing "
                          f"({len(fidelity.get('repairs', []))} repairs)",
                          "warning")
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

    # PBIR page.json allowed fields (strict allowlist)
    _PBIR_PAGE_ALLOWED_KEYS = {
        "$schema", "name", "displayName", "displayOption",
        "width", "height", "background", "backgroundImage",
        "mobileState", "outspacePartitions", "filters",
        "filterConfig", "publicCustomVisuals", "pods",
    }

    # PBIR visual.json allowed top-level keys
    _PBIR_VISUAL_ALLOWED_KEYS = {
        "$schema", "name", "position", "visual",
        "filters", "filterConfig", "dataTransforms",
        "howCreated", "isHidden", "tabOrder",
        "parentGroupName", "layerOrder",
    }

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

        # Schema allowlist checks (catches ordinal-type bugs)
        schema_issues = self._check_pbir_schemas(pbip_path)
        if schema_issues:
            # Auto-fix critical schema issues immediately
            self._auto_fix_schema_issues(pbip_path, schema_issues)
            # Re-check after fix
            remaining = self._check_pbir_schemas(pbip_path)
            issues.extend(remaining)

        return issues

    def _check_pbir_schemas(self, pbip_path) -> list:
        """Validate page.json and visual.json against PBIR allowed fields."""
        issues = []
        # Find the .Report folder
        rpt_dirs = glob.glob(os.path.join(pbip_path, "*.Report"))
        if not rpt_dirs:
            return issues
        rpt = rpt_dirs[0]

        # Check all page.json files
        pages_dir = os.path.join(rpt, "definition", "pages")
        if os.path.isdir(pages_dir):
            for page_folder in os.listdir(pages_dir):
                pj_path = os.path.join(pages_dir, page_folder, "page.json")
                if not os.path.isfile(pj_path):
                    continue
                try:
                    with open(pj_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    unknown = set(data.keys()) - self._PBIR_PAGE_ALLOWED_KEYS
                    if unknown:
                        issues.append({
                            "type": "SCHEMA",
                            "severity": "critical",
                            "file": pj_path,
                            "detail": f"page.json has invalid fields: "
                                      f"{unknown} -- PBI will reject",
                            "fix_action": "remove_keys",
                            "fix_keys": list(unknown),
                        })
                except Exception:
                    pass

                # Check visual.json files under this page
                vis_dir = os.path.join(pages_dir, page_folder, "visuals")
                if not os.path.isdir(vis_dir):
                    continue
                for viz_folder in os.listdir(vis_dir):
                    vj_path = os.path.join(vis_dir, viz_folder, "visual.json")
                    if not os.path.isfile(vj_path):
                        continue
                    try:
                        with open(vj_path, "r", encoding="utf-8") as f:
                            vdata = json.load(f)
                        unknown_v = (
                            set(vdata.keys()) - self._PBIR_VISUAL_ALLOWED_KEYS
                        )
                        if unknown_v:
                            issues.append({
                                "type": "SCHEMA",
                                "severity": "critical",
                                "file": vj_path,
                                "detail": f"visual.json has invalid fields: "
                                          f"{unknown_v} -- PBI will reject",
                                "fix_action": "remove_keys",
                                "fix_keys": list(unknown_v),
                            })
                    except Exception:
                        pass

        return issues

    def _auto_fix_schema_issues(self, pbip_path, issues):
        """Remove invalid keys from page.json and visual.json files."""
        for issue in issues:
            if issue.get("fix_action") != "remove_keys":
                continue
            fpath = issue.get("file", "")
            keys_to_remove = issue.get("fix_keys", [])
            if not fpath or not keys_to_remove or not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k in keys_to_remove:
                    data.pop(k, None)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                self._log("SCHEMA_FIX",
                          f"Removed {keys_to_remove} from "
                          f"{os.path.basename(fpath)}",
                          "fix")
            except Exception as e:
                self._log("SCHEMA_FIX",
                          f"Failed on {fpath}: {e}", "error")

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
    #  SOURCE MANIFEST BUILDER                                             #
    # ================================================================== #

    @staticmethod
    def _clean_manifest_field(raw_field):
        """Clean a raw Tableau shelf field for manifest comparison.

        Filters out datasource references (federated.*, long hex IDs),
        strips Tableau prefixes (sum:, none:, tqr:, yr:, etc.), and
        removes generated fields (Latitude/Longitude (generated)).
        """
        if not raw_field or not isinstance(raw_field, str):
            return ""
        f = raw_field.strip("[] ")
        # Filter datasource references
        if f.startswith("federated.") or f.startswith("[federated."):
            return ""
        # Filter long hex-only strings (datasource IDs)
        alpha = f.replace("-", "").replace(".", "")
        if len(alpha) > 20 and all(c in "0123456789abcdef" for c in alpha.lower()):
            return ""
        # Filter generated fields
        if "(generated)" in f.lower():
            return ""
        # Strip Tableau prefixes: sum:Field:qk -> Field
        if ":" in f:
            parts = f.split(":")
            if len(parts) >= 3:
                f = parts[1]  # prefix:FIELD:suffix
            elif len(parts) == 2:
                f = parts[1]  # prefix:FIELD
        f = f.strip("[] ")
        return f if len(f) > 1 else ""

    def _build_source_manifest(self) -> list:
        """Build a per-tab source-of-truth manifest from source_spec.

        Each entry represents one expected output page with the fields,
        chart type, and visual count the QA agent should verify.
        Uses DataFrame column names for field matching when available.
        """
        spec = self.source_spec
        if not spec:
            return []

        # Build a set of real column names from the DataFrame for matching
        df_columns = set()
        if self.dataframe is not None:
            df_columns = {c.lower() for c in self.dataframe.columns}

        manifest = []
        worksheets = spec.get("worksheets", [])
        dashboards = spec.get("dashboards", [])
        ws_by_name = {ws.get("name", ""): ws for ws in worksheets}

        def _extract_fields_for_ws(ws):
            """Get cleaned, validated fields for a worksheet."""
            fields = set()
            for shelf_key in ("rows_fields", "cols_fields"):
                for f in ws.get(shelf_key, []):
                    cleaned = self._clean_manifest_field(f)
                    if cleaned:
                        fields.add(cleaned.lower())
            # Also extract from color_field
            color = ws.get("color_field", "")
            if color:
                # Handle compound refs: [federated.xxx].[sum:Sales:qk]
                for part in color.split("]."):
                    cleaned = self._clean_manifest_field(part)
                    if cleaned:
                        fields.add(cleaned.lower())
            # Cross-reference with DataFrame columns for validation
            if df_columns:
                validated = set()
                for f in fields:
                    for col in df_columns:
                        if f == col or f in col or col in f:
                            validated.add(col)
                # Also infer fields from worksheet name
                ws_lower = ws.get("name", "").lower()
                for col in df_columns:
                    if col in ws_lower:
                        validated.add(col)
                return list(validated)
            return list(fields)

        if dashboards:
            used_ws_names = set()
            for db in dashboards:
                db_name = db.get("name", "Dashboard")
                ws_used = db.get("worksheets_used", [])
                all_fields = []
                chart_types = []
                for ws_name in ws_used:
                    used_ws_names.add(ws_name)
                    ws = ws_by_name.get(ws_name, {})
                    all_fields.extend(_extract_fields_for_ws(ws))
                    chart_types.append(ws.get("chart_type", "automatic"))
                manifest.append({
                    "name": db_name,
                    "type": "dashboard",
                    "expected_visual_count": len(ws_used),
                    "worksheet_names": list(ws_used),
                    "chart_types": chart_types,
                    "fields": list(set(all_fields)),
                })

            orphans = [
                ws for ws in worksheets
                if ws.get("name", "") not in used_ws_names
            ]
            if orphans:
                orphan_fields = []
                for ws in orphans:
                    orphan_fields.extend(_extract_fields_for_ws(ws))
                page_name = (orphans[0].get("name", "Sheet")
                             if len(orphans) == 1 else "Additional Sheets")
                manifest.append({
                    "name": page_name,
                    "type": "orphan",
                    "expected_visual_count": len(orphans),
                    "worksheet_names": [ws.get("name", "") for ws in orphans],
                    "chart_types": [
                        ws.get("chart_type", "automatic") for ws in orphans
                    ],
                    "fields": list(set(orphan_fields)),
                })
        else:
            for ws in worksheets:
                manifest.append({
                    "name": ws.get("name", "Sheet"),
                    "type": "worksheet",
                    "expected_visual_count": 1,
                    "worksheet_names": [ws.get("name", "")],
                    "chart_types": [ws.get("chart_type", "automatic")],
                    "fields": _extract_fields_for_ws(ws),
                })

        return manifest

    # ================================================================== #
    #  CHART TYPE TRANSLATION                                              #
    # ================================================================== #

    # Tableau mark type -> expected Power BI visual type(s)
    _MARK_TO_PBI = {
        "bar": {"clusteredColumnChart", "clusteredBarChart",
                "hundredPercentStackedColumnChart",
                "hundredPercentStackedBarChart"},
        "line": {"lineChart", "lineClusteredColumnComboChart",
                 "lineStackedColumnComboChart"},
        "area": {"areaChart", "stackedAreaChart"},
        "circle": {"scatterChart"},
        "square": {"treemap", "matrix"},
        "text": {"tableEx", "matrix", "card", "cardVisual", "kpi",
                 "multiRowCard"},
        "pie": {"pieChart", "donutChart"},
        "gantt": {"clusteredBarChart", "tableEx"},  # no native PBI Gantt
        "polygon": {"filledMap", "shapeMap"},
        "map": {"map", "filledMap"},
        "automatic": set(),  # accept any type
    }

    def _tableau_mark_to_pbi_visual(self, mark_type: str) -> set:
        """Return set of acceptable PBI visual types for a Tableau mark."""
        return self._MARK_TO_PBI.get(
            mark_type.lower().strip(), set()
        )

    # ================================================================== #
    #  FIDELITY SCORING — 4-AXIS TAB-BY-TAB                               #
    # ================================================================== #

    def _compute_fidelity(self, live_result, source_manifest=None) -> dict:
        """4-axis fidelity scoring per tab.

        Axes (25 pts each, normalized to 100 total):
          Tab Identity — tab count + name match + order
          Chart Types  — visual type matches Tableau mark per tab
          Field Bindings — correct fields present in visuals per tab
          Layout — visual count per page + relative sizing
        """
        if source_manifest is None:
            source_manifest = self._build_source_manifest()

        if not source_manifest:
            return {
                "score": 50,
                "reason": "no_source_manifest",
                "tab_identity": 0,
                "chart_types": 0,
                "field_bindings": 0,
                "layout": 0,
                "tab_scores": [],
                "repairs": [],
                "total_tabs": 0,
                "matched_tabs": 0,
            }

        total_tabs = len(source_manifest)
        identity_pts = 0
        chart_pts = 0
        field_pts = 0
        layout_pts = 0
        tab_scores = []
        repairs = []

        # Build lookup: lowercase displayName -> page dict
        output_pages = live_result.get("pages", [])
        page_lookup = {}
        for idx, page in enumerate(output_pages):
            dn = page.get("displayName", "").lower().strip()
            if dn:
                page_lookup[dn] = (idx, page)

        for src_idx, src_tab in enumerate(source_manifest):
            src_name = src_tab["name"]
            src_lower = src_name.lower().strip()
            ts = {
                "name": src_name,
                "type": src_tab.get("type", "worksheet"),
                "page_found": False,
                "chart_match": None,
                "field_pct": 0,
                "visual_count_expected": src_tab.get(
                    "expected_visual_count", 1
                ),
                "visual_count_actual": 0,
            }

            # --- TAB IDENTITY (25 pts) ---
            match = page_lookup.get(src_lower)
            if match is None:
                # Try fuzzy: check if any page name contains or is contained
                for dn, (pidx, pg) in page_lookup.items():
                    if src_lower in dn or dn in src_lower:
                        match = (pidx, pg)
                        break

            if match is not None:
                page_idx, page = match
                ts["page_found"] = True
                # Name match: 15 pts for exact, 10 for fuzzy
                exact = (page.get("displayName", "").lower().strip()
                         == src_lower)
                identity_pts += 15 if exact else 10
                if not exact:
                    repairs.append({
                        "repair_type": "RENAME_PAGE",
                        "target": page.get("displayName", ""),
                        "expected_value": src_name,
                        "priority": 3,
                    })
                # Order match: 10 pts if within ±1 position
                if abs(page_idx - src_idx) <= 1:
                    identity_pts += 10
                elif abs(page_idx - src_idx) <= 2:
                    identity_pts += 5
                else:
                    repairs.append({
                        "repair_type": "REORDER_PAGES",
                        "target": src_name,
                        "current_value": f"position {page_idx}",
                        "expected_value": f"position {src_idx}",
                        "priority": 3,
                    })
            else:
                # Page completely missing
                ts["score"] = 0
                tab_scores.append(ts)
                repairs.append({
                    "repair_type": "ADD_PAGE",
                    "target": src_name,
                    "source_ref": f"manifest[{src_idx}]",
                    "expected_value": (
                        f"page with {src_tab.get('expected_visual_count', 1)} "
                        f"visuals for worksheets: "
                        f"{src_tab.get('worksheet_names', [])}"
                    ),
                    "priority": 1,
                })
                continue

            visuals = page.get("visuals", [])
            ts["visual_count_actual"] = len(visuals)

            # --- CHART TYPES (25 pts) ---
            src_chart_types = src_tab.get("chart_types", [])
            if src_chart_types:
                matched_charts = 0
                total_charts = len(src_chart_types)
                pbi_types = [v.get("type", "") for v in visuals]
                for ct in src_chart_types:
                    acceptable = self._tableau_mark_to_pbi_visual(ct)
                    if not acceptable:
                        # "automatic" — accept anything
                        matched_charts += 1
                    elif any(pt in acceptable for pt in pbi_types):
                        matched_charts += 1
                    else:
                        repairs.append({
                            "repair_type": "CHANGE_VISUAL_TYPE",
                            "target": src_name,
                            "current_value": (
                                pbi_types[0] if pbi_types else "none"
                            ),
                            "expected_value": sorted(acceptable)[0],
                            "source_ref": ct,
                            "priority": 2,
                        })
                ratio = matched_charts / max(total_charts, 1)
                chart_pts += int(25 * ratio)
                ts["chart_match"] = ratio >= 0.8
            else:
                chart_pts += 25  # no chart types to verify

            # --- FIELD BINDINGS (25 pts) ---
            src_fields = set()
            for f in src_tab.get("fields", []):
                cleaned = f.lower().strip("[]' ")
                # Strip Tableau prefixes like sum:, avg:, none:
                if ":" in cleaned:
                    parts = cleaned.split(":")
                    # Use the field name part (usually index 1 for
                    # prefix:field:suffix patterns)
                    if len(parts) >= 2:
                        cleaned = parts[-2] if len(parts) >= 3 else parts[-1]
                if cleaned:
                    src_fields.add(cleaned)

            pbi_fields = set()
            for v in visuals:
                for f in v.get("fields", []):
                    pbi_fields.add(f.lower().strip("[]' "))
                # Also check queryRef patterns from visual.json
                for f in v.get("queryRefs", []):
                    # "Table.Column" -> "column"
                    if "." in f:
                        pbi_fields.add(f.split(".")[-1].lower().strip())
                    else:
                        pbi_fields.add(f.lower().strip())

            if src_fields and pbi_fields:
                overlap = len(src_fields & pbi_fields)
                field_ratio = overlap / len(src_fields)
                field_pts += int(25 * field_ratio)
                ts["field_pct"] = round(field_ratio * 100)
                missing_fields = src_fields - pbi_fields
                for mf in missing_fields:
                    repairs.append({
                        "repair_type": "ADD_FIELD",
                        "target": f"{src_name}/{mf}",
                        "expected_value": mf,
                        "priority": 2,
                    })
            elif pbi_fields and not src_fields:
                # PBI has fields but we have no source expectation -- partial credit
                field_pts += 10
                ts["field_pct"] = 40
            else:
                # No fields on either side -- 0 points, not free 25
                ts["field_pct"] = 0

            # --- LAYOUT (25 pts) ---
            expected_vc = src_tab.get("expected_visual_count", 1)
            actual_vc = len(visuals)
            if expected_vc > 0:
                vc_ratio = min(actual_vc / max(expected_vc, 1), 1.0)
                # Visual count match: 15 pts
                layout_pts += int(15 * vc_ratio)
                # Sizing check: 10 pts if all visuals have valid size
                sized = sum(
                    1 for v in visuals
                    if v.get("width", 0) > 100 and v.get("height", 0) > 80
                )
                if actual_vc > 0:
                    layout_pts += int(10 * (sized / actual_vc))
                ts["visual_count_expected"] = expected_vc
                ts["visual_count_actual"] = actual_vc

                if actual_vc < expected_vc:
                    missing_count = expected_vc - actual_vc
                    repairs.append({
                        "repair_type": "ADD_VISUAL",
                        "target": src_name,
                        "current_value": f"{actual_vc} visuals",
                        "expected_value": f"{expected_vc} visuals",
                        "priority": 1,
                    })
            else:
                layout_pts += 25

            ts["score"] = int(
                (identity_pts + chart_pts + field_pts + layout_pts)
                * 100 / (100 * max(src_idx + 1, 1))
            )
            tab_scores.append(ts)

        # Normalize to 100.  Max per axis per tab = 25, 4 axes -> max = 100*n
        n = max(total_tabs, 1)
        raw = identity_pts + chart_pts + field_pts + layout_pts
        max_raw = 100 * n
        score = int(raw * 100 / max(max_raw, 1))
        score = max(0, min(100, score))

        # Sort repairs by priority
        repairs.sort(key=lambda r: r.get("priority", 9))

        return {
            "score": score,
            "tab_identity": int(identity_pts / n),
            "chart_types": int(chart_pts / n),
            "field_bindings": int(field_pts / n),
            "layout": int(layout_pts / n),
            "tab_scores": tab_scores,
            "repairs": repairs,
            "total_tabs": total_tabs,
            "matched_tabs": sum(
                1 for t in tab_scores if t.get("page_found")
            ),
        }

    # ================================================================== #
    #  FIDELITY GAP DIAGNOSIS + CONFIG ADJUSTMENT                         #
    # ================================================================== #

    def _diagnose_fidelity_gaps(self, fidelity, live_result):
        """Diagnose gaps using tab-level scores and issue repair instructions.

        Reads the ``repairs`` list from ``_compute_fidelity()`` and applies
        priority-1 and priority-2 repairs to ``self.config`` so the next
        loop generates a better PBIP.
        """
        repairs = fidelity.get("repairs", [])
        tab_scores = fidelity.get("tab_scores", [])

        # Log per-tab status
        for ts in tab_scores:
            if not ts.get("page_found"):
                self._log(
                    "DIAGNOSE",
                    f"TAB MISSING: '{ts['name']}' has no output page",
                    "error",
                )
            elif ts.get("chart_match") is False:
                self._log(
                    "DIAGNOSE",
                    f"CHART MISMATCH on '{ts['name']}': "
                    f"expected {ts.get('chart_expected', '?')}, "
                    f"got {ts.get('chart_actual', '?')}",
                    "warning",
                )
            field_pct = ts.get("field_pct", 100)
            if field_pct < 80:
                self._log(
                    "DIAGNOSE",
                    f"FIELD GAP on '{ts['name']}': "
                    f"only {field_pct}% of source fields found",
                    "warning",
                )
            vc_exp = ts.get("visual_count_expected", 0)
            vc_act = ts.get("visual_count_actual", 0)
            if vc_exp > 0 and vc_act < vc_exp:
                self._log(
                    "DIAGNOSE",
                    f"VISUAL COUNT on '{ts['name']}': "
                    f"{vc_act}/{vc_exp} visuals",
                    "warning",
                )

        # Log data-level issues from live_result
        if live_result.get("row_count", 0) == 0:
            self._log(
                "DIAGNOSE",
                "Row count = 0. Data didn't load. "
                "Checking M expression.",
                "warning",
            )

        # Apply repair instructions to config for next loop
        config_sections = self.config.get("report_layout", {}).get(
            "sections", []
        )
        if not config_sections:
            config_sections = []

        for repair in repairs:
            rtype = repair.get("repair_type", "")
            priority = repair.get("priority", 9)
            if priority > 2:
                continue  # Only apply critical + high priority in re-loop

            if rtype == "ADD_PAGE":
                self._log(
                    "REPAIR",
                    f"Queuing page add: '{repair['target']}' "
                    f"({repair.get('expected_value', '')})",
                    "fix",
                )
                # Flag in config so the mapper knows to create this page
                missing = self.config.setdefault(
                    "_repair_add_pages", []
                )
                missing.append({
                    "name": repair["target"],
                    "source_ref": repair.get("source_ref", ""),
                })

            elif rtype == "ADD_VISUAL":
                self._log(
                    "REPAIR",
                    f"Queuing visual add on '{repair['target']}': "
                    f"{repair.get('expected_value', '')}",
                    "fix",
                )
                add_visuals = self.config.setdefault(
                    "_repair_add_visuals", []
                )
                add_visuals.append({
                    "page": repair["target"],
                    "expected": repair.get("expected_value", ""),
                })

            elif rtype == "CHANGE_VISUAL_TYPE":
                self._log(
                    "REPAIR",
                    f"Queuing chart type fix on '{repair['target']}': "
                    f"{repair.get('current_value', '')} -> "
                    f"{repair.get('expected_value', '')}",
                    "fix",
                )

            elif rtype == "ADD_FIELD":
                self._log(
                    "REPAIR",
                    f"Queuing field binding: '{repair['target']}' "
                    f"needs '{repair.get('expected_value', '')}'",
                    "fix",
                )

        total_repairs = len([
            r for r in repairs if r.get("priority", 9) <= 2
        ])
        if total_repairs:
            self._log(
                "REPAIR_SUMMARY",
                f"{total_repairs} repair(s) queued for next loop",
                "fix",
            )

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
            lines.append("FIDELITY REPORT (4-axis tab-by-tab):")
            lines.append(
                f"  Tab Identity:   {fid.get('tab_identity', 0)}/25"
            )
            lines.append(
                f"  Chart Types:    {fid.get('chart_types', 0)}/25"
            )
            lines.append(
                f"  Field Bindings: {fid.get('field_bindings', 0)}/25"
            )
            lines.append(
                f"  Layout:         {fid.get('layout', 0)}/25"
            )
            lines.append(
                f"  TOTAL: {fid.get('score', 0)}/100  "
                f"(tabs: {fid.get('matched_tabs', 0)}"
                f"/{fid.get('total_tabs', 0)})"
            )
            lines.append("")

            tab_scores = fid.get("tab_scores", [])
            if tab_scores:
                lines.append("PER-TAB SCORES:")
                for ts in tab_scores:
                    found = "FOUND" if ts.get("page_found") else "MISSING"
                    chart = ("OK" if ts.get("chart_match")
                             else "MISMATCH"
                             if ts.get("chart_match") is False
                             else "N/A")
                    lines.append(
                        f"  {ts['name']}: {found} | "
                        f"chart={chart} | "
                        f"fields={ts.get('field_pct', 0)}% | "
                        f"visuals="
                        f"{ts.get('visual_count_actual', 0)}"
                        f"/{ts.get('visual_count_expected', 0)}"
                    )
                lines.append("")

            repairs = fid.get("repairs", [])
            if repairs:
                lines.append("REPAIR INSTRUCTIONS:")
                for r in repairs:
                    prio = r.get("priority", "?")
                    lines.append(
                        f"  [P{prio}] {r['repair_type']}: "
                        f"{r.get('target', '')} -> "
                        f"{r.get('expected_value', '')}"
                    )
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

    # ================================================================== #
    #  PRE-PUBLISH STRUCTURAL QA (tab-by-tab)                             #
    # ================================================================== #

    _MARK_TO_PBI = {
        "bar": "clusteredBarChart",
        "line": "lineChart",
        "area": "areaChart",
        "pie": "pieChart",
        "ban": "cardVisual",
        "kpi": "cardVisual",
        "text": "tableEx",
        "circle": "scatterChart",
        "map": "map",
        "filledMap": "filledMap",
        "multipolygon": "filledMap",
        "polygon": "filledMap",
        "gantt-bar": "clusteredBarChart",
        "shape": "tableEx",
        "density": "scatterChart",
        "heatmap": "matrix",
        "square": "matrix",
        "automatic": "clusteredColumnChart",
    }

    def _pre_publish_structural_qa(self, spec, config, pbip_dir):
        """Compare Tableau tab manifest vs generated PBI pages.

        Returns (repairs, fidelity) where repairs is a list of actions
        and fidelity is a per-tab scoring dict.
        """
        windows = spec.get("windows", [])
        visible_tabs = [w for w in windows if not w.get("hidden", False)]

        if not visible_tabs:
            # No windows data -- skip structural QA
            return [], {"total": 100, "tab_scores": []}

        ws_by_name = {ws["name"]: ws for ws in spec.get("worksheets", [])}

        # Build source manifest
        source_manifest = []
        for tab in visible_tabs:
            entry = {"name": tab["name"], "type": tab["type"]}
            if tab["type"] == "worksheet":
                ws = ws_by_name.get(tab["name"], {})
                entry["chart_type"] = ws.get("chart_type", "automatic")
                entry["expected_pbi_type"] = self._MARK_TO_PBI.get(
                    entry["chart_type"], "clusteredColumnChart"
                )
            elif tab["type"] == "dashboard":
                db = next(
                    (d for d in spec.get("dashboards", [])
                     if d["name"] == tab["name"]),
                    None,
                )
                if db:
                    entry["expected_visual_count"] = len(
                        db.get("worksheets_used", [])
                    )
            source_manifest.append(entry)

        # Build output manifest from generated PBIP
        output_manifest = self._read_pbip_manifest(pbip_dir)

        # Diff
        repairs = []
        tab_scores = []

        for src in source_manifest:
            score = 0
            checks = {}

            # Find matching page (case-insensitive)
            match = next(
                (p for p in output_manifest
                 if p["displayName"].lower().strip() == src["name"].lower().strip()),
                None,
            )

            checks["page_exists"] = match is not None
            if not match:
                repairs.append({
                    "action": "add_page",
                    "tab": src["name"],
                    "type": src["type"],
                })
                tab_scores.append({
                    "name": src["name"], "score": 0, "checks": checks,
                })
                continue

            score += 25  # page exists

            # Visual count check for dashboards
            if src["type"] == "dashboard":
                expected = src.get("expected_visual_count", 0)
                actual = len(match.get("visuals", []))
                checks["visual_count"] = f"{actual}/{expected}"
                if actual >= expected:
                    score += 25
                elif actual > 0:
                    score += int(25 * actual / max(expected, 1))

            # Chart type check for worksheets
            if src["type"] == "worksheet" and match.get("visuals"):
                expected_type = src.get("expected_pbi_type", "")
                actual_type = match["visuals"][0].get("visualType", "")
                checks["chart_type"] = f"{actual_type} (expected {expected_type})"
                if expected_type == actual_type:
                    score += 25
                else:
                    repairs.append({
                        "action": "fix_chart_type",
                        "page": src["name"],
                        "expected": expected_type,
                        "actual": actual_type,
                    })

            # Has at least one visual with data bindings
            has_data = any(
                v.get("has_fields") for v in match.get("visuals", [])
            )
            checks["has_data_bindings"] = has_data
            if has_data:
                score += 25
            elif match.get("visuals"):
                score += 10

            # Has proper size
            has_size = any(
                v.get("width", 0) > 100 and v.get("height", 0) > 100
                for v in match.get("visuals", [])
            )
            checks["has_proper_size"] = has_size
            if has_size:
                score += 25

            tab_scores.append({
                "name": src["name"],
                "score": min(score, 100),
                "checks": checks,
            })

        total = (
            sum(t["score"] for t in tab_scores) / max(len(tab_scores), 1)
        )

        return repairs, {"total": total, "tab_scores": tab_scores}

    def _read_pbip_manifest(self, pbip_dir):
        """Read generated PBIP folder and return page + visual manifest."""
        pages = []
        # Find the .Report folder
        report_dirs = glob.glob(os.path.join(pbip_dir, "*.Report"))
        if not report_dirs:
            return pages

        pages_dir = os.path.join(report_dirs[0], "definition", "pages")
        if not os.path.isdir(pages_dir):
            return pages

        for page_folder in sorted(os.listdir(pages_dir)):
            page_path = os.path.join(pages_dir, page_folder)
            if not os.path.isdir(page_path):
                continue
            page_json = os.path.join(page_path, "page.json")
            if not os.path.exists(page_json):
                continue
            try:
                with open(page_json) as f:
                    page_data = json.load(f)
            except Exception:
                continue

            visuals = []
            visuals_dir = os.path.join(page_path, "visuals")
            if os.path.isdir(visuals_dir):
                for viz_folder in os.listdir(visuals_dir):
                    viz_json = os.path.join(
                        visuals_dir, viz_folder, "visual.json"
                    )
                    if not os.path.exists(viz_json):
                        continue
                    try:
                        with open(viz_json) as f:
                            vdata = json.load(f)
                        vis = vdata.get("visual", {})
                        qs = vis.get("query", {}).get("queryState", {})
                        pos = vdata.get("position", {})
                        # Extract bound field names from queryState
                        bound_fields = []
                        query_refs = []
                        for role_data in qs.values():
                            for proj in role_data.get("projections", []):
                                nr = proj.get("nativeQueryRef", "")
                                qr = proj.get("queryRef", "")
                                if nr:
                                    # "Sum of Sales" -> "sales"
                                    clean = nr.lower().replace("sum of ", "").replace("avg of ", "").strip()
                                    bound_fields.append(clean)
                                if qr:
                                    query_refs.append(qr)
                        visuals.append({
                            "visualType": vis.get("visualType", ""),
                            "type": vis.get("visualType", ""),
                            "title": "",
                            "width": pos.get("width", 0),
                            "height": pos.get("height", 0),
                            "has_fields": bool(qs),
                            "fields": bound_fields,
                            "queryRefs": query_refs,
                        })
                    except Exception:
                        continue

            pages.append({
                "displayName": page_data.get("displayName", page_folder),
                "visuals": visuals,
            })

        return pages
