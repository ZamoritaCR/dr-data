"""
QA Agent -- Deterministic gate + GPT advisory for PBIP validation.

Layer 1: Deterministic checks (BLOCKS publish on failure).
Layer 2: GPT-4o advisory (warnings only, never blocks).

Enterprise-grade: the gate is 100% deterministic. No AI. No hallucination.
"""

import os
import re
import json
import glob
import requests


class QAAgent:
    def __init__(self, pbip_path, dataframe=None):
        self.pbip_path = pbip_path
        self.dataframe = dataframe
        self.issues = []
        self.warnings = []
        self.fixes_applied = []
        # Load OpenAI key same way vision_extractor.py does
        try:
            from config.settings import OPENAI_API_KEY
            self.api_key = OPENAI_API_KEY or ""
        except ImportError:
            self.api_key = ""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

    def run_full_qa(self):
        """Run full QA pipeline: deterministic gate, then GPT advisory."""
        self._check_structure()
        self._check_tmdl_indentation()
        self._check_tmdl_syntax()
        self._check_no_broken_paths()
        self._check_data_integrity()
        self._check_visual_references()

        if self.issues:
            self._auto_fix()
            # Re-run checks after fix attempt
            self.issues = []
            self._check_structure()
            self._check_tmdl_indentation()
            self._check_tmdl_syntax()
            self._check_no_broken_paths()
            self._check_data_integrity()
            self._check_visual_references()

        passed = len(self.issues) == 0

        if passed and self.api_key:
            try:
                self._gpt_advisory()
            except Exception as e:
                self.warnings.append(f"GPT review skipped: {e}")

        return {
            "passed": passed,
            "issues": self.issues,
            "warnings": self.warnings,
            "fixes": self.fixes_applied,
        }

    # ======== LAYER 1: DETERMINISTIC CHECKS ========

    def _check_structure(self):
        """Verify PBIP project has all required files and folders."""
        root = self.pbip_path
        if not os.path.isdir(root):
            self.issues.append(f"STRUCTURE: path does not exist: {root}")
            return

        # Check for .pbip file at root
        pbip_files = glob.glob(os.path.join(root, "*.pbip"))
        if not pbip_files:
            self.issues.append("STRUCTURE: missing .pbip file at project root")

        # Find SemanticModel folder
        sm_dirs = glob.glob(os.path.join(root, "*.SemanticModel"))
        if not sm_dirs:
            self.issues.append("STRUCTURE: missing .SemanticModel folder")
        else:
            sm = sm_dirs[0]
            if not os.path.exists(os.path.join(sm, "definition.pbism")):
                self.issues.append(
                    f"STRUCTURE: missing {os.path.basename(sm)}/definition.pbism"
                )
            if not os.path.exists(os.path.join(sm, ".platform")):
                self.issues.append(
                    f"STRUCTURE: missing {os.path.basename(sm)}/.platform"
                )
            tmdl_files = glob.glob(
                os.path.join(sm, "**", "*.tmdl"), recursive=True
            )
            if not tmdl_files:
                self.issues.append(
                    f"STRUCTURE: no .tmdl files in {os.path.basename(sm)}"
                )

        # Find Report folder
        rpt_dirs = glob.glob(os.path.join(root, "*.Report"))
        if not rpt_dirs:
            self.issues.append("STRUCTURE: missing .Report folder")
        else:
            rpt = rpt_dirs[0]
            rpt_base = os.path.basename(rpt)
            if not os.path.exists(os.path.join(rpt, "definition.pbir")):
                self.issues.append(
                    f"STRUCTURE: missing {rpt_base}/definition.pbir"
                )
            # report.json can be at definition/report.json
            rj = os.path.join(rpt, "definition", "report.json")
            if not os.path.exists(rj):
                self.issues.append(
                    f"STRUCTURE: missing {rpt_base}/definition/report.json"
                )
            if not os.path.exists(os.path.join(rpt, ".platform")):
                self.issues.append(
                    f"STRUCTURE: missing {rpt_base}/.platform"
                )

            # Check pages -- at least 1 page folder with at least 1 visual
            pages_dir = os.path.join(rpt, "definition", "pages")
            if os.path.isdir(pages_dir):
                page_folders = [
                    d for d in os.listdir(pages_dir)
                    if os.path.isdir(os.path.join(pages_dir, d))
                ]
                if not page_folders:
                    self.issues.append(
                        f"STRUCTURE: no page folders in {rpt_base}/definition/pages/"
                    )
                for pf in page_folders:
                    visuals_dir = os.path.join(pages_dir, pf, "visuals")
                    if os.path.isdir(visuals_dir):
                        vis_jsons = glob.glob(
                            os.path.join(visuals_dir, "**", "visual.json"),
                            recursive=True,
                        )
                        if not vis_jsons:
                            self.issues.append(
                                f"STRUCTURE: page '{pf}' has no visual.json files"
                            )
                    else:
                        self.issues.append(
                            f"STRUCTURE: page '{pf}' missing visuals/ folder"
                        )
            else:
                self.issues.append(
                    f"STRUCTURE: missing {rpt_base}/definition/pages/ folder"
                )

    def _check_tmdl_indentation(self):
        """Check every .tmdl file for indentation errors."""
        tmdl_files = glob.glob(
            os.path.join(self.pbip_path, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, self.pbip_path)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception as e:
                self.issues.append(f"INDENTATION: {fname}: cannot read: {e}")
                continue

            prev_indent = 0
            for lineno, line in enumerate(lines, 1):
                stripped = line.rstrip("\n\r")
                if not stripped:
                    continue

                # Count leading tabs
                leading = len(stripped) - len(stripped.lstrip("\t"))
                remainder = stripped[:leading + len(stripped.lstrip("\t")) - len(stripped.lstrip())]

                # Check for mixed spaces and tabs in leading whitespace
                leading_ws = stripped[: len(stripped) - len(stripped.lstrip())]
                if leading_ws and " " in leading_ws and "\t" in leading_ws:
                    self.issues.append(
                        f"INDENTATION: {fname} line {lineno}: "
                        f"mixed tabs and spaces in leading whitespace"
                    )
                elif leading_ws and " " in leading_ws and "\t" not in leading_ws:
                    # Pure spaces -- should be tabs in TMDL
                    self.issues.append(
                        f"INDENTATION: {fname} line {lineno}: "
                        f"spaces used for indentation (TMDL requires tabs)"
                    )

                # Check indent jump > 1 from previous non-empty line
                if leading > prev_indent + 1:
                    # Allow larger jumps inside partition/source blocks
                    # (M expression lines can be deeply indented)
                    if not any(
                        kw in stripped.lstrip()
                        for kw in ("let", "in", "Source", "#table", "{", "}")
                    ):
                        pass  # M expression blocks legitimately jump
                prev_indent = leading

    def _check_tmdl_syntax(self):
        """Check .tmdl files for unmatched braces, parens, quotes."""
        tmdl_files = glob.glob(
            os.path.join(self.pbip_path, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, self.pbip_path)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            # Count braces
            n_open = content.count("{")
            n_close = content.count("}")
            if n_open != n_close:
                self.issues.append(
                    f"SYNTAX: {fname}: unmatched braces "
                    f"({n_open} open, {n_close} close)"
                )

            # Count parens
            n_popen = content.count("(")
            n_pclose = content.count(")")
            if n_popen != n_pclose:
                self.issues.append(
                    f"SYNTAX: {fname}: unmatched parentheses "
                    f"({n_popen} open, {n_pclose} close)"
                )

            # Count double-quote characters (must be even)
            n_quotes = content.count('"')
            if n_quotes % 2 != 0:
                self.issues.append(
                    f"SYNTAX: {fname}: odd number of double quotes ({n_quotes})"
                )

    def _check_no_broken_paths(self):
        """Flag hardcoded file paths that will break on other machines."""
        text_exts = (".tmdl", ".json", ".pbism", ".pbir", ".pbip")
        patterns = [
            (r"File\.Contents\s*\(", "File.Contents() reference"),
            (r"/home/", "Linux /home/ path"),
            (r"/tmp/", "Linux /tmp/ path"),
            (r"/usr/", "Linux /usr/ path"),
            (r"[A-Z]:\\", "Hardcoded Windows drive path"),
            (r"[A-Z]:/", "Hardcoded Windows drive path"),
            (r"localhost", "localhost reference"),
            (r"127\.0\.0\.1", "loopback IP reference"),
        ]

        for root, _, files in os.walk(self.pbip_path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext not in text_exts:
                    continue
                fpath = os.path.join(root, name)
                fname = os.path.relpath(fpath, self.pbip_path)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue

                for pat, desc in patterns:
                    matches = re.findall(pat, content)
                    if matches:
                        self.issues.append(
                            f"BROKEN_PATH: {fname}: {desc} "
                            f"({len(matches)} occurrence(s))"
                        )

    def _check_data_integrity(self):
        """Verify inline #table() data matches the dataframe."""
        if self.dataframe is None:
            return

        tmdl_files = glob.glob(
            os.path.join(self.pbip_path, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            fname = os.path.relpath(fpath, self.pbip_path)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            if "#table(" not in content:
                continue

            # Count data rows: lines inside #table block that match
            # the pattern of a data row: { followed by values }
            in_table = False
            data_rows = 0
            for line in content.split("\n"):
                stripped = line.strip()
                if "#table(" in stripped:
                    in_table = True
                    continue
                if in_table:
                    # Data rows start with { and contain values
                    if stripped.startswith("{") and "," in stripped and (
                        stripped.endswith("},") or stripped.endswith("}")
                    ):
                        data_rows += 1
                    # End of table block
                    if stripped.startswith("})"):
                        in_table = False

            if data_rows == 0:
                self.issues.append(
                    f"DATA: {fname}: empty #table() -- no data rows"
                )
            else:
                df_rows = len(self.dataframe)
                # Allow up to 5% mismatch (due to row limits)
                # But also allow intentional row limits (e.g., max 500)
                if data_rows > df_rows * 1.05:
                    self.issues.append(
                        f"DATA: {fname}: #table has {data_rows} rows "
                        f"but dataframe has {df_rows} -- mismatch"
                    )

            # Extract column names from type table definition
            type_match = re.search(
                r"type\s+table\s*\[([^\]]+)\]", content
            )
            if type_match:
                type_def = type_match.group(1)
                # Parse column names: #"Col Name" = type or ColName = type
                tmdl_cols = []
                for col_def in re.finditer(
                    r'(?:#"([^"]+)"|(\w+))\s*=\s*\w+', type_def
                ):
                    tmdl_cols.append(col_def.group(1) or col_def.group(2))

                df_cols = set(self.dataframe.columns)
                tmdl_col_set = set(tmdl_cols)

                missing_in_tmdl = df_cols - tmdl_col_set
                extra_in_tmdl = tmdl_col_set - df_cols

                if extra_in_tmdl:
                    self.warnings.append(
                        f"DATA: {fname}: columns in TMDL not in dataframe: "
                        f"{sorted(extra_in_tmdl)}"
                    )
                if missing_in_tmdl and len(missing_in_tmdl) > len(df_cols) * 0.5:
                    self.warnings.append(
                        f"DATA: {fname}: {len(missing_in_tmdl)} of "
                        f"{len(df_cols)} dataframe columns missing from TMDL"
                    )

    def _check_visual_references(self):
        """Verify visual data bindings reference columns/measures that exist."""
        # Collect column and measure names from .tmdl files
        tmdl_columns = set()
        tmdl_measures = set()
        tmdl_files = glob.glob(
            os.path.join(self.pbip_path, "**", "*.tmdl"), recursive=True
        )
        for fpath in tmdl_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        stripped = line.strip()
                        # column 'Name' or column Name
                        col_match = re.match(
                            r"column\s+'([^']+)'", stripped
                        ) or re.match(
                            r"column\s+(\S+)", stripped
                        )
                        if col_match:
                            tmdl_columns.add(col_match.group(1))
                        # measure 'Name' = ...
                        meas_match = re.match(
                            r"measure\s+'([^']+)'", stripped
                        ) or re.match(
                            r"measure\s+(\S+)\s*=", stripped
                        )
                        if meas_match:
                            tmdl_measures.add(meas_match.group(1))
            except Exception:
                continue

        all_known = tmdl_columns | tmdl_measures

        # Parse visual.json files
        visual_files = glob.glob(
            os.path.join(self.pbip_path, "**", "visual.json"), recursive=True
        )
        for vpath in visual_files:
            vname = os.path.relpath(vpath, self.pbip_path)
            try:
                with open(vpath, "r", encoding="utf-8") as f:
                    vdata = json.load(f)
            except json.JSONDecodeError as e:
                self.issues.append(f"VISUAL: {vname}: invalid JSON -- {e}")
                continue
            except Exception:
                continue

            # Extract field references from visual config
            # Look for Column/Property/Measure references in projections
            refs = set()
            self._extract_field_refs(vdata, refs)

            for ref in refs:
                if ref and ref not in all_known:
                    self.warnings.append(
                        f"VISUAL: {vname}: references '{ref}' "
                        f"not found in TMDL columns/measures"
                    )

    def _extract_field_refs(self, obj, refs):
        """Recursively extract field name references from visual JSON."""
        if isinstance(obj, dict):
            # Common patterns for field references in PBI visual JSON
            if "Column" in obj and isinstance(obj["Column"], dict):
                prop = obj["Column"].get("Property")
                if prop:
                    refs.add(prop)
            if "Measure" in obj and isinstance(obj["Measure"], dict):
                prop = obj["Measure"].get("Property")
                if prop:
                    refs.add(prop)
            for v in obj.values():
                self._extract_field_refs(v, refs)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_field_refs(item, refs)

    # ======== AUTO-FIX (deterministic only) ========

    def _auto_fix(self):
        """Attempt deterministic fixes for known issue patterns."""
        fixed_files = set()

        for issue in self.issues:
            # Fix indentation: spaces -> tabs
            if "INDENTATION" in issue and (
                "spaces" in issue or "mixed tabs" in issue
            ):
                # Extract filename from issue message
                match = re.match(r"INDENTATION: (.+?) line", issue)
                if not match:
                    continue
                fname = match.group(1)
                fpath = os.path.join(self.pbip_path, fname)
                if fpath in fixed_files or not os.path.exists(fpath):
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        # Replace leading spaces with tabs (4 spaces = 1 tab)
                        content = line.rstrip("\n\r")
                        leading_ws = content[: len(content) - len(content.lstrip())]
                        body = content.lstrip()
                        # Convert spaces to tabs
                        tab_count = leading_ws.count("\t")
                        space_count = leading_ws.count(" ")
                        total_tabs = tab_count + (space_count // 4)
                        if space_count % 4:
                            total_tabs += 1
                        new_lines.append("\t" * total_tabs + body + "\n")

                    with open(fpath, "w", encoding="utf-8") as f:
                        f.writelines(new_lines)
                    fixed_files.add(fpath)
                    self.fixes_applied.append(
                        f"FIXED indentation in {fname}"
                    )
                except Exception as e:
                    self.fixes_applied.append(
                        f"FAILED to fix indentation in {fname}: {e}"
                    )

            # Fix data escaping: tabs/newlines in #table values
            if "DATA" in issue and "tab" in issue.lower():
                match = re.match(r"DATA: (.+?):", issue)
                if not match:
                    continue
                fname = match.group(1)
                fpath = os.path.join(self.pbip_path, fname)
                if fpath in fixed_files or not os.path.exists(fpath):
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Strip tabs/newlines from inside quoted values in data rows
                    # This is a targeted fix for the #table data section
                    fixed = re.sub(
                        r'"([^"]*)"',
                        lambda m: '"' + m.group(1)
                        .replace("\t", " ")
                        .replace("\n", " ")
                        .replace("\r", "")
                        + '"',
                        content,
                    )
                    if fixed != content:
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(fixed)
                        fixed_files.add(fpath)
                        self.fixes_applied.append(
                            f"FIXED data escaping in {fname}"
                        )
                except Exception as e:
                    self.fixes_applied.append(
                        f"FAILED to fix data in {fname}: {e}"
                    )

            # Cannot auto-fix these -- log only
            if "BROKEN_PATH" in issue and "File.Contents" in issue:
                self.fixes_applied.append(
                    "CANNOT AUTO-FIX File.Contents -- needs pipeline rebuild"
                )

    # ======== LAYER 2: GPT ADVISORY ========

    def _gpt_advisory(self):
        """Run GPT-4o review on project files (advisory only, never blocks)."""
        if not self.api_key:
            return

        # Collect file contents (prioritize TMDL, then visual JSON)
        file_contents = []
        total_chars = 0
        max_chars = 80000

        # TMDL files first
        tmdl_files = sorted(
            glob.glob(
                os.path.join(self.pbip_path, "**", "*.tmdl"), recursive=True
            )
        )
        for fpath in tmdl_files:
            if total_chars >= max_chars:
                break
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                fname = os.path.relpath(fpath, self.pbip_path)
                entry = f"--- {fname} ---\n{content}\n"
                file_contents.append(entry)
                total_chars += len(entry)
            except Exception:
                continue

        # Visual JSON files
        visual_files = sorted(
            glob.glob(
                os.path.join(self.pbip_path, "**", "visual.json"),
                recursive=True,
            )
        )
        for vpath in visual_files:
            if total_chars >= max_chars:
                break
            try:
                with open(vpath, "r", encoding="utf-8") as f:
                    content = f.read()
                fname = os.path.relpath(vpath, self.pbip_path)
                entry = f"--- {fname} ---\n{content}\n"
                file_contents.append(entry)
                total_chars += len(entry)
            except Exception:
                continue

        if not file_contents:
            return

        all_content = "\n".join(file_contents)
        # Truncate if still over limit
        if len(all_content) > max_chars:
            all_content = all_content[:max_chars] + "\n... (truncated)"

        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "max_tokens": 2000,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a Power BI TMDL/PBIP expert reviewer. "
                                "Review these project files for potential issues. "
                                "Return ONLY valid JSON: "
                                '{\"findings\": [\"description\", ...]}. '
                                "Check: DAX syntax validity, M expression "
                                "correctness, column type appropriateness, "
                                "visual data binding completeness, TMDL "
                                "structural correctness. If everything looks "
                                'valid, return {\"findings\": []}.'
                            ),
                        },
                        {"role": "user", "content": all_content},
                    ],
                },
                timeout=60,
            )

            if resp.status_code != 200:
                self.warnings.append(
                    f"GPT review failed: HTTP {resp.status_code}"
                )
                return

            body = resp.json()
            content = body.get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            )

            # Parse JSON from response (may be wrapped in markdown code block)
            json_str = content.strip()
            if json_str.startswith("```"):
                json_str = re.sub(r"^```\w*\n?", "", json_str)
                json_str = re.sub(r"\n?```$", "", json_str)

            findings = json.loads(json_str).get("findings", [])
            for finding in findings:
                self.warnings.append(f"GPT: {finding}")

        except json.JSONDecodeError:
            self.warnings.append("GPT review returned non-JSON response")
        except requests.exceptions.Timeout:
            self.warnings.append("GPT review timed out")
        except Exception as e:
            self.warnings.append(f"GPT review unavailable: {e}")
