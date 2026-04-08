"""
PBIP Healer -- Surgical post-generation fixer.

Takes a PreflightReport with failures and applies deterministic fixes
to the generated PBIP files. Every fix is targeted at a specific error
class. No AI calls. No guessing. Just pattern-matched repairs.

The healer never modifies a file that passed all checks.
Returns a list of applied fixes for audit logging.
"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)


class HealerFix:
    """Record of a single fix applied."""
    __slots__ = ("check_name", "file_path", "description")

    def __init__(self, check_name, file_path, description):
        self.check_name = check_name
        self.file_path = file_path
        self.description = description

    def __repr__(self):
        fname = os.path.basename(self.file_path)
        return f"HEALED {self.check_name} in {fname}: {self.description}"


def heal(project_path, preflight_report):
    """Apply surgical fixes for all preflight failures.

    Args:
        project_path: root of the PBIP project directory.
        preflight_report: PreflightReport from preflight_validator.validate().

    Returns:
        list of HealerFix objects describing what was changed.
    """
    if preflight_report.all_passed:
        return []

    failed_names = {r.name for r in preflight_report.failed}
    fixes = []

    for root, dirs, files in os.walk(project_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                content = open(fpath, "rb").read().decode("utf-8", errors="replace")
            except Exception:
                continue

            new_content = content
            changed = False

            # --- .pbip fixes (ERROR CLASS 1) ---
            if fname.endswith(".pbip"):
                new_content, file_fixes = _heal_pbip(
                    new_content, fpath, failed_names
                )
                if file_fixes:
                    changed = True
                    fixes.extend(file_fixes)

            # --- definition.pbir fixes (ERROR CLASS 2) ---
            if fname == "definition.pbir":
                new_content, file_fixes = _heal_pbir(
                    new_content, fpath, failed_names
                )
                if file_fixes:
                    changed = True
                    fixes.extend(file_fixes)

            # --- TMDL fixes (ERROR CLASSES 3-8) ---
            if fname.endswith(".tmdl"):
                new_content, file_fixes = _heal_tmdl(
                    new_content, fname, fpath, root, failed_names
                )
                if file_fixes:
                    changed = True
                    fixes.extend(file_fixes)

            if changed:
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                except Exception as e:
                    logger.error(f"[HEALER] Failed to write {fpath}: {e}")

    if fixes:
        logger.info(f"[HEALER] Applied {len(fixes)} fixes")
    return fixes


def _heal_pbip(content, fpath, failed_names):
    """Fix .pbip manifest issues."""
    fixes = []
    try:
        d = json.loads(content)
    except json.JSONDecodeError:
        return content, fixes

    modified = False

    # Fix version
    if "pbip_version" in failed_names and d.get("version") != "1.0":
        d["version"] = "1.0"
        modified = True
        fixes.append(HealerFix("pbip_version", fpath, "version -> 1.0"))

    # Fix artifacts: keep only report entries
    if "pbip_artifacts" in failed_names or "pbip_no_forbidden_keys" in failed_names:
        arts = d.get("artifacts", [])
        cleaned = [{"report": a["report"]} for a in arts if "report" in a]
        if not cleaned:
            # No report entry at all -- reconstruct from directory
            parent = os.path.dirname(fpath)
            for item in os.listdir(parent):
                if item.endswith(".Report"):
                    cleaned = [{"report": {"path": item}}]
                    break
        if cleaned != arts:
            d["artifacts"] = cleaned
            modified = True
            fixes.append(HealerFix("pbip_artifacts", fpath,
                                   f"cleaned artifacts to {len(cleaned)} report-only entries"))

    # Ensure settings
    if "settings" not in d:
        d["settings"] = {"enableAutoRecovery": True}
        modified = True

    if modified:
        content = json.dumps(d, indent=2)

    return content, fixes


def _heal_pbir(content, fpath, failed_names):
    """Fix definition.pbir issues."""
    fixes = []
    try:
        d = json.loads(content)
    except json.JSONDecodeError:
        return content, fixes

    modified = False

    # Fix version
    if "pbir_version" in failed_names and d.get("version") != "1.0":
        d["version"] = "1.0"
        modified = True
        fixes.append(HealerFix("pbir_version", fpath, "version -> 1.0"))

    # Fix $schema
    if "pbir_schema" in failed_names and "$schema" not in d:
        d["$schema"] = (
            "https://developer.microsoft.com/json-schemas/fabric/"
            "item/report/definitionProperties/1.0.0/schema.json"
        )
        modified = True
        fixes.append(HealerFix("pbir_schema", fpath, "added $schema"))

    # Fix byConnection
    if "pbir_byconnection" in failed_names:
        ds_ref = d.get("datasetReference", {})
        if "byConnection" not in ds_ref:
            ds_ref["byConnection"] = None
            d["datasetReference"] = ds_ref
            modified = True
            fixes.append(HealerFix("pbir_byconnection", fpath,
                                   "added byConnection: null"))

    if modified:
        content = json.dumps(d, indent=2)

    return content, fixes


def _heal_tmdl(content, fname, fpath, root, failed_names):
    """Fix TMDL file issues."""
    fixes = []
    new_content = content

    # CLASS 4: Replace Int64.Type with type number
    check_name = f"{fname}_no_int64"
    if check_name in failed_names and "Int64.Type" in new_content:
        new_content = new_content.replace("Int64.Type", "type number")
        fixes.append(HealerFix(check_name, fpath,
                               "Int64.Type -> type number"))

    # CLASS 6: Replace old table name
    check_name = f"{fname}_no_old_name"
    if check_name in failed_names and "synthetic_tableau_data" in new_content:
        new_content = new_content.replace("synthetic_tableau_data", "Data")
        fixes.append(HealerFix(check_name, fpath,
                               "synthetic_tableau_data -> Data"))

    # CLASS 7: Remove server paths (replace with portable placeholder)
    check_name = f"{fname}_no_server_path"
    if check_name in failed_names:
        if "/home/" in new_content:
            # Replace Linux paths with Windows placeholder
            new_content = re.sub(
                r'/home/[^\s"]+/([^/\s"]+)',
                r'C:\\PBI_Data\\\1',
                new_content,
            )
            fixes.append(HealerFix(check_name, fpath,
                                   "Linux path -> C:\\PBI_Data\\"))
        if ".twbx" in new_content.lower():
            new_content = re.sub(
                r'(["\'])([^"\']*\.twbx)(["\'])',
                lambda m: m.group(1) + m.group(2).rsplit(".", 1)[0] + "_data.csv" + m.group(3),
                new_content,
                flags=re.IGNORECASE,
            )
            fixes.append(HealerFix(check_name, fpath,
                                   ".twbx reference -> _data.csv"))

    # CLASS 3: Fix space indentation in partition blocks
    check_name = f"{fname}_tab_indent"
    if check_name in failed_names and "partition" in new_content:
        lines = new_content.split("\n")
        fixed_lines = []
        for line in lines:
            if line.startswith("    ") and not line.startswith("\t"):
                # Convert leading groups of 4 spaces to tabs
                stripped = line.lstrip(" ")
                n_spaces = len(line) - len(stripped)
                n_tabs = n_spaces // 4
                remainder = " " * (n_spaces % 4)
                line = "\t" * n_tabs + remainder + stripped
            fixed_lines.append(line)
        new_content = "\n".join(fixed_lines)
        fixes.append(HealerFix(check_name, fpath,
                               "spaces -> tabs in partition block"))

    return new_content, fixes
