"""
PBIP Preflight Validator.

Walks a generated PBIP project directory and runs every known validation
check against the output files. Returns a structured report of passes
and failures that the healer can consume.

Checks cover all 8 error classes from production user reports:
  1. .pbip manifest (version, artifacts)
  2. definition.pbir (version, schema, byConnection)
  3. TMDL indentation (tabs not spaces)
  4. Int64.Type forbidden in M expressions
  5. model.tmdl must not contain table/partition blocks
  6. Table name consistency (no synthetic_tableau_data)
  7. No server paths (/home/, .twbx references)
  8. Inline data present when no external source
"""

import json
import os
import re
import logging

logger = logging.getLogger(__name__)


class PreflightResult:
    """Single check result."""
    __slots__ = ("name", "passed", "detail", "file_path")

    def __init__(self, name, passed, detail="", file_path=""):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.file_path = file_path

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"{status} {self.name}: {self.detail}"


class PreflightReport:
    """Collection of check results."""

    def __init__(self):
        self.results = []

    def add(self, name, passed, detail="", file_path=""):
        self.results.append(PreflightResult(name, passed, detail, file_path))

    @property
    def passed(self):
        return [r for r in self.results if r.passed]

    @property
    def failed(self):
        return [r for r in self.results if not r.passed]

    @property
    def all_passed(self):
        return len(self.failed) == 0

    @property
    def pass_count(self):
        return len(self.passed)

    @property
    def fail_count(self):
        return len(self.failed)

    def summary(self):
        return f"PASS={self.pass_count} FAIL={self.fail_count}"


def validate(project_path):
    """Run all preflight checks on a PBIP project directory.

    Args:
        project_path: path to the root of the generated PBIP project.

    Returns:
        PreflightReport with all check results.
    """
    report = PreflightReport()

    for root, dirs, files in os.walk(project_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                content = open(fpath, "rb").read().decode("utf-8", errors="replace")
            except Exception:
                continue

            if fname.endswith(".pbip"):
                _check_pbip(report, fname, fpath, content)

            if fname == "definition.pbir":
                _check_pbir(report, fname, fpath, content)

            if fname.endswith(".tmdl"):
                _check_tmdl(report, fname, fpath, content, root)

    if not report.results:
        report.add("project_structure", False, "No PBIP files found", project_path)

    logger.info(f"[PREFLIGHT] {report.summary()}")
    return report


def _check_pbip(report, fname, fpath, content):
    """ERROR CLASS 1: .pbip manifest checks."""
    try:
        d = json.loads(content)
    except json.JSONDecodeError as e:
        report.add("pbip_json", False, f"Invalid JSON: {e}", fpath)
        return

    # Version must be "1.0"
    v = d.get("version", "")
    report.add("pbip_version", v == "1.0", f"version={v}", fpath)

    # Artifacts must have exactly one entry with "report"
    arts = d.get("artifacts", [])
    ok = len(arts) == 1 and "report" in arts[0]
    report.add("pbip_artifacts", ok,
               f"artifacts={[list(a.keys()) for a in arts]}", fpath)

    # No forbidden keys (semanticModel, dataset)
    forbidden = any("semanticModel" in str(a) or "dataset" in str(a) for a in arts)
    report.add("pbip_no_forbidden_keys", not forbidden,
               "forbidden keys found" if forbidden else "clean", fpath)


def _check_pbir(report, fname, fpath, content):
    """ERROR CLASS 2: definition.pbir checks."""
    try:
        d = json.loads(content)
    except json.JSONDecodeError as e:
        report.add("pbir_json", False, f"Invalid JSON: {e}", fpath)
        return

    # Version must be "1.0"
    v = d.get("version", "")
    report.add("pbir_version", v == "1.0", f"version={v}", fpath)

    # $schema must be present
    report.add("pbir_schema", "$schema" in d,
               "present" if "$schema" in d else "MISSING", fpath)

    # byConnection must be present in datasetReference
    ds_ref = d.get("datasetReference", {})
    report.add("pbir_byconnection", "byConnection" in ds_ref,
               "present" if "byConnection" in ds_ref else "MISSING", fpath)


def _check_tmdl(report, fname, fpath, content, root):
    """ERROR CLASSES 3-8: TMDL checks."""

    # CLASS 4: No Int64.Type
    report.add(f"{fname}_no_int64", "Int64.Type" not in content,
               "FOUND" if "Int64.Type" in content else "clean", fpath)

    # CLASS 7: No server paths
    has_linux = "/home/" in content
    has_twbx = ".twbx" in content.lower()
    report.add(f"{fname}_no_server_path", not has_linux and not has_twbx,
               ("linux path" if has_linux else "") + (" .twbx ref" if has_twbx else "") or "clean",
               fpath)

    # Table-specific checks (files in tables/ directory)
    if "tables" in root:
        # CLASS 6: No old table name
        report.add(f"{fname}_no_old_name",
                   "synthetic_tableau_data" not in content,
                   "FOUND" if "synthetic_tableau_data" in content else "clean",
                   fpath)

        # CLASS 8: Inline data present
        report.add(f"{fname}_inline_data", "#table(" in content,
                   "present" if "#table(" in content else "MISSING", fpath)

    # CLASS 5: model.tmdl must not have table blocks
    if fname == "model.tmdl":
        has_block = bool(re.search(r"^table\s+", content, re.MULTILINE))
        report.add("model_tmdl_no_table_block", not has_block,
                   "table block found" if has_block else "clean", fpath)

    # CLASS 3: Tab indentation in partition blocks
    if "partition" in content:
        space_lines = sum(
            1 for line in content.split("\n")
            if line.startswith("    ") and not line.startswith("\t")
        )
        report.add(f"{fname}_tab_indent", space_lines == 0,
                   f"tabs OK" if space_lines == 0 else f"SPACES {space_lines} lines",
                   fpath)
