"""
Business Rules Engine -- define validation rules without code,
evaluate them against any DataFrame.
"""

import pandas as pd
import numpy as np
import re
import json
import os
from datetime import datetime, timezone


class BusinessRulesEngine:
    """Custom business rules engine for data validation."""

    def __init__(self, rules_path="business_rules.json"):
        self.rules_path = rules_path
        self.rules = self._load_rules()
        self.evaluation_results = {}

    # ── Persistence ──

    def _load_rules(self):
        try:
            if os.path.exists(self.rules_path):
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[RULES ENGINE] Loaded {len(data.get('rules', []))} rules "
                      f"from {self.rules_path}")
                return data
        except Exception as e:
            print(f"[RULES ENGINE] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rules": [],
            "rule_sets": {},
        }

    def _save_rules(self):
        try:
            self.rules["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.rules_path, "w", encoding="utf-8") as f:
                json.dump(self.rules, f, indent=2, default=str)
        except Exception as e:
            print(f"[RULES ENGINE] Save failed: {e}")

    # ── Rule CRUD ──

    def _next_id(self):
        existing = [r["id"] for r in self.rules["rules"] if r.get("id", "").startswith("BR-")]
        nums = []
        for rid in existing:
            try:
                nums.append(int(rid.split("-")[1]))
            except (IndexError, ValueError):
                pass
        nxt = max(nums, default=0) + 1
        return f"BR-{nxt:04d}"

    _VALID_TYPES = {
        "not_null", "unique", "range", "regex", "in_list", "not_in_list",
        "custom_expression", "cross_column", "referential", "length", "type_check",
    }
    _VALID_SEVERITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def add_rule(self, rule):
        try:
            if not isinstance(rule, dict):
                print("[RULES ENGINE] Rule must be a dict")
                return None
            if not rule.get("name"):
                print("[RULES ENGINE] Rule name is required")
                return None
            rtype = rule.get("rule_type")
            if rtype not in self._VALID_TYPES:
                print(f"[RULES ENGINE] Invalid rule_type: {rtype}")
                return None
            sev = rule.get("severity", "MEDIUM").upper()
            if sev not in self._VALID_SEVERITIES:
                sev = "MEDIUM"

            now = datetime.now(timezone.utc).isoformat()
            rid = self._next_id()
            entry = {
                "id": rid,
                "name": rule["name"],
                "description": rule.get("description", ""),
                "table_pattern": rule.get("table_pattern", "*"),
                "column": rule.get("column"),
                "rule_type": rtype,
                "parameters": rule.get("parameters", {}),
                "severity": sev,
                "category": rule.get("category", "Validity"),
                "regulatory_tag": rule.get("regulatory_tag"),
                "enabled": rule.get("enabled", True),
                "created_at": now,
                "created_by": rule.get("created_by", "system"),
                "last_evaluated": None,
                "last_result": None,
            }
            self.rules["rules"].append(entry)
            self._save_rules()
            print(f"[RULES ENGINE] Added rule {rid}: {entry['name']}")
            return rid
        except Exception as e:
            print(f"[RULES ENGINE] add_rule failed: {e}")
            return None

    def remove_rule(self, rule_id):
        try:
            before = len(self.rules["rules"])
            self.rules["rules"] = [r for r in self.rules["rules"] if r["id"] != rule_id]
            if len(self.rules["rules"]) < before:
                self._save_rules()
                print(f"[RULES ENGINE] Removed rule {rule_id}")
                return True
            print(f"[RULES ENGINE] Rule {rule_id} not found")
            return False
        except Exception as e:
            print(f"[RULES ENGINE] remove_rule failed: {e}")
            return False

    def enable_rule(self, rule_id):
        try:
            for r in self.rules["rules"]:
                if r["id"] == rule_id:
                    r["enabled"] = True
                    self._save_rules()
                    print(f"[RULES ENGINE] Enabled rule {rule_id}")
                    return True
            print(f"[RULES ENGINE] Rule {rule_id} not found")
            return False
        except Exception as e:
            print(f"[RULES ENGINE] enable_rule failed: {e}")
            return False

    def disable_rule(self, rule_id):
        try:
            for r in self.rules["rules"]:
                if r["id"] == rule_id:
                    r["enabled"] = False
                    self._save_rules()
                    print(f"[RULES ENGINE] Disabled rule {rule_id}")
                    return True
            print(f"[RULES ENGINE] Rule {rule_id} not found")
            return False
        except Exception as e:
            print(f"[RULES ENGINE] disable_rule failed: {e}")
            return False

    # ── Single Rule Evaluation ──

    def evaluate_rule(self, rule, df, reference_tables=None):
        try:
            col_name = rule.get("column")
            rtype = rule["rule_type"]
            params = rule.get("parameters", {})

            # Column existence check (for column-level rules)
            if col_name and col_name not in df.columns:
                return {
                    "rule_id": rule["id"], "rule_name": rule["name"],
                    "passed": None, "error": f"Column '{col_name}' not in DataFrame",
                }

            failing = pd.DataFrame()

            if rtype == "not_null":
                failing = df[df[col_name].isna()]

            elif rtype == "unique":
                failing = df[df.duplicated(subset=[col_name], keep=False)]

            elif rtype == "range":
                col = pd.to_numeric(df[col_name], errors="coerce")
                mask = pd.Series(True, index=df.index)
                inclusive = params.get("inclusive", True)
                if params.get("min") is not None:
                    if inclusive:
                        mask &= col >= params["min"]
                    else:
                        mask &= col > params["min"]
                if params.get("max") is not None:
                    if inclusive:
                        mask &= col <= params["max"]
                    else:
                        mask &= col < params["max"]
                failing = df[~mask & col.notna()]

            elif rtype == "regex":
                pattern = params.get("pattern", "")
                match_type = params.get("match_type", "full")
                col = df[col_name].astype(str)
                if match_type == "full":
                    m = col.str.match(pattern, na=False)
                else:
                    m = col.str.contains(pattern, na=False, regex=True)
                failing = df[~m & df[col_name].notna()]

            elif rtype == "in_list":
                valid_values = params.get("values", [])
                failing = df[~df[col_name].isin(valid_values) & df[col_name].notna()]

            elif rtype == "not_in_list":
                invalid_values = params.get("values", [])
                failing = df[df[col_name].isin(invalid_values)]

            elif rtype == "custom_expression":
                expr = params.get("expression", "True")
                try:
                    mask = df.eval(expr, engine='numexpr')
                    failing = df[~mask]
                except Exception as ex:
                    return {
                        "rule_id": rule["id"], "rule_name": rule["name"],
                        "passed": None,
                        "error": f"Expression eval failed: {ex}",
                    }

            elif rtype == "cross_column":
                col_b_name = params.get("column_b")
                if col_b_name not in df.columns:
                    return {
                        "rule_id": rule["id"], "rule_name": rule["name"],
                        "passed": None,
                        "error": f"Column '{col_b_name}' not in DataFrame",
                    }
                op = params.get("operator", "==")
                col_a = df[col_name]
                col_b = df[col_b_name]
                ops = {
                    ">": col_a > col_b, ">=": col_a >= col_b,
                    "<": col_a < col_b, "<=": col_a <= col_b,
                    "==": col_a == col_b, "!=": col_a != col_b,
                }
                mask = ops.get(op, col_a == col_b)
                failing = df[~mask & col_a.notna() & col_b.notna()]

            elif rtype == "referential":
                ref_table = params.get("reference_table")
                ref_col = params.get("reference_column")
                if not reference_tables or ref_table not in reference_tables:
                    return {
                        "rule_id": rule["id"], "rule_name": rule["name"],
                        "passed": None,
                        "error": f"Reference table '{ref_table}' not loaded",
                    }
                ref_values = set(reference_tables[ref_table][ref_col].dropna())
                failing = df[~df[col_name].isin(ref_values) & df[col_name].notna()]

            elif rtype == "length":
                col = df[col_name].astype(str)
                mask = pd.Series(True, index=df.index)
                if params.get("min_length") is not None:
                    mask &= col.str.len() >= params["min_length"]
                if params.get("max_length") is not None:
                    mask &= col.str.len() <= params["max_length"]
                failing = df[~mask & df[col_name].notna()]

            elif rtype == "type_check":
                expected = params.get("expected_type", "string")
                col = df[col_name]
                if expected == "numeric":
                    mask = pd.to_numeric(col, errors="coerce").notna() | col.isna()
                elif expected == "date":
                    mask = pd.to_datetime(col, errors="coerce").notna() | col.isna()
                elif expected == "string":
                    mask = col.apply(lambda x: isinstance(x, str) or pd.isna(x))
                elif expected == "boolean":
                    mask = col.isin(
                        [True, False, 0, 1, "true", "false", "True", "False", None]
                    ) | col.isna()
                else:
                    mask = pd.Series(True, index=df.index)
                failing = df[~mask]

            total = len(df)
            fail_count = len(failing)
            pass_count = total - fail_count
            pass_rate = (pass_count / total * 100) if total > 0 else 100.0

            # Collect up to 10 failing examples
            examples = []
            for idx in failing.head(10).index:
                row_data = {}
                if col_name and col_name in df.columns:
                    row_data[col_name] = str(failing.loc[idx, col_name])
                examples.append({"row_index": int(idx), "values": row_data})

            return {
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "passed": fail_count == 0,
                "total_rows": total,
                "passing_rows": pass_count,
                "failing_rows": fail_count,
                "pass_rate": round(pass_rate, 2),
                "severity": rule["severity"],
                "failing_examples": examples,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            print(f"[RULES ENGINE] evaluate_rule failed for {rule.get('id')}: {e}")
            return {
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
                "passed": None,
                "error": str(e),
            }

    # ── Bulk Evaluation ──

    def evaluate_all(self, df, table_name, reference_tables=None):
        try:
            now = datetime.now(timezone.utc).isoformat()
            matching_rules = []
            for r in self.rules["rules"]:
                if not r.get("enabled", True):
                    continue
                pat = r.get("table_pattern", "*")
                if pat == "*" or re.search(pat, table_name, re.IGNORECASE):
                    matching_rules.append(r)

            results = []
            passed = 0
            failed = 0
            errors = 0
            by_category = {}
            by_severity = {}
            critical_failures = []

            for r in matching_rules:
                res = self.evaluate_rule(r, df, reference_tables)
                results.append(res)

                # Update rule metadata
                r["last_evaluated"] = now
                r["last_result"] = "PASS" if res.get("passed") else (
                    "FAIL" if res.get("passed") is False else "ERROR")

                if res.get("passed") is True:
                    passed += 1
                elif res.get("passed") is False:
                    failed += 1
                    if r["severity"] == "CRITICAL":
                        critical_failures.append(res)
                else:
                    errors += 1

                # Category summary
                cat = r.get("category", "Other")
                if cat not in by_category:
                    by_category[cat] = {"total": 0, "passed": 0, "failed": 0}
                by_category[cat]["total"] += 1
                if res.get("passed") is True:
                    by_category[cat]["passed"] += 1
                elif res.get("passed") is False:
                    by_category[cat]["failed"] += 1

                # Severity summary
                sev = r.get("severity", "MEDIUM")
                if sev not in by_severity:
                    by_severity[sev] = {"total": 0, "passed": 0, "failed": 0}
                by_severity[sev]["total"] += 1
                if res.get("passed") is True:
                    by_severity[sev]["passed"] += 1
                elif res.get("passed") is False:
                    by_severity[sev]["failed"] += 1

            total_rules = len(matching_rules)
            overall_rate = (passed / total_rules * 100) if total_rules > 0 else 100.0

            summary = {
                "table_name": table_name,
                "evaluated_at": now,
                "total_rules": total_rules,
                "passed": passed,
                "failed": failed,
                "error": errors,
                "overall_pass_rate": round(overall_rate, 2),
                "results": results,
                "critical_failures": critical_failures,
                "summary_by_category": by_category,
                "summary_by_severity": by_severity,
            }
            self.evaluation_results[table_name] = summary
            self._save_rules()
            print(f"[RULES ENGINE] Evaluated {total_rules} rules on '{table_name}': "
                  f"{passed} passed, {failed} failed, {errors} errors")
            return summary
        except Exception as e:
            print(f"[RULES ENGINE] evaluate_all failed: {e}")
            return {"table_name": table_name, "error": str(e)}

    # ── Rule Sets ──

    def add_rule_set(self, name, description, rule_ids):
        try:
            self.rules["rule_sets"][name] = {
                "description": description,
                "rule_ids": rule_ids,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_rules()
            print(f"[RULES ENGINE] Added rule set '{name}' with {len(rule_ids)} rules")
        except Exception as e:
            print(f"[RULES ENGINE] add_rule_set failed: {e}")

    # ── Query Helpers ──

    def get_rules_by_category(self, category):
        try:
            return [r for r in self.rules["rules"]
                    if r.get("category", "").lower() == category.lower()]
        except Exception as e:
            print(f"[RULES ENGINE] get_rules_by_category failed: {e}")
            return []

    def get_rules_by_regulatory_tag(self, tag):
        try:
            return [r for r in self.rules["rules"]
                    if (r.get("regulatory_tag") or "").lower() == tag.lower()]
        except Exception as e:
            print(f"[RULES ENGINE] get_rules_by_regulatory_tag failed: {e}")
            return []

    def get_evaluation_summary(self):
        try:
            if not self.evaluation_results:
                return {"message": "No evaluations run yet"}
            summary = {}
            for tname, ev in self.evaluation_results.items():
                summary[tname] = {
                    "total_rules": ev.get("total_rules", 0),
                    "passed": ev.get("passed", 0),
                    "failed": ev.get("failed", 0),
                    "overall_pass_rate": ev.get("overall_pass_rate", 0),
                    "critical_failures": len(ev.get("critical_failures", [])),
                    "evaluated_at": ev.get("evaluated_at"),
                }
            return summary
        except Exception as e:
            print(f"[RULES ENGINE] get_evaluation_summary failed: {e}")
            return {}

    # ── Export ──

    def export_rules(self, fmt="json"):
        try:
            if fmt == "json":
                return json.dumps(self.rules, indent=2, default=str)

            if fmt != "markdown":
                return json.dumps(self.rules, indent=2, default=str)

            lines = [
                "# Business Rules Documentation",
                f"**Version:** {self.rules.get('version', '1.0')}",
                f"**Updated:** {self.rules.get('updated_at', '')}",
                f"**Total Rules:** {len(self.rules.get('rules', []))}",
                "",
            ]

            by_cat = {}
            for r in self.rules.get("rules", []):
                cat = r.get("category", "Other")
                by_cat.setdefault(cat, []).append(r)

            for cat, cat_rules in sorted(by_cat.items()):
                lines.append(f"## {cat}")
                lines.append("")
                lines.append("| ID | Name | Type | Severity | Regulatory | Enabled |")
                lines.append("|-----|------|------|----------|------------|---------|")
                for r in cat_rules:
                    enabled = "Yes" if r.get("enabled") else "No"
                    reg = r.get("regulatory_tag") or "-"
                    lines.append(
                        f"| {r['id']} | {r['name']} | {r['rule_type']} "
                        f"| {r['severity']} | {reg} | {enabled} |"
                    )
                lines.append("")

            rule_sets = self.rules.get("rule_sets", {})
            if rule_sets:
                lines.append("## Rule Sets")
                lines.append("")
                for name, rs in rule_sets.items():
                    lines.append(f"### {name}")
                    lines.append(f"{rs.get('description', '')}")
                    lines.append(f"Rules: {', '.join(rs.get('rule_ids', []))}")
                    lines.append("")

            return "\n".join(lines)
        except Exception as e:
            print(f"[RULES ENGINE] export_rules failed: {e}")
            return "{}"

    # ── Preset Rules ──

    def _match_columns(self, df, hints):
        """Return list of column names containing any of the hint substrings."""
        matched = []
        for col in df.columns:
            low = col.lower()
            if any(h in low for h in hints):
                matched.append(col)
        return matched

    def load_preset_rules(self, preset_name, df=None, table_name=None):
        try:
            if preset_name == "financial_services":
                return self._load_financial_preset(df, table_name)
            elif preset_name == "tpch_snowflake":
                return self._load_tpch_preset(df, table_name)
            else:
                print(f"[RULES ENGINE] Unknown preset: {preset_name}")
                return []
        except Exception as e:
            print(f"[RULES ENGINE] load_preset_rules failed: {e}")
            return []

    def _load_financial_preset(self, df=None, table_name=None):
        templates = [
            {
                "name": "Transaction Amount Positive",
                "hints": ["amount", "price", "total"],
                "rule_type": "range", "parameters": {"min": 0},
                "severity": "CRITICAL", "category": "Business Logic",
            },
            {
                "name": "Transaction Amount Limit",
                "hints": ["amount"],
                "rule_type": "range", "parameters": {"max": 50000},
                "severity": "HIGH", "category": "Business Logic",
                "regulatory_tag": "BSA/AML",
            },
            {
                "name": "Customer Name Not Null",
                "hints": ["name"],
                "rule_type": "not_null", "parameters": {},
                "severity": "HIGH", "category": "Completeness",
                "regulatory_tag": "KYC",
            },
            {
                "name": "Country Code Valid Length",
                "hints": ["country", "nation"],
                "rule_type": "length", "parameters": {"min_length": 2, "max_length": 3},
                "severity": "MEDIUM", "category": "Validity",
                "regulatory_tag": "Compliance",
            },
            {
                "name": "Email Format",
                "hints": ["email"],
                "rule_type": "regex",
                "parameters": {
                    "pattern": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
                    "match_type": "full",
                },
                "severity": "LOW", "category": "Validity",
            },
            {
                "name": "Phone Format",
                "hints": ["phone"],
                "rule_type": "regex",
                "parameters": {
                    "pattern": r"^\+?[\d\s\-\(\)]{7,20}$",
                    "match_type": "full",
                },
                "severity": "LOW", "category": "Validity",
            },
            {
                "name": "No Negative Balance",
                "hints": ["balance", "bal"],
                "rule_type": "range", "parameters": {"min": 0},
                "severity": "HIGH", "category": "Accuracy",
            },
            {
                "name": "Status Values Valid",
                "hints": ["status"],
                "rule_type": "in_list",
                "parameters": {
                    "values": ["Active", "Inactive", "Pending", "Closed", "Suspended"],
                },
                "severity": "MEDIUM", "category": "Validity",
            },
        ]
        return self._apply_templates(templates, df, table_name)

    def _load_tpch_preset(self, df=None, table_name=None):
        templates = [
            {
                "name": "ORDERS Total Price Positive",
                "table_pattern": "(?i)orders",
                "hints": ["o_totalprice", "totalprice"],
                "rule_type": "range", "parameters": {"min": 0.01},
                "severity": "CRITICAL", "category": "Business Logic",
            },
            {
                "name": "ORDERS Status Valid",
                "table_pattern": "(?i)orders",
                "hints": ["o_orderstatus", "orderstatus"],
                "rule_type": "in_list",
                "parameters": {"values": ["F", "O", "P"]},
                "severity": "HIGH", "category": "Validity",
            },
            {
                "name": "LINEITEM Quantity Positive",
                "table_pattern": "(?i)lineitem",
                "hints": ["l_quantity", "quantity"],
                "rule_type": "range", "parameters": {"min": 1},
                "severity": "CRITICAL", "category": "Business Logic",
            },
            {
                "name": "LINEITEM Extended Price Positive",
                "table_pattern": "(?i)lineitem",
                "hints": ["l_extendedprice", "extendedprice"],
                "rule_type": "range", "parameters": {"min": 0},
                "severity": "CRITICAL", "category": "Business Logic",
            },
            {
                "name": "LINEITEM Discount Range",
                "table_pattern": "(?i)lineitem",
                "hints": ["l_discount", "discount"],
                "rule_type": "range", "parameters": {"min": 0, "max": 1},
                "severity": "MEDIUM", "category": "Validity",
            },
            {
                "name": "LINEITEM Tax Range",
                "table_pattern": "(?i)lineitem",
                "hints": ["l_tax"],
                "rule_type": "range", "parameters": {"min": 0, "max": 0.1},
                "severity": "MEDIUM", "category": "Validity",
            },
            {
                "name": "CUSTOMER Account Balance Not Null",
                "table_pattern": "(?i)customer",
                "hints": ["c_acctbal", "acctbal"],
                "rule_type": "not_null", "parameters": {},
                "severity": "HIGH", "category": "Completeness",
            },
            {
                "name": "CUSTOMER Nation Key Referential",
                "table_pattern": "(?i)customer",
                "hints": ["c_nationkey", "nationkey"],
                "rule_type": "referential",
                "parameters": {"reference_table": "NATION", "reference_column": "N_NATIONKEY"},
                "severity": "CRITICAL", "category": "Consistency",
            },
            {
                "name": "ORDERS Customer Key Referential",
                "table_pattern": "(?i)orders",
                "hints": ["o_custkey", "custkey"],
                "rule_type": "referential",
                "parameters": {"reference_table": "CUSTOMER", "reference_column": "C_CUSTKEY"},
                "severity": "CRITICAL", "category": "Consistency",
            },
            {
                "name": "SUPPLIER Nation Key Referential",
                "table_pattern": "(?i)supplier",
                "hints": ["s_nationkey", "nationkey"],
                "rule_type": "referential",
                "parameters": {"reference_table": "NATION", "reference_column": "N_NATIONKEY"},
                "severity": "HIGH", "category": "Consistency",
            },
        ]
        return self._apply_templates(templates, df, table_name)

    def _apply_templates(self, templates, df=None, table_name=None):
        """Match templates against actual DataFrame columns and add rules."""
        added = []
        for t in templates:
            hints = t.get("hints", [])
            tpat = t.get("table_pattern")

            # If table-specific pattern, check table name matches
            if tpat and tpat != "*" and table_name:
                if not re.search(tpat, table_name, re.IGNORECASE):
                    continue

            # Match columns if DataFrame provided
            if df is not None and hints:
                matched_cols = self._match_columns(df, hints)
                if not matched_cols:
                    continue
                for col in matched_cols:
                    rule = {
                        "name": f"{t['name']} ({col})",
                        "description": t.get("description", f"Preset rule for {col}"),
                        "table_pattern": tpat or (table_name if table_name else "*"),
                        "column": col,
                        "rule_type": t["rule_type"],
                        "parameters": t.get("parameters", {}),
                        "severity": t.get("severity", "MEDIUM"),
                        "category": t.get("category", "Validity"),
                        "regulatory_tag": t.get("regulatory_tag"),
                        "created_by": f"preset:{t.get('name', 'unknown')}",
                    }
                    rid = self.add_rule(rule)
                    if rid:
                        added.append(rid)
            elif df is None:
                # No df: add rule with column as hint pattern placeholder
                rule = {
                    "name": t["name"],
                    "description": t.get("description", "Preset rule"),
                    "table_pattern": tpat or "*",
                    "column": hints[0] if hints else None,
                    "rule_type": t["rule_type"],
                    "parameters": t.get("parameters", {}),
                    "severity": t.get("severity", "MEDIUM"),
                    "category": t.get("category", "Validity"),
                    "regulatory_tag": t.get("regulatory_tag"),
                    "created_by": f"preset:{t.get('name', 'unknown')}",
                }
                rid = self.add_rule(rule)
                if rid:
                    added.append(rid)

        print(f"[RULES ENGINE] Loaded {len(added)} preset rules")
        return added
