"""
DQ API Gateway -- programmatic interface for pipeline integration.
Wraps all DQ modules into simple callable functions for Airflow, dbt,
or custom ETL pipelines.
"""

import json
from datetime import datetime, timezone


class DQAPIGateway:
    """Unified API for data quality checks from external pipelines."""

    def __init__(self, dq_engine=None, rules_engine=None, catalog=None,
                 trust_scorer=None, compliance=None):
        self.dq_engine = dq_engine
        self.rules_engine = rules_engine
        self.catalog = catalog
        self.trust_scorer = trust_scorer
        self.compliance = compliance
        self.api_log = []

    def _log(self, endpoint, params, result_summary):
        self.api_log.append({
            "endpoint": endpoint,
            "params": params,
            "result_summary": result_summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _eng(self):
        if not self.dq_engine:
            from core.dq_engine import DataQualityEngine
            self.dq_engine = DataQualityEngine()
        return self.dq_engine

    def scan(self, df, table_name, return_format="json"):
        """Run a full DAMA-DMBOK quality scan on a DataFrame."""
        try:
            result = self._eng().scan_table(df, table_name)
            score = result.get("overall_score", 0)
            self._log("scan", {"table": table_name, "rows": len(df)}, {"score": score})
            if return_format == "dict":
                return result
            if return_format == "summary":
                st = "PASS" if score >= 90 else "WARN" if score >= 70 else "FAIL"
                return {"table": table_name, "score": score, "status": st}
            return json.dumps(result, default=str)
        except Exception as e:
            print(f"[DQ API] scan failed: {e}")
            return {"error": str(e)} if return_format == "dict" else json.dumps({"error": str(e)})

    def quality_gate(self, df, table_name, min_score=80,
                     required_dimensions=None, fail_on_critical=True):
        """Pass/fail quality gate for pipeline integration."""
        try:
            result = self._eng().scan_table(df, table_name)
            score = result.get("overall_score", 0) or 0
            dims = result.get("dimensions", {})
            blocking, passed = [], score >= min_score
            if required_dimensions:
                for dn in required_dimensions:
                    dd = dims.get(dn, {})
                    ds = (dd.get("score", 0) if isinstance(dd, dict) else 0) or 0
                    if ds < min_score:
                        passed = False
                        blocking.append(f"{dn}: {ds:.1f} < {min_score}")
            if fail_on_critical:
                for dn, dd in dims.items():
                    if not isinstance(dd, dict):
                        continue
                    ds = dd.get("score", 100) or 100
                    if ds < 50:
                        passed = False
                        blocking.append(f"{dn}: CRITICAL ({ds:.1f})")
            dim_scores = {dn: (dd.get("score", 0) or 0)
                          for dn, dd in dims.items() if isinstance(dd, dict)}
            reason = (f"Score {score:.1f} meets threshold {min_score}" if passed
                      else f"Failed: {'; '.join(blocking[:5])}" if blocking
                      else f"Score {score:.1f} below threshold {min_score}")
            gate = {"passed": passed, "score": round(score, 1),
                    "gate_threshold": min_score, "reason": reason,
                    "dimension_scores": dim_scores, "blocking_issues": blocking}
            self._log("quality_gate", {"table": table_name, "threshold": min_score},
                      {"passed": passed, "score": score})
            return gate
        except Exception as e:
            print(f"[DQ API] quality_gate failed: {e}")
            return {"passed": False, "score": 0,
                    "gate_threshold": min_score,
                    "reason": f"Error: {e}",
                    "dimension_scores": {}, "blocking_issues": [str(e)]}

    def evaluate_rules(self, df, table_name, rule_set=None,
                       reference_tables=None):
        """Run business rules against a DataFrame."""
        try:
            if self.rules_engine is None:
                return {"error": "No rules_engine configured"}
            result = self.rules_engine.evaluate_all(
                df, table_name, reference_tables=reference_tables)
            if rule_set and isinstance(result, dict):
                rules = result.get("results", [])
                filtered = [r for r in rules
                            if r.get("rule_set") == rule_set]
                result["results"] = filtered
                result["total_rules"] = len(filtered)
            self._log("evaluate_rules",
                      {"table": table_name, "rule_set": rule_set},
                      {"total": result.get("total_rules", 0),
                       "passed": result.get("passed", 0)})
            return result
        except Exception as e:
            print(f"[DQ API] evaluate_rules failed: {e}")
            return {"error": str(e)}

    def get_trust_score(self, table_name, df=None, scan_first=True):
        """Calculate composite trust score."""
        try:
            if self.trust_scorer is None:
                return {"error": "No trust_scorer configured"}
            dq_result = None
            if scan_first and df is not None:
                engine = self._eng()
                dq_result = engine.scan_table(df, table_name)
            result = self.trust_scorer.calculate_trust_score(
                table_name, dq_result=dq_result)
            self._log("get_trust_score", {"table": table_name},
                      {"score": result.get("trust_score", 0)})
            return result
        except Exception as e:
            print(f"[DQ API] get_trust_score failed: {e}")
            return {"error": str(e)}

    def check_compliance(self, table_name, df=None, frameworks=None):
        """Assess regulatory compliance."""
        try:
            if self.compliance is None:
                return {"error": "No compliance mapper configured"}
            if df is not None:
                self.compliance.auto_map_columns(df, table_name)
            engine = self._eng()
            dq_result = engine.scan_table(df, table_name) if df is not None else {}
            result = self.compliance.assess_compliance(
                table_name, dq_result)
            if frameworks and isinstance(result, dict):
                fw_data = result.get("frameworks", {})
                filtered = {k: v for k, v in fw_data.items()
                            if k in frameworks}
                result["frameworks"] = filtered
            self._log("check_compliance",
                      {"table": table_name, "frameworks": frameworks},
                      {"score": result.get("overall_compliance_score", 0)})
            return result
        except Exception as e:
            print(f"[DQ API] check_compliance failed: {e}")
            return {"error": str(e)}

    def validate_and_catalog(self, df, table_name, source_system="API"):
        """Full validation + cataloging in one call."""
        try:
            engine = self._eng()
            scan_result = engine.scan_table(df, table_name)
            catalog_entry = None
            if self.catalog:
                catalog_entry = self.catalog.auto_catalog_from_dataframe(
                    df, table_name, source_system=source_system)
                self.catalog.auto_generate_glossary(df, table_name)
            compliance_result = None
            if self.compliance:
                self.compliance.auto_map_columns(df, table_name)
                compliance_result = self.compliance.assess_compliance(
                    table_name, scan_result)
            trust_result = None
            if self.trust_scorer:
                trust_result = self.trust_scorer.calculate_trust_score(
                    table_name, dq_result=scan_result)
            combined = {
                "table_name": table_name, "source_system": source_system,
                "scan": scan_result, "catalog": catalog_entry,
                "compliance": compliance_result, "trust": trust_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._log("validate_and_catalog",
                      {"table": table_name, "source": source_system},
                      {"score": scan_result.get("overall_score", 0)})
            return combined
        except Exception as e:
            print(f"[DQ API] validate_and_catalog failed: {e}")
            return {"error": str(e)}

    def get_api_spec(self):
        """Return self-documentation of all API endpoints."""
        def _ep(desc, params, ret):
            return {"description": desc, "params": params, "returns": ret}
        try:
            return {"name": "Dr. Data Quality API", "version": "1.0", "endpoints": {
                "scan": _ep("Run full DAMA-DMBOK quality scan",
                    ["df", "table_name", "return_format"], "DQ scan results"),
                "quality_gate": _ep("Pass/fail quality gate for pipeline integration",
                    ["df", "table_name", "min_score", "required_dimensions", "fail_on_critical"],
                    "Gate result with pass/fail"),
                "evaluate_rules": _ep("Run business rules against data",
                    ["df", "table_name", "rule_set", "reference_tables"], "Rule evaluation results"),
                "get_trust_score": _ep("Calculate composite trust score",
                    ["table_name", "df", "scan_first"], "Trust score with factor breakdown"),
                "check_compliance": _ep("Assess regulatory compliance",
                    ["table_name", "df", "frameworks"], "Compliance assessment"),
                "validate_and_catalog": _ep("Full validation + cataloging in one call",
                    ["df", "table_name", "source_system"], "Combined scan + catalog + compliance"),
                "batch_scan": _ep("Scan multiple tables with quality gates",
                    ["tables_dict", "quality_gate_score"], "Per-table results with overall pass/fail"),
            }}
        except Exception as e:
            print(f"[DQ API] get_api_spec failed: {e}")
            return {}

    def get_api_log(self, limit=50):
        """Return recent API calls."""
        try:
            return list(reversed(self.api_log[-limit:]))
        except Exception as e:
            print(f"[DQ API] get_api_log failed: {e}")
            return []

    def batch_scan(self, tables_dict, quality_gate_score=80):
        """Scan multiple tables and run quality gates on all."""
        try:
            engine = self._eng()
            tables_out = {}
            passed_count, failed_count = 0, 0
            scores = []
            for tname, df in tables_dict.items():
                try:
                    result = engine.scan_table(df, tname)
                    sc = result.get("overall_score", 0) or 0
                    ok = sc >= quality_gate_score
                    issues = []
                    for dn, dd in result.get("dimensions", {}).items():
                        if isinstance(dd, dict):
                            ds = dd.get("score", 100) or 100
                            if ds < quality_gate_score:
                                issues.append(f"{dn}: {ds:.1f}")
                    tables_out[tname] = {
                        "score": round(sc, 1), "passed_gate": ok,
                        "issues": issues,
                    }
                    scores.append(sc)
                    if ok:
                        passed_count += 1
                    else:
                        failed_count += 1
                except Exception as te:
                    tables_out[tname] = {
                        "score": 0, "passed_gate": False,
                        "issues": [str(te)],
                    }
                    failed_count += 1
            total = passed_count + failed_count
            avg = round(sum(scores) / len(scores), 1) if scores else 0
            batch_result = {
                "tables": tables_out,
                "all_passed": failed_count == 0 and total > 0,
                "summary": {
                    "total": total, "passed": passed_count,
                    "failed": failed_count, "avg_score": avg,
                },
            }
            self._log("batch_scan",
                      {"table_count": total, "threshold": quality_gate_score},
                      {"passed": passed_count, "failed": failed_count,
                       "avg": avg})
            return batch_result
        except Exception as e:
            print(f"[DQ API] batch_scan failed: {e}")
            return {"tables": {}, "all_passed": False,
                    "summary": {"total": 0, "passed": 0,
                                "failed": 0, "avg_score": 0}}
