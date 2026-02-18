"""
Trust Scoring -- composite trust score from multiple quality signals,
plus certification management.
"""

from datetime import datetime, timezone


class TrustScorer:
    """Calculate trust scores and manage data certification."""

    def __init__(self):
        self.weights = {
            "dq_score": 0.30,
            "completeness": 0.15,
            "freshness": 0.15,
            "stewardship": 0.10,
            "documentation": 0.10,
            "rule_compliance": 0.10,
            "scan_frequency": 0.05,
            "lineage_coverage": 0.05,
        }
        self.certification_thresholds = {
            "Certified": 85,
            "Warning": 65,
            "Quarantined": 0,
        }

    def calculate_trust_score(self, table_name, dq_result=None,
                              catalog_entry=None, rule_results=None,
                              history_data=None):
        try:
            scores = {}

            # DQ Score (30%) + Completeness (15%) + Freshness (15%)
            if dq_result:
                scores["dq_score"] = dq_result.get("overall_score", 0) or 0
                dims = dq_result.get("dimensions", {})
                comp = dims.get("completeness", {})
                scores["completeness"] = (
                    comp.get("score", 0) if isinstance(comp, dict) else 0
                ) or 0
                time_d = dims.get("timeliness", {})
                scores["freshness"] = (
                    time_d.get("score", 50)
                    if isinstance(time_d, dict) and time_d.get("score") is not None
                    else 50
                )
            else:
                scores["dq_score"] = 0
                scores["completeness"] = 0
                scores["freshness"] = 50

            # Stewardship (10%)
            if catalog_entry:
                steward_score = 0
                if catalog_entry.get("owner", "Unassigned") != "Unassigned":
                    steward_score += 50
                if catalog_entry.get("steward", "Unassigned") != "Unassigned":
                    steward_score += 50
                scores["stewardship"] = steward_score
            else:
                scores["stewardship"] = 0

            # Documentation (10%)
            if catalog_entry:
                cols = catalog_entry.get("columns", {})
                if cols:
                    documented = sum(
                        1 for c in cols.values()
                        if c.get("description", "").strip()
                    )
                    scores["documentation"] = (documented / len(cols)) * 100
                else:
                    scores["documentation"] = 0
            else:
                scores["documentation"] = 0

            # Rule compliance (10%)
            if rule_results:
                scores["rule_compliance"] = (
                    rule_results.get("overall_pass_rate", 0) or 0)
            else:
                scores["rule_compliance"] = 50  # neutral if no rules

            # Scan frequency (5%)
            if history_data:
                total_scans = history_data.get("total_scans", 0)
                if total_scans >= 10:
                    scores["scan_frequency"] = 100
                elif total_scans >= 5:
                    scores["scan_frequency"] = 80
                elif total_scans >= 2:
                    scores["scan_frequency"] = 60
                elif total_scans >= 1:
                    scores["scan_frequency"] = 40
                else:
                    scores["scan_frequency"] = 0
            else:
                scores["scan_frequency"] = 0

            # Lineage coverage (5%) - placeholder
            scores["lineage_coverage"] = 50

            # Weighted trust score
            trust_score = sum(
                scores.get(k, 0) * w
                for k, w in self.weights.items()
            )

            # Certification recommendation
            recommended_cert = self.get_certification_recommendation(
                trust_score)

            return {
                "table_name": table_name,
                "trust_score": round(trust_score, 1),
                "recommended_certification": recommended_cert,
                "component_scores": scores,
                "weights": dict(self.weights),
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "factors": {
                    "dq_score": {
                        "score": scores.get("dq_score", 0),
                        "weight": "30%",
                        "description": "Latest DQ scan overall score",
                    },
                    "completeness": {
                        "score": scores.get("completeness", 0),
                        "weight": "15%",
                        "description": "Data completeness (non-null percentage)",
                    },
                    "freshness": {
                        "score": scores.get("freshness", 50),
                        "weight": "15%",
                        "description": "Data timeliness and freshness",
                    },
                    "stewardship": {
                        "score": scores.get("stewardship", 0),
                        "weight": "10%",
                        "description": "Has assigned owner and data steward",
                    },
                    "documentation": {
                        "score": scores.get("documentation", 0),
                        "weight": "10%",
                        "description": "Percentage of columns with descriptions",
                    },
                    "rule_compliance": {
                        "score": scores.get("rule_compliance", 50),
                        "weight": "10%",
                        "description": "Business rule pass rate",
                    },
                    "scan_frequency": {
                        "score": scores.get("scan_frequency", 0),
                        "weight": "5%",
                        "description": "How regularly the table is scanned",
                    },
                    "lineage_coverage": {
                        "score": scores.get("lineage_coverage", 50),
                        "weight": "5%",
                        "description": "Data lineage documentation coverage",
                    },
                },
            }
        except Exception as e:
            print(f"[TRUST] calculate_trust_score failed: {e}")
            return {
                "table_name": table_name,
                "trust_score": 0,
                "recommended_certification": "Quarantined",
                "error": str(e),
            }

    def score_all_tables(self, dq_results, catalog=None,
                         rules_engine=None, history=None):
        try:
            all_scores = {}
            for tname, result in dq_results.items():
                cat_entry = (
                    catalog.get_table_catalog(tname) if catalog else None)
                rule_result = (
                    rules_engine.evaluation_results.get(tname)
                    if rules_engine else None
                )
                hist_data = (
                    history.get_scan_frequency(tname) if history else None)
                all_scores[tname] = self.calculate_trust_score(
                    tname, dq_result=result, catalog_entry=cat_entry,
                    rule_results=rule_result, history_data=hist_data,
                )
            print(f"[TRUST] Scored {len(all_scores)} tables")
            return all_scores
        except Exception as e:
            print(f"[TRUST] score_all_tables failed: {e}")
            return {}

    def get_certification_recommendation(self, trust_score):
        try:
            if trust_score >= self.certification_thresholds["Certified"]:
                return "Certified"
            if trust_score >= self.certification_thresholds["Warning"]:
                return "Warning"
            return "Quarantined"
        except Exception:
            return "Quarantined"

    def generate_trust_summary(self, all_scores):
        try:
            if not all_scores:
                return {"total_tables": 0}

            certified = sum(
                1 for t in all_scores.values()
                if t.get("recommended_certification") == "Certified"
            )
            warning = sum(
                1 for t in all_scores.values()
                if t.get("recommended_certification") == "Warning"
            )
            quarantined = sum(
                1 for t in all_scores.values()
                if t.get("recommended_certification") == "Quarantined"
            )
            trust_vals = [
                t.get("trust_score", 0) for t in all_scores.values()
            ]
            avg = sum(trust_vals) / len(trust_vals) if trust_vals else 0

            sorted_scores = sorted(
                all_scores.items(), key=lambda x: x[1].get("trust_score", 0))

            return {
                "total_tables": len(all_scores),
                "certified": certified,
                "warning": warning,
                "quarantined": quarantined,
                "avg_trust_score": round(avg, 1),
                "lowest_trust": {
                    "table": sorted_scores[0][0],
                    "score": sorted_scores[0][1].get("trust_score", 0),
                },
                "highest_trust": {
                    "table": sorted_scores[-1][0],
                    "score": sorted_scores[-1][1].get("trust_score", 0),
                },
            }
        except Exception as e:
            print(f"[TRUST] generate_trust_summary failed: {e}")
            return {"total_tables": 0, "error": str(e)}
