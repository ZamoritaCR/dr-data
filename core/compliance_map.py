"""
Compliance Mapper -- maps DQ rules and data elements to specific
regulatory requirements.  Critical for heavily regulated environments
(BSA/AML, KYC, SOX, PCI-DSS, GDPR, OFAC).
"""

import json
import os
import re
from datetime import datetime, timezone


def _req(title, desc, elems, dims, sev="HIGH"):
    return {"title": title, "description": desc,
            "data_elements": elems, "dq_dimensions": dims, "severity": sev}


def _fw(full, auth, reqs):
    return {"full_name": full, "authority": auth, "requirements": reqs}


def _default_frameworks():
    """Pre-populated regulatory frameworks for financial services."""
    return {
        "BSA/AML": _fw("Bank Secrecy Act / Anti-Money Laundering", "FinCEN", {
            "BSA-001": _req("Customer Identification Program (CIP)",
                "Verify identity of customers. Requires name, DOB, address, ID number.",
                ["customer_name", "date_of_birth", "address", "ssn", "id_number", "id_type"],
                ["completeness", "accuracy", "validity"], "CRITICAL"),
            "BSA-002": _req("Currency Transaction Reports",
                "Report transactions exceeding $10,000.",
                ["transaction_amount", "transaction_date", "customer_name", "customer_id"],
                ["completeness", "accuracy", "timeliness"], "CRITICAL"),
            "BSA-003": _req("Suspicious Activity Reports",
                "File SAR for suspicious transactions. Data must be complete and timely.",
                ["transaction_amount", "transaction_type", "sender_name", "receiver_name", "country"],
                ["completeness", "accuracy", "timeliness"], "CRITICAL"),
            "BSA-004": _req("Recordkeeping",
                "Maintain records of transactions for 5 years.",
                ["transaction_id", "transaction_date", "amount", "customer_id"],
                ["completeness", "uniqueness", "timeliness"]),
        }),
        "KYC": _fw("Know Your Customer", "FinCEN / OFAC", {
            "KYC-001": _req("Customer Due Diligence",
                "Collect and verify customer identity. Screen against OFAC sanctions lists.",
                ["customer_name", "date_of_birth", "nationality", "address", "id_document"],
                ["completeness", "accuracy", "validity"], "CRITICAL"),
            "KYC-002": _req("Enhanced Due Diligence",
                "Additional scrutiny for high-risk customers. Complete risk assessment.",
                ["risk_score", "source_of_funds", "occupation", "transaction_pattern"],
                ["completeness", "accuracy"]),
            "KYC-003": _req("Ongoing Monitoring",
                "Continuous monitoring of customer transactions for unusual patterns.",
                ["transaction_history", "alert_status", "review_date"],
                ["timeliness", "completeness"]),
        }),
        "SOX": _fw("Sarbanes-Oxley Act", "SEC", {
            "SOX-001": _req("Financial Data Integrity",
                "Financial reporting data must be accurate and complete.",
                ["revenue", "expenses", "transaction_amount", "balance"],
                ["accuracy", "completeness", "consistency"], "CRITICAL"),
            "SOX-002": _req("Audit Trail",
                "Maintain audit trails for all financial data changes.",
                ["modified_by", "modified_at", "change_type", "old_value", "new_value"],
                ["completeness", "timeliness"]),
            "SOX-003": _req("Internal Controls",
                "Data quality controls must be documented and tested.",
                ["control_id", "test_date", "test_result", "remediation"],
                ["completeness", "validity"]),
        }),
        "PCI-DSS": _fw("Payment Card Industry Data Security Standard", "PCI SSC", {
            "PCI-001": _req("Cardholder Data Protection",
                "Protect stored cardholder data. Mask PAN when displayed.",
                ["card_number", "cardholder_name", "expiry_date", "cvv"],
                ["validity", "accuracy"], "CRITICAL"),
            "PCI-002": _req("Data Retention",
                "Do not store sensitive authentication data after authorization.",
                ["card_number", "cvv", "pin_block"], ["validity"], "CRITICAL"),
        }),
        "GDPR": _fw("General Data Protection Regulation", "EU DPA", {
            "GDPR-001": _req("Data Accuracy (Article 5)",
                "Personal data must be accurate and kept up to date.",
                ["customer_name", "email", "phone", "address"],
                ["accuracy", "timeliness"]),
            "GDPR-002": _req("Data Minimization",
                "Only collect data necessary for the stated purpose.",
                [], ["validity"], "MEDIUM"),
            "GDPR-003": _req("Right to Erasure",
                "Ability to delete personal data on request. Requires complete data inventory.",
                ["customer_id", "pii_columns"], ["completeness", "uniqueness"]),
        }),
        "OFAC": _fw("Office of Foreign Assets Control Sanctions", "US Treasury", {
            "OFAC-001": _req("Sanctions Screening",
                "Screen all parties against OFAC SDN list. Name matching must be accurate.",
                ["sender_name", "receiver_name", "country", "address"],
                ["accuracy", "completeness", "consistency"], "CRITICAL"),
        }),
    }


_STRIP_PREFIXES = re.compile(r"^[a-z]_")


def _normalize(name):
    """Lowercase, strip underscores and common 1-letter prefixes."""
    n = name.lower().replace("_", "").replace("-", "").replace(" ", "")
    n = _STRIP_PREFIXES.sub("", name.lower().replace("-", " "))
    return n.replace("_", "").replace(" ", "")


class ComplianceMapper:
    """Map data elements to regulatory requirements and assess compliance."""

    def __init__(self, compliance_path="compliance_map.json"):
        self.compliance_path = compliance_path
        self.data = self._load()

    # ── Persistence ──

    def _load(self):
        try:
            if os.path.exists(self.compliance_path):
                with open(self.compliance_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[COMPLIANCE] Loaded from {self.compliance_path}")
                return data
        except Exception as e:
            print(f"[COMPLIANCE] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frameworks": _default_frameworks(),
            "column_mappings": {},
            "compliance_scores": {},
        }

    def _save(self):
        try:
            self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.compliance_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"[COMPLIANCE] Save failed: {e}")

    # ── Column Mapping ──

    def map_column_to_requirement(self, table_name, column_name,
                                  framework, requirement_id):
        try:
            key = f"{table_name}.{column_name}"
            if key not in self.data["column_mappings"]:
                self.data["column_mappings"][key] = []
            # Avoid duplicates
            for existing in self.data["column_mappings"][key]:
                if (existing.get("framework") == framework
                        and existing.get("requirement_id") == requirement_id):
                    return
            self.data["column_mappings"][key].append({
                "framework": framework,
                "requirement_id": requirement_id,
                "mapped_at": datetime.now(timezone.utc).isoformat(),
            })
            self._save()
        except Exception as e:
            print(f"[COMPLIANCE] map_column_to_requirement failed: {e}")

    def auto_map_columns(self, df, table_name):
        """Auto-detect column-to-regulation mappings by name matching."""
        try:
            mappings_created = []
            col_norms = {c: _normalize(c) for c in df.columns}

            for fw_name, fw in self.data["frameworks"].items():
                for req_id, req in fw.get("requirements", {}).items():
                    elems = req.get("data_elements", [])
                    elem_norms = [_normalize(e) for e in elems]
                    for col, cn in col_norms.items():
                        if any(en and en in cn or cn in en
                               for en in elem_norms if en):
                            self.map_column_to_requirement(
                                table_name, col, fw_name, req_id)
                            mappings_created.append(
                                f"{table_name}.{col} -> "
                                f"{fw_name}/{req_id}")
            print(f"[COMPLIANCE] Auto-mapped {len(mappings_created)} "
                  f"links for '{table_name}'")
            return mappings_created
        except Exception as e:
            print(f"[COMPLIANCE] auto_map_columns failed: {e}")
            return []

    # ── Assessment ──

    def assess_compliance(self, table_name, dq_result, rule_results=None):
        try:
            dims = dq_result.get("dimensions", {})
            now = datetime.now(timezone.utc).isoformat()
            fw_results, all_scores, critical_gaps = {}, [], []
            for fw_name, fw in self.data["frameworks"].items():
                req_results, comp, ncomp = {}, 0, 0
                for req_id, req in fw.get("requirements", {}).items():
                    mapped = []
                    for key, maps in self.data["column_mappings"].items():
                        if not key.startswith(f"{table_name}."):
                            continue
                        for m in maps:
                            if m.get("framework") == fw_name and m.get("requirement_id") == req_id:
                                mapped.append(key.split(".", 1)[1])
                    if not mapped:
                        continue
                    dsc, iss = [], []
                    for d in req.get("dq_dimensions", []):
                        dd = dims.get(d, {})
                        sc = (dd.get("score", 100) if isinstance(dd, dict) else 100) or 100
                        dsc.append(sc)
                        if sc < 90:
                            iss.append(f"{d}: {sc:.1f}% (threshold 90%)")
                    score = sum(dsc) / len(dsc) if dsc else 100
                    ok = score >= 80
                    comp += ok; ncomp += (not ok)
                    req_results[req_id] = {
                        "title": req.get("title", ""), "compliant": ok,
                        "score": round(score, 1), "mapped_columns": mapped,
                        "issues": iss, "severity": req.get("severity", "MEDIUM")}
                    all_scores.append(score)
                    if not ok:
                        critical_gaps.append({"framework": fw_name, "requirement_id": req_id,
                            "title": req.get("title", ""), "score": round(score, 1),
                            "severity": req.get("severity")})
                if req_results:
                    rsc = [r["score"] for r in req_results.values()]
                    fw_results[fw_name] = {
                        "requirements": req_results, "compliant_count": comp,
                        "non_compliant_count": ncomp,
                        "overall_score": round(sum(rsc) / len(rsc), 1)}
            overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 100
            result = {"table_name": table_name, "assessed_at": now, "frameworks": fw_results,
                      "overall_compliance_score": overall, "critical_gaps": critical_gaps}
            self.data["compliance_scores"][table_name] = result
            self._save()
            print(f"[COMPLIANCE] Assessed '{table_name}': {overall}% compliant")
            return result
        except Exception as e:
            print(f"[COMPLIANCE] assess_compliance failed: {e}")
            return {}

    # ── Summary ──

    def get_compliance_summary(self):
        try:
            scores = self.data.get("compliance_scores", {})
            if not scores:
                return {"tables_assessed": 0, "overall_score": 0}
            all_ov, by_fw, all_gaps = [], {}, []
            for tc in scores.values():
                all_ov.append(tc.get("overall_compliance_score", 0))
                all_gaps.extend(tc.get("critical_gaps", []))
                for fn, fd in tc.get("frameworks", {}).items():
                    if fn not in by_fw:
                        by_fw[fn] = {"s": [], "c": 0, "nc": 0}
                    by_fw[fn]["s"].append(fd.get("overall_score", 0))
                    by_fw[fn]["c"] += fd.get("compliant_count", 0)
                    by_fw[fn]["nc"] += fd.get("non_compliant_count", 0)
            fw_sum = {fw: {"score": round(sum(d["s"])/len(d["s"]), 1) if d["s"] else 0,
                           "compliant": d["c"], "non_compliant": d["nc"]}
                      for fw, d in by_fw.items()}
            overall = round(sum(all_ov) / len(all_ov), 1) if all_ov else 0
            return {"tables_assessed": len(scores), "overall_score": overall,
                    "by_framework": fw_sum, "critical_gaps": all_gaps,
                    "top_risk_areas": sorted(all_gaps, key=lambda x: x.get("score", 100))[:10]}
        except Exception as e:
            print(f"[COMPLIANCE] get_compliance_summary failed: {e}")
            return {"tables_assessed": 0, "overall_score": 0}

    # ── Getters ──

    def get_requirements_by_framework(self, framework):
        try:
            fw = self.data["frameworks"].get(framework, {})
            return fw.get("requirements", {})
        except Exception:
            return {}

    def get_column_regulatory_tags(self, table_name, column_name):
        try:
            key = f"{table_name}.{column_name}"
            return self.data["column_mappings"].get(key, [])
        except Exception:
            return []

    # ── Export ──

    def export_compliance_report(self, fmt="json"):
        try:
            scores = self.data.get("compliance_scores", {})
            if fmt == "json":
                return json.dumps(scores, indent=2, default=str)
            if fmt != "markdown":
                return json.dumps(scores, indent=2, default=str)
            summary = self.get_compliance_summary()
            lines = [
                "# Regulatory Compliance Report",
                f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
                f"**Overall Score:** {summary.get('overall_score', 0):.1f}%",
                f"**Tables Assessed:** {summary.get('tables_assessed', 0)}",
                "", "## Framework Summary", "",
            ]
            for fw, fd in summary.get("by_framework", {}).items():
                lines.append(
                    f"- **{fw}**: {fd.get('score', 0):.1f}% "
                    f"({fd.get('compliant', 0)} compliant / "
                    f"{fd.get('non_compliant', 0)} non-compliant)")
            lines += ["", "## Critical Gaps", ""]
            for g in summary.get("critical_gaps", []):
                lines.append(
                    f"- [{g.get('severity', '')}] "
                    f"{g.get('framework', '')} / "
                    f"{g.get('requirement_id', '')}: "
                    f"{g.get('title', '')} ({g.get('score', 0):.1f}%)")
            lines += ["", "## Per-Table Detail", ""]
            for tname, tcomp in scores.items():
                lines.append(
                    f"### {tname} "
                    f"({tcomp.get('overall_compliance_score', 0):.1f}%)")
                for fw_n, fw_d in tcomp.get("frameworks", {}).items():
                    lines.append(f"#### {fw_n}")
                    for rid, rd in fw_d.get("requirements", {}).items():
                        tag = "[PASS]" if rd.get("compliant") else "[FAIL]"
                        lines.append(
                            f"- {tag} {rid}: {rd.get('title', '')} "
                            f"({rd.get('score', 0):.1f}%)")
                        if rd.get("issues"):
                            for iss in rd["issues"]:
                                lines.append(f"  - {iss}")
                lines.append("")
            return "\n".join(lines)
        except Exception as e:
            print(f"[COMPLIANCE] export_compliance_report failed: {e}")
            return "{}"
