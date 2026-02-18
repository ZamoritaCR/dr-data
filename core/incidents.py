"""
Incident Manager -- formal incident records for data failures with root
cause analysis, impact assessment, and prevention measures.
Think PagerDuty for data.
"""

import json
import os
from datetime import datetime, timezone
from collections import defaultdict

_OPEN_STATES = {"Open", "Acknowledged"}


def _t(title, sev, cat, prompts):
    return {"title_template": title, "severity_default": sev,
            "category": cat, "rca_prompts": prompts}


class IncidentManager:
    """Create, track, and resolve data incidents with postmortem workflow."""

    def __init__(self, incidents_path="incidents.json"):
        self.incidents_path = incidents_path
        self.data = self._load()

    # -- Persistence --

    def _load(self):
        try:
            if os.path.exists(self.incidents_path):
                with open(self.incidents_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[INCIDENTS] Loaded {len(data.get('incidents', {}))} incidents")
                return data
        except Exception as e:
            print(f"[INCIDENTS] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "incidents": {},
            "templates": {
                "data_quality_degradation": _t("DQ Degradation: {table_name} - {dimension}", "HIGH", "Data Quality",
                    ["What changed in upstream ETL?", "Schema change?", "Source system outage?", "Code deployment?"]),
                "schema_drift": _t("Schema Drift: {table_name}", "CRITICAL", "Schema",
                    ["Intentional schema change?", "Which team owns source?", "Downstream consumers notified?"]),
                "volume_anomaly": _t("Volume Anomaly: {table_name} ({change_pct}% change)", "HIGH", "Volume",
                    ["Batch job failure?", "Source system issues?", "Filter change in ETL?"]),
                "sla_breach": _t("Data Freshness SLA Breach: {table_name}", "HIGH", "Timeliness",
                    ["ETL pipeline running?", "Source system available?", "Network issue?"]),
                "compliance_gap": _t("Compliance Gap: {framework} - {requirement}", "CRITICAL", "Compliance",
                    ["Which data elements non-compliant?", "Regulatory exposure?", "Remediation timeline?"]),
            },
            "stats": {"total_created": 0, "total_resolved": 0},
        }

    def _save(self):
        try:
            self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.incidents_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"[INCIDENTS] Save failed: {e}")

    def _now(self):
        return datetime.now(timezone.utc).isoformat()

    def _get(self, incident_id):
        return self.data["incidents"].get(incident_id)

    def _tl(self, inc, event, by="System"):
        inc["timeline"].append({"event": event, "at": self._now(), "by": by})

    # -- Create --

    def create_incident(self, title, description, severity, category,
                        table_name=None, template=None, metadata=None):
        try:
            seq = self.data["stats"]["total_created"] + 1
            iid = f"INC-{seq:04d}"
            now = self._now()
            incident = {
                "id": iid, "title": title, "description": description,
                "severity": severity, "category": category, "status": "Open",
                "table_name": table_name,
                "detected_at": now, "acknowledged_at": None,
                "resolved_at": None, "closed_at": None,
                "assigned_to": None, "responders": [],
                "impact_assessment": {
                    "affected_tables": [], "affected_reports": [],
                    "affected_users": "Unknown", "business_impact": "",
                    "data_records_affected": 0, "estimated_financial_impact": 0},
                "root_cause": {"category": None, "description": "",
                    "identified_at": None, "identified_by": None,
                    "contributing_factors": []},
                "resolution": {"description": "", "steps_taken": [],
                    "resolved_by": None, "data_recovered": False, "permanent_fix": False},
                "prevention": {"measures": [], "monitoring_added": False,
                    "rule_added": False, "documentation_updated": False},
                "timeline": [{"event": "Incident created", "at": now, "by": "System"}],
                "metadata": metadata or {}, "rca_prompts": [],
                "linked_issues": [], "postmortem_complete": False,
            }
            if template and template in self.data["templates"]:
                incident["rca_prompts"] = self.data["templates"][template].get("rca_prompts", [])
            self.data["incidents"][iid] = incident
            self.data["stats"]["total_created"] += 1
            self._save()
            print(f"[INCIDENTS] Created {iid}: {title[:60]}")
            return iid
        except Exception as e:
            print(f"[INCIDENTS] create_incident failed: {e}")
            return None

    def auto_create_from_degradation(self, table_name, degradation_result):
        try:
            if not degradation_result.get("degraded"):
                return None
            change = abs(degradation_result.get("overall_change", 0))
            dims = degradation_result.get("degraded_dimensions", [])
            dim_str = ", ".join(d.get("dimension", "?") for d in dims[:3]) if dims else "multiple"
            return self.create_incident(
                f"DQ Degradation: {table_name} - score dropped {change:.1f} points",
                f"DQ degradation in {table_name}. Dimensions: {dim_str}. Change: -{change:.1f}pts.",
                "HIGH", "Data Quality", table_name=table_name,
                template="data_quality_degradation", metadata={"degradation": degradation_result})
        except Exception as e:
            print(f"[INCIDENTS] auto_create_from_degradation failed: {e}")
            return None

    def auto_create_from_schema_drift(self, table_name, drift_result):
        try:
            if not drift_result.get("has_drift"):
                return None
            parts = []
            for key, label in [("new_columns", "New"), ("removed_columns", "Removed")]:
                cols = drift_result.get(key, [])
                if cols:
                    parts.append(f"{label}: {', '.join(cols[:5])}")
            tc = drift_result.get("type_changes", [])
            if tc:
                parts.append(f"{len(tc)} type change(s)")
            return self.create_incident(
                f"Schema Drift Detected: {table_name}",
                f"Schema drift in {table_name}. {'; '.join(parts)}",
                "CRITICAL", "Schema", table_name=table_name,
                template="schema_drift", metadata={"drift": drift_result})
        except Exception as e:
            print(f"[INCIDENTS] auto_create_from_schema_drift failed: {e}")
            return None

    def auto_create_from_volume_anomaly(self, table_name, volume_result):
        try:
            if not volume_result.get("is_anomaly"):
                return None
            pct = volume_result.get("change_pct", 0)
            return self.create_incident(
                f"Volume Anomaly: {table_name} ({pct:.0f}% change)",
                f"Volume anomaly in {table_name}. Change: {pct:.1f}%.",
                "HIGH", "Volume", table_name=table_name,
                template="volume_anomaly", metadata={"volume": volume_result})
        except Exception as e:
            print(f"[INCIDENTS] auto_create_from_volume_anomaly failed: {e}")
            return None

    def acknowledge_incident(self, incident_id, acknowledged_by):
        try:
            inc = self._get(incident_id)
            if not inc:
                return
            inc["acknowledged_at"] = self._now()
            inc["status"] = "Acknowledged"
            self._tl(inc, f"Acknowledged by {acknowledged_by}", acknowledged_by)
            self._save()
        except Exception as e:
            print(f"[INCIDENTS] acknowledge_incident failed: {e}")

    def update_impact(self, incident_id, impact_dict):
        try:
            inc = self._get(incident_id)
            if not inc:
                return
            for k, v in impact_dict.items():
                if k in inc["impact_assessment"]:
                    inc["impact_assessment"][k] = v
            self._tl(inc, "Impact assessment updated")
            self._save()
        except Exception as e:
            print(f"[INCIDENTS] update_impact failed: {e}")

    def set_root_cause(self, incident_id, category, description,
                       contributing_factors=None, identified_by=None):
        try:
            inc = self._get(incident_id)
            if not inc:
                return
            by = identified_by or "System"
            inc["root_cause"] = {
                "category": category, "description": description,
                "identified_at": self._now(), "identified_by": by,
                "contributing_factors": contributing_factors or []}
            self._tl(inc, f"Root cause set: {category}", by)
            self._save()
            print(f"[INCIDENTS] {incident_id} RCA: {category}")
        except Exception as e:
            print(f"[INCIDENTS] set_root_cause failed: {e}")

    def resolve_incident(self, incident_id, resolution_description,
                         steps_taken, resolved_by,
                         permanent_fix=False, data_recovered=False):
        try:
            inc = self._get(incident_id)
            if not inc:
                return
            inc["status"] = "Resolved"
            inc["resolved_at"] = self._now()
            inc["resolution"] = {
                "description": resolution_description,
                "steps_taken": steps_taken or [], "resolved_by": resolved_by,
                "data_recovered": data_recovered, "permanent_fix": permanent_fix}
            self._tl(inc, f"Resolved by {resolved_by}", resolved_by)
            self.data["stats"]["total_resolved"] += 1
            self._save()
            print(f"[INCIDENTS] {incident_id} resolved")
        except Exception as e:
            print(f"[INCIDENTS] resolve_incident failed: {e}")

    def add_prevention_measure(self, incident_id, measure):
        try:
            inc = self._get(incident_id)
            if inc:
                inc["prevention"]["measures"].append(measure)
                self._save()
        except Exception as e:
            print(f"[INCIDENTS] add_prevention_measure failed: {e}")

    def complete_postmortem(self, incident_id):
        try:
            inc = self._get(incident_id)
            if not inc:
                return
            inc["postmortem_complete"] = True
            inc["status"] = "Closed"
            inc["closed_at"] = self._now()
            self._tl(inc, "Postmortem completed, incident closed")
            self._save()
        except Exception as e:
            print(f"[INCIDENTS] complete_postmortem failed: {e}")

    def link_to_issue(self, incident_id, issue_id):
        try:
            inc = self._get(incident_id)
            if inc and issue_id not in inc["linked_issues"]:
                inc["linked_issues"].append(issue_id)
                self._save()
        except Exception as e:
            print(f"[INCIDENTS] link_to_issue failed: {e}")

    def get_incidents(self, status=None, severity=None, category=None):
        try:
            sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
            results = []
            for inc in self.data["incidents"].values():
                if status and inc.get("status") != status:
                    continue
                if severity and inc.get("severity") != severity:
                    continue
                if category and inc.get("category") != category:
                    continue
                results.append(inc)
            return sorted(results, key=lambda x: (
                sev_order.get(x.get("severity"), 9), x.get("detected_at", "")))
        except Exception as e:
            print(f"[INCIDENTS] get_incidents failed: {e}")
            return []

    def _parse_ts(self, ts_str):
        try:
            return datetime.fromisoformat(
                ts_str.replace("Z", "+00:00")) if ts_str else None
        except (ValueError, TypeError):
            return None

    def get_dashboard_stats(self):
        try:
            incs = list(self.data["incidents"].values())
            if not incs:
                return {"total_incidents": 0, "open": 0, "resolved": 0,
                        "closed": 0, "mttr": None, "postmortems_pending": 0}
            by_sev, by_cat, rca_cats = defaultdict(int), defaultdict(int), defaultdict(int)
            open_c = resolved_c = closed_c = pm_done = pm_pending = 0
            res_hours = []
            for inc in incs:
                st = inc.get("status", "Open")
                by_sev[inc.get("severity", "LOW")] += 1
                by_cat[inc.get("category", "Other")] += 1
                if st in _OPEN_STATES: open_c += 1
                elif st == "Resolved":
                    resolved_c += 1
                    if not inc.get("postmortem_complete"): pm_pending += 1
                elif st == "Closed": closed_c += 1
                if inc.get("postmortem_complete"): pm_done += 1
                rca = inc.get("root_cause", {}).get("category")
                if rca: rca_cats[rca] += 1
                ct, rt = self._parse_ts(inc.get("detected_at")), self._parse_ts(inc.get("resolved_at"))
                if ct and rt:
                    res_hours.append((rt - ct).total_seconds() / 3600)
            mttr = round(sum(res_hours) / len(res_hours), 1) if res_hours else None
            return {
                "total_incidents": len(incs), "open": open_c,
                "resolved": resolved_c, "closed": closed_c,
                "by_severity": dict(by_sev), "by_category": dict(by_cat),
                "avg_resolution_hours": mttr, "mttr": mttr,
                "postmortems_complete": pm_done, "postmortems_pending": pm_pending,
                "most_common_root_cause": max(rca_cats, key=rca_cats.get) if rca_cats else None,
            }
        except Exception as e:
            print(f"[INCIDENTS] get_dashboard_stats failed: {e}")
            return {"total_incidents": 0, "open": 0, "resolved": 0,
                    "closed": 0, "mttr": None}

    def generate_postmortem_report(self, incident_id):
        try:
            inc = self._get(incident_id)
            if not inc:
                return ""
            rca, res, prev = inc.get("root_cause", {}), inc.get("resolution", {}), inc.get("prevention", {})
            impact = inc.get("impact_assessment", {})
            _ts = lambda k: (inc.get(k) or "N/A")[:16]
            lines = [
                f"# Postmortem Report: {inc['id']}", f"**Title:** {inc.get('title', '')}",
                f"**Severity:** {inc.get('severity', '')}  |  **Category:** {inc.get('category', '')}",
                f"**Table:** {inc.get('table_name', 'N/A')}  |  **Status:** {inc.get('status', '')}",
                "", "## Timeline",
                f"- Detected: {_ts('detected_at')}  |  Acknowledged: {_ts('acknowledged_at')}",
                f"- Resolved: {_ts('resolved_at')}  |  Closed: {_ts('closed_at')}",
                "", "### Events"]
            for ev in inc.get("timeline", []):
                lines.append(f"- {ev.get('at', '')[:16]} | {ev.get('event', '')} ({ev.get('by', 'System')})")
            aff_t = ", ".join(impact.get("affected_tables", [])) or "N/A"
            aff_r = ", ".join(impact.get("affected_reports", [])) or "N/A"
            lines += ["", "## Impact",
                f"- Tables: {aff_t}  |  Reports: {aff_r}",
                f"- Users: {impact.get('affected_users', 'Unknown')}  |  Records: {impact.get('data_records_affected', 0):,}",
                f"- Business Impact: {impact.get('business_impact', 'Not assessed')}",
                "", "## Root Cause",
                f"- Category: {rca.get('category', 'Not identified')}",
                f"- Description: {rca.get('description', 'Not provided')}",
                f"- Identified By: {rca.get('identified_by', 'N/A')}"]
            factors = rca.get("contributing_factors", [])
            if factors:
                lines.append("- Contributing: " + "; ".join(factors))
            yn = lambda v: "Yes" if v else "No"
            lines += ["", "## Resolution",
                f"- {res.get('description', 'N/A')}  (by {res.get('resolved_by', 'N/A')})",
                f"- Data Recovered: {yn(res.get('data_recovered'))}  |  Permanent Fix: {yn(res.get('permanent_fix'))}"]
            steps = res.get("steps_taken", [])
            if steps:
                for i, s in enumerate(steps, 1):
                    lines.append(f"  {i}. {s}")
            lines += ["", "## Prevention"]
            measures = prev.get("measures", [])
            for m in (measures or ["No prevention measures documented yet."]):
                lines.append(f"- {m}")
            lines += [f"- Monitoring: {yn(prev.get('monitoring_added'))} | Rule: {yn(prev.get('rule_added'))} | Docs: {yn(prev.get('documentation_updated'))}",
                "", "---", f"*Generated: {self._now()[:16]}*"]
            return "\n".join(lines)
        except Exception as e:
            print(f"[INCIDENTS] generate_postmortem_report failed: {e}")
            return ""

    def export_incidents(self, fmt="json"):
        try:
            incs = self.data.get("incidents", {})
            if fmt == "json":
                return json.dumps(incs, indent=2, default=str)
            if fmt == "csv":
                rows = ["id,title,severity,category,status,table,detected_at,resolved_at"]
                for inc in incs.values():
                    ra = inc.get("resolved_at", "")
                    rows.append(f"{inc.get('id','')},{inc.get('title','')[:50]},"
                        f"{inc.get('severity','')},{inc.get('category','')},"
                        f"{inc.get('status','')},{inc.get('table_name','N/A')},"
                        f"{inc.get('detected_at','')[:16]},{ra[:16] if ra else 'N/A'}")
                return "\n".join(rows)
            if fmt == "markdown":
                lines = ["# Data Incident Log",
                         f"**Generated:** {self._now()[:16]}", f"**Total:** {len(incs)}", ""]
                for inc in incs.values():
                    lines += [f"## {inc.get('id','')} - {inc.get('title','')}",
                        f"- Severity: {inc.get('severity','')} | Status: {inc.get('status','')}",
                        f"- Detected: {inc.get('detected_at','')[:16]}", ""]
                return "\n".join(lines)
            return json.dumps(incs, indent=2, default=str)
        except Exception as e:
            print(f"[INCIDENTS] export_incidents failed: {e}")
            return "[]"
