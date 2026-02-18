"""
Stewardship Workflow -- issue lifecycle manager for DQ problems.
When quality issues are found they get assigned, tracked, and resolved.
Think Jira for data quality.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict


_SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
_OPEN_STATES = {"Open", "Assigned", "In Progress"}
_CLOSED_STATES = {"Resolved", "Verified", "Closed", "Wont Fix"}
_ALL_STATUSES = _OPEN_STATES | _CLOSED_STATES


class StewardshipWorkflow:
    """Issue lifecycle manager for data quality problems."""

    def __init__(self, workflow_path="stewardship.json"):
        self.workflow_path = workflow_path
        self.data = self._load()

    # ── Persistence ──

    def _load(self):
        try:
            if os.path.exists(self.workflow_path):
                with open(self.workflow_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[STEWARDSHIP] Loaded {len(data.get('issues', {}))} "
                      f"issues from {self.workflow_path}")
                return data
        except Exception as e:
            print(f"[STEWARDSHIP] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "issues": {},
            "stewards": {},
            "teams": {},
            "sla_config": {
                "CRITICAL": {"response_hours": 4, "resolution_hours": 24},
                "HIGH": {"response_hours": 24, "resolution_hours": 72},
                "MEDIUM": {"response_hours": 48, "resolution_hours": 168},
                "LOW": {"response_hours": 168, "resolution_hours": 720},
            },
            "escalation_rules": {
                "auto_escalate_after_hours": 48,
                "escalation_chain": [
                    "Data Steward", "Data Owner", "Domain Lead", "CDO"],
            },
            "stats": {
                "total_created": 0,
                "total_resolved": 0,
                "total_escalated": 0,
            },
        }

    def _save(self):
        try:
            self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.workflow_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"[STEWARDSHIP] Save failed: {e}")

    # ── Issue CRUD ──

    def create_issue(self, title, description, table_name,
                     column_name=None, dimension=None,
                     severity="MEDIUM", source="DQ Scan",
                     assigned_to=None, regulatory_tag=None):
        try:
            seq = self.data["stats"]["total_created"] + 1
            issue_id = f"DQ-{seq:04d}"
            now = datetime.now(timezone.utc)

            now_s = now.isoformat()
            issue = {
                "id": issue_id, "title": title, "description": description,
                "table_name": table_name, "column_name": column_name,
                "dimension": dimension, "severity": severity,
                "source": source, "status": "Open",
                "assigned_to": assigned_to or "Unassigned",
                "created_at": now_s, "updated_at": now_s,
                "due_at": None, "resolved_at": None,
                "resolution_notes": None, "resolution_type": None,
                "escalated": False, "escalation_level": 0,
                "regulatory_tag": regulatory_tag, "comments": [],
                "history": [{"action": "Created", "by": "System",
                             "at": now_s,
                             "details": f"Issue created from {source}"}],
            }

            sla = self.data["sla_config"].get(severity, {})
            if sla.get("resolution_hours"):
                due = now + timedelta(hours=sla["resolution_hours"])
                issue["due_at"] = due.isoformat()

            self.data["issues"][issue_id] = issue
            self.data["stats"]["total_created"] = seq
            self._save()
            print(f"[STEWARDSHIP] Created {issue_id}: {title[:60]}")
            return issue_id
        except Exception as e:
            print(f"[STEWARDSHIP] create_issue failed: {e}")
            return None

    def create_issues_from_dq_scan(self, scan_result, table_name):
        """Auto-create issues from DQ scan recommendations."""
        try:
            created = 0
            recs = scan_result.get("recommendations", [])
            for rec in recs:
                prio = rec.get("priority", "LOW")
                if prio not in ("CRITICAL", "HIGH"):
                    continue
                dim = rec.get("dimension")
                finding = rec.get("finding", "")
                existing = self.find_similar_issue(
                    table_name, dim, finding)
                if existing:
                    continue
                self.create_issue(
                    title=finding[:100] or "DQ Issue",
                    description=(
                        f"{finding}\n\n"
                        f"Recommendation: {rec.get('recommendation', '')}\n\n"
                        f"Impact: {rec.get('impact', '')}"
                    ),
                    table_name=table_name,
                    dimension=dim,
                    severity=prio,
                    source="DQ Scan",
                )
                created += 1
            print(f"[STEWARDSHIP] Created {created} issues from scan "
                  f"of '{table_name}'")
            return created
        except Exception as e:
            print(f"[STEWARDSHIP] create_issues_from_dq_scan failed: {e}")
            return 0

    def find_similar_issue(self, table_name, dimension,
                           finding_text=None):
        """Find an open issue for the same table+dimension."""
        try:
            for iid, iss in self.data["issues"].items():
                if iss.get("status") in _CLOSED_STATES:
                    continue
                if iss.get("table_name") != table_name:
                    continue
                if dimension and iss.get("dimension") != dimension:
                    continue
                if finding_text:
                    snippet = finding_text[:50].lower()
                    title = (iss.get("title") or "").lower()
                    desc = (iss.get("description") or "").lower()
                    if snippet not in title and snippet not in desc:
                        continue
                return iid
            return None
        except Exception as e:
            print(f"[STEWARDSHIP] find_similar_issue failed: {e}")
            return None

    # ── Lifecycle ──

    def assign_issue(self, issue_id, assigned_to):
        try:
            iss = self.data["issues"].get(issue_id)
            if not iss:
                print(f"[STEWARDSHIP] Issue {issue_id} not found")
                return
            iss["assigned_to"] = assigned_to
            if iss["status"] == "Open":
                iss["status"] = "Assigned"
            iss["updated_at"] = datetime.now(timezone.utc).isoformat()
            iss["history"].append({
                "action": "Assigned", "by": "System",
                "at": iss["updated_at"],
                "details": f"Assigned to {assigned_to}",
            })
            self._save()
            print(f"[STEWARDSHIP] {issue_id} assigned to {assigned_to}")
        except Exception as e:
            print(f"[STEWARDSHIP] assign_issue failed: {e}")

    def update_status(self, issue_id, new_status,
                      updated_by="User", notes=None):
        try:
            if new_status not in _ALL_STATUSES:
                print(f"[STEWARDSHIP] Invalid status: {new_status}")
                return
            iss = self.data["issues"].get(issue_id)
            if not iss:
                print(f"[STEWARDSHIP] Issue {issue_id} not found")
                return
            old = iss["status"]
            iss["status"] = new_status
            iss["updated_at"] = datetime.now(timezone.utc).isoformat()
            if new_status in ("Resolved", "Closed"):
                iss["resolved_at"] = iss["updated_at"]
                self.data["stats"]["total_resolved"] += 1
            if notes:
                iss["resolution_notes"] = notes
            iss["history"].append({
                "action": "Status Change", "by": updated_by,
                "at": iss["updated_at"],
                "details": f"{old} -> {new_status}"
                           + (f". Notes: {notes}" if notes else ""),
            })
            self._save()
            print(f"[STEWARDSHIP] {issue_id}: {old} -> {new_status}")
        except Exception as e:
            print(f"[STEWARDSHIP] update_status failed: {e}")

    def add_comment(self, issue_id, comment_text, author="User"):
        try:
            iss = self.data["issues"].get(issue_id)
            if not iss:
                print(f"[STEWARDSHIP] Issue {issue_id} not found")
                return
            now = datetime.now(timezone.utc).isoformat()
            iss["comments"].append({
                "text": comment_text, "author": author, "at": now,
            })
            iss["updated_at"] = now
            iss["history"].append({
                "action": "Comment", "by": author,
                "at": now, "details": comment_text[:80],
            })
            self._save()
        except Exception as e:
            print(f"[STEWARDSHIP] add_comment failed: {e}")

    def escalate_issue(self, issue_id, reason=None):
        try:
            iss = self.data["issues"].get(issue_id)
            if not iss:
                print(f"[STEWARDSHIP] Issue {issue_id} not found")
                return
            chain = self.data["escalation_rules"]["escalation_chain"]
            new_level = min(
                iss.get("escalation_level", 0) + 1, len(chain) - 1)
            iss["escalation_level"] = new_level
            iss["escalated"] = True
            iss["updated_at"] = datetime.now(timezone.utc).isoformat()
            target = chain[new_level]
            iss["history"].append({
                "action": "Escalated", "by": "System",
                "at": iss["updated_at"],
                "details": f"Escalated to {target}"
                           + (f". Reason: {reason}" if reason else ""),
            })
            self.data["stats"]["total_escalated"] += 1
            self._save()
            print(f"[STEWARDSHIP] {issue_id} escalated to {target}")
        except Exception as e:
            print(f"[STEWARDSHIP] escalate_issue failed: {e}")

    # ── SLA ──

    def check_sla_breaches(self):
        try:
            now = datetime.now(timezone.utc)
            breached = []
            for iid, iss in self.data["issues"].items():
                if iss.get("status") not in _OPEN_STATES:
                    continue
                due_str = iss.get("due_at")
                if not due_str:
                    continue
                try:
                    due = datetime.fromisoformat(
                        due_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if now > due:
                    delta = (now - due).total_seconds() / 3600
                    breached.append({
                        "issue_id": iid,
                        "title": iss.get("title", ""),
                        "severity": iss.get("severity"),
                        "hours_overdue": round(delta, 1),
                    })
            return sorted(breached, key=lambda x: -x["hours_overdue"])
        except Exception as e:
            print(f"[STEWARDSHIP] check_sla_breaches failed: {e}")
            return []

    # ── Query ──

    def get_issues(self, status=None, severity=None,
                   table_name=None, assigned_to=None):
        try:
            results = []
            for iss in self.data["issues"].values():
                if status and iss.get("status") != status:
                    continue
                if severity and iss.get("severity") != severity:
                    continue
                if table_name and iss.get("table_name") != table_name:
                    continue
                if assigned_to and iss.get("assigned_to") != assigned_to:
                    continue
                results.append(iss)
            results.sort(key=lambda x: (
                _SEVERITY_ORDER.get(x.get("severity"), 9),
                x.get("created_at", ""),
            ))
            return results
        except Exception as e:
            print(f"[STEWARDSHIP] get_issues failed: {e}")
            return []

    def _parse_ts(self, ts_str):
        """Parse ISO timestamp, return datetime or None."""
        try:
            return datetime.fromisoformat(
                ts_str.replace("Z", "+00:00")) if ts_str else None
        except (ValueError, TypeError):
            return None

    def get_dashboard_stats(self):
        try:
            issues = list(self.data["issues"].values())
            if not issues:
                return {"total_open": 0, "total_resolved": 0,
                        "total_wont_fix": 0, "sla_breaches": 0,
                        "unassigned": 0}
            by_sev, by_status, by_table = defaultdict(int), defaultdict(int), defaultdict(int)
            open_iss, resolved_iss = [], []
            for iss in issues:
                st = iss.get("status", "Open")
                by_sev[iss.get("severity", "MEDIUM")] += 1
                by_status[st] += 1
                by_table[iss.get("table_name", "Unknown")] += 1
                if st in _OPEN_STATES:
                    open_iss.append(iss)
                if st in ("Resolved", "Verified", "Closed"):
                    resolved_iss.append(iss)
            # Avg resolution hours
            rh = []
            for iss in resolved_iss:
                ct, rt = self._parse_ts(iss.get("created_at")), self._parse_ts(iss.get("resolved_at"))
                if ct and rt:
                    rh.append((rt - ct).total_seconds() / 3600)
            avg_res = round(sum(rh) / len(rh), 1) if rh else 0
            # Oldest open
            oldest = None
            if open_iss:
                o = min(open_iss, key=lambda x: x.get("created_at", ""))
                oldest = {"id": o.get("id"), "title": o.get("title"),
                          "created_at": o.get("created_at")}
            # Recent resolved (last 5)
            recent = sorted(resolved_iss, key=lambda x: x.get("resolved_at", ""), reverse=True)[:5]
            recent_list = [{"id": r.get("id"), "title": r.get("title"),
                            "resolved_at": r.get("resolved_at")} for r in recent]
            unassigned = sum(1 for i in open_iss if i.get("assigned_to") == "Unassigned")
            return {
                "total_open": len(open_iss), "total_resolved": len(resolved_iss),
                "total_wont_fix": by_status.get("Wont Fix", 0),
                "by_severity": dict(by_sev), "by_status": dict(by_status),
                "by_table": dict(by_table),
                "sla_breaches": len(self.check_sla_breaches()),
                "avg_resolution_hours": avg_res, "oldest_open": oldest,
                "recently_resolved": recent_list, "unassigned": unassigned,
            }
        except Exception as e:
            print(f"[STEWARDSHIP] get_dashboard_stats failed: {e}")
            return {"total_open": 0, "total_resolved": 0,
                    "sla_breaches": 0, "error": str(e)}

    # ── Stewards & Teams ──

    def add_steward(self, name, email=None, role="Data Steward",
                    tables=None, domains=None):
        try:
            self.data["stewards"][name] = {
                "email": email or "",
                "role": role,
                "assigned_tables": tables or [],
                "domains": domains or [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()
            print(f"[STEWARDSHIP] Added steward: {name}")
        except Exception as e:
            print(f"[STEWARDSHIP] add_steward failed: {e}")

    def add_team(self, name, members, lead=None):
        try:
            self.data["teams"][name] = {
                "members": members,
                "lead": lead,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()
            print(f"[STEWARDSHIP] Added team: {name}")
        except Exception as e:
            print(f"[STEWARDSHIP] add_team failed: {e}")

    def auto_assign_by_table(self, table_name):
        try:
            for sname, sd in self.data["stewards"].items():
                if table_name in sd.get("assigned_tables", []):
                    return sname
            return "Unassigned"
        except Exception as e:
            print(f"[STEWARDSHIP] auto_assign_by_table failed: {e}")
            return "Unassigned"

    # ── Export ──

    def export_issues(self, fmt="json", status=None):
        try:
            issues = self.get_issues(status=status) if status else list(
                self.data["issues"].values())

            if fmt == "csv":
                header = (
                    "id,title,table,column,dimension,severity,"
                    "status,assigned_to,created_at,due_at,resolved_at"
                )
                lines = [header]
                for iss in issues:
                    row = ",".join([
                        iss.get("id", ""),
                        f'"{(iss.get("title") or "").replace(chr(34), "")}"',
                        iss.get("table_name", ""),
                        iss.get("column_name") or "",
                        iss.get("dimension") or "",
                        iss.get("severity", ""),
                        iss.get("status", ""),
                        iss.get("assigned_to", ""),
                        iss.get("created_at", ""),
                        iss.get("due_at") or "",
                        iss.get("resolved_at") or "",
                    ])
                    lines.append(row)
                return "\n".join(lines)

            return json.dumps(issues, indent=2, default=str)
        except Exception as e:
            print(f"[STEWARDSHIP] export_issues failed: {e}")
            return "[]"
