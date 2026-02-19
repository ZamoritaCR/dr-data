"""Dashboard Rationalization Engine -- identify zombie dashboards, duplicates,
refresh waste, and hidden costs across an enterprise BI portfolio.
"""

import uuid
import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class DashboardRationalizationEngine:
    """Analyze a dashboard inventory for zombies, duplicates, waste, and cost."""

    def __init__(self):
        self.inventory = None
        self.analysis = {}

    # ------------------------------------------------------------------ #
    #  Sample data generation                                              #
    # ------------------------------------------------------------------ #

    def generate_sample_data(self, num_dashboards=10000):
        """Generate realistic enterprise dashboard inventory."""
        rng = np.random.default_rng(42)

        # --- Name pools ---
        prefixes = [
            "Monthly", "Weekly", "Daily", "Q1", "Q2", "Q3", "Q4", "Annual",
            "YTD", "MTD", "WTD", "Ad-Hoc", "Executive", "Regional",
            "Global", "Departmental", "Team", "Individual",
        ]
        topics = [
            "Revenue", "Compliance", "Risk", "Transactions", "Customer",
            "AML", "KYC", "Operations", "HR", "Sales", "Marketing",
            "Treasury", "Payroll", "Expense", "Fraud", "Audit",
            "Inventory", "Supply Chain", "Procurement", "Licensing",
            "Attrition", "Onboarding", "NPS", "Churn", "Retention",
            "Pipeline", "Conversion", "Lead", "Campaign", "Digital",
            "Settlement", "Reconciliation", "P&L", "Balance Sheet",
            "Cash Flow", "FX Exposure", "Credit", "Collections",
            "Delinquency", "Dispute", "Sanctions", "PEP Screening",
        ]
        suffixes = [
            "by Region", "by Country", "by Product", "by Channel",
            "by Segment", "by BU", "by Agent", "Overview", "Summary",
            "Detail", "Trend", "Breakdown", "Comparison", "Scorecard",
            "Tracker", "Monitor", "Analysis", "Report", "Dashboard",
            "KPIs", "Metrics", "Health Check",
        ]

        workspaces = [
            "Finance", "Compliance", "Operations", "Marketing", "HR",
            "Executive", "Sales", "Risk Management", "IT", "Analytics",
            "Treasury", "Legal",
        ]
        workspace_weights = np.array([
            15, 12, 14, 8, 6, 5, 12, 10, 6, 6, 3, 3,
        ], dtype=float)
        workspace_weights /= workspace_weights.sum()

        first_names = [
            "James", "Maria", "David", "Sarah", "Michael", "Jennifer",
            "Robert", "Linda", "William", "Patricia", "Richard", "Barbara",
            "Joseph", "Elizabeth", "Charles", "Susan", "Thomas", "Jessica",
            "Christopher", "Karen", "Daniel", "Nancy", "Matthew", "Lisa",
            "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
            "Paul", "Dorothy", "Andrew", "Kimberly", "Joshua", "Emily",
            "Kenneth", "Donna", "Kevin", "Michelle", "Brian", "Carol",
            "George", "Amanda", "Timothy", "Melissa", "Ronald", "Deborah",
            "Edward", "Stephanie", "Priya", "Raj", "Wei", "Mei",
            "Omar", "Fatima", "Carlos", "Ana", "Yuki", "Hiroshi",
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
            "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez",
            "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez",
            "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez",
            "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
            "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
            "Campbell", "Mitchell", "Carter", "Roberts", "Patel", "Shah",
            "Chen", "Wang", "Kumar", "Singh", "Ali", "Kim", "Park",
        ]

        platforms = ["Power BI", "Tableau", "Qlik"]
        platform_weights = np.array([0.60, 0.25, 0.15])

        refresh_options = [
            "4x daily", "2x daily", "1x daily", "weekly", "manual", "none",
        ]
        refresh_weights = np.array([0.08, 0.12, 0.30, 0.20, 0.18, 0.12])
        refresh_weights /= refresh_weights.sum()

        source_tables = [
            "fact_transactions", "fact_compliance_checks", "fact_settlements",
            "fact_payments", "fact_transfers", "fact_agent_activity",
            "fact_customer_interactions", "fact_fraud_alerts",
            "fact_aml_screening", "fact_kyc_checks", "fact_disputes",
            "fact_collections", "fact_campaigns", "fact_web_events",
            "fact_hr_events", "fact_expenses", "fact_invoices",
            "fact_orders", "fact_returns", "fact_inventory_moves",
            "dim_customer", "dim_agent", "dim_country", "dim_product",
            "dim_channel", "dim_currency", "dim_date", "dim_geography",
            "dim_employee", "dim_department", "dim_cost_center",
            "dim_gl_account", "dim_regulatory_body", "dim_risk_category",
            "agg_monthly_revenue", "agg_daily_transactions",
            "agg_weekly_compliance", "agg_quarterly_financials",
            "agg_agent_performance", "agg_customer_ltv",
            "stg_bank_feeds", "stg_partner_data", "stg_market_rates",
            "ref_sanctions_list", "ref_pep_list", "ref_country_risk",
            "ref_exchange_rates", "ref_fee_schedule", "ref_thresholds",
            "mart_executive_summary",
        ]

        # --- Generate rows ---
        rows = []
        now = datetime(2026, 2, 1)
        start_date = datetime(2020, 1, 1)
        date_range_days = (now - start_date).days

        # Pre-generate owner pool (~300 owners)
        owner_pool = []
        for _ in range(300):
            fn = rng.choice(first_names)
            ln = rng.choice(last_names)
            owner_pool.append(f"{fn.lower()}.{ln.lower()}@westernunion.com")
        owner_pool = list(set(owner_pool))

        for _ in range(num_dashboards):
            prefix = rng.choice(prefixes)
            topic = rng.choice(topics)
            suffix = rng.choice(suffixes)
            name = f"{prefix} {topic} {suffix}"

            ws = rng.choice(workspaces, p=workspace_weights)
            owner = rng.choice(owner_pool)
            platform = rng.choice(platforms, p=platform_weights)

            created_days_ago = rng.integers(0, date_range_days)
            created = start_date + timedelta(days=int(created_days_ago))

            # Last viewed: bias heavily towards stale
            #   70% > 90 days, 40% > 180 days, 20% > 365 days
            stale_roll = rng.random()
            if stale_roll < 0.20:
                # >365 days ago
                days_since = rng.integers(365, 800)
            elif stale_roll < 0.40:
                # 180-365 days ago
                days_since = rng.integers(180, 365)
            elif stale_roll < 0.70:
                # 90-180 days ago
                days_since = rng.integers(90, 180)
            elif stale_roll < 0.85:
                # 30-90 days ago
                days_since = rng.integers(30, 90)
            else:
                # active: 0-30 days ago
                days_since = rng.integers(0, 30)
            last_viewed = now - timedelta(days=int(days_since))

            # View count: correlated with staleness
            if days_since > 180:
                views = int(rng.exponential(1))
            elif days_since > 90:
                views = int(rng.exponential(3))
            elif days_since > 30:
                views = int(rng.exponential(10))
            else:
                views = int(rng.exponential(40)) + rng.integers(1, 20)
            views = max(0, min(views, 500))

            refresh = rng.choice(refresh_options, p=refresh_weights)

            n_tables = rng.integers(1, 5)
            tables = rng.choice(source_tables, size=n_tables, replace=False)
            source_str = ", ".join(tables)

            ds_size = int(rng.exponential(300)) + 10
            ds_size = min(ds_size, 5000)

            n_visuals = rng.integers(1, 26)
            certified = rng.random() < 0.15

            rows.append({
                "dashboard_id": str(uuid.uuid4()),
                "dashboard_name": name,
                "workspace": ws,
                "owner_email": owner,
                "platform": platform,
                "created_date": created.strftime("%Y-%m-%d"),
                "last_viewed_date": last_viewed.strftime("%Y-%m-%d"),
                "view_count_30d": views,
                "refresh_schedule": refresh,
                "source_tables": source_str,
                "dataset_size_mb": ds_size,
                "num_visuals": int(n_visuals),
                "is_certified": certified,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Load and prepare inventory                                          #
    # ------------------------------------------------------------------ #

    def load_inventory(self, df):
        """Store the inventory dataframe and compute derived fields."""
        self.inventory = df.copy()
        now = pd.Timestamp(datetime(2026, 2, 1))

        # Parse dates
        for col in ("created_date", "last_viewed_date"):
            if col in self.inventory.columns:
                self.inventory[col] = pd.to_datetime(
                    self.inventory[col], errors="coerce")

        # Derived
        if "last_viewed_date" in self.inventory.columns:
            self.inventory["days_since_viewed"] = (
                (now - self.inventory["last_viewed_date"]).dt.days
            ).fillna(9999).astype(int)
        else:
            self.inventory["days_since_viewed"] = 9999

        if "created_date" in self.inventory.columns:
            self.inventory["age_days"] = (
                (now - self.inventory["created_date"]).dt.days
            ).fillna(0).astype(int)
        else:
            self.inventory["age_days"] = 0

        self.analysis = {}

    # ------------------------------------------------------------------ #
    #  Full analysis                                                       #
    # ------------------------------------------------------------------ #

    def analyze(self):
        """Run the full rationalization analysis. Returns a dict of results."""
        if self.inventory is None or self.inventory.empty:
            return {}

        inv = self.inventory
        n = len(inv)

        # 1. Summary
        summary = {
            "total_dashboards": n,
            "total_workspaces": inv["workspace"].nunique()
                if "workspace" in inv.columns else 0,
            "total_owners": inv["owner_email"].nunique()
                if "owner_email" in inv.columns else 0,
            "avg_age_days": int(inv["age_days"].mean()) if "age_days" in inv.columns else 0,
            "platform_breakdown": (
                inv["platform"].value_counts().to_dict()
                if "platform" in inv.columns else {}
            ),
        }

        # 2. Zombies
        z90 = inv[inv["days_since_viewed"] > 90]
        z180 = inv[inv["days_since_viewed"] > 180]
        z365 = inv[inv["days_since_viewed"] > 365]
        zombies = {
            "over_90d": {"count": len(z90), "pct": round(100 * len(z90) / n, 1)},
            "over_180d": {"count": len(z180), "pct": round(100 * len(z180) / n, 1)},
            "over_365d": {"count": len(z365), "pct": round(100 * len(z365) / n, 1)},
            "zombie_df": z90.sort_values("days_since_viewed", ascending=False),
        }

        # 3. Duplicates (group by exact source_tables match)
        dup_groups = []
        if "source_tables" in inv.columns:
            grouped = inv.groupby("source_tables")
            for src, grp in grouped:
                if len(grp) >= 3:
                    dup_groups.append({
                        "source_tables": src,
                        "count": len(grp),
                        "dashboards": grp[["dashboard_name", "workspace",
                                           "owner_email", "view_count_30d"]].to_dict("records"),
                    })
            dup_groups.sort(key=lambda x: x["count"], reverse=True)

        total_in_dup_groups = sum(g["count"] for g in dup_groups)
        duplicates = {
            "groups": dup_groups,
            "total_groups": len(dup_groups),
            "total_dashboards_in_groups": total_in_dup_groups,
        }

        # 4. Refresh waste
        high_refresh = {"4x daily", "2x daily", "1x daily"}
        if "refresh_schedule" in inv.columns and "view_count_30d" in inv.columns:
            wasted = inv[
                (inv["refresh_schedule"].isin(high_refresh))
                & (inv["view_count_30d"] == 0)
            ]
        else:
            wasted = pd.DataFrame()

        refresh_cost_per_run = 0.12
        runs_per_day = {"4x daily": 4, "2x daily": 2, "1x daily": 1}
        annual_waste = 0
        for _, row in wasted.iterrows():
            sched = row.get("refresh_schedule", "none")
            daily_runs = runs_per_day.get(sched, 0)
            annual_waste += daily_runs * 365 * refresh_cost_per_run

        refresh_waste = {
            "wasted_refreshes_count": len(wasted),
            "annual_cost": round(annual_waste, 2),
            "wasted_df": wasted,
        }

        # 5. Workspace health
        ws_health = {}
        if "workspace" in inv.columns:
            for ws, grp in inv.groupby("workspace"):
                ws_n = len(grp)
                ws_zombies = len(grp[grp["days_since_viewed"] > 90])
                ws_dup = 0
                if "source_tables" in grp.columns:
                    ws_src_counts = grp["source_tables"].value_counts()
                    ws_dup = int((ws_src_counts >= 3).sum())
                ws_health[ws] = {
                    "total": ws_n,
                    "zombie_pct": round(100 * ws_zombies / ws_n, 1) if ws_n else 0,
                    "duplicate_groups": ws_dup,
                    "avg_views": round(grp["view_count_30d"].mean(), 1)
                        if "view_count_30d" in grp.columns else 0,
                }

        # 6. Cost impact
        dup_maint_cost = len(dup_groups) * 20 * 120  # 20hrs x $120/hr
        analyst_search_cost = 30 / 60 * 120 * 200 * 250  # 30min/day x 200 analysts x 250 days
        premium_capacity_waste = len(z90) * 8.50 * 12  # $8.50/mo per zombie dashboard
        total_annual_waste = (
            annual_waste + dup_maint_cost + analyst_search_cost + premium_capacity_waste
        )

        cost_impact = {
            "zombie_refresh_waste": round(annual_waste, 2),
            "duplicate_maintenance": round(dup_maint_cost, 2),
            "analyst_search_time": round(analyst_search_cost, 2),
            "premium_capacity_waste": round(premium_capacity_waste, 2),
            "total_annual_waste": round(total_annual_waste, 2),
        }

        # 7. Recommendations
        retire_list = z365[["dashboard_name", "workspace", "owner_email",
                            "days_since_viewed"]].to_dict("records") if len(z365) > 0 else []
        archive_180 = z180[~z180.index.isin(z365.index)]
        archive_list = archive_180[["dashboard_name", "workspace", "owner_email",
                                    "days_since_viewed"]].to_dict("records") if len(archive_180) > 0 else []

        consolidate_list = []
        for g in dup_groups[:50]:
            consolidate_list.append({
                "source_tables": g["source_tables"],
                "count": g["count"],
                "action": "CONSOLIDATE",
            })

        reduce_refresh_list = wasted[["dashboard_name", "workspace",
                                       "refresh_schedule"]].to_dict("records") if len(wasted) > 0 else []

        certify_candidates = inv[
            (inv["days_since_viewed"] <= 30)
            & (inv["view_count_30d"] >= 20)
            & (~inv.get("is_certified", pd.Series(dtype=bool)).fillna(False))
        ] if "is_certified" in inv.columns else pd.DataFrame()

        certify_list = certify_candidates[
            ["dashboard_name", "workspace", "view_count_30d"]
        ].to_dict("records") if len(certify_candidates) > 0 else []

        recommendations = {
            "retire": {"count": len(retire_list), "items": retire_list[:100]},
            "archive": {"count": len(archive_list), "items": archive_list[:100]},
            "consolidate": {"count": len(consolidate_list), "items": consolidate_list},
            "reduce_refresh": {"count": len(reduce_refresh_list), "items": reduce_refresh_list[:100]},
            "certify": {"count": len(certify_list), "items": certify_list[:100]},
        }

        # 8. Retirement plan
        retirement_plan = {
            "phase_1": {
                "label": "Immediate (>365 days stale)",
                "count": len(z365),
                "timeline": "This week",
            },
            "phase_2": {
                "label": "30-day notice (180-365 days stale)",
                "count": len(archive_180),
                "timeline": "Next 30 days",
            },
            "phase_3": {
                "label": "Duplicate consolidation",
                "count": total_in_dup_groups,
                "timeline": "Next 60 days",
            },
        }

        self.analysis = {
            "summary": summary,
            "zombies": zombies,
            "duplicates": duplicates,
            "refresh_waste": refresh_waste,
            "workspace_health": ws_health,
            "cost_impact": cost_impact,
            "recommendations": recommendations,
            "retirement_plan": retirement_plan,
        }
        return self.analysis

    # ------------------------------------------------------------------ #
    #  Executive summary (text)                                            #
    # ------------------------------------------------------------------ #

    def get_executive_summary(self):
        """Return a formatted text summary for chat display."""
        if not self.analysis:
            return "No analysis has been run yet. Load an inventory and run analyze() first."

        a = self.analysis
        s = a["summary"]
        z = a["zombies"]
        d = a["duplicates"]
        c = a["cost_impact"]
        r = a["retirement_plan"]

        lines = [
            "DASHBOARD RATIONALIZATION - EXECUTIVE SUMMARY",
            "=" * 50,
            "",
            f"Portfolio: {s['total_dashboards']:,} dashboards across "
            f"{s['total_workspaces']} workspaces, {s['total_owners']} owners",
            "",
            "ZOMBIE DASHBOARDS (not viewed):",
            f"  >90 days:  {z['over_90d']['count']:,} ({z['over_90d']['pct']}%)",
            f"  >180 days: {z['over_180d']['count']:,} ({z['over_180d']['pct']}%)",
            f"  >365 days: {z['over_365d']['count']:,} ({z['over_365d']['pct']}%)",
            "",
            f"DUPLICATE GROUPS: {d['total_groups']} groups "
            f"({d['total_dashboards_in_groups']:,} dashboards)",
            "",
            "ANNUAL COST IMPACT:",
            f"  Zombie refresh waste:    ${c['zombie_refresh_waste']:>12,.2f}",
            f"  Duplicate maintenance:   ${c['duplicate_maintenance']:>12,.2f}",
            f"  Analyst search time:     ${c['analyst_search_time']:>12,.2f}",
            f"  Premium capacity waste:  ${c['premium_capacity_waste']:>12,.2f}",
            f"  TOTAL ANNUAL WASTE:      ${c['total_annual_waste']:>12,.2f}",
            "",
            "RETIREMENT PLAN:",
            f"  Phase 1 (immediate):     {r['phase_1']['count']:,} dashboards",
            f"  Phase 2 (30-day notice): {r['phase_2']['count']:,} dashboards",
            f"  Phase 3 (consolidation): {r['phase_3']['count']:,} dashboards",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  HTML report                                                         #
    # ------------------------------------------------------------------ #

    def generate_html_report(self):
        """Generate a self-contained dark-themed HTML rationalization report."""
        if not self.analysis:
            return "<html><body>No analysis data.</body></html>"

        a = self.analysis
        s = a["summary"]
        z = a["zombies"]
        d = a["duplicates"]
        rw = a["refresh_waste"]
        ws = a["workspace_health"]
        c = a["cost_impact"]
        rec = a["recommendations"]
        rp = a["retirement_plan"]

        zombie_pct = z["over_90d"]["pct"]
        dup_pct = round(100 * d["total_dashboards_in_groups"] / max(s["total_dashboards"], 1), 1)
        refresh_pct = round(100 * rw["wasted_refreshes_count"] / max(s["total_dashboards"], 1), 1)
        cert_count = sum(1 for _ in (self.inventory["is_certified"] if "is_certified" in self.inventory.columns else []) if _)
        cert_pct = round(100 * cert_count / max(s["total_dashboards"], 1), 1)

        def svg_gauge(value, label, color, gauge_id):
            pct = min(max(value, 0), 100)
            angle = pct * 1.8  # 0-180 degrees
            rad = math.radians(180 - angle)
            end_x = 100 + 70 * math.cos(rad)
            end_y = 100 - 70 * math.sin(rad)
            large_arc = 1 if angle > 90 else 0
            return f'''<svg width="200" height="140" viewBox="0 0 200 140">
  <path d="M 30 100 A 70 70 0 0 1 170 100" fill="none" stroke="#333" stroke-width="12" stroke-linecap="round"/>
  <path d="M 30 100 A 70 70 0 {large_arc} 1 {end_x:.1f} {end_y:.1f}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>
  <text x="100" y="90" text-anchor="middle" fill="white" font-size="24" font-weight="bold">{pct:.0f}%</text>
  <text x="100" y="130" text-anchor="middle" fill="#aaa" font-size="12">{label}</text>
</svg>'''

        gauges_html = f'''<div style="display:flex;justify-content:space-around;flex-wrap:wrap;margin:20px 0;">
  {svg_gauge(zombie_pct, "Zombie Rate", "#ff4757", "g1")}
  {svg_gauge(dup_pct, "Duplicate Rate", "#ffa502", "g2")}
  {svg_gauge(refresh_pct, "Refresh Waste", "#ff6348", "g3")}
  {svg_gauge(cert_pct, "Certified", "#2ed573", "g4")}
</div>'''

        # Platform donut
        plat = s.get("platform_breakdown", {})
        plat_colors = {"Power BI": "#F2C811", "Tableau": "#E97627", "Qlik": "#009845"}
        donut_segments = []
        offset = 0
        total_plat = sum(plat.values()) or 1
        for p_name, p_count in plat.items():
            pct = p_count / total_plat * 100
            color = plat_colors.get(p_name, "#888")
            donut_segments.append(
                f'<circle r="40" cx="60" cy="60" fill="none" stroke="{color}" '
                f'stroke-width="20" stroke-dasharray="{pct * 2.51:.1f} 251.2" '
                f'stroke-dashoffset="{-offset * 2.512:.1f}" />'
            )
            offset += pct

        plat_legend = " ".join(
            f'<span style="color:{plat_colors.get(k,"#888")}">&#9679;</span> {k}: {v:,}'
            for k, v in plat.items()
        )
        donut_html = f'''<div style="text-align:center;margin:20px;">
<svg width="120" height="120" viewBox="0 0 120 120">
  {"".join(donut_segments)}
  <circle r="25" cx="60" cy="60" fill="#1a1a2e"/>
</svg>
<div style="color:#aaa;font-size:13px;margin-top:8px;">{plat_legend}</div>
</div>'''

        # Cost bar chart
        cost_items = [
            ("Zombie Refreshes", c["zombie_refresh_waste"], "#ff4757"),
            ("Duplicate Maint.", c["duplicate_maintenance"], "#ffa502"),
            ("Analyst Search", c["analyst_search_time"], "#ff6348"),
            ("Premium Capacity", c["premium_capacity_waste"], "#e84393"),
        ]
        max_cost = max((v for _, v, _ in cost_items), default=1) or 1
        cost_bars = ""
        for label, val, color in cost_items:
            w = val / max_cost * 300
            cost_bars += (
                f'<div style="margin:8px 0;">'
                f'<div style="color:#ccc;font-size:13px;margin-bottom:2px;">{label}</div>'
                f'<div style="display:flex;align-items:center;">'
                f'<div style="background:{color};height:24px;width:{w:.0f}px;'
                f'border-radius:4px;"></div>'
                f'<span style="color:white;margin-left:8px;font-size:14px;">'
                f'${val:,.0f}</span>'
                f'</div></div>'
            )

        # Workspace health table
        ws_rows = ""
        for ws_name in sorted(ws.keys()):
            wi = ws[ws_name]
            z_color = "#2ed573" if wi["zombie_pct"] < 30 else (
                "#ffa502" if wi["zombie_pct"] < 60 else "#ff4757")
            ws_rows += (
                f'<tr><td>{ws_name}</td><td>{wi["total"]}</td>'
                f'<td style="color:{z_color}">{wi["zombie_pct"]}%</td>'
                f'<td>{wi["duplicate_groups"]}</td>'
                f'<td>{wi["avg_views"]}</td></tr>'
            )

        # Duplicate groups table (top 20)
        dup_rows = ""
        for g in d["groups"][:20]:
            names = ", ".join(di["dashboard_name"] for di in g["dashboards"][:5])
            if g["count"] > 5:
                names += f" (+{g['count']-5} more)"
            dup_rows += (
                f'<tr><td>{g["source_tables"][:60]}</td>'
                f'<td>{g["count"]}</td>'
                f'<td style="font-size:12px">{names}</td></tr>'
            )

        # Retirement plan
        phases_html = ""
        for phase_key, phase in [("phase_1", rp["phase_1"]),
                                  ("phase_2", rp["phase_2"]),
                                  ("phase_3", rp["phase_3"])]:
            color = "#ff4757" if "1" in phase_key else (
                "#ffa502" if "2" in phase_key else "#2ed573")
            phases_html += (
                f'<div style="background:#16213e;border-left:4px solid {color};'
                f'padding:16px;margin:10px 0;border-radius:4px;">'
                f'<div style="color:{color};font-weight:bold;font-size:16px;">'
                f'{phase["label"]}</div>'
                f'<div style="color:white;font-size:24px;margin:8px 0;">'
                f'{phase["count"]:,} dashboards</div>'
                f'<div style="color:#aaa;">{phase["timeline"]}</div>'
                f'</div>'
            )

        # Recommendations summary
        rec_rows = ""
        for action, data in rec.items():
            badge_color = {
                "retire": "#ff4757", "archive": "#ffa502",
                "consolidate": "#1e90ff", "reduce_refresh": "#ff6348",
                "certify": "#2ed573",
            }.get(action, "#888")
            rec_rows += (
                f'<tr><td><span style="background:{badge_color};color:white;'
                f'padding:2px 10px;border-radius:12px;font-size:12px;'
                f'text-transform:uppercase;">{action}</span></td>'
                f'<td style="font-size:20px;font-weight:bold;">{data["count"]:,}</td></tr>'
            )

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard Rationalization Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#1a1a2e; color:#eee; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:40px 20px; }}
  .container {{ max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:28px; margin-bottom:4px; }}
  h2 {{ font-size:20px; color:#a8d8ea; margin:30px 0 12px; border-bottom:1px solid #333; padding-bottom:6px; }}
  .subtitle {{ color:#888; font-size:14px; margin-bottom:30px; }}
  .hero {{ display:flex; justify-content:space-around; flex-wrap:wrap; margin:24px 0; }}
  .hero-card {{ background:#16213e; border-radius:8px; padding:20px 30px; text-align:center; min-width:200px; margin:8px; }}
  .hero-num {{ font-size:36px; font-weight:bold; }}
  .hero-label {{ color:#aaa; font-size:13px; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; }}
  th {{ background:#16213e; color:#a8d8ea; padding:10px 12px; text-align:left; font-size:13px; }}
  td {{ padding:8px 12px; border-bottom:1px solid #2a2a4a; font-size:13px; }}
  tr:hover {{ background:#16213e55; }}
  .print-break {{ page-break-before:always; }}
  @media print {{ body {{ background:white; color:black; }} .hero-card {{ border:1px solid #ccc; }} th {{ background:#eee; color:black; }} }}
  @media (max-width:700px) {{ .hero {{ flex-direction:column; align-items:center; }} .hero-card {{ width:90%; }} }}
</style>
</head>
<body>
<div class="container">
<h1>Dashboard Rationalization Report</h1>
<div class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | {s["total_dashboards"]:,} dashboards analyzed</div>

<div class="hero">
  <div class="hero-card"><div class="hero-num" style="color:#ff4757">{z["over_90d"]["count"]:,}</div><div class="hero-label">Zombie Dashboards ({z["over_90d"]["pct"]}%)</div></div>
  <div class="hero-card"><div class="hero-num" style="color:#ffa502">{d["total_groups"]}</div><div class="hero-label">Duplicate Groups</div></div>
  <div class="hero-card"><div class="hero-num" style="color:#ff6348">{rw["wasted_refreshes_count"]:,}</div><div class="hero-label">Wasted Refreshes</div></div>
  <div class="hero-card"><div class="hero-num" style="color:#e84393">${c["total_annual_waste"]:,.0f}</div><div class="hero-label">Annual Waste</div></div>
</div>

{gauges_html}

<h2>Platform Breakdown</h2>
{donut_html}

<h2>Cost Impact Breakdown</h2>
<div style="background:#16213e;padding:20px;border-radius:8px;">
{cost_bars}
<div style="margin-top:16px;padding-top:12px;border-top:1px solid #333;">
<div style="color:white;font-size:18px;font-weight:bold;">Total Annual Waste: <span style="color:#ff4757">${c["total_annual_waste"]:,.0f}</span></div>
</div>
</div>

<h2>Workspace Health</h2>
<table>
<thead><tr><th>Workspace</th><th>Total</th><th>Zombie %</th><th>Dup Groups</th><th>Avg Views (30d)</th></tr></thead>
<tbody>{ws_rows}</tbody>
</table>

<div class="print-break"></div>
<h2>Top Duplicate Groups</h2>
<table>
<thead><tr><th>Source Tables</th><th>Count</th><th>Dashboards</th></tr></thead>
<tbody>{dup_rows}</tbody>
</table>

<h2>Retirement Plan</h2>
{phases_html}

<h2>Recommendations</h2>
<table>
<thead><tr><th>Action</th><th>Count</th></tr></thead>
<tbody>{rec_rows}</tbody>
</table>

<div style="margin-top:40px;padding:20px;background:#16213e;border-radius:8px;text-align:center;">
<div style="color:#888;font-size:12px;">Generated by Dr. Data - Dashboard Rationalization Engine</div>
</div>
</div>
</body>
</html>'''
        return html
