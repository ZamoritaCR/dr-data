"""
COPDQ Calculator -- Cost of Poor Data Quality.
Translates DQ scores into dollar impact so CFOs care.
"""

from datetime import datetime, timezone


class COPDQCalculator:
    """Calculate financial impact of data quality issues."""

    def __init__(self):
        self.cost_parameters = {
            "avg_analyst_hourly_rate": 67,
            "avg_manual_fix_minutes_per_record": 5,
            "avg_failed_transaction_cost": 25,
            "avg_compliance_fine_per_violation": 1000,
            "avg_customer_impact_cost": 50,
            "avg_rework_hours_per_report": 4,
            "reports_per_month": 50,
            "transactions_per_month": 24_000_000,
            "annual_revenue": 4_100_000_000,
        }

    def calculate_copdq(self, dq_results, cost_params=None):
        try:
            params = {**self.cost_parameters, **(cost_params or {})}
            total_cost = 0.0
            cost_breakdown = {}

            for table_name, result in dq_results.items():
                dims = result.get("dimensions", {})
                row_count = result.get("row_count", 0) or 0

                tc = {
                    "table_name": table_name,
                    "row_count": row_count,
                    "costs": {},
                    "total": 0.0,
                }

                # 1. Completeness -- missing data needs manual research
                comp = dims.get("completeness", {})
                comp_score = (
                    comp.get("score", 100)
                    if isinstance(comp, dict) else 100
                ) or 100
                if comp_score < 100:
                    missing_pct = (100 - comp_score) / 100
                    missing_records = int(row_count * missing_pct)
                    fix_hours = (
                        missing_records
                        * params["avg_manual_fix_minutes_per_record"]
                    ) / 60
                    monthly = fix_hours * params["avg_analyst_hourly_rate"]
                    tc["costs"]["completeness"] = {
                        "description": (
                            "Manual research and data entry for "
                            "missing values"
                        ),
                        "missing_records": missing_records,
                        "estimated_fix_hours": round(fix_hours, 1),
                        "annual_cost": round(monthly * 12, 2),
                        "monthly_cost": round(monthly, 2),
                    }
                    tc["total"] += monthly * 12

                # 2. Accuracy -- bad data causes failed txns + rework
                acc = dims.get("accuracy", {})
                acc_score = (
                    acc.get("score", 100)
                    if isinstance(acc, dict) else 100
                ) or 100
                if acc_score < 100:
                    err = (100 - acc_score) / 100
                    aff_txn = int(
                        params["transactions_per_month"] * err * 0.001)
                    txn_cost = (
                        aff_txn * params["avg_failed_transaction_cost"])
                    rework = (
                        params["avg_rework_hours_per_report"]
                        * params["avg_analyst_hourly_rate"]
                        * params["reports_per_month"]
                        * err
                    )
                    monthly = txn_cost + rework
                    tc["costs"]["accuracy"] = {
                        "description": (
                            "Failed transactions + report rework from "
                            "inaccurate data"
                        ),
                        "affected_transactions_monthly": aff_txn,
                        "transaction_loss_monthly": round(txn_cost, 2),
                        "rework_cost_monthly": round(rework, 2),
                        "annual_cost": round(monthly * 12, 2),
                        "monthly_cost": round(monthly, 2),
                    }
                    tc["total"] += monthly * 12

                # 3. Uniqueness -- duplicates cause double-processing
                uniq = dims.get("uniqueness", {})
                uniq_score = (
                    uniq.get("score", 100)
                    if isinstance(uniq, dict) else 100
                ) or 100
                if uniq_score < 100:
                    dup_pct = (100 - uniq_score) / 100
                    dup_records = int(row_count * dup_pct)
                    proc_cost = (
                        (dup_records * 2 / 60)
                        * params["avg_analyst_hourly_rate"]
                    )
                    cust_impact = (
                        int(dup_records * 0.01)
                        * params["avg_customer_impact_cost"]
                    )
                    monthly = proc_cost + cust_impact
                    tc["costs"]["uniqueness"] = {
                        "description": (
                            "Duplicate investigation + customer impact "
                            "from duplicate records"
                        ),
                        "duplicate_records": dup_records,
                        "processing_cost_monthly": round(proc_cost, 2),
                        "customer_impact_monthly": round(cust_impact, 2),
                        "annual_cost": round(monthly * 12, 2),
                        "monthly_cost": round(monthly, 2),
                    }
                    tc["total"] += monthly * 12

                # 4. Validity -- invalid data triggers compliance reviews
                val = dims.get("validity", {})
                val_score = (
                    val.get("score", 100)
                    if isinstance(val, dict) else 100
                ) or 100
                if val_score < 100:
                    viol_pct = (100 - val_score) / 100
                    violations = int(row_count * viol_pct)
                    annual = (
                        min(violations, 100)
                        * params["avg_compliance_fine_per_violation"]
                    )
                    tc["costs"]["validity"] = {
                        "description": (
                            "Compliance review and potential regulatory "
                            "exposure from invalid data"
                        ),
                        "violations": violations,
                        "compliance_exposure_annual": round(annual, 2),
                        "annual_cost": round(annual, 2),
                        "monthly_cost": round(annual / 12, 2),
                    }
                    tc["total"] += annual

                # 5. Timeliness -- stale data leads to wrong decisions
                time_d = dims.get("timeliness", {})
                time_score = (
                    time_d.get("score")
                    if isinstance(time_d, dict) else None
                )
                if time_score is not None and time_score < 100:
                    stale = (100 - time_score) / 100
                    monthly = (
                        stale
                        * params["reports_per_month"]
                        * params["avg_rework_hours_per_report"]
                        * params["avg_analyst_hourly_rate"]
                        * 0.5
                    )
                    tc["costs"]["timeliness"] = {
                        "description": (
                            "Decision delays and report invalidation "
                            "from stale data"
                        ),
                        "staleness_factor": round(stale, 2),
                        "annual_cost": round(monthly * 12, 2),
                        "monthly_cost": round(monthly, 2),
                    }
                    tc["total"] += monthly * 12

                cost_breakdown[table_name] = tc
                total_cost += tc["total"]

            rev = params.get("annual_revenue", 0) or 1
            return {
                "total_annual_cost": round(total_cost, 2),
                "total_monthly_cost": round(total_cost / 12, 2),
                "revenue_impact_pct": round(
                    (total_cost / rev) * 100, 4
                ) if rev > 0 else 0,
                "cost_breakdown": cost_breakdown,
                "parameters_used": params,
                "calculated_at": datetime.now(timezone.utc).isoformat(),
                "gartner_benchmark": (
                    "Gartner estimates poor data quality costs "
                    "organizations an average of $12.9M per year"
                ),
            }
        except Exception as e:
            print(f"[COPDQ] calculate_copdq failed: {e}")
            return {
                "total_annual_cost": 0,
                "total_monthly_cost": 0,
                "revenue_impact_pct": 0,
                "error": str(e),
            }

    def format_currency(self, amount):
        try:
            amount = float(amount)
            if amount >= 1_000_000:
                return f"${amount / 1_000_000:.1f}M"
            if amount >= 1_000:
                return f"${amount / 1_000:.0f}K"
            return f"${amount:.0f}"
        except Exception:
            return "$0"
