"""
DQ History -- tracks data quality scores over time for trend analysis,
degradation detection, and improvement tracking.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

_DIMENSIONS = [
    "completeness", "accuracy", "consistency",
    "timeliness", "uniqueness", "validity",
]


class DQHistory:
    """Persistent DQ scan history with trend & degradation detection."""

    def __init__(self, history_path="dq_history.json"):
        self.history_path = history_path
        self.history = self._load_history()

    # ── Persistence ──

    def _load_history(self):
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[DQ HISTORY] Loaded {len(data.get('scans', []))} scans "
                      f"from {self.history_path}")
                return data
        except Exception as e:
            print(f"[DQ HISTORY] Load failed: {e}")
        return {
            "version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "scans": [],
            "table_history": {},
        }

    def _save_history(self):
        try:
            # Trim global scans to 500
            self.history["scans"] = self.history["scans"][-500:]
            self.history["updated_at"] = datetime.now(timezone.utc).isoformat()
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print(f"[DQ HISTORY] Save failed: {e}")

    # ── Record ──

    def record_scan(self, table_name, scan_result):
        try:
            ts = scan_result.get(
                "scan_timestamp", datetime.now(timezone.utc).isoformat())
            entry = {
                "timestamp": ts,
                "table_name": table_name,
                "overall_score": scan_result.get("overall_score", 0),
                "row_count": scan_result.get("row_count", 0),
                "column_count": scan_result.get("column_count", 0),
                "dimensions": {},
            }
            for dim, dim_data in scan_result.get("dimensions", {}).items():
                if isinstance(dim_data, dict):
                    entry["dimensions"][dim.lower()] = {
                        "score": dim_data.get("score"),
                        "status": dim_data.get("status", "N/A"),
                    }

            self.history["scans"].append(entry)

            if table_name not in self.history["table_history"]:
                self.history["table_history"][table_name] = []
            self.history["table_history"][table_name].append(entry)
            # Trim per-table to 100
            self.history["table_history"][table_name] = (
                self.history["table_history"][table_name][-100:])

            self._save_history()
            print(f"[DQ HISTORY] Recorded scan for '{table_name}': "
                  f"score={entry['overall_score']}")
        except Exception as e:
            print(f"[DQ HISTORY] record_scan failed: {e}")

    # ── Trends ──

    def get_table_trend(self, table_name, dimension=None, limit=30):
        try:
            entries = self.history["table_history"].get(table_name, [])
            entries = entries[-limit:]
            if dimension:
                return [
                    {
                        "timestamp": e["timestamp"],
                        "score": e.get("dimensions", {}).get(
                            dimension.lower(), {}).get("score"),
                    }
                    for e in entries
                ]
            return [
                {"timestamp": e["timestamp"], "score": e.get("overall_score")}
                for e in entries
            ]
        except Exception as e:
            print(f"[DQ HISTORY] get_table_trend failed: {e}")
            return []

    def get_dimension_trend(self, table_name, dimension, limit=30):
        try:
            entries = self.history["table_history"].get(table_name, [])
            entries = entries[-limit:]
            dim_low = dimension.lower()
            return [
                {
                    "timestamp": e["timestamp"],
                    "score": e.get("dimensions", {}).get(dim_low, {}).get("score"),
                    "status": e.get("dimensions", {}).get(dim_low, {}).get("status", "N/A"),
                }
                for e in entries
            ]
        except Exception as e:
            print(f"[DQ HISTORY] get_dimension_trend failed: {e}")
            return []

    def get_all_table_latest(self):
        try:
            latest = {}
            for tname, entries in self.history["table_history"].items():
                if entries:
                    latest[tname] = entries[-1]
            return latest
        except Exception as e:
            print(f"[DQ HISTORY] get_all_table_latest failed: {e}")
            return {}

    # ── Degradation Detection ──

    def detect_degradation(self, table_name, threshold_pct=5):
        try:
            entries = self.history["table_history"].get(table_name, [])
            if len(entries) < 2:
                return {"degraded": False, "note": "Insufficient history"}

            prev, curr = entries[-2], entries[-1]
            overall_change = (
                (curr.get("overall_score", 0) or 0)
                - (prev.get("overall_score", 0) or 0)
            )

            dim_changes = {}
            degraded_dims = []
            improved_dims = []
            all_dims = set(
                list(curr.get("dimensions", {}).keys())
                + list(prev.get("dimensions", {}).keys())
            )
            for dim in all_dims:
                c_score = curr.get("dimensions", {}).get(dim, {}).get("score")
                p_score = prev.get("dimensions", {}).get(dim, {}).get("score")
                if c_score is not None and p_score is not None:
                    change = c_score - p_score
                    dim_changes[dim] = round(change, 2)
                    if change < -threshold_pct:
                        degraded_dims.append(dim)
                    elif change > threshold_pct:
                        improved_dims.append(dim)

            return {
                "degraded": overall_change < -threshold_pct,
                "overall_change": round(overall_change, 2),
                "dimension_changes": dim_changes,
                "degraded_dimensions": degraded_dims,
                "improved_dimensions": improved_dims,
            }
        except Exception as e:
            print(f"[DQ HISTORY] detect_degradation failed: {e}")
            return {"degraded": False, "error": str(e)}

    # ── Stats ──

    def get_best_worst(self, table_name):
        try:
            entries = self.history["table_history"].get(table_name, [])
            if not entries:
                return {"error": "No history for this table"}

            scores = [
                (e.get("overall_score", 0) or 0, e.get("timestamp", ""))
                for e in entries
            ]
            best = max(scores, key=lambda x: x[0])
            worst = min(scores, key=lambda x: x[0])
            avg = sum(s[0] for s in scores) / len(scores)

            return {
                "best_score": best[0],
                "best_date": best[1],
                "worst_score": worst[0],
                "worst_date": worst[1],
                "current_score": scores[-1][0],
                "total_scans": len(entries),
                "avg_score": round(avg, 2),
            }
        except Exception as e:
            print(f"[DQ HISTORY] get_best_worst failed: {e}")
            return {}

    def get_scan_frequency(self, table_name):
        try:
            entries = self.history["table_history"].get(table_name, [])
            if not entries:
                return {"total_scans": 0}

            timestamps = []
            for e in entries:
                try:
                    ts = e.get("timestamp", "")
                    if ts:
                        timestamps.append(
                            datetime.fromisoformat(ts.replace("Z", "+00:00")))
                except (ValueError, TypeError):
                    pass

            if not timestamps:
                return {"total_scans": len(entries)}

            timestamps.sort()
            now = datetime.now(timezone.utc)
            avg_days = 0.0
            if len(timestamps) > 1:
                deltas = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]
                avg_days = (sum(deltas) / len(deltas)) / 86400

            last_7 = sum(1 for t in timestamps if (now - t).days <= 7)
            last_30 = sum(1 for t in timestamps if (now - t).days <= 30)

            return {
                "total_scans": len(entries),
                "first_scan": timestamps[0].isoformat(),
                "last_scan": timestamps[-1].isoformat(),
                "avg_days_between_scans": round(avg_days, 2),
                "scans_last_7_days": last_7,
                "scans_last_30_days": last_30,
            }
        except Exception as e:
            print(f"[DQ HISTORY] get_scan_frequency failed: {e}")
            return {"total_scans": 0, "error": str(e)}

    # ── Chart Data ──

    def generate_trend_chart_data(self, table_name, limit=30):
        try:
            entries = self.history["table_history"].get(table_name, [])
            entries = entries[-limit:]

            data = {
                "timestamps": [],
                "overall": [],
            }
            for dim in _DIMENSIONS:
                data[dim] = []

            for e in entries:
                data["timestamps"].append(e.get("timestamp", ""))
                data["overall"].append(e.get("overall_score"))
                dims = e.get("dimensions", {})
                for dim in _DIMENSIONS:
                    data[dim].append(
                        dims.get(dim, {}).get("score"))

            return data
        except Exception as e:
            print(f"[DQ HISTORY] generate_trend_chart_data failed: {e}")
            return {"timestamps": []}

    # ── Compare ──

    def compare_tables(self, table_names):
        try:
            tables = {}
            for tname in table_names:
                entries = self.history["table_history"].get(tname, [])
                if entries:
                    latest = entries[-1]
                    dims = {}
                    for dim in _DIMENSIONS:
                        dims[dim] = latest.get("dimensions", {}).get(
                            dim, {}).get("score")
                    tables[tname] = {
                        "overall": latest.get("overall_score", 0) or 0,
                        "dimensions": dims,
                    }

            if not tables:
                return {"tables": {}, "note": "No history for any table"}

            scores = {n: t["overall"] for n, t in tables.items()}
            avg_overall = sum(scores.values()) / len(scores)

            return {
                "tables": tables,
                "best_table": max(scores, key=scores.get),
                "worst_table": min(scores, key=scores.get),
                "avg_overall": round(avg_overall, 2),
            }
        except Exception as e:
            print(f"[DQ HISTORY] compare_tables failed: {e}")
            return {"tables": {}, "error": str(e)}

    # ── Export ──

    def export_history(self, table_name=None, fmt="json"):
        try:
            if table_name:
                entries = self.history["table_history"].get(table_name, [])
            else:
                entries = self.history.get("scans", [])

            if fmt == "csv":
                header = "timestamp,table,overall"
                for dim in _DIMENSIONS:
                    header += f",{dim}"
                lines = [header]
                for e in entries:
                    row = (
                        f"{e.get('timestamp', '')},"
                        f"{e.get('table_name', '')},"
                        f"{e.get('overall_score', '')}"
                    )
                    dims = e.get("dimensions", {})
                    for dim in _DIMENSIONS:
                        val = dims.get(dim, {}).get("score", "")
                        row += f",{val if val is not None else ''}"
                    lines.append(row)
                return "\n".join(lines)

            # Default: JSON
            return json.dumps(entries, indent=2, default=str)
        except Exception as e:
            print(f"[DQ HISTORY] export_history failed: {e}")
            return "[]"
