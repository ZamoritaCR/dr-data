"""
Relationship Detector -- Auto-detect joinable columns across multiple DataFrames.

Given a dict of {name: DataFrame}, detects shared keys (exact name match,
ID patterns, value overlap) and can auto-join them into a single unified
DataFrame.
"""
import re
import pandas as pd


class RelationshipDetector:
    """Detect and exploit relationships between multiple DataFrames."""

    # Common ID suffixes that signal a join key
    _ID_PATTERNS = re.compile(
        r"(?:_?id|_?key|_?code|_?num|_?no|_?number)$", re.IGNORECASE
    )

    # Abbreviation pairs (both directions checked)
    _ABBREVS = {
        "cust": "customer", "prod": "product", "cat": "category",
        "emp": "employee", "dept": "department", "loc": "location",
        "qty": "quantity", "amt": "amount", "txn": "transaction",
        "acct": "account", "inv": "invoice", "ord": "order",
        "mgr": "manager", "org": "organization", "desc": "description",
    }

    def detect(self, dataframes_dict):
        """Find joinable column pairs across all DataFrames.

        Args:
            dataframes_dict: {name: DataFrame}

        Returns:
            list of relationship dicts:
                {left_table, right_table, left_col, right_col,
                 match_type, confidence, overlap_pct}
        """
        relationships = []
        names = list(dataframes_dict.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                left_name = names[i]
                right_name = names[j]
                left_df = dataframes_dict[left_name]
                right_df = dataframes_dict[right_name]

                pairs = self._find_join_candidates(left_df, right_df)

                for left_col, right_col, match_type, confidence in pairs:
                    overlap = self._value_overlap(
                        left_df[left_col], right_df[right_col]
                    )
                    if overlap > 0:
                        # Boost confidence if values actually overlap well
                        if overlap >= 0.5:
                            confidence = min(1.0, confidence + 0.2)
                        relationships.append({
                            "left_table": left_name,
                            "right_table": right_name,
                            "left_col": left_col,
                            "right_col": right_col,
                            "match_type": match_type,
                            "confidence": round(confidence, 2),
                            "overlap_pct": round(overlap * 100, 1),
                        })

        # Sort by confidence descending
        relationships.sort(key=lambda r: r["confidence"], reverse=True)
        return relationships

    def auto_join(self, dataframes_dict, relationships=None):
        """Join multiple DataFrames into one using detected relationships.

        Args:
            dataframes_dict: {name: DataFrame}
            relationships: pre-computed list from detect(). If None, calls detect().

        Returns:
            (joined_df, join_log) -- join_log is a list of strings describing
            each merge step.
        """
        if len(dataframes_dict) < 2:
            name = next(iter(dataframes_dict))
            return dataframes_dict[name].copy(), [f"Single table: {name}"]

        if relationships is None:
            relationships = self.detect(dataframes_dict)

        if not relationships:
            # No relationships found -- return the largest DataFrame
            largest_name = max(
                dataframes_dict, key=lambda k: len(dataframes_dict[k])
            )
            return (
                dataframes_dict[largest_name].copy(),
                [f"No relationships found. Using largest table: {largest_name}"],
            )

        # Build the join plan: greedily pick the best relationship for each
        # unjoined table, starting from the table in the highest-confidence
        # relationship.
        joined_tables = set()
        join_log = []

        # Start with the best relationship
        best = relationships[0]
        result = pd.merge(
            dataframes_dict[best["left_table"]],
            dataframes_dict[best["right_table"]],
            left_on=best["left_col"],
            right_on=best["right_col"],
            how="left",
            suffixes=("", f"_{best['right_table'][:20]}"),
        )
        joined_tables.add(best["left_table"])
        joined_tables.add(best["right_table"])
        join_log.append(
            f"Joined '{best['left_table']}' with '{best['right_table']}' "
            f"on {best['left_col']}={best['right_col']} "
            f"(confidence {best['confidence']}, overlap {best['overlap_pct']}%)"
        )

        # Try to join remaining tables
        remaining = set(dataframes_dict.keys()) - joined_tables
        for _ in range(len(remaining)):
            if not remaining:
                break
            best_rel = None
            for rel in relationships:
                # One side must already be joined, the other must be remaining
                left_in = rel["left_table"] in joined_tables
                right_in = rel["right_table"] in joined_tables
                left_rem = rel["left_table"] in remaining
                right_rem = rel["right_table"] in remaining
                if (left_in and right_rem) or (right_in and left_rem):
                    best_rel = rel
                    break

            if best_rel is None:
                break

            # Determine which side is the new table
            if best_rel["right_table"] in remaining:
                new_table = best_rel["right_table"]
                on_left = best_rel["left_col"]
                on_right = best_rel["right_col"]
            else:
                new_table = best_rel["left_table"]
                on_left = best_rel["right_col"]
                on_right = best_rel["left_col"]

            result = pd.merge(
                result,
                dataframes_dict[new_table],
                left_on=on_left,
                right_on=on_right,
                how="left",
                suffixes=("", f"_{new_table[:20]}"),
            )
            joined_tables.add(new_table)
            remaining.discard(new_table)
            join_log.append(
                f"Joined '{new_table}' on {on_left}={on_right} "
                f"(confidence {best_rel['confidence']}, "
                f"overlap {best_rel['overlap_pct']}%)"
            )

        if remaining:
            join_log.append(
                f"Could not join: {', '.join(remaining)} (no matching keys)"
            )

        return result, join_log

    def summarize(self, relationships):
        """Return a human-readable summary string of detected relationships.

        Args:
            relationships: list from detect()

        Returns:
            str -- multi-line summary
        """
        if not relationships:
            return "No relationships detected between the uploaded tables."

        lines = [f"Detected {len(relationships)} relationship(s):"]
        for r in relationships:
            lines.append(
                f"  {r['left_table']}.{r['left_col']} <-> "
                f"{r['right_table']}.{r['right_col']}  "
                f"({r['match_type']}, confidence={r['confidence']}, "
                f"value overlap={r['overlap_pct']}%)"
            )
        return "\n".join(lines)

    def get_join_diagram(self, relationships):
        """Return an ASCII-art diagram of the relationships.

        Args:
            relationships: list from detect()

        Returns:
            str -- ASCII diagram
        """
        if not relationships:
            return "(no relationships)"

        tables = set()
        for r in relationships:
            tables.add(r["left_table"])
            tables.add(r["right_table"])

        lines = []
        for r in relationships:
            arrow = (
                f"  [{r['left_table']}].{r['left_col']}  "
                f"---({r['confidence']})---  "
                f"[{r['right_table']}].{r['right_col']}"
            )
            lines.append(arrow)

        header = f"Tables: {', '.join(sorted(tables))}"
        return header + "\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_join_candidates(self, left_df, right_df):
        """Find potential join column pairs between two DataFrames.

        Returns list of (left_col, right_col, match_type, confidence).
        """
        candidates = []
        left_cols = list(left_df.columns)
        right_cols = list(right_df.columns)

        # Normalize names for comparison
        left_norm = {c: self._normalize(c) for c in left_cols}
        right_norm = {c: self._normalize(c) for c in right_cols}

        for lc in left_cols:
            ln = left_norm[lc]
            for rc in right_cols:
                rn = right_norm[rc]

                # 1. Exact name match (case-insensitive)
                if ln == rn:
                    conf = 0.9 if self._ID_PATTERNS.search(lc) else 0.7
                    candidates.append((lc, rc, "exact_name", conf))
                    continue

                # 2. ID pattern match (e.g., "CustomerID" vs "CustID")
                if self._ID_PATTERNS.search(lc) and self._ID_PATTERNS.search(rc):
                    ln_base = self._ID_PATTERNS.sub("", ln)
                    rn_base = self._ID_PATTERNS.sub("", rn)
                    if ln_base and rn_base and ln_base == rn_base:
                        candidates.append((lc, rc, "id_pattern", 0.8))
                        continue

                # 3. Abbreviation match
                if self._abbrev_match(ln, rn):
                    candidates.append((lc, rc, "abbreviation", 0.6))

        return candidates

    def _normalize(self, name):
        """Normalize a column name for matching."""
        # Lowercase, strip common delimiters
        n = name.lower().strip()
        n = re.sub(r"[\s_\-\.]+", "", n)
        return n

    def _abbrev_match(self, a, b):
        """Check if a and b are abbreviation variants of each other."""
        for short, full in self._ABBREVS.items():
            if (short in a and full in b) or (full in a and short in b):
                return True
        return False

    def _value_overlap(self, series_a, series_b):
        """Calculate the fraction of values in series_a that also appear in series_b."""
        try:
            a_vals = set(series_a.dropna().unique())
            b_vals = set(series_b.dropna().unique())
            if not a_vals or not b_vals:
                return 0.0
            overlap = a_vals & b_vals
            # Use the smaller set as denominator
            smaller = min(len(a_vals), len(b_vals))
            return len(overlap) / smaller
        except (TypeError, ValueError):
            return 0.0
