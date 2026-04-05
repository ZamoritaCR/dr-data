"""
Sprint 2B -- Synthetic Data Generator (V2).

Two modes:
  QUICK  — Faker-based, ~1s, pattern-matched field generation with trends.
  DEEP   — Claude Opus-powered, ~15s, domain-aware coherent business stories.

Also provides detect_no_data() for determining when a parsed TWB has no
embedded or live data and synthetic generation is needed.
"""

import json
import random
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(0)


class SyntheticDataGenerator:
    """Generates realistic synthetic data from a schema hint."""

    # ------------------------------------------------------------------ #
    #  Field name → Faker expression map                                   #
    # ------------------------------------------------------------------ #
    FIELD_TYPE_MAP = {
        # date / time
        "date": "date_between",
        "year": "year",
        "month": "month_name",
        "quarter": "quarter",
        # geo
        "region": "_pick_weighted_region",
        "country": "country",
        "city": "city",
        "state": "state",
        # money / metrics
        "revenue": "_float_range_10000_500000",
        "sales": "_float_range_1000_100000",
        "profit": "_float_range_neg5000_50000",
        "cost": "_float_range_500_80000",
        "price": "_float_range_10_10000",
        "amount": "_float_range_1000_100000",
        "discount": "_float_range_0_0_35",
        # counts
        "quantity": "_int_range_1_500",
        "count": "_int_range_1_1000",
        "orders": "_int_range_1_200",
        # names / labels
        "customer": "company",
        "product": "catch_phrase",
        "category": "_pick_category",
        "sub_category": "_pick_subcategory",
        "segment": "_pick_segment",
        "name": "name",
        "employee": "name",
        # ids
        "id": "_sequential_id",
        "code": "_bothify_code",
    }

    # Static pools for realistic distributions
    _REGIONS = ["East", "West", "Central", "South", "North"]
    _REGION_WEIGHTS = [0.28, 0.25, 0.18, 0.17, 0.12]
    _CATEGORIES = ["Technology", "Furniture", "Office Supplies", "Electronics", "Clothing"]
    _SUBCATEGORIES = [
        "Phones", "Chairs", "Storage", "Tables", "Accessories",
        "Copiers", "Bookcases", "Appliances", "Binders", "Paper",
    ]
    _SEGMENTS = ["Consumer", "Corporate", "Home Office"]
    _SEGMENT_WEIGHTS = [0.52, 0.30, 0.18]

    # ------------------------------------------------------------------ #
    #  Value generators                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_weighted_region():
        return random.choices(
            SyntheticDataGenerator._REGIONS,
            weights=SyntheticDataGenerator._REGION_WEIGHTS, k=1
        )[0]

    @staticmethod
    def _pick_category():
        return random.choice(SyntheticDataGenerator._CATEGORIES)

    @staticmethod
    def _pick_subcategory():
        return random.choice(SyntheticDataGenerator._SUBCATEGORIES)

    @staticmethod
    def _pick_segment():
        return random.choices(
            SyntheticDataGenerator._SEGMENTS,
            weights=SyntheticDataGenerator._SEGMENT_WEIGHTS, k=1
        )[0]

    @staticmethod
    def _float_range_10000_500000():
        return round(random.uniform(10000, 500000), 2)

    @staticmethod
    def _float_range_1000_100000():
        return round(random.uniform(1000, 100000), 2)

    @staticmethod
    def _float_range_neg5000_50000():
        return round(random.uniform(-5000, 50000), 2)

    @staticmethod
    def _float_range_500_80000():
        return round(random.uniform(500, 80000), 2)

    @staticmethod
    def _float_range_10_10000():
        return round(random.uniform(9.99, 9999.99), 2)

    @staticmethod
    def _float_range_0_0_35():
        return round(random.uniform(0.0, 0.35), 2)

    @staticmethod
    def _int_range_1_500():
        return random.randint(1, 500)

    @staticmethod
    def _int_range_1_1000():
        return random.randint(1, 1000)

    @staticmethod
    def _int_range_1_200():
        return random.randint(1, 200)

    _id_counter = 0

    @classmethod
    def _sequential_id(cls):
        cls._id_counter += 1
        return cls._id_counter

    @staticmethod
    def _bothify_code():
        return fake.bothify(text="??-###").upper()

    # ------------------------------------------------------------------ #
    #  Field name → generator resolution                                   #
    # ------------------------------------------------------------------ #
    def _resolve_generator(self, field_name: str, field_type: str = ""):
        """Match a field name to the best generator function.

        Checks field name against FIELD_TYPE_MAP keys using substring matching.
        Falls back to type-based defaults.
        """
        name_lower = field_name.lower().replace(" ", "_").replace("-", "_")

        # Direct substring match against known patterns
        for pattern, method_name in self.FIELD_TYPE_MAP.items():
            if pattern in name_lower:
                return method_name

        # Fall back to declared type
        type_lower = (field_type or "").lower()
        if type_lower in ("date", "datetime"):
            return "date_between"
        if type_lower in ("float", "real", "double", "decimal", "number"):
            return "_float_range_1000_100000"
        if type_lower in ("int", "integer", "whole"):
            return "_int_range_1_1000"
        if type_lower in ("string", "str", "text"):
            return "word"

        # Ultimate fallback
        return "word"

    def _generate_value(self, method_name: str):
        """Execute a generator by method name."""
        # Check if it's one of our static methods
        if method_name.startswith("_"):
            fn = getattr(self, method_name, None)
            if fn and callable(fn):
                return fn()

        # Otherwise it's a Faker method
        fn = getattr(fake, method_name, None)
        if fn and callable(fn):
            # Special handling for date_between
            if method_name == "date_between":
                return fn(start_date="-2y", end_date="today")
            if method_name == "quarter":
                return f"Q{random.randint(1, 4)}"
            return fn()

        return fake.word()

    # ------------------------------------------------------------------ #
    #  QUICK mode — Faker-based with trend injection                       #
    # ------------------------------------------------------------------ #
    def generate_quick(self, schema_hint: dict, n_rows: int = 100) -> pd.DataFrame:
        """Fast Faker-based generation with trend/seasonality for date+metric combos.

        Args:
            schema_hint: {"fields": [{"name": str, "type": str, "sample_values": list}, ...]}
            n_rows: number of rows to generate.

        Returns:
            pd.DataFrame with generated data.
        """
        fields = schema_hint.get("fields", [])
        if not fields:
            return pd.DataFrame()

        # Reset ID counter
        SyntheticDataGenerator._id_counter = 0

        # Resolve generators for each field
        generators = {}
        for f in fields:
            fname = f.get("name", "unnamed")
            ftype = f.get("type", "")
            sample_vals = f.get("sample_values", [])
            generators[fname] = {
                "method": self._resolve_generator(fname, ftype),
                "sample_values": sample_vals,
            }

        # Generate base data
        data = {f.get("name", "unnamed"): [] for f in fields}
        for _ in range(n_rows):
            for f in fields:
                fname = f.get("name", "unnamed")
                gen_info = generators[fname]

                # If sample_values provided, use them with some variation
                if gen_info["sample_values"]:
                    data[fname].append(random.choice(gen_info["sample_values"]))
                else:
                    data[fname].append(self._generate_value(gen_info["method"]))

        df = pd.DataFrame(data)

        # Inject trend/seasonality for date + metric combos
        df = self._inject_trends(df, fields)

        return df

    def _inject_trends(self, df: pd.DataFrame, fields: list) -> pd.DataFrame:
        """Add upward trend and seasonal variance to metric fields when dates exist."""
        # Find date column
        date_col = None
        for f in fields:
            method = self._resolve_generator(f.get("name", ""), f.get("type", ""))
            if method == "date_between":
                date_col = f.get("name")
                break

        if date_col is None or date_col not in df.columns:
            return df

        # Sort by date for trend injection
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        # Find numeric metric columns (revenue, sales, profit, etc.)
        metric_patterns = {"revenue", "sales", "profit", "cost", "amount", "price"}
        for f in fields:
            fname = f.get("name", "")
            name_lower = fname.lower().replace(" ", "_")
            is_metric = any(p in name_lower for p in metric_patterns)

            if is_metric and fname in df.columns and pd.api.types.is_numeric_dtype(df[fname]):
                n = len(df)
                # Linear growth trend: +30% over the date range
                trend = 1.0 + 0.3 * (pd.Series(range(n)) / max(n - 1, 1))
                # Monthly seasonality: ±10% sine wave
                seasonality = 1.0 + 0.10 * pd.Series(
                    [__import__("math").sin(2 * 3.14159 * i / max(n / 12, 1)) for i in range(n)]
                )
                # Random noise: ±5%
                noise = pd.Series([random.uniform(0.95, 1.05) for _ in range(n)])

                df[fname] = (df[fname] * trend * seasonality * noise).round(2)

        return df

    # ------------------------------------------------------------------ #
    #  DEEP mode — Claude Opus-powered                                     #
    # ------------------------------------------------------------------ #
    def generate_deep(
        self, schema_hint: dict, domain: str, n_rows: int = 100
    ) -> pd.DataFrame:
        """Claude Opus-powered generation with domain-aware coherent business stories.

        Args:
            schema_hint: {"fields": [{"name": str, "type": str}, ...]}
            domain: business domain context (e.g., "retail sales", "healthcare")
            n_rows: number of rows (capped at 200 for token budget)

        Returns:
            pd.DataFrame with generated data.
        """
        import anthropic

        # Cap rows to avoid blowing token budget
        capped_rows = min(n_rows, 200)

        client = anthropic.Anthropic()
        prompt = f"""You are generating synthetic data for a {domain} dashboard.

Schema: {json.dumps(schema_hint, indent=2)}

Generate exactly {capped_rows} rows of realistic {domain} data as a JSON array.

Rules:
- Make it tell a realistic business story (trends, seasonality, realistic distributions)
- Revenue/Sales fields should trend upward over time with monthly variance
- Geographic fields should have realistic weighted distributions (not equal splits)
- Date fields should span the last 24 months
- Numeric fields should have realistic ranges and distributions (not uniform random)
- Category fields should have a power-law distribution (some values more common)
- IDs should be unique and sequential or UUID-style
- Profit should correlate with Revenue but with realistic margins (5-25%)
- Return ONLY the JSON array, no markdown fences, no explanation
- Each row must have all schema fields: {[f["name"] for f in schema_hint.get("fields", [])]}

Return format: [{{"field1": val1, "field2": val2, ...}}, ...]"""

        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=16384,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        data = json.loads(raw_text)
        return pd.DataFrame(data)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                     #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        schema_hint: dict,
        mode: str = "quick",
        domain: str = "business",
        n_rows: int = 100,
    ) -> dict:
        """Generate synthetic data.

        Args:
            schema_hint: {"fields": [{"name", "type", "sample_values"(optional)}]}
            mode: "quick" (Faker) or "deep" (Claude Opus)
            domain: business domain context for deep mode
            n_rows: number of rows to generate

        Returns:
            {
                "dataframe": pd.DataFrame,
                "mode_used": str,
                "row_count": int,
                "preview": list of dicts (first 5 rows),
                "error": None or str,
            }
        """
        result = {
            "dataframe": None,
            "mode_used": mode,
            "row_count": 0,
            "preview": [],
            "error": None,
        }

        try:
            if mode == "deep":
                df = self.generate_deep(schema_hint, domain, n_rows)
            else:
                df = self.generate_quick(schema_hint, n_rows)

            result["dataframe"] = df
            result["row_count"] = len(df)
            result["preview"] = df.head(5).to_dict(orient="records")

        except Exception as e:
            result["error"] = str(e)
            # Fallback: if deep fails, try quick
            if mode == "deep":
                try:
                    df = self.generate_quick(schema_hint, n_rows)
                    result["dataframe"] = df
                    result["row_count"] = len(df)
                    result["preview"] = df.head(5).to_dict(orient="records")
                    result["mode_used"] = "quick (fallback)"
                    result["error"] = f"Deep mode failed ({e}), fell back to quick"
                except Exception as e2:
                    result["error"] = f"Both modes failed: deep={e}, quick={e2}"

        return result


# ------------------------------------------------------------------ #
#  TWB no-data detection                                               #
# ------------------------------------------------------------------ #
def detect_no_data(twb_spec: dict) -> bool:
    """Returns True if the parsed TWB spec has no embedded or live data.

    Checks for hyper extracts, data-engine connections, and known live
    connection types (postgres, mysql, snowflake, etc.).
    """
    has_hyper = twb_spec.get("has_hyper", False)
    datasources = twb_spec.get("datasources", [])

    if not datasources:
        return True

    for ds in datasources:
        conn_type = ds.get("connection_type", "")
        # Embedded extract present
        if conn_type in ("hyper", "dataengine", "federated") and has_hyper:
            return False
        # Live connection to a database
        if conn_type in (
            "postgres", "mysql", "snowflake", "sqlserver", "bigquery",
            "redshift", "oracle", "databricks", "azure_sql",
        ):
            return False
        # Excel/CSV file reference with a filename
        if conn_type in ("excel-direct", "textscan") and ds.get("filename"):
            return False

    return True


# ------------------------------------------------------------------ #
#  Self-test                                                            #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    schema = {
        "fields": [
            {"name": "Date", "type": "date"},
            {"name": "Revenue", "type": "float"},
            {"name": "Region", "type": "string"},
        ]
    }
    gen = SyntheticDataGenerator()
    result = gen.generate(schema, mode="quick", n_rows=20)

    passed = result["dataframe"] is not None and len(result["dataframe"]) == 20
    print(
        "SYNTHETIC GEN TEST:",
        "PASS" if passed else "FAIL",
    )
    print(result["preview"][:2])

    # Verify trends: Revenue should generally increase over sorted dates
    df = result["dataframe"]
    first_half = df["Revenue"].iloc[: len(df) // 2].mean()
    second_half = df["Revenue"].iloc[len(df) // 2 :].mean()
    trend_ok = second_half > first_half
    print(f"TREND TEST: {'PASS' if trend_ok else 'FAIL'} (first_half_avg={first_half:.0f}, second_half_avg={second_half:.0f})")

    # Verify detect_no_data
    spec_no_data = {"has_hyper": False, "datasources": [{"connection_type": "federated"}]}
    spec_has_data = {"has_hyper": True, "datasources": [{"connection_type": "hyper"}]}
    detect_ok = detect_no_data(spec_no_data) is True and detect_no_data(spec_has_data) is False
    print(f"DETECT_NO_DATA TEST: {'PASS' if detect_ok else 'FAIL'}")
