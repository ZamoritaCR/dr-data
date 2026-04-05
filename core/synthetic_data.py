"""
Synthetic Data Generator -- creates realistic fake data from Tableau structure.

When a .twbx has no extractable data (e.g., .hyper format), this module
reads the Tableau spec (datasources, columns, worksheets, calculated fields)
and generates a DataFrame with the right schema, types, and distributions.
"""

import re
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# -- Realistic value pools for common business dimensions --

_DIMENSION_POOLS = {
    "region": ["East", "West", "Central", "South"],
    "state": ["California", "New York", "Texas", "Florida", "Illinois",
              "Pennsylvania", "Ohio", "Georgia", "Washington", "Virginia"],
    "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
             "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"],
    "country": ["United States", "Canada", "United Kingdom", "Germany", "France"],
    "category": ["Technology", "Furniture", "Office Supplies"],
    "sub-category": ["Phones", "Chairs", "Storage", "Tables", "Accessories",
                     "Copiers", "Bookcases", "Appliances", "Binders", "Paper"],
    "segment": ["Consumer", "Corporate", "Home Office"],
    "ship mode": ["Standard Class", "Second Class", "First Class", "Same Day"],
    "status": ["Won", "Lost", "Open", "Pending", "Closed"],
    "stage": ["Lead", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"],
    "priority": ["Low", "Medium", "High", "Critical"],
    "department": ["Sales", "Marketing", "Engineering", "Finance", "Operations"],
    "product": ["Widget A", "Widget B", "Service X", "Service Y", "Premium Z"],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "month": ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"],
    "year": ["2022", "2023", "2024", "2025"],
    "type": ["Type A", "Type B", "Type C"],
    "channel": ["Direct", "Online", "Partner", "Retail"],
}

_NAME_POOLS = {
    "customer": [f"Customer {i}" for i in range(1, 201)],
    "person": [f"Person {i}" for i in range(1, 51)],
    "sales person": [f"Rep {chr(65+i)}" for i in range(20)],
    "salesperson": [f"Rep {chr(65+i)}" for i in range(20)],
    "manager": [f"Manager {chr(65+i)}" for i in range(10)],
    "product name": [f"Product {i:03d}" for i in range(1, 101)],
    "order id": None,  # generated as sequence
    "id": None,
}

# Measures: typical range by keyword
_MEASURE_RANGES = {
    "sales": (10, 50000),
    "revenue": (100, 100000),
    "profit": (-5000, 20000),
    "quantity": (1, 50),
    "discount": (0.0, 0.5),
    "cost": (5, 30000),
    "price": (1, 5000),
    "amount": (10, 50000),
    "count": (1, 100),
    "score": (0, 100),
    "rate": (0.0, 1.0),
    "ratio": (0.0, 2.0),
    "days": (1, 365),
    "quota": (10000, 500000),
    "target": (10000, 500000),
    "commission": (100, 50000),
    "salary": (30000, 200000),
    "budget": (1000, 100000),
    "weight": (0.1, 100.0),
    "percentage": (0.0, 100.0),
}


def _guess_dimension_pool(col_name: str) -> Optional[list]:
    """Find the best matching value pool for a dimension column."""
    name_lower = col_name.lower().strip()

    # Direct match
    for key, pool in _DIMENSION_POOLS.items():
        if key in name_lower:
            return pool

    # Name-type columns
    for key, pool in _NAME_POOLS.items():
        if key in name_lower:
            return pool

    return None


def _guess_measure_range(col_name: str) -> Tuple[float, float]:
    """Find a reasonable numeric range for a measure column."""
    name_lower = col_name.lower().strip()

    for key, (lo, hi) in _MEASURE_RANGES.items():
        if key in name_lower:
            return (lo, hi)

    # Default range
    return (0, 10000)


def _is_date_column(col_name: str, datatype: str = "") -> bool:
    """Check if a column is likely a date based on name/type."""
    if datatype in ("date", "datetime"):
        return True
    date_hints = ["date", "time", "timestamp", "created", "updated", "day", "month"]
    name_lower = col_name.lower()
    return any(h in name_lower for h in date_hints)


def _is_id_column(col_name: str) -> bool:
    """Check if a column is likely an ID/key."""
    name_lower = col_name.lower().strip()
    id_hints = ["order id", "row id", "product id", "customer id", "id"]
    return any(h in name_lower for h in id_hints)


def _clean_field_name(name: str) -> str:
    """Strip surrounding quotes and whitespace from a Tableau field name.

    Tableau shelf expressions can embed quoted field names like:
        [none:"Region":nk]  ->  after split on ':', parts[1] is '"Region"'
    This strips those outer quotes so the column name is clean.
    Returns empty string for names that are only punctuation/whitespace.
    """
    name = name.strip()
    if name.startswith('"') and name.endswith('"') and len(name) >= 2:
        name = name[1:-1].strip()
    if name.startswith("'") and name.endswith("'") and len(name) >= 2:
        name = name[1:-1].strip()
    # Reject names that are only punctuation or whitespace
    if not name or not re.search(r'[a-zA-Z0-9]', name):
        return ""
    return name


def extract_schema_from_tableau(tableau_spec: dict) -> List[dict]:
    """Extract column schema from a Tableau spec.

    Combines information from:
    - datasource columns (explicit definitions with name, datatype, role)
    - worksheet shelf references (field names used in rows/cols/filters)
    - calculated field references (field names in formulas)

    Returns a list of column dicts: {name, datatype, role, source}
    """
    columns = {}  # name -> {datatype, role, source}

    # 1. Explicit columns from datasources
    for ds in tableau_spec.get("datasources", []):
        for col in ds.get("columns", []):
            raw_name = col.get("caption") or col.get("name", "")
            # Skip internal Tableau fields
            if raw_name.startswith(":") or raw_name.startswith("["):
                continue
            if not raw_name or raw_name == "Number of Records":
                continue

            name = _clean_field_name(raw_name)
            if not name:
                continue
            if name not in columns:
                columns[name] = {
                    "name": name,
                    "datatype": col.get("datatype", "string"),
                    "role": col.get("role", "dimension"),
                    "source": ds.get("caption", ds.get("name", "")),
                }

    # 2. Field references from worksheet shelves
    field_ref_pattern = re.compile(r'\[([^\]]+)\]')
    # Aggregation prefixes that indicate a measure
    _agg_prefixes = {"sum", "avg", "count", "min", "max", "countd", "median"}
    # All known Tableau shelf prefixes (aggregation + dimension qualifiers).
    # Format: prefix:FieldName:suffix  (e.g. sum:Sales:qk, none:Region:nk,
    # tqr:Date:qk, yr:Date:ok, mn:Date:ok, qr:Date:ok, tmn:Date:ok, etc.)
    _all_shelf_prefix_pattern = re.compile(
        r'^(sum|avg|count|min|max|countd|median|attr|'
        r'none|usr|yr|mn|qr|tqr|tmn|twk|day|'
        r'mdy|wk|md|qy|my)\:(.+?)(?:\:[a-z]+)?$',
        re.IGNORECASE,
    )
    agg_pattern = re.compile(r'(SUM|AVG|COUNT|MIN|MAX|COUNTD|ATTR)\s*\(', re.IGNORECASE)

    for ws in tableau_spec.get("worksheets", []):
        for shelf_key in ("rows", "cols"):
            shelf_text = ws.get(shelf_key, "")
            if not shelf_text:
                continue
            refs = field_ref_pattern.findall(shelf_text)
            for ref in refs:
                # Skip datasource IDs and internal refs
                if ref.startswith("federated.") or ref.startswith(":"):
                    continue
                # Parse Tableau shelf prefix:FieldName:suffix pattern
                role = "dimension"
                m = _all_shelf_prefix_pattern.match(ref)
                if m:
                    prefix = m.group(1).lower()
                    ref = m.group(2)
                    if prefix in _agg_prefixes:
                        role = "measure"
                elif ":" in ref:
                    # Unknown prefix:value:suffix -- extract middle part
                    parts = ref.split(":")
                    if len(parts) >= 2:
                        ref = parts[1]

                name = _clean_field_name(ref)
                if name and name not in columns:
                    columns[name] = {
                        "name": name,
                        "datatype": "real" if role == "measure" else "string",
                        "role": role,
                        "source": "shelf_reference",
                    }

        # Filters (handles both legacy string filters and dict filters
        # from the enhanced parser)
        for filt in ws.get("filters", []):
            if isinstance(filt, dict):
                name = _clean_field_name(
                    filt.get("field") or filt.get("column_ref") or ""
                )
                if name and name not in columns:
                    columns[name] = {
                        "name": name,
                        "datatype": "string",
                        "role": "dimension",
                        "source": "filter_reference",
                    }
            elif isinstance(filt, str):
                refs = field_ref_pattern.findall(filt)
                for ref in refs:
                    if ref.startswith("federated.") or ref.startswith(":"):
                        continue
                    parts = ref.split(":")
                    raw = parts[1] if len(parts) >= 2 else ref
                    name = _clean_field_name(raw)
                    if name and name not in columns:
                        columns[name] = {
                            "name": name,
                            "datatype": "string",
                            "role": "dimension",
                            "source": "filter_reference",
                        }

    # 3. Field references from calculated fields
    for cf in tableau_spec.get("calculated_fields", []):
        formula = cf.get("formula", "")
        refs = field_ref_pattern.findall(formula)
        for ref in refs:
            if ref.startswith("Parameters.") or ref.startswith(":"):
                continue
            name = _clean_field_name(ref)
            if name and name not in columns:
                # If used in an aggregation, it's likely a measure
                role = "measure" if agg_pattern.search(formula) else "dimension"
                columns[name] = {
                    "name": name,
                    "datatype": "real" if role == "measure" else "string",
                    "role": role,
                    "source": "calculated_field",
                }

    # Filter out Tableau internal / noise columns
    _internal_prefixes = (
        "Action (", "Tooltip (", "__tableau", "usr:", "pcto:", "cnt:",
        "twk:", "tmn:", "yr:", "mn:", "qr:", "tqr:", "Calculation_",
        "federated.", "Multiple Values",
    )
    _internal_suffixes = (
        "(generated)",  # Tableau auto-generated fields like Latitude/Longitude
        "(copy)",
        "(bin)",
    )
    cleaned = []
    for col in columns.values():
        name = col["name"]
        if any(name.startswith(p) for p in _internal_prefixes):
            continue
        if any(name.endswith(s) for s in _internal_suffixes):
            continue
        if len(name) > 60:  # Overly long names are usually internal
            continue
        if len(name) < 2 or not re.search(r'[a-zA-Z0-9]', name):
            continue
        cleaned.append(col)

    return cleaned


def generate_synthetic_dataframe(
    schema: List[dict],
    num_rows: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame from a schema.

    Args:
        schema: list of column dicts from extract_schema_from_tableau()
        num_rows: number of rows to generate
        seed: random seed for reproducibility

    Returns:
        pd.DataFrame with synthetic data matching the schema
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    data = {}

    for col in schema:
        name = col["name"]
        datatype = col.get("datatype", "string")
        role = col.get("role", "dimension")

        if _is_date_column(name, datatype):
            # Date column: last 2 years relative to now
            end = datetime.now()
            start = end.replace(year=end.year - 2)
            delta = (end - start).days
            dates = [start + timedelta(days=int(rng.integers(0, delta)))
                     for _ in range(num_rows)]
            data[name] = dates

        elif _is_id_column(name):
            # ID column: sequential
            prefix = name.replace(" ", "-").upper()[:3]
            data[name] = [f"{prefix}-{i+1:05d}" for i in range(num_rows)]

        elif role == "measure" or datatype in ("real", "integer", "float", "double"):
            lo, hi = _guess_measure_range(name)
            if datatype == "integer" or (isinstance(lo, int) and isinstance(hi, int) and hi - lo > 5):
                if lo >= 0 and hi <= 1:
                    data[name] = rng.uniform(lo, hi, num_rows).round(2)
                else:
                    data[name] = rng.integers(int(lo), int(hi) + 1, num_rows)
            else:
                data[name] = rng.uniform(lo, hi, num_rows).round(2)

        else:
            # Dimension: try to find a pool
            pool = _guess_dimension_pool(name)
            if pool is None:
                # Generate generic categories
                n_cats = min(20, max(3, num_rows // 100))
                pool = [f"{name} {i+1}" for i in range(n_cats)]

            if name.lower().endswith("name") or "customer" in name.lower():
                # Higher cardinality for name fields
                n_vals = min(len(pool), max(50, num_rows // 20))
                pool = pool[:n_vals] if len(pool) >= n_vals else pool
            data[name] = [random.choice(pool) for _ in range(num_rows)]

    df = pd.DataFrame(data)
    return df


def generate_from_tableau_spec(
    tableau_spec: dict,
    num_rows: int = 2000,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, List[dict]]:
    """Full pipeline: extract schema from Tableau spec, generate data, optionally save.

    Args:
        tableau_spec: parsed Tableau spec dict
        num_rows: rows to generate
        output_dir: if provided, saves CSV to this directory

    Returns:
        (DataFrame, csv_path_or_empty, schema)
    """
    schema = extract_schema_from_tableau(tableau_spec)

    if not schema:
        # Fallback: minimal schema from worksheet names
        schema = [
            {"name": "Category", "datatype": "string", "role": "dimension"},
            {"name": "Value", "datatype": "real", "role": "measure"},
            {"name": "Date", "datatype": "date", "role": "dimension"},
        ]

    df = generate_synthetic_dataframe(schema, num_rows=num_rows)

    csv_path = ""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "synthetic_tableau_data.csv")
        df.to_csv(csv_path, index=False)

    return df, csv_path, schema
