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
    "branch": ["Branch Alpha", "Branch Beta", "Branch Gamma", "Branch Delta",
               "Branch Epsilon", "Branch Zeta", "Branch Eta", "Branch Theta",
               "Branch Iota", "Branch Kappa"],
    "store": ["Store A", "Store B", "Store C", "Store D", "Store E",
              "Store F", "Store G", "Store H"],
    "office": ["Head Office", "Regional Office North", "Regional Office South",
               "East Office", "West Office", "Central Office"],
    "category": ["Technology", "Furniture", "Office Supplies"],
    "sub-category": ["Phones", "Chairs", "Storage", "Tables", "Accessories",
                     "Copiers", "Bookcases", "Appliances", "Binders", "Paper"],
    "segment": ["Consumer", "Corporate", "Home Office"],
    "ship mode": ["Standard Class", "Second Class", "First Class", "Same Day"],
    "status": ["Won", "Lost", "Open", "Pending", "Closed"],
    "salestatus": ["Sold", "Reserved", "Available", "Cancelled", "Pending"],
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
    "saleschannel": ["Direct", "Online", "Dealer", "Distributor", "Partner"],
    "paymentmethod": ["Cash", "Credit Card", "Debit Card", "Bank Transfer", "Installment"],
    "customersegment": ["Individual", "SME", "Corporate", "Government", "Education"],
    "carmodel": ["Model A", "Model B", "Model C", "Model D", "Model E",
                 "Model F", "Model G", "Model H", "Model X", "Model Y"],
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
    # IMPORTANT: Use RAW column name (col["name"]) as the primary key, NOT caption.
    # The Tableau shelf expressions always reference the RAW name (e.g. sum:NetSales:qk).
    # The TMDL and CSV must use the same raw names so Power BI field refs resolve.
    for ds in tableau_spec.get("datasources", []):
        for col in ds.get("columns", []):
            raw_name = col.get("name", "") or col.get("caption", "")
            # Skip internal Tableau fields (start with [ or : indicates internal ref)
            if not raw_name or raw_name.startswith(":") or raw_name.startswith("["):
                continue
            # Skip Tableau-generated internal object ID columns
            if "__tableau_internal_object_id__" in raw_name:
                continue
            if raw_name == "Number of Records":
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

    # 4. Add calculated fields BY CAPTION NAME as actual columns.
    # These are the fields that visuals reference after calc_id_map resolution.
    for cf in tableau_spec.get("calculated_fields", []):
        caption = cf.get("name", "")
        if not caption or caption in columns:
            continue
        # Infer data type from formula or explicit datatype
        dt = cf.get("datatype", "")
        role_hint = cf.get("role", "")
        if dt in ("real", "integer", "float"):
            dtype = "real"
            role = "measure"
        elif dt in ("date", "datetime"):
            dtype = "date"
            role = "dimension"
        elif dt == "boolean":
            dtype = "boolean"
            role = "dimension"
        elif role_hint == "measure":
            dtype = "real"
            role = "measure"
        else:
            # Guess from formula: if it has SUM/AVG/COUNT it's numeric
            formula = cf.get("formula", "")
            if agg_pattern.search(formula) or any(kw in formula.upper()
                    for kw in ("SUM(", "AVG(", "COUNT(", "MIN(", "MAX(")):
                dtype = "real"
                role = "measure"
            elif "IF " in formula.upper() and ("THEN" in formula.upper()):
                # IF/THEN often returns string categories
                dtype = "string"
                role = "dimension"
            else:
                dtype = "real"
                role = "measure"
        columns[caption] = {
            "name": caption,
            "datatype": dtype,
            "role": role,
            "source": "calculated_field_caption",
        }

    # Filter out Tableau internal / noise columns
    _internal_prefixes = (
        "Action (", "Tooltip (", "__tableau", "usr:", "pcto:", "cnt:",
        "twk:", "tmn:", "yr:", "mn:", "qr:", "tqr:",
        "federated.", "Multiple Values",
    )
    # NOTE: Calculation_ prefix is NO LONGER filtered.
    # Internal Calculation_XXX IDs are resolved to captions via calc_id_map.
    _internal_suffixes = (
        "(generated)",  # Tableau auto-generated fields like Latitude/Longitude
    )
    cleaned = []
    for col in columns.values():
        name = col["name"]
        if any(name.startswith(p) for p in _internal_prefixes):
            continue
        if any(name.endswith(s) for s in _internal_suffixes):
            continue
        if len(name) > 80:  # Very long names are usually internal
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
            # Date column: last 2 years relative to now (date-only, no time)
            end = datetime.now().date()
            start = end.replace(year=end.year - 2)
            delta = (end - start).days
            dates = [start + timedelta(days=int(rng.integers(0, delta)))
                     for _ in range(num_rows)]
            data[name] = pd.to_datetime(dates)

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


def extract_twbx_embedded_data(
    twbx_path: str,
    output_dir: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Extract real embedded data from a .twbx ZIP archive.

    A .twbx is a ZIP file containing a .twb (XML) and optionally:
    - Data/Extracts/*.hyper  — Tableau Hyper extract (most common)
    - *.csv / *.xlsx / *.xls — raw data files

    Returns:
        (DataFrame, csv_path) if data found, (None, "") otherwise.
    """
    import zipfile

    if not twbx_path or not os.path.exists(twbx_path):
        return None, ""

    try:
        zf = zipfile.ZipFile(twbx_path)
    except Exception:
        return None, ""

    names = zf.namelist()

    # --- Try CSV/Excel files first (no extra deps) ---
    for name in names:
        lower = name.lower()
        if lower.endswith(".csv") and not lower.endswith(".twb"):
            try:
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                if df is not None and len(df) > 0:
                    csv_path = _save_extracted(df, twbx_path, output_dir)
                    print(f"[TWBX] Extracted CSV '{name}': {df.shape}")
                    return df, csv_path
            except Exception as e:
                print(f"[TWBX] CSV read failed for '{name}': {e}")

    for name in names:
        lower = name.lower()
        if lower.endswith(".xlsx") or lower.endswith(".xls"):
            try:
                import tempfile
                tmp = tempfile.mkdtemp()
                extracted = zf.extract(name, tmp)
                df = pd.read_excel(extracted)
                if df is not None and len(df) > 0:
                    csv_path = _save_extracted(df, twbx_path, output_dir)
                    print(f"[TWBX] Extracted Excel '{name}': {df.shape}")
                    return df, csv_path
            except Exception as e:
                print(f"[TWBX] Excel read failed for '{name}': {e}")

    # --- Try Hyper files (requires tableauhyperapi) ---
    hyper_names = [n for n in names if n.lower().endswith(".hyper")]
    if hyper_names:
        try:
            import tempfile
            from tableauhyperapi import (
                HyperProcess, Telemetry, Connection,
            )
            tmp = tempfile.mkdtemp()
            hyper_path = zf.extract(hyper_names[0], tmp)

            with HyperProcess(
                telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU
            ) as hyper:
                with Connection(hyper.endpoint, hyper_path) as conn:
                    df = _hyper_to_dataframe(conn)
                    if df is not None and len(df) > 0:
                        csv_path = _save_extracted(df, twbx_path, output_dir)
                        print(f"[TWBX] Extracted Hyper '{hyper_names[0]}': {df.shape}")
                        return df, csv_path
        except ImportError:
            print("[TWBX] tableauhyperapi not installed; cannot read .hyper file")
        except Exception as e:
            print(f"[TWBX] Hyper read failed: {e}")

    return None, ""


def _hyper_to_dataframe(conn) -> Optional[pd.DataFrame]:
    """Read the first non-empty table from a HyperProcess Connection."""
    try:
        schemas = conn.catalog.get_schema_names()
        # Prefer 'Extract' schema, fall back to others
        ordered = sorted(schemas, key=lambda s: (str(s) != "Extract", str(s)))
        for schema in ordered:
            tables = conn.catalog.get_table_names(schema)
            for tbl in tables:
                try:
                    count = conn.execute_scalar_query(f"SELECT COUNT(*) FROM {tbl}")
                    if not count:
                        continue
                    rows = []
                    with conn.execute_query(f"SELECT * FROM {tbl}") as result:
                        cols = [c.name.unescaped for c in result.schema.columns]
                        for row in result:
                            rows.append(list(row))
                    if rows:
                        df = pd.DataFrame(rows, columns=cols)
                        # Convert Tableau Date objects to Python date strings
                        for col in df.columns:
                            sample = df[col].dropna().head(1)
                            if not sample.empty:
                                val = sample.iloc[0]
                                if hasattr(val, "year") and hasattr(val, "month"):
                                    df[col] = df[col].apply(
                                        lambda v: f"{v.year}-{v.month:02d}-{v.day:02d}"
                                        if v is not None else None
                                    )
                        return df
                except Exception:
                    continue
    except Exception as e:
        print(f"[TWBX] _hyper_to_dataframe error: {e}")
    return None


def _save_extracted(
    df: pd.DataFrame, twbx_path: str, output_dir: Optional[str]
) -> str:
    """Save DataFrame to CSV alongside the TWBX or in output_dir."""
    if not output_dir:
        output_dir = os.path.dirname(twbx_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(twbx_path))[0]
    base = base.replace(" ", "_") + "_extracted.csv"
    csv_path = os.path.join(output_dir, base)
    df.to_csv(csv_path, index=False)
    return csv_path


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

    # --- ADD CALCULATED FIELD COLUMNS ---
    # Visual bindings reference calc fields by name. Without these columns,
    # PBI visuals can't bind and fall back to defaults.
    calc_fields = tableau_spec.get("calculated_fields", [])
    if calc_fields:
        import numpy as np
        for cf in calc_fields:
            col_name = cf.get("caption") or cf.get("name", "")
            if not col_name or col_name in df.columns:
                continue
            formula = cf.get("formula", "").lower()
            # Infer data type from formula content
            if any(kw in formula for kw in ["sum(", "avg(", "count(", "min(", "max(", "ratio", "profit", "sales", "budget", "rank"]):
                # Numeric measure
                df[col_name] = np.random.uniform(0, 10000, size=len(df)).round(2)
            elif any(kw in formula for kw in ["datepart", "date", "day", "month", "year"]):
                # Numeric (date part)
                df[col_name] = np.random.randint(1, 365, size=len(df))
            elif any(kw in formula for kw in ["if ", "case", "elseif", "then"]):
                # Categorical from conditional
                df[col_name] = np.random.choice(["Category A", "Category B", "Category C", "Category D"], size=len(df))
            elif any(kw in formula for kw in ["left(", "right(", "mid(", "upper(", "lower(", "str("]):
                # String manipulation
                df[col_name] = [f"Val_{i}" for i in range(len(df))]
            elif any(kw in formula for kw in ["int(", "round(", "floor(", "ceiling("]):
                # Integer
                df[col_name] = np.random.randint(0, 100, size=len(df))
            elif "window" in formula or "running" in formula or "index" in formula:
                # Window function — sequential numeric
                df[col_name] = np.arange(len(df), dtype=float)
            else:
                # Default to numeric
                df[col_name] = np.random.uniform(0, 1000, size=len(df)).round(2)
        # Calc field columns added: {sum(1 for cf in calc_fields if (cf.get('caption') or cf.get('name','')) in df.columns)}

    return df, csv_path, schema
