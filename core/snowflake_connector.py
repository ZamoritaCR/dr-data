"""
Snowflake Connector - Direct Snowflake warehouse integration.

Provides connection management, schema discovery, query execution,
and table profiling against Snowflake data warehouses.
"""

import os
import re

try:
    import snowflake.connector as sf_connector
    _HAS_SNOWFLAKE = True
except ImportError:
    _HAS_SNOWFLAKE = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


class SnowflakeConnector:
    """Direct Snowflake warehouse connector for querying and profiling."""

    def __init__(self):
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        self.conn = None

    def connect(self, account=None, user=None, password=None,
                warehouse=None, database=None, schema=None):
        """Open a Snowflake connection. Params override env vars."""
        if not _HAS_SNOWFLAKE:
            print("[SNOWFLAKE] snowflake-connector-python not installed.")
            return False

        acct = account or self.account
        usr = user or self.user
        pwd = password or self.password
        wh = warehouse or self.warehouse
        db = database or self.database
        sch = schema or self.schema

        try:
            self.conn = sf_connector.connect(
                account=acct, user=usr, password=pwd,
                warehouse=wh, database=db, schema=sch,
            )
            print(f"[SNOWFLAKE] Connected to {db}.{sch}")
            return True
        except Exception as e:
            print(f"[SNOWFLAKE] Connection failed: {e}")
            self.conn = None
            return False

    def list_databases(self):
        """Return list of database names."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return []
        try:
            cur = self.conn.cursor()
            cur.execute("SHOW DATABASES")
            rows = cur.fetchall()
            return [r[1] for r in rows]
        except Exception as e:
            print(f"[SNOWFLAKE] list_databases failed: {e}")
            return []

    def list_schemas(self, database=None):
        """Return list of schema names in the given database."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return []
        db = database or self.database
        try:
            cur = self.conn.cursor()
            cur.execute(f"SHOW SCHEMAS IN DATABASE {db}")
            rows = cur.fetchall()
            return [r[1] for r in rows]
        except Exception as e:
            print(f"[SNOWFLAKE] list_schemas failed: {e}")
            return []

    def list_tables(self, database=None, schema=None):
        """Return list of table names in the given database.schema."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return []
        db = database or self.database
        sch = schema or self.schema
        try:
            cur = self.conn.cursor()
            cur.execute(f"SHOW TABLES IN {db}.{sch}")
            rows = cur.fetchall()
            return [r[1] for r in rows]
        except Exception as e:
            print(f"[SNOWFLAKE] list_tables failed: {e}")
            return []

    def query_to_df(self, sql, limit=50000):
        """Execute SQL and return a pandas DataFrame."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return None
        if not _HAS_PANDAS:
            print("[SNOWFLAKE] pandas not installed.")
            return None
        try:
            clean = sql.strip().rstrip(";")
            if not re.search(r"\bLIMIT\b", clean, re.IGNORECASE):
                clean += f" LIMIT {limit}"
            cur = self.conn.cursor()
            cur.execute(clean)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            print(f"[SNOWFLAKE] Query returned {len(rows)} rows")
            df = pd.DataFrame(rows, columns=cols)
            # Fix Snowflake type mismatches that confuse pandas/PBI:
            #  - Decimal -> float64  (NUMBER columns with scale > 0)
            #  - datetime.date -> datetime64  (DATE columns)
            #  - object columns that are all-numeric -> float64
            import datetime as _dt
            from decimal import Decimal as _Dec
            for c in df.columns:
                sample = df[c].dropna().head(20)
                if sample.empty:
                    continue
                first_type = type(sample.iloc[0])
                if first_type is _Dec:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                elif first_type is _dt.date:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                elif df[c].dtype == object:
                    # Last resort: try numeric coercion on object columns
                    converted = pd.to_numeric(df[c], errors="coerce")
                    if converted.notna().sum() > sample.notna().sum() * 0.8:
                        df[c] = converted
            return df
        except Exception as e:
            print(f"[SNOWFLAKE] query_to_df failed: {e}")
            return None

    def table_to_df(self, table_name, limit=50000):
        """Load a full table into a DataFrame with a row limit."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return None
        try:
            sql = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.query_to_df(sql, limit=limit)
        except Exception as e:
            print(f"[SNOWFLAKE] table_to_df failed: {e}")
            return None

    def get_table_profile(self, table_name):
        """Return a dict with row_count, column_info, and sample rows."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return None
        try:
            profile = {}
            cur = self.conn.cursor()

            # Row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            profile["row_count"] = cur.fetchone()[0]

            # Column info
            cur.execute(f"DESCRIBE TABLE {table_name}")
            profile["column_info"] = [
                {"name": r[0], "type": r[1], "nullable": r[3]}
                for r in cur.fetchall()
            ]

            # Sample
            sample_df = self.query_to_df(
                f"SELECT * FROM {table_name} LIMIT 5", limit=5
            )
            profile["sample"] = (
                sample_df.to_dict(orient="records") if sample_df is not None else []
            )

            print(f"[SNOWFLAKE] Profiled {table_name}: "
                  f"{profile['row_count']} rows, "
                  f"{len(profile['column_info'])} columns")
            return profile

        except Exception as e:
            print(f"[SNOWFLAKE] get_table_profile failed: {e}")
            return None

    def get_sample_data_tables(self):
        """List tables in SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 with row counts."""
        if not self.conn:
            print("[SNOWFLAKE] Not connected.")
            return {}

        expected = [
            "CUSTOMER", "LINEITEM", "NATION", "ORDERS",
            "PART", "PARTSUPP", "REGION", "SUPPLIER",
        ]
        result = {}
        try:
            cur = self.conn.cursor()
            for tbl in expected:
                fqn = f"SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.{tbl}"
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {fqn}")
                    result[tbl] = cur.fetchone()[0]
                except Exception:
                    result[tbl] = None
            print(f"[SNOWFLAKE] Sample data: {len(result)} tables found")
            return result
        except Exception as e:
            print(f"[SNOWFLAKE] get_sample_data_tables failed: {e}")
            return {}

    def disconnect(self):
        """Close the Snowflake connection safely."""
        if self.conn:
            try:
                self.conn.close()
                print("[SNOWFLAKE] Disconnected.")
            except Exception as e:
                print(f"[SNOWFLAKE] Disconnect error: {e}")
            finally:
                self.conn = None
