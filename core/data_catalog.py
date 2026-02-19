"""
Enterprise Data Catalog & Business Glossary -- DAMA-DMBOK compliant.

Provides table/column registration, auto-cataloging from DataFrames,
business glossary generation, data classification, stewardship, and
domain management.  Persists to JSON for portability.
"""
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd


# ── PII / classification hints ─────────────────────────────────────── #

_PII_HINTS = {
    "name", "email", "phone", "address", "ssn", "dob", "birth",
    "passport", "license", "social_security",
}
_FINANCIAL_HINTS = {
    "amount", "balance", "price", "cost", "revenue", "salary",
    "fee", "payment", "transaction",
}
_REFERENCE_HINTS = {"code", "type", "status", "category", "flag"}
_IDENTIFIER_HINTS = {"_id", "_key", "_pk", "_fk", "uuid"}

# ── Prefix-to-domain mapping (Snowflake TPC-H style) ───────────────── #

_PREFIX_DOMAIN = {
    "c_": "Customer", "o_": "Transaction", "l_": "Transaction",
    "s_": "Operations", "p_": "Operations", "ps_": "Operations",
    "n_": "Reference", "r_": "Reference",
}

_DEFAULT_CLASSIFICATIONS = {
    "PII": {
        "description": "Personally Identifiable Information",
        "sensitivity_level": "Restricted",
        "handling_rules": [
            "Encrypt at rest", "Mask in non-prod", "Log all access"],
    },
    "Financial": {
        "description": "Financial data including transactions and balances",
        "sensitivity_level": "Confidential",
        "handling_rules": ["Audit trail required", "SOX compliance"],
    },
    "Reference": {
        "description": "Reference and lookup data",
        "sensitivity_level": "Internal",
        "handling_rules": [
            "Cache-friendly", "Change management required"],
    },
    "Identifier": {
        "description": "System identifiers and keys",
        "sensitivity_level": "Internal",
        "handling_rules": ["Do not expose externally"],
    },
    "General": {
        "description": "General business data",
        "sensitivity_level": "Internal",
        "handling_rules": [],
    },
}

_DEFAULT_DOMAINS = {
    "Customer": "Customer master data and profiles",
    "Transaction": "Payment and money transfer transactions",
    "Compliance": "KYC, AML, and regulatory data",
    "Operations": "Agent network and operational data",
    "Financial": "Revenue, pricing, and financial reporting",
    "Reference": "Lookup tables, country codes, currencies",
}


class DataCatalog:
    """Enterprise data catalog with glossary, classification, and lineage."""

    def __init__(self, catalog_path="data_catalog.json"):
        self.catalog_path = catalog_path
        self.catalog = self._load_catalog()

    # ── Persistence ──

    def _load_catalog(self):
        try:
            if os.path.exists(self.catalog_path):
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[CATALOG] Loaded catalog from {self.catalog_path}")
                return data
        except Exception as e:
            print(f"[CATALOG] Could not load catalog: {e}")

        print("[CATALOG] Creating new catalog")
        now = datetime.now(timezone.utc).isoformat()
        cat = {
            "version": "1.0",
            "created_at": now,
            "updated_at": now,
            "organization": "Organization",
            "tables": {},
            "glossary": {},
            "domains": {},
            "stewards": {},
            "classifications": {},
            "tags": {},
        }
        # Pre-populate default classifications
        for name, info in _DEFAULT_CLASSIFICATIONS.items():
            cat["classifications"][name] = {
                "description": info["description"],
                "sensitivity_level": info["sensitivity_level"],
                "handling_rules": info["handling_rules"],
                "created_at": now,
            }
        # Pre-populate default domains
        for name, desc in _DEFAULT_DOMAINS.items():
            cat["domains"][name] = {
                "description": desc,
                "owner": "Unassigned",
                "steward": "Unassigned",
                "tables": [],
                "created_at": now,
            }
        return cat

    def _save_catalog(self):
        try:
            self.catalog["updated_at"] = (
                datetime.now(timezone.utc).isoformat())
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(self.catalog, f, indent=2, default=str)
        except Exception as e:
            print(f"[CATALOG] Save failed: {e}")

    # ── Table & Column Registration ──

    def register_table(self, table_name, metadata=None):
        try:
            m = metadata or {}
            now = datetime.now(timezone.utc).isoformat()
            ex = self.catalog["tables"].get(table_name, {})
            self.catalog["tables"][table_name] = {
                "registered_at": ex.get("registered_at", now), "updated_at": now,
                "business_name": m.get("business_name", table_name),
                "description": m.get("description", ""),
                "domain": m.get("domain", "Unassigned"),
                "owner": m.get("owner", "Unassigned"),
                "steward": m.get("steward", "Unassigned"),
                "source_system": m.get("source_system", "Unknown"),
                "refresh_frequency": m.get("refresh_frequency", "Unknown"),
                "certification_status": ex.get("certification_status", "Uncertified"),
                "trust_score": ex.get("trust_score"),
                "last_dq_scan": ex.get("last_dq_scan"),
                "last_dq_score": ex.get("last_dq_score"),
                "tags": m.get("tags", ex.get("tags", [])),
                "columns": ex.get("columns", {}),
                "relationships": ex.get("relationships", []),
                "regulatory_mappings": m.get("regulatory_mappings", []),
                "row_count": m.get("row_count", 0),
            }
            self._save_catalog()
            print(f"[CATALOG] Registered table: {table_name}")
        except Exception as e:
            print(f"[CATALOG] register_table failed: {e}")

    def register_column(self, table_name, column_name, metadata=None):
        try:
            if table_name not in self.catalog["tables"]:
                self.register_table(table_name)
            m = metadata or {}
            self.catalog["tables"][table_name]["columns"][column_name] = {
                "business_name": m.get("business_name", column_name),
                "description": m.get("description", ""),
                "data_type": m.get("data_type", "unknown"),
                "classification": m.get("classification", "General"),
                "pii": m.get("pii", False),
                "nullable": m.get("nullable", True),
                "business_rules": m.get("business_rules", []),
                "valid_values": m.get("valid_values"),
                "format_pattern": m.get("format_pattern"),
                "steward": m.get("steward", "Unassigned"),
                "tags": m.get("tags", []),
                "regulatory_tags": m.get("regulatory_tags", []),
            }
            self._save_catalog()
        except Exception as e:
            print(f"[CATALOG] register_column failed: {e}")

    # ── Auto-cataloging ──

    def _classify_column(self, col_name):
        """Detect classification from column name hints."""
        low = col_name.lower()
        if any(h in low for h in _PII_HINTS):
            return "PII", True
        if any(h in low for h in _FINANCIAL_HINTS):
            return "Financial", False
        if any(h in low for h in _IDENTIFIER_HINTS):
            return "Identifier", False
        if any(h in low for h in _REFERENCE_HINTS):
            return "Reference", False
        return "General", False

    def auto_catalog_from_dataframe(self, df, table_name,
                                     source_system="Unknown"):
        """Register a table and all columns from a DataFrame."""
        try:
            self.register_table(table_name, {
                "row_count": len(df),
                "source_system": source_system,
            })
            for col in df.columns:
                classification, pii = self._classify_column(col)
                s = df[col]
                nullable = bool(s.isna().any())
                dtype_str = str(s.dtype)

                valid_values = None
                if s.dtype == object and s.nunique() < 20:
                    valid_values = sorted(
                        s.dropna().unique().astype(str).tolist())

                self.register_column(table_name, col, {
                    "data_type": dtype_str,
                    "classification": classification,
                    "pii": pii,
                    "nullable": nullable,
                    "valid_values": valid_values,
                })

            print(f"[CATALOG] Auto-cataloged {table_name}: "
                  f"{len(df.columns)} columns")
            return self.catalog["tables"].get(table_name)
        except Exception as e:
            print(f"[CATALOG] auto_catalog_from_dataframe failed: {e}")
            return None

    # ── Business Glossary ──

    def add_glossary_term(self, term, definition, domain=None,
                          synonyms=None, related_terms=None, owner=None):
        try:
            self.catalog["glossary"][term.lower()] = {
                "term": term,
                "definition": definition,
                "domain": domain or "General",
                "synonyms": synonyms or [],
                "related_terms": related_terms or [],
                "owner": owner or "Unassigned",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "approved": False,
            }
            self._save_catalog()
        except Exception as e:
            print(f"[CATALOG] add_glossary_term failed: {e}")

    def _split_column_name(self, col_name):
        """Split a column name into readable words."""
        # Strip common prefixes
        name = col_name
        for prefix in _PREFIX_DOMAIN:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break
        # camelCase split
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
        # underscore/hyphen split
        name = re.sub(r"[_\-]+", " ", name)
        return name.strip().title()

    def _detect_domain_from_name(self, col_name):
        """Guess domain from column name prefix or suffix."""
        low = col_name.lower()
        for prefix, domain in _PREFIX_DOMAIN.items():
            if low.startswith(prefix):
                return domain
        _suffix_map = [
            (("_date", "_time", "timestamp"), "Operations"),
            (("_amt", "_price", "_cost", "_bal", "revenue"), "Financial"),
            (("_name", "_address", "_phone", "_email"), "Customer"),
            (("_key", "_id", "_pk", "_fk"), "Reference"),
            (("_status", "_type", "_code", "_flag"), "Reference"),
        ]
        for hints, domain in _suffix_map:
            if any(s in low for s in hints):
                return domain
        return "General"

    def auto_generate_glossary(self, df, table_name):
        """Generate glossary entries from column names and data types."""
        try:
            generated = []
            _type_hints = {
                "int": "Numeric value representing ", "float": "Numeric value representing ",
                "datetime": "Date/time value indicating ", "object": "Text value describing ",
                "bool": "Boolean flag indicating ",
            }
            _domain_suffix = {
                "Financial": " for financial reporting and analysis",
                "Customer": " associated with customer records",
                "Transaction": " related to transactions",
                "Operations": " for operational tracking",
            }
            _overrides = [
                (("date", "_dt"), "The date when the {r} event occurred."),
                (("key", "_id"), "Unique identifier for {r} records."),
                (("_amt", "amount"), "Monetary amount for {r} in the base currency."),
                (("_name", "name"), "The name or label for {r}."),
                (("status",), "Current status of {r} in the business process."),
            ]
            for col in df.columns:
                readable = self._split_column_name(col)
                domain = self._detect_domain_from_name(col)
                dtype = str(df[col].dtype)
                hint = next((v for k, v in _type_hints.items() if k in dtype), "")
                definition = f"{hint}the {readable.lower()}"
                definition += _domain_suffix.get(domain, "") + "."
                low = col.lower()
                for suffixes, tpl in _overrides:
                    if any(low.endswith(s) for s in suffixes):
                        definition = tpl.format(r=readable.lower())
                        break
                self.add_glossary_term(readable, definition, domain=domain,
                                       related_terms=[table_name])
                generated.append(readable)
            print(f"[CATALOG] Generated {len(generated)} glossary terms from {table_name}")
            return generated
        except Exception as e:
            print(f"[CATALOG] auto_generate_glossary failed: {e}")
            return []

    # ── Domains & Stewards ──

    def add_domain(self, domain_name, description, owner=None,
                   steward=None):
        try:
            self.catalog["domains"][domain_name] = {
                "description": description,
                "owner": owner or "Unassigned",
                "steward": steward or "Unassigned",
                "tables": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_catalog()
            print(f"[CATALOG] Added domain: {domain_name}")
        except Exception as e:
            print(f"[CATALOG] add_domain failed: {e}")

    def add_steward(self, name, email=None, role=None, domains=None):
        try:
            self.catalog["stewards"][name] = {
                "email": email or "",
                "role": role or "Data Steward",
                "domains": domains or [],
                "assigned_tables": [],
                "assigned_columns": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_catalog()
            print(f"[CATALOG] Added steward: {name}")
        except Exception as e:
            print(f"[CATALOG] add_steward failed: {e}")

    def add_classification(self, name, description, sensitivity_level,
                           handling_rules=None):
        try:
            valid = ("Public", "Internal", "Confidential", "Restricted")
            if sensitivity_level not in valid:
                print(f"[CATALOG] Invalid sensitivity_level: "
                      f"{sensitivity_level}. Must be one of {valid}")
                return
            self.catalog["classifications"][name] = {
                "description": description,
                "sensitivity_level": sensitivity_level,
                "handling_rules": handling_rules or [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_catalog()
            print(f"[CATALOG] Added classification: {name}")
        except Exception as e:
            print(f"[CATALOG] add_classification failed: {e}")

    # ── Search ──

    def search_catalog(self, query):
        """Search tables, columns, and glossary. Case-insensitive."""
        try:
            q = query.lower()
            results = []

            def _hit(rtype, name, field, ctx):
                results.append({"type": rtype, "name": name,
                                "match_field": field, "context": ctx[:120]})

            for tname, td in self.catalog.get("tables", {}).items():
                if q in tname.lower():
                    _hit("table", tname, "table_name", td.get("description", ""))
                bn = td.get("business_name", "")
                if bn and q in bn.lower() and q not in tname.lower():
                    _hit("table", tname, "business_name", bn)
                ds = td.get("description", "")
                if ds and q in ds.lower():
                    _hit("table", tname, "description", ds)
                for cn, cd in td.get("columns", {}).items():
                    fqn = f"{tname}.{cn}"
                    if q in cn.lower():
                        _hit("column", fqn, "column_name", cd.get("description", ""))
                    cb = cd.get("business_name", "")
                    if cb and q in cb.lower() and q not in cn.lower():
                        _hit("column", fqn, "business_name", cb)
                    cds = cd.get("description", "")
                    if cds and q in cds.lower():
                        _hit("column", fqn, "description", cds)

            for gk, gd in self.catalog.get("glossary", {}).items():
                if q in gk:
                    _hit("glossary", gd.get("term", gk), "term", gd.get("definition", ""))
                gdf = gd.get("definition", "")
                if gdf and q in gdf.lower() and q not in gk:
                    _hit("glossary", gd.get("term", gk), "definition", gdf)
            return results
        except Exception as e:
            print(f"[CATALOG] search failed: {e}")
            return []

    # ── Getters ──

    def get_table_catalog(self, table_name):
        try:
            return self.catalog["tables"].get(table_name)
        except Exception:
            return None

    def get_column_catalog(self, table_name, column_name):
        try:
            tbl = self.catalog["tables"].get(table_name)
            if tbl:
                return tbl.get("columns", {}).get(column_name)
            return None
        except Exception:
            return None

    def get_glossary(self):
        try:
            terms = list(self.catalog.get("glossary", {}).values())
            terms.sort(key=lambda t: t.get("term", "").lower())
            return terms
        except Exception:
            return []

    def get_catalog_stats(self):
        try:
            tables = self.catalog.get("tables", {})
            total_cols = 0
            pii_cols = 0
            described_cols = 0
            certified = 0

            for tdata in tables.values():
                cols = tdata.get("columns", {})
                total_cols += len(cols)
                if tdata.get("certification_status") == "Certified":
                    certified += 1
                for cdata in cols.values():
                    if cdata.get("pii"):
                        pii_cols += 1
                    if cdata.get("description"):
                        described_cols += 1

            coverage = (
                round(described_cols / total_cols * 100, 1)
                if total_cols else 0
            )
            return {
                "tables_cataloged": len(tables),
                "columns_cataloged": total_cols,
                "glossary_terms": len(self.catalog.get("glossary", {})),
                "domains": len(self.catalog.get("domains", {})),
                "stewards": len(self.catalog.get("stewards", {})),
                "pii_columns": pii_cols,
                "certified_tables": certified,
                "uncatalogued_columns": total_cols - described_cols,
                "coverage_pct": coverage,
            }
        except Exception as e:
            print(f"[CATALOG] get_catalog_stats failed: {e}")
            return {}

    # ── Export ──

    def export_catalog(self, fmt="json"):
        """Export the full catalog as JSON or Markdown."""
        try:
            if fmt == "json":
                return json.dumps(self.catalog, indent=2, default=str)
            if fmt != "markdown":
                return json.dumps(self.catalog, indent=2, default=str)

            lines = [
                "# Enterprise Data Catalog",
                f"**Organization:** {self.catalog.get('organization', 'N/A')}",
                f"**Updated:** {self.catalog.get('updated_at', '')}", "",
                "## Table of Contents",
                "1. [Tables](#tables)",
                "2. [Business Glossary](#business-glossary)",
                "3. [Domains](#domains)", "", "## Tables", "",
            ]
            for tname, td in self.catalog.get("tables", {}).items():
                dq = td.get("last_dq_score")
                dq_s = f"{dq:.1f}" if dq else "N/A"
                lines += [
                    f"### {tname}",
                    f"- **Business Name:** {td.get('business_name', tname)}",
                    f"- **Description:** {td.get('description', 'N/A')}",
                    f"- **Domain:** {td.get('domain', 'N/A')}",
                    f"- **Owner:** {td.get('owner', 'N/A')}",
                    f"- **Certification:** {td.get('certification_status', '?')}",
                    f"- **DQ Score:** {dq_s}",
                    f"- **Rows:** {td.get('row_count', 'N/A'):,}", "",
                ]
                cols = td.get("columns", {})
                if cols:
                    lines.append("| Column | Type | Classification | PII | Nullable |")
                    lines.append("|--------|------|---------------|-----|----------|")
                    for cn, cd in cols.items():
                        pii = "Yes" if cd.get("pii") else "No"
                        nul = "Yes" if cd.get("nullable") else "No"
                        lines.append(
                            f"| {cn} | {cd.get('data_type', '?')} "
                            f"| {cd.get('classification', '?')} | {pii} | {nul} |")
                    lines.append("")

            lines += ["## Business Glossary", ""]
            for gd in self.get_glossary():
                lines.append(f"**{gd.get('term', '?')}** ({gd.get('domain', '?')})")
                lines.append(f": {gd.get('definition', 'N/A')}")
                syns = gd.get("synonyms", [])
                if syns:
                    lines.append(f"  Synonyms: {', '.join(syns)}")
                lines.append("")

            lines += ["## Domains", ""]
            for dn, dd in self.catalog.get("domains", {}).items():
                lines += [
                    f"### {dn}", dd.get("description", "N/A"),
                    f"- Owner: {dd.get('owner', 'N/A')}",
                    f"- Steward: {dd.get('steward', 'N/A')}", "",
                ]
            return "\n".join(lines)
        except Exception as e:
            print(f"[CATALOG] export failed: {e}")
            return "{}"

    # ── DQ Integration ──

    def update_table_dq(self, table_name, dq_score, scan_timestamp):
        try:
            if table_name not in self.catalog["tables"]:
                print(f"[CATALOG] Table {table_name} not in catalog")
                return
            self.catalog["tables"][table_name]["last_dq_scan"] = (
                scan_timestamp)
            self.catalog["tables"][table_name]["last_dq_score"] = dq_score
            self._save_catalog()
            print(f"[CATALOG] Updated DQ for {table_name}: {dq_score}")
        except Exception as e:
            print(f"[CATALOG] update_table_dq failed: {e}")

    def set_certification(self, table_name, status, certified_by=None,
                          notes=None):
        try:
            valid = ("Certified", "Warning", "Quarantined", "Uncertified")
            if status not in valid:
                print(f"[CATALOG] Invalid status: {status}. "
                      f"Must be one of {valid}")
                return
            if table_name not in self.catalog["tables"]:
                print(f"[CATALOG] Table {table_name} not in catalog")
                return
            tbl = self.catalog["tables"][table_name]
            tbl["certification_status"] = status
            tbl["certification_date"] = (
                datetime.now(timezone.utc).isoformat())
            tbl["certified_by"] = certified_by
            tbl["certification_notes"] = notes
            self._save_catalog()
            print(f"[CATALOG] {table_name} certified as: {status}")
        except Exception as e:
            print(f"[CATALOG] set_certification failed: {e}")
