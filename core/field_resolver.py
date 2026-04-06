"""
Sprint 1 -- Field Resolution Engine.

Resolves Tableau field references to actual Excel/data column names
using exact matching, alias mapping, and rapidfuzz fuzzy matching.
Uses tableaudocumentapi for structured TWB/TWBX parsing when available.
"""
import re
import os
import shutil
from rapidfuzz import fuzz, process
from core.utils import remove_tableau_brackets


class TableauFieldResolver:
    """Resolves Tableau field names to actual data column names."""

    def __init__(self, similarity_threshold=80, table_threshold=90):
        self.similarity_threshold = similarity_threshold
        self.table_threshold = table_threshold
        self._alias_map = {}
        self._reverse_alias_map = {}
        self._resolution_log = []

    def resolve_field(self, tableau_name, candidates, threshold=None):
        """Resolve a single Tableau field name against a list of candidate column names.

        Args:
            tableau_name: Tableau field reference (e.g. '[Sales Amount]', '[Orders].[Sales Amount]')
            candidates: iterable of real column names
            threshold: similarity threshold (default: self.similarity_threshold)

        Returns:
            dict with {matched, score, method} or None if no match above threshold.
        """
        if threshold is None:
            threshold = self.similarity_threshold

        candidates = list(candidates)
        if not candidates or not tableau_name:
            return None

        # Strip Tableau brackets and datasource prefix
        clean = self._strip_tableau_ref(tableau_name)

        # 1. Exact match (case-insensitive)
        for c in candidates:
            if c.lower().strip() == clean.lower().strip():
                return {"matched": c, "score": 100.0, "method": "exact"}

        # 2. Alias lookup
        alias_target = self._alias_map.get(clean.lower())
        if alias_target:
            for c in candidates:
                if c.lower() == alias_target.lower():
                    return {"matched": c, "score": 100.0, "method": "alias"}

        # 3. Normalized match (strip spaces, underscores, hyphens)
        clean_norm = self._normalize(clean)
        for c in candidates:
            if self._normalize(c) == clean_norm:
                return {"matched": c, "score": 98.0, "method": "normalized"}

        # 4. rapidfuzz token_sort_ratio
        result = process.extractOne(
            clean, candidates, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        if result:
            matched, score, _idx = result
            self._log(clean, matched, score, "fuzzy_token_sort")
            return {"matched": matched, "score": score, "method": "fuzzy_token_sort"}

        # 5. rapidfuzz WRatio as fallback
        result = process.extractOne(
            clean, candidates, scorer=fuzz.WRatio, score_cutoff=threshold
        )
        if result:
            matched, score, _idx = result
            self._log(clean, matched, score, "fuzzy_wratio")
            return {"matched": matched, "score": score, "method": "fuzzy_wratio"}

        return None

    def resolve_table_name(self, tableau_datasource_name, excel_sheet_names):
        """Fuzzy match a Tableau datasource name against Excel sheet names.

        Args:
            tableau_datasource_name: datasource caption/name from Tableau
            excel_sheet_names: list of sheet names from Excel file

        Returns:
            dict with {matched, score, method} or None
        """
        if not excel_sheet_names or not tableau_datasource_name:
            return None

        sheets = list(excel_sheet_names)
        clean = self._strip_tableau_ref(tableau_datasource_name)

        # Exact match first
        for s in sheets:
            if s.lower().strip() == clean.lower().strip():
                return {"matched": s, "score": 100.0, "method": "exact"}

        # Normalized match
        clean_norm = self._normalize(clean)
        for s in sheets:
            if self._normalize(s) == clean_norm:
                return {"matched": s, "score": 98.0, "method": "normalized"}

        # Fuzzy match with tight threshold
        result = process.extractOne(
            clean, sheets, scorer=fuzz.token_sort_ratio,
            score_cutoff=self.table_threshold
        )
        if result:
            matched, score, _idx = result
            return {"matched": matched, "score": score, "method": "fuzzy"}

        return None

    def resolve_all_datasource_fields(self, twb_path_or_xml, excel_df):
        """Extract all fields from a TWB/TWBX and resolve against Excel columns.

        Args:
            twb_path_or_xml: path to .twb or .twbx file
            excel_df: pandas DataFrame whose columns are the resolution targets

        Returns:
            dict: {tableau_field_id: {excel_column, score, method, is_calculated, formula, ...}}
        """
        import pandas as pd
        candidates = list(excel_df.columns)
        resolution_map = {}

        # Try tableaudocumentapi first
        fields = self._extract_fields_tda(twb_path_or_xml)
        if not fields:
            # Fallback to XML parsing
            fields = self._extract_fields_xml(twb_path_or_xml)

        # Build alias map from extracted fields
        self.build_alias_map_from_fields(fields)

        for field in fields:
            field_id = field.get("id", field.get("name", ""))
            caption = field.get("caption", "")
            name = field.get("name", "")
            formula = field.get("formula", "")
            is_calculated = bool(formula)

            # Try resolving caption first (more human-readable), then name
            resolution = None
            for try_name in [caption, name, field_id]:
                if try_name:
                    resolution = self.resolve_field(try_name, candidates)
                    if resolution:
                        break

            entry = {
                "field_id": field_id,
                "name": name,
                "caption": caption,
                "is_calculated": is_calculated,
                "formula": formula,
                "datatype": field.get("datatype", ""),
                "role": field.get("role", ""),
                "default_aggregation": field.get("default_aggregation", ""),
            }

            if resolution:
                entry["excel_column"] = resolution["matched"]
                entry["score"] = resolution["score"]
                entry["method"] = resolution["method"]
            else:
                entry["excel_column"] = None
                entry["score"] = 0
                entry["method"] = "unresolved"

            resolution_map[field_id or name or caption] = entry

        return resolution_map

    def build_alias_map(self, workbook):
        """Build alias map from a tableaudocumentapi Workbook object."""
        self._alias_map = {}
        self._reverse_alias_map = {}

        try:
            for ds in workbook.datasources:
                for field in ds.fields.values():
                    name = getattr(field, 'id', '') or ''
                    caption = getattr(field, 'caption', '') or ''
                    if name and caption and name != caption:
                        self._alias_map[caption.lower()] = name
                        self._alias_map[name.lower()] = caption
                        self._reverse_alias_map[name.lower()] = caption
                        self._reverse_alias_map[caption.lower()] = name
        except Exception:
            pass

    def build_alias_map_from_fields(self, fields):
        """Build alias map from a list of field dicts."""
        self._alias_map = {}
        self._reverse_alias_map = {}

        for field in fields:
            name = field.get("name", "")
            caption = field.get("caption", "")
            if name and caption and name.lower() != caption.lower():
                self._alias_map[caption.lower()] = name
                self._alias_map[name.lower()] = caption
                self._reverse_alias_map[name.lower()] = caption
                self._reverse_alias_map[caption.lower()] = name

    def get_resolution_log(self):
        """Return the log of fuzzy resolutions performed."""
        return list(self._resolution_log)

    # --- Internal methods ---

    def _extract_fields_tda(self, twb_path):
        """Extract fields using tableaudocumentapi."""
        try:
            from tableaudocumentapi import Workbook
            wb = Workbook(twb_path)
            self.build_alias_map(wb)

            fields = []
            for ds in wb.datasources:
                ds_name = getattr(ds, 'caption', '') or getattr(ds, 'name', '')
                for field_id, field in ds.fields.items():
                    f = {
                        "id": field_id,
                        "name": getattr(field, 'id', field_id) or field_id,
                        "caption": getattr(field, 'caption', '') or '',
                        "datatype": getattr(field, 'datatype', '') or '',
                        "role": getattr(field, 'role', '') or '',
                        "default_aggregation": getattr(field, 'default_aggregation', '') or '',
                        "formula": '',
                        "datasource": ds_name,
                    }
                    calc = getattr(field, 'calculation', None)
                    if calc:
                        f["formula"] = str(calc) if calc else ''
                    fields.append(f)
            return fields
        except Exception as e:
            print(f"[FIELD_RESOLVER] tableaudocumentapi failed: {e}, falling back to XML")
            return None

    def _extract_fields_xml(self, twb_path):
        """Extract fields by parsing TWB XML directly."""
        import defusedxml.ElementTree as ET
        import zipfile
        import tempfile

        actual_path = twb_path
        tmpdir = None

        # Handle .twbx (ZIP containing .twb)
        if str(twb_path).lower().endswith('.twbx'):
            try:
                tmpdir = tempfile.mkdtemp(prefix="field_resolver_")
                with zipfile.ZipFile(twb_path, 'r') as z:
                    for name in z.namelist():
                        if name.endswith('.twb'):
                            z.extract(name, tmpdir)
                            actual_path = os.path.join(tmpdir, name)
                            break
            except Exception:
                if tmpdir:
                    shutil.rmtree(tmpdir, ignore_errors=True)
                return []

        fields = []
        try:
            tree = ET.parse(actual_path)
            root = tree.getroot()

            for ds in root.iter("datasource"):
                ds_name = ds.get("name", "") or ds.get("caption", "")
                if ds_name.startswith("Parameters"):
                    continue

                for col in ds.findall("column"):
                    name = remove_tableau_brackets(col.get("name", ""))
                    caption = col.get("caption", "")
                    formula = ""
                    calc = col.find("calculation")
                    if calc is not None:
                        formula = calc.get("formula", "")

                    fields.append({
                        "id": col.get("name", ""),
                        "name": name,
                        "caption": caption,
                        "datatype": col.get("datatype", ""),
                        "role": col.get("role", ""),
                        "default_aggregation": col.get("aggregation", ""),
                        "formula": formula,
                        "datasource": ds_name,
                    })
        except Exception as e:
            print(f"[FIELD_RESOLVER] XML parse error: {e}")
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return fields

    @staticmethod
    def _strip_tableau_ref(ref):
        """Strip Tableau brackets and datasource prefix.

        '[Orders].[Sales Amount]' -> 'Sales Amount'
        '[Sales Amount]' -> 'Sales Amount'
        'Sales Amount' -> 'Sales Amount'
        """
        ref = ref.strip()
        # Pattern: [Datasource].[Field]
        m = re.match(r'\[([^\]]+)\]\.\[([^\]]+)\]', ref)
        if m:
            return m.group(2)
        # Pattern: [Field]
        m = re.match(r'^\[([^\]]+)\]$', ref)
        if m:
            return m.group(1)
        return ref

    @staticmethod
    def _normalize(s):
        """Normalize a string for comparison: lowercase, strip whitespace/underscores/hyphens."""
        return s.lower().replace("_", "").replace("-", "").replace(" ", "")

    def _log(self, original, matched, score, method):
        """Log a fuzzy resolution."""
        self._resolution_log.append({
            "original": original,
            "matched": matched,
            "score": score,
            "method": method,
        })
