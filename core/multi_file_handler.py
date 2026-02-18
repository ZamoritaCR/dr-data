"""
Multi-File Session Manager.

Handles multiple uploads and intelligently pairs them:
- .twb + .csv  -> Tableau structure + data = Power BI project
- .twb + .xlsx -> same
- .twbx alone  -> has both structure + embedded data
- .yxmd + .csv -> Alteryx workflow + data
- .csv + .csv  -> multiple data sources (ask user which is primary)
- .wid + .csv  -> Business Objects structure + data
- any combo    -> figure it out

Drop this in: core/multi_file_handler.py
"""
import os
import sys
import pandas as pd
import zipfile
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..")))
from app.file_handler import load_excel_smart
from core.relationship_detector import RelationshipDetector


# File type classification
STRUCTURE_FILES = {"twb", "twbx", "yxmd", "yxwz", "yxmc", "yxzp", "wid"}
DATA_FILES = {"csv", "xlsx", "xls", "tsv", "parquet", "json", "xml", "txt"}
ARCHIVE_FILES = {"zip", "twbx", "yxzp"}


class MultiFileSession:
    """
    Manages multiple uploaded files in a session.

    Attributes:
        files: dict of {filename: FileInfo}
        structure_file: the primary structure file (Tableau/Alteryx/BOBJ) or None
        data_files: list of data files loaded as DataFrames
        primary_df: the main DataFrame for dashboard building
        tableau_spec: parsed Tableau structure (if applicable)
        alteryx_spec: parsed Alteryx structure (if applicable)
        data_source_map: which data file maps to which Tableau data source
    """

    def __init__(self):
        self.files = {}
        self.structure_file = None
        self.data_files = []
        self.primary_df = None
        self.primary_sheet_name = None
        self.tableau_spec = None
        self.alteryx_spec = None
        self.data_source_map = {}
        self._temp_dir = tempfile.mkdtemp(prefix="drdata_")
        self.detector = RelationshipDetector()
        self.relationships = []
        self.unified_df = None
        self.join_log = []

    def add_file(self, filename, file_bytes_or_path):
        """
        Add a file to the session.
        Returns a dict describing what was loaded and what Dr. Data should say.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        # Save to temp
        if isinstance(file_bytes_or_path, (bytes, bytearray)):
            path = os.path.join(self._temp_dir, filename)
            with open(path, "wb") as f:
                f.write(file_bytes_or_path)
        elif isinstance(file_bytes_or_path, str) and os.path.exists(file_bytes_or_path):
            path = file_bytes_or_path
        else:
            # Assume it's a file-like with getbuffer()
            path = os.path.join(self._temp_dir, filename)
            with open(path, "wb") as f:
                f.write(file_bytes_or_path.getbuffer())

        info = {
            "filename": filename,
            "path": path,
            "ext": ext,
            "size": os.path.getsize(path),
            "category": self._classify(ext),
            "df": None,
            "structure": None,
        }

        # Process based on type
        if ext == "twbx":
            # Packaged Tableau -- has structure AND data
            info["category"] = "tableau_packaged"
            twb_spec, embedded_dfs = self._parse_twbx(path)
            info["structure"] = twb_spec
            self.tableau_spec = twb_spec
            self.structure_file = info
            if embedded_dfs:
                # Use largest embedded DataFrame as primary
                biggest = max(embedded_dfs, key=lambda x: len(x))
                info["df"] = biggest
                self.primary_df = biggest
                self.data_files.append({
                    "filename": f"{filename} (embedded data)",
                    "df": biggest,
                    "rows": len(biggest),
                    "columns": list(biggest.columns),
                })

        elif ext == "twb":
            info["category"] = "tableau_structure"
            twb_spec = self._parse_twb(path)
            info["structure"] = twb_spec
            self.tableau_spec = twb_spec
            self.structure_file = info

        elif ext in ("yxmd", "yxwz", "yxmc", "yxzp"):
            info["category"] = "alteryx"
            alteryx_spec = self._parse_alteryx(path)
            info["structure"] = alteryx_spec
            self.alteryx_spec = alteryx_spec
            self.structure_file = info

        elif ext == "wid":
            info["category"] = "bobj_structure"
            bobj_spec = self._parse_bobj(path)
            info["structure"] = bobj_spec
            self.structure_file = info

        elif ext in DATA_FILES:
            df, sheet_name = self._load_data(path, ext)
            if df is not None:
                info["df"] = df
                info["sheet_name"] = sheet_name
                info["category"] = "data"
                self.data_files.append({
                    "filename": filename,
                    "df": df,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sheet_name": sheet_name,
                })
                # Set as primary if we don't have one
                if self.primary_df is None:
                    self.primary_df = df
                    self.primary_sheet_name = sheet_name
                # If we already have a structure file, try to map
                if self.structure_file and self.tableau_spec:
                    self._try_map_data_to_tableau(filename, df)

        elif ext == "zip":
            info["category"] = "archive"
            extracted = self._process_zip(path)
            for item in extracted:
                self.add_file(item["filename"], item["path"])

        self.files[filename] = info
        return self._describe_session()

    def add_dataframe(self, table_name, df):
        """Add a pre-loaded DataFrame (e.g. from Snowflake) to the session."""
        self.data_files.append({
            "filename": table_name,
            "df": df,
            "rows": len(df),
            "columns": list(df.columns),
            "sheet_name": table_name,
        })
        if self.primary_df is None:
            self.primary_df = df
            self.primary_sheet_name = table_name
        self.files[table_name] = {
            "filename": table_name,
            "path": None,
            "ext": "snowflake",
            "size": 0,
            "category": "data",
            "df": df,
            "structure": None,
        }
        # Auto-detect relationships when 2+ tables loaded
        if len(self.data_files) >= 2:
            self.relationships = self.detector.detect_all(self.data_files)
        return self._describe_session()

    def get_summary(self):
        """Get current session state for the agent."""
        return self._describe_session()

    def get_primary_dataframe(self):
        """Get the main DataFrame for building dashboards.

        Prefers the unified (auto-joined) DataFrame if available.
        """
        if self.unified_df is not None:
            return self.unified_df
        if self.primary_df is not None:
            return self.primary_df
        # Fall back to first data file
        if self.data_files:
            return self.data_files[0]["df"]
        return None

    def get_all_dataframes(self):
        """Get all loaded DataFrames."""
        return {d["filename"]: d["df"] for d in self.data_files}

    def set_primary_dataframe(self, filename):
        """Set which data file is the primary one."""
        for d in self.data_files:
            if d["filename"] == filename:
                self.primary_df = d["df"]
                return True
        return False

    def auto_unify(self):
        """Detect relationships and auto-join all data files into one DataFrame.

        Sets self.relationships, self.unified_df, self.join_log.
        Returns (unified_df, relationships, join_log).
        """
        dfs = self.get_all_dataframes()
        if len(dfs) < 2:
            return self.get_primary_dataframe(), [], []

        self.relationships = self.detector.detect(dfs)
        self.unified_df, self.join_log = self.detector.auto_join(
            dfs, self.relationships
        )
        return self.unified_df, self.relationships, self.join_log

    def has_structure(self):
        """Do we have a Tableau/Alteryx/BOBJ structure file?"""
        return self.structure_file is not None

    def has_data(self):
        """Do we have at least one data file loaded?"""
        return len(self.data_files) > 0

    def needs_data(self):
        """Do we have a structure file but no data?"""
        return self.has_structure() and not self.has_data()

    def is_ready_for_build(self):
        """Do we have everything needed to build a dashboard?"""
        return self.has_data()

    # ==================================================================
    # INTERNAL: File classification
    # ==================================================================

    def _classify(self, ext):
        if ext in STRUCTURE_FILES:
            return "structure"
        if ext in DATA_FILES:
            return "data"
        if ext in ARCHIVE_FILES:
            return "archive"
        return "unknown"

    # ==================================================================
    # INTERNAL: Data loading
    # ==================================================================

    def _load_data(self, path, ext):
        """Load a data file into a DataFrame.

        Returns:
            tuple: (DataFrame, sheet_name) -- sheet_name is None for non-Excel.
        """
        try:
            if ext == "csv":
                return pd.read_csv(path, low_memory=False), None
            elif ext == "tsv":
                return pd.read_csv(path, sep="\t", low_memory=False), None
            elif ext in ("xlsx", "xls"):
                return load_excel_smart(path)
            elif ext == "parquet":
                return pd.read_parquet(path), None
            elif ext == "json":
                try:
                    return pd.read_json(path), None
                except Exception:
                    return pd.json_normalize(pd.read_json(path, typ="series")), None
            elif ext == "xml":
                return pd.read_xml(path), None
            elif ext == "txt":
                # Try common delimiters
                for sep in [",", "\t", "|", ";"]:
                    try:
                        df = pd.read_csv(path, sep=sep, low_memory=False)
                        if len(df.columns) > 1:
                            return df, None
                    except Exception:
                        continue
                return pd.read_csv(path, low_memory=False), None
            else:
                # Try CSV as fallback
                return pd.read_csv(path, low_memory=False), None
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
        return None, None

    # ==================================================================
    # INTERNAL: Tableau parsing
    # ==================================================================

    def _parse_twbx(self, path):
        """Parse a packaged Tableau workbook (.twbx)."""
        spec = None
        dfs = []

        try:
            with zipfile.ZipFile(path, "r") as z:
                names = z.namelist()

                # Find the .twb inside
                twb_files = [n for n in names if n.endswith(".twb")]
                if twb_files:
                    twb_data = z.read(twb_files[0])
                    # Save to temp and parse
                    twb_path = os.path.join(self._temp_dir, "extracted.twb")
                    with open(twb_path, "wb") as f:
                        f.write(twb_data)
                    spec = self._parse_twb(twb_path)

                # Find data files inside
                data_exts = (".csv", ".xlsx", ".xls", ".hyper", ".tde", ".tsv")
                for name in names:
                    if name.lower().endswith(data_exts) and not name.startswith("__"):
                        try:
                            z.extract(name, self._temp_dir)
                            extracted_path = os.path.join(self._temp_dir, name)
                            ext = name.rsplit(".", 1)[-1].lower()

                            if ext in ("hyper", "tde"):
                                # Hyper/TDE need special handling
                                # For now, skip -- the user should provide the data separately
                                if spec:
                                    spec["has_hyper"] = True
                                continue

                            df, _sn = self._load_data(extracted_path, ext)
                            if df is not None:
                                dfs.append(df)
                        except Exception:
                            continue

        except Exception as e:
            print(f"[WARN] Error parsing .twbx: {e}")

        return spec, dfs

    def _parse_twb(self, path):
        """Parse a Tableau workbook XML."""
        spec = {
            "type": "tableau",
            "datasources": [],
            "worksheets": [],
            "dashboards": [],
            "calculated_fields": [],
            "parameters": [],
            "filters": [],
            "relationships": [],
            "has_hyper": False,
        }

        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # Data sources
            for ds in root.iter("datasource"):
                ds_name = ds.get("name", "") or ds.get("caption", "")
                if ds_name.startswith("Parameters"):
                    # Extract parameters
                    for col in ds.iter("column"):
                        if col.get("param-domain-type"):
                            spec["parameters"].append({
                                "name": col.get("caption", col.get("name", "")),
                                "type": col.get("datatype", ""),
                                "domain": col.get("param-domain-type", ""),
                            })
                    continue

                ds_info = {
                    "name": ds_name,
                    "caption": ds.get("caption", ds_name),
                    "connection_type": "",
                    "tables": [],
                    "columns": [],
                }

                # Connection info
                conn = ds.find(".//connection")
                if conn is not None:
                    ds_info["connection_type"] = conn.get("class", "unknown")
                    ds_info["server"] = conn.get("server", "")
                    ds_info["dbname"] = conn.get("dbname", "")
                    ds_info["filename"] = conn.get("filename", "")

                    # Tables
                    for rel in conn.iter("relation"):
                        table_name = rel.get("table", rel.get("name", ""))
                        if table_name:
                            ds_info["tables"].append(table_name)

                # Columns
                for col in ds.findall("column"):
                    col_info = {
                        "name": col.get("name", "").strip("[]"),
                        "caption": col.get("caption", ""),
                        "datatype": col.get("datatype", ""),
                        "role": col.get("role", ""),
                        "type": col.get("type", ""),
                    }

                    # Calculated fields
                    calc = col.find("calculation")
                    if calc is not None:
                        formula = calc.get("formula", "")
                        if formula:
                            col_info["formula"] = formula
                            spec["calculated_fields"].append({
                                "name": col.get("caption", col.get("name", "")),
                                "formula": formula,
                                "datasource": ds_name,
                            })

                    ds_info["columns"].append(col_info)

                if ds_name:
                    spec["datasources"].append(ds_info)

            # Worksheets
            for ws in root.iter("worksheet"):
                ws_name = ws.get("name", "")
                ws_info = {
                    "name": ws_name,
                    "mark_type": "",
                    "rows": "",
                    "cols": "",
                    "filters": [],
                }

                # Mark type (chart type)
                mark = ws.find(".//mark")
                if mark is not None:
                    ws_info["mark_type"] = mark.get("class", "")

                table = ws.find(".//table")
                if table is not None:
                    # Rows and columns on shelves
                    rows_el = table.find("rows")
                    cols_el = table.find("cols")
                    if rows_el is not None and rows_el.text:
                        ws_info["rows"] = rows_el.text
                    if cols_el is not None and cols_el.text:
                        ws_info["cols"] = cols_el.text

                    # Filters
                    for filt in table.iter("filter"):
                        f_col = filt.get("column", "")
                        if f_col:
                            ws_info["filters"].append(f_col)
                            spec["filters"].append({
                                "worksheet": ws_name,
                                "column": f_col,
                            })

                spec["worksheets"].append(ws_info)

            # Dashboards
            for db in root.iter("dashboard"):
                db_name = db.get("name", "")
                db_info = {
                    "name": db_name,
                    "worksheets_used": [],
                }
                for zone in db.iter("zone"):
                    ws_ref = zone.get("name", "")
                    if ws_ref and ws_ref != db_name:
                        db_info["worksheets_used"].append(ws_ref)

                spec["dashboards"].append(db_info)

        except Exception as e:
            spec["parse_error"] = str(e)

        return spec

    # ==================================================================
    # INTERNAL: Alteryx parsing
    # ==================================================================

    def _parse_alteryx(self, path):
        """Parse Alteryx workflow. Uses the full parser if available."""
        try:
            from migration.alteryx_parser import AlteryxParser
            parser = AlteryxParser()
            return parser.parse(path)
        except ImportError:
            # Lightweight fallback
            return self._parse_alteryx_lite(path)

    def _parse_alteryx_lite(self, path):
        """Lightweight Alteryx parsing without the full parser."""
        spec = {"type": "alteryx", "tools": [], "connections": [], "data_sources": []}
        try:
            ext = path.rsplit(".", 1)[-1].lower()
            if ext == "yxzp":
                with zipfile.ZipFile(path, "r") as z:
                    for name in z.namelist():
                        if name.endswith((".yxmd", ".yxwz", ".yxmc")):
                            z.extract(name, self._temp_dir)
                            path = os.path.join(self._temp_dir, name)
                            break

            tree = ET.parse(path)
            root = tree.getroot()

            for node in root.iter("Node"):
                tool_id = node.get("ToolID", "")
                gui = node.find("GuiSettings")
                plugin = gui.get("Plugin", "Unknown") if gui is not None else "Unknown"
                short_name = plugin.split(".")[-1] if "." in plugin else plugin
                spec["tools"].append({"id": tool_id, "plugin": plugin, "name": short_name})

            for conn in root.iter("Connection"):
                origin = conn.find("Origin")
                dest = conn.find("Destination")
                if origin is not None and dest is not None:
                    spec["connections"].append({
                        "from": origin.get("ToolID", ""),
                        "to": dest.get("ToolID", ""),
                    })

            spec["tool_count"] = len(spec["tools"])
            spec["connection_count"] = len(spec["connections"])

        except Exception as e:
            spec["parse_error"] = str(e)

        return spec

    # ==================================================================
    # INTERNAL: Business Objects parsing
    # ==================================================================

    def _parse_bobj(self, path):
        """Best-effort parsing of Business Objects .wid files."""
        spec = {"type": "bobj", "queries": [], "columns": [], "raw_text": ""}
        try:
            with open(path, "rb") as f:
                raw = f.read()

            # Extract readable text
            text = raw.decode("utf-8", errors="ignore")
            text = "".join(c if c.isprintable() or c in "\n\r\t" else " " for c in text)

            # Look for SQL queries
            import re
            sql_patterns = re.findall(r"SELECT\s+.+?FROM\s+.+?(?:WHERE|GROUP|ORDER|;|\Z)",
                                       text, re.IGNORECASE | re.DOTALL)
            spec["queries"] = sql_patterns[:10]

            # Look for column names (heuristic)
            col_patterns = re.findall(r"\b([A-Z][a-z]+(?:_[A-Z][a-z]+)+)\b", text)
            spec["columns"] = list(set(col_patterns))[:50]

            spec["raw_text"] = text[:5000]

        except Exception as e:
            spec["parse_error"] = str(e)

        return spec

    # ==================================================================
    # INTERNAL: ZIP processing
    # ==================================================================

    def _process_zip(self, path):
        """Extract a ZIP and return list of files to process."""
        results = []
        try:
            with zipfile.ZipFile(path, "r") as z:
                z.extractall(self._temp_dir)
                for name in z.namelist():
                    if name.startswith("__") or name.startswith("."):
                        continue
                    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                    full = os.path.join(self._temp_dir, name)
                    if os.path.isfile(full) and ext in (STRUCTURE_FILES | DATA_FILES):
                        results.append({"filename": os.path.basename(name), "path": full})
        except Exception as e:
            print(f"[WARN] ZIP extraction error: {e}")
        return results

    # ==================================================================
    # INTERNAL: Data-to-Tableau mapping
    # ==================================================================

    def _try_map_data_to_tableau(self, data_filename, df):
        """Try to map a data file to a Tableau data source."""
        if not self.tableau_spec:
            return

        base = data_filename.rsplit(".", 1)[0].lower().replace("_", "").replace("-", "").replace(" ", "")
        df_cols = set(c.lower().strip() for c in df.columns)

        for ds in self.tableau_spec.get("datasources", []):
            ds_name = (ds.get("name", "") or "").lower().replace("_", "").replace("-", "").replace(" ", "")
            ds_caption = (ds.get("caption", "") or "").lower().replace("_", "").replace("-", "").replace(" ", "")
            ds_file = (ds.get("filename", "") or "").lower()

            # Match by filename similarity
            if base in ds_name or base in ds_caption or base in ds_file:
                self.data_source_map[ds.get("name", "")] = {
                    "data_file": data_filename,
                    "match_type": "filename",
                }
                continue

            # Match by column overlap
            ds_cols = set()
            for col in ds.get("columns", []):
                ds_cols.add(col.get("name", "").lower().strip("[]"))
                ds_cols.add(col.get("caption", "").lower())

            if ds_cols and df_cols:
                overlap = ds_cols & df_cols
                if len(overlap) > min(3, len(ds_cols) * 0.3):
                    self.data_source_map[ds.get("name", "")] = {
                        "data_file": data_filename,
                        "match_type": "column_overlap",
                        "matched_columns": list(overlap)[:10],
                    }

    # ==================================================================
    # Session description for the agent
    # ==================================================================

    def _describe_session(self):
        """Build a summary dict the agent uses to know what it has."""
        desc = {
            "file_count": len(self.files),
            "files": [],
            "has_structure": self.has_structure(),
            "has_data": self.has_data(),
            "needs_data": self.needs_data(),
            "ready_for_build": self.is_ready_for_build(),
            "structure_type": None,
            "data_file_count": len(self.data_files),
            "primary_df_shape": None,
            "data_source_mapping": self.data_source_map,
        }

        for filename, info in self.files.items():
            f = {
                "filename": filename,
                "category": info["category"],
                "size_kb": round(info["size"] / 1024, 1),
            }
            if info.get("df") is not None:
                f["rows"] = len(info["df"])
                f["columns"] = list(info["df"].columns)
            if info.get("structure"):
                f["structure_summary"] = self._summarize_structure(info["structure"])
            desc["files"].append(f)

        if self.structure_file:
            desc["structure_type"] = self.structure_file["category"]

        if self.primary_df is not None:
            desc["primary_df_shape"] = {
                "rows": len(self.primary_df),
                "columns": len(self.primary_df.columns),
                "column_names": list(self.primary_df.columns),
            }

        # Tableau-specific
        if self.tableau_spec:
            desc["tableau"] = {
                "datasource_count": len(self.tableau_spec.get("datasources", [])),
                "worksheet_count": len(self.tableau_spec.get("worksheets", [])),
                "dashboard_count": len(self.tableau_spec.get("dashboards", [])),
                "calculated_fields": len(self.tableau_spec.get("calculated_fields", [])),
                "parameters": len(self.tableau_spec.get("parameters", [])),
                "filters": len(self.tableau_spec.get("filters", [])),
                "has_hyper": self.tableau_spec.get("has_hyper", False),
                "datasource_names": [
                    ds.get("caption", ds.get("name", ""))
                    for ds in self.tableau_spec.get("datasources", [])
                ],
                "unmapped_sources": [
                    ds.get("caption", ds.get("name", ""))
                    for ds in self.tableau_spec.get("datasources", [])
                    if ds.get("name", "") not in self.data_source_map
                ],
            }

        # Alteryx-specific
        if self.alteryx_spec:
            desc["alteryx"] = {
                "tool_count": self.alteryx_spec.get("tool_count",
                    len(self.alteryx_spec.get("tools", []))),
                "connection_count": self.alteryx_spec.get("connection_count",
                    len(self.alteryx_spec.get("connections", []))),
            }

        return desc

    def _summarize_structure(self, structure):
        """One-line summary of a structure file."""
        stype = structure.get("type", "unknown")
        if stype == "tableau":
            ws = len(structure.get("worksheets", []))
            db = len(structure.get("dashboards", []))
            cf = len(structure.get("calculated_fields", []))
            return f"Tableau: {db} dashboards, {ws} worksheets, {cf} calculated fields"
        elif stype == "alteryx":
            tc = structure.get("tool_count", len(structure.get("tools", [])))
            return f"Alteryx: {tc} tools"
        elif stype == "bobj":
            q = len(structure.get("queries", []))
            return f"Business Objects: {q} queries found"
        return f"{stype} structure"
