"""
Universal file handler. Takes any uploaded file and returns:
- A pandas DataFrame (if data file)
- A parsed structure dict (if report/dashboard file)
- Both (if packaged file like .twbx with embedded data)
"""
import pandas as pd
import json
import os
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import shutil

from core.enhanced_tableau_parser import parse_twb as _enhanced_parse_twb

SUPPORTED_DATA = {
    "csv", "tsv", "xlsx", "xls", "parquet", "json",
    "xml", "txt", "dat", "rpt"
}
SUPPORTED_REPORTS = {"twb", "twbx", "wid", "yxmd", "yxwz", "yxmc", "yxzp"}
SUPPORTED_ARCHIVES = {"zip"}

ALL_SUPPORTED = SUPPORTED_DATA | SUPPORTED_REPORTS | SUPPORTED_ARCHIVES


def sanitize_columns(df):
    """Clean up DataFrame column names for Power BI compatibility.

    Fixes:
    - Pandas duplicate suffixes (e.g. 'Event.1' -> 'Event_2')
    - Bracket expressions that conflict with M/DAX syntax
    - Leading/trailing whitespace
    """
    import re
    seen = {}
    new_cols = []
    for col in df.columns:
        c = str(col).strip()
        # Remove brackets that conflict with M/DAX bracket notation
        # e.g. '[Event] = [Parameter 1]' -> 'Event = Parameter 1'
        c = c.replace("[", "").replace("]", "")
        # Replace pandas duplicate suffix (.1, .2) with underscore version
        m = re.match(r'^(.+)\.(\d+)$', c)
        if m:
            base, num = m.group(1), int(m.group(2))
            c = f"{base}_{num + 1}"
        # Ensure uniqueness
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 1
        new_cols.append(c)
    df.columns = new_cols
    return df


def identify_file(file_path):
    """Identify file type and return category."""
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    if ext in SUPPORTED_DATA:
        return "data", ext
    elif ext in SUPPORTED_REPORTS:
        return "report", ext
    elif ext in SUPPORTED_ARCHIVES:
        return "archive", ext
    else:
        return "unknown", ext


def ingest_file(file_path):
    """
    Universal file ingestion.
    Returns dict with:
      "dataframes": {name: pd.DataFrame}
      "report_structure": dict or None
      "file_type": str
      "file_name": str
      "errors": list
    """
    result = {
        "dataframes": {},
        "report_structure": None,
        "file_type": "",
        "file_name": os.path.basename(file_path),
        "errors": []
    }

    category, ext = identify_file(file_path)
    result["file_type"] = ext

    try:
        if category == "data":
            df = load_data_file(file_path, ext)
            if df is not None:
                result["dataframes"]["main"] = df

        elif category == "report":
            if ext in ("twbx", "twb"):
                result["report_structure"] = _enhanced_parse_twb(file_path)
                if ext == "twbx":
                    # Also extract embedded data files
                    _, dfs = _extract_twbx_data(file_path)
                    result["dataframes"].update(dfs)
            elif ext == "wid":
                result["report_structure"] = parse_bobj(file_path)
            elif ext in ("yxmd", "yxwz", "yxmc", "yxzp"):
                from migration.alteryx_parser import AlteryxParser
                parser = AlteryxParser()
                result["report_structure"] = parser.parse(file_path)
            elif ext == "rpt":
                df = load_rpt(file_path)
                if df is not None:
                    result["dataframes"]["main"] = df
                else:
                    result["report_structure"] = {
                        "type": "crystal_reports",
                        "note": "Binary .rpt -- extracted what text was readable"
                    }

        elif category == "archive":
            result = ingest_zip(file_path)

        else:
            # Try loading as text/CSV anyway
            try:
                df = pd.read_csv(file_path, sep=None, engine="python")
                result["dataframes"]["main"] = df
                result["file_type"] = "auto_detected_delimited"
            except Exception:
                result["errors"].append(
                    f"Unrecognized file type: .{ext}"
                )

    except Exception as e:
        result["errors"].append(str(e))

    # Sanitize column names in all loaded DataFrames
    for key, df in result["dataframes"].items():
        if isinstance(df, pd.DataFrame):
            result["dataframes"][key] = sanitize_columns(df)

    return result


def load_excel_smart(file_path):
    """Load the largest sheet from an Excel file.

    Always picks the sheet with the most rows so that the primary data table
    is loaded regardless of sheet ordering.

    Returns:
        tuple: (DataFrame, sheet_name)
    """
    xls = pd.ExcelFile(file_path)
    if len(xls.sheet_names) == 1:
        name = xls.sheet_names[0]
        return pd.read_excel(file_path, sheet_name=name), name

    best_df = None
    best_name = xls.sheet_names[0]
    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        if best_df is None or len(df) > len(best_df):
            best_df = df
            best_name = sheet
    return best_df, best_name


def load_data_file(file_path, ext):
    """Load a data file into a DataFrame."""
    try:
        if ext == "csv":
            return pd.read_csv(file_path)
        elif ext == "tsv":
            return pd.read_csv(file_path, sep="\t")
        elif ext in ("xlsx", "xls"):
            df, _sheet = load_excel_smart(file_path)
            return df
        elif ext == "parquet":
            return pd.read_parquet(file_path)
        elif ext == "json":
            return pd.read_json(file_path)
        elif ext == "xml":
            return pd.read_xml(file_path)
        elif ext in ("txt", "dat"):
            return pd.read_csv(file_path, sep=None, engine="python")
        elif ext == "rpt":
            return load_rpt(file_path)
    except Exception as e:
        print(f"[WARN] Failed to load {file_path} as {ext}: {e}")
        return None


def load_rpt(file_path):
    """Load .rpt file -- could be Crystal Reports export or fixed-width."""
    try:
        return pd.read_fwf(file_path)
    except Exception:
        try:
            return pd.read_csv(file_path, sep="|", engine="python")
        except Exception:
            try:
                return pd.read_csv(file_path, sep=None, engine="python")
            except Exception:
                return None



def parse_twb(file_path):
    """Parse Tableau .twb XML file.

    Legacy wrapper -- delegates to core.enhanced_tableau_parser.parse_twb().
    """
    return _enhanced_parse_twb(file_path)


def _extract_twbx_data(file_path):
    """Extract embedded data files (CSV/Excel) from a .twbx archive.

    Returns (None, dataframes_dict). Structure parsing is handled by
    the enhanced parser which processes .twbx natively.
    """
    dataframes = {}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmp)
            for root_dir, dirs, files in os.walk(tmp):
                for f in files:
                    fpath = os.path.join(root_dir, f)
                    if f.endswith((".csv", ".xlsx")):
                        try:
                            if f.endswith(".csv"):
                                df = pd.read_csv(fpath)
                            else:
                                df = pd.read_excel(fpath)
                            dataframes[f] = df
                        except Exception:
                            pass
        except Exception:
            pass
    return None, dataframes


def parse_twbx(file_path):
    """Parse packaged Tableau workbook -- extract data + structure.

    Legacy wrapper -- delegates structure parsing to the enhanced parser.
    """
    structure = _enhanced_parse_twb(file_path)
    _, dataframes = _extract_twbx_data(file_path)
    return structure, dataframes


def parse_bobj(file_path):
    """Parse Business Objects .wid file (best effort)."""
    structure = {
        "type": "business_objects_webi",
        "file_name": os.path.basename(file_path),
        "note": (
            "Business Objects .wid is a proprietary binary format. "
            "I extracted readable text content. For full structure, "
            "export from BusinessObjects as CSV or Excel, or provide "
            "the report specification separately."
        ),
        "extracted_text": [],
        "possible_queries": [],
        "possible_columns": []
    }

    try:
        with open(file_path, "rb") as f:
            content = f.read()

        text_chunks = []
        current = []
        for byte in content:
            if 32 <= byte < 127:
                current.append(chr(byte))
            else:
                if len(current) > 4:
                    text_chunks.append("".join(current))
                current = []

        for chunk in text_chunks:
            chunk = chunk.strip()
            if len(chunk) > 10:
                lower = chunk.lower()
                if any(kw in lower for kw in [
                    "select", "from", "where", "group by",
                    "order by", "join"
                ]):
                    structure["possible_queries"].append(chunk)
                elif any(kw in lower for kw in [
                    "sum(", "count(", "avg(", "max(", "min("
                ]):
                    structure["possible_columns"].append(chunk)
                else:
                    structure["extracted_text"].append(chunk)

        structure["extracted_text"] = structure["extracted_text"][:50]

    except Exception as e:
        structure["error"] = str(e)

    return structure


def ingest_zip(file_path):
    """Extract ZIP and process all files inside."""
    result = {
        "dataframes": {},
        "report_structure": None,
        "file_type": "zip_archive",
        "file_name": os.path.basename(file_path),
        "contained_files": [],
        "errors": []
    }

    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmp)

            for root_dir, dirs, files in os.walk(tmp):
                for f in files:
                    fpath = os.path.join(root_dir, f)
                    result["contained_files"].append(f)

                    category, ext = identify_file(f)
                    if category == "data":
                        df = load_data_file(fpath, ext)
                        if df is not None:
                            result["dataframes"][f] = df
                    elif category == "report":
                        sub = ingest_file(fpath)
                        if sub.get("report_structure"):
                            result["report_structure"] = sub["report_structure"]
                        result["dataframes"].update(sub.get("dataframes", {}))

        except Exception as e:
            result["errors"].append(str(e))

    return result
