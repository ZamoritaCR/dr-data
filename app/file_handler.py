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

SUPPORTED_DATA = {
    "csv", "tsv", "xlsx", "xls", "parquet", "json",
    "xml", "txt", "dat", "rpt"
}
SUPPORTED_REPORTS = {"twb", "twbx", "wid", "yxmd", "yxwz", "yxmc", "yxzp"}
SUPPORTED_ARCHIVES = {"zip"}

ALL_SUPPORTED = SUPPORTED_DATA | SUPPORTED_REPORTS | SUPPORTED_ARCHIVES


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
            if ext == "twbx":
                structure, dfs = parse_twbx(file_path)
                result["report_structure"] = structure
                result["dataframes"].update(dfs)
            elif ext == "twb":
                result["report_structure"] = parse_twb(file_path)
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
    """Parse Tableau .twb XML file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        structure = {
            "type": "tableau_workbook",
            "version": root.get("version", "unknown"),
            "worksheets": [],
            "dashboards": [],
            "datasources": [],
            "calculated_fields": [],
            "parameters": []
        }

        for ds in root.iter("datasource"):
            ds_name = ds.get("name", ds.get("caption", "unknown"))
            if ds_name != "Parameters":
                structure["datasources"].append({
                    "name": ds_name,
                    "caption": ds.get("caption", ds_name),
                })

            for col in ds.iter("column"):
                calc = col.find("calculation")
                if calc is not None:
                    structure["calculated_fields"].append({
                        "name": col.get("caption", col.get("name", "")),
                        "formula": calc.get("formula", ""),
                        "datasource": ds_name
                    })

            if ds_name == "Parameters":
                for col in ds.iter("column"):
                    structure["parameters"].append({
                        "name": col.get("caption", col.get("name", "")),
                        "datatype": col.get("datatype", ""),
                        "value": col.get("value", "")
                    })

        for ws in root.iter("worksheet"):
            ws_info = {
                "name": ws.get("name", ""),
                "marks": [],
                "dimensions": [],
                "measures": [],
                "filters": []
            }

            for mark in ws.iter("mark"):
                ws_info["marks"].append(mark.get("class", ""))

            for enc in ws.iter("encoding"):
                shelf = enc.get("attr", "")
                col_name = enc.get("column", "")
                if shelf in ("columns", "rows"):
                    ws_info["dimensions"].append(col_name)
                elif shelf in ("size", "color", "text"):
                    ws_info["measures"].append(col_name)

            structure["worksheets"].append(ws_info)

        for db in root.iter("dashboard"):
            db_info = {
                "name": db.get("name", ""),
                "zones": []
            }
            for zone in db.iter("zone"):
                zone_name = zone.get("name", "")
                if zone_name:
                    db_info["zones"].append(zone_name)
            structure["dashboards"].append(db_info)

        return structure

    except Exception as e:
        return {
            "type": "tableau_workbook",
            "error": str(e),
            "note": "Partial parse -- file may use newer Tableau format"
        }


def parse_twbx(file_path):
    """Parse packaged Tableau workbook -- extract data + structure."""
    structure = None
    dataframes = {}

    with tempfile.TemporaryDirectory() as tmp:
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(tmp)

            for root_dir, dirs, files in os.walk(tmp):
                for f in files:
                    fpath = os.path.join(root_dir, f)
                    if f.endswith(".twb"):
                        structure = parse_twb(fpath)
                    elif f.endswith((".csv", ".xlsx")):
                        try:
                            if f.endswith(".csv"):
                                df = pd.read_csv(fpath)
                            elif f.endswith(".xlsx"):
                                df = pd.read_excel(fpath)
                            else:
                                continue
                            dataframes[f] = df
                        except Exception:
                            pass
        except Exception as e:
            structure = {"type": "tableau_packaged", "error": str(e)}

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
