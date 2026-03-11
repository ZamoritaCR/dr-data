"""
Enhanced Tableau Parser.

Wraps the existing TWB/TWBX parsing logic and adds field resolution
via the TableauFieldResolver from Sprint 1.
"""
import xml.etree.ElementTree as ET
import zipfile
import tempfile
import shutil
import os


def parse_twb(path):
    """Parse a Tableau workbook XML with enhanced field extraction.

    Returns a dict with:
        type, version, datasources, worksheets, dashboards,
        calculated_fields, parameters, filters, relationships,
        has_hyper, field_resolution_map (if resolver is run)
    """
    spec = {
        "type": "tableau",
        "version": "",
        "datasources": [],
        "worksheets": [],
        "dashboards": [],
        "calculated_fields": [],
        "parameters": [],
        "filters": [],
        "relationships": [],
        "has_hyper": False,
    }

    actual_path = path
    tmpdir = None
    # Handle .twbx
    if str(path).lower().endswith('.twbx'):
        try:
            tmpdir = tempfile.mkdtemp(prefix="etp_")
            with zipfile.ZipFile(path, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.twb'):
                        z.extract(name, tmpdir)
                        actual_path = os.path.join(tmpdir, name)
                        break
        except Exception as e:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            spec["parse_error"] = str(e)
            return spec

    try:
        tree = ET.parse(actual_path)
        root = tree.getroot()
        spec["version"] = root.get("version", "unknown")

        # Data sources
        for ds in root.iter("datasource"):
            ds_name = ds.get("name", "") or ds.get("caption", "")
            if ds_name.startswith("Parameters"):
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

            conn = ds.find(".//connection")
            if conn is not None:
                ds_info["connection_type"] = conn.get("class", "unknown")
                ds_info["server"] = conn.get("server", "")
                ds_info["dbname"] = conn.get("dbname", "")
                ds_info["filename"] = conn.get("filename", "")

                for rel in conn.iter("relation"):
                    table_name = rel.get("table", rel.get("name", ""))
                    if table_name:
                        ds_info["tables"].append(table_name)

            for col in ds.findall("column"):
                col_info = {
                    "name": col.get("name", "").strip("[]"),
                    "caption": col.get("caption", ""),
                    "datatype": col.get("datatype", ""),
                    "role": col.get("role", ""),
                    "type": col.get("type", ""),
                }

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

            mark = ws.find(".//mark")
            if mark is not None:
                ws_info["mark_type"] = mark.get("class", "")

            table = ws.find(".//table")
            if table is not None:
                rows_el = table.find("rows")
                cols_el = table.find("cols")
                if rows_el is not None and rows_el.text:
                    ws_info["rows"] = rows_el.text
                if cols_el is not None and cols_el.text:
                    ws_info["cols"] = cols_el.text

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
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return spec


def parse_twb_with_resolution(path, excel_df=None):
    """Parse TWB and optionally resolve fields against an Excel DataFrame.

    Args:
        path: path to .twb or .twbx file
        excel_df: optional pandas DataFrame for field resolution

    Returns:
        dict: parsed spec with optional 'field_resolution_map' key
    """
    spec = parse_twb(path)

    if excel_df is not None:
        try:
            from core.field_resolver import TableauFieldResolver
            resolver = TableauFieldResolver()
            resolution_map = resolver.resolve_all_datasource_fields(path, excel_df)
            spec["field_resolution_map"] = resolution_map
        except Exception as e:
            spec["field_resolution_error"] = str(e)

    return spec
