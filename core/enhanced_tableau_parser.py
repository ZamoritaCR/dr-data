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


def _safe_strip_brackets(name):
    """Strip surrounding [] from Tableau column names without mangling edge cases.

    strip("[]") is dangerous: it strips ALL leading/trailing [ and ] characters,
    so "[Sales [Net]]" becomes "Sales [Net" instead of "Sales [Net]".
    """
    if name.startswith('[') and name.endswith(']') and name.count('[') == 1:
        return name[1:-1]
    return name


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

                    # Parse join relationships
                    if rel.get("type") == "join":
                        join_type = rel.get("join", "inner")
                        for clause in rel.iter("expression"):
                            if clause.get("op") == "=":
                                operands = list(clause)
                                if len(operands) >= 2:
                                    left = operands[0].get("op", "")
                                    right = operands[1].get("op", "")
                                    # op contains "[Table].[Column]" reference
                                    if left and right:
                                        spec["relationships"].append({
                                            "left_ref": left,
                                            "right_ref": right,
                                            "join_type": join_type,
                                            "datasource": ds_name,
                                        })

            for col in ds.findall("column"):
                col_info = {
                    "name": _safe_strip_brackets(col.get("name", "")),
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
                "color_field": "",
                "size_field": "",
                "label_fields": [],
                "tooltip_fields": [],
                "sort_field": "",
            }

            # Mark types (all layers, not just first)
            marks = [m.get("class", "") for m in ws.iter("mark") if m.get("class")]
            ws_info["mark_type"] = marks[0] if marks else ""
            if len(marks) > 1:
                ws_info["mark_layers"] = marks

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

            # Visual encoding shelves
            for enc in ws.iter("encoding"):
                attr = enc.get("attr", "")
                field = enc.get("column", enc.get("field", ""))
                if attr == "color" and field:
                    ws_info["color_field"] = field
                elif attr == "size" and field:
                    ws_info["size_field"] = field
                elif attr == "label" and field:
                    ws_info["label_fields"].append(field)
                elif attr == "tooltip" and field:
                    ws_info["tooltip_fields"].append(field)

            # Sort field
            for sort_el in ws.iter("sort"):
                sort_field = sort_el.get("column", sort_el.get("field", ""))
                if sort_field:
                    ws_info["sort_field"] = sort_field
                    break

            spec["worksheets"].append(ws_info)

        # Dashboards
        for db in root.iter("dashboard"):
            db_name = db.get("name", "")
            db_info = {
                "name": db_name,
                "worksheets_used": [],
                "zones": [],
            }
            for zone in db.iter("zone"):
                ws_ref = zone.get("name", "")
                zone_data = {"name": ws_ref}
                for attr in ("x", "y", "w", "h"):
                    val = zone.get(attr, "")
                    if val:
                        zone_data[attr] = val
                zone_type = zone.get("type-v2", zone.get("type", ""))
                if zone_type:
                    zone_data["type"] = zone_type
                if ws_ref:
                    db_info["zones"].append(zone_data)
                    if ws_ref != db_name:
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
