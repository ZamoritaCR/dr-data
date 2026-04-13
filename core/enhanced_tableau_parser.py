"""
Enhanced Tableau Parser.

Single authoritative TWB/TWBX parser for Dr. Data.
Extracts worksheets with chart types, shelf fields, visual encodings,
filters; dashboards with deduplicated zone spatial layout; calculated
fields; parameters; join relationships; hyper file detection.
"""
import re
import defusedxml.ElementTree as ET
import zipfile
import tempfile
import shutil
import os


def _safe_strip_brackets(name):
    """Strip surrounding [] from Tableau column names without mangling edge cases."""
    if name.startswith('[') and name.endswith(']') and name.count('[') == 1:
        return name[1:-1]
    return name


def _extract_mark_type(ws_element):
    """Extract the primary mark/chart type from a Tableau worksheet element.

    Checks <table><pane><mark class="..."> first (most specific),
    then falls back to any <mark> element under the worksheet.
    Returns lowercase hyphenated mark class, or "automatic".
    """
    for pane in ws_element.iter("pane"):
        for mark in pane.iter("mark"):
            mc = mark.get("class", "")
            if mc and mc.lower() != "automatic":
                return mc.lower().replace(" ", "-")

    for mark in ws_element.iter("mark"):
        mc = mark.get("class", "")
        if mc and mc.lower() != "automatic":
            return mc.lower().replace(" ", "-")

    return "automatic"


def _disambiguate_chart_type(ws_info):
    """Refine chart type using shelf fields and encodings after initial extraction.

    The raw mark type from _extract_mark_type() is ambiguous in many cases:
    - Circle can be a scatter plot OR a bubble map (when lat/lon fields present)
    - Multipolygon/polygon are filled maps, not tables
    - Shape with no shelf fields = UI decoration (zoom buttons), not a visual
    - Shape with dims on rows = table, not scatter
    - Automatic with empty shelves = card/BAN, not column chart

    Must be called AFTER rows_fields/cols_fields are populated on ws_info.
    Returns the refined chart type string.
    """
    mark = ws_info.get("chart_type", "automatic")
    rows_f = ws_info.get("rows_fields", [])
    cols_f = ws_info.get("cols_fields", [])
    all_fields = rows_f + cols_f
    field_str = " ".join(all_fields).lower()

    # Circle + lat/lon fields = map, not scatter
    if mark == "circle" and ("latitude" in field_str or "longitude" in field_str):
        return "map"

    # Multipolygon/polygon = filled map (choropleth)
    if mark in ("multipolygon", "polygon"):
        return "filled-map"

    # Shape with no shelf fields = UI decoration (zoom buttons, icons)
    if mark == "shape" and not rows_f and not cols_f:
        return "skip"

    # Shape with dims on rows = table, not scatter
    if mark == "shape" and rows_f:
        return "text"

    # Automatic with empty shelves = card/BAN
    if mark in ("automatic", "") and not rows_f and not cols_f:
        return "ban"

    return mark


def _extract_shelf_fields(text):
    """Extract field references from shelf expression text.

    Tableau stores shelf expressions like:
      [Category]
      SUM([Sales])
      ([federated.xxx].[none:Region:nk] * [federated.xxx].[sum:Sales:qk])
    Returns a list of field reference strings found inside brackets.
    """
    if not text:
        return []
    return re.findall(r'\[([^\]]+)\]', text)


def _extract_ws_filters(ws_element):
    """Extract worksheet-level filters from <filter> elements.

    Skips Tableau-internal filters (Action filters, Tooltip filters)
    that do not correspond to real data columns.
    """
    # Prefixes that indicate Tableau-internal (non-data) filters
    _INTERNAL_FILTER_PREFIXES = ("Action (", "Tooltip (", "Multiple Values")
    filters = []
    for filt in ws_element.iter("filter"):
        col = filt.get("column", "")
        fclass = filt.get("class", "")
        if col:
            field = col
            m = re.search(r'\[([^\]]+)\]$', col)
            if m:
                field = m.group(1)
            # Strip Tableau shelf encoding from field name.
            # e.g. "none:Region:nk" -> "Region", "sum:Sales:qk" -> "Sales"
            if ":" in field:
                parts = field.split(":")
                if len(parts) >= 2:
                    # Middle part is the actual field name
                    candidate = parts[1].strip().strip('"').strip("'")
                    if candidate:
                        field = candidate
            # Skip Tableau-internal action/tooltip filters
            if any(field.startswith(p) for p in _INTERNAL_FILTER_PREFIXES):
                continue
            filters.append({
                "field": field,
                "column_ref": col,
                "type": fclass or "categorical",
            })
    return filters


def _extract_global_design(root):
    """Extract workbook-level design metadata: color palettes and global fonts.

    Returns a dict with 'color_palettes' and 'global_fonts' keys.
    """
    design = {
        "color_palettes": [],
        "global_fonts": {},
    }

    # Color palettes defined at workbook level
    for cp in root.iter("color-palette"):
        palette_name = cp.get("name", "")
        palette_type = cp.get("type", "")
        colors = [c.text for c in cp.findall("color") if c.text]
        design["color_palettes"].append({
            "name": palette_name,
            "type": palette_type,
            "colors": colors,
        })

    # Global font settings from top-level <style> or <preferences> format elements.
    # Only capture font-related attributes from direct workbook-level style elements.
    font_attrs = {}
    for style_el in root.findall("style"):
        for sr in style_el.findall("style-rule"):
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr in ("font-family", "font-size", "font-color",
                            "font-style", "font-weight") and val:
                    font_attrs[attr] = val
    if font_attrs:
        design["global_fonts"] = font_attrs

    # Datasource-level color encodings (discrete palette maps applied globally)
    ds_color_maps = []
    for ds in root.iter("datasource"):
        for sr in ds.iter("style-rule"):
            if sr.get("element") == "mark":
                for enc in sr.findall("encoding"):
                    if enc.get("attr") == "color" and enc.get("type") == "palette":
                        field = enc.get("field", "")
                        mappings = []
                        for m in enc.findall("map"):
                            color = m.get("to", "")
                            bucket = m.find("bucket")
                            if color:
                                mappings.append({
                                    "color": color,
                                    "value": bucket.text.strip('"') if bucket is not None and bucket.text else "",
                                })
                        if mappings:
                            ds_color_maps.append({
                                "field": field,
                                "mappings": mappings,
                            })
    if ds_color_maps:
        design["datasource_color_maps"] = ds_color_maps

    return design


def _extract_ws_design(ws_element):
    """Extract per-worksheet design metadata from style-rules and mark elements.

    Returns a dict with mark_colors, background_color, title_font,
    axis_config, mark_style, and border keys.
    """
    ws_design = {
        "mark_colors": [],
        "background_color": "",
        "title_font": {},
        "axis_config": [],
        "mark_style": {},
        "border": {},
    }

    seen_colors = set()

    for sr in ws_element.iter("style-rule"):
        element_type = sr.get("element", "")

        if element_type == "mark":
            # Mark colors from <encoding> with color maps
            for enc in sr.findall("encoding"):
                if enc.get("attr") == "color":
                    palette = enc.get("palette", "")
                    enc_type = enc.get("type", "")
                    # Discrete palette color mappings
                    for m in enc.findall("map"):
                        color = m.get("to", "")
                        if color and color not in seen_colors:
                            seen_colors.add(color)
                            bucket = m.find("bucket")
                            ws_design["mark_colors"].append({
                                "color": color,
                                "value": bucket.text.strip('"') if bucket is not None and bucket.text else "",
                            })
                    # If no discrete maps but palette is named, record palette info
                    if not ws_design["mark_colors"] and (palette or enc_type):
                        ws_design["mark_style"]["color_palette"] = palette
                        ws_design["mark_style"]["color_type"] = enc_type

            # Mark style from <format> attrs
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr == "mark-color" and val:
                    if val not in seen_colors:
                        seen_colors.add(val)
                        ws_design["mark_colors"].append({"color": val, "value": ""})
                elif attr == "mark-labels-show" and val:
                    ws_design["mark_style"]["labels_visible"] = val.lower() == "true"
                elif attr == "mark-labels-mode" and val:
                    ws_design["mark_style"]["labels_mode"] = val
                elif attr == "mark-labels-cull" and val:
                    ws_design["mark_style"]["labels_cull"] = val.lower() == "true"
                elif attr.startswith("mark-labels-line-") and val:
                    ws_design["mark_style"][attr] = val.lower() == "true"
                elif attr.startswith("mark-labels-range-") and val:
                    ws_design["mark_style"][attr] = val

        elif element_type == "worksheet":
            # Background color from worksheet style rules
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr == "background-color" and val:
                    ws_design["background_color"] = val

        elif element_type == "axis":
            # Axis configuration
            for enc in sr.findall("encoding"):
                axis_info = {
                    "field": enc.get("field", ""),
                    "scope": enc.get("scope", ""),
                    "type": enc.get("type", ""),
                    "synchronized": enc.get("synchronized", ""),
                }
                ws_design["axis_config"].append(axis_info)
            for fmt in sr.findall("format"):
                attr = fmt.get("attr", "")
                val = fmt.get("value", "")
                if attr == "display" and val == "false":
                    # Axis hidden
                    field = fmt.get("field", "")
                    ws_design["axis_config"].append({
                        "field": field,
                        "hidden": True,
                    })

    # Title font from <title> element with <run> children or <format> elements
    for title_el in ws_element.iter("title"):
        for fmt in title_el.iter("formatted-text"):
            for run in fmt.iter("run"):
                font_info = {}
                for attr_name in ("fontname", "fontsize", "fontcolor", "bold",
                                  "italic", "underline"):
                    v = run.get(attr_name, "")
                    if v:
                        font_info[attr_name] = v
                if run.text:
                    font_info["text"] = run.text.strip()
                if font_info:
                    ws_design["title_font"] = font_info
                break  # first run is enough
        break  # only first title

    # Mark opacity/size from <mark> elements directly
    for pane in ws_element.iter("pane"):
        for mark in pane.iter("mark"):
            for child in mark:
                if child.tag == "encoding" and child.get("attr") == "size":
                    size_val = child.get("value", "")
                    if size_val:
                        ws_design["mark_style"]["size"] = size_val
                elif child.tag == "encoding" and child.get("attr") == "opacity":
                    opacity_val = child.get("value", "")
                    if opacity_val:
                        ws_design["mark_style"]["opacity"] = opacity_val

    # Border from zone-style (will be populated at dashboard level instead)
    # Keep border empty here -- dashboard zones carry border info

    return ws_design


def _parse_colors(ws_element, root):
    """Extract color hex values used by a worksheet.

    Checks three sources in priority order:
    1. Worksheet style-rule mark encodings: <style-rule element="mark"><encoding attr="color"><map to="...">
    2. Pane mark color values: <panes><pane><mark><color value="...">
    3. Workbook-level custom palettes: <preferences><color-palette><color>

    Returns a list of unique hex color strings (e.g. ["#4E79A7", "#F28E2B"]).
    Returns empty list if no custom colors found.
    """
    colors = []
    seen = set()

    def _add(c):
        if c and c.startswith("#") and c not in seen:
            seen.add(c)
            colors.append(c)

    # Source 1: worksheet style-rule mark color encodings
    for sr in ws_element.iter("style-rule"):
        if sr.get("element") == "mark":
            for enc in sr.findall("encoding"):
                if enc.get("attr") == "color":
                    for m in enc.findall("map"):
                        _add(m.get("to", ""))
            for fmt in sr.findall("format"):
                if fmt.get("attr") == "mark-color":
                    _add(fmt.get("value", ""))

    # Source 2: pane mark color values
    for pane in ws_element.iter("pane"):
        for mark in pane.iter("mark"):
            for color_el in mark.iter("color"):
                _add(color_el.get("value", ""))

    # Source 3: workbook-level custom color palettes (only if no ws-level colors)
    if not colors:
        for cp in root.iter("color-palette"):
            for c_el in cp.findall("color"):
                if c_el.text:
                    _add(c_el.text.strip())

    return colors


def _parse_dashboard_zones(dashboard_element):
    """Parse all zones from a dashboard element with worksheet references.

    For each zone, extracts name, spatial layout, and the worksheet it
    references (from param attribute or nested zone-xml content).

    Returns list of dicts:
        [{"name": str, "x": int, "y": int, "w": int, "h": int,
          "worksheet_ref": str or "", "type": str}]
    """
    zones = []
    seen = set()

    for zone in dashboard_element.iter("zone"):
        zone_name = zone.get("name", "")
        if not zone_name or zone_name in seen:
            continue
        seen.add(zone_name)

        zone_type = zone.get("type-v2", zone.get("type", ""))
        layout = _extract_zone_layout(zone)

        # Worksheet reference: check param attribute first (most common),
        # then look for nested zone content that names a worksheet
        ws_ref = zone.get("param", "")

        # Some zones store the worksheet reference in a nested <zone> with
        # a name attribute matching a worksheet
        if not ws_ref:
            for child_zone in zone.findall("zone"):
                child_name = child_zone.get("name", "")
                if child_name:
                    ws_ref = child_name
                    break

        zones.append({
            "name": zone_name,
            "x": layout.get("x", 0),
            "y": layout.get("y", 0),
            "w": layout.get("w", 0),
            "h": layout.get("h", 0),
            "worksheet_ref": ws_ref,
            "type": zone_type,
        })

    return zones


def _extract_zone_layout(zone_element):
    """Extract spatial layout dict from a zone element's x/y/w/h attributes.

    Returns a dict with int values for x, y, w, h (0 if not present).
    """
    layout = {}
    for attr in ("x", "y", "w", "h"):
        raw = zone_element.get(attr, "")
        if raw:
            try:
                layout[attr] = int(raw)
            except (ValueError, TypeError):
                layout[attr] = 0
    return layout


def _extract_zone_style(zone_element):
    """Extract border and style info from a zone's <zone-style> child."""
    border = {}
    for zs in zone_element.findall("zone-style"):
        for fmt in zs.findall("format"):
            attr = fmt.get("attr", "")
            val = fmt.get("value", "")
            if attr and val:
                border[attr] = val
    return border


def get_xml_root(path):
    """Get the XML Element root from a .twb or .twbx file.

    Returns the root Element or None on error. Caller must handle cleanup.
    Used by Sprint 2 color_extractor for deep color extraction.
    """
    actual_path = path
    tmpdir = None
    if str(path).lower().endswith('.twbx'):
        try:
            tmpdir = tempfile.mkdtemp(prefix="etp_root_")
            with zipfile.ZipFile(path, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.twb'):
                        z.extract(name, tmpdir)
                        actual_path = os.path.join(tmpdir, name)
                        break
        except Exception:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return None
    try:
        tree = ET.parse(actual_path)
        return tree.getroot()
    except Exception:
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_windows(root):
    """Extract the visible tab list from the <windows> section.

    Returns ordered list:
        [{"name": str, "type": str, "order": int, "hidden": bool}]
    """
    windows_el = root.find("windows")
    if windows_el is None:
        return []
    tabs = []
    for idx, win in enumerate(windows_el):
        if win.tag != "window":
            continue
        name = win.get("name", "")
        wclass = win.get("class", "")
        hidden = win.get("hidden", "").lower() == "true"
        if not name or not wclass:
            continue
        tabs.append({
            "name": name,
            "type": wclass,
            "order": idx,
            "hidden": hidden,
        })
    return tabs


def _parse_stories(root):
    """Extract story elements with their storypoints.

    Returns:
        [{"name": str, "storypoints": [{"caption": str, "sheet": str}]}]
    """
    stories = []
    for story in root.iter("story"):
        s_name = story.get("name", "")
        if not s_name:
            continue
        points = []
        for sp in story.iter("storypoint"):
            caption = sp.get("caption", "")
            sheet = ""
            for child in sp:
                if child.tag in ("dashboard", "worksheet"):
                    sheet = child.get("name", "")
                    break
            if not sheet:
                sheet = sp.get("captured-sheet", "")
            points.append({"caption": caption, "sheet": sheet})
        stories.append({"name": s_name, "storypoints": points})
    return stories


def parse_twb(path):
    """Parse a Tableau workbook XML with full metadata extraction.

    Handles both .twb and .twbx files (extracts .twb from archive).

    Returns a dict with:
        type, version, datasources, worksheets, dashboards,
        calculated_fields, parameters, filters, relationships,
        has_hyper, windows, stories, worksheet_colors
    """
    spec = {
        "type": "tableau_workbook",
        "version": "",
        "datasources": [],
        "worksheets": [],
        "dashboards": [],
        "calculated_fields": [],
        "parameters": [],
        "filters": [],
        "relationships": [],
        "has_hyper": False,
        "windows": [],
        "stories": [],
        "worksheet_colors": {},
        "calc_id_map": {},  # internal Calculation_XXX -> caption name
    }

    actual_path = path
    tmpdir = None
    has_hyper = False

    # Handle .twbx (packaged workbook)
    if str(path).lower().endswith('.twbx'):
        try:
            tmpdir = tempfile.mkdtemp(prefix="etp_")
            with zipfile.ZipFile(path, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.twb'):
                        z.extract(name, tmpdir)
                        actual_path = os.path.join(tmpdir, name)
                    if name.endswith('.hyper') or name.endswith('.tde'):
                        has_hyper = True
        except Exception as e:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            spec["parse_error"] = str(e)
            return spec

    spec["has_hyper"] = has_hyper

    try:
        tree = ET.parse(actual_path)
        root = tree.getroot()
        spec["version"] = root.get("version", "unknown")

        # --- Global design metadata (color palettes, fonts) ---
        spec["design"] = _extract_global_design(root)

        # --- Datasources ---
        for ds in root.iter("datasource"):
            ds_name = ds.get("name", "") or ds.get("caption", "")

            # Parameters datasource
            if ds_name == "Parameters" or ds_name.startswith("Parameters"):
                for col in ds.iter("column"):
                    if col.get("param-domain-type"):
                        spec["parameters"].append({
                            "name": col.get("caption", col.get("name", "")),
                            "datatype": col.get("datatype", ""),
                            "domain": col.get("param-domain-type", ""),
                            "value": col.get("value", ""),
                        })
                continue

            ds_info = {
                "name": ds_name,
                "caption": ds.get("caption", ds_name),
                "connection_type": "",
                "server": "",
                "dbname": "",
                "filename": "",
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
                    if table_name and table_name not in ds_info["tables"]:
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
                        internal_name = _safe_strip_brackets(col.get("name", ""))
                        caption_name = col.get("caption", "") or internal_name
                        spec["calculated_fields"].append({
                            "name": caption_name,
                            "internal_name": internal_name,
                            "formula": formula,
                            "datatype": col.get("datatype", ""),
                            "role": col.get("role", ""),
                            "datasource": ds_name,
                        })
                        # Map internal ID -> caption for visual binding resolution
                        if internal_name != caption_name:
                            spec["calc_id_map"][internal_name] = caption_name

                ds_info["columns"].append(col_info)

            if ds_name:
                spec["datasources"].append(ds_info)

        # --- Worksheets ---
        for ws in root.iter("worksheet"):
            ws_name = ws.get("name", "")
            chart_type = _extract_mark_type(ws)
            ws_filters = _extract_ws_filters(ws)

            ws_info = {
                "name": ws_name,
                "chart_type": chart_type,
                "mark_type": chart_type,
                "marks": [],
                "rows": "",
                "cols": "",
                "rows_fields": [],
                "cols_fields": [],
                "dimensions": [],
                "measures": [],
                "filters": ws_filters,
                "color_field": "",
                "size_field": "",
                "label_fields": [],
                "tooltip_fields": [],
                "sort_field": "",
            }

            # Collect all mark layers
            for mark in ws.iter("mark"):
                mc = mark.get("class", "")
                if mc:
                    ws_info["marks"].append(mc)

            # Shelf expressions from <rows> and <cols>
            table = ws.find(".//table")
            if table is not None:
                rows_el = table.find("rows")
                cols_el = table.find("cols")
                if rows_el is not None and rows_el.text:
                    ws_info["rows"] = rows_el.text
                    ws_info["rows_fields"] = _extract_shelf_fields(rows_el.text)
                if cols_el is not None and cols_el.text:
                    ws_info["cols"] = cols_el.text
                    ws_info["cols_fields"] = _extract_shelf_fields(cols_el.text)

                # Filters from the table element (append to spec-level list too)
                for filt in table.iter("filter"):
                    f_col = filt.get("column", "")
                    if f_col:
                        spec["filters"].append({
                            "worksheet": ws_name,
                            "column": f_col,
                        })

            # Visual encoding shelves
            for enc in ws.iter("encoding"):
                attr = enc.get("attr", "")
                field = enc.get("column", enc.get("field", ""))
                if not field:
                    continue
                if attr in ("columns", "rows"):
                    ws_info["dimensions"].append(field)
                elif attr in ("size", "color", "text", "detail"):
                    ws_info["measures"].append(field)

                if attr == "color":
                    ws_info["color_field"] = field
                elif attr == "size":
                    ws_info["size_field"] = field
                elif attr == "label":
                    ws_info["label_fields"].append(field)
                elif attr == "tooltip":
                    ws_info["tooltip_fields"].append(field)

            # Sort field
            for sort_el in ws.iter("sort"):
                sort_field = sort_el.get("column", sort_el.get("field", ""))
                if sort_field:
                    ws_info["sort_field"] = sort_field
                    break

            # Per-worksheet design metadata
            ws_info["design"] = _extract_ws_design(ws)

            # Per-worksheet color extraction
            ws_info["colors"] = _parse_colors(ws, root)

            # -- Mark type disambiguation using shelf context --
            # Refine ambiguous mark types (circle, shape, multipolygon, etc.)
            # now that shelf fields are populated.
            refined = _disambiguate_chart_type(ws_info)
            if refined != ws_info["chart_type"]:
                ws_info["chart_type"] = refined
                ws_info["mark_type"] = refined

            # -- Chart type inference from shelf bindings --
            # When the mark class is "automatic", infer a better chart type
            # from the actual row/col shelf composition rather than defaulting
            # to clusteredColumnChart for everything.
            # Uses a priority chain: first match wins.
            if ws_info["chart_type"] == "automatic":
                rows_f = ws_info["rows_fields"]
                cols_f = ws_info["cols_fields"]
                all_f = rows_f + cols_f

                # Helper: check if a shelf field is a measure (aggregated)
                _agg_prefixes = ("sum:", "avg:", "cnt:", "count:", "countd:",
                                 "min:", "max:", "median:", "attr:")
                _date_prefixes = ("yr:", "mn:", "qr:", "tqr:", "tmn:",
                                  "twk:", "day:", "mdy:", "wk:", "md:")
                _geo_keywords = ("latitude", "longitude", "country", "state",
                                 "city", "zip", "geo", "region", "province",
                                 "county", "postal", "location")

                def _is_measure_ref(f):
                    fl = f.lower()
                    return (fl.endswith(":qk") or
                            any(fl.startswith(p) for p in _agg_prefixes))

                def _is_date_ref(f):
                    fl = f.lower()
                    return (fl.endswith(":ok") or
                            any(fl.startswith(p) for p in _date_prefixes))

                def _is_geo_ref(f):
                    fl = f.lower()
                    return any(kw in fl for kw in _geo_keywords)

                def _is_dim_ref(f):
                    return (not f.startswith("federated.")
                            and ":" in f
                            and not _is_measure_ref(f)
                            and not _is_date_ref(f))

                # Count field types across both shelves
                has_real_fields = any(
                    ":" in f and not f.startswith("federated")
                    for f in all_f
                )
                measure_refs = [f for f in all_f if _is_measure_ref(f)]
                date_refs = [f for f in all_f if _is_date_ref(f)]
                geo_refs = [f for f in all_f if _is_geo_ref(f)]
                dim_refs = [f for f in all_f if _is_dim_ref(f)]

                rows_measures = [f for f in rows_f if _is_measure_ref(f)]
                cols_measures = [f for f in cols_f if _is_measure_ref(f)]
                rows_dates = [f for f in rows_f if _is_date_ref(f)]
                cols_dates = [f for f in cols_f if _is_date_ref(f)]
                rows_dims = [f for f in rows_f if _is_dim_ref(f)]
                cols_dims = [f for f in cols_f if _is_dim_ref(f)]

                # Also check for "Measure Names"/"Measure Values" on shelves
                has_measure_names = any(
                    "Measure Names" in f or "Measure Values" in f
                    for f in all_f
                )

                inferred = None

                # 1. Geographic fields -> map
                if geo_refs:
                    inferred = "map"

                # 2. No real fields at all -> card (single aggregate)
                elif not has_real_fields:
                    inferred = "ban"

                # 3. Only measures, no dimensions or dates -> card
                elif measure_refs and not dim_refs and not date_refs:
                    inferred = "ban"

                # 4. Measure Names/Values pattern -> multi-measure line
                elif has_measure_names and date_refs:
                    inferred = "line"

                # 5. Date on either shelf + measure -> line chart
                elif (cols_dates or rows_dates) and measure_refs:
                    inferred = "line"

                # 6. Both shelves have measures (X=measure, Y=measure) -> scatter
                elif rows_measures and cols_measures and not dim_refs:
                    inferred = "circle"

                # 7. Both shelves have dims (no measures) -> text table
                elif rows_dims and cols_dims and not measure_refs:
                    inferred = "text"

                # 8. Two+ measures + one dimension -> line
                elif len(measure_refs) >= 2 and len(dim_refs) == 1:
                    inferred = "line"

                # 9. Default fallback (current behavior)
                # else: keep "automatic" -> clusteredColumnChart

                if inferred:
                    ws_info["chart_type"] = inferred
                    ws_info["mark_type"] = inferred

            spec["worksheets"].append(ws_info)

        # --- Dashboards (with zone deduplication) ---
        # Build a set of known worksheet names for cross-referencing
        known_ws_names = {ws.get("name", "") for ws in spec["worksheets"]}
        # Also build case-insensitive lookup
        known_ws_lower = {n.lower().strip(): n for n in known_ws_names if n}

        for db in root.iter("dashboard"):
            db_name = db.get("name", "")
            db_info = {
                "name": db_name,
                "size": {},
                "worksheets_used": [],
                "zones": [],
            }

            # Dashboard canvas size
            size_el = db.find("size")
            if size_el is not None:
                db_info["size"] = {
                    "width": size_el.get("maxwidth", size_el.get("width", "")),
                    "height": size_el.get("maxheight", size_el.get("height", "")),
                }

            # Canvas dimensions as integers for design replication
            canvas_w = ""
            canvas_h = ""
            if size_el is not None:
                canvas_w = size_el.get("maxwidth", size_el.get("width", ""))
                canvas_h = size_el.get("maxheight", size_el.get("height", ""))
            try:
                canvas_w_int = int(canvas_w) if canvas_w else 0
            except (ValueError, TypeError):
                canvas_w_int = 0
            try:
                canvas_h_int = int(canvas_h) if canvas_h else 0
            except (ValueError, TypeError):
                canvas_h_int = 0
            db_info["canvas"] = {
                "width": canvas_w_int,
                "height": canvas_h_int,
            }

            # Track seen zone names for deduplication
            seen_zones = set()
            # Track seen worksheet names for worksheets_used dedup
            seen_ws_used = set()

            for zone in db.iter("zone"):
                zone_name = zone.get("name", "")
                if not zone_name:
                    continue

                zone_type = zone.get("type-v2", zone.get("type", ""))
                zone_data = {
                    "name": zone_name,
                    "type": zone_type,
                }
                for attr in ("x", "y", "w", "h"):
                    val = zone.get(attr, "")
                    if val:
                        zone_data[attr] = val

                # Zone layout (x, y, w, h as integers)
                zone_data["layout"] = _extract_zone_layout(zone)

                # Zone border/style from <zone-style>
                zone_style = _extract_zone_style(zone)
                if zone_style:
                    zone_data["zone_style"] = zone_style

                # Keep the first (outermost) zone occurrence with spatial data;
                # skip duplicates from nested zone references.
                zone_key = zone_name
                if zone_key not in seen_zones:
                    seen_zones.add(zone_key)
                    db_info["zones"].append(zone_data)

                    # Only add to worksheets_used if the zone name matches
                    # an actual worksheet (not layout containers, filters, etc.)
                    # Also skip the dashboard's own name and deduplicate.
                    if zone_name != db_name and zone_name not in seen_ws_used:
                        # Check if this zone name is a real worksheet
                        ws_match = zone_name in known_ws_names
                        if not ws_match:
                            # Try case-insensitive match
                            ws_match = zone_name.lower().strip() in known_ws_lower
                        if ws_match:
                            seen_ws_used.add(zone_name)
                            db_info["worksheets_used"].append(zone_name)

            # Parsed zones with worksheet references (for QA manifest)
            db_info["parsed_zones"] = _parse_dashboard_zones(db)

            spec["dashboards"].append(db_info)

        # --- Windows (visible tab list) ---
        spec["windows"] = _parse_windows(root)

        # --- Stories ---
        spec["stories"] = _parse_stories(root)

        # --- Per-worksheet colors ---
        ws_colors = {}
        for ws in root.iter("worksheet"):
            ws_name = ws.get("name", "")
            if ws_name:
                colors = _parse_colors(ws, root)
                if colors:
                    ws_colors[ws_name] = colors
        spec["worksheet_colors"] = ws_colors

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
