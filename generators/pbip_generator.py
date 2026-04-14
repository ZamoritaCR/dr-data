"""
PBIP Generator - Creates Power BI Project files matching PBI Desktop Jan 2026 output.

Reverse-engineered from PBI Desktop 2.150.2455.0 64-bit with preview features:
  - PBI_gitIntegration
  - PBI_enhancedReportFormat
  - PBI_tmdlInDataset

Structure produced (flat siblings, no parent folder):
  {name}.pbip                              project pointer
  {name}.Report/
    .platform                              type: Report
    definition.pbir                        report -> semantic model pointer
    definition/
      report.json                          theme + settings (small)
      version.json                         version metadata
      pages/
        pages.json                         page ordering
        {pageId}/
          page.json                        page dimensions
          visuals/
            {vizId}/
              visual.json                  individual visual
    StaticResources/SharedResources/BaseThemes/
      CY25SU12.json                        theme file (copied from reference)
  {name}.SemanticModel/
    .platform                              type: SemanticModel
    definition.pbism                       model pointer
    definition/
      database.tmdl
      model.tmdl
      cultures/en-US.tmdl
      tables/{table}.tmdl                  one per table
"""

import json
import os
import uuid
import shutil
from pathlib import Path
from copy import deepcopy

from generators.pbip_reference import (
    PBIP_TEMPLATE, PLATFORM_TEMPLATE, PBIR_TEMPLATE,
    REPORT_JSON_TEMPLATE, VERSION_JSON_TEMPLATE, PAGES_JSON_TEMPLATE,
    PAGE_JSON_TEMPLATE, VISUAL_JSON_TEMPLATE, PBISM_TEMPLATE,
    DATABASE_TMDL, CULTURE_TMDL, THEME_SOURCE_PATH,
)


def _parse_tableau_field_ref(ref):
    """Parse a Tableau field reference like '[Table].[Column]' or 'Table.Column'.

    Returns (table_name, column_name). If no table part, returns ('', column).
    """
    import re
    ref = ref.strip()
    # Pattern: [Table].[Column]
    m = re.match(r'\[([^\]]+)\]\.\[([^\]]+)\]', ref)
    if m:
        return m.group(1), m.group(2)
    # Pattern: Table.[Column]
    m = re.match(r'([^.\[]+)\.\[([^\]]+)\]', ref)
    if m:
        return m.group(1).strip(), m.group(2)
    # Pattern: [Table].Column
    m = re.match(r'\[([^\]]+)\]\.(.+)', ref)
    if m:
        return m.group(1), m.group(2).strip()
    # Pattern: Table.Column
    if '.' in ref:
        parts = ref.split('.', 1)
        return parts[0].strip(), parts[1].strip()
    # Just a column name
    return '', ref


class PBIPGenerator:
    """Generate a complete PBIP project from AI config."""

    # Map our simplified dataRoles keys to PBI queryState role names.
    # Different visual types use different roles.
    ROLE_MAP_CHART = {"category": "Category", "values": "Y", "series": "Series"}
    ROLE_MAP_CARD = {"values": "Values"}
    ROLE_MAP_TABLE = {"values": "Values"}
    ROLE_MAP_SLICER = {"category": "Values", "values": "Values"}
    ROLE_MAP_MAP = {"category": "Location", "values": "Category", "series": "Legend"}
    ROLE_MAP_SCATTER = {"category": "X", "values": "Y", "series": "Details"}

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).parent.parent.resolve()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, config, data_profile, dashboard_spec, data_file_path=None,
                 sheet_name=None, relationships=None, snowflake_config=None,
                 csv_url=None, dataframe=None):
        """Create the full PBIP project.

        Args:
            config:           dict from OpenAIEngine (report_layout + tmdl_model).
            data_profile:     dict from DataAnalyzer (auto-generated if None + dataframe).
            dashboard_spec:   dict from ClaudeInterpreter (for title/theme).
            data_file_path:   optional path to the source data file.
            sheet_name:       Excel sheet name that was loaded (for M expression).
            relationships:    optional list of relationship dicts for TMDL model.
            snowflake_config: optional dict with account/warehouse/database/schema
                              for Snowflake DirectQuery M expression.
            dataframe:        optional pandas DataFrame. When provided with no
                              data_file_path/csv_url/snowflake, data is embedded
                              inline in TMDL via #table() M expression.

        Returns:
            dict with path, field_audit, table_names, valid_measures, etc.
        """
        # Auto-profile DataFrame if data_profile not provided
        if data_profile is None and dataframe is not None:
            from core.deep_analyzer import DeepAnalyzer
            analyzer = DeepAnalyzer()
            data_profile = analyzer.profile(dataframe)
            table_name_hint = config.get("table_name", "Data")
            data_profile["table_name"] = table_name_hint

        # Store dataframe for inline embedding if no external data source
        self._inline_dataframe = dataframe

        title = dashboard_spec.get("dashboard_title", "Dashboard")
        safe = self._safe_name(title)

        # Create a clean subdirectory for this project (not mixed with other output)
        project_dir = self.output_dir / f"{safe}_pbip"
        if project_dir.exists():
            self._safe_remove(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create folder structure inside the clean project directory
        report_root = project_dir / f"{safe}.Report"
        report_def = report_root / "definition"
        pages_dir = report_def / "pages"
        theme_dir = report_root / "StaticResources" / "SharedResources" / "BaseThemes"
        model_root = project_dir / f"{safe}.SemanticModel"
        model_def = model_root / "definition"
        cultures_dir = model_def / "cultures"
        tables_dir = model_def / "tables"

        for d in [report_def, pages_dir, theme_dir, model_def, cultures_dir, tables_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Gather info -- table_name must match what _write_tables_tmdl uses.
        # Force "Data" when csv_url provided so TMDL table name
        # matches Entity="Data" in all visual JSON files.
        if csv_url is not None:
            table_name = "Data"
        else:
            table_name = data_profile.get("table_name", self._primary_table(config))
            # Defensive: never allow internal synthetic filename through
            if table_name == "synthetic_tableau_data":
                table_name = "Data"

        # Propagate normalized name back into the profile so _write_tables_tmdl
        # and all other downstream readers agree on the table name.
        if data_profile.get("table_name") != table_name:
            data_profile = {**data_profile, "table_name": table_name}

        # Column semantic types (dimension vs measure) from data profile
        col_types = {}
        col_cardinality = {}
        profile_col_names = set()
        for col_info in data_profile.get("columns", []):
            col_types[col_info["name"]] = col_info.get("semantic_type", "dimension")
            col_cardinality[col_info["name"]] = col_info.get("unique_count", 0)
            profile_col_names.add(col_info["name"])
        self._col_cardinality = col_cardinality

        # --- Build Semantic Model FIRST so we know which measures are valid ---

        # 9. {name}.SemanticModel/.platform
        self._write_platform(model_root, title, "SemanticModel")

        # 10. definition.pbism
        self._write_pbism(model_root)

        # 11. database.tmdl
        self._write_text(model_def / "database.tmdl", DATABASE_TMDL)

        # 12. cultures/en-US.tmdl
        self._write_text(cultures_dir / "en-US.tmdl", CULTURE_TMDL)

        # 13. Table TMDL files (returns table_names AND validated measure names)
        tm = config.get("tmdl_model", {})
        table_names, valid_measure_names = self._write_tables_tmdl(
            tables_dir, tm, data_profile, data_file_path, sheet_name,
            snowflake_config=snowflake_config,
            csv_url=csv_url,
        )

        # 14. model.tmdl (with relationships if provided)
        all_relationships = list(relationships or [])
        # Also check config for relationships (passed from agent)
        if config.get("relationships"):
            all_relationships.extend(config["relationships"])
        self._write_model_tmdl(model_def, table_names, all_relationships or None)

        # --- Now build Report side using only validated names ---

        # 1. {name}.pbip
        self._write_pbip(safe, project_dir)

        # 2. {name}.Report/.platform
        self._write_platform(report_root, title, "Report")

        # 3. {name}.Report/definition.pbir
        self._write_pbir(report_root, safe)

        # 4. {name}.Report/definition/report.json
        self._write_report_json(report_def)

        # 5. {name}.Report/definition/version.json
        self._write_version_json(report_def)

        # 6. Pages and visuals (using validated measure names + profile columns)
        rl = config.get("report_layout", {})
        raw_sections = rl.get("sections", [])
        page_ids, field_audit = self._write_pages(
            pages_dir, raw_sections, table_name, valid_measure_names,
            col_types, profile_col_names,
            dashboard_spec=dashboard_spec,
        )

        # Visual field audit summary
        print(f"[VISUAL AUDIT] {field_audit['valid']} valid fields, "
              f"{field_audit['fixed']} auto-fixed, "
              f"{field_audit['removed']} removed")

        # 7. pages.json
        self._write_pages_json(pages_dir, page_ids)

        # 8. Theme file -- Sprint 2: use color_extractor + visual_styler for
        # enhanced theme when deep color data is available, fallback to
        # design_translator for basic cases.
        tableau_design = dashboard_spec.get("design") or {}
        unified_palette = dashboard_spec.get("_unified_palette", [])
        if unified_palette:
            from core.visual_styler import build_enhanced_theme
            from core.design_translator import translate_fonts
            fonts = translate_fonts(tableau_design.get("global_fonts", {}))
            custom_theme = build_enhanced_theme(unified_palette, fonts, title)
        else:
            from core.design_translator import build_pbi_theme
            custom_theme = build_pbi_theme(tableau_design, title)
        self._write_json(theme_dir / "CY25SU12.json", custom_theme)

        # 15. Auto-launcher .bat (patches data path + opens .pbip)
        data_filename = ""
        if data_file_path:
            try:
                data_filename = Path(data_file_path).name
            except Exception:
                data_filename = str(data_file_path).split("/")[-1].split("\\")[-1]
        self._write_launcher(project_dir, safe, data_filename)

        # 16. README for the user
        self._write_readme(project_dir, safe, title)

        # Summary
        total = sum(1 for _ in project_dir.rglob("*") if _.is_file())
        print(f"[OK] PBIP project generated: {project_dir}")
        print(f"     {total} files created")

        if not page_ids:
            print("[ERROR] PBIP has 0 pages -- output will be empty in Power BI")

        return {
            "path": str(project_dir),
            "field_audit": field_audit,
            "table_names": list(table_names),
            "valid_measures": list(valid_measure_names),
            "page_count": len(page_ids),
            "file_count": total,
        }

    # ------------------------------------------------------------------ #
    #  {name}.pbip                                                        #
    # ------------------------------------------------------------------ #

    def _write_pbip(self, safe, project_dir):
        doc = deepcopy(PBIP_TEMPLATE)
        doc["artifacts"][0]["report"]["path"] = f"{safe}.Report"
        self._write_json(project_dir / f"{safe}.pbip", doc)
        print(f"    [+] {safe}.pbip")

    def _write_launcher(self, project_dir, safe, data_filename):
        """Write Open_Dashboard.bat + setup.ps1 for one-click opening.

        The launcher:
        1. setup.ps1 detects the extraction folder
        2. Replaces the placeholder C:\\PBI_Data\\ in all .tmdl files
           with the actual folder so PBI Desktop finds the data file
        3. Opens the .pbip in Power BI Desktop
        """
        # PowerShell setup script (reliable string replacement)
        ps1 = (
            '$folder = Split-Path -Parent $MyInvocation.MyCommand.Path\r\n'
            '$folder = $folder + "\\"\r\n'
            'Get-ChildItem -Path $folder -Recurse -Filter "*.tmdl" '
            '| ForEach-Object {\r\n'
            '    $c = [IO.File]::ReadAllText($_.FullName)\r\n'
            '    if ($c.Contains("C:\\PBI_Data\\")) {\r\n'
            '        $c = $c.Replace("C:\\PBI_Data\\", $folder)\r\n'
            '        [IO.File]::WriteAllText($_.FullName, $c)\r\n'
            '        Write-Host "  Updated:" $_.Name\r\n'
            '    }\r\n'
            '}\r\n'
        )
        (project_dir / "setup.ps1").write_text(ps1, encoding="utf-8")

        # Batch launcher (calls PS1 then opens .pbip)
        bat = (
            '@echo off\r\n'
            'cd /d "%~dp0"\r\n'
            'echo Setting up data source path...\r\n'
            'powershell -NoProfile -ExecutionPolicy Bypass '
            '-File "setup.ps1"\r\n'
            'echo.\r\n'
            f'echo Opening {safe}.pbip in Power BI Desktop...\r\n'
            f'start "" "{safe}.pbip"\r\n'
        )
        (project_dir / "Open_Dashboard.bat").write_text(bat, encoding="utf-8")
        print(f"    [+] Open_Dashboard.bat + setup.ps1")

    def _write_readme(self, project_dir, safe, title):
        """Write a README so the user knows how to open the project."""
        readme = (
            f"Power BI Project: {title}\n"
            f"{'=' * (len(title) + 21)}\n\n"
            f"HOW TO OPEN:\n"
            f"  1. Extract this entire ZIP to a folder on your computer\n"
            f"  2. Double-click: Open_Dashboard.bat\n"
            f"     (This sets up the data source path and opens Power BI)\n\n"
            f"REQUIREMENTS:\n"
            f"  - Power BI Desktop (January 2026 or later)\n"
            f"  - Enable preview features if not already enabled:\n"
            f"    File > Options > Preview features > check all PBIP options\n\n"
            f"FILES:\n"
            f"  Open_Dashboard.bat        -- Double-click this to open\n"
            f"  {safe}.pbip               -- Power BI project file\n"
            f"  {safe}.Report/            -- Report layout, pages, visuals\n"
            f"  {safe}.SemanticModel/     -- Data model, DAX measures\n"
        )
        self._write_text(project_dir / "README.txt", readme)

    # ------------------------------------------------------------------ #
    #  .platform                                                           #
    # ------------------------------------------------------------------ #

    def _write_platform(self, folder, display_name, artifact_type):
        doc = deepcopy(PLATFORM_TEMPLATE)
        doc["metadata"]["type"] = artifact_type
        doc["metadata"]["displayName"] = display_name
        doc["config"]["logicalId"] = str(uuid.uuid4())
        self._write_json(folder / ".platform", doc)

    # ------------------------------------------------------------------ #
    #  definition.pbir                                                     #
    # ------------------------------------------------------------------ #

    def _write_pbir(self, report_root, safe):
        doc = deepcopy(PBIR_TEMPLATE)
        doc["datasetReference"]["byPath"]["path"] = f"../{safe}.SemanticModel"
        self._write_json(report_root / "definition.pbir", doc)
        print(f"    [+] definition.pbir")

    # ------------------------------------------------------------------ #
    #  definition/report.json                                              #
    # ------------------------------------------------------------------ #

    def _write_report_json(self, report_def):
        doc = deepcopy(REPORT_JSON_TEMPLATE)
        self._write_json(report_def / "report.json", doc)
        print(f"    [+] report.json")

    # ------------------------------------------------------------------ #
    #  definition/version.json                                             #
    # ------------------------------------------------------------------ #

    def _write_version_json(self, report_def):
        self._write_json(report_def / "version.json", deepcopy(VERSION_JSON_TEMPLATE))

    # ------------------------------------------------------------------ #
    #  definition/pages/pages.json                                         #
    # ------------------------------------------------------------------ #

    def _write_pages_json(self, pages_dir, page_ids):
        doc = deepcopy(PAGES_JSON_TEMPLATE)
        doc["pageOrder"] = page_ids
        doc["activePageName"] = page_ids[0] if page_ids else ""
        self._write_json(pages_dir / "pages.json", doc)

    # ------------------------------------------------------------------ #
    #  Pages and visuals                                                   #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Deterministic grid layout (applied as final pass over GPT-4 output) #
    # ------------------------------------------------------------------ #

    # Canvas and spacing constants for deterministic layout
    _CANVAS_W = 1280
    _CANVAS_H = 720
    _MARGIN = 12       # Edge margin
    _GAP = 8           # Gap between visuals
    _SLICER_H = 52     # Height of a slicer row
    _CARD_H = 80       # Height of a KPI card row
    _MIN_VIS_H = 100   # Minimum height for a chart visual

    # Visual types classified as KPI cards
    _CARD_TYPES = {"card", "kpi", "multiRowCard"}
    # Visual types classified as slicers
    _SLICER_TYPES = {"slicer"}

    @classmethod
    def _deterministic_layout(cls, visual_containers):
        """Re-compute x/y/width/height for all visuals using a clean grid.

        Layout strategy:
          - Row 1 (top):    Slicer visuals in an evenly-spaced horizontal row
          - Row 2:          KPI / card visuals in an evenly-spaced horizontal row
          - Remaining area: Main charts in a 2-column grid

        All positions respect canvas bounds (1280x720), margins (12px edges),
        and gaps (8px between visuals). No visual will overlap another or
        exceed the canvas.

        Args:
            visual_containers: list of dicts, each with at least "config"
                (str or dict with "visualType").

        Returns:
            list of dicts -- same containers with x/y/width/height updated.
        """
        if not visual_containers:
            return visual_containers

        CW = cls._CANVAS_W
        CH = cls._CANVAS_H
        M = cls._MARGIN
        G = cls._GAP

        # Classify visuals into buckets while preserving original order index
        slicers = []
        cards = []
        charts = []

        for i, vc in enumerate(visual_containers):
            vtype = cls._extract_visual_type(vc)
            if vtype in cls._SLICER_TYPES:
                slicers.append((i, vc))
            elif vtype in cls._CARD_TYPES:
                cards.append((i, vc))
            else:
                charts.append((i, vc))

        usable_w = CW - 2 * M
        cursor_y = M

        # -- Slicer row (top) --
        if slicers:
            n = len(slicers)
            sw = (usable_w - (n - 1) * G) // n
            for idx, (orig_i, vc) in enumerate(slicers):
                vc["x"] = M + idx * (sw + G)
                vc["y"] = cursor_y
                vc["width"] = sw
                vc["height"] = cls._SLICER_H
            cursor_y += cls._SLICER_H + G

        # -- KPI card row --
        if cards:
            n = len(cards)
            cw = (usable_w - (n - 1) * G) // n
            for idx, (orig_i, vc) in enumerate(cards):
                vc["x"] = M + idx * (cw + G)
                vc["y"] = cursor_y
                vc["width"] = cw
                vc["height"] = cls._CARD_H
            cursor_y += cls._CARD_H + G

        # -- Main charts: 2-column grid in remaining space --
        if charts:
            remaining_h = CH - cursor_y - M
            n = len(charts)
            cols = 2 if n > 1 else 1
            rows = (n + cols - 1) // cols
            cell_w = (usable_w - (cols - 1) * G) // cols
            cell_h = max(cls._MIN_VIS_H, (remaining_h - (rows - 1) * G) // rows)

            # Clamp: ensure charts do not exceed canvas bottom
            if cursor_y + rows * cell_h + (rows - 1) * G > CH - M:
                cell_h = max(
                    cls._MIN_VIS_H,
                    (CH - M - cursor_y - (rows - 1) * G) // rows,
                )

            for idx, (orig_i, vc) in enumerate(charts):
                row = idx // cols
                col = idx % cols
                vc["x"] = M + col * (cell_w + G)
                vc["y"] = cursor_y + row * (cell_h + G)
                vc["width"] = cell_w
                vc["height"] = cell_h

        print(f"    [LAYOUT] Deterministic grid: "
              f"{len(slicers)} slicers, {len(cards)} cards, "
              f"{len(charts)} charts")

        return visual_containers

    @staticmethod
    def _extract_visual_type(vc):
        """Extract the visualType string from a visual container dict."""
        raw_cfg = vc.get("config", "{}")
        if isinstance(raw_cfg, str):
            try:
                cfg_obj = json.loads(raw_cfg)
            except (json.JSONDecodeError, TypeError):
                cfg_obj = {}
        else:
            cfg_obj = raw_cfg if isinstance(raw_cfg, dict) else {}
        return cfg_obj.get("visualType", "unknown").lower()

    def _write_pages(self, pages_dir, raw_sections, table_name, measure_names,
                     col_types=None, profile_col_names=None,
                     dashboard_spec=None):
        """Write page.json and visual.json files.

        Args:
            dashboard_spec: optional full dashboard_spec dict. When present,
                Tableau zone positions and worksheet designs are used for
                layout and visual formatting instead of the deterministic grid.

        Returns (page_ids, audit_totals) where audit_totals is
        {"valid": int, "fixed": int, "removed": int}.
        """
        if dashboard_spec is None:
            dashboard_spec = {}

        # Pre-compute Tableau zone positions if available
        tableau_positions = {}
        tableau_dashboards = dashboard_spec.get("tableau_dashboards", [])
        if tableau_dashboards:
            from core.design_translator import translate_positions
            db = tableau_dashboards[0]  # primary dashboard
            zones = db.get("zones", [])
            canvas = db.get("canvas", {})
            if zones and canvas.get("width") and canvas.get("height"):
                pbi_positions = translate_positions(zones, canvas)
                for p in pbi_positions:
                    tableau_positions[p["name"]] = p

        # Pre-compute worksheet designs for visual formatting
        ws_designs = dashboard_spec.get("worksheet_designs", {})

        # Pre-compute Tableau chart type mapping (worksheet name -> mark type)
        ws_chart_types = dashboard_spec.get("worksheet_chart_types", {})

        page_ids = []
        audit_totals = {"valid": 0, "fixed": 0, "removed": 0}

        for idx, raw in enumerate(raw_sections):
            page_id = uuid.uuid4().hex[:20]
            page_ids.append(page_id)
            display_name = raw.get("displayName", raw.get("name", f"Page {idx + 1}"))

            page_dir = pages_dir / page_id
            page_dir.mkdir(exist_ok=True)

            # page.json
            page = deepcopy(PAGE_JSON_TEMPLATE)
            page["name"] = page_id
            page["displayName"] = display_name
            page["width"] = int(raw.get("width", 1280))
            page["height"] = int(raw.get("height", 720))
            self._write_json(page_dir / "page.json", page)

            containers = list(raw.get("visualContainers", []))

            # --- Layout pass ---
            # If Tableau zone positions are available, apply them by matching
            # zone names to visual titles. Otherwise use deterministic grid.
            applied_tableau_layout = False
            if tableau_positions:
                applied_tableau_layout = self._apply_tableau_positions(
                    containers, tableau_positions
                )

            if not applied_tableau_layout:
                containers = self._deterministic_layout(containers)

            # Inject Tableau mark types into visual containers
            if ws_chart_types:
                for vc in containers:
                    raw_cfg = vc.get("config", "{}")
                    if isinstance(raw_cfg, str):
                        try:
                            cfg_obj = json.loads(raw_cfg)
                        except (json.JSONDecodeError, TypeError):
                            cfg_obj = {}
                    else:
                        cfg_obj = raw_cfg if isinstance(raw_cfg, dict) else {}
                    # Match by title or name to worksheet chart type
                    title = cfg_obj.get("title", "").lower().strip()
                    ws_name = cfg_obj.get("worksheet_name", "").lower().strip()
                    for ws, ct in ws_chart_types.items():
                        ws_lower = ws.lower().strip()
                        if ws_lower in (title, ws_name) or ws_lower in title:
                            cfg_obj["tableau_mark_type"] = ct
                            vc["config"] = cfg_obj
                            break

            # Visuals
            visuals_root = page_dir / "visuals"
            vc_count = 0
            for vidx, vc in enumerate(containers):
                viz_id = uuid.uuid4().hex[:20]
                viz_dir = visuals_root / viz_id
                viz_dir.mkdir(parents=True, exist_ok=True)
                fix_stats = self._write_visual_json(
                    viz_dir, vc, vidx, viz_id,
                    table_name, measure_names, col_types,
                    profile_col_names,
                    ws_designs=ws_designs,
                )
                if fix_stats:
                    for k in audit_totals:
                        audit_totals[k] += fix_stats.get(k, 0)
                vc_count += 1

            layout_src = "Tableau zones" if applied_tableau_layout else "deterministic grid"
            if vc_count == 0:
                print(f"    [WARNING] page '{display_name}': 0 visuals -- page will be empty in Power BI")
            else:
                print(f"    [+] page '{display_name}': {vc_count} visuals ({layout_src})")

        return page_ids, audit_totals

    @classmethod
    def _apply_tableau_positions(cls, containers, tableau_positions):
        """Try to apply Tableau zone positions to visual containers.

        Matches zone names to visual config titles or worksheet names.
        Returns True if at least one container was positioned from Tableau data.
        """
        if not tableau_positions or not containers:
            return False

        matched = 0
        for vc in containers:
            raw_cfg = vc.get("config", "{}")
            if isinstance(raw_cfg, str):
                try:
                    cfg_obj = json.loads(raw_cfg)
                except (json.JSONDecodeError, TypeError):
                    cfg_obj = {}
            else:
                cfg_obj = raw_cfg if isinstance(raw_cfg, dict) else {}

            # Try to match by title, then by worksheet_name
            title = cfg_obj.get("title", "")
            ws_name = cfg_obj.get("worksheet_name", "")

            pos = tableau_positions.get(title) or tableau_positions.get(ws_name)
            if pos:
                vc["x"] = pos["x"]
                vc["y"] = pos["y"]
                vc["width"] = pos["width"]
                vc["height"] = pos["height"]
                matched += 1

        if matched > 0:
            print(f"    [LAYOUT] Applied Tableau zone positions to {matched} visuals")
            return True
        return False

    def _write_visual_json(self, viz_dir, vc, vidx, viz_id, table_name,
                           measure_names, col_types=None,
                           profile_col_names=None, ws_designs=None):
        """Write a single visual.json file."""
        if ws_designs is None:
            ws_designs = {}

        x = vc.get("x", 0)
        y = vc.get("y", 0)
        w = vc.get("width", 300)
        h = vc.get("height", 200)

        # Parse the simplified config from OpenAI
        raw_cfg = vc.get("config", "{}")
        if isinstance(raw_cfg, str):
            try:
                cfg_obj = json.loads(raw_cfg)
            except json.JSONDecodeError:
                cfg_obj = {}
        else:
            cfg_obj = raw_cfg

        # Chart type: prefer Tableau-translated type over GPT-4 guess
        tableau_mark = cfg_obj.get("tableau_mark_type", "")
        if tableau_mark:
            from core.design_translator import translate_chart_type
            visual_type = translate_chart_type(tableau_mark)
        else:
            visual_type = cfg_obj.get("visualType", "card")

        # Safety net: only allow known PBI visual types through.
        # Tenant has all visuals enabled (maps, AI, custom, R/Python).
        _SAFE_VISUALS = {
            "clusteredBarChart", "clusteredColumnChart", "stackedBarChart",
            "stackedColumnChart", "hundredPercentStackedBarChart",
            "hundredPercentStackedColumnChart", "lineChart", "pieChart",
            "donutChart", "card", "cardVisual", "multiRowCard", "areaChart",
            "stackedAreaChart", "scatterChart", "tableEx", "matrix",
            "treemap", "lineClusteredColumnComboChart",
            "lineStackedColumnComboChart", "waterfallChart", "funnelChart",
            "gauge", "kpi", "slicer", "textbox", "image", "shape",
            "decompositionTreeVisual", "ribbonChart",
            "filledMap", "map", "shapeMap",
        }
        if visual_type not in _SAFE_VISUALS:
            visual_type = "clusteredBarChart"

        title_text = cfg_obj.get("title", "")
        data_roles = cfg_obj.get("dataRoles", {})

        doc = deepcopy(VISUAL_JSON_TEMPLATE)
        doc["name"] = viz_id
        doc["position"] = {
            "x": x, "y": y, "z": vidx,
            "height": h, "width": w,
            "tabOrder": vidx,
        }
        doc["visual"]["visualType"] = visual_type
        doc["visual"]["query"]["queryState"] = self._build_query_state(
            visual_type, data_roles, table_name, measure_names, col_types,
            profile_col_names,
        )

        # NOTE: Do NOT inject visual.objects (title, categoryAxis, valueAxis,
        # labels). PBI Desktop auto-generates default formatting. Injecting
        # objects can cause visuals to render as blank if the format is not
        # exactly what PBI expects for each visual type.

        # Write PBIR visual-level filters from Tableau filter_values.
        # Fabric rejects the older query.filterState payload during publish.
        filter_values = cfg_obj.get("filter_values", [])
        if filter_values and profile_col_names:
            # Find the most likely filter field: first dimension in the visual
            filter_field = None
            for role_key in ("category", "Category", "series", "Series"):
                fields = data_roles.get(role_key, [])
                if fields:
                    filter_field = fields[0]
                    break
            # Also check column_instances for a nominal field used as filter
            if not filter_field:
                dims = data_roles.get("category", data_roles.get("Category", []))
                if dims:
                    filter_field = dims[0]
            if filter_field and filter_field in (profile_col_names | measure_names):
                value_literals = []
                for value in filter_values[:10]:
                    if isinstance(value, bool):
                        literal = "true" if value else "false"
                    elif isinstance(value, (int, float)) and not isinstance(value, bool):
                        literal = str(value)
                    else:
                        escaped = str(value).replace("'", "''")
                        literal = f"'{escaped}'"
                    value_literals.append([{"Literal": {"Value": literal}}])

                pbi_filter = {
                    "name": f"visualFilter_{viz_id}_{filter_field}",
                    "displayName": filter_field,
                    "field": {
                        "Column": {
                            "Expression": {"SourceRef": {"Entity": table_name}},
                            "Property": filter_field,
                        }
                    },
                    "type": "Categorical",
                    "howCreated": "User",
                    "isHiddenInViewMode": False,
                    "isLockedInViewMode": False,
                    "filter": {
                        "Version": 2,
                        "From": [{"Name": "d", "Entity": table_name, "Type": 0}],
                        "Where": [{
                            "Condition": {
                                "In": {
                                    "Expressions": [{
                                        "Column": {
                                            "Expression": {"SourceRef": {"Source": "d"}},
                                            "Property": filter_field,
                                        }
                                    }],
                                    "Values": value_literals,
                                }
                            }
                        }],
                    },
                }
                doc["filterConfig"] = {
                    "filters": [pbi_filter],
                }

        # Validate & fix all field references against real column names
        valid_cols = list(profile_col_names | measure_names) if profile_col_names else []
        doc, fix_stats = self._fix_field_references(doc, valid_cols, table_name)

        self._write_json(viz_dir / "visual.json", doc)

        return fix_stats

    # ------------------------------------------------------------------ #
    #  Visual query state                                                  #
    # ------------------------------------------------------------------ #

    def _build_query_state(self, visual_type, data_roles, table_name,
                           measure_names, col_types=None,
                           profile_col_names=None):
        """Build queryState matching PBI Desktop format.

        PBI Desktop wraps numeric columns in Aggregation blocks (Function 0 = Sum)
        when they appear in value roles (Y, Values).  Category/axis roles get
        plain Column refs.  This matches the reference_template output.

        Fields that don't exist as columns or valid measures are skipped.
        """
        if col_types is None:
            col_types = {}
        if profile_col_names is None:
            profile_col_names = set(col_types.keys())

        # All valid field names = real columns + validated measures
        valid_fields = profile_col_names | measure_names

        # Choose role mapping based on visual type
        if visual_type in ("card", "cardVisual", "multiRowCard", "kpi"):
            role_map = self.ROLE_MAP_CARD
        elif visual_type in ("tableEx", "pivotTable", "table"):
            role_map = self.ROLE_MAP_TABLE
        elif visual_type == "slicer":
            role_map = self.ROLE_MAP_SLICER
        elif visual_type in ("filledMap", "map", "shapeMap"):
            role_map = self.ROLE_MAP_MAP
        elif visual_type == "scatterChart":
            role_map = self.ROLE_MAP_SCATTER
        else:
            role_map = self.ROLE_MAP_CHART

        # Value roles get aggregated, category roles do not
        value_roles = {"Y", "Values", "Tooltips"}
        category_roles = {"Category", "X", "Series", "Rows", "Columns", "Location", "Legend"}
        # filledMap: "Category" is the color-saturation bucket (a measure),
        # not a dimension axis. Move it to value_roles so it gets aggregated.
        if visual_type in ("filledMap", "map", "shapeMap"):
            value_roles = value_roles | {"Category"}
            category_roles = category_roles - {"Category"}
        # Max unique values allowed on a category axis before PBI throws
        # DataViewMappingError_ConditionRangeTooLarge.
        # Date columns are exempted because PBI auto-hierarchies handle them.
        # String dimensions with extreme cardinality (>30000) are still blocked.
        MAX_CATEGORY_CARDINALITY = 30000
        col_card = getattr(self, "_col_cardinality", {})

        # Extract derivation map from data_roles (if provided by direct_mapper)
        derivations = data_roles.get("derivations", {})

        query_state = {}
        for our_role, pbi_role in role_map.items():
            fields = data_roles.get(our_role, [])
            if not fields:
                continue

            aggregate = pbi_role in value_roles
            is_category = pbi_role in category_roles
            projections = []
            for field_name in fields:
                if field_name not in valid_fields:
                    print(f"    [SKIP] visual field '{field_name}': "
                          f"not in dataset columns or valid measures")
                    continue
                # Block extremely high-cardinality string dimensions on
                # category axes. Date columns are never blocked because
                # PBI handles date hierarchies natively.
                sem_type = (col_types or {}).get(field_name, "dimension")
                if (is_category and sem_type == "dimension"
                        and field_name in col_card):
                    card = col_card[field_name]
                    if card > MAX_CATEGORY_CARDINALITY:
                        print(f"    [SKIP] visual field '{field_name}': "
                              f"cardinality {card:,} exceeds {MAX_CATEGORY_CARDINALITY} "
                              f"for category axis (would cause ConditionRangeTooLarge)")
                        continue
                # Pass Tableau derivation (Sum/Avg/CountD/etc.) to field projection
                field_derivation = derivations.get(field_name)
                proj = self._build_field_projection(
                    field_name, table_name, measure_names,
                    aggregate=aggregate, col_types=col_types,
                    derivation=field_derivation,
                )
                projections.append(proj)

            if projections:
                query_state[pbi_role] = {"projections": projections}

        # Map-specific fallback: filledMap has Location but the color
        # saturation measure (Category) was dropped because it referenced
        # a Tableau Calculation_* ID not in the dataset. Assign the first
        # available measure so the map actually shades regions.
        if (visual_type in ("filledMap", "map", "shapeMap")
                and "Location" in query_state and "Category" not in query_state
                and profile_col_names):
            measures = sorted([c for c in profile_col_names
                               if col_types.get(c) == "measure"])
            if measures:
                proj = self._build_field_projection(
                    measures[0], table_name, measure_names,
                    aggregate=True, col_types=col_types,
                )
                query_state["Category"] = {"projections": [proj]}
                print(f"    [FALLBACK] filledMap: assigned '{measures[0]}' "
                      f"as color saturation (original calc field not in dataset)")

        # Fallback: if ALL requested fields were invalid and query_state is
        # empty, assign a default dimension + measure from the profile so the
        # visual is never blank. This handles Tableau Calculation_* IDs that
        # do not exist in the synthetic dataset.
        if not query_state and profile_col_names:
            dims = sorted([c for c in profile_col_names
                           if col_types.get(c) not in ("measure", "date")])
            measures = sorted([c for c in profile_col_names
                               if col_types.get(c) == "measure"])
            fallback_fields = []
            fallback_role = None

            if visual_type in ("card", "cardVisual", "multiRowCard", "kpi"):
                # Cards need a single value
                fallback_role = "Values"
                if measures:
                    fallback_fields = [measures[0]]
                elif dims:
                    fallback_fields = [dims[0]]
            elif visual_type in ("tableEx", "pivotTable", "table"):
                fallback_role = "Values"
                fallback_fields = (dims[:1] + measures[:1]) or dims[:1] or measures[:1]
            else:
                # Charts: assign category + Y
                if dims:
                    cat_proj = self._build_field_projection(
                        dims[0], table_name, measure_names,
                        aggregate=False, col_types=col_types,
                    )
                    query_state["Category"] = {"projections": [cat_proj]}
                if measures:
                    val_proj = self._build_field_projection(
                        measures[0], table_name, measure_names,
                        aggregate=True, col_types=col_types,
                    )
                    query_state["Y"] = {"projections": [val_proj]}
                elif dims and len(dims) > 1:
                    val_proj = self._build_field_projection(
                        dims[1], table_name, measure_names,
                        aggregate=False, col_types=col_types,
                    )
                    query_state["Y"] = {"projections": [val_proj]}

            if fallback_role and fallback_fields:
                projs = []
                agg = fallback_role in value_roles
                for fn in fallback_fields:
                    projs.append(self._build_field_projection(
                        fn, table_name, measure_names,
                        aggregate=agg, col_types=col_types,
                    ))
                query_state[fallback_role] = {"projections": projs}

            if query_state:
                used = []
                for role_data in query_state.values():
                    for p in role_data.get("projections", []):
                        used.append(p.get("nativeQueryRef", "?"))
                print(f"    [FALLBACK] Assigned default fields for empty visual: "
                      f"{', '.join(used)}")

        # ── Enforce PBI visual type field limits ──
        _ROLE_LIMITS = {
            "card":              {"Values": 1},
            "cardVisual":        {"Values": 1},
            "multiRowCard":      {"Values": 4},
            "kpi":               {"Values": 1},
            "clusteredBarChart": {"Category": 1, "Y": 3, "Series": 1},
            "barChart":          {"Category": 1, "Y": 2, "Series": 1},
            "stackedBarChart":   {"Category": 1, "Y": 2, "Legend": 1},
            "hundredPercentStackedBarChart": {"Category": 1, "Y": 2, "Legend": 1},
            "stackedColumnChart": {"Category": 1, "Y": 2, "Legend": 1},
            "hundredPercentStackedColumnChart": {"Category": 1, "Y": 2, "Legend": 1},
            "clusteredColumnChart": {"Category": 1, "Y": 3, "Series": 1},
            "scatterChart":      {"X": 1, "Y": 1, "Size": 1, "Details": 1},
            "lineChart":         {"Category": 1, "Y": 3, "Series": 1},
            "areaChart":         {"Category": 1, "Y": 3, "Series": 1},
            "stackedAreaChart":  {"Category": 1, "Y": 3, "Series": 1},
            "pieChart":          {"Category": 1, "Y": 1, "Legend": 1},
            "donutChart":        {"Category": 1, "Y": 1, "Legend": 1},
            "treemap":           {"Category": 1, "Values": 1, "Legend": 1},
            "filledMap":         {"Location": 1, "Category": 1, "Legend": 1},
            "map":               {"Location": 1, "Category": 1, "Legend": 1},
            "shapeMap":          {"Location": 1, "Category": 1, "Legend": 1},
            "funnelChart":       {"Category": 1, "Y": 1},
            "waterfallChart":    {"Category": 1, "Y": 1},
            "slicer":            {"Values": 1},
        }
        limits = _ROLE_LIMITS.get(visual_type, {})
        for role, max_count in limits.items():
            if role in query_state:
                projs = query_state[role].get("projections", [])
                if len(projs) > max_count:
                    dropped = [p.get("nativeQueryRef", "?") for p in projs[max_count:]]
                    query_state[role]["projections"] = projs[:max_count]
                    print(f"    [LIMIT] {visual_type}.{role}: truncated to {max_count} "
                          f"(dropped: {', '.join(dropped)})")

        return query_state

    # ── Aggregation function mapping (Tableau derivation → PBI Function enum) ──
    _AGG_FUNCTION_MAP = {
        "sum": 0,
        "avg": 1, "average": 1,
        "count": 2, "cnt": 2,
        "min": 3,
        "max": 4,
        "countd": 5, "distinctcount": 5,
    }

    def _build_field_projection(self, field_name, table_name, measure_names,
                                aggregate=False, col_types=None,
                                derivation=None):
        """Build a single field projection for queryState.

        Matches the reference_template structure from PBI Desktop:
        - Measures: Measure ref (no aggregation)
        - Numeric columns in value roles: Aggregation wrapper (Function 0 = Sum)
        - Dimension columns: plain Column ref
        """
        if col_types is None:
            col_types = {}

        if field_name in measure_names:
            # Measure reference (no aggregation wrapper)
            return {
                "field": {
                    "Measure": {
                        "Expression": {"SourceRef": {"Entity": table_name}},
                        "Property": field_name,
                    }
                },
                "queryRef": f"{table_name}.{field_name}",
                "nativeQueryRef": field_name,
            }

        col_ref = {
            "Column": {
                "Expression": {"SourceRef": {"Entity": table_name}},
                "Property": field_name,
            }
        }

        # Wrap numeric columns in Aggregation when in value roles.
        # Use derivation from Tableau parser when available, else default Sum.
        sem = col_types.get(field_name, "dimension")
        if aggregate and sem == "measure":
            # Resolve aggregation function from Tableau derivation
            agg_func = 0  # Default: Sum
            agg_label = "Sum"
            if derivation:
                deriv_lower = derivation.lower().strip()
                mapped = self._AGG_FUNCTION_MAP.get(deriv_lower)
                if mapped is not None:
                    agg_func = mapped
                    agg_label = derivation.capitalize()

            return {
                "field": {
                    "Aggregation": {
                        "Expression": col_ref,
                        "Function": agg_func,
                    }
                },
                "queryRef": f"{agg_label}({table_name}.{field_name})",
                "nativeQueryRef": f"{agg_label} of {field_name}",
            }

        # Plain column reference for dimensions / category axes
        return {
            "field": col_ref,
            "queryRef": f"{table_name}.{field_name}",
            "nativeQueryRef": field_name,
        }

    # ------------------------------------------------------------------ #
    #  Field reference validation & auto-fix                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fuzzy_match_column(name, valid_columns):
        """Find the best matching column name from valid_columns.

        Tries:
          1. Case-insensitive exact match
          2. Stripped/collapsed whitespace match
          3. Best substring / edit-distance heuristic
        Returns the matched column name or None.
        """
        if not name or not valid_columns:
            return None

        name_lower = name.lower().strip()

        # 1. Case-insensitive exact match
        for vc in valid_columns:
            if vc.lower().strip() == name_lower:
                return vc

        # 2. Whitespace-collapsed match  (e.g. "DishCategory" vs "Dish Category")
        def collapse(s):
            return "".join(s.lower().split())

        name_collapsed = collapse(name)
        for vc in valid_columns:
            if collapse(vc) == name_collapsed:
                return vc

        # 3. Underscore/space normalised  (e.g. "dish_name" vs "Dish Name")
        def normalise(s):
            return s.lower().replace("_", "").replace("-", "").replace(" ", "")

        name_norm = normalise(name)
        for vc in valid_columns:
            if normalise(vc) == name_norm:
                return vc

        return None

    def _fix_field_references(self, visual_config, valid_columns, table_name):
        """Recursively walk a visual config and fix/remove broken field refs.

        Returns (fixed_config, stats) where stats is a dict with counts:
          valid, fixed, removed
        """
        stats = {"valid": 0, "fixed": 0, "removed": 0}
        valid_set = set(valid_columns)

        def _fix_property(node, parent_list=None, parent_idx=None):
            """Fix a Property value inside a Column/Measure/Aggregation node."""
            if isinstance(node, dict):
                # Fix Entity (table name) references
                if "Entity" in node:
                    old_entity = node["Entity"]
                    if old_entity != table_name:
                        print(f"    [FIELD FIX] Entity '{old_entity}' -> '{table_name}'")
                        node["Entity"] = table_name

                # Fix Property (column/field name) references
                if "Property" in node:
                    prop = node["Property"]
                    if prop in valid_set:
                        stats["valid"] += 1
                    else:
                        match = self._fuzzy_match_column(prop, valid_columns)
                        if match:
                            print(f"    [FIELD FIX] {prop} -> {match}")
                            node["Property"] = match
                            stats["fixed"] += 1
                        else:
                            print(f"    [FIELD REMOVED] {prop} - no match in data")
                            stats["removed"] += 1
                            # Signal removal to parent
                            return False

                for v in node.values():
                    result = _fix_property(v, None, None)
                    if result is False:
                        return False

            elif isinstance(node, list):
                to_remove = []
                for i, item in enumerate(node):
                    result = _fix_property(item, node, i)
                    if result is False:
                        to_remove.append(i)
                for i in reversed(to_remove):
                    node.pop(i)

            return True

        # Walk the queryState projections
        query_state = (visual_config.get("visual", {})
                       .get("query", {})
                       .get("queryState", {}))
        roles_to_remove = []
        for role_name, role_data in query_state.items():
            projections = role_data.get("projections", [])
            to_remove = []
            for idx, proj in enumerate(projections):
                field = proj.get("field", {})
                result = _fix_property(field)
                if result is False:
                    to_remove.append(idx)
                else:
                    # Also fix queryRef and nativeQueryRef to match
                    self._sync_query_refs(proj, table_name)
            for i in reversed(to_remove):
                projections.pop(i)
            if not projections:
                roles_to_remove.append(role_name)
        for role_name in roles_to_remove:
            del query_state[role_name]

        return visual_config, stats

    @staticmethod
    def _sync_query_refs(projection, table_name):
        """Re-derive queryRef and nativeQueryRef from the field structure."""
        field = projection.get("field", {})

        if "Measure" in field:
            prop = field["Measure"].get("Property", "")
            projection["queryRef"] = f"{table_name}.{prop}"
            projection["nativeQueryRef"] = prop

        elif "Aggregation" in field:
            expr = field["Aggregation"].get("Expression", {})
            col = expr.get("Column", expr.get("column", {}))
            prop = col.get("Property", "")
            projection["queryRef"] = f"Sum({table_name}.{prop})"
            projection["nativeQueryRef"] = f"Sum of {prop}"

        elif "Column" in field:
            prop = field["Column"].get("Property", "")
            projection["queryRef"] = f"{table_name}.{prop}"
            projection["nativeQueryRef"] = prop

    # ------------------------------------------------------------------ #
    #  Theme file                                                          #
    # ------------------------------------------------------------------ #

    def _write_theme(self, theme_dir):
        """Copy the CY25SU12 theme from the reference template."""
        src = self.project_root / THEME_SOURCE_PATH
        dst = theme_dir / "CY25SU12.json"
        if src.exists():
            shutil.copy2(src, dst)
        else:
            # Fallback: write a minimal default theme
            theme = {
                "name": "CY25SU12",
                "dataColors": [
                    "#118DFF", "#12239E", "#E66C37", "#6B007B",
                    "#E044A7", "#744EC2", "#D9B300", "#D64550",
                ],
                "foreground": "#252423",
                "background": "#FFFFFF",
                "tableAccent": "#118DFF",
            }
            self._write_json(dst, theme)

    # ------------------------------------------------------------------ #
    #  definition.pbism                                                    #
    # ------------------------------------------------------------------ #

    def _write_pbism(self, model_root):
        self._write_json(model_root / "definition.pbism", deepcopy(PBISM_TEMPLATE))
        print(f"    [+] definition.pbism")

    # ------------------------------------------------------------------ #
    #  TMDL files                                                          #
    # ------------------------------------------------------------------ #

    def _write_model_tmdl(self, model_def, table_names, relationships=None):
        """Write definition/model.tmdl."""
        lines = [
            "model Model",
            "\tculture: en-US",
            "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
            "\tsourceQueryCulture: en-US",
            "\tdataAccessOptions",
            "\t\tlegacyRedirects",
            "\t\treturnErrorValuesAsNull",
            "",
            "annotation __PBI_TimeIntelligenceEnabled = 1",
            "",
            f'annotation PBI_QueryOrder = {json.dumps(table_names)}',
            "",
            'annotation PBI_ProTooling = ["DevMode"]',
            "",
        ]
        for tbl in table_names:
            lines.append(f"ref table {self._tmdl_quote(tbl)}")
        lines.append("")

        # Relationships
        rel_lines = self._generate_relationships(relationships, table_names)
        if rel_lines:
            lines.extend(rel_lines)
            lines.append("")

        lines.append("ref cultureInfo en-US")
        lines.append("")

        self._write_text(model_def / "model.tmdl", "\n".join(lines) + "\n")
        rel_count = len(relationships) if relationships else 0
        print(f"    [+] model.tmdl ({len(table_names)} tables, {rel_count} relationships)")

    def _generate_relationships(self, relationships, table_names):
        """Generate TMDL relationship blocks from detected relationships.

        Args:
            relationships: list of dicts, each with keys:
                - from_table, from_column, to_table, to_column
                - Optional: cardinality ('many-to-one', 'one-to-many', etc.)
                - Or: left_table, right_table, left_col, right_col (RelationshipDetector format)
                - Or: left, right, type (Tableau parser format)
            table_names: list of table names actually present in the model.

        Returns:
            list of TMDL lines (empty if no valid relationships).
        """
        if not relationships:
            return []

        table_set = set(table_names)
        lines = []

        for rel in relationships:
            # Normalize the different relationship formats into from/to
            from_table, from_col, to_table, to_col = self._normalize_relationship(rel)

            if not all([from_table, from_col, to_table, to_col]):
                continue

            # Both tables must exist in the model
            if from_table not in table_set or to_table not in table_set:
                # If tables don't match exactly, check if one is a substring
                # (e.g., "Orders" matching "Orders" table)
                from_match = self._find_table_match(from_table, table_set)
                to_match = self._find_table_match(to_table, table_set)
                if from_match and to_match:
                    from_table = from_match
                    to_table = to_match
                else:
                    print(f"    [SKIP] relationship {from_table}[{from_col}] -> "
                          f"{to_table}[{to_col}]: table not in model")
                    continue

            from_q = self._tmdl_quote(from_table)
            to_q = self._tmdl_quote(to_table)
            from_col_q = self._tmdl_quote(from_col)
            to_col_q = self._tmdl_quote(to_col)

            lines.append(f"relationship {from_q}[{from_col_q}] -> {to_q}[{to_col_q}]")
            lines.append("\tcrossFilteringBehavior: oneDirection")
            lines.append("\tfromCardinality: many")
            lines.append("\ttoCardinality: one")
            lines.append("\tisActive")
            lines.append("\tsecurityFilteringBehavior: oneDirection")
            lines.append("")

        if lines:
            print(f"    [+] {len([l for l in lines if l.startswith('relationship')])} "
                  f"relationships generated")
        return lines

    @staticmethod
    def _normalize_relationship(rel):
        """Normalize different relationship dict formats to (from_table, from_col, to_table, to_col)."""
        # Format 1: explicit from/to keys (our canonical format)
        if "from_table" in rel:
            fc = rel.get("from_column", "")
            tt = rel.get("to_table", "")
            tc = rel.get("to_column", "")
            if not all([fc, tt, tc]):
                return (None, None, None, None)
            return (
                rel["from_table"], fc, tt, tc,
            )

        # Format 2: RelationshipDetector format (left/right)
        if "left_table" in rel:
            return (
                rel["left_table"], rel["left_col"],
                rel["right_table"], rel["right_col"],
            )

        # Format 3: Tableau parser format (left/right are "table.column" strings)
        if "left" in rel and "right" in rel:
            left = rel["left"]
            right = rel["right"]
            # Parse "table.column" or "[table].[column]" patterns
            from_table, from_col = _parse_tableau_field_ref(left)
            to_table, to_col = _parse_tableau_field_ref(right)
            return (from_table, from_col, to_table, to_col)

        return (None, None, None, None)

    @staticmethod
    def _find_table_match(name, table_set):
        """Find a table in table_set that matches name (case-insensitive, underscore-agnostic)."""
        norm = name.lower().replace(" ", "_").replace("-", "_")
        for t in table_set:
            t_norm = t.lower().replace(" ", "_").replace("-", "_")
            if norm == t_norm:
                return t
        return None

    @classmethod
    def _pandas_dtype_to_pbi(cls, dtype_str, semantic_type):
        """Map a pandas dtype string to a PBI dataType.

        Uses substring matching instead of an exact-key dict so that all
        int-like (int8, int16, int32, Int64, UInt32 ...) and float-like
        (float16, float32, Float64 ...) dtypes map correctly.  Measures
        are NEVER typed as string -- that would break DAX aggregation
        functions (SUM, AVERAGE, COUNT, etc.).
        """
        dl = dtype_str.lower()

        # 1. Dates
        if semantic_type == "date":
            return "dateTime"
        if "datetime" in dl or "timestamp" in dl:
            return "dateTime"

        # 2. Boolean
        if "bool" in dl:
            return "boolean"

        # 3. Integer-like
        if "int" in dl:
            return "int64"

        # 4. Float / double / decimal
        if "float" in dl or "double" in dl or "decimal" in dl:
            return "double"

        # 5. Safety net: measures must ALWAYS be numeric
        if semantic_type == "measure":
            return "double"

        return "string"

    def _write_tables_tmdl(self, tables_dir, tmdl_model, data_profile,
                           data_file_path, sheet_name=None,
                           snowflake_config=None, csv_url=None):
        """Write one .tmdl file per table.

        Returns:
            tuple: (list of table names, set of valid measure names)

        COLUMNS are built from data_profile (ground truth from actual data),
        not from the AI-generated tmdl_model -- this prevents column mismatches.
        MEASURES come from the AI but are validated against real columns.
        """
        tables_raw = tmdl_model.get("tables", [])
        table_names = []

        # Ground-truth columns from the data profile
        profile_columns = data_profile.get("columns", [])
        profile_col_names = {c["name"] for c in profile_columns}

        # The table name comes from the data profile (set from sheet name)
        profile_table_name = data_profile.get("table_name", "Data")

        # Collect measures from ALL AI-generated tables
        all_measures = []
        for tbl in tables_raw:
            all_measures.extend(tbl.get("measures", []))

        # We write exactly ONE table using the profiled data
        tbl_name = profile_table_name
        table_names.append(tbl_name)

        lines = []
        lineage = str(uuid.uuid4())
        quoted_tbl = self._tmdl_quote(tbl_name)
        lines.append(f"table {quoted_tbl}")
        lines.append(f"\tlineageTag: {lineage}")
        lines.append("")

        # Columns -- from data_profile (always matches the actual data)
        for col_info in profile_columns:
            col_name = self._strip_control_chars(col_info["name"])
            sem = col_info.get("semantic_type", "dimension")
            dtype_str = col_info.get("dtype", "object")
            dt = self._pandas_dtype_to_pbi(dtype_str, sem)

            quoted = self._tmdl_quote(col_name)
            lines.append(f"\tcolumn {quoted}")
            lines.append(f"\t\tdataType: {dt}")

            # Format string
            if dt == "int64":
                lines.append("\t\tformatString: 0")
            elif dt == "dateTime":
                lines.append("\t\tformatString: Long Date")

            lines.append(f"\t\tlineageTag: {uuid.uuid4()}")

            # Summarize
            if sem == "measure":
                lines.append("\t\tsummarizeBy: sum")
            elif dt == "int64" and sem == "dimension":
                lines.append("\t\tsummarizeBy: count")
            else:
                lines.append("\t\tsummarizeBy: none")

            lines.append(f"\t\tsourceColumn: {col_name}")
            lines.append("")
            lines.append("\t\tannotation SummarizationSetBy = Automatic")

            if dt == "double":
                lines.append("")
                lines.append('\t\tannotation PBI_FormatHint = {"isGeneralNumber":true}')

            lines.append("")

        # Measures -- from AI output, validated against real columns + DAX syntax
        # Build case-insensitive set of column names for conflict detection.
        # Power BI rejects a model if a measure shares its name with a column
        # in the same table (error: "measure cannot be created because a column
        # with the same name already exists").
        col_names_lower = {c.lower() for c in profile_col_names}

        valid_measure_names = set()
        for m in all_measures:
            m_name = self._strip_control_chars(m["name"])
            dax = self._sanitize_dax(
                m.get("dax", m.get("expression", "BLANK()"))
            )
            fmt = self._strip_control_chars(
                m.get("format", m.get("formatString", "#,0"))
            )

            # --- Name conflict check: measure vs column ---
            if m_name.lower() in col_names_lower:
                # Rename the measure to avoid the Power BI load error.
                # Append " Measure" suffix -- unambiguous and discoverable.
                new_name = m_name + " Measure"
                # Edge case: the suffixed name also collides (unlikely but safe)
                while new_name.lower() in col_names_lower:
                    new_name += " Calc"
                print(f"    [RENAME] measure '{m_name}' -> '{new_name}': "
                      f"conflicts with column of the same name")
                m_name = new_name

            # Validate DAX syntax (balanced parens, no foreign syntax, etc.)
            syntax_issue = self._validate_dax_syntax(dax)
            if syntax_issue:
                print(f"    [SKIP] measure '{m_name}': {syntax_issue}")
                continue

            # Validate: extract column refs like [ColName] or Table[ColName]
            # and check they exist in the actual data
            bad_refs = self._find_bad_column_refs(dax, profile_col_names)
            if bad_refs:
                print(f"    [SKIP] measure '{m_name}': "
                      f"references non-existent column(s): {bad_refs}")
                continue

            # Deduplicate: skip if we already emitted a measure with this name
            if m_name in valid_measure_names:
                print(f"    [SKIP] measure '{m_name}': duplicate measure name")
                continue

            valid_measure_names.add(m_name)
            quoted = self._tmdl_quote(m_name)

            lines.append(f"\tmeasure {quoted} = {dax}")
            lines.append(f"\t\tformatString: {fmt}")
            lines.append(f"\t\tlineageTag: {uuid.uuid4()}")
            lines.append("")

        # Partition (M expression)
        if snowflake_config:
            m_expr = self._build_snowflake_m_expression(
                snowflake_config, tbl_name
            )
            partition_mode = "directQuery"
        else:
            # Build a synthetic table_cfg from the profile for type transforms
            profile_table_cfg = {
                "name": tbl_name,
                "columns": [
                    {
                        "name": c["name"],
                        "dataType": self._pandas_dtype_to_pbi(
                            c.get("dtype", "object"),
                            c.get("semantic_type", "dimension"),
                        ),
                    }
                    for c in profile_columns
                ],
            }
            if csv_url is not None:
                m_expr = self._build_web_contents_m_expression(csv_url, profile_table_cfg)
                print(f"    [WEB] M query -> Web.Contents({csv_url})")
            elif self._inline_dataframe is not None:
                # Inline data ALWAYS wins over File.Contents when a dataframe
                # is available. This prevents broken C:\Users paths in TMDL.
                m_expr = self._build_inline_m_table(
                    self._inline_dataframe, profile_table_cfg
                )
                print(f"    [INLINE] M query -> #table() with "
                      f"{len(self._inline_dataframe)} rows embedded "
                      f"(data_file_path={'set' if data_file_path else 'none'} -- inline wins)")
            else:
                m_expr = self._build_m_expression(
                    profile_table_cfg, data_file_path, sheet_name
                )
            partition_mode = "import"
        lines.append(f"\tpartition {quoted_tbl} = m")
        lines.append(f"\t\tmode: {partition_mode}")
        lines.append("\t\tsource =")
        for m_line in m_expr:
            # Convert any remaining 4-space indents to tabs for TMDL compliance,
            # then prepend 4 tabs to place inside the partition block.
            _cleaned = m_line.replace("    ", "\t")
            lines.append(f"\t\t\t\t{_cleaned}")
        lines.append("")

        lines.append("\tannotation PBI_ResultType = Table")
        lines.append("")

        # Final sanitization: strip any control chars that slipped through
        # (AI-generated DAX can contain invisible 0x1F etc.)
        tmdl_text = self._strip_control_chars("\n".join(lines) + "\n")
        self._write_text(tables_dir / f"{tbl_name}.tmdl", tmdl_text)

        return table_names, valid_measure_names

    def _build_web_contents_m_expression(self, csv_url, table_cfg):
        """Build M expression that fetches data from a live CSV URL via Web.Contents().

        Produces:
            let
                Source = Csv.Document(Web.Contents("https://..."), [...]),
                #"Promoted Headers" = Table.PromoteHeaders(Source, [...]),
                #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {...})
            in
                #"Changed Type"
        """
        pbi_to_m_type = {
            "string": "type text",
            "int64": "type number",
            "double": "type number",
            "dateTime": "type datetime",
            "date": "type datetime",
            "boolean": "type logical",
        }
        cols = table_cfg.get("columns", [])
        n_cols = len(cols) if cols else None
        cols_arg = str(n_cols) if n_cols else "null"
        type_list = ", ".join(
            '{{"' + c["name"] + '", ' + pbi_to_m_type.get(c.get("dataType", "string"), "type text") + "}}"
            for c in cols
        )
        return [
            "let",
            f'    Source = Csv.Document(Web.Contents("{csv_url}"),',
            f'        [Delimiter=",", Columns={cols_arg}, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
            '    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
            f'    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers", {{{type_list}}})',
            "in",
            '    #"Changed Type"',
        ]

    def _build_m_expression(self, table_cfg, data_file_path, sheet_name=None):
        """Build Power Query M expression lines for a table partition.

        The M expression must use a valid Windows absolute path for
        File.Contents(). If the generator runs on Linux (Streamlit Cloud)
        or uses a temp path, we substitute a portable Windows path so
        PBI Desktop can open the file. The data file is bundled in the
        ZIP -- the user just needs to update the path once in Power Query.
        """
        if data_file_path:
            try:
                filename = Path(data_file_path).name
            except Exception:
                filename = str(data_file_path).split("/")[-1].split("\\")[-1]

            # Guard: .twbx/.twb files are ZIP archives, not valid data sources
            # for Power BI M expressions. If one slips through, rewrite the
            # filename to reference a CSV instead (the data file should have
            # been exported to CSV upstream).
            if filename.lower().endswith((".twbx", ".twb")):
                print(f"    [WARN] data_file_path points to Tableau archive: {filename}")
                filename = filename.rsplit(".", 1)[0] + "_data.csv"
                data_file_path = str(Path(data_file_path).parent / filename)
                print(f"    [FIX]  Rewritten to: {filename}")

            # Build a valid Windows path for PBI Desktop.
            # If we're already on Windows with a real path, use it.
            # Otherwise use a placeholder the user can update.
            file_path_for_m = self._windows_path_for_m(data_file_path, filename)
            is_excel = filename.lower().endswith((".xlsx", ".xls"))

            if is_excel:
                if sheet_name:
                    nav_line = (
                        f'    Navigation = Source{{[Item="{sheet_name}",'
                        f'Kind="Sheet"]}}[Data],'
                    )
                else:
                    nav_line = '    Navigation = Source{0}[Data],'
                return [
                    "let",
                    f'    Source = Excel.Workbook(File.Contents("{file_path_for_m}"), null, true),',
                    nav_line,
                    f'    #"Promoted Headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars=true]),',
                    f'    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{self._build_type_transforms(table_cfg)})',
                    "in",
                    '    #"Changed Type"',
                ]
            else:
                return [
                    "let",
                    f'    Source = Csv.Document(File.Contents("{file_path_for_m}"),[Delimiter=",",Columns={len(table_cfg.get("columns", []))},Encoding=65001,QuoteStyle=QuoteStyle.None]),',
                    f'    #"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
                    f'    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{self._build_type_transforms(table_cfg)})',
                    "in",
                    '    #"Changed Type"',
                ]

        # Last resort: empty table
        return [
            "let",
            '    Source = #table({"Col"}, {})',
            "in",
            "    Source",
        ]

    def _build_inline_m_table(self, df, table_cfg):
        """Build a #table() M expression with data rows embedded inline.

        Produces a complete M let/in expression that creates a typed table
        from literal values. No external file reference needed.

        Args:
            df: pandas DataFrame with the actual data.
            table_cfg: dict with columns[{name, dataType}] for type schema.

        Returns:
            list of M expression lines (tab-indented, no leading spaces).
        """
        import pandas as pd
        import numpy as np

        # Filter footnote rows -- rows where first column starts with * or **
        # (common in .hyper extracts, breaks Analysis Services)
        if not df.empty:
            first_col = df.columns[0]
            mask = df[first_col].astype(str).str.startswith("*", na=False)
            if mask.any():
                n_removed = mask.sum()
                df = df[~mask].reset_index(drop=True)
                print(f"    [INLINE] Removed {n_removed} footnote rows from inline data")

        columns = table_cfg.get("columns", [])
        col_names = [c["name"] for c in columns]

        # Build type schema using M "type table" syntax:
        #   type table [#"Col Name" = text, Col2 = number]
        pbi_to_m = {
            "string": "text",
            "int64": "number",
            "double": "number",
            "dateTime": "date",
            "boolean": "logical",
        }
        type_cols = []
        for c in columns:
            m_type = pbi_to_m.get(c.get("dataType", "string"), "text")
            col_name = c["name"]
            # Quote column names that contain spaces or special chars
            if " " in col_name or "-" in col_name or not col_name.isidentifier():
                type_cols.append(f'#"{col_name}" = {m_type}')
            else:
                type_cols.append(f'{col_name} = {m_type}')
        type_schema = "type table [" + ", ".join(type_cols) + "]"

        # Build data rows -- limit to 500 rows to keep TMDL size reasonable
        max_rows = min(len(df), 500)
        row_lines = []
        for idx in range(max_rows):
            vals = []
            for c in columns:
                col_name = c["name"]
                if col_name not in df.columns:
                    vals.append('""')
                    continue
                raw = df[col_name].iloc[idx]

                # Handle null/NaN
                if pd.isna(raw):
                    vals.append("null")
                    continue

                dt = c.get("dataType", "string")
                if dt in ("double", "int64"):
                    vals.append(str(raw))
                elif dt == "dateTime":
                    # M date literal: #date(year, month, day)
                    try:
                        ts = pd.Timestamp(raw)
                        vals.append(
                            f"#date({ts.year}, {ts.month}, {ts.day})"
                        )
                    except Exception:
                        vals.append(f'"{raw}"')
                elif dt == "boolean":
                    vals.append("true" if raw else "false")
                else:
                    # String -- escape double quotes, strip control chars
                    s = str(raw).replace('"', '""')
                    s = s.replace("\t", " ").replace("\n", " ").replace("\r", "")
                    vals.append(f'"{s}"')

            # Sanitize entire row: strip any remaining control chars
            clean_vals = ", ".join(vals).replace("\t", " ").replace("\n", " ").replace("\r", "")
            row_lines.append("{" + clean_vals + "}")

        # Build the M expression -- each line is a separate list element.
        # Use tab indentation throughout -- the caller (_write_tables_tmdl)
        # prepends 4 tabs. TMDL requires pure tab indentation; any spaces
        # in leading whitespace cause Fabric API "Indentation" errors.
        lines = [
            "let",
            "\tSource = #table(" + type_schema + ",",
            "\t{",
        ]
        for i, row in enumerate(row_lines):
            suffix = "," if i < len(row_lines) - 1 else ""
            lines.append("\t" + row + suffix)
        lines.append("\t})")
        lines.append("in")
        lines.append("\tSource")
        return lines

    @staticmethod
    def _build_snowflake_m_expression(sf_config, table_name):
        """Build a Snowflake DirectQuery M expression.

        Uses Snowflake.Databases() to connect directly to the warehouse.
        PBI Desktop will prompt for Snowflake credentials on first open.
        """
        acct = sf_config.get("account", "")
        wh = sf_config.get("warehouse", "")
        db = sf_config.get("database", "")
        schema = sf_config.get("schema", "")

        # Build server URL (account can be org-acctname or full URL)
        if ".snowflakecomputing.com" in acct:
            server = acct
        else:
            server = f"{acct}.snowflakecomputing.com"

        return [
            "let",
            f'    Source = Snowflake.Databases("{server}", "{wh}"),',
            f'    Database = Source{{[Name="{db}"]}}[Data],',
            f'    Schema = Database{{[Name="{schema}"]}}[Data],',
            f'    Table = Schema{{[Name="{table_name}"]}}[Data]',
            "in",
            "    Table",
        ]

    @staticmethod
    def _windows_path_for_m(data_file_path, filename):
        """Return a valid Windows absolute path for PBI M expressions.

        - On Windows with a real local path: use as-is (resolved).
        - On Linux / temp paths / Streamlit Cloud: use C:\\PBI_Data\\{filename}
          so PBI Desktop can open the file (user updates path once).
        """
        import sys
        raw = str(data_file_path)

        if sys.platform == "win32":
            try:
                resolved = str(Path(raw).resolve())
                # Only use it if it looks like a real Windows path (not temp)
                if resolved[1] == ":" and "\\Temp\\" not in resolved:
                    return resolved
            except Exception:
                pass

        # Fallback: portable placeholder path
        return f"C:\\PBI_Data\\{filename}"

    def _build_type_transforms(self, table_cfg):
        """Build the type transform list for Table.TransformColumnTypes."""
        type_map = {
            "string": "type text",
            "int64": "type number",
            "double": "type number",
            "dateTime": "type date",
            "boolean": "type logical",
        }
        pairs = []
        for col in table_cfg.get("columns", []):
            name = col["name"]
            dt = col.get("dataType", "string")
            m_type = type_map.get(dt, "type text")
            pairs.append(f'{{"{name}", {m_type}}}')
        return "{" + ", ".join(pairs) + "}"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _primary_table(self, config):
        tables = config.get("tmdl_model", {}).get("tables", [])
        return tables[0].get("name", "Data") if tables else "Data"

    def _collect_measures(self, config):
        measures = []
        for t in config.get("tmdl_model", {}).get("tables", []):
            measures.extend(t.get("measures", []))
        return measures

    @staticmethod
    def _find_bad_column_refs(dax, valid_columns):
        """Extract column references from DAX and return any not in valid_columns.

        Matches patterns like [Column Name] and Table[Column Name].
        Ignores DAX function names and known measure self-references.
        """
        import re
        # Find all [bracketed] references in the DAX
        refs = re.findall(r'\[([^\]]+)\]', dax)
        if not refs:
            return set()

        # DAX built-in functions/properties to ignore
        dax_builtins = {
            "Value", "BLANK", "TRUE", "FALSE", "ALL", "ALLEXCEPT",
            "CALCULATE", "FILTER", "SUM", "AVERAGE", "COUNT", "COUNTROWS",
            "MIN", "MAX", "DIVIDE", "IF", "SWITCH", "RELATED",
            "TOTALYTD", "TOTALMTD", "TOTALQTD", "DATESYTD",
            "PREVIOUSYEAR", "PREVIOUSMONTH", "PREVIOUSQUARTER",
            "SAMEPERIODLASTYEAR", "DATEADD", "VALUES",
        }

        bad = set()
        for ref in refs:
            if ref in valid_columns:
                continue
            if ref in dax_builtins:
                continue
            bad.add(ref)
        return bad

    @staticmethod
    def _validate_dax_syntax(formula):
        """Basic DAX syntax validation -- safety net, not a full parser.

        Returns None if the formula looks OK, or a reason string if it should be skipped.
        Better to skip a bad measure than crash Power BI Desktop.
        """
        if not formula or not formula.strip():
            return "empty formula"

        f = formula.strip()

        # a. Balanced parentheses
        if f.count("(") != f.count(")"):
            return (f"unbalanced parentheses: {f.count('(')} open vs "
                    f"{f.count(')')} close")

        # b. Balanced double quotes
        if f.count('"') % 2 != 0:
            return f"unbalanced double quotes ({f.count('\"')} found)"

        # c. No Python/SQL/Tableau syntax leaking through
        import re
        foreign_patterns = [
            (r'\bdef\s+', "Python 'def' keyword"),
            (r'\bimport\s+', "Python 'import' keyword"),
            (r'\bSELECT\s+', "SQL SELECT statement"),
            (r'\bFROM\s+\w+\s+WHERE\b', "SQL FROM...WHERE pattern"),
            (r'\blambda\s+', "Python 'lambda' keyword"),
            (r'\bclass\s+', "Python 'class' keyword"),
            (r'\bTHEN\b', "Tableau IF/THEN syntax (not DAX)"),
            (r'\bELSEIF\b', "Tableau ELSEIF syntax (not DAX)"),
            (r'\bEND\b(?!\s*\))', "Tableau END keyword (not DAX)"),
        ]
        for pattern, desc in foreign_patterns:
            if re.search(pattern, f, re.IGNORECASE):
                return f"contains {desc}"

        # d. No bare Tableau syntax: [Field] without Table prefix in aggregation context
        # Tableau uses SUM([Field]) -- DAX requires SUM(Table[Field])
        # Look for AGG_FUNC([Field]) where [Field] is NOT preceded by a table name
        tableau_agg_leak = re.findall(
            r'\b(SUM|AVERAGE|COUNT|MIN|MAX|COUNTD|AVG)\s*\(\s*\[',
            f, re.IGNORECASE
        )
        if tableau_agg_leak:
            return (f"Tableau syntax detected: {tableau_agg_leak[0]}([...]) "
                    f"-- DAX requires {tableau_agg_leak[0]}(Table[Column])")

        # e. Completely nonsensical: just a number or a bare string
        if re.match(r'^[\d.]+$', f):
            return "formula is just a literal number"

        return None  # All checks passed

    def _safe_name(self, name):
        safe = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in ("_", "-", "."))
        return safe[:80]

    @staticmethod
    def _tmdl_quote(name):
        """Quote a TMDL identifier if it contains special characters.

        TMDL requires single-quote wrapping for identifiers that contain
        anything other than letters, digits, and underscores.  This covers
        spaces, parentheses, hyphens, dots, etc.
        """
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return f"'{name}'"
        return name

    @classmethod
    def _sanitize_dax(cls, text):
        """Clean a DAX expression for safe TMDL embedding.

        1. Strip control characters
        2. Remove // line comments and /* block comments */
        3. Collapse multi-line expressions to a single line
        4. Normalize whitespace

        TMDL requires measure expressions on one line after the = sign.
        Multi-line comments or expressions cause InvalidLineType errors.
        """
        if not isinstance(text, str):
            return "BLANK()"
        import re
        text = cls._strip_control_chars(text)
        # Remove /* ... */ block comments
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
        # Remove // line comments (everything from // to end of line)
        text = re.sub(r'//[^\n]*', ' ', text)
        # Collapse newlines and excess whitespace to single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return "BLANK()"
        return text

    @staticmethod
    def _strip_control_chars(text):
        """Remove XML-illegal control characters (0x00-0x1F except tab/newline/CR).

        Power BI Desktop serializes TMDL to XMLA internally. Any control
        character in the file causes: 'hexadecimal value 0x1F, is an invalid
        character'. AI-generated DAX can contain these invisibly.
        """
        if not isinstance(text, str):
            return text
        import re
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)

    def _safe_remove(self, target):
        """Remove a file or folder, handling OneDrive locks."""
        if target.is_file():
            try:
                target.unlink()
            except (PermissionError, OSError):
                pass
            return
        try:
            shutil.rmtree(target)
        except (PermissionError, OSError):
            import time
            time.sleep(1)
            for f in sorted(target.rglob("*"), reverse=True):
                try:
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        f.rmdir()
                except (PermissionError, OSError):
                    pass
            try:
                target.rmdir()
            except (PermissionError, OSError):
                pass

    def _write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_text(self, path, text):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)


# ------------------------------------------------------------------ #
#  CLI test                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    import io

    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    config_path = project_root / "output" / "openai_pbip_config.json"
    profile_path = project_root / "output" / "data_profile.json"
    spec_path = project_root / "output" / "claude_dashboard_spec.json"

    for p in [config_path, profile_path, spec_path]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        data_profile = json.load(f)
    with open(spec_path, "r", encoding="utf-8") as f:
        dashboard_spec = json.load(f)

    data_file = None
    for pattern in ["*.xlsx", "*.csv"]:
        matches = sorted(project_root.glob(pattern))
        if matches:
            data_file = str(matches[0])
            break

    print("=" * 60)
    print("PBIP Generator Test")
    print("=" * 60)

    gen = PBIPGenerator(str(project_root / "output"))
    result = gen.generate(config, data_profile, dashboard_spec, data_file)
    out = result["path"] if isinstance(result, dict) else result

    safe = gen._safe_name(dashboard_spec.get("dashboard_title", "Dashboard"))
    print(f"\nField audit: {result.get('field_audit', {})}")
    print(f"\nFile tree:")
    for suffix in [".pbip"]:
        p = Path(out) / f"{safe}{suffix}"
        if p.exists():
            print(f"  {p.name:60s} {p.stat().st_size:>8,} bytes")
    for suffix in [".Report", ".SemanticModel"]:
        folder = Path(out) / f"{safe}{suffix}"
        if folder.exists():
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    rel = f"{safe}{suffix}/{f.relative_to(folder)}"
                    print(f"  {rel:60s} {f.stat().st_size:>8,} bytes")
