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


class PBIPGenerator:
    """Generate a complete PBIP project from AI config."""

    # Map our simplified dataRoles keys to PBI queryState role names.
    # Different visual types use different roles.
    ROLE_MAP_CHART = {"category": "Category", "values": "Y", "series": "Series"}
    ROLE_MAP_CARD = {"values": "Values"}
    ROLE_MAP_TABLE = {"values": "Values"}

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).parent.parent.resolve()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, config, data_profile, dashboard_spec, data_file_path=None,
                 sheet_name=None):
        """Create the full PBIP project.

        Args:
            config:         dict from OpenAIEngine (report_layout + tmdl_model).
            data_profile:   dict from DataAnalyzer.
            dashboard_spec: dict from ClaudeInterpreter (for title/theme).
            data_file_path: optional path to the source data file.
            sheet_name:     Excel sheet name that was loaded (for M expression).

        Returns:
            str: path to the clean project directory containing ONLY the PBIP files.
        """
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

        # Gather info -- table_name must match what _write_tables_tmdl uses
        # (data_profile table name = ground truth from the actual data)
        table_name = data_profile.get("table_name", self._primary_table(config))

        # Column semantic types (dimension vs measure) from data profile
        col_types = {}
        profile_col_names = set()
        for col_info in data_profile.get("columns", []):
            col_types[col_info["name"]] = col_info.get("semantic_type", "dimension")
            profile_col_names.add(col_info["name"])

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
            tables_dir, tm, data_profile, data_file_path, sheet_name
        )

        # 14. model.tmdl
        self._write_model_tmdl(model_def, table_names)

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
        page_ids = self._write_pages(
            pages_dir, raw_sections, table_name, valid_measure_names,
            col_types, profile_col_names,
        )

        # 7. pages.json
        self._write_pages_json(pages_dir, page_ids)

        # 8. Theme file (copy from reference)
        self._write_theme(theme_dir)

        # 15. README for the user
        self._write_readme(project_dir, safe, title)

        # Summary
        total = sum(1 for _ in project_dir.rglob("*") if _.is_file())
        print(f"[OK] PBIP project generated: {project_dir}")
        print(f"     {total} files created")

        return str(project_dir)

    # ------------------------------------------------------------------ #
    #  {name}.pbip                                                        #
    # ------------------------------------------------------------------ #

    def _write_pbip(self, safe, project_dir):
        doc = deepcopy(PBIP_TEMPLATE)
        doc["artifacts"][0]["report"]["path"] = f"{safe}.Report"
        self._write_json(project_dir / f"{safe}.pbip", doc)
        print(f"    [+] {safe}.pbip")

    def _write_readme(self, project_dir, safe, title):
        """Write a README so the user knows how to open the project."""
        readme = (
            f"Power BI Project: {title}\n"
            f"{'=' * (len(title) + 21)}\n\n"
            f"HOW TO OPEN:\n"
            f"  1. Extract this ZIP to a folder on your computer\n"
            f"  2. Open Power BI Desktop (January 2026 or later)\n"
            f"  3. Enable preview features if not already enabled:\n"
            f"     File > Options > Preview features > check all PBIP options\n"
            f"  4. Double-click the file: {safe}.pbip\n"
            f"  5. Power BI Desktop will open with all visuals, measures, and data model\n\n"
            f"FILES:\n"
            f"  {safe}.pbip              -- Open this file in PBI Desktop\n"
            f"  {safe}.Report/           -- Report layout, pages, visuals\n"
            f"  {safe}.SemanticModel/    -- Data model, DAX measures, table definitions\n\n"
            f"NOTES:\n"
            f"  - This is a Power BI Project (.pbip) format, NOT the legacy .pbix format\n"
            f"  - PBIP is the modern format used by PBI Desktop for version control and collaboration\n"
            f"  - If PBI Desktop does not recognize the .pbip file, enable the preview features above\n"
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

    def _write_pages(self, pages_dir, raw_sections, table_name, measure_names,
                     col_types=None, profile_col_names=None):
        """Write page.json and visual.json files. Returns list of page IDs."""
        page_ids = []

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

            # Visuals
            visuals_root = page_dir / "visuals"
            vc_count = 0
            for vidx, vc in enumerate(raw.get("visualContainers", [])):
                viz_id = uuid.uuid4().hex[:20]
                viz_dir = visuals_root / viz_id
                viz_dir.mkdir(parents=True, exist_ok=True)
                self._write_visual_json(
                    viz_dir, vc, vidx, viz_id,
                    table_name, measure_names, col_types,
                    profile_col_names,
                )
                vc_count += 1

            print(f"    [+] page '{display_name}': {vc_count} visuals")

        return page_ids

    def _write_visual_json(self, viz_dir, vc, vidx, viz_id, table_name,
                           measure_names, col_types=None,
                           profile_col_names=None):
        """Write a single visual.json file."""
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

        visual_type = cfg_obj.get("visualType", "card")
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

        # Add title if present
        if title_text:
            doc["visual"]["objects"] = {
                "title": [{
                    "properties": {
                        "text": {"expr": {"Literal": {"Value": f"'{title_text}'"}}},
                        "show": {"expr": {"Literal": {"Value": "true"}}},
                    }
                }]
            }

        self._write_json(viz_dir / "visual.json", doc)

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
        else:
            role_map = self.ROLE_MAP_CHART

        # Value roles get aggregated, category roles do not
        value_roles = {"Y", "Values"}

        query_state = {}
        for our_role, pbi_role in role_map.items():
            fields = data_roles.get(our_role, [])
            if not fields:
                continue

            aggregate = pbi_role in value_roles
            projections = []
            for field_name in fields:
                if field_name not in valid_fields:
                    print(f"    [SKIP] visual field '{field_name}': "
                          f"not in dataset columns or valid measures")
                    continue
                proj = self._build_field_projection(
                    field_name, table_name, measure_names,
                    aggregate=aggregate, col_types=col_types,
                )
                projections.append(proj)

            if projections:
                query_state[pbi_role] = {"projections": projections}

        return query_state

    def _build_field_projection(self, field_name, table_name, measure_names,
                                aggregate=False, col_types=None):
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

        # Wrap numeric columns in Aggregation (Sum) when in value roles
        # This matches how PBI Desktop writes visual.json (see reference_template)
        sem = col_types.get(field_name, "dimension")
        if aggregate and sem == "measure":
            return {
                "field": {
                    "Aggregation": {
                        "Expression": col_ref,
                        "Function": 0,  # 0 = Sum
                    }
                },
                "queryRef": f"Sum({table_name}.{field_name})",
                "nativeQueryRef": f"Sum of {field_name}",
            }

        # Plain column reference for dimensions / category axes
        return {
            "field": col_ref,
            "queryRef": f"{table_name}.{field_name}",
            "nativeQueryRef": field_name,
        }

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

    def _write_model_tmdl(self, model_def, table_names):
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
        lines.append("ref cultureInfo en-US")
        lines.append("")

        self._write_text(model_def / "model.tmdl", "\n".join(lines) + "\n")
        print(f"    [+] model.tmdl ({len(table_names)} tables)")

    # Pandas dtype -> PBI dataType mapping
    _DTYPE_MAP = {
        "int64": "int64",
        "Int64": "int64",
        "float64": "double",
        "Float64": "double",
        "object": "string",
        "string": "string",
        "bool": "boolean",
        "boolean": "boolean",
    }

    @classmethod
    def _pandas_dtype_to_pbi(cls, dtype_str, semantic_type):
        """Map a pandas dtype string to a PBI dataType."""
        if semantic_type == "date":
            return "dateTime"
        # Check for datetime-like dtypes (datetime64[ns], etc.)
        if "datetime" in dtype_str.lower():
            return "dateTime"
        return cls._DTYPE_MAP.get(dtype_str, "string")

    def _write_tables_tmdl(self, tables_dir, tmdl_model, data_profile,
                           data_file_path, sheet_name=None):
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
            col_name = col_info["name"]
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

        # Measures -- from AI output, validated against real columns
        valid_measure_names = set()
        for m in all_measures:
            m_name = m["name"]
            dax = m.get("dax", m.get("expression", "BLANK()"))
            fmt = m.get("format", m.get("formatString", "#,0"))

            # Validate: extract column refs like [ColName] or Table[ColName]
            # and check they exist in the actual data
            bad_refs = self._find_bad_column_refs(dax, profile_col_names)
            if bad_refs:
                print(f"    [SKIP] measure '{m_name}': "
                      f"references non-existent column(s): {bad_refs}")
                continue

            valid_measure_names.add(m_name)
            quoted = self._tmdl_quote(m_name)

            lines.append(f"\tmeasure {quoted} = {dax}")
            lines.append(f"\t\tformatString: {fmt}")
            lines.append(f"\t\tlineageTag: {uuid.uuid4()}")
            lines.append("")

        # Partition (M expression)
        # Build a synthetic table_cfg from the profile for type transforms
        profile_table_cfg = {
            "name": tbl_name,
            "columns": [
                {
                    "name": c["name"],
                    "dataType": self._pandas_dtype_to_pbi(
                        c.get("dtype", "object"), c.get("semantic_type", "dimension")
                    ),
                }
                for c in profile_columns
            ],
        }
        m_expr = self._build_m_expression(
            profile_table_cfg, data_file_path, sheet_name
        )
        lines.append(f"\tpartition {quoted_tbl} = m")
        lines.append("\t\tmode: import")
        lines.append("\t\tsource =")
        for m_line in m_expr:
            lines.append(f"\t\t\t\t{m_line}")
        lines.append("")

        lines.append("\tannotation PBI_ResultType = Table")
        lines.append("")

        self._write_text(tables_dir / f"{tbl_name}.tmdl", "\n".join(lines) + "\n")

        return table_names, valid_measure_names

    def _build_m_expression(self, table_cfg, data_file_path, sheet_name=None):
        """Build Power Query M expression lines for a table partition.

        Uses generic step names (Navigation, Source) to avoid M parser errors
        when identifiers contain special characters.

        The M expression uses a FilePath parameter pattern so PBI Desktop
        can locate the data file wherever the user extracts the ZIP.

        Args:
            table_cfg:      dict with 'name' and 'columns' keys.
            data_file_path: path to the source data file.
            sheet_name:     Excel sheet name to load (e.g. "Orders").
        """
        if data_file_path:
            try:
                p = Path(data_file_path)
                filename = p.name
                original_path = str(p.resolve())
            except Exception:
                filename = str(data_file_path).replace("/", "\\").split("\\")[-1]
                original_path = str(data_file_path)
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
                    f'    // DataFile: {filename}',
                    f'    Source = Excel.Workbook(File.Contents("{original_path}"), null, true),',
                    nav_line,
                    f'    #"Promoted Headers" = Table.PromoteHeaders(Navigation, [PromoteAllScalars=true]),',
                    f'    #"Changed Type" = Table.TransformColumnTypes(#"Promoted Headers",{self._build_type_transforms(table_cfg)})',
                    "in",
                    '    #"Changed Type"',
                ]
            else:
                return [
                    "let",
                    f'    // DataFile: {filename}',
                    f'    Source = Csv.Document(File.Contents("{original_path}"),[Delimiter=",",Columns={len(table_cfg.get("columns", []))},Encoding=65001,QuoteStyle=QuoteStyle.None]),',
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

    def _build_type_transforms(self, table_cfg):
        """Build the type transform list for Table.TransformColumnTypes."""
        type_map = {
            "string": "type text",
            "int64": "Int64.Type",
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
    out = gen.generate(config, data_profile, dashboard_spec, data_file)

    safe = gen._safe_name(dashboard_spec.get("dashboard_title", "Dashboard"))
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
