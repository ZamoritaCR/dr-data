"""
Dr. Data Agent -- Claude tool-calling conversational agent.

Maintains conversation state across messages. Uses Claude's tool_use
to call analysis, design, and build functions.
"""
import sys
import io
import os
import json
import time
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

import anthropic
import pandas as pd

# Windows encoding fix
if sys.platform == "win32" and getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import _get_secret
from config.prompts import DR_DATA_SYSTEM_PROMPT, DASHBOARD_DESIGN_PROMPT
from core.deep_analyzer import DeepAnalyzer
from core.html_dashboard import HTMLDashboardBuilder
from core.pdf_report import PDFReportBuilder

# ------------------------------------------------------------------ #
#  Tool Definitions for Claude                                         #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "name": "analyze_data",
        "description": (
            "Deep statistical profiling of a dataset. Returns distributions, "
            "correlations, outliers, null patterns, business signals, and "
            "data quality assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file to analyze"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "design_dashboard",
        "description": (
            "Design optimal dashboard layout with visual hierarchy, chart types, "
            "calculated metrics. Returns a structured dashboard specification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What the user wants the dashboard to show"
                },
                "data_profile": {
                    "type": "string",
                    "description": "JSON string of the data profile from analyze_data"
                },
                "audience": {
                    "type": "string",
                    "description": "Target audience (e.g., 'CFO', 'operations team', 'general')"
                }
            },
            "required": ["request", "data_profile"]
        }
    },
    {
        "name": "build_html_dashboard",
        "description": (
            "Build an interactive HTML dashboard directly from the loaded data. "
            "Opens in any browser, filters update all charts live, works offline. "
            "Auto-detects best charts from the data -- no spec needed. "
            "Returns the file path to the generated HTML."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Dashboard title"
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file (optional if already loaded)"
                }
            },
            "required": ["title"]
        }
    },
    {
        "name": "build_pdf_report",
        "description": (
            "Generate a professional PDF report with embedded charts, tables, "
            "and executive narrative. Returns the file path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": "JSON string of the dashboard specification"
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to the data file"
                }
            },
            "required": ["spec", "data_path"]
        }
    },
    {
        "name": "build_powerbi",
        "description": (
            "Build a Power BI Project (.pbip) with DAX measures, relationships, "
            "and formatted visuals. Uses a full AI pipeline (Claude + GPT-4) "
            "to generate the project, then ZIPs it for download. "
            "Returns the path to the ZIP file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What the dashboard should show (e.g. 'sales by region and category')"
                },
                "audience": {
                    "type": "string",
                    "description": "Target audience (e.g. 'CFO', 'operations', 'general')"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name for the Power BI project"
                }
            },
            "required": ["request", "project_name"]
        }
    },
    {
        "name": "build_documentation",
        "description": (
            "Generate documentation package. Types: 'executive_summary', "
            "'technical_spec', 'data_dictionary', 'full_package'. "
            "Returns the file path(s)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "string",
                    "description": "JSON string of the dashboard specification"
                },
                "profile": {
                    "type": "string",
                    "description": "JSON string of the data profile"
                },
                "doc_type": {
                    "type": "string",
                    "description": "Type of documentation to generate",
                    "enum": ["executive_summary", "technical_spec",
                             "data_dictionary", "full_package"]
                }
            },
            "required": ["doc_type"]
        }
    },
    {
        "name": "parse_legacy_report",
        "description": (
            "Extract structure from Tableau (.twb/.twbx) or Business Objects "
            "(.rpt/.wid) files. Returns the parsed structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the legacy report file"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "parse_alteryx_workflow",
        "description": (
            "Parse an Alteryx workflow (.yxmd, .yxwz, .yxmc, .yxzp) "
            "and generate a complete Dataiku DSS migration plan. "
            "Returns workflow structure, tool-by-tool translation map, "
            "generated Python recipe code, and migration effort estimate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the Alteryx file"
                },
                "generate_dataiku_code": {
                    "type": "boolean",
                    "description": "Whether to generate Dataiku Python recipe code"
                }
            },
            "required": ["file_path"]
        }
    }
]


# ------------------------------------------------------------------ #
#  Dr. Data Agent                                                      #
# ------------------------------------------------------------------ #

class DrDataAgent:
    """Conversational Claude agent with tool-calling for data analysis."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 8192

    def __init__(self):
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. "
                "Add it in Streamlit Cloud Secrets or your local .env file."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages = []
        self.analyzer = DeepAnalyzer()
        self.html_builder = HTMLDashboardBuilder()
        self.pdf_builder = PDFReportBuilder()

        self.output_dir = str(PROJECT_ROOT / "output")

        # State carried across turns
        self.data_profile = None
        self.dashboard_spec = None
        self.dataframe = None
        self.data_path = None
        self.data_file_path = None
        self.sheet_name = None
        self.tableau_spec = None
        self.generated_files = []

        # Multi-file session support
        self.session = None
        self.session_bridge = None

    def _fix_orphaned_tool_calls(self):
        """Ensure every assistant tool_use block has a matching tool_result.

        If a previous call crashed after appending an assistant message
        with tool_use but before appending the tool_results, the message
        list becomes invalid for the API. This method scans and patches
        any such gaps with error-carrying tool_result stubs.
        """
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            if msg["role"] != "assistant":
                i += 1
                continue

            content = msg.get("content", [])
            if not isinstance(content, list):
                i += 1
                continue

            tool_ids = [
                b["id"] for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if not tool_ids:
                i += 1
                continue

            # This assistant message has tool_use -- check next message
            if i + 1 < len(self.messages):
                nxt = self.messages[i + 1]
                nxt_content = nxt.get("content", "")

                if nxt["role"] == "user" and isinstance(nxt_content, list):
                    # Next message is a tool_result list -- patch missing ids
                    existing = {
                        b.get("tool_use_id") for b in nxt_content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    for tid in tool_ids:
                        if tid not in existing:
                            nxt_content.append({
                                "type": "tool_result",
                                "tool_use_id": tid,
                                "content": json.dumps({"error": "Interrupted"})
                            })
                else:
                    # Next message is a plain user text -- insert tool_results
                    results = [{
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": json.dumps({"error": "Interrupted"})
                    } for tid in tool_ids]
                    self.messages.insert(i + 1, {
                        "role": "user", "content": results
                    })
            else:
                # tool_use is the very last message -- append results
                results = [{
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": json.dumps({"error": "Interrupted"})
                } for tid in tool_ids]
                self.messages.append({
                    "role": "user", "content": results
                })

            i += 1

    def chat(self, user_message, progress_callback=None):
        """Send a message and get Dr. Data's response.

        Args:
            user_message: The user's text message.
            progress_callback: Optional callable(status_text) for UI updates.

        Returns:
            dict with:
              "text": Dr. Data's text response
              "files": list of generated file paths
              "profile": data profile dict or None
              "spec": dashboard spec dict or None
        """
        self.messages.append({"role": "user", "content": user_message})
        self.generated_files = []
        self._progress_callback = progress_callback

        # Fix any orphaned tool_use from a previous crashed call
        self._fix_orphaned_tool_calls()

        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        # Loop for tool-calling conversation
        max_iterations = 10
        for iteration in range(max_iterations):
            _progress(f"Dr. Data is thinking... (step {iteration + 1})")

            try:
                response = self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    temperature=0,
                    system=DR_DATA_SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=self.messages,
                )
            except Exception as api_err:
                # If the API call itself fails, remove the user message
                # we just added so self.messages stays consistent
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()
                raise api_err

            # Process response blocks -- convert to plain dicts for re-serialization
            assistant_content = response.content
            content_dicts = []
            for block in assistant_content:
                if block.type == "text":
                    content_dicts.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    content_dicts.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            self.messages.append({"role": "assistant", "content": content_dicts})

            # Check if we need to handle tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                tool_labels = {
                    "analyze_data": "Analyzing your data...",
                    "design_dashboard": "Designing dashboard layout...",
                    "build_html_dashboard": "Building HTML dashboard...",
                    "build_pdf_report": "Generating PDF report...",
                    "build_powerbi": "Building Power BI project...",
                    "build_documentation": "Writing documentation...",
                    "parse_legacy_report": "Parsing legacy report...",
                    "parse_alteryx_workflow": "Parsing Alteryx workflow...",
                }
                for block in assistant_content:
                    if block.type == "tool_use":
                        label = tool_labels.get(block.name, f"Running: {block.name}")
                        _progress(label)
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                self.messages.append({"role": "user", "content": tool_results})
                continue  # Let Claude process the tool results

            # No more tool calls -- extract final text
            text_parts = []
            for block in assistant_content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            return {
                "text": "\n".join(text_parts),
                "files": self.generated_files,
                "profile": self.data_profile,
                "spec": self.dashboard_spec,
            }

        # Shouldn't reach here, but safety net
        return {
            "text": "I hit the maximum processing steps. Please try rephrasing your request.",
            "files": self.generated_files,
            "profile": self.data_profile,
            "spec": self.dashboard_spec,
        }

    def set_session(self, session):
        """Connect a MultiFileSession to this agent for multi-file support."""
        from core.agent_session import AgentSessionBridge
        self.session = session
        self.session_bridge = AgentSessionBridge(session)

        # Sync state from session into agent
        df = session.get_primary_dataframe()
        if df is not None:
            self.dataframe = df
            if self.data_profile is None:
                self.data_profile = self.analyzer.profile(df)

        if session.tableau_spec:
            self.tableau_spec = session.tableau_spec
        if session.alteryx_spec:
            self.tableau_spec = session.alteryx_spec

        # Get primary data file path and sheet name
        for fname, info in session.files.items():
            if info.get("df") is not None:
                self.data_file_path = info["path"]
                self.data_path = info["path"]
                self.sheet_name = info.get("sheet_name") or getattr(
                    session, "primary_sheet_name", None
                )
                break

    def inject_file(self, file_path, df=None):
        """Inject a file into the agent's state (called when user uploads)."""
        self.data_path = file_path
        self.data_file_path = file_path
        if df is not None:
            self.dataframe = df

    def analyze_uploaded_file(self, file_path=None):
        """Auto-analyze uploaded file(s) and return a text summary.

        If a MultiFileSession is connected via set_session(), uses that.
        Otherwise falls back to single-file analysis via ingest_file().
        """
        # Multi-file session path
        if self.session:
            return self._analyze_from_session()

        # Single-file fallback
        if not file_path:
            return "No file provided."

        from app.file_handler import ingest_file

        self.data_file_path = file_path
        self.data_path = file_path

        result = ingest_file(file_path)

        # Store dataframes
        dfs = result.get("dataframes", {})
        if dfs:
            first_key = next(iter(dfs))
            self.dataframe = dfs[first_key]
            self.data_profile = self.analyzer.profile(self.dataframe)

        # Store report structure (Tableau, Alteryx, Business Objects)
        if result.get("report_structure"):
            self.tableau_spec = result["report_structure"]

        # Build summary text
        fname = os.path.basename(file_path)
        parts = [f"I have loaded {fname}."]

        if self.dataframe is not None:
            df = self.dataframe
            cols_preview = ", ".join(df.columns[:8].tolist())
            if len(df.columns) > 8:
                cols_preview += "..."
            parts.append(
                f"The dataset contains {len(df):,} rows and "
                f"{len(df.columns)} columns: {cols_preview}."
            )
            if self.data_profile:
                for insight in self.data_profile.get("quick_insights", [])[:3]:
                    parts.append(insight)

        if self.tableau_spec:
            rtype = self.tableau_spec.get("type", "")
            if "tableau" in rtype:
                ws_count = len(self.tableau_spec.get("worksheets", []))
                ds_count = len(self.tableau_spec.get("datasources", []))
                parts.append(
                    f"This is a Tableau workbook with {ws_count} worksheets "
                    f"and {ds_count} data sources."
                )
            elif "alteryx" in rtype:
                tool_count = self.tableau_spec.get(
                    "tool_summary", {}
                ).get("total_tools", 0)
                parts.append(
                    f"This is an Alteryx workflow with {tool_count} tools."
                )
            elif "business_objects" in rtype:
                parts.append("This is a Business Objects report.")

        parts.append(self._build_options_text())
        return " ".join(parts)

    def _analyze_from_session(self):
        """Build analysis summary from the MultiFileSession."""
        session = self.session
        summary = session.get_summary()
        parts = []

        # Sync data from session
        df = session.get_primary_dataframe()
        if df is not None and self.dataframe is None:
            self.dataframe = df
            self.data_profile = self.analyzer.profile(df)

        # File listing
        file_names = [f["filename"] for f in summary["files"]]
        if len(file_names) == 1:
            parts.append(f"I have loaded {file_names[0]}.")
        else:
            parts.append(
                f"I have loaded {len(file_names)} files: {', '.join(file_names)}."
            )

        # Data summary with specific numbers
        if self.dataframe is not None:
            df = self.dataframe
            cols_preview = ", ".join(df.columns[:8].tolist())
            if len(df.columns) > 8:
                cols_preview += "..."
            parts.append(
                f"The primary dataset contains {len(df):,} rows and "
                f"{len(df.columns)} columns: {cols_preview}."
            )
            if self.data_profile:
                for insight in self.data_profile.get("quick_insights", [])[:3]:
                    parts.append(insight)

        # Structure info
        if summary.get("has_structure"):
            stype = summary.get("structure_type", "")
            if "tableau" in stype and "tableau" in summary:
                t = summary["tableau"]
                parts.append(
                    f"This is a Tableau workbook with {t['worksheet_count']} worksheets, "
                    f"{t['dashboard_count']} dashboards, and "
                    f"{t['calculated_fields']} calculated fields."
                )
                unmapped = t.get("unmapped_sources", [])
                if unmapped:
                    parts.append(
                        f"I need the data files for these Tableau sources: "
                        f"{', '.join(unmapped)}. Please upload them so I can "
                        f"build your dashboard with real data."
                    )
            elif "alteryx" in str(stype) and "alteryx" in summary:
                a = summary["alteryx"]
                parts.append(
                    f"This is an Alteryx workflow with {a['tool_count']} tools."
                )

        # Multiple data files
        if summary["data_file_count"] > 1:
            data_names = [d["filename"] for d in session.data_files]
            parts.append(
                f"I have {summary['data_file_count']} data files loaded: "
                f"{', '.join(data_names)}. Using the largest as the primary dataset."
            )

        # Data-to-Tableau mapping
        mapping = summary.get("data_source_mapping", {})
        if mapping:
            for ds_name, info in mapping.items():
                parts.append(
                    f"Mapped Tableau source '{ds_name}' to {info['data_file']}."
                )

        # Present options (skip if still needs data)
        if not summary.get("needs_data"):
            parts.append(self._build_options_text())

        return " ".join(parts)

    def _build_options_text(self):
        """Build the numbered output options per CORE_BEHAVIOR."""
        if self.tableau_spec:
            return (
                "\n\nWhat would you like me to do?\n\n"
                "1. Translate to Power BI -- equivalent DAX measures, visuals, "
                "and data model in a .pbip project you can open in PBI Desktop.\n\n"
                "2. Build an interactive HTML dashboard -- from the data, "
                "opens in any browser, filters update every chart live.\n\n"
                "3. Both -- Power BI project for your analytics team plus an "
                "HTML dashboard for quick sharing.\n\n"
                "4. Migration report -- every calculated field translated with "
                "explanations.\n\n"
                "Who is the audience?"
            )
        return (
            "\n\nWhat would you like me to build?\n\n"
            "1. Interactive HTML Dashboard -- opens in any browser, filters "
            "update all charts live, works offline. Best for sharing and "
            "presenting.\n\n"
            "2. Power BI Project -- full .pbip with DAX measures, relationships, "
            "and formatted visuals. Best if your team uses Power BI Desktop.\n\n"
            "3. Both -- I'll generate the HTML dashboard for quick sharing "
            "and the Power BI project for your analytics team.\n\n"
            "Who is the audience?"
        )

    def respond(self, user_message, conversation_history, uploaded_files,
                progress_callback=None):
        """High-level response method for the split-panel UI.

        Auto-detects uploaded files the agent does not yet know about,
        injects context so Claude never asks for file paths, then
        delegates to chat().

        Returns:
            dict with "content", "downloads", "scores"
        """
        # Auto-detect uploaded files the agent doesn't know about yet
        if uploaded_files and self.dataframe is None:
            for name, info in uploaded_files.items():
                path = info.get("path", "")
                if path and os.path.exists(path):
                    self.data_file_path = path
                    self.data_path = path
                    ext = info.get("ext", path.rsplit(".", 1)[-1].lower())
                    try:
                        if ext == "csv":
                            self.dataframe = pd.read_csv(path)
                        elif ext in ("xlsx", "xls"):
                            from app.file_handler import load_excel_smart
                            self.dataframe, self.sheet_name = load_excel_smart(path)
                        elif ext == "parquet":
                            self.dataframe = pd.read_parquet(path)
                        elif ext == "json":
                            self.dataframe = pd.read_json(path)
                        if self.dataframe is not None and self.data_profile is None:
                            self.data_profile = self.analyzer.profile(self.dataframe)
                    except Exception:
                        pass
                    break

        # Enrich user message with context so Claude never asks for paths
        enriched = self._build_context_message(user_message)

        # Use existing chat() method with progress callback
        result = self.chat(enriched, progress_callback=progress_callback)

        # Translate to workspace-friendly format
        downloads = []
        for fpath in result.get("files", []):
            if os.path.exists(fpath):
                fname = os.path.basename(fpath)
                downloads.append({
                    "name": fname,
                    "description": "Generated file",
                    "path": fpath,
                    "filename": fname,
                })

        scores = None
        analyst_path = os.path.join(self.output_dir, "analyst_scores.json")
        if os.path.exists(analyst_path):
            try:
                with open(analyst_path, "r", encoding="utf-8") as f:
                    scores = json.load(f).get("scorecard")
            except Exception:
                pass

        return {
            "content": result.get("text", ""),
            "downloads": downloads,
            "scores": scores,
        }

    def _build_context_message(self, user_message):
        """Inject context about loaded data so Claude never asks for file paths."""
        # Prefer AgentSessionBridge for multi-file sessions
        if self.session_bridge:
            context = self.session_bridge.get_context()
            if context:
                return context + "\n\n" + user_message
            return user_message

        # Fallback: single-file context injection
        context_parts = []

        if self.data_file_path:
            context_parts.append(
                f"[SYSTEM: File loaded at {self.data_file_path}. "
                f"Do NOT ask user for file path -- you already have it.]"
            )

        if self.data_profile:
            rows = self.data_profile.get("row_count", "?")
            cols = self.data_profile.get("column_count", "?")
            col_names = [
                c["name"] for c in self.data_profile.get("columns", [])
            ]
            context_parts.append(
                f"[SYSTEM: Dataset has {rows} rows, {cols} columns: "
                f"{', '.join(col_names[:10])}. Profile is available. "
                f"Proceed with analysis -- do not ask for more info.]"
            )

        if self.dashboard_spec:
            context_parts.append(
                "[SYSTEM: Dashboard has been designed and is ready to build.]"
            )

        if self.tableau_spec:
            context_parts.append(
                "[SYSTEM: Tableau/Alteryx file has been parsed. Structure available.]"
            )

        if context_parts:
            return "\n".join(context_parts) + "\n\n" + user_message
        return user_message

    # ------------------------------------------------------------------ #
    #  Tool Execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute_tool(self, tool_name, tool_input):
        """Execute a tool and return the result string."""
        try:
            if tool_name == "analyze_data":
                return self._tool_analyze_data(tool_input)
            elif tool_name == "design_dashboard":
                return self._tool_design_dashboard(tool_input)
            elif tool_name == "build_html_dashboard":
                return self._tool_build_html(tool_input)
            elif tool_name == "build_pdf_report":
                return self._tool_build_pdf(tool_input)
            elif tool_name == "build_powerbi":
                return self._tool_build_powerbi(tool_input)
            elif tool_name == "build_documentation":
                return self._tool_build_docs(tool_input)
            elif tool_name == "parse_legacy_report":
                return self._tool_parse_legacy(tool_input)
            elif tool_name == "parse_alteryx_workflow":
                return self._tool_parse_alteryx(tool_input)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _tool_analyze_data(self, inputs):
        """Run DeepAnalyzer on the data."""
        file_path = inputs.get("file_path", self.data_path)

        if self.dataframe is not None:
            df = self.dataframe
        elif file_path:
            ext = file_path.rsplit(".", 1)[-1].lower()
            if ext == "csv":
                df = pd.read_csv(file_path)
            elif ext in ("xlsx", "xls"):
                from app.file_handler import load_excel_smart
                df, self.sheet_name = load_excel_smart(file_path)
            elif ext == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, sep=None, engine="python")
            self.dataframe = df
        else:
            return json.dumps({"error": "No data file available"})

        profile = self.analyzer.profile(df)
        self.data_profile = profile

        # Save profile to disk for internal pipeline use (NOT a user deliverable)
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        profile_path = output_dir / "data_profile.json"
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, default=str, ensure_ascii=False)

        return json.dumps(profile, default=str)

    def _tool_design_dashboard(self, inputs):
        """Design a dashboard using Claude with DASHBOARD_DESIGN_PROMPT."""
        request = inputs.get("request", "")
        audience = inputs.get("audience", "executive")
        profile_str = inputs.get("data_profile", "")

        if not profile_str and self.data_profile:
            profile_str = json.dumps(self.data_profile, default=str)

        user_msg = (
            f"USER REQUEST: {request}\n\n"
            f"TARGET AUDIENCE: {audience}\n\n"
            f"DATA PROFILE:\n{profile_str}\n\n"
            "Generate the complete dashboard specification JSON."
        )

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=0,
            system=DASHBOARD_DESIGN_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            import re
            raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
            raw = raw.strip()

        spec = json.loads(raw)
        self.dashboard_spec = spec

        # Save spec to disk for internal pipeline use (NOT a user deliverable)
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        spec_path = output_dir / "dashboard_spec.json"
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False, default=str)

        return json.dumps(spec, default=str)

    def _tool_build_html(self, inputs):
        """Build an interactive HTML dashboard directly from loaded data."""
        self._report_progress("Building interactive HTML dashboard...")
        title = inputs.get("title", "Dashboard")
        data_path = inputs.get("data_path", self.data_path)

        # Load data if we don't have a DataFrame yet
        if self.dataframe is None:
            if data_path and os.path.exists(data_path):
                ext = data_path.rsplit(".", 1)[-1].lower()
                if ext == "csv":
                    self.dataframe = pd.read_csv(data_path)
                elif ext in ("xlsx", "xls"):
                    from app.file_handler import load_excel_smart
                    self.dataframe, self.sheet_name = load_excel_smart(data_path)
                elif ext == "parquet":
                    self.dataframe = pd.read_parquet(data_path)
                elif ext == "json":
                    self.dataframe = pd.read_json(data_path)
                else:
                    self.dataframe = pd.read_csv(data_path, sep=None, engine="python")
            else:
                return json.dumps({"error": "No data available to build dashboard"})

        output_dir = str(PROJECT_ROOT / "output")
        filepath = self.html_builder.build(
            self.dataframe,
            output_dir=output_dir,
            title=title,
            data_path=data_path,
        )
        self.generated_files.append(filepath)
        return json.dumps({"status": "success", "file_path": filepath})

    def _tool_build_pdf(self, inputs):
        """Build a PDF report."""
        spec_str = inputs.get("spec", "")
        data_path = inputs.get("data_path", self.data_path)

        if spec_str:
            spec = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
        elif self.dashboard_spec:
            spec = self.dashboard_spec
        else:
            return json.dumps({"error": "No dashboard specification available"})

        profile = self.data_profile or {}
        output_dir = str(PROJECT_ROOT / "output")

        if self.dataframe is not None:
            filepath = self.pdf_builder.build_from_dataframe(
                spec, self.dataframe, profile, output_dir
            )
        elif data_path:
            filepath = self.pdf_builder.build(spec, data_path, profile, output_dir)
        else:
            return json.dumps({"error": "No data file available"})

        self.generated_files.append(filepath)
        return json.dumps({"status": "success", "file_path": filepath})

    def _report_progress(self, msg):
        """Send a progress update to the UI if a callback is set."""
        if hasattr(self, "_progress_callback") and self._progress_callback:
            self._progress_callback(msg)

    def _tool_build_powerbi(self, inputs):
        """Build a Power BI project using the full AI pipeline and ZIP it."""
        request = inputs.get("request", "Build a comprehensive dashboard")
        audience = inputs.get("audience", "executive")
        project_name = inputs.get("project_name", "Dashboard")

        if self.dataframe is None:
            return json.dumps({"error": "No data available. Upload a data file first."})

        self._report_progress("Step 1/5: Profiling data for Power BI...")

        # Use DataAnalyzer (not DeepAnalyzer) for the PBI pipeline --
        # ClaudeInterpreter expects table_name, column semantic types, etc.
        from core.data_analyzer import DataAnalyzer
        pbi_analyzer = DataAnalyzer()

        # Derive table name: prefer sheet name (matches Excel data), else file name
        if self.sheet_name:
            table_name = self.sheet_name.replace(" ", "_").replace("-", "_")
        elif self.data_file_path:
            base = os.path.basename(self.data_file_path)
            table_name = base.rsplit(".", 1)[0].replace(" ", "_").replace("-", "_")
        else:
            table_name = project_name.replace(" ", "_")

        pbi_profile = pbi_analyzer.analyze(self.dataframe, table_name=table_name)

        try:
            # Step 2: Claude interprets request -> dashboard spec
            self._report_progress(
                "Step 2/5: Claude is designing your dashboard "
                "(pages, visuals, DAX measures)..."
            )
            from core.claude_interpreter import ClaudeInterpreter
            interpreter = ClaudeInterpreter()
            dashboard_spec = interpreter.interpret(request, pbi_profile)
            self.dashboard_spec = dashboard_spec

            page_count = len(dashboard_spec.get("pages", []))
            visual_count = sum(
                len(p.get("visuals", []))
                for p in dashboard_spec.get("pages", [])
            )

            # Step 3: OpenAI generates PBIP config (report layout + data model)
            self._report_progress(
                f"Step 3/5: GPT-4 is generating Power BI layout "
                f"({page_count} pages, {visual_count} visuals)..."
            )
            from core.openai_engine import OpenAIEngine
            engine = OpenAIEngine()
            config = engine.generate_pbip_config(dashboard_spec, pbi_profile)

            # Step 4: PBIPGenerator creates the actual project files
            self._report_progress(
                "Step 4/5: Building Power BI project files "
                "(visuals, TMDL model, themes)..."
            )
            from generators.pbip_generator import PBIPGenerator
            output_dir = str(PROJECT_ROOT / "output")
            generator = PBIPGenerator(output_dir)
            result_path = generator.generate(
                config, pbi_profile, dashboard_spec,
                data_file_path=self.data_file_path,
                sheet_name=self.sheet_name,
            )

            # Step 5: Bundle data file + ZIP for download
            self._report_progress(
                "Step 5/5: Packaging Power BI project for download..."
            )

            # Copy the data file into the project so the user has it
            data_filename = ""
            if self.data_file_path and os.path.isfile(self.data_file_path):
                data_filename = os.path.basename(self.data_file_path)
                dst = os.path.join(result_path, data_filename)
                if not os.path.exists(dst):
                    shutil.copy2(self.data_file_path, dst)

            zip_name = project_name.replace(" ", "_")
            zip_path = shutil.make_archive(
                os.path.join(output_dir, zip_name), "zip", result_path
            )
            self.generated_files.append(zip_path)

            self._report_progress("Power BI project ready for download")

            # Include clear instructions in the result
            safe_name = generator._safe_name(
                dashboard_spec.get("dashboard_title", "Dashboard")
            )
            return json.dumps({
                "status": "success",
                "file_path": zip_path,
                "project_name": project_name,
                "pages": page_count,
                "visuals": visual_count,
                "measures": len(dashboard_spec.get("measures", [])),
                "instructions": (
                    f"Extract the ZIP to a folder. The data file "
                    f"({data_filename}) is included. Double-click "
                    f"{safe_name}.pbip to open in Power BI Desktop. "
                    f"If Power BI says 'file not found', update the "
                    f"data source path in Power Query Editor to point "
                    f"to {data_filename} in the extracted folder."
                ),
            })

        except Exception as e:
            self._report_progress(f"Power BI generation failed: {e}")
            return json.dumps({
                "error": f"Power BI generation failed: {e}",
                "suggestion": "Try building an HTML dashboard instead."
            })

    def _tool_build_docs(self, inputs):
        """Build documentation."""
        doc_type = inputs.get("doc_type", "executive_summary")
        spec_str = inputs.get("spec", "")
        profile_str = inputs.get("profile", "")

        spec = None
        if spec_str:
            spec = json.loads(spec_str) if isinstance(spec_str, str) else spec_str
        elif self.dashboard_spec:
            spec = self.dashboard_spec

        profile = None
        if profile_str:
            profile = json.loads(profile_str) if isinstance(profile_str, str) else profile_str
        elif self.data_profile:
            profile = self.data_profile

        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        if doc_type == "data_dictionary":
            content = self._generate_data_dictionary(profile)
            filepath = output_dir / f"data_dictionary_{timestamp}.md"
        elif doc_type == "technical_spec":
            content = self._generate_technical_spec(spec, profile)
            filepath = output_dir / f"technical_spec_{timestamp}.md"
        elif doc_type == "executive_summary":
            content = self._generate_exec_summary(spec, profile)
            filepath = output_dir / f"executive_summary_{timestamp}.md"
        elif doc_type == "full_package":
            content = (
                self._generate_exec_summary(spec, profile) + "\n\n---\n\n" +
                self._generate_technical_spec(spec, profile) + "\n\n---\n\n" +
                self._generate_data_dictionary(profile)
            )
            filepath = output_dir / f"full_documentation_{timestamp}.md"
        else:
            return json.dumps({"error": f"Unknown doc type: {doc_type}"})

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        self.generated_files.append(str(filepath))
        return json.dumps({"status": "success", "file_path": str(filepath)})

    def _tool_parse_legacy(self, inputs):
        """Parse a Tableau or Business Objects file."""
        from app.file_handler import ingest_file
        file_path = inputs.get("file_path", "")
        if not file_path:
            return json.dumps({"error": "No file path provided"})

        result = ingest_file(file_path)

        # If we got dataframes, store the first one
        dfs = result.get("dataframes", {})
        if dfs and self.dataframe is None:
            first_key = next(iter(dfs))
            self.dataframe = dfs[first_key]

        # Can't serialize DataFrames directly
        serializable = {
            k: v for k, v in result.items() if k != "dataframes"
        }
        serializable["dataframe_names"] = list(dfs.keys())
        serializable["dataframe_shapes"] = {
            k: list(v.shape) for k, v in dfs.items()
        }

        return json.dumps(serializable, default=str)

    def _tool_parse_alteryx(self, inputs):
        """Parse Alteryx workflow and generate Dataiku migration plan."""
        from migration.alteryx_parser import AlteryxParser
        from migration.alteryx_to_dataiku import AlteryxToDataikuTranslator

        file_path = inputs.get("file_path", "")
        generate_code = inputs.get("generate_dataiku_code", True)

        # Parse the Alteryx file
        parser = AlteryxParser()
        alteryx_spec = parser.parse(file_path)

        # Translate to Dataiku
        translator = AlteryxToDataikuTranslator()
        translation = translator.translate(alteryx_spec)

        # Generate markdown migration report
        report_md = translator.generate_migration_report_md(translation)
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "alteryx_migration_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        self.generated_files.append(report_path)

        # Save full translation as JSON
        json_path = os.path.join(self.output_dir, "alteryx_translation.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(translation, f, indent=2, default=str)
        self.generated_files.append(json_path)

        # Generate Python recipes if requested
        if generate_code and translation.get("python_recipes"):
            for recipe_name, code in translation["python_recipes"].items():
                code_path = os.path.join(
                    self.output_dir, f"dataiku_recipe_{recipe_name}.py"
                )
                with open(code_path, "w", encoding="utf-8") as f:
                    f.write(code)
                self.generated_files.append(code_path)

        # Return summary for Claude to narrate
        return json.dumps({
            "alteryx_spec": {
                "workflow_name": alteryx_spec.get("workflow_name", ""),
                "total_tools": alteryx_spec.get("tool_summary", {}).get("total_tools", 0),
                "tool_summary": alteryx_spec.get("tool_summary", {}),
                "complexity": alteryx_spec.get("complexity_score", ""),
                "data_sources": alteryx_spec.get("data_sources", []),
                "formulas_count": len(alteryx_spec.get("formulas", [])),
                "joins_count": len(alteryx_spec.get("joins", [])),
                "filters_count": len(alteryx_spec.get("filters", [])),
                "summarizations_count": len(alteryx_spec.get("summarizations", [])),
                "warnings": alteryx_spec.get("warnings", []),
            },
            "translation": {
                "auto_pct": translation["migration_plan"]["auto_pct"],
                "auto_count": translation["migration_plan"]["auto_translatable"],
                "manual_count": translation["migration_plan"]["manual_required"],
                "estimated_effort": translation["migration_plan"]["estimated_effort"],
                "recommendations": translation.get("recommendations", []),
            },
            "generated_files": [report_path, json_path],
        }, default=str)

    # ------------------------------------------------------------------ #
    #  Documentation generators                                            #
    # ------------------------------------------------------------------ #

    def _generate_data_dictionary(self, profile):
        """Generate a markdown data dictionary from profile."""
        if not profile:
            return "# Data Dictionary\n\nNo data profile available."

        lines = [
            "# Data Dictionary",
            "",
            f"**Rows:** {profile.get('row_count', 'N/A'):,}",
            f"**Columns:** {profile.get('column_count', 'N/A')}",
            f"**Memory:** {profile.get('memory_mb', 'N/A')} MB",
            "",
            "## Column Details",
            "",
            "| Column | Type | Semantic | Nulls | Unique | Notes |",
            "|--------|------|----------|-------|--------|-------|",
        ]
        for col in profile.get("columns", []):
            null_info = f"{col.get('null_pct', 0)}%"
            unique_info = f"{col.get('unique_count', 'N/A')}"
            notes = ""
            if col.get("distribution"):
                notes = col["distribution"]
            elif col.get("mode"):
                notes = f"mode: {col['mode']}"
            lines.append(
                f"| {col['name']} | {col['dtype']} | {col.get('semantic_type', '')} "
                f"| {null_info} | {unique_info} | {notes} |"
            )

        # Quality section
        dq = profile.get("data_quality", {})
        lines.extend([
            "",
            "## Data Quality",
            "",
            f"- Total null cells: {dq.get('total_nulls', 0):,} ({dq.get('total_null_pct', 0)}%)",
            f"- Duplicate rows: {dq.get('duplicate_rows', 0):,} ({dq.get('duplicate_row_pct', 0)}%)",
        ])
        if dq.get("constant_columns"):
            lines.append(f"- Constant columns: {', '.join(dq['constant_columns'])}")

        return "\n".join(lines)

    def _generate_technical_spec(self, spec, profile):
        """Generate a technical specification document."""
        if not spec:
            return "# Technical Specification\n\nNo dashboard specification available."

        lines = [
            "# Technical Specification",
            "",
            f"**Dashboard:** {spec.get('title', 'N/A')}",
            f"**Audience:** {spec.get('audience', 'N/A')}",
            "",
        ]

        # Visuals
        for page in spec.get("pages", []):
            lines.append(f"## Page: {page.get('page_name', 'Unnamed')}")
            for v in page.get("visuals", []):
                lines.append(f"- **{v.get('title', 'N/A')}** ({v.get('type', '?')})")
                if v.get("design_rationale"):
                    lines.append(f"  Rationale: {v['design_rationale']}")

        # Measures
        measures = spec.get("calculated_measures", [])
        if measures:
            lines.extend(["", "## Calculated Measures", ""])
            for m in measures:
                lines.append(f"- **{m.get('name', '')}**: `{m.get('formula', '')}`")
                if m.get("explanation"):
                    lines.append(f"  {m['explanation']}")

        return "\n".join(lines)

    def _generate_exec_summary(self, spec, profile):
        """Generate an executive summary."""
        lines = [
            "# Executive Summary",
            "",
            f"**Dashboard:** {spec.get('title', 'N/A') if spec else 'N/A'}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if spec and spec.get("design_notes"):
            lines.extend([spec["design_notes"], ""])

        if profile:
            lines.extend([
                "## Data Overview",
                "",
                f"The dataset contains {profile.get('row_count', 0):,} records "
                f"across {profile.get('column_count', 0)} fields.",
                "",
            ])
            for insight in profile.get("quick_insights", []):
                lines.append(f"- {insight}")

        return "\n".join(lines)
