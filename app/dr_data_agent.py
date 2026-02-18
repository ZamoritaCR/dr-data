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
import numpy as np

try:
    from openai import OpenAI as _OpenAIClient
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

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
from core.export_engine import ExportEngine
from core.interactive_dashboard import InteractiveDashboard
from core.trace_logger import TraceLogger

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
        self.export_engine = ExportEngine()
        self.dashboard_builder = InteractiveDashboard()

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

        # Snowflake data
        self.snowflake_tables = {}

        # Trace logger for audit trail
        self.trace = TraceLogger()

        # OpenAI secondary engine (for large-context / data-heavy tasks)
        self.openai_client = None
        openai_key = os.getenv("OPENAI_API_KEY") or _get_secret("OPENAI_API_KEY")
        if openai_key and _HAS_OPENAI:
            self.openai_client = _OpenAIClient(api_key=openai_key)
        elif not openai_key:
            print("[INFO] OPENAI_API_KEY not set -- using Claude for all requests.")

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

            # Retry loop for API resilience
            response = None
            last_err = None
            for attempt in range(3):
                try:
                    response = self.client.messages.create(
                        model=self.MODEL,
                        max_tokens=self.MAX_TOKENS,
                        temperature=0,
                        system=DR_DATA_SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=self.messages,
                        timeout=120.0,
                    )
                    break
                except anthropic.RateLimitError as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(2)
                except anthropic.APIError as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(2)
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(2)

            if response is None:
                # All retries failed -- clean up and return friendly error
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()
                return {
                    "text": (
                        "Hit a snag reaching my analysis engine. "
                        "Give me a second and try again."
                    ),
                    "files": [],
                    "profile": None,
                    "spec": None,
                }

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

            # Trace: log Claude call
            _resp_summary = ""
            for b in assistant_content:
                if hasattr(b, "text") and b.text:
                    _resp_summary = b.text[:200]
                    break
                if hasattr(b, "name"):
                    _resp_summary = f"tool_use:{b.name}"
                    break
            self.trace.log_llm_call(
                "claude", self.MODEL,
                user_message[:200] if isinstance(user_message, str) else "tool_results",
                _resp_summary,
            )

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

        # Trace: log each loaded file
        for d in session.data_files:
            ddf = d.get("df")
            if ddf is not None:
                self.trace.log_file_loaded(
                    d["filename"], len(ddf), len(ddf.columns),
                    str(ddf.dtypes.value_counts().to_dict()),
                )

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

    # ------------------------------------------------------------------ #
    #  OpenAI secondary engine                                             #
    # ------------------------------------------------------------------ #

    def _call_openai(self, system_prompt, user_message, model="gpt-4o"):
        """Call OpenAI chat completions with retry logic.

        Returns response text, or None if all retries fail.
        """
        if not self.openai_client:
            return None

        last_err = None
        for attempt in range(3):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=model,
                    temperature=0.7,
                    max_tokens=4000,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    timeout=120.0,
                )
                text = resp.choices[0].message.content or ""
                self.trace.log_llm_call(
                    "openai", model,
                    user_message[:200], text[:200],
                )
                return text
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2)

        print(f"[WARN] OpenAI failed after 3 attempts: {last_err}")
        return None

    # Keywords that signal a heavy analysis / generation task
    _HEAVY_KEYWORDS = (
        "report", "dashboard", "analysis", "analyze", "summary",
        "generate", "build", "create", "design", "insight",
        "trend", "forecast", "model", "compare", "breakdown",
    )

    def _route_request(self, message, context=""):
        """Route a request to OpenAI or Claude based on context size and intent.

        Args:
            message: The user/internal message text.
            context: Additional context string (data stats, etc.).

        Returns:
            (response_text, engine_used) tuple.
            engine_used is "openai" or "claude".
        """
        full_text = context + "\n" + message if context else message
        estimated_tokens = len(full_text) // 4

        msg_lower = message.lower()
        is_heavy = (
            estimated_tokens > 5000
            or any(kw in msg_lower for kw in self._HEAVY_KEYWORDS)
        )

        # Try OpenAI for heavy tasks
        if is_heavy and self.openai_client:
            result = self._call_openai(DR_DATA_SYSTEM_PROMPT, full_text)
            if result is not None:
                return result, "openai"
            # OpenAI failed -- fall back to Claude with truncated context
            if estimated_tokens > 5000:
                # Truncate context to ~4000 tokens worth
                max_chars = 16000
                if len(full_text) > max_chars:
                    full_text = full_text[:max_chars] + "\n[...context truncated...]"

        # Use Claude (chat loop with tool calling)
        chat_result = self.chat(full_text)
        return chat_result.get("text", ""), "claude"

    def analyze_uploaded_file(self, file_path=None):
        """Auto-analyze uploaded file(s) and return Dr. Data's LLM response.

        Gathers data context (rows, columns, stats, structure) then sends
        it through the LLM so Dr. Data's personality controls the greeting.
        """
        # Multi-file session path
        if self.session:
            context = self._build_upload_context_from_session()
        elif file_path:
            context = self._build_upload_context_from_file(file_path)
        else:
            return "No file provided."

        # Route through the LLM so Dr. Data's voice comes through
        internal_prompt = (
            "A new file was uploaded. Here is what I computed from "
            "the raw data. Use these specific numbers in your analysis. "
            "Lead with the most interesting finding -- a correlation, "
            "an outlier, a trend, a concentration risk, whatever jumps out. "
            "Be specific with numbers. Suggest what to build based on what "
            "the data is telling you. Do NOT use numbered lists or bullet "
            "points. Do NOT just describe the file metadata. "
            "Find the STORY in the data."
        )

        text, engine = self._route_request(internal_prompt, context=context)
        return text

    def _build_upload_context_from_file(self, file_path):
        """Load a single file and return context string for the LLM."""
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
            self.trace.log_file_loaded(
                os.path.basename(file_path),
                len(self.dataframe), len(self.dataframe.columns),
                str(self.dataframe.dtypes.value_counts().to_dict()),
            )

        # Store report structure (Tableau, Alteryx, Business Objects)
        if result.get("report_structure"):
            self.tableau_spec = result["report_structure"]

        return self._format_data_context(
            file_names=[os.path.basename(file_path)]
        )

    def _build_upload_context_from_session(self):
        """Build context string from MultiFileSession for the LLM."""
        session = self.session
        summary = session.get_summary()

        # Sync data from session
        df = session.get_primary_dataframe()
        if df is not None and self.dataframe is None:
            self.dataframe = df
            self.data_profile = self.analyzer.profile(df)

        file_names = [f["filename"] for f in summary["files"]]

        # Extra structure context
        extra_parts = []

        if summary.get("has_structure"):
            stype = summary.get("structure_type", "")
            if "tableau" in stype and "tableau" in summary:
                t = summary["tableau"]
                extra_parts.append(
                    f"Tableau workbook: {t['worksheet_count']} worksheets, "
                    f"{t['dashboard_count']} dashboards, "
                    f"{t['calculated_fields']} calculated fields."
                )
                unmapped = t.get("unmapped_sources", [])
                if unmapped:
                    extra_parts.append(
                        f"Still need data files for: {', '.join(unmapped)}."
                    )
            elif "alteryx" in str(stype) and "alteryx" in summary:
                a = summary["alteryx"]
                extra_parts.append(
                    f"Alteryx workflow: {a['tool_count']} tools."
                )

        if summary["data_file_count"] > 1:
            data_names = [d["filename"] for d in session.data_files]
            extra_parts.append(
                f"Multiple data files: {', '.join(data_names)}. "
                f"Using the largest as primary."
            )

        mapping = summary.get("data_source_mapping", {})
        if mapping:
            for ds_name, info in mapping.items():
                extra_parts.append(
                    f"Mapped source '{ds_name}' to {info['data_file']}."
                )

        # Include auto-detected relationships between tables
        if session.relationships:
            rel_summary = session.detector.summarize(session.relationships)
            extra_parts.append(rel_summary)
            self.trace.log_relationships(session.relationships)
        if session.join_log:
            extra_parts.append("JOIN LOG: " + " | ".join(session.join_log))

        # If session was auto-unified, use the unified DataFrame
        if session.unified_df is not None and self.dataframe is not session.unified_df:
            self.dataframe = session.unified_df
            self.data_profile = self.analyzer.profile(self.dataframe)

        return self._format_data_context(
            file_names=file_names,
            extra=extra_parts,
            needs_data=summary.get("needs_data", False),
        )

    def _format_data_context(self, file_names, extra=None, needs_data=False):
        """Compute deep insights from the DataFrame and format for the LLM."""
        parts = [f"Files loaded: {', '.join(file_names)}."]

        if self.dataframe is not None:
            df = self.dataframe
            parts.append(
                f"Dataset: {len(df):,} rows, {len(df.columns)} columns."
            )
            parts.append(f"Columns: {', '.join(df.columns.tolist())}.")

            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()

            # --- Numeric column stats ---
            if num_cols:
                stats_lines = []
                for col in num_cols[:10]:
                    s = df[col].dropna()
                    if len(s) == 0:
                        continue
                    stats_lines.append(
                        f"  {col}: min={s.min():,.2f}, max={s.max():,.2f}, "
                        f"mean={s.mean():,.2f}, median={s.median():,.2f}, "
                        f"std={s.std():,.2f}"
                    )
                parts.append("NUMERIC STATS:\n" + "\n".join(stats_lines))

            # --- Top correlations ---
            if len(num_cols) >= 2:
                try:
                    corr = df[num_cols].corr()
                    # Get upper triangle pairs
                    pairs = []
                    for i in range(len(corr.columns)):
                        for j in range(i + 1, len(corr.columns)):
                            val = corr.iloc[i, j]
                            if pd.notna(val):
                                pairs.append((
                                    corr.columns[i], corr.columns[j],
                                    val
                                ))
                    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    if pairs:
                        corr_lines = []
                        for a, b, r in pairs[:5]:
                            direction = "positive" if r > 0 else "negative"
                            corr_lines.append(
                                f"  {a} <-> {b}: r={r:.3f} ({direction})"
                            )
                        parts.append(
                            "TOP CORRELATIONS:\n" + "\n".join(corr_lines)
                        )
                except Exception:
                    pass

            # --- Outliers (>3 std from mean) ---
            outlier_findings = []
            for col in num_cols[:10]:
                s = df[col].dropna()
                if len(s) < 10:
                    continue
                mean, std = s.mean(), s.std()
                if std == 0:
                    continue
                outlier_mask = (s - mean).abs() > 3 * std
                n_outliers = int(outlier_mask.sum())
                if n_outliers > 0:
                    pct = n_outliers / len(s) * 100
                    outlier_findings.append(
                        f"  {col}: {n_outliers} outliers ({pct:.1f}% of values)"
                    )
            if outlier_findings:
                parts.append(
                    "OUTLIERS (>3 std from mean):\n"
                    + "\n".join(outlier_findings)
                )

            # --- Null analysis per column ---
            null_cols = []
            for col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null > 0:
                    pct = n_null / len(df) * 100
                    null_cols.append(f"  {col}: {n_null:,} nulls ({pct:.1f}%)")
            if null_cols:
                parts.append(
                    "NULL ANALYSIS:\n" + "\n".join(null_cols[:10])
                )

            # --- Date/time analysis ---
            date_cols = [
                c for c in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[c])
            ]
            # Also try to detect string date columns
            if not date_cols:
                for col in cat_cols[:5]:
                    try:
                        parsed = pd.to_datetime(
                            df[col], infer_datetime_format=True, errors="raise"
                        )
                        date_cols.append(col)
                        df[col] = parsed  # convert in place for trend calc
                        break
                    except Exception:
                        pass

            if date_cols:
                dc = date_cols[0]
                d_series = df[dc].dropna()
                if len(d_series) > 0:
                    parts.append(
                        f"TIME RANGE: {dc} spans from {d_series.min()} "
                        f"to {d_series.max()}."
                    )
                    # Trend direction for numeric cols
                    if num_cols:
                        trend_lines = []
                        sorted_df = df.dropna(subset=[dc]).sort_values(dc)
                        n = len(sorted_df)
                        if n >= 4:
                            first_q = sorted_df.head(n // 4)
                            last_q = sorted_df.tail(n // 4)
                            for col in num_cols[:5]:
                                early = first_q[col].mean()
                                late = last_q[col].mean()
                                if early != 0:
                                    change = (late - early) / abs(early) * 100
                                    direction = "UP" if change > 0 else "DOWN"
                                    trend_lines.append(
                                        f"  {col}: {direction} {abs(change):.1f}% "
                                        f"(first quarter avg {early:,.2f} -> "
                                        f"last quarter avg {late:,.2f})"
                                    )
                            if trend_lines:
                                parts.append(
                                    "TREND DIRECTION (first vs last quarter "
                                    "of date range):\n"
                                    + "\n".join(trend_lines)
                                )

            # --- Categorical value counts ---
            cat_insights = []
            for col in cat_cols[:8]:
                nunique = df[col].nunique()
                if 1 < nunique <= 20:
                    top = df[col].value_counts().head(5)
                    top_str = ", ".join(
                        f"{v}={c}" for v, c in top.items()
                    )
                    cat_insights.append(
                        f"  {col} ({nunique} unique): top values: {top_str}"
                    )
            if cat_insights:
                parts.append(
                    "CATEGORICAL BREAKDOWN:\n" + "\n".join(cat_insights)
                )

            # --- Concentration analysis ---
            if num_cols and cat_cols:
                try:
                    nc = num_cols[0]
                    cc = cat_cols[0]
                    grouped = df.groupby(cc)[nc].sum().sort_values(
                        ascending=False
                    )
                    total = grouped.sum()
                    if total > 0 and len(grouped) >= 3:
                        top_n = min(5, len(grouped))
                        top_share = grouped.head(top_n).sum() / total * 100
                        parts.append(
                            f"CONCENTRATION: Top {top_n} values of '{cc}' "
                            f"account for {top_share:.1f}% of total '{nc}'."
                        )
                except Exception:
                    pass

            # Quick insights from profiler
            if self.data_profile:
                insights = self.data_profile.get("quick_insights", [])
                if insights:
                    parts.append(
                        "PROFILER INSIGHTS: " + " ".join(insights[:4])
                    )

        if self.tableau_spec:
            rtype = self.tableau_spec.get("type", "")
            if "tableau" in rtype:
                ws_count = len(self.tableau_spec.get("worksheets", []))
                ds_count = len(self.tableau_spec.get("datasources", []))
                parts.append(
                    f"Tableau workbook: {ws_count} worksheets, "
                    f"{ds_count} data sources."
                )
            elif "alteryx" in rtype:
                tool_count = self.tableau_spec.get(
                    "tool_summary", {}
                ).get("total_tools", 0)
                parts.append(f"Alteryx workflow: {tool_count} tools.")
            elif "business_objects" in rtype:
                parts.append("Business Objects report.")

        if extra:
            parts.extend(extra)

        if needs_data:
            parts.append("NOTE: Still waiting for data files to be uploaded.")

        # Trace: log the analysis performed
        if self.dataframe is not None:
            methods = ["statistical_profiling"]
            insights_found = []
            cols_analyzed = list(self.dataframe.columns[:15])
            if len(self.dataframe.select_dtypes(include="number").columns) >= 2:
                methods.append("correlation_analysis")
            methods.extend(["outlier_detection", "null_analysis", "trend_analysis"])
            # Extract short insight summaries from parts
            for p in parts:
                if any(tag in p for tag in ("CORRELATION", "OUTLIER", "TREND", "CONCENTRATION")):
                    insights_found.append(p.split("\n")[0][:150])
            self.trace.log_analysis(insights_found, cols_analyzed, methods)

        return "\n".join(parts)

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

        # --- Export interception: handle deliverables before LLM ---
        msg_lower = user_message.lower()

        # Power BI checked FIRST — highest priority
        want_pbi = any(k in msg_lower for k in (
            "power bi", "powerbi", "pbi", "pbip", "pbix", "power_bi",
        ))
        # Dashboard only when NOT a PBI request (avoid "powerbi dashboard" -> HTML)
        want_dash = (
            any(k in msg_lower for k in (
                "dashboard", "html", "interactive", "explore",
                "drill down", "filter",
            ))
            and not want_pbi
        )
        want_pptx = any(k in msg_lower for k in (
            "powerpoint", "pptx", "presentation", "slides", "deck",
        ))
        want_pdf = any(k in msg_lower for k in ("pdf", "report"))
        want_docx = any(k in msg_lower for k in ("word", "docx", "document"))
        want_all = any(k in msg_lower for k in (
            "all formats", "all three", "everything",
        ))

        is_export = (
            want_pptx or want_pdf or want_docx
            or want_dash or want_pbi or want_all
        )

        print(f"[EXPORT CHECK] Message: {user_message[:80]}")
        print(f"[EXPORT CHECK] wants_pbi={want_pbi}, wants_dashboard={want_dash}, "
              f"wants_pptx={want_pptx}, wants_pdf={want_pdf}, wants_docx={want_docx}")
        print(f"[EXPORT CHECK] DataFrame available: {self.dataframe is not None}, "
              f"shape: {self.dataframe.shape if self.dataframe is not None else 'None'}")
        print(f'[ROUTE] Message: {user_message[:50]}... -> {"export" if is_export else "chat"}')

        if is_export:
            # Guard: need data first
            if self.dataframe is None:
                return {
                    "type": "chat",
                    "content": (
                        "Upload a file first and I will build that for you. "
                        "Drop a CSV, Excel, or any data file in the sidebar "
                        "and I will get right on it."
                    ),
                }

            if want_all:
                want_pptx = want_pdf = want_docx = want_dash = want_pbi = True

            # -- Build instant acknowledgment --
            dataset_name = "your dataset"
            if self.data_file_path:
                dataset_name = os.path.splitext(
                    os.path.basename(self.data_file_path)
                )[0].replace("_", " ").strip() or "your dataset"
            items = []
            if want_dash:
                items.append("interactive dashboard")
            if want_pbi:
                items.append("Power BI project")
            if want_pptx:
                items.append("PowerPoint")
            if want_pdf:
                items.append("PDF report")
            if want_docx:
                items.append("Word document")
            if not items:
                items.append("deliverables")
            item_str = (" and ".join(items) if len(items) <= 2
                        else ", ".join(items[:-1]) + ", and " + items[-1])
            row_count = len(self.dataframe)
            col_count = len(self.dataframe.columns)
            acknowledgment = (
                f"On it. Building your {item_str} from {dataset_name} "
                f"({row_count:,} rows, {col_count} columns). "
                f"This will take a moment."
            )

            # -- Progress tracking --
            progress_steps = []
            def _progress(msg):
                progress_steps.append(msg)
                if progress_callback:
                    progress_callback(msg)

            title = "Dr. Data Report"
            if self.data_file_path:
                base = os.path.splitext(
                    os.path.basename(self.data_file_path)
                )[0]
                title = base.replace("_", " ").strip() or title
            safe = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in title
            )
            import traceback as _tb
            os.makedirs(self.output_dir, exist_ok=True)
            downloads = []
            errors = []

            _progress(
                f"Reading data profile... "
                f"{row_count:,} rows across {col_count} columns"
            )

            # -- Interactive HTML Dashboard --
            if want_dash:
                print('[BUILD] Starting dashboard generation...')
                _progress("Generating interactive dashboard...")
                try:
                    p = self.dashboard_builder.generate(
                        self.dataframe, title,
                        os.path.join(
                            self.output_dir, f"{safe}_dashboard.html"
                        ),
                    )
                    if p:
                        downloads.append({
                            "name": "Interactive Dashboard",
                            "filename": f"{safe}_dashboard.html",
                            "path": p,
                            "description": (
                                "Filterable charts and data table with "
                                "search, sort, copy, and CSV export"
                            ),
                        })
                        self.trace.log_deliverable(
                            "Interactive Dashboard", p,
                            {"title": title, "type": "html"},
                        )
                        _progress("  Dashboard complete.")
                except Exception as e:
                    print('[BUILD] Dashboard FAILED:')
                    _tb.print_exc()
                    errors.append(f"Interactive Dashboard: {e}")
                    _progress(f"  Dashboard failed: {str(e)[:80]}")

            # -- PowerPoint --
            if want_pptx:
                print('[BUILD] Starting PowerPoint generation...')
                _progress("Generating PowerPoint presentation...")
                try:
                    p = self.export_engine.generate_pptx(
                        self.dataframe, title,
                        os.path.join(self.output_dir, f"{safe}.pptx"),
                    )
                    if p:
                        downloads.append({
                            "name": "PowerPoint Presentation",
                            "filename": f"{safe}.pptx",
                            "path": p,
                            "description": (
                                "Professional presentation with key "
                                "stats and trends"
                            ),
                        })
                        self.trace.log_deliverable(
                            "PowerPoint", p,
                            {"title": title, "type": "pptx"},
                        )
                        _progress("  PowerPoint complete.")
                except Exception as e:
                    print('[BUILD] PowerPoint FAILED:')
                    _tb.print_exc()
                    errors.append(f"PowerPoint: {e}")
                    _progress(f"  PowerPoint failed: {str(e)[:80]}")

            # -- PDF --
            if want_pdf:
                print('[BUILD] Starting PDF generation...')
                _progress("Generating PDF report...")
                try:
                    p = self.export_engine.generate_pdf(
                        self.dataframe, title,
                        os.path.join(self.output_dir, f"{safe}.pdf"),
                    )
                    if p:
                        downloads.append({
                            "name": "PDF Report",
                            "filename": f"{safe}.pdf",
                            "path": p,
                            "description": (
                                "Executive PDF report with data summary"
                            ),
                        })
                        self.trace.log_deliverable(
                            "PDF Report", p,
                            {"title": title, "type": "pdf"},
                        )
                        _progress("  PDF complete.")
                except Exception as e:
                    print('[BUILD] PDF FAILED:')
                    _tb.print_exc()
                    errors.append(f"PDF Report: {e}")
                    _progress(f"  PDF failed: {str(e)[:80]}")

            # -- Word --
            if want_docx:
                print('[BUILD] Starting Word document generation...')
                _progress("Generating Word document...")
                try:
                    p = self.export_engine.generate_docx(
                        self.dataframe, title,
                        os.path.join(self.output_dir, f"{safe}.docx"),
                    )
                    if p:
                        downloads.append({
                            "name": "Word Document",
                            "filename": f"{safe}.docx",
                            "path": p,
                            "description": (
                                "Detailed Word document with data analysis"
                            ),
                        })
                        self.trace.log_deliverable(
                            "Word Document", p,
                            {"title": title, "type": "docx"},
                        )
                        _progress("  Word document complete.")
                except Exception as e:
                    print('[BUILD] Word FAILED:')
                    _tb.print_exc()
                    errors.append(f"Word Document: {e}")
                    _progress(f"  Word document failed: {str(e)[:80]}")

            # -- Power BI project (direct call, no LLM routing) --
            if want_pbi:
                _progress("Generating Power BI project...")
                print(f"[PBI] Starting Power BI generation...")
                print(f"[PBI] DataFrame available: {self.dataframe is not None}, "
                      f"shape: {self.dataframe.shape if self.dataframe is not None else 'None'}")
                try:
                    pbi_inputs = {
                        "request": user_message,
                        "audience": "executive",
                        "project_name": title,
                    }
                    pbi_result_json = self._tool_build_powerbi(pbi_inputs)
                    print(f"[PBI] Generation complete: {pbi_result_json[:200] if pbi_result_json else 'None'}")
                    pbi_result = json.loads(pbi_result_json)
                    if "error" in pbi_result:
                        errors.append(f"Power BI Project: {pbi_result['error']}")
                        _progress(f"  Power BI failed: {pbi_result['error'][:80]}")
                    else:
                        zip_path = pbi_result.get("file_path")
                        pbi_build_context = pbi_result.get("build_context", {})
                        if zip_path and os.path.exists(zip_path):
                            fname = os.path.basename(zip_path)
                            downloads.append({
                                "name": "Power BI Project",
                                "filename": fname,
                                "path": zip_path,
                                "description": (
                                    "Full .pbip project with DAX measures, "
                                    "visuals, and data model"
                                ),
                            })
                            self.trace.log_deliverable(
                                "Power BI Project", zip_path,
                                {"title": title, "type": "pbip",
                                 "build_context": pbi_build_context},
                            )
                        _progress("  Power BI project complete.")
                except Exception as e:
                    print(f"[PBI] GENERATION FAILED:")
                    _tb.print_exc()
                    errors.append(f"Power BI Project: {e}")
                    _progress(f"  Power BI failed: {str(e)[:80]}")

            # -- Generate trace log and add to downloads --
            if downloads:
                _progress("Generating trace log...")
                try:
                    trace_path = self.trace.generate_trace_doc(
                        os.path.join(self.output_dir, "trace_log.html"),
                        format="html",
                    )
                    downloads.append({
                        "name": "Processing Trace Log",
                        "filename": "trace_log.html",
                        "path": trace_path,
                        "description": (
                            "Full audit trail: files loaded, analysis "
                            "performed, LLM calls, QA recommendations"
                        ),
                    })
                except Exception:
                    pass

            # Build followup context for LLM narration
            detail_lines = []
            for d in downloads:
                detail_lines.append(
                    f"- {d['name']}: {d.get('description', '')}"
                )
            if errors:
                for e in errors:
                    detail_lines.append(f"- Failed: {e}")
            data_hint = ""
            if self.data_profile:
                insights = self.data_profile.get("quick_insights", [])
                if insights:
                    data_hint = (
                        "\nInteresting things in the data: "
                        + "; ".join(insights[:3])
                    )

            # Enhanced followup: include PBI build context when available
            pbi_bc = locals().get("pbi_build_context", {})
            if pbi_bc:
                # Build the detailed context for LLM narration
                page_detail = ", ".join(
                    f"{n} ({c} visuals)"
                    for n, c in zip(
                        pbi_bc.get("page_names", []),
                        pbi_bc.get("visuals_per_page", []),
                    )
                )
                measures_created = pbi_bc.get("measures_created", [])
                measures_skipped = pbi_bc.get("measures_skipped", [])
                fields_fixed = pbi_bc.get("fields_fixed", 0)
                fields_removed = pbi_bc.get("fields_removed", 0)

                pbi_detail = (
                    f"\n\nPOWER BI BUILD DETAILS:\n"
                    f"- Data file: {pbi_bc.get('data_file', '?')}, "
                    f"{pbi_bc.get('row_count', 0):,} rows x "
                    f"{pbi_bc.get('col_count', 0)} columns "
                    f"({pbi_bc.get('numeric_count', 0)} numeric, "
                    f"{pbi_bc.get('categorical_count', 0)} categorical)\n"
                    f"- Table name in model: {pbi_bc.get('table_name', '?')}\n"
                    f"- Pages: {page_detail}\n"
                    f"- DAX measures created: "
                    f"{', '.join(measures_created) if measures_created else 'none'}\n"
                )
                if measures_skipped:
                    pbi_detail += (
                        f"- DAX measures SKIPPED (bad column refs): "
                        f"{', '.join(measures_skipped)}\n"
                    )
                if fields_fixed:
                    pbi_detail += (
                        f"- Visual fields auto-fixed (name mismatch): "
                        f"{fields_fixed}\n"
                    )
                if fields_removed:
                    pbi_detail += (
                        f"- Visual fields REMOVED (no match in data): "
                        f"{fields_removed}\n"
                    )

                # Tableau migration details
                if pbi_bc.get("tableau_file"):
                    pbi_detail += (
                        f"\nTABLEAU TRANSLATION:\n"
                        f"- Source: {pbi_bc['tableau_file']}\n"
                        f"- Worksheets found: "
                        f"{', '.join(pbi_bc.get('tableau_worksheets', []))}\n"
                        f"- Calc fields found: "
                        f"{len(pbi_bc.get('tableau_calcs_found', []))}\n"
                        f"- Converted to DAX: "
                        f"{', '.join(pbi_bc.get('tableau_calcs_converted', [])) or 'none'}\n"
                        f"- Could not convert: "
                        f"{', '.join(pbi_bc.get('tableau_calcs_failed', [])) or 'none'}\n"
                    )

                if pbi_bc.get("relationships"):
                    pbi_detail += (
                        f"- Relationships: "
                        f"{', '.join(pbi_bc['relationships'])}\n"
                    )

                pbi_detail += (
                    "\nWrite a 3-4 sentence summary telling the user what "
                    "was built, what to check, and what might need manual "
                    "adjustment. Be specific about any fields removed or "
                    "measures skipped. If a Tableau workbook was provided, "
                    "mention how closely the translation matched."
                )
            else:
                pbi_detail = ""

            followup_context = (
                f"I just built the following deliverables for the user:\n"
                + "\n".join(detail_lines)
                + f"\nThe dataset has {row_count:,} rows and "
                f"{col_count} columns.{data_hint}"
                + pbi_detail
            )

            # Generate LLM content summary if PBI build context available
            content = ""
            if pbi_bc and self.client:
                try:
                    _progress("Generating build summary...")
                    summary_resp = self.client.messages.create(
                        model=self.MODEL,
                        max_tokens=500,
                        temperature=0.3,
                        system=DR_DATA_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": followup_context}],
                        timeout=30.0,
                    )
                    for block in summary_resp.content:
                        if hasattr(block, "text"):
                            content = block.text
                            break
                    self.trace.log_llm_call(
                        "claude", self.MODEL,
                        "PBI build summary narration",
                        content[:200],
                    )
                except Exception:
                    pass

            return {
                "type": "export",
                "acknowledgment": acknowledgment,
                "content": content,
                "downloads": downloads,
                "errors": errors,
                "progress_steps": progress_steps,
                "followup_context": followup_context,
            }

        else:
            # ---- Normal chat path (guaranteed if not export) ----
            enriched = self._build_context_message(user_message)

            # Route: use OpenAI for large-context or analysis-heavy requests,
            # but ONLY when no tool calling is needed.
            _tool_keywords = (
                "build", "power bi", "pbi", "pbip", "tableau", "alteryx",
                "parse", "migrate",
            )
            needs_tools = any(kw in msg_lower for kw in _tool_keywords)

            if not needs_tools and self.openai_client:
                estimated_tokens = len(enriched) // 4
                is_heavy = (
                    estimated_tokens > 5000
                    or any(kw in msg_lower for kw in self._HEAVY_KEYWORDS)
                )
                if is_heavy:
                    oai_text = self._call_openai(
                        DR_DATA_SYSTEM_PROMPT, enriched
                    )
                    if oai_text is not None:
                        return {
                            "type": "chat",
                            "content": oai_text,
                        }
                    # OpenAI failed -- fall through to Claude

            # Use Claude chat() with tool-calling loop
            result = self.chat(enriched, progress_callback=progress_callback)

            # Translate to workspace-friendly format
            chat_downloads = []
            for fpath in result.get("files", []):
                if os.path.exists(fpath):
                    fname = os.path.basename(fpath)
                    chat_downloads.append({
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
                "type": "chat",
                "content": result.get("text", ""),
                "downloads": chat_downloads,
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

        if self.snowflake_tables:
            sf_detail = ", ".join(
                f"{t} ({len(df)} rows x {len(df.columns)} cols)"
                for t, df in self.snowflake_tables.items()
            )
            context_parts.append(
                f"[SYSTEM: Data loaded from Snowflake warehouse. "
                f"Tables: {sf_detail}. This is live enterprise data "
                f"from SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 schema -- "
                f"contains customer, order, lineitem, supplier, part, "
                f"nation, and region tables for a simulated business "
                f"dataset. Proceed with analysis.]"
            )

        if context_parts:
            return "\n".join(context_parts) + "\n\n" + user_message
        return user_message

    def _narrate_export(self, downloads, errors):
        """Ask the LLM to narrate what was just built in Dr. Data's voice."""
        details = []
        for d in downloads:
            details.append(f"- {d['name']}: {d.get('description', '')}")
        if errors:
            for e in errors:
                details.append(f"- Failed: {e}")

        if not details:
            return (
                "Hmm, nothing came out of that build. "
                "Tell me what you need and I will try a different angle."
            )

        row_count = len(self.dataframe) if self.dataframe is not None else 0
        col_count = (
            len(self.dataframe.columns) if self.dataframe is not None else 0
        )

        data_hint = ""
        if self.data_profile:
            insights = self.data_profile.get("quick_insights", [])
            if insights:
                data_hint = (
                    "\nInteresting things in the data: "
                    + "; ".join(insights[:3])
                )

        detail_text = "\n".join(details)
        llm_prompt = (
            f"I just built the following deliverables for the user:\n"
            f"{detail_text}\n"
            f"The dataset has {row_count:,} rows and "
            f"{col_count} columns.{data_hint}\n\n"
            f"Write a short 1-2 sentence response confirming what you built. "
            f"Mention something specific and interesting you noticed about "
            f"the data while building it. Be natural and conversational, "
            f"not robotic."
        )

        try:
            resp = self.client.messages.create(
                model=self.MODEL,
                max_tokens=300,
                temperature=0.7,
                system=DR_DATA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": llm_prompt}],
                timeout=30.0,
            )
            text = resp.content[0].text.strip()
            if text:
                self.trace.log_llm_call(
                    "claude", self.MODEL,
                    "narrate_export", text[:200],
                )
                return text
        except Exception:
            pass

        # Fallback -- still natural, not robotic
        if downloads:
            fmt = downloads[0]["name"] if len(downloads) == 1 else "deliverables"
            fallback = (
                f"Done -- your {fmt} is in the downloads. "
                f"Take a look and tell me what you want to tweak."
            )
        else:
            fallback = (
                "Ran into some issues with the build. "
                "Tell me what you need and I will try again."
            )
        if errors:
            err_names = [e.split(":")[0] for e in errors]
            fallback += f" (Heads up: {', '.join(err_names)} had issues.)"
        return fallback

    def narrate_export_stream(self, followup_context):
        """Stream Dr. Data's response about what was built. Yields text chunks.

        Use with st.write_stream() in Streamlit for live typing effect.
        """
        prompt = (
            f"{followup_context}\n\n"
            "Write a short 1-2 sentence response confirming what you built. "
            "Mention something specific and interesting you noticed about "
            "the data while building it. Be natural and conversational, "
            "not robotic."
        )
        try:
            with self.client.messages.stream(
                model=self.MODEL,
                max_tokens=300,
                temperature=0.7,
                system=DR_DATA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                full_text = ""
                for text in stream.text_stream:
                    full_text += text
                    yield text
                if full_text:
                    self.trace.log_llm_call(
                        "claude", self.MODEL,
                        "narrate_export", full_text[:200],
                    )
        except Exception:
            yield (
                "Done -- your deliverables are ready. "
                "Take a look and tell me what you want to tweak."
            )

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
            from core.claude_interpreter import ClaudeInterpreter
            interpreter = ClaudeInterpreter()

            if self.tableau_spec:
                # --- TABLEAU MIGRATION PATH ---
                self._report_progress(
                    "Step 2/5: Claude is translating your Tableau "
                    "workbook to Power BI..."
                )
                translate_request = self._build_tableau_translate_request(
                    request, table_name
                )
                dashboard_spec = interpreter.interpret(
                    translate_request, pbi_profile
                )

                # Enrich spec with Tableau metadata for downstream stages
                dashboard_spec["source"] = "tableau_migration"
                dashboard_spec["tableau_worksheets"] = (
                    self.tableau_spec.get("worksheets", [])
                )
                dashboard_spec["tableau_dashboards"] = (
                    self.tableau_spec.get("dashboards", [])
                )
                dashboard_spec["tableau_calcs"] = (
                    self.tableau_spec.get("calculated_fields", [])
                )
            else:
                # --- FRESH DESIGN PATH (CSV / Excel) ---
                self._report_progress(
                    "Step 2/5: Claude is designing your dashboard "
                    "(pages, visuals, DAX measures)..."
                )
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

            # Collect relationships from all sources
            relationships = []
            if self.session and self.session.relationships:
                relationships.extend(self.session.relationships)
            if self.tableau_spec and self.tableau_spec.get("relationships"):
                relationships.extend(self.tableau_spec["relationships"])

            gen_result = generator.generate(
                config, pbi_profile, dashboard_spec,
                data_file_path=self.data_file_path,
                sheet_name=self.sheet_name,
                relationships=relationships or None,
            )
            # generator.generate() returns a dict with path + audit info
            result_path = gen_result["path"]
            field_audit = gen_result.get("field_audit", {})
            valid_measures = gen_result.get("valid_measures", [])

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

            # -- Collect detailed build context for summary --
            data_file_name = (os.path.basename(self.data_file_path)
                              if self.data_file_path else "unknown")
            row_count = len(self.dataframe) if self.dataframe is not None else 0
            col_count = (len(self.dataframe.columns)
                         if self.dataframe is not None else 0)

            # Page and visual details
            page_names = [p.get("title", p.get("name", f"Page {i+1}"))
                          for i, p in enumerate(dashboard_spec.get("pages", []))]
            visuals_per_page = [len(p.get("visuals", []))
                                for p in dashboard_spec.get("pages", [])]

            # Measures: all requested vs validated
            all_measures = [m.get("name", "?")
                            for m in dashboard_spec.get("measures", [])]
            skipped_measures = [m for m in all_measures
                                if m not in valid_measures]

            # Column breakdown
            numeric_count = sum(
                1 for c in pbi_profile.get("columns", [])
                if c.get("semantic_type") == "measure"
            )
            categorical_count = col_count - numeric_count

            # Tableau info (if migration)
            tableau_file_name = ""
            tableau_worksheets = []
            tableau_calcs_found = []
            tableau_calcs_converted = []
            tableau_calcs_failed = []
            if self.tableau_spec:
                tableau_file_name = self.tableau_spec.get(
                    "file_name", "Tableau workbook"
                )
                tableau_worksheets = [
                    w.get("name", "?")
                    for w in self.tableau_spec.get("worksheets", [])
                ]
                for cf in self.tableau_spec.get("calculated_fields", []):
                    cf_name = cf.get("name", "?")
                    tableau_calcs_found.append(cf_name)
                    if cf_name in valid_measures:
                        tableau_calcs_converted.append(cf_name)
                    else:
                        tableau_calcs_failed.append(cf_name)

            build_context = {
                "data_file": data_file_name,
                "row_count": row_count,
                "col_count": col_count,
                "numeric_count": numeric_count,
                "categorical_count": categorical_count,
                "table_name": table_name,
                "page_names": page_names,
                "visuals_per_page": visuals_per_page,
                "measures_created": list(valid_measures),
                "measures_skipped": skipped_measures,
                "fields_valid": field_audit.get("valid", 0),
                "fields_fixed": field_audit.get("fixed", 0),
                "fields_removed": field_audit.get("removed", 0),
                "relationships": [str(r) for r in (relationships or [])],
                "m_expression_source": data_file_name,
                "tableau_file": tableau_file_name,
                "tableau_worksheets": tableau_worksheets,
                "tableau_calcs_found": tableau_calcs_found,
                "tableau_calcs_converted": tableau_calcs_converted,
                "tableau_calcs_failed": tableau_calcs_failed,
            }

            # Log the full build context to trace logger
            self.trace.log("powerbi_build_summary", build_context)

            return json.dumps({
                "status": "success",
                "file_path": zip_path,
                "project_name": project_name,
                "pages": page_count,
                "visuals": visual_count,
                "measures": len(valid_measures),
                "build_context": build_context,
                "instructions": (
                    f"Extract the ZIP to a folder, then double-click "
                    f"Open_Dashboard.bat -- it will set up the data "
                    f"source and open the dashboard in Power BI Desktop "
                    f"automatically."
                ),
            })

        except Exception as e:
            self._report_progress(f"Power BI generation failed: {e}")
            return json.dumps({
                "error": f"Power BI generation failed: {e}",
                "suggestion": "Try building an HTML dashboard instead."
            })

    # ------------------------------------------------------------------ #
    #  Tableau -> Power BI translation helper                               #
    # ------------------------------------------------------------------ #

    # Tableau -> Power BI chart type mapping
    _TABLEAU_CHART_MAP = {
        "bar": "clusteredBarChart",
        "stacked-bar": "stackedBarChart",
        "line": "lineChart",
        "area": "areaChart",
        "map": "map",
        "filled-map": "filledMap",
        "text": "tableEx",
        "text-table": "tableEx",
        "crosstab": "matrix",
        "scatter": "scatterChart",
        "pie": "pieChart",
        "treemap": "treemap",
        "heatmap": "matrix",
        "highlight-table": "matrix",
        "dual-axis": "lineClusteredColumnComboChart",
        "combo": "lineClusteredColumnComboChart",
        "histogram": "clusteredBarChart",
        "box-plot": "clusteredBarChart",
        "gantt": "clusteredBarChart",
        "bullet": "clusteredBarChart",
        "waterfall": "waterfallChart",
        "funnel": "funnelChart",
        "donut": "donutChart",
    }

    # Tableau formula -> DAX mapping reference (included in prompt)
    _TABLEAU_DAX_MAP = (
        "Tableau formula syntax mapping:\n"
        "- SUM([Field]) -> SUM(TableName[Field])\n"
        "- AVG([Field]) -> AVERAGE(TableName[Field])\n"
        "- COUNTD([Field]) -> DISTINCTCOUNT(TableName[Field])\n"
        "- COUNT([Field]) -> COUNT(TableName[Field])\n"
        "- MIN([Field]) -> MIN(TableName[Field])\n"
        "- MAX([Field]) -> MAX(TableName[Field])\n"
        "- IF condition THEN x ELSE y END -> IF(condition, x, y)\n"
        "- DATETRUNC('month', [Date]) -> STARTOFMONTH(TableName[Date])\n"
        "- DATEDIFF('unit', [start], [end]) -> "
        "DATEDIFF(TableName[start], TableName[end], unit)\n"
        "- ZN([Field]) -> IF(ISBLANK(TableName[Field]), 0, "
        "TableName[Field])\n"
        "- CONTAINS([String], 'text') -> "
        "CONTAINSSTRING(TableName[String], \"text\")\n"
        "- LEFT/RIGHT/MID -> LEFT/RIGHT/MID (same in DAX)\n"
        "- ATTR([Field]) -> SELECTEDVALUE(TableName[Field])\n"
        "- WINDOW_SUM/AVG/COUNT -> use DAX window functions\n"
        "- RUNNING_SUM -> use CALCULATE with FILTER\n"
        "Keep the same measure names as the Tableau calculated fields."
    )

    def _build_tableau_translate_request(self, user_request, table_name):
        """Build a Tableau-to-Power BI translation prompt for ClaudeInterpreter.

        Instead of asking Claude to design from scratch, we give it the
        full Tableau structure and ask it to RECREATE the same layout
        in Power BI format.
        """
        spec = self.tableau_spec
        parts = []

        parts.append(
            "The user uploaded a Tableau workbook. Your job is to "
            "TRANSLATE this into an equivalent Power BI dashboard -- "
            "do NOT redesign from scratch. Preserve the structure, "
            "layout, and intent of the original."
        )

        # -- Worksheets --
        worksheets = spec.get("worksheets", [])
        if worksheets:
            ws_lines = []
            for ws in worksheets:
                name = ws.get("name", "Untitled")
                chart_type = ws.get("type", ws.get("chart_type", "unknown"))
                pbi_type = self._TABLEAU_CHART_MAP.get(
                    chart_type.lower().replace(" ", "-"),
                    "clusteredBarChart",
                )
                rows = ws.get("rows", [])
                cols = ws.get("columns", [])
                marks = ws.get("marks", ws.get("mark_type", ""))
                filters = ws.get("filters", [])

                line = f'  - "{name}": {chart_type} -> PBI {pbi_type}'
                if rows:
                    line += f", rows={rows}"
                if cols:
                    line += f", columns={cols}"
                if marks:
                    line += f", marks={marks}"
                if filters:
                    line += f", filters={filters}"
                ws_lines.append(line)
            parts.append(
                "TABLEAU WORKSHEETS:\n" + "\n".join(ws_lines)
            )

        # -- Dashboards --
        dashboards = spec.get("dashboards", [])
        if dashboards:
            db_lines = []
            for db in dashboards:
                name = db.get("name", "Dashboard")
                sheets = db.get("worksheets", db.get("sheets", []))
                size = db.get("size", {})
                w = size.get("width", size.get("maxwidth", ""))
                h = size.get("height", size.get("maxheight", ""))

                line = f'  - "{name}": contains {sheets}'
                if w and h:
                    line += f", size={w}x{h}"
                    # Proportional mapping to PBI 1280x720
                    try:
                        tw = int(str(w).replace("px", ""))
                        th = int(str(h).replace("px", ""))
                        if tw > 0 and th > 0:
                            line += (
                                f" (map proportionally to PBI "
                                f"1280x720 canvas)"
                            )
                    except (ValueError, TypeError):
                        pass

                # Include layout positions if available
                zones = db.get("zones", db.get("layout", []))
                if zones:
                    line += f", layout_zones={len(zones)} zones"

                db_lines.append(line)
            parts.append(
                "TABLEAU DASHBOARDS:\n" + "\n".join(db_lines)
            )

        # -- Calculated fields --
        calc_fields = spec.get("calculated_fields", [])
        if calc_fields:
            calc_lines = []
            for cf in calc_fields:
                if isinstance(cf, dict):
                    name = cf.get("name", "?")
                    formula = cf.get("formula", "?")
                    calc_lines.append(f'  - "{name}": {formula}')
                elif isinstance(cf, str):
                    calc_lines.append(f"  - {cf}")
            parts.append(
                "TABLEAU CALCULATED FIELDS (convert to DAX):\n"
                + "\n".join(calc_lines[:30])
            )
            parts.append(
                self._TABLEAU_DAX_MAP.replace("TableName", table_name)
            )

        # -- Parameters --
        parameters = spec.get("parameters", [])
        if parameters:
            param_lines = []
            for p in parameters:
                if isinstance(p, dict):
                    name = p.get("name", "?")
                    dtype = p.get("datatype", p.get("type", "?"))
                    value = p.get("value", p.get("current_value", ""))
                    param_lines.append(
                        f'  - "{name}": {dtype}, value={value}'
                    )
            if param_lines:
                parts.append(
                    "TABLEAU PARAMETERS (convert to What-If "
                    "parameters or slicers):\n"
                    + "\n".join(param_lines)
                )

        # -- Filters --
        filters = spec.get("filters", [])
        if filters:
            filter_lines = []
            for f in filters:
                if isinstance(f, dict):
                    col = f.get("column", f.get("field", "?"))
                    ftype = f.get("type", "?")
                    filter_lines.append(f"  - {col}: {ftype}")
                elif isinstance(f, str):
                    filter_lines.append(f"  - {f}")
            if filter_lines:
                parts.append(
                    "TABLEAU FILTERS:\n"
                    + "\n".join(filter_lines[:20])
                )

        # -- Chart type mapping reference --
        parts.append(
            "CHART TYPE MAPPING:\n"
            "  Tableau bar -> PBI clusteredBarChart\n"
            "  Tableau stacked bar -> PBI stackedBarChart\n"
            "  Tableau line -> PBI lineChart\n"
            "  Tableau area -> PBI areaChart\n"
            "  Tableau map -> PBI map\n"
            "  Tableau text table -> PBI tableEx\n"
            "  Tableau scatter -> PBI scatterChart\n"
            "  Tableau pie -> PBI pieChart\n"
            "  Tableau treemap -> PBI treemap\n"
            "  Tableau heatmap -> PBI matrix with conditional formatting\n"
            "  Tableau dual-axis -> PBI lineClusteredColumnComboChart\n"
            "  Tableau donut -> PBI donutChart\n"
            "  Tableau waterfall -> PBI waterfallChart"
        )

        # -- Original user request --
        if user_request and "power bi" not in user_request.lower():
            parts.append(
                f"ADDITIONAL USER REQUEST: {user_request}"
            )

        return "\n\n".join(parts)

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
