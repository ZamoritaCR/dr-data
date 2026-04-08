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
from core.deliverable_registry import save_deliverable as _save_deliverable
from core.deliverable_registry import search as _search_deliverables
from core.audit_engine import AuditEngine as _AuditEngine
from core.dashboard_rationalization import DashboardRationalizationEngine

# ------------------------------------------------------------------ #
#  Vision extraction for uploaded images                                #
# ------------------------------------------------------------------ #

_VISION_PROMPT = (
    "Analyze this dashboard/report screenshot in detail. Extract:\n"
    "1. Every chart type visible (bar, line, pie, map, KPI card, table, etc.)\n"
    "2. Every field name, metric, KPI, and dimension visible\n"
    "3. All filter/slicer controls and their values\n"
    "4. Page layout structure (how visuals are arranged)\n"
    "5. Color palette (list hex codes if detectable)\n"
    "6. Titles, subtitles, and any visible text\n"
    "7. Number of pages/tabs if indicated\n"
    "8. Data source hints (table names, connection info)\n\n"
    "Return a structured analysis. Be exhaustive -- do not skip any visible element."
)


def _extract_image_with_vision(image_structure):
    """Run AI vision on an uploaded image. Gemini primary, Claude fallback."""
    b64 = image_structure.get("base64", "")
    media_type = image_structure.get("media_type", "image/png")
    if not b64:
        return ""

    # Try Gemini 2.5 Pro first
    gemini_key = _get_secret("GEMINI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            import base64 as b64_mod
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")
            image_bytes = b64_mod.b64decode(b64)
            response = model.generate_content([
                _VISION_PROMPT,
                {"mime_type": media_type, "data": image_bytes},
            ])
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[VISION] Gemini failed: {e}")

    # Fallback: Claude Opus vision
    anthropic_key = _get_secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        }},
                        {"type": "text", "text": _VISION_PROMPT},
                    ],
                }],
            )
            if resp and resp.content:
                return resp.content[0].text
        except Exception as e:
            print(f"[VISION] Claude fallback failed: {e}")

    # Fallback: GPT-4o vision
    if _HAS_OPENAI:
        openai_key = _get_secret("OPENAI_API_KEY")
        if openai_key:
            try:
                client = _OpenAIClient(api_key=openai_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=4096,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                            }},
                            {"type": "text", "text": _VISION_PROMPT},
                        ],
                    }],
                )
                if resp and resp.choices:
                    return resp.choices[0].message.content
            except Exception as e:
                print(f"[VISION] GPT-4o fallback failed: {e}")

    return ""


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
    },
    {
        "name": "rationalize_dashboards",
        "description": (
            "Run a dashboard rationalization analysis on the enterprise BI portfolio. "
            "Identifies zombie dashboards, duplicates, refresh waste, and hidden costs. "
            "Can generate sample data or analyze an uploaded inventory. "
            "Returns executive summary with recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Action: 'generate_sample' to create 10K sample dashboards, "
                        "'analyze' to run full analysis, 'report' to generate HTML report"
                    ),
                    "enum": ["generate_sample", "analyze", "report"]
                }
            },
            "required": ["action"]
        }
    }
]


# ------------------------------------------------------------------ #
#  Dr. Data Agent                                                      #
# ------------------------------------------------------------------ #

class DrDataAgent:
    """Conversational Claude agent with tool-calling for data analysis."""

    MODEL = "claude-opus-4-20250514"
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
        self._dashboard_iteration = 0     # Increments each build for variation

        # Multi-file session support
        self.session = None
        self.session_bridge = None

        # Snowflake data
        self.snowflake_tables = {}
        self.snowflake_config = None

        # Data quality scan results (set via set_dq_results)
        self.dq_scan_results = None
        self.trust_scores = None
        self.copdq_result = None
        self.compliance_summary = None
        self.stewardship_stats = None
        self.incident_stats = None

        # Dashboard rationalization engine
        self.rationalization_engine = DashboardRationalizationEngine()

        # Trace logger for audit trail
        self.trace = TraceLogger()

        # OpenAI secondary engine (for large-context / data-heavy tasks)
        self.openai_client = None
        openai_key = os.getenv("OPENAI_API_KEY") or _get_secret("OPENAI_API_KEY")
        if openai_key and _HAS_OPENAI:
            self.openai_client = _OpenAIClient(api_key=openai_key)
        elif not openai_key:
            print("[INFO] OPENAI_API_KEY not set -- using Claude for all requests.")

        # Gemini tertiary engine (large-context analysis, vision, failover)
        self.gemini_engine = None
        try:
            from core.gemini_engine import GeminiEngine
            self.gemini_engine = GeminiEngine()
        except Exception as e:
            print(f"[INFO] Gemini engine not loaded: {e}")

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
                        f"Hit a snag reaching my analysis engine. "
                        f"Error: {last_err}. Give me a second and try again."
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

    def chat_stream(self, user_message, progress_callback=None):
        """Streaming version of chat() -- yields text chunks as Claude generates.

        Use with st.write_stream() in Streamlit for a live typing effect.
        Tool-calling rounds run synchronously; the final text response streams.
        """
        self.messages.append({"role": "user", "content": user_message})
        self.generated_files = []
        self._progress_callback = progress_callback
        self._fix_orphaned_tool_calls()

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

        max_iterations = 10
        for iteration in range(max_iterations):

            # Retry loop
            response = None
            last_err = None
            for attempt in range(3):
                try:
                    with self.client.messages.stream(
                        model=self.MODEL,
                        max_tokens=self.MAX_TOKENS,
                        temperature=0,
                        system=DR_DATA_SYSTEM_PROMPT,
                        tools=TOOLS,
                        messages=self.messages,
                        timeout=120.0,
                    ) as stream:
                        for text in stream.text_stream:
                            yield text
                        response = stream.get_final_message()
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
                if self.messages and self.messages[-1]["role"] == "user":
                    self.messages.pop()
                yield (
                    f"\n\nHit a snag reaching the analysis engine. "
                    f"Error: {last_err}. Give me a second and try again."
                )
                return

            # Store assistant message
            content_dicts = []
            for block in response.content:
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

            # Trace
            _resp_summary = ""
            for b in response.content:
                if hasattr(b, "text") and b.text:
                    _resp_summary = b.text[:200]
                    break
                if hasattr(b, "name"):
                    _resp_summary = f"tool_use:{b.name}"
                    break
            self.trace.log_llm_call(
                "claude", self.MODEL,
                user_message[:200] if isinstance(user_message, str) else "stream",
                _resp_summary,
            )

            # Handle tool calls (blocking -- cannot stream tool execution)
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        label = tool_labels.get(block.name, f"Running: {block.name}")
                        if progress_callback:
                            progress_callback(label)
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                self.messages.append({"role": "user", "content": tool_results})
                continue  # next iteration streams the tool-result response

            # end_turn -- all text has already been yielded
            return

        # Safety net
        yield "\n\nReached maximum processing steps. Try rephrasing your request."

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

        # Run AI vision extraction on any uploaded images
        for fname, finfo in session.files.items():
            structure = finfo.get("structure")
            if (structure and isinstance(structure, dict)
                    and structure.get("type") == "image"
                    and structure.get("base64")
                    and "vision_analysis" not in structure):
                vision_text = _extract_image_with_vision(structure)
                if vision_text:
                    structure["vision_analysis"] = vision_text

        # Get primary data file path and sheet name.
        # IMPORTANT: if the source is a .twbx/.twb, the path points to the
        # Tableau archive, which is NOT a valid data file for Power BI.
        # In that case, export the DataFrame to CSV so the M expression
        # references a real CSV file that Power BI Desktop can load.
        for fname, info in session.files.items():
            if info.get("df") is not None:
                raw_path = info["path"]
                raw_ext = raw_path.rsplit(".", 1)[-1].lower() if "." in raw_path else ""
                if raw_ext in ("twbx", "twb"):
                    # Tableau archive -- save embedded data to CSV
                    csv_dir = str(PROJECT_ROOT / "output")
                    os.makedirs(csv_dir, exist_ok=True)
                    csv_name = os.path.splitext(os.path.basename(raw_path))[0]
                    csv_name = csv_name.replace(" ", "_") + "_data.csv"
                    csv_path = os.path.join(csv_dir, csv_name)
                    info["df"].to_csv(csv_path, index=False)
                    self.data_file_path = csv_path
                    self.data_path = csv_path
                    print(f"[SESSION] Exported .twbx embedded data to {csv_path}")
                else:
                    self.data_file_path = raw_path
                    self.data_path = raw_path
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

    _CSV_ENDPOINT = "https://joao.theartofthepossible.io/drdata-csv"
    _CSV_NAME_MAP = {
        "AF_Berliana": "AF_Berliana",
        "Berliana": "AF_Berliana",
        "COVID19": "COVID19",
        "COVID-19": "COVID19",
        "Coronavirus": "COVID19",
        "DigitalAds": "DigitalAds",
        "Digital": "DigitalAds",
        "Superstore": "Superstore",
    }

    def _get_csv_url(self):
        """Derive the live CSV URL from the uploaded TWBX filename.

        Returns None if the source is not a known TWBX, so the generator
        falls back to the local-file M expression.
        """
        source = ""
        if self.tableau_spec and self.tableau_spec.get("file_name"):
            source = self.tableau_spec["file_name"]
        elif self.data_file_path:
            source = os.path.basename(self.data_file_path)
        if not source:
            return None
        stem = os.path.splitext(os.path.basename(source))[0]
        for key, csv_name in self._CSV_NAME_MAP.items():
            if key.lower() in stem.lower():
                return f"{self._CSV_ENDPOINT}/{csv_name}.csv"
        return None

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

            # Wire field resolution: if we have both a Tableau spec and data,
            # resolve field names against actual columns using fuzzy matching.
            if self.dataframe is not None and self.tableau_spec.get("type", "").startswith("tableau"):
                try:
                    from core.field_resolver import TableauFieldResolver
                    resolver = TableauFieldResolver()
                    col_names = list(self.dataframe.columns)
                    resolution_map = {}
                    for ws in self.tableau_spec.get("worksheets", []):
                        for field in ws.get("rows_fields", []) + ws.get("cols_fields", []) + ws.get("dimensions", []) + ws.get("measures", []):
                            if field and field not in resolution_map:
                                match = resolver.resolve_field(field, col_names)
                                if match:
                                    resolution_map[field] = match["matched"]
                    for cf in self.tableau_spec.get("calculated_fields", []):
                        fname = cf.get("name", "")
                        if fname and fname not in resolution_map:
                            match = resolver.resolve_field(fname, col_names)
                            if match:
                                resolution_map[fname] = match["matched"]
                    self.tableau_spec["field_resolution_map"] = resolution_map
                    print(f"[FIELD RESOLVER] Resolved {len(resolution_map)}/{len(resolution_map)} fields via fuzzy matching")
                except ImportError:
                    print("[FIELD RESOLVER] rapidfuzz not installed -- skipping field resolution")
                except Exception as e:
                    print(f"[FIELD RESOLVER] Failed: {e}")

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

        # Include extracted content from images, documents, and BI files
        for fname, finfo in session.files.items():
            structure = finfo.get("structure")
            if not structure or not isinstance(structure, dict):
                continue
            stype = structure.get("type", "")
            if stype == "image":
                vision = structure.get("vision_analysis", "")
                if vision:
                    extra_parts.append(
                        f"Image uploaded: {fname} ({structure.get('size_mb', '?')} MB). "
                        f"AI vision analysis:\n{vision}"
                    )
                else:
                    extra_parts.append(
                        f"Image uploaded: {fname} ({structure.get('size_mb', '?')} MB). "
                        "Vision extraction pending -- image content available for analysis."
                    )
            elif stype == "document":
                text = structure.get("text", "")
                if text and text.strip():
                    preview = text[:2000]
                    extra_parts.append(
                        f"Document uploaded: {fname} ({structure.get('format', '?')}). "
                        f"Extracted text:\n{preview}"
                    )
            elif stype == "power_bi":
                tables = structure.get("tables", [])
                if tables:
                    tbl_desc = "; ".join(
                        f"{t['name']} ({len(t.get('columns', []))} cols, "
                        f"{len(t.get('measures', []))} measures)"
                        for t in tables[:10]
                    )
                    extra_parts.append(
                        f"Power BI file uploaded: {fname}. Tables: {tbl_desc}"
                    )
            elif stype == "bi_file":
                note = structure.get("note", "")
                if note:
                    extra_parts.append(f"BI file uploaded: {fname}. {note}")
            elif stype == "tableau_datasource":
                cols = structure.get("columns", [])
                conn = structure.get("connection_type", "")
                extra_parts.append(
                    f"Tableau datasource: {fname} ({conn}), {len(cols)} columns."
                )

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
                            df[col], errors="raise"
                        )
                        # don't mutate original df; use parsed locally
                        date_cols.append(col)
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
        self._progress_callback = progress_callback

        # Auto-detect uploaded files the agent doesn't know about yet
        if uploaded_files and self.dataframe is None:
            for name, info in uploaded_files.items():
                path = info.get("path", "")
                if not path or not os.path.exists(path):
                    continue
                ext = info.get("ext", path.rsplit(".", 1)[-1].lower())
                # Skip structure-only files -- look for actual data files
                if ext in ("twb", "twbx", "wid", "yxmd", "yxwz", "yxmc", "yxzp"):
                    continue
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
                    if self.dataframe is not None:
                        self.data_file_path = path
                        self.data_path = path
                        if self.data_profile is None:
                            self.data_profile = self.analyzer.profile(self.dataframe)
                        break
                except Exception:
                    pass

        # --- LLM-powered intent classification ---
        from core.intent_classifier import (
            classify as _classify_intent,
            BUILD_DASHBOARD, BUILD_POWERBI, BUILD_PDF, BUILD_PPTX,
            BUILD_WORD, BUILD_ALL, BUILD_INTENTS,
        )

        # Build conversation context from recent messages
        _recent_ctx = ""
        if self.messages:
            _ctx_msgs = self.messages[-4:]
            _ctx_lines = []
            for _m in _ctx_msgs:
                role = _m.get("role", "?")
                content = ""
                if isinstance(_m.get("content"), str):
                    content = _m["content"][:150]
                elif isinstance(_m.get("content"), list):
                    for _b in _m["content"]:
                        if isinstance(_b, dict) and _b.get("type") == "text":
                            content = _b.get("text", "")[:150]
                            break
                _ctx_lines.append(f"{role}: {content}")
            _recent_ctx = "\n".join(_ctx_lines)

        # Build file context for intent classifier
        _file_context = {
            "source_type": "tableau" if self.tableau_spec else (
                "image" if (self.tableau_spec and
                            self.tableau_spec.get("source") == "vision_screenshot")
                else "data" if self.dataframe is not None else "none"
            ),
            "file_name": os.path.basename(self.data_file_path or ""),
            "has_tableau_spec": bool(self.tableau_spec),
            "has_dataframe": self.dataframe is not None,
            "worksheet_count": len(self.tableau_spec.get("worksheets", []))
                if self.tableau_spec else 0,
            "dashboard_count": len(self.tableau_spec.get("dashboards", []))
                if self.tableau_spec else 0,
        }

        _intent = _classify_intent(user_message, _recent_ctx, _file_context)
        _intent_type = _intent["intent"]
        _intent_mode = _intent.get("mode", "creative")

        # Propagate mode to session state for downstream routing
        # (replicate vs creative for Tableau/vision specs)
        try:
            import streamlit as _st_mode
            if _intent_mode == "replicate":
                _st_mode.session_state["user_intent"] = "replicate"
            elif _intent_mode == "creative" and _intent_type in (BUILD_DASHBOARD,):
                _st_mode.session_state["user_intent"] = "reimagine"
        except Exception:
            pass

        print(f"[INTENT] {_intent_type} mode={_intent_mode} "
              f"conf={_intent.get('confidence', '?')} "
              f"via={_intent.get('details', '?')}")

        want_pbi = _intent_type == BUILD_POWERBI
        want_dash = _intent_type == BUILD_DASHBOARD
        want_pptx = _intent_type == BUILD_PPTX
        want_pdf = _intent_type == BUILD_PDF
        want_docx = _intent_type == BUILD_WORD
        want_all = _intent_type == BUILD_ALL

        is_export = _intent_type in BUILD_INTENTS

        print(f"[EXPORT CHECK] Message: {user_message[:80]}")
        print(f"[EXPORT CHECK] wants_pbi={want_pbi}, wants_dashboard={want_dash}, "
              f"wants_pptx={want_pptx}, wants_pdf={want_pdf}, wants_docx={want_docx}")
        print(f"[EXPORT CHECK] DataFrame available: {self.dataframe is not None}, "
              f"shape: {self.dataframe.shape if self.dataframe is not None else 'None'}")
        print(f'[ROUTE] Message: {user_message[:50]}... -> {"export" if is_export else "chat"}')

        # --- Check deliverable registry for similar past work ---
        _past_work_context = ""
        try:
            _similar = _search_deliverables(user_message)
            if _similar:
                _past_lines = []
                for _item in _similar[:3]:
                    _past_lines.append(
                        f"- {_item['type'].upper()}: {_item['name']} "
                        f"from {_item['source_file']} "
                        f"({_item['created_at'][:10]}). "
                        f"{_item['description']}"
                    )
                _past_work_context = (
                    "I have built similar deliverables before:\n"
                    + "\n".join(_past_lines)
                    + "\nIf any of these are relevant, suggest reusing "
                    "or building on them. Let the user decide. "
                    "Do not auto-reuse without asking."
                )
        except Exception:
            pass

        if is_export:
            # Try to recover data before checking
            if self.dataframe is None:
                self._try_recover_data()

            print(f"[EXPORT] is_export=True. dataframe={self.dataframe is not None}, "
                  f"tableau_spec={bool(self.tableau_spec)}, "
                  f"session={bool(self.session)}")
            # Guard: need data first -- but auto-generate if we have Tableau structure
            if self.dataframe is None and self.tableau_spec:
                self._report_progress(
                    "No extractable data found in the Tableau file "
                    "(uses .hyper format). Generating synthetic data "
                    "from the workbook structure..."
                )
                try:
                    from core.synthetic_data import generate_from_tableau_spec
                    ws_count = len(self.tableau_spec.get("worksheets", []))
                    cf_count = len(self.tableau_spec.get("calculated_fields", []))
                    self._report_progress(
                        f"Reading Tableau structure: {ws_count} worksheets, "
                        f"{cf_count} calculated fields"
                    )

                    output_dir = str(PROJECT_ROOT / "output")
                    df, csv_path, schema = generate_from_tableau_spec(
                        self.tableau_spec,
                        num_rows=2000,
                        output_dir=output_dir,
                    )
                    self.dataframe = df
                    self.data_file_path = csv_path
                    self.data_path = csv_path
                    # Profile immediately so context builder sees it
                    if self.data_profile is None:
                        self.data_profile = self.analyzer.profile(df)

                    self._report_progress(
                        f"Synthetic data generated: {len(df)} rows x "
                        f"{len(df.columns)} columns ({len(schema)} fields "
                        f"inferred from Tableau structure). "
                        f"Saved to {os.path.basename(csv_path)}"
                    )
                except Exception as synth_err:
                    print(f"[SYNTH] Failed to generate synthetic data: {synth_err}")
                    import traceback
                    traceback.print_exc()
                    self._report_progress(
                        f"Synthetic data generation failed: {synth_err}. "
                        "Upload a CSV or Excel file with your data to proceed."
                    )

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

            # Title priority: Tableau workbook name > original file name > default
            title = "Dr. Data Report"
            if self.tableau_spec:
                # Use Tableau workbook/dashboard name
                _tb_name = ""
                for _db in self.tableau_spec.get("dashboards", []):
                    _tb_name = _db.get("name", "")
                    if _tb_name:
                        break
                if not _tb_name:
                    _tb_name = self.tableau_spec.get("file_name", "")
                if _tb_name:
                    title = _tb_name.replace(".twbx", "").replace(".twb", "").strip()
            if title == "Dr. Data Report" and self.data_file_path:
                base = os.path.splitext(
                    os.path.basename(self.data_file_path)
                )[0]
                # Don't use "synthetic_tableau_data" as title
                if "synthetic" not in base.lower():
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
                self._dashboard_iteration += 1

                # Check if this is a vision-sourced spec that should be
                # replicated deterministically instead of AI-generated
                _is_vision_replica = (
                    self.tableau_spec
                    and self.tableau_spec.get("source") == "vision_screenshot"
                    and self.tableau_spec.get("worksheets")
                )

                if _is_vision_replica:
                    _progress(
                        f"Building replica dashboard from screenshot "
                        f"({len(self.tableau_spec.get('worksheets', []))} visuals)..."
                    )
                    try:
                        from core.vision_html_builder import build_replica_html
                        p = build_replica_html(
                            self.tableau_spec,
                            output_dir=self.output_dir,
                            title=title,
                        )
                        if p:
                            fname = os.path.basename(p)
                            downloads.append({
                                "name": "Dashboard Replica",
                                "filename": fname,
                                "path": p,
                                "description": (
                                    "Screenshot replica -- matching layout, "
                                    "chart types, and colors"
                                ),
                            })
                            self.trace.log_deliverable(
                                "Dashboard Replica", p,
                                {"title": title, "type": "html"},
                            )
                            _progress("  Replica dashboard complete.")
                    except Exception as e:
                        print('[BUILD] Vision replica FAILED, falling back to freeform:')
                        _tb.print_exc()
                        _progress(f"  Replica failed ({str(e)[:60]}), trying AI generation...")
                        _is_vision_replica = False

                if not _is_vision_replica:
                    _progress(
                        f"Claude is creating your dashboard from scratch "
                        f"(version {self._dashboard_iteration})..."
                    )

                try:
                    if not _is_vision_replica:
                        p = self._generate_freeform_dashboard(
                            user_message, title, _progress
                        )
                        if p:
                            fname = os.path.basename(p)
                            downloads.append({
                                "name": "Interactive Dashboard",
                                "filename": fname,
                                "path": p,
                                "description": (
                                    "AI-generated interactive dashboard -- "
                                    "unique design, charts, and insights"
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

            # -- Save each deliverable to the persistent registry --
            _src_file = getattr(self, "current_file_name", "") or ""
            _cols_used = (
                self.dataframe.columns.tolist()
                if self.dataframe is not None else []
            )
            _type_map = {
                "Interactive Dashboard": "dashboard",
                "PowerPoint Presentation": "pptx",
                "PDF Report": "pdf",
                "Word Document": "docx",
                "Power BI Project": "powerbi",
            }
            for _dl in downloads:
                try:
                    _save_deliverable({
                        "type": _type_map.get(_dl["name"], "other"),
                        "name": title,
                        "source_file": _src_file,
                        "columns_used": _cols_used,
                        "row_count": row_count,
                        "file_path": _dl.get("path", ""),
                        "description": _dl.get("description", ""),
                        "insights_found": "",
                    })
                except Exception:
                    pass

            # -- Generate QA audit report and add to downloads --
            if downloads and self.dataframe is not None:
                _progress("Running quality audit...")
                try:
                    _auditor = _AuditEngine()
                    _sub_reports = []

                    # 1. Audit the source data
                    _data_report = _auditor.audit_dataframe(
                        self.dataframe,
                        source_name=getattr(self, "current_file_name", "data"),
                    )
                    _sub_reports.append(_data_report)

                    # 2. Audit each generated deliverable file
                    for _dl in downloads:
                        _fpath = _dl.get("path", "")
                        if _fpath and os.path.exists(_fpath):
                            _ext = os.path.splitext(_fpath)[1].lower()
                            _ftype = "html" if _ext == ".html" else "binary"
                            _sub_reports.append(
                                _auditor.audit_deliverable(_fpath, _ftype)
                            )

                    # 3. Cross-validate key metrics against pandas ground truth
                    _cv_report = _auditor.cross_validate(
                        self.dataframe,
                        source_name=getattr(self, "current_file_name", "data"),
                    )
                    if _cv_report.findings:
                        _sub_reports.append(_cv_report)

                    # 4. Combine into single report
                    _combined = _AuditEngine.combine_reports(
                        _sub_reports,
                        title=f"Quality Audit -- {title}",
                    )
                    _combined.compute_scores()

                    # 5. Write standalone HTML with engine activity
                    _audit_path = os.path.join(
                        self.output_dir, "quality_audit.html"
                    )
                    _engine_summary = self.trace.get_engine_summary()
                    with open(_audit_path, "w", encoding="utf-8") as _af:
                        _af.write(_combined.to_standalone_html(
                            title=f"Quality Audit -- {title}",
                            engine_summary=_engine_summary,
                        ))

                    downloads.append({
                        "name": "Quality Audit Report",
                        "filename": "quality_audit.html",
                        "path": _audit_path,
                        "description": (
                            f"Score {_combined.overall_score}/100 -- "
                            f"{_combined.passed} passed, "
                            f"{_combined.warnings} warnings, "
                            f"{_combined.critical + _combined.blockers} critical. "
                            f"{_combined.release_decision}"
                        ),
                    })
                    _progress(
                        f"  Quality score: {_combined.overall_score}/100 "
                        f"({_combined.release_decision})"
                    )
                except Exception:
                    pass

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
            if _past_work_context:
                followup_context += f"\n\n{_past_work_context}"

            # Generate Claude narration for every build
            content = ""
            if self.client and (downloads or errors):
                try:
                    _narration_prompt = (
                        f"The user asked: \"{user_message}\"\n\n"
                        f"You just built these deliverables:\n"
                        + "\n".join(detail_lines)
                        + f"\n\nDataset: {row_count:,} rows, {col_count} columns."
                        + (f"\n{data_hint}" if data_hint else "")
                        + (f"\n\n{pbi_detail}" if pbi_detail else "")
                        + "\n\nNow respond AS DR. DATA. Tell the user:"
                        + "\n- What you built and what makes it interesting"
                        + "\n- One specific insight you noticed in the data while building"
                        + "\n- What you would do next if they want more"
                        + "\nBe direct, sharp, and opinionated. No bullet lists. "
                        + "No generic filler. Talk like a senior analyst sharing findings "
                        + "with a colleague. 2-3 short paragraphs max."
                    )
                    if errors:
                        _narration_prompt += (
                            f"\n\nSome builds had issues: {'; '.join(errors[:3])}"
                        )

                    _progress("Dr. Data is reviewing the results...")
                    summary_resp = self.client.messages.create(
                        model=self.MODEL,
                        max_tokens=600,
                        temperature=0.5,
                        system=DR_DATA_SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": _narration_prompt}],
                        timeout=45.0,
                    )
                    for block in summary_resp.content:
                        if hasattr(block, "text"):
                            content = block.text
                            break
                    self.trace.log_llm_call(
                        "claude", self.MODEL,
                        "Build narration",
                        content[:200],
                    )
                except Exception as _narr_err:
                    print(f"[NARRATE] Failed: {_narr_err}")

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
            _msg_lower = user_message.lower()

            # Route: use OpenAI for large-context or analysis-heavy requests,
            # but ONLY when no tool calling is needed.
            _tool_keywords = (
                "build", "power bi", "pbi", "pbip", "tableau", "alteryx",
                "parse", "migrate",
            )
            needs_tools = any(kw in _msg_lower for kw in _tool_keywords)

            if not needs_tools and self.openai_client:
                estimated_tokens = len(enriched) // 4
                is_heavy = (
                    estimated_tokens > 5000
                    or any(kw in _msg_lower for kw in self._HEAVY_KEYWORDS)
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

    def set_dq_results(self, scan_results):
        """Accept DQ scan results so they flow into conversational context."""
        self.dq_scan_results = scan_results

    def set_trust_scores(self, trust_scores):
        self.trust_scores = trust_scores

    def set_copdq_result(self, copdq_result):
        self.copdq_result = copdq_result

    def set_compliance_summary(self, compliance_summary):
        self.compliance_summary = compliance_summary

    def set_stewardship_stats(self, stewardship_stats):
        self.stewardship_stats = stewardship_stats

    def set_incident_stats(self, incident_stats):
        self.incident_stats = incident_stats

    def _build_dq_context(self):
        """Build a context snippet summarising all DQ module results."""
        if not self.dq_scan_results:
            return ""
        parts = ["\n\nDATA QUALITY CONTEXT:"]
        for tname, result in self.dq_scan_results.items():
            score = result.get("overall_score", 0)
            parts.append(f"Table {tname}: DQ Score {score:.1f}/100")
            dims = result.get("dimensions", {})
            for dim_name, dim_data in dims.items():
                if isinstance(dim_data, dict) and dim_data.get("score") is not None:
                    parts.append(f"  {dim_name}: {dim_data['score']:.1f}%")
            recs = result.get("recommendations", [])
            critical = [r for r in recs if r.get("priority") == "CRITICAL"]
            if critical:
                parts.append(f"  CRITICAL ISSUES: {len(critical)}")
                for c in critical[:3]:
                    parts.append(f"    - {c.get('finding', '')[:80]}")

        if self.trust_scores:
            parts.append("\nTRUST SCORES:")
            for tname, tscore in self.trust_scores.items():
                parts.append(
                    f"  {tname}: {tscore.get('trust_score', 0):.1f}/100 "
                    f"({tscore.get('recommended_certification', 'Uncertified')})")

        if self.copdq_result:
            parts.append(
                f"\nCOST OF POOR DATA QUALITY: "
                f"${self.copdq_result.get('total_annual_cost', 0):,.0f}/year")

        if self.compliance_summary:
            parts.append(
                f"\nCOMPLIANCE: Overall "
                f"{self.compliance_summary.get('overall_score', 0):.1f}%")
            gaps = self.compliance_summary.get("critical_gaps", [])
            if gaps:
                parts.append(f"  Critical gaps: {len(gaps)}")

        if self.stewardship_stats:
            parts.append(
                f"\nOPEN DQ ISSUES: "
                f"{self.stewardship_stats.get('total_open', 0)}")
            parts.append(
                f"SLA BREACHES: "
                f"{self.stewardship_stats.get('sla_breaches', 0)}")

        if self.incident_stats:
            parts.append(
                f"\nOPEN INCIDENTS: "
                f"{self.incident_stats.get('open', 0)}")

        parts.append(
            "\nYou are a DAMA-DMBOK expert. Use this context to "
            "answer DQ questions with authority.")
        return "\n".join(parts)

    def _build_context_message(self, user_message):
        """Inject context about loaded data so Claude never asks for file paths."""
        # Prefer AgentSessionBridge for multi-file sessions
        if self.session_bridge:
            context = self.session_bridge.get_context()
            dq_ctx = self._build_dq_context()
            if context:
                ctx = context
                if dq_ctx:
                    ctx = ctx + "\n\n" + dq_ctx
                return ctx + "\n\n" + user_message
            if dq_ctx:
                return dq_ctx + "\n\n" + user_message
            return user_message

        # Fallback: single-file context injection
        context_parts = []

        if self.dataframe is not None:
            rows = len(self.dataframe)
            cols = len(self.dataframe.columns)
            col_list = ", ".join(self.dataframe.columns[:15])
            context_parts.append(
                f"[SYSTEM: Data is loaded and ready. {rows:,} rows x {cols} columns. "
                f"Columns: {col_list}. "
                f"Do NOT ask the user to upload data -- you already have it. "
                f"Use the tools (build_powerbi, build_html_dashboard, analyze_data) directly.]"
            )

        if self.data_file_path:
            context_parts.append(
                f"[SYSTEM: Source file: {os.path.basename(self.data_file_path)}. "
                f"Do NOT ask user for file path -- you already have it.]"
            )

        if self.data_profile:
            rows = self.data_profile.get("row_count", "?")
            cols = self.data_profile.get("column_count", "?")
            col_names = [
                c["name"] for c in self.data_profile.get("columns", [])
            ]
            context_parts.append(
                f"[SYSTEM: Data profile available: {rows} rows, {cols} columns: "
                f"{', '.join(col_names[:10])}. "
                f"Proceed with analysis -- do not ask for more info.]"
            )

        if self.dashboard_spec:
            context_parts.append(
                "[SYSTEM: A dashboard has already been designed and built. "
                "You can modify it, build another format, or analyze the data further.]"
            )

        if self.tableau_spec:
            ws_count = len(self.tableau_spec.get("worksheets", []))
            db_count = len(self.tableau_spec.get("dashboards", []))
            cf_count = len(self.tableau_spec.get("calculated_fields", []))
            context_parts.append(
                f"[SYSTEM: Tableau workbook parsed: {ws_count} worksheets, "
                f"{db_count} dashboards, {cf_count} calculated fields. "
                f"You CAN build dashboards and Power BI projects from this. "
                f"Synthetic data will be generated automatically if needed. "
                f"Do NOT ask the user to upload data -- just call the tools directly. "
                f"Do NOT say you need data or that data is missing.]"
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

        # Inject past deliverable awareness
        try:
            _similar = _search_deliverables(user_message)
            if _similar:
                _lines = []
                for _item in _similar[:3]:
                    _lines.append(
                        f"- {_item['type'].upper()}: {_item['name']} "
                        f"from {_item['source_file']} "
                        f"({_item['created_at'][:10]}). "
                        f"{_item['description']}"
                    )
                context_parts.append(
                    "[SYSTEM: Past deliverables that may be relevant:\n"
                    + "\n".join(_lines)
                    + "\nMention these naturally if the user's request "
                    "relates to past work. Do not force it.]"
                )
        except Exception:
            pass

        # Inject DQ scan results if available
        dq_ctx = self._build_dq_context()
        if dq_ctx:
            context_parts.append(dq_ctx)

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
            elif tool_name == "rationalize_dashboards":
                return self._tool_rationalize_dashboards(tool_input)
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

        # Try to recover data from session
        if self.dataframe is None:
            self._try_recover_data()
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

    def _generate_freeform_dashboard(self, user_request, title, progress_fn=None):
        """Let Claude generate the ENTIRE HTML dashboard from scratch.

        No template. No constraints. Claude analyzes the data and writes
        a complete standalone HTML file with whatever design, charts,
        colors, animations, and layout it decides is best.
        """
        df = self.dataframe
        if df is None:
            return None

        if progress_fn:
            progress_fn("Analyzing data for dashboard design...")

        # Build a rich data summary
        col_info = []
        for col in df.columns[:40]:
            nunique = df[col].nunique()
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                if len(s) > 0:
                    col_info.append(
                        f"  {col} (numeric): min={s.min():.2f}, max={s.max():.2f}, "
                        f"mean={s.mean():.2f}, median={s.median():.2f}, nulls={s.isna().sum()}"
                    )
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info.append(f"  {col} (date): {df[col].min()} to {df[col].max()}")
            else:
                top_vals = df[col].value_counts().head(5).to_dict()
                col_info.append(
                    f"  {col} (categorical, {nunique} unique): top values = {top_vals}"
                )

        # Embed a data sample as JSON for Plotly
        sample_size = min(5000, len(df))
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
        for c in df_sample.columns:
            if pd.api.types.is_datetime64_any_dtype(df_sample[c]):
                df_sample[c] = df_sample[c].dt.strftime("%Y-%m-%d")
        data_json = df_sample.to_json(orient="records", date_format="iso")

        data_summary = (
            f"Dataset: \"{title}\" -- {len(df):,} rows x {len(df.columns)} columns\n"
            f"Column details:\n" + "\n".join(col_info)
        )

        if progress_fn:
            progress_fn(
                f"Claude Opus is designing and coding your dashboard "
                f"({len(df.columns)} columns, {len(df):,} rows)..."
            )

        prompt = f"""You are Dr. Data, the most talented data visualization designer alive.

The user said: "{user_request}"
This is version {self._dashboard_iteration} -- make it DIFFERENT from any previous version.

{data_summary}

Generate a COMPLETE, standalone HTML file that renders an interactive dashboard.
The data is provided below as a JSON array -- embed it in a <script> tag.

REQUIREMENTS:
1. Use Plotly.js (CDN: https://cdn.plot.ly/plotly-2.35.0.min.js) for ALL charts
2. The HTML must be 100% self-contained (inline CSS, inline JS, embedded data)
3. Make it BEAUTIFUL. You are a design god. Think Bloomberg Terminal meets Apple.
4. Use a sophisticated color palette -- NOT just one color. Be creative.
5. Include interactive filters/dropdowns that actually filter all charts
6. Include KPI cards at the top showing the most important numbers
7. Include at least 5-8 different charts (mix of types: line, bar, heatmap, scatter, donut, area, treemap -- whatever fits the data best)
8. Each chart should have a title that states the INSIGHT, not just "X by Y"
9. Make the layout responsive (works on desktop and tablet)
10. Add smooth transitions/animations where appropriate
11. Add a dark professional theme
12. The footer should say: "Built by Dr. Data | The Art of the Possible"
13. Think about what STORY the data tells. Lead with the most important finding.
14. Use Google Fonts (Inter or similar professional font)

DO NOT:
- Use any external dependencies other than Plotly.js CDN and Google Fonts
- Use placeholder data -- use the ACTUAL data provided below
- Make it boring or generic
- Use the same layout as a typical template

DATA (JSON array, {sample_size:,} rows):
The data is too large to include in the prompt. Instead, use this variable in your script:
var DATA = {{DATA_PLACEHOLDER}};

Output ONLY the complete HTML file. No markdown fences. No explanation. Just the HTML.
Start with <!DOCTYPE html> and end with </html>."""

        try:
            collected = []
            with self.client.messages.stream(
                model=self.MODEL,
                max_tokens=16000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    collected.append(chunk)
            html = "".join(collected)

            # Strip markdown fences if present
            import re as _re
            html = _re.sub(r'^```(?:html)?\s*\n?', '', html.strip())
            html = _re.sub(r'\n?```\s*$', '', html)

            # Inject the actual data
            html = html.replace("{DATA_PLACEHOLDER}", data_json)
            html = html.replace("{{DATA_PLACEHOLDER}}", data_json)

            # If Claude didn't include the data placeholder, inject before </script>
            if "DATA_PLACEHOLDER" not in "".join(collected) and "var DATA" not in html:
                html = html.replace(
                    "</script>",
                    f"\nvar DATA = {data_json};\n</script>",
                    1
                )

            # Validate it's actual HTML
            if not html.strip().startswith("<!") and not html.strip().startswith("<html"):
                print("[FREEFORM] Output doesn't look like HTML, falling back to template")
                return self._fallback_template_dashboard(title)

            # Save
            safe_title = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in title
            ).strip().replace(" ", "_") or "Dashboard"
            filepath = os.path.join(
                self.output_dir,
                f"{safe_title}_v{self._dashboard_iteration}.html"
            )
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)

            if progress_fn:
                progress_fn(f"Dashboard generated: {len(html):,} characters of custom HTML")

            return os.path.abspath(filepath)

        except Exception as e:
            print(f"[FREEFORM] Claude generation failed: {e}")
            if progress_fn:
                progress_fn("Free-form generation failed, using designed template...")
            return self._fallback_template_dashboard(title)

    def _fallback_template_dashboard(self, title):
        """Fallback to the template builder if free-form generation fails."""
        try:
            return self.html_builder.build(
                self.dataframe,
                output_dir=self.output_dir,
                title=title,
            )
        except Exception:
            return None

    def _design_html_dashboard(self, user_request, title, progress_fn=None):
        """Use Claude to design a dashboard spec for HTML rendering.

        Analyzes the actual data and user request to decide what charts,
        KPIs, and layout to produce. Each call produces different output
        based on the iteration counter and user request.
        """
        if progress_fn:
            progress_fn("Claude is analyzing your data and designing the dashboard...")

        # Build a compact data summary for Claude
        df = self.dataframe
        col_info = []
        for col in df.columns[:30]:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            if pd.api.types.is_numeric_dtype(df[col]):
                stats = f"min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}"
                col_info.append(f"  {col} (numeric, {nunique} unique): {stats}")
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info.append(f"  {col} (date): {df[col].min()} to {df[col].max()}")
            else:
                col_info.append(f"  {col} (text, {nunique} unique): {sample[:3]}")

        data_summary = (
            f"Dataset: {len(df)} rows x {len(df.columns)} columns\n"
            f"Columns:\n" + "\n".join(col_info)
        )

        prompt = f"""Design an interactive HTML dashboard for this data.

USER REQUEST: {user_request}
ITERATION: {self._dashboard_iteration} (make this version DIFFERENT from previous versions)

{data_summary}

Return a JSON object with this exact structure:
{{
  "title": "Dashboard Title",
  "subtitle": "One-line insight",
  "kpis": [
    {{"column": "col_name", "label": "Display Label", "agg": "sum|avg|count|max|min"}}
  ],
  "charts": [
    {{
      "type": "line|bar|hbar|donut|scatter|area|stacked_bar|heatmap|treemap",
      "title": "Chart Title",
      "x": "column_name",
      "y": "column_name",
      "color": "column_name_or_null",
      "agg": "sum|avg|count|max|min",
      "sort": "desc|asc|none",
      "top_n": null,
      "width": "full|half",
      "insight": "One sentence explaining what this chart reveals"
    }}
  ],
  "filters": [
    {{"column": "col_name"}}
  ]
}}

RULES:
- Use ONLY columns that exist in the dataset above
- Design 4-8 charts that tell a compelling DATA STORY
- Lead with the most important insight, not alphabetical columns
- For iteration {self._dashboard_iteration}: {"Focus on TRENDS and TIME PATTERNS" if self._dashboard_iteration % 3 == 1 else "Focus on COMPARISONS and RANKINGS" if self._dashboard_iteration % 3 == 2 else "Focus on DISTRIBUTIONS and OUTLIERS"}
- Include at least one donut or treemap for composition
- Include at least one time series if date columns exist
- KPIs should be the 3-4 most important numbers a decision-maker needs
- Every chart must have a specific insight explaining what it shows
- Do NOT just list all columns -- be SELECTIVE and ANALYTICAL
- Use descriptive chart titles that state the finding, not just "X by Y"

Output ONLY valid JSON. No markdown. No commentary."""

        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=4096,
                temperature=0.7,  # Some creativity
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences
            import re as _re
            raw = _re.sub(r'^```(?:json)?\s*\n?', '', raw)
            raw = _re.sub(r'\n?```\s*$', '', raw)
            spec = json.loads(raw)
            if progress_fn:
                n_charts = len(spec.get("charts", []))
                n_kpis = len(spec.get("kpis", []))
                progress_fn(
                    f"Dashboard designed: {n_charts} charts, {n_kpis} KPIs, "
                    f"{len(spec.get('filters', []))} filters"
                )
            return spec
        except Exception as e:
            print(f"[DESIGN] Claude dashboard design failed: {e}")
            if progress_fn:
                progress_fn("Design failed, using auto-detected layout...")
            return None

    def _report_progress(self, msg):
        """Send a progress update to the UI if a callback is set."""
        if hasattr(self, "_progress_callback") and self._progress_callback:
            self._progress_callback(msg)

    def _try_recover_data(self):
        """Try every available source to recover a dataframe.

        Called when self.dataframe is None but the user expects data to be loaded.
        Sources tried in order:
        1. MultiFileSession primary dataframe
        2. data_file_path (CSV/Excel/Parquet on disk)
        3. Synthetic generation from tableau_spec
        """
        # 1. Session
        if self.session:
            try:
                df = self.session.get_primary_dataframe()
                if df is not None:
                    self.dataframe = df
                    if self.data_profile is None:
                        self.data_profile = self.analyzer.profile(df)
                    print(f"[RECOVER] Loaded from session: {df.shape}")
                    return
            except Exception:
                pass

        # 2. File on disk
        if self.data_file_path and os.path.exists(self.data_file_path):
            try:
                ext = self.data_file_path.rsplit(".", 1)[-1].lower()
                if ext == "csv":
                    self.dataframe = pd.read_csv(self.data_file_path)
                elif ext in ("xlsx", "xls"):
                    from app.file_handler import load_excel_smart
                    self.dataframe, self.sheet_name = load_excel_smart(self.data_file_path)
                elif ext == "parquet":
                    self.dataframe = pd.read_parquet(self.data_file_path)
                elif ext == "json":
                    self.dataframe = pd.read_json(self.data_file_path)
                if self.dataframe is not None:
                    if self.data_profile is None:
                        self.data_profile = self.analyzer.profile(self.dataframe)
                    print(f"[RECOVER] Loaded from file: {self.dataframe.shape}")
                    return
            except Exception as e:
                print(f"[RECOVER] File load failed: {e}")

        # 3. Synthetic from Tableau spec
        if self.tableau_spec:
            try:
                from core.synthetic_data import generate_from_tableau_spec
                output_dir = str(PROJECT_ROOT / "output")
                df, csv_path, _ = generate_from_tableau_spec(
                    self.tableau_spec, num_rows=2000, output_dir=output_dir
                )
                self.dataframe = df
                self.data_file_path = csv_path
                self.data_path = csv_path
                if self.data_profile is None:
                    self.data_profile = self.analyzer.profile(df)
                print(f"[RECOVER] Generated synthetic: {df.shape}")
                return
            except Exception as e:
                print(f"[RECOVER] Synthetic failed: {e}")
                import traceback
                traceback.print_exc()

    def _tool_build_powerbi(self, inputs):
        """Build a Power BI project using the full AI pipeline and ZIP it."""
        request = inputs.get("request", "Build a comprehensive dashboard")
        audience = inputs.get("audience", "executive")
        project_name = inputs.get("project_name", "Dashboard")

        # Try to recover data from session if not on agent
        if self.dataframe is None:
            self._try_recover_data()
        if self.dataframe is None:
            return json.dumps({"error": "No data available. Upload a data file first."})

        row_count = len(self.dataframe)
        col_count = len(self.dataframe.columns)

        # Safety: if data_file_path points to a Tableau archive (.twbx/.twb),
        # Power BI cannot load it as CSV/Excel. Export the DataFrame to CSV.
        if self.data_file_path:
            _ext = self.data_file_path.rsplit(".", 1)[-1].lower() if "." in self.data_file_path else ""
            if _ext in ("twbx", "twb"):
                csv_dir = str(PROJECT_ROOT / "output")
                os.makedirs(csv_dir, exist_ok=True)
                _base = os.path.splitext(os.path.basename(self.data_file_path))[0]
                _base = _base.replace(" ", "_") + "_data.csv"
                _csv_path = os.path.join(csv_dir, _base)
                self.dataframe.to_csv(_csv_path, index=False)
                self.data_file_path = _csv_path
                self.data_path = _csv_path
                print(f"[PBI] Exported DataFrame to CSV (was .twbx): {_csv_path}")

        self._report_progress(
            f"Profiling data: {row_count:,} rows x {col_count} columns -- "
            f"detecting column types, distributions, and relationships"
        )

        # Use DataAnalyzer (not DeepAnalyzer) for the PBI pipeline --
        # ClaudeInterpreter expects table_name, column semantic types, etc.
        from core.data_analyzer import DataAnalyzer
        pbi_analyzer = DataAnalyzer()

        # Derive table name: prefer sheet name (matches Excel data), else file name
        if self.snowflake_config and self.snowflake_tables:
            # Use first loaded Snowflake table name as-is
            table_name = list(self.snowflake_tables.keys())[0]
        elif self.sheet_name:
            table_name = self.sheet_name.replace(" ", "_").replace("-", "_")
        elif self.data_file_path:
            base = os.path.basename(self.data_file_path)
            table_name = base.rsplit(".", 1)[0].replace(" ", "_").replace("-", "_")
        else:
            table_name = project_name.replace(" ", "_")

        # Never expose the internal synthetic extraction filename as PBI table name
        if table_name == "synthetic_tableau_data":
            table_name = "Data"

        pbi_profile = pbi_analyzer.analyze(self.dataframe, table_name=table_name)

        n_measures = sum(
            1 for c in pbi_profile.get("columns", [])
            if c.get("semantic_type") == "measure"
        )
        n_dims = col_count - n_measures
        self._report_progress(
            f"Data profiled: {n_dims} dimensions, {n_measures} measures, "
            f"quality score {pbi_profile.get('data_quality_score', '?')}/100"
        )

        # --- DIRECT TABLEAU REPLICA PATH ---
        # When we have a Tableau spec with dashboards, bypass AI for visual
        # structure. This produces a faithful replica instead of an AI
        # "interpretation". The existing AI path is preserved as fallback.
        # Use direct mapper whenever we have Tableau structure — dashboards OR
        # bare worksheets. Never fall through to AI for Tableau files.
        # user_intent from the intelligence card buttons overrides:
        #   "replicate" -> force direct mapper path
        #   "reimagine" -> force full AI pipeline path
        try:
            import streamlit as _st_intent
            _user_intent = _st_intent.session_state.get("user_intent", "")
        except Exception:
            _user_intent = ""
        _has_tableau_structure = bool(
            self.tableau_spec and (
                self.tableau_spec.get("dashboards") or
                self.tableau_spec.get("worksheets")
            )
        )
        # If user chose "reimagine", skip direct mapper -- fall through to AI
        if _user_intent == "reimagine":
            _has_tableau_structure = False
            self._report_progress(
                "Reimagine mode: using full AI pipeline for creative dashboard design"
            )
        if _has_tableau_structure:
            try:
                from core.direct_mapper import build_pbip_config_from_tableau

                ws_count = len(self.tableau_spec.get("worksheets", []))
                db_count = len(self.tableau_spec.get("dashboards", []))
                cf_count = len(self.tableau_spec.get("calculated_fields", []))
                self._report_progress(
                    f"Direct Tableau replica mode: mapping {ws_count} worksheets, "
                    f"{db_count} dashboards, {cf_count} calculated fields "
                    f"-- no AI for visual structure"
                )

                config, dashboard_spec = build_pbip_config_from_tableau(
                    self.tableau_spec, pbi_profile, table_name
                )

                sections = config.get("report_layout", {}).get("sections", [])
                total_vc = sum(
                    len(s.get("visualContainers", [])) for s in sections
                )
                measure_count = len(
                    config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", [])
                )
                self._report_progress(
                    f"Direct layout built: {len(sections)} pages, "
                    f"{total_vc} visuals, {measure_count} DAX measures "
                    f"-- deterministic mapping (no AI interpretation)"
                )

                self.dashboard_spec = dashboard_spec

                # Jump to PBIP generator (skip Claude + GPT-4 pipeline)
                self._report_progress(
                    "Writing Power BI project files: page layouts, "
                    "visual.json, TMDL model, DAX measures, theme..."
                )
                from generators.pbip_generator import PBIPGenerator
                output_dir = str(PROJECT_ROOT / "output")
                generator = PBIPGenerator(output_dir)

                relationships = []
                if self.session and self.session.relationships:
                    relationships.extend(self.session.relationships)
                if self.tableau_spec.get("relationships"):
                    relationships.extend(self.tableau_spec["relationships"])

                gen_result = generator.generate(
                    config, pbi_profile, dashboard_spec,
                    data_file_path=self.data_file_path,
                    sheet_name=self.sheet_name,
                    relationships=relationships or None,
                    snowflake_config=self.snowflake_config,
                    csv_url=self._get_csv_url(),
                )
                result_path = gen_result["path"]
                field_audit = gen_result.get("field_audit", {})
                valid_measures = gen_result.get("valid_measures", [])

                valid_count = field_audit.get("valid", 0)
                fixed_count = field_audit.get("fixed", 0)
                removed_count = field_audit.get("removed", 0)
                self._report_progress(
                    f"PBIP project built (direct mapper): "
                    f"{gen_result.get('file_count', '?')} files, "
                    f"{len(valid_measures)} DAX measures validated. "
                    f"Field audit: {valid_count} valid, {fixed_count} auto-fixed, "
                    f"{removed_count} removed. Packaging ZIP..."
                )

                # Copy data file into project
                data_filename = ""
                if self.data_file_path and os.path.isfile(self.data_file_path):
                    data_filename = os.path.basename(self.data_file_path)
                    dst = os.path.join(result_path, data_filename)
                    if not os.path.exists(dst):
                        shutil.copy2(self.data_file_path, dst)

                # Calculation validation
                calc_audit_result = None
                if (self.tableau_spec.get("calculated_fields")
                        and self.dataframe is not None):
                    try:
                        from core.calc_validator import (
                            validate_calculations,
                            extract_dax_measures_from_config,
                            write_calculation_audit,
                        )
                        dax_measures = extract_dax_measures_from_config(config)
                        calc_audit_result = validate_calculations(
                            self.tableau_spec, dax_measures, self.dataframe,
                        )
                        audit_path = write_calculation_audit(
                            calc_audit_result, result_path,
                        )
                        self.generated_files.append(audit_path)
                        self._report_progress(
                            f"Calculation audit: {calc_audit_result['validated']} "
                            f"matched, {calc_audit_result['mismatched']} mismatched, "
                            f"{calc_audit_result['skipped_tableau'] + calc_audit_result['skipped_dax']} skipped"
                        )
                    except Exception as cv_err:
                        print(f"[CALC-VALIDATOR] Audit failed (non-fatal): {cv_err}")

                # Generate formula translation audit report (HTML)
                try:
                    from core.audit_report import build_audit_data, generate_audit_report
                    _measures_full = dashboard_spec.get("measures_full", [])
                    if not _measures_full:
                        # Fallback: reconstruct from config measures
                        _measures_full = [
                            {"name": m["name"], "dax": m.get("dax", m.get("expression", "")),
                             "format": m.get("format", "#,0")}
                            for m in config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", [])
                        ]
                    _wb_name = self.tableau_spec.get("workbook_name", "")
                    if not _wb_name and self.data_file_path:
                        _wb_name = os.path.splitext(os.path.basename(self.data_file_path))[0]
                    _audit_data = build_audit_data(
                        measures=_measures_full,
                        tableau_spec=self.tableau_spec,
                        calc_audit_result=calc_audit_result,
                        workbook_name=_wb_name,
                        tableau_version=self.tableau_spec.get("version", ""),
                    )
                    _audit_html = generate_audit_report(
                        fields=_audit_data["fields"],
                        dq=_audit_data["dq"],
                        warnings=_audit_data["warnings"],
                        meta=_audit_data["meta"],
                    )
                    _audit_html_path = os.path.join(self.output_dir, "translation_audit.html")
                    with open(_audit_html_path, "w", encoding="utf-8") as _af:
                        _af.write(_audit_html)
                    self.generated_files.append(_audit_html_path)
                    self._report_progress(
                        f"Translation audit report: DQ {_audit_data['dq']['score']}/100, "
                        f"{_audit_data['meta']['auto_good']} mapped, "
                        f"{_audit_data['meta']['blocked']} blocked"
                    )
                except Exception as _ar_err:
                    print(f"[AUDIT-REPORT] Generation failed (non-fatal): {_ar_err}")

                # Preflight validation + self-healing
                try:
                    from core.preflight_validator import validate as _preflight
                    from core.pbip_healer import heal as _heal
                    _pf = _preflight(result_path)
                    if not _pf.all_passed:
                        _fixes = _heal(result_path, _pf)
                        self._report_progress(
                            f"Preflight: {_pf.fail_count} issues found, "
                            f"{len(_fixes)} auto-healed"
                        )
                        _pf2 = _preflight(result_path)
                        if not _pf2.all_passed:
                            for _f in _pf2.failed:
                                print(f"    [PREFLIGHT] still failing: {_f}")
                    else:
                        self._report_progress(
                            f"Preflight: {_pf.pass_count} checks passed"
                        )
                except Exception as _pf_err:
                    print(f"[PREFLIGHT] Non-fatal: {_pf_err}")

                zip_name = project_name.replace(" ", "_")
                zip_path = shutil.make_archive(
                    os.path.join(output_dir, zip_name), "zip", result_path
                )
                self.generated_files.append(zip_path)
                self._report_progress("Power BI project ready for download (direct Tableau replica)")

                # Build context for summary
                data_file_name = (os.path.basename(self.data_file_path)
                                  if self.data_file_path else "unknown")
                page_names = [p.get("name", f"Page {i+1}")
                              for i, p in enumerate(dashboard_spec.get("pages", []))]
                visuals_per_page = [len(p.get("visuals", []))
                                    for p in dashboard_spec.get("pages", [])]
                measure_names = [m["name"] for m in dashboard_spec.get("measures", [])]

                return json.dumps({
                    "status": "success",
                    "method": "direct_tableau_mapper",
                    "project_path": result_path,
                    "file_path": zip_path,
                    "zip_path": zip_path,
                    "data_file": data_file_name,
                    "data_shape": f"{row_count} rows x {col_count} columns",
                    "pages": page_names,
                    "visuals_per_page": visuals_per_page,
                    "total_visuals": sum(visuals_per_page),
                    "measures": measure_names,
                    "field_audit": field_audit,
                    "build_context": {
                        "data_file": data_file_name,
                        "row_count": row_count,
                        "col_count": col_count,
                        "table_name": table_name,
                        "page_names": page_names,
                        "visuals_per_page": visuals_per_page,
                        "measures_created": measure_names,
                        "measures_skipped": [],
                        "fields_fixed": field_audit.get("fixed", 0),
                        "fields_removed": field_audit.get("removed", 0),
                        "numeric_count": n_measures,
                        "categorical_count": n_dims,
                        "tableau_file": os.path.basename(
                            self.data_path or ""
                        ) if self.data_path else "",
                        "tableau_worksheets": [
                            ws["name"] for ws in
                            self.tableau_spec.get("worksheets", [])
                        ],
                        "method": "direct_tableau_mapper",
                    },
                    "message": (
                        f"Power BI project built via direct Tableau mapper "
                        f"(deterministic, no AI interpretation). "
                        f"{len(page_names)} pages, {sum(visuals_per_page)} visuals, "
                        f"{len(measure_names)} DAX measures."
                    ),
                })

            except Exception as dm_err:
                # Direct mapper failed -- fall through to AI pipeline
                print(f"[DIRECT-MAPPER] Failed (falling back to AI pipeline): {dm_err}")
                import traceback
                traceback.print_exc()
                self._report_progress(
                    f"Direct mapper encountered an issue -- "
                    f"falling back to AI-assisted pipeline"
                )

        try:
            # -- REQUIREMENTS CONTRACT (runs before any LLM calls) --
            from core.requirements_contract import (
                build_contract, validate_contract, enforce_contract,
                format_contract_for_prompt,
            )

            tableau_wb_for_contract = None
            if self.tableau_spec:
                try:
                    from core.tableau_extractor import extract_workbook
                    # Try to extract rich spec; fall back to None
                    # (the multi_file_handler may have stored a simpler dict)
                    tableau_wb_for_contract = extract_workbook(
                        self.tableau_spec
                    ) if isinstance(self.tableau_spec, (str, bytes)) else None
                except Exception:
                    pass

            contract = build_contract(
                user_request=request,
                tableau_wb=tableau_wb_for_contract,
                source_file=os.path.basename(self.data_file_path or ""),
                data_columns=list(self.dataframe.columns) if self.dataframe is not None else [],
            )

            contract_errors = validate_contract(contract)
            if contract_errors:
                print(f"[CONTRACT] Validation warnings: {contract_errors}")

            contract_prompt = format_contract_for_prompt(contract)

            self._report_progress(
                f"Requirements contract built: {contract.page_count} page(s), "
                f"{len(contract.visuals)} visuals, "
                f"{len(contract.top_filters)} slicers, "
                f"{len(contract.manual_review_items)} items for review"
            )

            # Step 2: Claude interprets request -> dashboard spec
            from core.claude_interpreter import ClaudeInterpreter
            interpreter = ClaudeInterpreter()

            if self.tableau_spec:
                # --- TABLEAU MIGRATION PATH ---
                ws_count = len(self.tableau_spec.get("worksheets", []))
                db_count = len(self.tableau_spec.get("dashboards", []))
                cf_count = len(self.tableau_spec.get("calculated_fields", []))
                self._report_progress(
                    f"Mapping Tableau structure: {ws_count} worksheets, "
                    f"{db_count} dashboards, {cf_count} calculated fields "
                    f"-- building migration intent"
                )

                # Build structured migration intent
                from core.visual_intent import (
                    extract_migration_intent, format_intent_for_prompt,
                )
                migration_intent = extract_migration_intent(
                    self.tableau_spec, request
                )

                n_pages = len(migration_intent.pages)
                n_visuals = sum(len(p.visuals) for p in migration_intent.pages)
                n_warnings = len(migration_intent.warnings)
                self._report_progress(
                    f"Migration plan ready: {n_pages} pages, "
                    f"{n_visuals} visuals mapped to Power BI equivalents"
                    + (f", {n_warnings} compatibility warnings" if n_warnings else "")
                )

                self._report_progress(
                    "Claude Opus 4 is translating Tableau visuals, "
                    "DAX measures, and layout to Power BI format -- "
                    "this takes 15-30 seconds"
                )

                translate_request = format_intent_for_prompt(
                    migration_intent, table_name
                )

                # Prepend the user's actual instruction so Claude knows
                # what they want (replicate vs. reimagine vs. specific changes)
                translate_request = (
                    f"USER REQUEST: {request}\n\n"
                    f"Follow the user's request above. If they asked for a "
                    f"replica, match the Tableau visuals closely. If they "
                    f"asked for something different, use the Tableau structure "
                    f"as a data reference but create a fresh design.\n\n"
                    + translate_request
                )

                # Append DAX mapping reference + contract
                translate_request += (
                    "\n\n" + self._TABLEAU_DAX_MAP.replace(
                        "TableName", table_name
                    )
                    + "\n\n" + contract_prompt
                )

                dashboard_spec = interpreter.interpret_migration(
                    translate_request, pbi_profile
                )

                # Enforce contract on LLM output
                dashboard_spec, contract_violations = enforce_contract(
                    contract, dashboard_spec
                )
                if contract_violations:
                    self._report_progress(
                        f"Contract enforcement: {len(contract_violations)} "
                        f"correction(s) applied to LLM output"
                    )
                    for cv in contract_violations:
                        print(f"[CONTRACT FIX] {cv}")

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
                dashboard_spec["migration_warnings"] = (
                    dashboard_spec.get("migration_warnings", [])
                    + migration_intent.warnings
                )

                # Attach design metadata for visual replication
                dashboard_spec["design"] = self.tableau_spec.get("design", {})
                ws_designs = {}
                ws_chart_types = {}
                for ws in self.tableau_spec.get("worksheets", []):
                    if ws.get("design"):
                        ws_designs[ws["name"]] = ws["design"]
                    if ws.get("chart_type"):
                        ws_chart_types[ws["name"]] = ws["chart_type"]
                dashboard_spec["worksheet_designs"] = ws_designs
                dashboard_spec["worksheet_chart_types"] = ws_chart_types
            else:
                # --- FRESH DESIGN PATH (CSV / Excel) ---
                self._report_progress(
                    "Claude Opus 4 is designing your dashboard "
                    "(pages, visuals, DAX measures)..."
                )
                # Inject contract into the request
                enriched_request = request + "\n\n" + contract_prompt
                dashboard_spec = interpreter.interpret(enriched_request, pbi_profile)

                # Enforce contract on LLM output
                dashboard_spec, contract_violations = enforce_contract(
                    contract, dashboard_spec
                )
                if contract_violations:
                    self._report_progress(
                        f"Contract enforcement: {len(contract_violations)} "
                        f"correction(s) applied to LLM output"
                    )
                    for cv in contract_violations:
                        print(f"[CONTRACT FIX] {cv}")

            self.dashboard_spec = dashboard_spec

            page_count = len(dashboard_spec.get("pages", []))
            visual_count = sum(
                len(p.get("visuals", []))
                for p in dashboard_spec.get("pages", [])
            )

            # Step 3: OpenAI generates PBIP config (report layout + data model)
            measure_count = len(dashboard_spec.get("measures", []))
            self._report_progress(
                f"GPT-4 is generating Power BI visual containers "
                f"and TMDL data model -- {page_count} pages, "
                f"{visual_count} visuals, {measure_count} DAX measures"
            )
            from core.openai_engine import OpenAIEngine
            engine = OpenAIEngine()
            config = engine.generate_pbip_config(
                dashboard_spec, pbi_profile,
                user_instructions=request,
            )

            sections = config.get("report_layout", {}).get("sections", [])
            total_vc = sum(
                len(s.get("visualContainers", [])) for s in sections
            )
            self._report_progress(
                f"Layout generated: {len(sections)} pages with "
                f"{total_vc} visual containers positioned on 1280x720 canvas"
            )

            # Step 4: PBIPGenerator creates the actual project files
            self._report_progress(
                "Writing Power BI project files: page layouts, "
                "visual.json, TMDL model, DAX measures, theme..."
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
                snowflake_config=self.snowflake_config,
                csv_url=self._get_csv_url(),
            )
            # generator.generate() returns a dict with path + audit info
            result_path = gen_result["path"]
            field_audit = gen_result.get("field_audit", {})
            valid_measures = gen_result.get("valid_measures", [])

            # Step 5: Bundle data file + ZIP for download
            valid_count = field_audit.get("valid", 0)
            fixed_count = field_audit.get("fixed", 0)
            removed_count = field_audit.get("removed", 0)
            self._report_progress(
                f"PBIP project built: {gen_result.get('file_count', '?')} files, "
                f"{len(valid_measures)} DAX measures validated. "
                f"Field audit: {valid_count} valid, {fixed_count} auto-fixed, "
                f"{removed_count} removed. Packaging ZIP..."
            )

            # Copy the data file into the project so the user has it
            data_filename = ""
            if self.data_file_path and os.path.isfile(self.data_file_path):
                data_filename = os.path.basename(self.data_file_path)
                dst = os.path.join(result_path, data_filename)
                if not os.path.exists(dst):
                    shutil.copy2(self.data_file_path, dst)

            # -- Calculation Validation (Tableau vs DAX) --
            calc_audit_result = None
            if (self.tableau_spec
                    and self.tableau_spec.get("calculated_fields")
                    and self.dataframe is not None):
                try:
                    from core.calc_validator import (
                        validate_calculations,
                        extract_dax_measures_from_config,
                        write_calculation_audit,
                    )
                    dax_measures = extract_dax_measures_from_config(config)
                    calc_audit_result = validate_calculations(
                        self.tableau_spec, dax_measures, self.dataframe,
                    )
                    audit_path = write_calculation_audit(
                        calc_audit_result, result_path,
                    )
                    self.generated_files.append(audit_path)
                    self._report_progress(
                        f"Calculation audit: {calc_audit_result['validated']} "
                        f"matched, {calc_audit_result['mismatched']} mismatched, "
                        f"{calc_audit_result['skipped_tableau'] + calc_audit_result['skipped_dax']} skipped"
                    )
                except Exception as cv_err:
                    print(f"[CALC-VALIDATOR] Audit failed (non-fatal): {cv_err}")

            # Preflight validation + self-healing
            try:
                from core.preflight_validator import validate as _preflight
                from core.pbip_healer import heal as _heal
                _pf = _preflight(result_path)
                if not _pf.all_passed:
                    _fixes = _heal(result_path, _pf)
                    self._report_progress(
                        f"Preflight: {_pf.fail_count} issues found, "
                        f"{len(_fixes)} auto-healed"
                    )
                    _pf2 = _preflight(result_path)
                    if not _pf2.all_passed:
                        for _f in _pf2.failed:
                            print(f"    [PREFLIGHT] still failing: {_f}")
                else:
                    self._report_progress(
                        f"Preflight: {_pf.pass_count} checks passed"
                    )
            except Exception as _pf_err:
                print(f"[PREFLIGHT] Non-fatal: {_pf_err}")

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
                "calc_validation": calc_audit_result,
            }

            # Log the full build context to trace logger
            self.trace.log("powerbi_build_summary", build_context)

            # -- QA MANIFEST (mandatory) --
            self._report_progress("Generating QA audit manifest...")
            try:
                from core.qa_manifest import build_manifest, save_manifest
                qa_manifest = build_manifest(
                    contract=locals().get("contract"),
                    equivalency_results=None,  # TODO: wire equivalency engine
                    build_context=build_context,
                    dashboard_spec=dashboard_spec,
                    field_audit=field_audit,
                    contract_violations=locals().get("contract_violations", []),
                    output_path=zip_path,
                )
                qa_paths = save_manifest(qa_manifest, self.output_dir)
                self.generated_files.append(qa_paths["json"])
                self.generated_files.append(qa_paths["markdown"])

                must_verify = sum(
                    1 for c in qa_manifest.qa_checklist
                    if c.severity == "must_verify"
                )
                auto_fail = sum(
                    1 for c in qa_manifest.qa_checklist
                    if c.auto_result == "fail"
                )
                self._report_progress(
                    f"QA manifest generated: {len(qa_manifest.qa_checklist)} checks "
                    f"({must_verify} must-verify, {auto_fail} auto-fail)"
                )
            except Exception as qa_err:
                print(f"[QA] Manifest generation failed: {qa_err}")

            return json.dumps({
                "status": "success",
                "file_path": zip_path,
                "project_name": project_name,
                "pages": page_count,
                "visuals": visual_count,
                "measures": len(valid_measures),
                "build_context": build_context,
                "qa_manifest": qa_paths.get("markdown", "") if 'qa_paths' in dir() else "",
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
        "multipolygon": "filledMap",
        "polygon": "filledMap",
        "circle": "scatterChart",
        "shape": "scatterChart",
        "square": "treemap",
        "automatic": "clusteredBarChart",
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
                rows = ws.get("rows_fields", ws.get("rows", []))
                cols = ws.get("cols_fields", ws.get("cols", []))
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
                sheets = db.get("worksheets_used", db.get("worksheets", db.get("sheets", [])))
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

    def _tool_rationalize_dashboards(self, inputs):
        """Run dashboard rationalization analysis."""
        action = inputs.get("action", "analyze")
        eng = self.rationalization_engine

        if action == "generate_sample":
            df = eng.generate_sample_data(num_dashboards=10000)
            eng.load_inventory(df)
            result = eng.analyze()
            summary = eng.get_executive_summary()
            return json.dumps({
                "status": "ok",
                "action": "generate_sample",
                "total_dashboards": len(df),
                "executive_summary": summary,
            }, default=str)

        elif action == "analyze":
            if eng.inventory is None:
                df = eng.generate_sample_data(num_dashboards=10000)
                eng.load_inventory(df)
            result = eng.analyze()
            summary = eng.get_executive_summary()
            return json.dumps({
                "status": "ok",
                "action": "analyze",
                "executive_summary": summary,
                "cost_impact": result.get("cost_impact", {}),
                "zombie_counts": {
                    "over_90d": result["zombies"]["over_90d"],
                    "over_180d": result["zombies"]["over_180d"],
                    "over_365d": result["zombies"]["over_365d"],
                },
                "duplicate_groups": result["duplicates"]["total_groups"],
                "retirement_plan": result["retirement_plan"],
            }, default=str)

        elif action == "report":
            if not eng.analysis:
                if eng.inventory is None:
                    df = eng.generate_sample_data(num_dashboards=10000)
                    eng.load_inventory(df)
                eng.analyze()
            html = eng.generate_html_report()
            os.makedirs(self.output_dir, exist_ok=True)
            report_path = os.path.join(
                self.output_dir, "dashboard_rationalization_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            self.generated_files.append(report_path)
            return json.dumps({
                "status": "ok",
                "action": "report",
                "file": report_path,
                "executive_summary": eng.get_executive_summary(),
            }, default=str)

        return json.dumps({"error": f"Unknown action: {action}"})

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
