"""
Dr. Data -- Dashboard Intelligence Platform
Split-panel layout: Workspace (center) + Chat (right)
+ Iteration 2: Audit layer, audience toggle, Dr. Data avatar
"""
import sys
import io
import os
import json
import time
import html as html_module
import tempfile
from pathlib import Path
from datetime import datetime

# Windows encoding fix
if sys.platform == "win32" and getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from app.dr_data_agent import DrDataAgent
from app.file_handler import ingest_file, ALL_SUPPORTED
from core.multi_file_handler import MultiFileSession
from core.audit_engine import AuditEngine
from core.deliverable_registry import get_recent as _get_recent_deliverables
from core.dq_engine import DataQualityEngine
from core.data_catalog import DataCatalog
from core.rules_engine import BusinessRulesEngine
from core.dq_history import DQHistory
from core.trust_scoring import TrustScorer
from core.copdq import COPDQCalculator
from core.stewardship import StewardshipWorkflow
from core.dashboard_rationalization import DashboardRationalizationEngine
from app.data_modeling import render_data_modeling
from app.chatbots import (
    render_tab_chatbot,
    MIGRATION_EXPERT_PROMPT, DQ_ENGINE_PROMPT,
    RATIONALIZATION_EXPERT_PROMPT, MODELER_EXPERT_PROMPT,
)
from app.ui_constants import GLOBAL_CSS, tab_header, badge


def _safe_html(html_str, fallback_text=""):
    """Render HTML via st.markdown with fallback to st.write on error."""
    try:
        st.markdown(html_str, unsafe_allow_html=True)
    except Exception:
        st.write(fallback_text or html_str)


def _escape_user_text(text):
    """Escape user-provided text before embedding in HTML to prevent XSS."""
    return html_module.escape(str(text)) if text else ""


# === PAGE CONFIG (must be first Streamlit call) ===
st.set_page_config(
    page_title="Dr. Data -- Dashboard Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DR. DATA AVATAR (inline SVG)
# =============================================================================

DR_DATA_AVATAR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="36" height="36"><defs><linearGradient id="abg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#1a1a1a"/><stop offset="100%" style="stop-color:#2d2d2d"/></linearGradient><linearGradient id="agl" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#3a3a3a"/><stop offset="100%" style="stop-color:#2d2d2d"/></linearGradient></defs><circle cx="24" cy="24" r="23" fill="url(#abg)" stroke="#4a4a4a" stroke-width="1"/><rect x="10" y="9" width="28" height="24" rx="8" fill="url(#agl)" stroke="#FFDE00" stroke-width="1"/><rect x="13" y="15" width="22" height="8" rx="4" fill="#1a1a1a" stroke="#FFDE00" stroke-width="0.8"/><circle cx="19" cy="19" r="2" fill="#FFDE00"/><circle cx="29" cy="19" r="2" fill="#FFDE00"/><path d="M 18 28 Q 24 33 30 28" fill="none" stroke="#FFDE00" stroke-width="1.5" stroke-linecap="round"/><line x1="24" y1="9" x2="24" y2="4" stroke="#4a4a4a" stroke-width="1.5"/><circle cx="24" cy="3" r="2" fill="#FFE600"/><rect x="12" y="38" width="24" height="6" rx="2" fill="#1a1a1a" stroke="#4a4a4a" stroke-width="0.5"/><text x="24" y="43" text-anchor="middle" fill="#FFDE00" font-family="monospace" font-size="4" font-weight="bold">DR.DATA</text></svg>'


# === CUSTOM CSS -- THE ENTIRE LOOK (Western Union) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Kill Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Dark foundation + Inter font */
    .stApp {
        background-color: #0D0D0D;
        font-family: 'Inter', sans-serif !important;
    }
    html, body {
        font-family: 'Inter', sans-serif;
    }

    /* === FULL HEIGHT CHAT LAYOUT === */
    section.main > div.block-container {
        max-width: 100% !important;
        padding: 0 2rem 120px 2rem !important;
    }

    /* Chat messages container -- full viewport height */
    div[data-testid='stChatMessageContainer'] {
        height: calc(100vh - 200px) !important;
        max-height: calc(100vh - 200px) !important;
        overflow-y: auto !important;
        padding: 1rem !important;
        scroll-behavior: smooth !important;
    }

    /* Chat input pinned to bottom */
    div[data-testid='stBottom'] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: #0D0D0D !important;
        border-top: 2px solid #FFE600 !important;
        padding: 12px 2rem !important;
        z-index: 999 !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        background: #0D0D0D !important;
        padding: 10px 20px !important;
        border-top: 2px solid #FFE600 !important;
    }

    /* Chat input wider and styled */
    div[data-testid='stChatInput'] {
        max-width: 100% !important;
    }
    div[data-testid='stChatInput'] textarea,
    div[data-testid='stChatInput'] div[contenteditable='true'] {
        background: #0d1117 !important;
        border: 2px solid #FFE600 !important;
        color: #ffffff !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 15px !important;
        min-height: 50px !important;
        border-radius: 8px !important;
    }
    div[data-testid='stChatInput'] textarea::placeholder {
        color: #808080 !important;
        -webkit-text-fill-color: #808080 !important;
    }

    /* Assistant messages -- wider, more readable */
    div[data-testid='stChatMessage'] {
        max-width: 90% !important;
        padding: 12px 16px !important;
        margin-bottom: 8px !important;
        line-height: 1.7 !important;
        font-size: 15px !important;
    }

    /* Scrollbar styling */
    div[data-testid='stChatMessageContainer']::-webkit-scrollbar {
        width: 6px;
    }
    div[data-testid='stChatMessageContainer']::-webkit-scrollbar-thumb {
        background: #4a4a4a;
        border-radius: 3px;
    }
    div[data-testid='stChatMessageContainer']::-webkit-scrollbar-track {
        background: #0D0D0D;
    }

    /* Download buttons inside chat */
    div[data-testid='stChatMessage'] .stDownloadButton button {
        background: #FFE600 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        margin-top: 6px !important;
        font-size: 13px !important;
    }

    /* Status containers inside chat */
    div[data-testid='stChatMessage'] div[data-testid='stStatusWidget'] {
        background: #2d2d2d !important;
        border: 1px solid #4a4a4a !important;
        border-radius: 8px !important;
        margin: 8px 0 !important;
    }

    /* === COMPACT HEADER BAR === */
    .top-header {
        background: linear-gradient(135deg, #0D0D0D 0%, #1A1A1A 100%);
        border-bottom: 1px solid #333333;
        padding: 6px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 50px;
        max-height: 50px;
        position: relative;
    }
    .top-header::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: #FFE600;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .top-header h1 {
        font-size: 16px;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.3px;
        display: inline;
    }
    .top-header .role {
        font-size: 11px;
        color: #B0B0B0;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline;
        margin-left: 8px;
    }

    /* === COMPACT SIDEBAR === */
    section[data-testid='stSidebar'] {
        width: 280px !important;
        min-width: 280px !important;
        background: #0D0D0D !important;
        border-right: 1px solid #333333;
    }
    section[data-testid='stSidebar'] .block-container {
        padding: 1rem !important;
    }
    section[data-testid="stSidebar"] .stFileUploader {
        border: 1px dashed #4a4a4a;
        border-radius: 8px;
        padding: 12px;
    }

    /* === CHAT PANEL BORDER === */
    div[data-testid="column"]:last-child {
        border-left: 1px solid #4a4a4a;
        padding-left: 16px !important;
    }

    /* === WORKSPACE AREA === */
    .workspace-card {
        background: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .workspace-card h3 {
        font-size: 14px;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #4a4a4a;
    }

    /* === KPI CARDS === */
    .kpi-row {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .kpi-card {
        flex: 1;
        background: linear-gradient(135deg, #2d2d2d, #3a3a3a);
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-card .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #FFDE00;
        line-height: 1.2;
    }
    .kpi-card .kpi-label {
        font-size: 10px;
        color: #B0B0B0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* === SCORE BADGE === */
    .score-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        font-size: 20px;
        font-weight: 700;
        border: 3px solid;
    }
    .score-green { color: #238636; border-color: #238636; }
    .score-amber { color: #d29922; border-color: #d29922; }
    .score-red { color: #da3633; border-color: #da3633; }

    /* === PROGRESS INDICATOR === */
    .phase-status {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        background: #3a3a3a;
        border: 1px solid #4a4a4a;
        border-left: 3px solid #FFDE00;
        border-radius: 6px;
        margin-bottom: 12px;
        font-size: 13px;
        color: #FFFFFF;
    }
    .phase-status.complete {
        border-left-color: #FFE600;
    }
    .phase-status .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #FFDE00;
        animation: pulse 1.5s infinite;
    }
    .phase-status.complete .dot {
        background: #FFE600;
        animation: none;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Chat messages custom classes */
    .dr-msg {
        background: #2d2d2d;
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13px;
        line-height: 1.6;
        color: #FFFFFF;
    }
    .user-msg {
        background: #3a3a3a;
        border: 1px solid #4a4a4a;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #B0B0B0;
        text-align: right;
    }
    .dr-name {
        font-size: 11px;
        font-weight: 600;
        color: #FFDE00;
        margin-bottom: 4px;
        letter-spacing: 0.3px;
    }

    /* Download cards */
    .dl-card {
        background: #3a3a3a;
        border: 1px solid #FFDE00;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
    }
    .dl-card .dl-name {
        font-size: 12px;
        font-weight: 600;
        color: #FFDE00;
    }
    .dl-card .dl-desc {
        font-size: 11px;
        color: #B0B0B0;
    }

    /* Download buttons -- WU branding */
    div.stDownloadButton > button {
        background-color: #FFE600 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        width: 100% !important;
        margin-top: 4px !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #FFE600 !important;
        filter: brightness(1.1);
    }

    /* Make data tables dark */
    .stDataFrame {
        border: 1px solid #4a4a4a;
        border-radius: 8px;
    }

    /* === CAPABILITY CARD BUTTONS === */
    button[kind="secondary"][data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #1A1A1A 0%, #262626 100%) !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 20px 16px !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        min-height: 60px !important;
        transition: border-color 0.2s !important;
    }
    button[kind="secondary"][data-testid="stBaseButton-secondary"]:hover {
        border-color: #FFE600 !important;
        color: #FFE600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Art of the Possible -- global design system overlay
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ============================================
# SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "workspace_content" not in st.session_state:
    st.session_state.workspace_content = {
        "phase": "waiting",
        "data_preview": None,
        "charts": [],
        "kpis": [],
        "scores": None,
        "deliverables": [],
        "progress_messages": [],
        "audit_html": None,        # NEW: audit report HTML
        "audit_summary": None,     # NEW: executive summary text
        "audit_releasable": None,  # NEW: release gate
    }

if ("agent" not in st.session_state
        or (st.session_state.agent is not None
            and not hasattr(st.session_state.agent, "chat_stream"))):
    try:
        from config.settings import require_at_least_one_engine
        available = require_at_least_one_engine()
        st.session_state.agent = DrDataAgent()
        st.session_state._engine_status = f"Engines: {', '.join(available)}"
    except RuntimeError as e:
        st.error(f"Dr. Data cannot start: {e}")
        st.session_state.agent = None
        st.session_state._engine_status = "No engines available"
    except Exception as e:
        st.error(f"Failed to initialize Dr. Data: {e}")
        st.session_state.agent = None
        st.session_state._engine_status = f"Error: {e}"

if "multi_session" not in st.session_state:
    st.session_state.multi_session = MultiFileSession()

if "file_just_uploaded" not in st.session_state:
    st.session_state.file_just_uploaded = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "audit_engine" not in st.session_state:
    st.session_state.audit_engine = AuditEngine()

if "audience_mode" not in st.session_state:
    st.session_state.audience_mode = "analyst"


# ============================================
# HEADER (with avatar)
# ============================================
_safe_html(f'<div class="top-header"><div class="header-left">{DR_DATA_AVATAR}<h1>Dr. Data</h1><span class="role">|&nbsp; Chief Data Intelligence Officer</span></div></div>', "Dr. Data -- Chief Data Intelligence Officer")


# ============================================
# SIDEBAR -- File Upload + Audience Toggle
# ============================================
with st.sidebar:
    st.markdown("**Upload Data**")
    ext_list = sorted(ALL_SUPPORTED)
    uploaded_files_list = st.file_uploader(
        "Drop files here",
        type=ext_list,
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    # Handle uploads through MultiFileSession
    if uploaded_files_list:
        # Build a fingerprint of current files: name + size
        current_fingerprint = {
            (uf.name, uf.size) for uf in uploaded_files_list
        }
        tracked_fingerprint = {
            (name, info.get("size", 0))
            for name, info in st.session_state.uploaded_files.items()
        }

        if current_fingerprint != tracked_fingerprint:
            # Files changed (added, removed, or replaced with different content)
            # Full reset to avoid stale data from previous session
            st.session_state.multi_session = MultiFileSession()
            st.session_state.uploaded_files = {}
            st.session_state.messages = []
            st.session_state.file_just_uploaded = None
            st.session_state.workspace_content = {
                "phase": "waiting",
                "data_preview": None,
                "charts": [],
                "kpis": [],
                "scores": None,
                "deliverables": [],
                "progress_messages": [],
                "audit_html": None,
                "audit_summary": None,
                "audit_releasable": None,
            }
            # Reset agent so it doesn't hold old dataframes
            try:
                st.session_state.agent = DrDataAgent()
            except Exception:
                st.session_state.agent = None
            # Reset any DQ / catalog / trust state
            for stale_key in [
                "dq_engine", "dq_cross_results", "dq_history",
                "data_catalog", "trust_scorer", "copdq",
                "rules_engine", "stewardship", "compliance",
                "incidents", "lineage", "dr_engine",
                "dr_inventory", "dr_analysis", "dr_report_html",
                "snowflake", "sf_tables", "audit_engine",
            ]:
                if stale_key in st.session_state:
                    del st.session_state[stale_key]
            # Re-init audit engine
            st.session_state.audit_engine = AuditEngine()

        session = st.session_state.multi_session
        new_files = []
        file_errors = []
        for uf in uploaded_files_list:
            if uf.name not in st.session_state.uploaded_files:
                try:
                    session.add_file(uf.name, uf)
                    file_info = session.files.get(uf.name, {})
                    st.session_state.uploaded_files[uf.name] = {
                        "path": file_info.get("path", ""),
                        "ext": file_info.get("ext", ""),
                        "size": uf.size,
                    }
                    new_files.append(uf.name)
                except Exception as e:
                    file_errors.append(f"{uf.name}: {str(e)[:100]}")

        if file_errors:
            for err in file_errors:
                st.warning(f"Could not process {err}")

        if new_files:
            # Auto-unify if 2+ data files
            if len(session.data_files) >= 2:
                try:
                    unified_df, rels, log = session.auto_unify()
                    if unified_df is not None:
                        st.session_state.workspace_content["join_log"] = log
                        st.session_state.workspace_content["relationships"] = rels
                except Exception:
                    pass

            try:
                primary_df = session.get_primary_dataframe()
            except Exception:
                primary_df = None

            if primary_df is not None:
                st.session_state.workspace_content["data_preview"] = primary_df

                # --- AUDIT on upload ---
                try:
                    audit = st.session_state.audit_engine.audit_dataframe(
                        primary_df, source_name=", ".join(new_files)
                    )
                    audit.compute_scores()
                    ws = st.session_state.workspace_content
                    ws["audit_html"] = audit.to_html(
                        audience=st.session_state.audience_mode
                    )
                    ws["audit_summary"] = audit.to_executive_summary()
                    ws["audit_releasable"] = audit.is_releasable
                except Exception:
                    pass

            if st.session_state.agent:
                try:
                    st.session_state.agent.set_session(session)
                    print(f"[UPLOAD] set_session OK -- "
                          f"dataframe={'YES' if st.session_state.agent.dataframe is not None else 'NO'}, "
                          f"tableau_spec={'YES' if st.session_state.agent.tableau_spec else 'NO'}, "
                          f"session.tableau_spec={'YES' if session.tableau_spec else 'NO'}")
                except Exception as _set_err:
                    print(f"[UPLOAD] set_session FAILED: {_set_err}")
                    import traceback as _tb_upload
                    _tb_upload.print_exc()
            else:
                print(f"[UPLOAD] No agent in session_state! Cannot set_session.")

            st.session_state.file_just_uploaded = {"names": new_files}

    # If uploader is empty but we have tracked files, full session wipe
    if not uploaded_files_list and st.session_state.uploaded_files:
        next_key = st.session_state.get("uploader_key", 0) + 1
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.uploader_key = next_key
        st.rerun()

    # Show uploaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("**Session Files**")
        for name, info in st.session_state.uploaded_files.items():
            size_str = (
                f"{info['size']/1024:.0f} KB"
                if info["size"] < 1024 * 1024
                else f"{info['size']/1024/1024:.1f} MB"
            )
            st.markdown(f"**{name}** ({size_str})")

    # --- AUDIENCE TOGGLE ---
    st.markdown("---")
    st.markdown("**Report Audience**")
    audience = st.radio(
        "Detail level",
        ["analyst", "executive"],
        index=0 if st.session_state.audience_mode == "analyst" else 1,
        format_func=lambda x: "Analyst -- full technical detail" if x == "analyst" else "Executive -- summary + decisions",
        label_visibility="collapsed",
    )
    if audience != st.session_state.audience_mode:
        st.session_state.audience_mode = audience
        # Re-render audit if we have data
        ws = st.session_state.workspace_content
        if ws.get("data_preview") is not None:
            audit = st.session_state.audit_engine.audit_dataframe(
                ws["data_preview"],
                source_name="session data"
            )
            audit.compute_scores()
            ws["audit_html"] = audit.to_html(audience=audience)
            ws["audit_summary"] = audit.to_executive_summary()

    # Deliverables in sidebar
    ws = st.session_state.workspace_content
    if ws["deliverables"]:
        st.markdown("---")
        st.markdown("**Downloads**")
        for idx, dl in enumerate(ws["deliverables"]):
            if os.path.exists(dl["path"]):
                with open(dl["path"], "rb") as f:
                    st.download_button(
                        label=dl["name"],
                        data=f.read(),
                        file_name=dl["filename"],
                        key=f"sb_dl_{idx}_{dl['filename']}",
                    )

    # ---- Generated Files ----
    _output_dir = None
    for _candidate in ["/mount/src/dr-data/output", os.path.join(os.path.dirname(__file__), "..", "output")]:
        if os.path.isdir(_candidate):
            _output_dir = _candidate
            break
    if _output_dir:
        _gen_files = [f for f in os.listdir(_output_dir) if os.path.isfile(os.path.join(_output_dir, f))]
        if _gen_files:
            st.markdown("---")
            with st.expander("Generated Files", expanded=False):
                _mime_map = {
                    ".pdf": "application/pdf",
                    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    ".csv": "text/csv",
                    ".json": "application/json",
                    ".html": "text/html",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".svg": "image/svg+xml",
                    ".txt": "text/plain",
                    ".pbix": "application/octet-stream",
                }
                for _gf in sorted(_gen_files):
                    _gf_path = os.path.join(_output_dir, _gf)
                    _ext = os.path.splitext(_gf)[1].lower()
                    _mime = _mime_map.get(_ext, "application/octet-stream")
                    try:
                        with open(_gf_path, "rb") as _fh:
                            st.download_button(
                                label=_gf,
                                data=_fh.read(),
                                file_name=_gf,
                                mime=_mime,
                                key=f"dl_{_gf}",
                            )
                    except Exception:
                        pass

    # ---- Snowflake Connection Panel ----
    st.markdown("---")
    st.markdown("**Snowflake Connection**")

    if "snowflake" not in st.session_state:
        st.session_state.snowflake = None
    if "sf_tables" not in st.session_state:
        st.session_state.sf_tables = []

    if st.session_state.snowflake is None:
        if st.button("Connect to Snowflake", key="sf_connect"):
            with st.status("Connecting...", expanded=True) as sf_status:
                from core.snowflake_connector import SnowflakeConnector
                sf = SnowflakeConnector()
                sf_status.write("Reaching Snowflake...")
                if sf.connect():
                    st.session_state.snowflake = sf
                    sf_status.write("Connected. Loading tables...")
                    try:
                        tables = sf.get_sample_data_tables()
                        st.session_state.sf_tables = (
                            tables if tables else sf.list_tables()
                        )
                        sf_status.update(
                            label="Connected", state="complete"
                        )
                    except Exception:
                        st.session_state.sf_tables = sf.list_tables()
                        sf_status.update(
                            label="Connected", state="complete"
                        )
                    st.rerun()
                else:
                    sf_status.update(
                        label="Connection failed - check .env credentials",
                        state="error",
                    )
    else:
        st.success("Snowflake connected")

        # Table selector
        tables = st.session_state.sf_tables
        if tables:
            if isinstance(tables, dict):
                table_list = list(tables.keys())
            elif isinstance(tables, list):
                table_list = tables
            else:
                table_list = []

            selected_tables = st.multiselect(
                "Select tables to load",
                table_list,
                key="sf_table_select",
            )

            row_limit = st.slider(
                "Row limit per table",
                1000, 100000, 50000, step=5000,
                key="sf_row_limit",
            )

            if st.button("Load selected tables", key="sf_load"):
                with st.status(
                    "Loading tables...", expanded=True
                ) as load_status:
                    sf = st.session_state.snowflake
                    loaded_count = 0
                    for tbl_name in selected_tables:
                        try:
                            load_status.write(f"Loading {tbl_name}...")
                            df = sf.table_to_df(tbl_name, limit=row_limit)
                            if df is not None and len(df) > 0:
                                load_status.write(
                                    f"{tbl_name}: {len(df)} rows x "
                                    f"{len(df.columns)} columns"
                                )
                                # Add to multi-file session
                                ms = st.session_state.multi_session
                                if ms:
                                    ms.add_dataframe(tbl_name, df)

                                # Set on agent
                                agent = st.session_state.get("agent")
                                if agent:
                                    if loaded_count == 0:
                                        agent.dataframe = df
                                        agent.current_file_name = tbl_name
                                    if not hasattr(agent, "snowflake_tables"):
                                        agent.snowflake_tables = {}
                                    agent.snowflake_tables[tbl_name] = df
                                    agent.snowflake_config = {
                                        "account": sf.account,
                                        "warehouse": sf.warehouse,
                                        "database": sf.database,
                                        "schema": sf.schema,
                                    }
                                loaded_count += 1
                        except Exception as e:
                            load_status.write(
                                f"Error loading {tbl_name}: "
                                f"{str(e)[:100]}"
                            )

                    if loaded_count > 0:
                        # Auto-detect relationships for TPCH tables
                        ms = st.session_state.multi_session
                        if ms and len(ms.data_files) >= 2:
                            try:
                                unified_df, rels, log = ms.auto_unify()
                                if rels:
                                    ws = st.session_state.workspace_content
                                    ws["relationships"] = rels
                                    ws["join_log"] = log
                            except Exception:
                                pass

                        load_status.update(
                            label=f"Loaded {loaded_count} table(s)",
                            state="complete",
                        )
                        st.rerun()
                    else:
                        load_status.update(
                            label="No tables loaded", state="error"
                        )

        # Custom SQL
        with st.expander("Custom SQL Query"):
            sql = st.text_area("Enter SQL:", height=80, key="sf_sql")
            if st.button("Run Query", key="sf_run_sql"):
                if sql.strip():
                    with st.spinner("Running..."):
                        try:
                            df = st.session_state.snowflake.query_to_df(sql)
                            if df is not None:
                                st.success(f"{len(df)} rows returned")
                                agent = st.session_state.get("agent")
                                if agent:
                                    agent.dataframe = df
                                    agent.current_file_name = "SQL Query Result"
                                st.rerun()
                        except Exception as e:
                            st.error(f"SQL error: {str(e)[:200]}")

        if st.button("Disconnect", key="sf_disconnect"):
            st.session_state.snowflake.disconnect()
            st.session_state.snowflake = None
            st.session_state.sf_tables = []
            st.rerun()

    st.markdown("---")
    col_reset, col_new = st.columns(2)
    with col_reset:
        if st.button("Reset Session", width="stretch"):
            # Keep the uploader key the same so the file_uploader widget
            # retains its files -- they'll be re-ingested into a fresh
            # MultiFileSession since uploaded_files dict is cleared.
            cur_key = st.session_state.get("uploader_key", 0)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.uploader_key = cur_key
            st.rerun()
    with col_new:
        if st.button("New Session", width="stretch"):
            next_key = st.session_state.get("uploader_key", 0) + 1
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.uploader_key = next_key
            st.rerun()




# ============================================
# MIME types for download buttons
# ============================================
_MIME_TYPES = {
    ".html": "text/html",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pbip": "application/zip",
    ".zip": "application/zip",
    ".csv": "text/csv",
}


def _render_downloads(downloads, key_prefix, ts):
    """Render download buttons inside the current container."""
    if not downloads:
        return
    st.markdown("---")
    dl_cols = st.columns(min(len(downloads), 3))
    for idx, dl in enumerate(downloads):
        file_path = dl.get("path", "")
        file_name = dl.get("filename", os.path.basename(file_path) if file_path else "file")
        with dl_cols[idx % len(dl_cols)]:
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                ext = os.path.splitext(file_name)[1].lower()
                mime = _MIME_TYPES.get(ext, "application/octet-stream")
                st.download_button(
                    label=f"Download {file_name}",
                    data=file_bytes,
                    file_name=file_name,
                    mime=mime,
                    key=f"{key_prefix}_{idx}_{file_name}_{ts}",
                )
            else:
                st.caption(f"(file expired: {file_name})")



# ============================================
# MAIN LAYOUT: Tabs + Workspace/Chat split
# ============================================
tab1, tab2, tab3, tab4 = st.tabs(["Dr. Data Agent", "Data Quality Engine", "Dashboard Rationalization", "Data Modeler"])

with tab1:
    workspace_col, chat_col = st.columns([65, 35])


    # ============================================
    # LEFT: WORKSPACE -- Shows Dr. Data's work
    # ============================================
    with workspace_col:
        ws = st.session_state.workspace_content

        # === EXPORT EXECUTION (runs in workspace with visible progress) ===
        if st.session_state.get("pending_export"):
            _export_req = st.session_state.pop("pending_export")
            _export_prompt = _export_req["prompt"]
            _export_ts = _export_req["timestamp"]
            _agent = st.session_state.agent

            ws["phase"] = "building"
            ws["progress_messages"].append(
                f"Building: {_export_prompt[:80]}..."
                if len(_export_prompt) > 80 else f"Building: {_export_prompt}"
            )

            if _agent:
                with st.status("Building your deliverables...", expanded=True) as _build_status:
                    _step_counter = [0]

                    def _live_progress(msg):
                        _step_counter[0] += 1
                        _build_status.update(label=msg)
                        _build_status.write(
                            f"**Step {_step_counter[0]}:** {msg}"
                        )

                    _live_progress("Starting generation pipeline...")
                    _response = None
                    try:
                        _response = _agent.respond(
                            _export_prompt,
                            st.session_state.messages,
                            st.session_state.uploaded_files,
                            progress_callback=_live_progress,
                        )

                        if _response is None:
                            _build_status.update(label="Build failed", state="error")
                            ws["progress_messages"].append("Build failed")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Something went wrong. Try again.",
                                "timestamp": _export_ts,
                            })

                        elif isinstance(_response, dict):
                            _content = _response.get("content", "")
                            _downloads = _response.get("downloads", []) or []

                            if _downloads:
                                _fnames = [
                                    dl.get("filename", dl.get("name", "file"))
                                    for dl in _downloads
                                ]
                                _build_status.write(
                                    f"Generated {len(_downloads)} file(s): "
                                    + ", ".join(_fnames)
                                )
                                _build_status.update(
                                    label=f"Done -- {len(_downloads)} deliverable(s)",
                                    state="complete",
                                )

                                # Save to workspace deliverables
                                for _dl in _downloads:
                                    if os.path.exists(_dl.get("path", "")):
                                        try:
                                            _dl_audit = (
                                                st.session_state.audit_engine
                                                .audit_deliverable(
                                                    _dl["path"],
                                                    file_type=_dl.get(
                                                        "filename", ""
                                                    ).split(".")[-1],
                                                )
                                            )
                                            _dl_audit.compute_scores()
                                            _dl["audit_score"] = _dl_audit.overall_score
                                            _dl["audit_releasable"] = _dl_audit.is_releasable
                                        except Exception:
                                            pass
                                ws["deliverables"].extend(_downloads)
                                ws["phase"] = "complete"
                                ws["progress_messages"].append(
                                    "Deliverables ready -- audited"
                                )

                                _chat_note = (
                                    f"Built {len(_downloads)} deliverable(s): "
                                    f"{', '.join(_fnames)}. "
                                    f"Check the workspace for downloads."
                                )
                                if _content:
                                    _chat_note = _content + "\n\n" + _chat_note
                            else:
                                _build_status.update(
                                    label="Done", state="complete"
                                )
                                _chat_note = _content or "Done. No files generated."

                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": _chat_note,
                                "downloads": _downloads,
                                "timestamp": _export_ts,
                            })

                        elif isinstance(_response, str) and _response.strip():
                            _build_status.update(label="Done", state="complete")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": _response,
                                "timestamp": _export_ts,
                            })
                        else:
                            _build_status.update(
                                label="No output", state="error"
                            )
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "No response. Try rephrasing.",
                                "timestamp": _export_ts,
                            })

                    except Exception as _e:
                        import traceback
                        traceback.print_exc()
                        _build_status.update(label="Error", state="error")
                        ws["progress_messages"].append("Build error")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Hit a snag: {str(_e)[:200]}",
                            "timestamp": _export_ts,
                        })

                    # Update data preview if needed
                    if _agent.dataframe is not None:
                        if ws.get("data_preview") is None:
                            ws["data_preview"] = _agent.dataframe

            st.rerun()  # refresh to show results in chat + workspace

        if ws["phase"] == "waiting" and ws.get("data_preview") is None:
            # Empty state -- clickable capability cards
            _safe_html(
                '<div style="text-align:center;padding:32px 0 8px 0;">'
                '<div style="font-size:18px;font-weight:600;color:#FFFFFF;margin-bottom:4px;">'
                'This is Creative Engineering!</div>'
                '<div style="font-size:13px;color:#808080;margin-bottom:24px;">'
                'Upload data in the sidebar, or click a card to get started</div>'
                '</div>',
                "This is Creative Engineering! -- Upload data or click a card to get started."
            )

            _capabilities = [
                ("Interactive Dashboard", "dashboard",
                 "HTML dashboard with charts, KPIs, and filters"),
                ("Power BI Project", "powerbi",
                 "PBIP with semantic model, DAX measures, themed visuals"),
                ("Reports & Exports", "reports",
                 "PDF, PowerPoint, Word, Excel -- any audience"),
                ("Tableau Migration", "tableau",
                 "Convert .twb/.twbx workbooks into Power BI"),
                ("Live Data", "data_connection",
                 "Connect to Snowflake and build from warehouse data"),
                ("Data Audit", "audit",
                 "Quality checks, completeness scores, release gates"),
            ]

            _row1 = st.columns(3)
            for i, (title, key, desc) in enumerate(_capabilities[:3]):
                with _row1[i]:
                    if st.button(title, key=f"cap_{key}", use_container_width=True):
                        st.session_state.user_interest = key
                        st.rerun()
                    st.caption(desc)

            _row2 = st.columns(3)
            for i, (title, key, desc) in enumerate(_capabilities[3:]):
                with _row2[i]:
                    if st.button(title, key=f"cap_{key}", use_container_width=True):
                        st.session_state.user_interest = key
                        st.rerun()
                    st.caption(desc)

        # === RECENT WORK (always visible if history exists) ===
        _recent = _get_recent_deliverables(6)
        if _recent:
            st.markdown("---")
            _safe_html(
                '<div style="font-size:15px;font-weight:600;color:#FFFFFF;'
                'margin-bottom:12px;">Recent Work</div>',
                "**Recent Work**"
            )
            _type_icons = {
                "dashboard": "&#9783;",
                "powerbi": "&#9638;",
                "pptx": "&#9776;",
                "pdf": "&#9776;",
                "docx": "&#9776;",
                "other": "&#9679;",
            }
            _vis_count = min(len(_recent), 6)
            _rw_cols = st.columns(min(_vis_count, 3))
            for _ri, _rec in enumerate(_recent[:6]):
                if _ri < 6:
                    with _rw_cols[_ri % 3]:
                        _icon = _type_icons.get(_rec.get("type", ""), "&#9679;")
                        _rdate = _rec.get("created_at", "")[:10]
                        _safe_html(
                            f'<div style="background:#1A1A1A;border:1px solid #333;'
                            f'border-radius:8px;padding:12px;margin-bottom:8px;">'
                            f'<div style="color:#FFE600;font-size:18px;">{_icon}</div>'
                            f'<div style="color:#FFF;font-size:13px;font-weight:600;'
                            f'margin:4px 0 2px 0;">{html_module.escape(_rec.get("name", "Untitled"))}</div>'
                            f'<div style="color:#808080;font-size:11px;">'
                            f'{html_module.escape(_rec.get("type", "").upper())} | '
                            f'{html_module.escape(_rec.get("source_file", "")[:30])} | '
                            f'{_rdate}</div>'
                            f'<div style="color:#666;font-size:11px;margin-top:4px;">'
                            f'{html_module.escape(_rec.get("description", "")[:80])}</div>'
                            f'</div>',
                            f'{_rec.get("name", "")} - {_rec.get("type", "")}'
                        )
                        _rpath = _rec.get("file_path", "")
                        if _rpath and os.path.exists(_rpath):
                            with open(_rpath, "rb") as _rf:
                                st.download_button(
                                    label="Download",
                                    data=_rf.read(),
                                    file_name=os.path.basename(_rpath),
                                    key=f"rw_dl_{_rec.get('id', _ri)}",
                                    use_container_width=True,
                                )

            if len(_recent) > 6:
                with st.expander("Show all"):
                    for _rec in _recent[6:]:
                        st.markdown(
                            f"**{_rec.get('name', '')}** -- "
                            f"{_rec.get('type', '').upper()} | "
                            f"{_rec.get('created_at', '')[:10]}"
                        )

        if ws["phase"] != "waiting" or ws.get("data_preview") is not None:
            # === KPI CARDS ===
            if ws["kpis"]:
                kpi_cols = st.columns(len(ws["kpis"]))
                for col, kpi in zip(kpi_cols, ws["kpis"]):
                    with col:
                        color = kpi.get("color", "#FFDE00")
                        _safe_html(f'<div class="kpi-card"><div class="kpi-value" style="color:{color};">{html_module.escape(str(kpi["value"]))}</div><div class="kpi-label">{html_module.escape(str(kpi["label"]))}</div></div>', f'{kpi["value"]} -- {kpi["label"]}')

            # === PROGRESS MESSAGES ===
            if ws["progress_messages"]:
                for i, msg in enumerate(ws["progress_messages"]):
                    is_last = i == len(ws["progress_messages"]) - 1
                    status_class = "" if is_last else "complete"
                    _safe_html(f'<div class="phase-status {status_class}"><div class="dot"></div>{html_module.escape(str(msg))}</div>', str(msg))

            # === DATA QUALITY AUDIT (shows BEFORE data preview) ===
            if ws.get("audit_html"):
                _safe_html('<div class="workspace-card"><h3>Data Quality Audit</h3></div>', "**Data Quality Audit**")
                components.html(ws["audit_html"], height=600, scrolling=True)

                # Executive summary always accessible
                if st.session_state.audience_mode == "analyst" and ws.get("audit_summary"):
                    with st.expander("Executive Summary"):
                        st.write(ws["audit_summary"])
                elif st.session_state.audience_mode == "executive" and ws.get("audit_summary"):
                    st.info(ws["audit_summary"])

            # === DATA PREVIEW ===
            if ws.get("data_preview") is not None:
                df_preview = ws["data_preview"]
                _safe_html('<div class="workspace-card"><h3>Data Preview</h3></div>', "**Data Preview**")
                st.dataframe(
                    df_preview.head(20),
                    width="stretch",
                    height=250,
                )
                st.caption(f"{len(df_preview):,} rows x {len(df_preview.columns)} columns")

            # === SCORES ===
            if ws["scores"]:
                scores = ws["scores"]
                _safe_html('<div class="workspace-card"><h3>Quality Scorecard</h3></div>', "**Quality Scorecard**")
                score_cols = st.columns(5)
                labels = [
                    ("Overall", "total_score"),
                    ("Data", "data_accuracy"),
                    ("Measures", "measure_quality"),
                    ("Visuals", "visual_effectiveness"),
                    ("UX", "user_experience"),
                ]
                for col, (label, key) in zip(score_cols, labels):
                    val = scores.get(key, 0)
                    css_class = (
                        "score-green" if val >= 80
                        else "score-amber" if val >= 60
                        else "score-red"
                    )
                    with col:
                        _safe_html(f'<div style="text-align:center;"><div class="score-badge {css_class}">{val}</div><div style="font-size:11px;color:#B0B0B0;margin-top:6px;">{label}</div></div>', f'{label}: {val}')

            # === DELIVERABLES (with audit gate) ===
            if ws["deliverables"]:
                _safe_html('<div class="workspace-card"><h3>Your Deliverables</h3></div>', "**Your Deliverables**")

                # Audit gate message
                if ws.get("audit_releasable") is False:
                    st.warning(
                        "Data quality audit flagged issues. "
                        "Review the audit findings above before distributing deliverables."
                    )

                for idx, dl in enumerate(ws["deliverables"]):
                    dl_col1, dl_col2 = st.columns([3, 1])
                    with dl_col1:
                        _safe_html(f'<div class="dl-card"><div class="dl-name">{html_module.escape(dl["name"])}</div><div class="dl-desc">{html_module.escape(dl.get("description", ""))}</div></div>', f'{dl["name"]}: {dl.get("description", "")}')
                    with dl_col2:
                        if os.path.exists(dl["path"]):
                            # Audit each deliverable file
                            dl_audit = st.session_state.audit_engine.audit_deliverable(
                                dl["path"],
                                file_type=dl.get("filename", "").split(".")[-1]
                            )
                            dl_audit.compute_scores()

                            with open(dl["path"], "rb") as f:
                                label = "Download"
                                if not dl_audit.is_releasable:
                                    label = "Download (review flagged)"
                                st.download_button(
                                    label=label,
                                    data=f.read(),
                                    file_name=dl["filename"],
                                    key=f"ws_dl_{idx}_{dl['filename']}",
                                )


    # ============================================
    # MIME types for download buttons
    # ============================================


    # ============================================
    # RIGHT: CHAT PANEL -- Conversation with Dr. Data
    # ============================================
    with chat_col:
        _safe_html(f'<div style="padding:4px 0 8px 0;border-bottom:1px solid #4a4a4a;margin-bottom:8px;display:flex;align-items:center;gap:8px;">{DR_DATA_AVATAR}<span style="font-size:13px;font-weight:600;color:#FFFFFF;">Chat with Dr. Data</span><span style="font-size:11px;color:#B0B0B0;">|&nbsp; This is Creative Engineering!</span></div>', "Chat with Dr. Data -- This is Creative Engineering!")

        # Chat container with scroll
        chat_container = st.container()

        with chat_container:
            # Opening message if empty
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown(
                        "Hello! Great to have you here. I am Dr. Data, your personal "
                        "data intelligence partner. I believe in **This is Creative Engineering!** "
                        "-- every dataset has a story waiting to be told, and I am here "
                        "to help you tell it.\n\n"
                        "Upload a file in the sidebar (CSV, Excel, Tableau, Alteryx, or "
                        "anything else you have) and tell me what you need. I can build "
                        "interactive dashboards, Power BI projects, PDF reports, PowerPoint "
                        "presentations, and more. Whatever helps you shine -- I have got "
                        "you covered. Let us make something great together!"
                    )

            # Render full chat history on every rerender
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # === HANDLE FILE UPLOAD -- instant acknowledgment, no blocking LLM call ===
        if st.session_state.file_just_uploaded:
            file_info = st.session_state.file_just_uploaded
            st.session_state.file_just_uploaded = None

            names = file_info.get("names", [])
            name_str = ", ".join(names)

            ws = st.session_state.workspace_content
            ws["phase"] = "analyzed"
            ws["progress_messages"].append(f"Loaded {name_str}")

            # Build a quick summary from data already profiled in sidebar
            preview_note = ""
            if ws.get("data_preview") is not None:
                df = ws["data_preview"]
                cols = ", ".join(df.columns[:6].tolist())
                more = "..." if len(df.columns) > 6 else ""
                preview_note = (
                    f" I can see {len(df):,} rows and "
                    f"{len(df.columns)} columns: {cols}{more}."
                )

            audit_note = ""
            if ws.get("audit_releasable") is True:
                audit_note = " Data quality audit passed."
                ws["progress_messages"].append("Audit passed")
            elif ws.get("audit_releasable") is False:
                audit_note = " Data quality audit flagged some issues -- check the workspace."
                ws["progress_messages"].append("Audit flagged issues")

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Loaded {name_str}.{preview_note}{audit_note} "
                    f"What would you like me to build? I can create dashboards, "
                    f"Power BI projects, reports, presentations -- whatever you need."
                ),
                "timestamp": time.time(),
            })

            st.rerun()

        # === HANDLE CAPABILITY CARD CLICKS ===
        if st.session_state.get("user_interest"):
            _interest = st.session_state.pop("user_interest")
            _agent = st.session_state.agent
            if _agent:
                # Build context based on whether data is loaded
                if _agent.dataframe is not None:
                    _fname = getattr(_agent, "current_file_name", "their data")
                    _rows = len(_agent.dataframe)
                    _cols = len(_agent.dataframe.columns)
                    _context = (
                        f"The user just clicked on the {_interest} capability card. "
                        f"They have {_fname} loaded with {_rows} rows and {_cols} columns. "
                        f"Respond naturally -- suggest what you would build and why, "
                        f"based on what you see in their data. If it makes sense, "
                        f"just start building it."
                    )
                else:
                    _context = (
                        f"The user just clicked on the {_interest} capability card. "
                        f"They have no data loaded yet. Guide them naturally."
                    )

                with chat_container:
                    with st.chat_message("assistant"):
                        try:
                            _enriched = _agent._build_context_message(_context)
                            _full = st.write_stream(_agent.chat_stream(_enriched))
                        except (AttributeError, Exception):
                            _resp = _agent.respond(
                                _context,
                                st.session_state.messages,
                                st.session_state.uploaded_files,
                            )
                            _full = (
                                _resp.get("content", "")
                                if isinstance(_resp, dict) else str(_resp or "")
                            )
                            st.markdown(_full)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": _full or "",
                            "timestamp": time.time(),
                        })

        # === CHAT INPUT (always active, never disabled) ===
        if prompt := st.chat_input("Ask Dr. Data anything -- I am here to help!", key="chat_input"):
            now = time.time()

            # Append user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": now,
            })

            if not st.session_state.agent:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "I am just warming up my analysis engines -- almost "
                        "ready! Give me one moment and try again."
                    ),
                    "timestamp": now,
                })
                st.rerun()

            agent = st.session_state.agent

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):

                    # -- LLM-powered intent classification --
                    from core.intent_classifier import classify, BUILD_INTENTS

                    # Build conversation context for the classifier
                    _recent = st.session_state.messages[-4:] if st.session_state.messages else []
                    _ctx_lines = []
                    for _rm in _recent:
                        role = _rm.get("role", "?")
                        content = str(_rm.get("content", ""))[:150]
                        has_dl = " [BUILT FILES]" if _rm.get("downloads") else ""
                        _ctx_lines.append(f"{role}: {content}{has_dl}")
                    _conv_ctx = "\n".join(_ctx_lines) if _ctx_lines else "(first message)"

                    _intent = classify(prompt, _conv_ctx)
                    is_export = _intent["intent"] in BUILD_INTENTS
                    print(f"[INTENT] '{prompt[:60]}...' -> {_intent['intent']} "
                          f"(confidence={_intent['confidence']})")

                    if is_export:
                        # ====== EXPORT PATH -- hand off to workspace ======
                        st.session_state.pending_export = {
                            "prompt": prompt,
                            "timestamp": now,
                        }
                        st.rerun()  # workspace will pick this up

                    else:
                        # ====== CHAT PATH -- streaming, no spinner ======

                        # Auto-load data if the agent hasn't ingested it yet
                        if agent.dataframe is None:
                            agent._try_recover_data()

                        if st.session_state.uploaded_files and agent.dataframe is None:
                            for name, info in st.session_state.uploaded_files.items():
                                path = info.get("path", "")
                                if not path or not os.path.exists(path):
                                    continue
                                ext = info.get("ext", path.rsplit(".", 1)[-1].lower())
                                # Skip structure-only files
                                if ext in ("twb", "twbx", "wid", "yxmd", "yxwz", "yxmc", "yxzp"):
                                    continue
                                try:
                                    if ext == "csv":
                                        agent.dataframe = pd.read_csv(path)
                                    elif ext in ("xlsx", "xls"):
                                        from app.file_handler import load_excel_smart
                                        agent.dataframe, agent.sheet_name = load_excel_smart(path)
                                    elif ext == "parquet":
                                        agent.dataframe = pd.read_parquet(path)
                                    elif ext == "json":
                                        agent.dataframe = pd.read_json(path)
                                    if agent.dataframe is not None:
                                        agent.data_file_path = path
                                        agent.data_path = path
                                        if agent.data_profile is None:
                                            agent.data_profile = agent.analyzer.profile(agent.dataframe)
                                        break
                                except Exception:
                                    pass

                        # Build context and stream response word-by-word
                        enriched = agent._build_context_message(prompt)
                        try:
                            full_text = st.write_stream(agent.chat_stream(enriched))
                        except AttributeError:
                            # Fallback: blocking call if streaming unavailable
                            response = agent.respond(
                                prompt, st.session_state.messages,
                                st.session_state.uploaded_files,
                            )
                            full_text = (
                                response.get("content", "")
                                if isinstance(response, dict) else str(response or "")
                            )
                            st.markdown(full_text)

                        # Check for any files generated during tool calls
                        chat_downloads = []
                        for fpath in agent.generated_files:
                            if os.path.exists(fpath):
                                fname = os.path.basename(fpath)
                                chat_downloads.append({
                                    "name": fname,
                                    "filename": fname,
                                    "path": fpath,
                                    "description": "Generated file",
                                })
                        if chat_downloads:
                            # Notify in chat -- downloads go to workspace
                            file_names = [dl["filename"] for dl in chat_downloads]
                            st.markdown(
                                f"Built {len(chat_downloads)} deliverable(s): "
                                f"{', '.join(file_names)}. "
                                f"Check the workspace for downloads."
                            )

                        # Save chat response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_text or "",
                            "downloads": chat_downloads,
                            "timestamp": now,
                        })

                        ws = st.session_state.workspace_content
                        if chat_downloads:
                            ws["deliverables"].extend(chat_downloads)
                            ws["phase"] = "complete"
                        if agent.dataframe is not None:
                            if ws.get("data_preview") is None:
                                ws["data_preview"] = agent.dataframe

    # ============================================
    # TAB 1 -- Migration Intelligence Chatbot
    # ============================================
    st.markdown("---")
    _safe_html(tab_header(
        "Migration Intelligence Chat",
        "DAX -- Tableau -- Power BI -- Snowflake -- Formula Transpilation",
        accent="#00D47E",
        icon="//",
    ), "Migration Intelligence Chat")

    def _migration_context():
        ctx_parts = []
        agent = st.session_state.get("agent")
        if agent and hasattr(agent, "tableau_spec") and agent.tableau_spec:
            spec = agent.tableau_spec
            ctx_parts.append(f"Tableau workbook loaded")
            ctx_parts.append(f"Worksheets: {len(spec.get('worksheets', []))}")
            ctx_parts.append(f"Calculated fields: {len(spec.get('calculated_fields', []))}")
            calcs = spec.get("calculated_fields", [])[:3]
            if calcs:
                ctx_parts.append("Sample calculated fields:")
                for c in calcs:
                    ctx_parts.append(f"  [{c.get('name','')}]: {c.get('formula','')[:80]}")
        return "\n".join(ctx_parts) if ctx_parts else None

    render_tab_chatbot(
        tab_key="migration",
        system_prompt=MIGRATION_EXPERT_PROMPT,
        placeholder="Ask about DAX translations, LOD expressions, Tableau to PBI migration patterns...",
        accent_color="#00D47E",
        tab_label="Migration Intelligence",
        context_injector=_migration_context,
        height=420,
    )


with tab2:
    # ── DQ Engine session state ──
    if "dq_engine" not in st.session_state:
        st.session_state.dq_engine = DataQualityEngine()

    _dq = st.session_state.dq_engine

    # Collect available tables from all sources
    _dq_tables = {}
    if "agent" in st.session_state and st.session_state.agent:
        _dq_agent = st.session_state.agent
        if hasattr(_dq_agent, "dataframe") and _dq_agent.dataframe is not None:
            _dq_name = getattr(_dq_agent, "current_file_name", "Uploaded File")
            _dq_tables[_dq_name] = _dq_agent.dataframe
        if hasattr(_dq_agent, "loaded_tables"):
            for _tn, _td in _dq_agent.loaded_tables.items():
                _dq_tables[_tn] = _td
        if hasattr(_dq_agent, "snowflake_tables"):
            for _tn, _td in _dq_agent.snowflake_tables.items():
                _dq_tables[_tn] = _td

    # Also check multi-file session (data_files is a list of dicts)
    if "multi_session" in st.session_state:
        _ms = st.session_state.multi_session
        for _fi in getattr(_ms, "data_files", []):
            _fn = _fi.get("filename", "")
            if _fn and _fn not in _dq_tables and "df" in _fi:
                _dq_tables[_fn] = _fi["df"]

    if not _dq_tables:
        st.markdown("#### No Data Sources Connected")
        st.markdown(
            "Use the sidebar to either upload files or connect to a "
            "database. Then return here to run a full data quality assessment."
        )
        st.markdown(
            "**Supported sources:** CSV, Excel, Snowflake, PostgreSQL, "
            "MySQL, SQL Server, SQLite"
        )
    else:
        dq_subtab1, dq_subtab2, dq_subtab3, dq_subtab4, dq_subtab5, dq_subtab6, dq_subtab7, dq_subtab8, dq_subtab9 = st.tabs([
            "Quality Scanner",
            "Data Catalog",
            "Business Rules",
            "Trending",
            "Stewardship",
            "Compliance",
            "Incidents",
            "Observability",
            "Advanced Analytics",
        ])

        # ============================================================
        # SUBTAB 1: Quality Scanner (all existing DQ content)
        # ============================================================
        with dq_subtab1:
            # ── Header row ──
            _dqh1, _dqh2 = st.columns([3, 1])
            with _dqh1:
                st.markdown("#### Data Quality Scanner")
            with _dqh2:
                _dq_scan_all = st.button(
                    "Scan All Tables", type="primary", key="dq_scan_all")

            _dq_selected = st.multiselect(
                "Select tables to scan",
                list(_dq_tables.keys()),
                default=list(_dq_tables.keys()),
                key="dq_table_select",
            )

            # ── Threshold config ──
            with st.expander("Quality Thresholds (DAMA Defaults)"):
                _tc1, _tc2, _tc3, _tc4 = st.columns(4)
                with _tc1:
                    _dq.thresholds["completeness_warn"] = st.number_input(
                        "Completeness Warn %", value=95,
                        min_value=50, max_value=100, key="dq_comp_warn")
                    _dq.thresholds["completeness_fail"] = st.number_input(
                        "Completeness Fail %", value=80,
                        min_value=0, max_value=100, key="dq_comp_fail")
                with _tc2:
                    _dq.thresholds["uniqueness_warn"] = st.number_input(
                        "Uniqueness Warn %", value=99,
                        min_value=50, max_value=100, key="dq_uniq_warn")
                    _dq.thresholds["uniqueness_fail"] = st.number_input(
                        "Uniqueness Fail %", value=95,
                        min_value=0, max_value=100, key="dq_uniq_fail")
                with _tc3:
                    _dq.thresholds["validity_warn"] = st.number_input(
                        "Validity Warn %", value=95,
                        min_value=50, max_value=100, key="dq_val_warn")
                    _dq.thresholds["validity_fail"] = st.number_input(
                        "Validity Fail %", value=85,
                        min_value=0, max_value=100, key="dq_val_fail")
                with _tc4:
                    _dq.thresholds["freshness_hours_warn"] = st.number_input(
                        "Freshness Warn (hrs)", value=24,
                        min_value=1, max_value=720, key="dq_fresh_warn")
                    _dq.thresholds["freshness_hours_fail"] = st.number_input(
                        "Freshness Fail (hrs)", value=72,
                        min_value=1, max_value=720, key="dq_fresh_fail")

            # ── Run scan ──
            if _dq_scan_all or st.button("Scan Selected", key="dq_scan_selected"):
                _dq_to_scan = {
                    t: _dq_tables[t] for t in _dq_selected
                    if t in _dq_tables
                }
                if _dq_to_scan:
                    _scan_count = len(_dq_to_scan)
                    if _scan_count > 1:
                        _scan_bar = st.progress(0, text="Starting scan...")
                    else:
                        _scan_bar = None

                    with st.status(
                        "Dr. Data is analyzing your data...",
                        expanded=True,
                    ) as _dq_status:
                        for _tbl_idx, (_tn, _td) in enumerate(_dq_to_scan.items()):
                            if _scan_bar:
                                _scan_bar.progress(
                                    _tbl_idx / _scan_count,
                                    text=f"Scanning {_tn} ({_tbl_idx + 1}/{_scan_count})...")
                            _dq_status.write(
                                f"Profiling {_tn} - "
                                f"{len(_td):,} rows x "
                                f"{len(_td.columns)} columns")
                            try:
                                _dq.scan_table(_td, _tn)
                                _res = _dq.scan_results[_tn]
                                _dims = _res.get("dimensions", {})
                                # Show each dimension score
                                for _dim_key, _dim_label in [
                                    ("completeness", "Completeness"),
                                    ("accuracy", "Accuracy"),
                                    ("consistency", "Consistency"),
                                    ("timeliness", "Timeliness"),
                                    ("uniqueness", "Uniqueness"),
                                    ("validity", "Validity"),
                                ]:
                                    _ds = _dims.get(_dim_key, {}).get("score")
                                    if _ds is not None:
                                        _dq_status.write(
                                            f"{_dim_label}: {_ds:.1f}%")
                                    else:
                                        _dq_status.write(
                                            f"{_dim_label}: N/A")

                                _dq_status.write("Generating recommendations...")
                                # Record to DQ history
                                _dq_status.write("Recording history...")
                                if "dq_history" not in st.session_state:
                                    st.session_state.dq_history = DQHistory()
                                st.session_state.dq_history.record_scan(
                                    _tn, _res)
                                # Update data catalog
                                _dq_status.write("Updating catalog...")
                                if "data_catalog" in st.session_state:
                                    st.session_state.data_catalog.update_table_dq(
                                        _tn,
                                        _res["overall_score"],
                                        _res["scan_timestamp"],
                                    )
                                # Auto-create stewardship issues
                                _dq_status.write(
                                    "Creating stewardship issues...")
                                if "stewardship" in st.session_state:
                                    st.session_state.stewardship.create_issues_from_dq_scan(
                                        _res, _tn)
                            except Exception as _e:
                                _dq_status.write(
                                    f"{_tn}: Error - {str(_e)[:200]}")

                        if _scan_count > 1:
                            _dq_status.write(
                                "Running cross-table analysis...")
                            if _scan_bar:
                                _scan_bar.progress(
                                    0.95,
                                    text="Cross-table analysis...")
                            try:
                                _dq_cross = _dq.scan_multiple_tables(
                                    _dq_to_scan)
                                st.session_state.dq_cross_results = (
                                    _dq_cross)
                            except Exception as _e:
                                _dq_status.write(
                                    f"Cross-table error: "
                                    f"{str(_e)[:200]}")

                        if _scan_bar:
                            _scan_bar.progress(
                                1.0, text="Scan complete")

                        # Final overall score
                        _final_scores = [
                            r["overall_score"]
                            for r in _dq.scan_results.values()
                        ]
                        _avg_score = (
                            sum(_final_scores) / len(_final_scores)
                            if _final_scores else 0
                        )
                        _dq_status.update(
                            label=f"Analysis Complete - Overall "
                                  f"Score: {_avg_score:.1f}%",
                            state="complete",
                        )

                        # Bridge: pass DQ results into Dr. Data agent
                        if ("agent" in st.session_state
                                and st.session_state.agent):
                            st.session_state.agent.set_dq_results(
                                _dq.scan_results)

                    st.rerun()

            # ── Display results ──
            if _dq.scan_results:
                _dq_sc = _dq.generate_scorecard_data()

                st.markdown("---")
                st.markdown("#### Overall Data Quality Scorecard")

                _sc1, _sc2, _sc3, _sc4, _sc5 = st.columns(5)
                _sc1.metric("Overall Score",
                            f"{_dq_sc['overall_score']:.1f}/100")
                _sc2.metric("Tables Scanned", _dq_sc["tables_scanned"])
                _sc3.metric("Total Columns", _dq_sc["total_columns"])
                _sc4.metric("Critical Issues", _dq_sc["critical_issues"])
                _sc5.metric("Auto-Fixable", _dq_sc["auto_fixable_count"])

                # ── Dimension scores ──
                st.markdown("#### DAMA Dimension Scores")
                _dim_cols = st.columns(6)
                _dim_keys = [
                    "completeness", "accuracy", "consistency",
                    "timeliness", "uniqueness", "validity",
                ]
                _dim_labels = [
                    "Completeness", "Accuracy", "Consistency",
                    "Timeliness", "Uniqueness", "Validity",
                ]
                _dims = _dq_sc.get("dimension_scores", {})
                for _i, (_dk, _dl) in enumerate(zip(_dim_keys, _dim_labels)):
                    _dscore = _dims.get(_dk, 0)
                    if _dscore is None:
                        _dim_cols[_i].metric(_dl, "N/A")
                    else:
                        _dim_cols[_i].metric(_dl, f"{_dscore:.1f}%")

                # ── Per-table details ──
                if _dq.scan_results:
                    st.markdown("---")
                    st.markdown("#### Table Details")
                    _tbl_tabs = st.tabs(list(_dq.scan_results.keys()))

                    for _ttab, (_tname, _tres) in zip(
                        _tbl_tabs, _dq.scan_results.items()
                    ):
                        with _ttab:
                            _tc1, _tc2, _tc3 = st.columns(3)
                            _tc1.metric(
                                "Score",
                                f"{_tres['overall_score']:.1f}/100")
                            _tc2.metric("Rows", f"{_tres['row_count']:,}")
                            _tc3.metric("Columns", _tres["column_count"])

                            _ddims = _tres.get("dimensions", {})

                            with st.expander("Completeness Details"):
                                _comp = _ddims.get("completeness", {})
                                if _comp:
                                    st.write(
                                        f"Table Score: {_comp.get('score', 0):.1f}%"
                                        f" ({_comp.get('status', 'N/A')})")
                                    _cc = _comp.get("columns", {})
                                    if _cc:
                                        _comp_df = pd.DataFrame([
                                            {
                                                "Column": c,
                                                "Complete %": f"{v['score']:.1f}%",
                                                "Nulls": v.get("null_count", 0),
                                                "Status": v.get("status", ""),
                                            }
                                            for c, v in _cc.items()
                                        ])
                                        st.dataframe(
                                            _comp_df,
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                            with st.expander("Accuracy Details"):
                                _acc = _ddims.get("accuracy", {})
                                if _acc:
                                    st.write(
                                        f"Table Score: {_acc.get('score', 0):.1f}%"
                                        f" ({_acc.get('status', 'N/A')})")
                                    for _acol, _ainfo in _acc.get(
                                        "outliers", {}
                                    ).items():
                                        if _ainfo.get("count", 0) > 0:
                                            st.write(
                                                f"**{_acol}**: "
                                                f"{_ainfo['count']} outliers "
                                                f"({_ainfo.get('pct', 0):.1f}%)")
                                            if _ainfo.get("examples"):
                                                st.caption(
                                                    "Examples: " + ", ".join(
                                                        str(x) for x in
                                                        _ainfo["examples"][:5]))

                            with st.expander("Uniqueness Details"):
                                _uniq = _ddims.get("uniqueness", {})
                                if _uniq:
                                    st.write(
                                        f"Table Score: "
                                        f"{_uniq.get('score', 0):.1f}%"
                                        f" ({_uniq.get('status', 'N/A')})")
                                    _frd = _uniq.get(
                                        "full_row_duplicates", {})
                                    if _frd:
                                        st.write(
                                            f"Full row duplicates: "
                                            f"{_frd.get('count', 0)} "
                                            f"({_frd.get('pct', 0):.1f}%)")
                                    _pks = _uniq.get(
                                        "potential_primary_keys", [])
                                    if _pks:
                                        st.write(
                                            f"Potential primary keys: "
                                            f"{', '.join(_pks)}")

                            with st.expander("Validity Details"):
                                _val = _ddims.get("validity", {})
                                if _val:
                                    st.write(
                                        f"Table Score: "
                                        f"{_val.get('score', 0):.1f}%"
                                        f" ({_val.get('status', 'N/A')})")
                                    for _vcol, _vinfo in _val.get(
                                        "column_validity", {}
                                    ).items():
                                        if _vinfo.get("violations", 0) > 0:
                                            st.write(
                                                f"**{_vcol}**: "
                                                f"{_vinfo['violations']} "
                                                f"violations (rule: "
                                                f"{_vinfo.get('rule', 'N/A')})")

                            with st.expander("Consistency Details"):
                                _cons = _ddims.get("consistency", {})
                                if _cons:
                                    st.write(
                                        f"Table Score: "
                                        f"{_cons.get('score', 0):.1f}%"
                                        f" ({_cons.get('status', 'N/A')})")
                                    for _fcol, _finfo in _cons.get(
                                        "format_issues", {}
                                    ).items():
                                        st.write(
                                            f"**{_fcol}**: "
                                            f"{_finfo.get('conforming_pct', 0):.1f}% "
                                            f"conform to "
                                            f"{_finfo.get('dominant_pattern', '?')}")

                            with st.expander("Timeliness Details"):
                                _timed = _ddims.get("timeliness", {})
                                if _timed:
                                    _tscore = _timed.get("score")
                                    if _tscore is not None:
                                        st.write(
                                            f"Table Score: {_tscore:.1f}%"
                                            f" ({_timed.get('status', 'N/A')})")
                                        for _dcol, _dinfo in _timed.get(
                                            "datetime_columns", {}
                                        ).items():
                                            st.write(
                                                f"**{_dcol}**: most recent "
                                                f"{_dinfo.get('most_recent', '?')}"
                                                f", staleness "
                                                f"{_dinfo.get('staleness_hours', 0):.0f}h"
                                                f", {_dinfo.get('gaps_detected', 0)}"
                                                f" gap(s)")
                                    else:
                                        st.write(
                                            "No temporal columns detected "
                                            "in this table.")

                # ── Recommendations ──
                st.markdown("---")
                st.markdown("#### Recommendations & Remediation")

                _all_recs = []
                for _rn, _rr in _dq.scan_results.items():
                    for _rec in _rr.get("recommendations", []):
                        _rec["table"] = _rn
                        _all_recs.append(_rec)

                if _all_recs:
                    _pri_order = {
                        "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
                    _all_recs.sort(
                        key=lambda r: _pri_order.get(
                            r.get("priority", "LOW"), 4))

                    for _ri, _rec in enumerate(_all_recs[:20]):
                        _pri = _rec.get("priority", "LOW")
                        _pri_colors = {
                            "CRITICAL": "#da3633",
                            "HIGH": "#d29922",
                            "MEDIUM": "#FFE600",
                            "LOW": "#238636",
                        }
                        _pc = _pri_colors.get(_pri, "#808080")
                        with st.expander(
                            f"[{_pri}] {_rec.get('table', '')} -- "
                            f"{_rec.get('finding', '')[:80]}"
                        ):
                            st.write(
                                f"**Dimension:** "
                                f"{_rec.get('dimension', 'N/A')}")
                            st.write(
                                f"**Finding:** "
                                f"{_rec.get('finding', '')}")
                            st.write(
                                f"**Recommendation:** "
                                f"{_rec.get('recommendation', '')}")
                            st.write(
                                f"**Impact:** "
                                f"{_rec.get('impact', '')}")
                            if _rec.get("auto_fixable"):
                                if st.button(
                                    f"Auto-Fix: {_rec.get('fix_type', '')}",
                                    key=f"fix_{_rec.get('table')}_"
                                        f"{_rec.get('dimension')}_{_ri}",
                                ):
                                    with st.spinner("Applying fix..."):
                                        try:
                                            _fix = _dq.auto_remediate(
                                                _dq_tables[_rec["table"]],
                                                _rec["table"],
                                                fix_types=[
                                                    _rec.get("fix_type")],
                                            )
                                            if _fix and "fixed_df" in _fix:
                                                _dq_tables[_rec["table"]] = (
                                                    _fix["fixed_df"])
                                                _changes = _fix.get(
                                                    "changes_made", [])
                                                st.success(
                                                    f"Fixed. {len(_changes)} "
                                                    f"change(s) applied.")
                                        except Exception as _e:
                                            st.error(
                                                f"Fix failed: "
                                                f"{str(_e)[:200]}")
                else:
                    st.info(
                        "Run a scan to see recommendations.")

                # ── Trust Scores & Certification ──
                st.markdown("---")
                st.markdown("#### Trust Scores & Certification")

                if "trust_scorer" not in st.session_state:
                    st.session_state.trust_scorer = TrustScorer()

                _scorer = st.session_state.trust_scorer

                if _dq.scan_results:
                    _t_catalog = st.session_state.get(
                        "data_catalog", None)
                    _t_rules = st.session_state.get(
                        "rules_engine", None)
                    _t_hist = st.session_state.get(
                        "dq_history", None)

                    _trust_scores = {}
                    for _tsn, _tsr in _dq.scan_results.items():
                        _t_cat = (
                            _t_catalog.get_table_catalog(_tsn)
                            if _t_catalog else None
                        )
                        _t_rr = (
                            _t_rules.evaluation_results.get(_tsn)
                            if _t_rules else None
                        )
                        _t_hd = (
                            _t_hist.get_scan_frequency(_tsn)
                            if _t_hist else None
                        )
                        _trust_scores[_tsn] = (
                            _scorer.calculate_trust_score(
                                _tsn, dq_result=_tsr,
                                catalog_entry=_t_cat,
                                rule_results=_t_rr,
                                history_data=_t_hd,
                            )
                        )

                    _t_summary = _scorer.generate_trust_summary(
                        _trust_scores)

                    # Bridge trust scores to agent
                    if "agent" in st.session_state and st.session_state.agent and hasattr(st.session_state.agent, "set_trust_scores"):
                        st.session_state.agent.set_trust_scores(_trust_scores)

                    ts1, ts2, ts3, ts4, ts5 = st.columns(5)
                    ts1.metric(
                        "Avg Trust Score",
                        f"{_t_summary.get('avg_trust_score', 0):.1f}",
                    )
                    ts2.metric(
                        "Certified",
                        _t_summary.get("certified", 0),
                    )
                    ts3.metric(
                        "Warning",
                        _t_summary.get("warning", 0),
                    )
                    ts4.metric(
                        "Quarantined",
                        _t_summary.get("quarantined", 0),
                    )
                    ts5.metric(
                        "Tables Scored",
                        _t_summary.get("total_tables", 0),
                    )

                    _cert_icons = {
                        "Certified": "[OK]",
                        "Warning": "[!!]",
                        "Quarantined": "[XX]",
                    }
                    for _tsn, _tres in _trust_scores.items():
                        _tscore = _tres["trust_score"]
                        _tcert = _tres["recommended_certification"]
                        _ticon = _cert_icons.get(_tcert, "[--]")

                        with st.expander(
                            f"{_ticon} {_tsn}: Trust Score "
                            f"{_tscore:.1f}/100 ({_tcert})"
                        ):
                            _factors = _tres.get("factors", {})
                            _frows = []
                            for _fn, _fd in _factors.items():
                                _frows.append({
                                    "Factor": _fn.replace(
                                        "_", " ").title(),
                                    "Score": f"{_fd['score']:.1f}",
                                    "Weight": _fd["weight"],
                                    "Description": _fd["description"],
                                })
                            st.dataframe(
                                pd.DataFrame(_frows),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Apply certification
                            if _t_catalog:
                                if st.button(
                                    f"Apply {_tcert} certification "
                                    f"to catalog",
                                    key=f"trust_apply_{_tsn}",
                                ):
                                    _t_catalog.set_certification(
                                        _tsn, _tcert,
                                        certified_by="Trust Scorer",
                                        notes=(
                                            f"Auto-certified with "
                                            f"trust score {_tscore:.1f}"
                                        ),
                                    )
                                    st.success(
                                        f"{_tsn} marked as {_tcert}")

                # ── COPDQ Calculator ──
                st.markdown("---")
                st.markdown("#### Cost of Poor Data Quality (COPDQ)")

                if "copdq" not in st.session_state:
                    st.session_state.copdq = COPDQCalculator()

                _copdq = st.session_state.copdq

                if _dq.scan_results:
                    with st.expander(
                        "Cost Parameters (adjust for your organization)"
                    ):
                        _cp1, _cp2, _cp3 = st.columns(3)
                        with _cp1:
                            _analyst_rate = st.number_input(
                                "Analyst hourly rate ($)",
                                value=67, key="copdq_rate",
                            )
                            _fix_min = st.number_input(
                                "Minutes to fix per record",
                                value=5, key="copdq_fix_min",
                            )
                        with _cp2:
                            _txn_cost = st.number_input(
                                "Failed transaction cost ($)",
                                value=25, key="copdq_txn",
                            )
                            _fine_cost = st.number_input(
                                "Compliance fine per violation ($)",
                                value=1000, key="copdq_fine",
                            )
                        with _cp3:
                            _rpts_month = st.number_input(
                                "Reports per month",
                                value=50, key="copdq_reports",
                            )
                            _annual_rev = st.number_input(
                                "Annual revenue ($)",
                                value=4_100_000_000,
                                key="copdq_revenue",
                            )

                    _custom_params = {
                        "avg_analyst_hourly_rate": _analyst_rate,
                        "avg_manual_fix_minutes_per_record": _fix_min,
                        "avg_failed_transaction_cost": _txn_cost,
                        "avg_compliance_fine_per_violation": _fine_cost,
                        "reports_per_month": _rpts_month,
                        "annual_revenue": _annual_rev,
                    }

                    if st.button(
                        "Calculate COPDQ", type="primary",
                        key="copdq_calc",
                    ):
                        _copdq_result = _copdq.calculate_copdq(
                            _dq.scan_results, cost_params=_custom_params)
                        st.session_state.copdq_result = _copdq_result
                        # Bridge COPDQ to agent
                        if "agent" in st.session_state and st.session_state.agent and hasattr(st.session_state.agent, "set_copdq_result"):
                            st.session_state.agent.set_copdq_result(_copdq_result)

                    if (
                        "copdq_result" in st.session_state
                        and st.session_state.copdq_result
                    ):
                        _cr = st.session_state.copdq_result

                        co1, co2, co3 = st.columns(3)
                        co1.metric(
                            "Annual Cost of Poor Data",
                            _copdq.format_currency(
                                _cr["total_annual_cost"]),
                        )
                        co2.metric(
                            "Monthly Cost",
                            _copdq.format_currency(
                                _cr["total_monthly_cost"]),
                        )
                        co3.metric(
                            "Revenue Impact",
                            f"{_cr['revenue_impact_pct']:.3f}%",
                        )

                        st.caption(
                            _cr.get("gartner_benchmark", ""))

                        # Per-table breakdown
                        for _ctn, _ctc in _cr.get(
                            "cost_breakdown", {}
                        ).items():
                            with st.expander(
                                f"{_ctn}: "
                                f"{_copdq.format_currency(_ctc['total'])}"
                                f"/year"
                            ):
                                for _cdim, _cdi in _ctc.get(
                                    "costs", {}
                                ).items():
                                    st.write(
                                        f"**{_cdim.title()}** - "
                                        f"{_copdq.format_currency(_cdi.get('annual_cost', 0))}"
                                        f"/year"
                                    )
                                    st.caption(
                                        _cdi.get("description", ""))

                # ── Export ──
                st.markdown("---")
                _exc1, _exc2 = st.columns(2)
                with _exc1:
                    if st.button(
                        "Export Scorecard as JSON", key="dq_export_json"
                    ):
                        _sc_json = json.dumps(
                            _dq_sc, indent=2, default=str)
                        st.download_button(
                            "Download JSON", _sc_json,
                            "dq_scorecard.json", "application/json",
                            key="dq_dl_json",
                        )
                with _exc2:
                    if st.button(
                        "Ask Dr. Data to analyze DQ results",
                        key="dq_ask_agent",
                    ):
                        st.session_state.dq_analysis_requested = True
                        st.info(
                            "Switch to the Dr. Data Agent tab. "
                            "Dr. Data now has your DQ scan results in context."
                        )

                # ── HTML Scorecard Report ──
                st.markdown("---")
                if st.button(
                    "Generate DQ Scorecard Report",
                    key="dq_export_html",
                ):
                    with st.status(
                        "Generating Enterprise DQ Report...",
                        expanded=True,
                    ) as _rpt_status:
                        _rpt_status.write(
                            "Building executive summary...")
                        _rpt_status.write(
                            "Rendering dimension gauges...")
                        _rpt_status.write(
                            "Building trust score cards...")
                        _rpt_status.write(
                            "Calculating COPDQ impact...")
                        _rpt_status.write(
                            "Mapping compliance frameworks...")
                        _rpt_status.write(
                            "Generating heatmaps...")
                        _html = _dq.generate_html_scorecard(
                            catalog=st.session_state.get(
                                "data_catalog"),
                            rules_engine=st.session_state.get(
                                "rules_engine"),
                            history=st.session_state.get(
                                "dq_history"),
                            trust_scorer=st.session_state.get(
                                "trust_scorer"),
                            copdq_result=st.session_state.get(
                                "copdq_result"),
                            compliance=st.session_state.get(
                                "compliance"),
                            stewardship=st.session_state.get(
                                "stewardship"),
                            incidents=st.session_state.get(
                                "incidents"),
                        )
                        _rpt_status.update(
                            label="Report ready for download",
                            state="complete",
                        )
                    if _html:
                        st.download_button(
                            "Download HTML Scorecard", _html,
                            "dq_scorecard.html", "text/html",
                            key="dq_dl_html",
                        )
                        components.html(_html, height=800, scrolling=True)

        # ============================================================
        # SUBTAB 2: Data Catalog & Business Glossary
        # ============================================================
        with dq_subtab2:
            if "data_catalog" not in st.session_state:
                st.session_state.data_catalog = DataCatalog()

            catalog = st.session_state.data_catalog
            stats = catalog.get_catalog_stats()

            st.markdown("#### Data Catalog & Business Glossary")

            # Stats row
            cat_c1, cat_c2, cat_c3, cat_c4, cat_c5 = st.columns(5)
            cat_c1.metric("Tables Cataloged", stats.get("tables_cataloged", 0))
            cat_c2.metric("Columns Cataloged", stats.get("columns_cataloged", 0))
            cat_c3.metric("Glossary Terms", stats.get("glossary_terms", 0))
            cat_c4.metric("PII Columns", stats.get("pii_columns", 0))
            _cov = stats.get("coverage_pct", 0) or 0
            cat_c5.metric("Catalog Coverage", f"{_cov:.0f}%")

            # Auto-catalog button
            if _dq_tables:
                if st.button("Auto-Catalog All Loaded Tables", type="primary", key="cat_auto"):
                    with st.status("Cataloging your data...", expanded=True) as cat_status:
                        _total_cols = 0
                        _total_pii = 0
                        for tname, tdf in _dq_tables.items():
                            cat_status.write(f"Registering {tname}...")
                            _src = "Snowflake" if (
                                hasattr(st.session_state.get("agent", None), "snowflake_tables")
                                and tname in getattr(st.session_state.agent, "snowflake_tables", {})
                            ) else "File Upload"
                            catalog.auto_catalog_from_dataframe(tdf, tname, source_system=_src)
                            _total_cols += len(tdf.columns)
                            cat_status.write("Detecting PII columns...")
                            cat_status.write("Classifying data types...")
                            cat_status.write(f"Generating business glossary for {tname}...")
                            catalog.auto_generate_glossary(tdf, tname)
                            # Count PII
                            _tinfo = catalog.catalog.get("tables", {}).get(tname, {})
                            for _ci in _tinfo.get("columns", {}).values():
                                if _ci.get("pii"):
                                    _total_pii += 1
                        cat_status.update(
                            label=f"Cataloged {len(_dq_tables)} tables, "
                                  f"{_total_cols} columns, "
                                  f"{_total_pii} PII fields detected",
                            state="complete")
                    st.rerun()

            # Search
            search_query = st.text_input(
                "Search catalog...", key="cat_search",
                placeholder="Search tables, columns, glossary terms...")
            if search_query:
                results = catalog.search_catalog(search_query)
                if results:
                    for r in results[:20]:
                        st.write(
                            f"**[{r['type'].upper()}]** {r['name']} - "
                            f"{r.get('context', '')[:100]}")
                else:
                    st.info("No results found.")

            # Catalog tabs
            cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs(
                ["Tables", "Glossary", "Domains", "Classifications"])

            with cat_tab1:
                table_catalog = catalog.catalog.get("tables", {})
                if table_catalog:
                    for tname, tinfo in table_catalog.items():
                        cert = tinfo.get("certification_status", "Uncertified")
                        _cert_icons = {
                            "Certified": "[OK]", "Warning": "[!!]",
                            "Quarantined": "[XX]", "Uncertified": "[--]",
                        }
                        cert_icon = _cert_icons.get(cert, "[--]")
                        with st.expander(
                            f"{cert_icon} {tinfo.get('business_name', tname)} ({tname})"
                        ):
                            ec1, ec2, ec3 = st.columns(3)
                            ec1.write(f"**Domain:** {tinfo.get('domain', 'Unassigned')}")
                            ec2.write(f"**Owner:** {tinfo.get('owner', 'Unassigned')}")
                            ec3.write(f"**Source:** {tinfo.get('source_system', 'Unknown')}")

                            desc = st.text_area(
                                "Description",
                                value=tinfo.get("description", ""),
                                key=f"cat_desc_{tname}", height=60)
                            if desc != tinfo.get("description", ""):
                                catalog.catalog["tables"][tname]["description"] = desc
                                catalog._save_catalog()

                            # Certification
                            _cert_opts = ["Uncertified", "Certified", "Warning", "Quarantined"]
                            _cert_idx = _cert_opts.index(cert) if cert in _cert_opts else 0
                            new_cert = st.selectbox(
                                "Certification", _cert_opts,
                                index=_cert_idx, key=f"cat_cert_{tname}")
                            if new_cert != cert:
                                catalog.set_certification(tname, new_cert)

                            # DQ info
                            last_score = tinfo.get("last_dq_score")
                            if last_score is not None:
                                st.write(
                                    f"**Last DQ Score:** {last_score:.1f}/100 "
                                    f"(scanned: {tinfo.get('last_dq_scan', 'never')})")

                            # Columns
                            cols = tinfo.get("columns", {})
                            if cols:
                                st.write(f"**Columns ({len(cols)}):**")
                                col_rows = []
                                for cname, cinfo in cols.items():
                                    col_rows.append({
                                        "Column": cname,
                                        "Business Name": cinfo.get("business_name", cname),
                                        "Type": cinfo.get("data_type", ""),
                                        "Classification": cinfo.get("classification", "General"),
                                        "PII": "Yes" if cinfo.get("pii") else "",
                                        "Nullable": "Yes" if cinfo.get("nullable") else "No",
                                    })
                                st.dataframe(
                                    pd.DataFrame(col_rows),
                                    use_container_width=True, hide_index=True)

                            # Edit column metadata
                            if cols:
                                with st.expander("Edit Column Metadata"):
                                    col_to_edit = st.selectbox(
                                        "Column", list(cols.keys()),
                                        key=f"cat_coledit_{tname}")
                                    if col_to_edit and col_to_edit in cols:
                                        cdata = cols[col_to_edit]
                                        new_bname = st.text_input(
                                            "Business Name",
                                            value=cdata.get("business_name", ""),
                                            key=f"cat_bn_{tname}_{col_to_edit}")
                                        new_cdesc = st.text_area(
                                            "Description",
                                            value=cdata.get("description", ""),
                                            key=f"cat_cd_{tname}_{col_to_edit}",
                                            height=60)
                                        _class_opts = [
                                            "General", "PII", "Financial",
                                            "Reference", "Identifier"]
                                        _cur_class = cdata.get("classification", "General")
                                        _class_idx = (
                                            _class_opts.index(_cur_class)
                                            if _cur_class in _class_opts else 0)
                                        new_class = st.selectbox(
                                            "Classification", _class_opts,
                                            index=_class_idx,
                                            key=f"cat_cl_{tname}_{col_to_edit}")
                                        new_pii = st.checkbox(
                                            "Contains PII",
                                            value=cdata.get("pii", False),
                                            key=f"cat_pii_{tname}_{col_to_edit}")
                                        if st.button(
                                            "Save Column Metadata",
                                            key=f"cat_save_{tname}_{col_to_edit}",
                                        ):
                                            catalog.catalog["tables"][tname]["columns"][col_to_edit].update({
                                                "business_name": new_bname,
                                                "description": new_cdesc,
                                                "classification": new_class,
                                                "pii": new_pii,
                                            })
                                            catalog._save_catalog()
                                            st.success("Saved")
                else:
                    st.info("No tables cataloged yet. Load data and click Auto-Catalog.")

            with cat_tab2:
                glossary = catalog.get_glossary()
                if glossary:
                    st.write(f"**{len(glossary)} terms defined**")
                    for term_data in glossary:
                        _term_key = term_data.get("term", "").lower()
                        with st.expander(term_data.get("term", "?")):
                            st.write(f"**Definition:** {term_data.get('definition', '')}")
                            st.write(f"**Domain:** {term_data.get('domain', 'General')}")
                            if term_data.get("synonyms"):
                                st.write(
                                    f"**Synonyms:** {', '.join(term_data['synonyms'])}")
                            # Edit definition
                            new_def = st.text_area(
                                "Edit definition",
                                value=term_data.get("definition", ""),
                                key=f"glos_{_term_key}", height=60)
                            if new_def != term_data.get("definition", ""):
                                catalog.catalog["glossary"][_term_key]["definition"] = new_def
                                catalog._save_catalog()
                else:
                    st.info("No glossary terms yet. Auto-catalog tables to generate glossary.")

                # Add manual term
                with st.expander("Add Glossary Term Manually"):
                    new_term = st.text_input("Term", key="glos_new_term")
                    new_gdef = st.text_area("Definition", key="glos_new_def", height=60)
                    new_domain = st.selectbox(
                        "Domain",
                        ["General", "Customer", "Transaction", "Compliance",
                         "Operations", "Financial", "Reference"],
                        key="glos_new_domain")
                    if st.button("Add Term", key="glos_add") and new_term and new_gdef:
                        catalog.add_glossary_term(new_term, new_gdef, domain=new_domain)
                        st.success(f"Added: {new_term}")
                        st.rerun()

            with cat_tab3:
                domains = catalog.catalog.get("domains", {})
                if domains:
                    for dname, dinfo in domains.items():
                        with st.expander(dname):
                            st.write(f"**Description:** {dinfo.get('description', '')}")
                            st.write(f"**Owner:** {dinfo.get('owner', 'Unassigned')}")
                            st.write(f"**Steward:** {dinfo.get('steward', 'Unassigned')}")
                else:
                    st.info("No domains configured.")

                with st.expander("Add Domain"):
                    new_dname = st.text_input("Domain Name", key="dom_new_name")
                    new_ddesc = st.text_area("Description", key="dom_new_desc", height=60)
                    new_downer = st.text_input("Owner", key="dom_new_owner")
                    if st.button("Add Domain", key="dom_add") and new_dname:
                        catalog.add_domain(new_dname, new_ddesc, owner=new_downer)
                        st.success(f"Added domain: {new_dname}")
                        st.rerun()

            with cat_tab4:
                classifications = catalog.catalog.get("classifications", {})
                if classifications:
                    class_rows = []
                    for cname, cinfo in classifications.items():
                        class_rows.append({
                            "Name": cname,
                            "Sensitivity": cinfo.get("sensitivity_level", ""),
                            "Description": cinfo.get("description", "")[:80],
                            "Handling Rules": len(cinfo.get("handling_rules", [])),
                        })
                    st.dataframe(
                        pd.DataFrame(class_rows),
                        use_container_width=True, hide_index=True)
                else:
                    st.info("No classifications configured.")

            # Lineage
            st.markdown("---")
            st.markdown("##### Data Lineage")

            from core.lineage import DataLineage
            if "lineage" not in st.session_state:
                st.session_state.lineage = DataLineage()

            _lin = st.session_state.lineage
            _lin_stats = _lin.get_lineage_stats()

            _l1, _l2, _l3 = st.columns(3)
            _l1.metric("Nodes", _lin_stats.get("total_nodes", 0))
            _l2.metric("Edges", _lin_stats.get("total_edges", 0))
            _l3.metric(
                "Coverage",
                f"{_lin_stats.get('coverage_pct', 0):.0f}%",
            )

            if _dq_tables:
                if st.button(
                    "Build Lineage Graph", type="primary",
                    key="lin_build",
                ):
                    with st.status(
                        "Building lineage graph...",
                        expanded=True,
                    ) as _lin_status:
                        _lin_status.write(
                            "Discovering table relationships...")
                        _lin_status.write(
                            "Detecting foreign key references...")
                        _lin_counts = _lin.auto_build_lineage(
                            _dq_tables,
                            source_system="Snowflake",
                            database="SNOWFLAKE_SAMPLE_DATA",
                            schema="TPCH_SF1",
                        )
                        _ln = _lin_counts.get("nodes", 0)
                        _le = _lin_counts.get("edges", 0)
                        _lin_status.update(
                            label=f"Lineage complete - "
                                  f"{_ln} nodes, {_le} edges",
                            state="complete",
                        )
                    st.rerun()

            if _lin_stats.get("total_nodes", 0) > 0:
                st.markdown("##### Data Lineage Graph")

                _dot = _lin.generate_graphviz()
                if _dot:
                    st.graphviz_chart(_dot, use_container_width=True)

                # Table-level focus view
                _focus_tbl_nodes = [
                    nid for nid, n in _lin.data.get("nodes", {}).items()
                    if n.get("type") == "table"
                ]
                if _focus_tbl_nodes:
                    _focus_tbl_names = [
                        _lin.data["nodes"][n]["name"]
                        for n in _focus_tbl_nodes
                    ]
                    _selected_focus = st.selectbox(
                        "Focus on table",
                        ["Full Graph"] + _focus_tbl_names,
                        key="lin_focus",
                    )
                    if _selected_focus != "Full Graph":
                        _focus_id = f"table_{_selected_focus.lower()}"
                        _focused_dot = _lin.generate_graphviz(
                            center_node=_focus_id, depth=2)
                        if _focused_dot:
                            st.graphviz_chart(
                                _focused_dot, use_container_width=True)

                # Impact analysis
                st.markdown("##### Impact Analysis")
                _tbl_nodes = [
                    nid for nid, n in _lin.data.get(
                        "nodes", {}).items()
                    if n.get("type") == "table"
                ]
                if _tbl_nodes:
                    _tbl_names = [
                        _lin.data["nodes"][n]["name"]
                        for n in _tbl_nodes
                    ]
                    _sel_lin = st.selectbox(
                        "Select table for impact analysis",
                        _tbl_names, key="lin_impact_select",
                    )
                    if _sel_lin:
                        _lin_nid = f"table_{_sel_lin.lower()}"
                        _lin_impact = _lin.impact_analysis(_lin_nid)
                        if _lin_impact:
                            st.write(
                                f"**Total downstream assets:** "
                                f"{_lin_impact.get('total_downstream', 0)}")
                            st.write(
                                f"**Impact severity:** "
                                f"{_lin_impact.get('impact_severity', 'LOW')}")
                            _lin_bt = _lin_impact.get("by_type", {})
                            if _lin_bt:
                                for _lt, _lc in _lin_bt.items():
                                    st.write(f"  {_lt}: {_lc}")

            # Export catalog
            st.markdown("---")
            exp1, exp2 = st.columns(2)
            with exp1:
                if st.button("Export Catalog (JSON)", key="cat_export_json"):
                    cat_json = catalog.export_catalog(fmt="json")
                    st.download_button(
                        "Download", cat_json,
                        "data_catalog.json", "application/json",
                        key="cat_dl_json")
            with exp2:
                if st.button("Export Catalog (Markdown)", key="cat_export_md"):
                    cat_md = catalog.export_catalog(fmt="markdown")
                    st.download_button(
                        "Download", cat_md,
                        "data_catalog.md", "text/markdown",
                        key="cat_dl_md")

        # ============================================================
        # SUBTAB 3: Business Rules (placeholder)
        # ============================================================
        with dq_subtab3:
            st.markdown("#### Custom Business Rules Engine")

            if "rules_engine" not in st.session_state:
                st.session_state.rules_engine = BusinessRulesEngine()
            _bre = st.session_state.rules_engine

            _bre_rules = _bre.rules
            st.write(f"**{len(_bre_rules)} rules configured**")

            # Add rule manually
            with st.expander("Add Business Rule"):
                _br_name = st.text_input("Rule Name", key="br_name")
                _br_col = st.text_input("Column", key="br_col")
                _br_expr = st.text_input(
                    "Expression (pandas eval syntax)",
                    key="br_expr",
                    placeholder="e.g. Amount > 0")
                _br_sev = st.selectbox(
                    "Severity", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    index=1, key="br_sev")
                if st.button("Add Rule", key="br_add") and _br_name and _br_expr:
                    _bre.add_rule(
                        _br_name, column=_br_col, expression=_br_expr,
                        severity=_br_sev)
                    st.success(f"Added: {_br_name}")
                    st.rerun()

            # Evaluate rules
            if _dq_tables and _bre_rules:
                _br_sel = st.selectbox(
                    "Evaluate against table",
                    list(_dq_tables.keys()), key="br_table_sel")
                if st.button("Evaluate Rules", type="primary", key="br_eval"):
                    _br_df = _dq_tables[_br_sel]
                    with st.status(
                        f"Evaluating {len(_bre_rules)} rules "
                        f"against {_br_sel}...",
                        expanded=True,
                    ) as _br_status:
                        _br_pass = 0
                        _br_fail = 0
                        _br_results = _bre.evaluate_all(
                            _br_df, _br_sel)
                        for _br_r in _br_results.get("results", []):
                            _r_name = _br_r.get("rule_name", "")
                            _r_id = _br_r.get("rule_id", "")
                            if _br_r.get("passed"):
                                _br_status.write(
                                    f"Rule {_r_id}: {_r_name} - "
                                    f":green[PASS]")
                                _br_pass += 1
                            else:
                                _br_status.write(
                                    f"Rule {_r_id}: {_r_name} - "
                                    f":red[FAIL]")
                                _br_fail += 1
                        _br_status.update(
                            label=f"Rules complete - {_br_pass} "
                                  f"passed, {_br_fail} failed",
                            state="complete")
                        # Bridge to agent
                        if ("agent" in st.session_state
                                and st.session_state.agent
                                and hasattr(st.session_state.agent,
                                            "set_rules_results")):
                            st.session_state.agent.set_rules_results(
                                _bre.evaluation_results)
                    st.rerun()

            # Show evaluation results
            if _bre.evaluation_results:
                for _er_tbl, _er_data in _bre.evaluation_results.items():
                    with st.expander(
                        f"{_er_tbl}: "
                        f"{_er_data.get('pass_count', 0)} passed, "
                        f"{_er_data.get('fail_count', 0)} failed"
                    ):
                        for _er_r in _er_data.get("results", []):
                            _pass_txt = (
                                ":green[PASS]"
                                if _er_r.get("passed")
                                else ":red[FAIL]"
                            )
                            st.markdown(
                                f"**{_er_r.get('rule_name', '')}** "
                                f"- {_pass_txt} "
                                f"({_er_r.get('violations', 0)} "
                                f"violations)")
            elif not _bre_rules:
                st.info(
                    "No rules configured. Add business rules above "
                    "or load data and run a DQ scan to auto-generate "
                    "validation rules.")

        # ============================================================
        # SUBTAB 4: Trending (placeholder)
        # ============================================================
        with dq_subtab4:
            if "dq_history" not in st.session_state:
                st.session_state.dq_history = DQHistory()

            history = st.session_state.dq_history

            st.markdown("#### Quality Score Trending")

            # Check for history data
            all_latest = history.get_all_table_latest()

            if not all_latest:
                st.info(
                    "No scan history yet. Run a quality scan in the "
                    "Scanner tab first. Each scan is automatically "
                    "recorded for trending."
                )
            else:
                # Table selector
                _trend_names = list(all_latest.keys())
                selected_trend_table = st.selectbox(
                    "Select table", _trend_names,
                    key="trend_table_select",
                )

                if selected_trend_table:
                    # Degradation check
                    degradation = history.detect_degradation(
                        selected_trend_table)
                    if degradation.get("degraded"):
                        st.warning(
                            f"Quality degradation detected: overall score "
                            f"dropped by "
                            f"{abs(degradation['overall_change']):.1f} "
                            f"points since last scan."
                        )
                        _deg_dims = degradation.get(
                            "degraded_dimensions", [])
                        if _deg_dims:
                            st.write(
                                f"Degraded dimensions: "
                                f"{', '.join(_deg_dims)}"
                            )
                    elif degradation.get("overall_change", 0) > 0:
                        st.success(
                            f"Quality improved by "
                            f"{degradation['overall_change']:.1f} "
                            f"points since last scan."
                        )

                    # Stats row
                    best_worst = history.get_best_worst(
                        selected_trend_table)
                    freq = history.get_scan_frequency(
                        selected_trend_table)

                    tw1, tw2, tw3, tw4, tw5 = st.columns(5)
                    tw1.metric(
                        "Current",
                        f"{best_worst.get('current_score', 0):.1f}",
                    )
                    tw2.metric(
                        "Best",
                        f"{best_worst.get('best_score', 0):.1f}",
                    )
                    tw3.metric(
                        "Worst",
                        f"{best_worst.get('worst_score', 0):.1f}",
                    )
                    tw4.metric(
                        "Average",
                        f"{best_worst.get('avg_score', 0):.1f}",
                    )
                    tw5.metric(
                        "Total Scans",
                        best_worst.get("total_scans", 0),
                    )

                    # Trend chart
                    chart_data = history.generate_trend_chart_data(
                        selected_trend_table, limit=30)

                    if (chart_data
                            and chart_data.get("timestamps")
                            and len(chart_data["timestamps"]) > 1):

                        st.markdown("##### Overall Score Trend")
                        trend_df = pd.DataFrame({
                            "Scan": range(
                                1,
                                len(chart_data["timestamps"]) + 1,
                            ),
                            "Overall": chart_data.get("overall", []),
                        })
                        st.line_chart(trend_df, x="Scan", y="Overall")

                        st.markdown("##### Dimension Trends")
                        dim_df = pd.DataFrame({
                            "Scan": range(
                                1,
                                len(chart_data["timestamps"]) + 1,
                            ),
                        })
                        _dim_names = [
                            "completeness", "accuracy", "consistency",
                            "timeliness", "uniqueness", "validity",
                        ]
                        for _dn in _dim_names:
                            _ds = chart_data.get(_dn, [])
                            if _ds and any(s is not None for s in _ds):
                                dim_df[_dn.title()] = [
                                    s if s is not None else 0
                                    for s in _ds
                                ]

                        if len(dim_df.columns) > 1:
                            _ycols = [
                                c for c in dim_df.columns if c != "Scan"
                            ]
                            st.line_chart(
                                dim_df, x="Scan", y=_ycols)

                        # Scan frequency
                        st.markdown("##### Scan Activity")
                        sf1, sf2, sf3 = st.columns(3)
                        sf1.write(
                            f"**First scan:** "
                            f"{freq.get('first_scan', 'N/A')[:10]}"
                        )
                        sf2.write(
                            f"**Last scan:** "
                            f"{freq.get('last_scan', 'N/A')[:10]}"
                        )
                        sf3.write(
                            f"**Avg days between:** "
                            f"{freq.get('avg_days_between_scans', 0):.1f}"
                        )

                    elif (chart_data
                          and len(
                              chart_data.get("timestamps", [])) == 1):
                        st.info(
                            "Only one scan recorded. Run more scans "
                            "to see trends."
                        )

                # Cross-table comparison
                if len(_trend_names) > 1:
                    st.markdown("---")
                    st.markdown("##### Cross-Table Comparison")
                    compare = history.compare_tables(_trend_names)

                    if compare:
                        comp_rows = []
                        for _ctn, _ctd in compare.get(
                                "tables", {}).items():
                            row = {
                                "Table": _ctn,
                                "Overall": f"{_ctd.get('overall', 0):.1f}",
                            }
                            _cdims = _ctd.get("dimensions", {})
                            for _cdn in [
                                "completeness", "accuracy",
                                "consistency", "timeliness",
                                "uniqueness", "validity",
                            ]:
                                _cs = _cdims.get(_cdn)
                                row[_cdn.title()] = (
                                    f"{_cs:.1f}"
                                    if _cs is not None else "N/A"
                                )
                            comp_rows.append(row)

                        st.dataframe(
                            pd.DataFrame(comp_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

                        cc1, cc2, cc3 = st.columns(3)
                        cc1.write(
                            f"**Best:** "
                            f"{compare.get('best_table', 'N/A')}"
                        )
                        cc2.write(
                            f"**Worst:** "
                            f"{compare.get('worst_table', 'N/A')}"
                        )
                        cc3.write(
                            f"**Average:** "
                            f"{compare.get('avg_overall', 0):.1f}"
                        )

                # Export
                st.markdown("---")
                if st.button(
                    "Export History (CSV)", key="trend_export_csv"
                ):
                    csv_data = history.export_history(fmt="csv")
                    if csv_data:
                        st.download_button(
                            "Download CSV", csv_data,
                            "dq_history.csv", "text/csv",
                            key="trend_dl_csv",
                        )

        # ============================================================
        # SUBTAB 5: Stewardship & Issue Tracking
        # ============================================================
        with dq_subtab5:
            if "stewardship" not in st.session_state:
                st.session_state.stewardship = StewardshipWorkflow()

            _sw = st.session_state.stewardship
            _sw_stats = _sw.get_dashboard_stats()

            st.markdown("#### Data Stewardship & Issue Tracking")

            # Dashboard stats row
            _sw1, _sw2, _sw3, _sw4, _sw5 = st.columns(5)
            _sw1.metric("Open Issues", _sw_stats.get("total_open", 0))
            _sw2.metric("Resolved", _sw_stats.get("total_resolved", 0))
            _sw3.metric("SLA Breaches", _sw_stats.get("sla_breaches", 0))
            _sw4.metric("Unassigned", _sw_stats.get("unassigned", 0))
            _sw_avg = _sw_stats.get("avg_resolution_hours")
            _sw5.metric(
                "Avg Resolution",
                f"{_sw_avg:.1f}h" if _sw_avg else "N/A",
            )

            # SLA breaches warning
            _sw_breaches = _sw.check_sla_breaches()
            if _sw_breaches:
                st.warning(
                    f"{len(_sw_breaches)} issues have breached their SLA")
                for _b in _sw_breaches[:5]:
                    st.write(
                        f"**{_b.get('issue_id')}** - "
                        f"{_b.get('title', '')[:60]} "
                        f"({_b.get('severity')}) - "
                        f"{_b.get('hours_overdue', 0):.0f}h overdue"
                    )

            # Auto-create issues from DQ scan
            if _dq.scan_results:
                if st.button(
                    "Create Issues from Latest DQ Scan",
                    key="sw_auto_create",
                ):
                    _sw_total_created = 0
                    for _swtn, _swres in _dq.scan_results.items():
                        _swc = _sw.create_issues_from_dq_scan(
                            _swres, _swtn)
                        _sw_total_created += _swc if _swc else 0
                    if _sw_total_created > 0:
                        st.success(
                            f"Created {_sw_total_created} new issues "
                            f"from DQ scan results")
                    else:
                        st.info(
                            "No new issues to create (all CRITICAL/HIGH "
                            "issues already tracked)")
                    # Bridge stewardship stats to agent
                    if "agent" in st.session_state and st.session_state.agent and hasattr(st.session_state.agent, "set_stewardship_stats"):
                        st.session_state.agent.set_stewardship_stats(_sw.get_dashboard_stats())
                    st.rerun()

            st.markdown("---")

            # Issue list with filters
            _fc1, _fc2, _fc3 = st.columns(3)
            with _fc1:
                _sw_filt_status = st.selectbox(
                    "Filter by Status",
                    ["All", "Open", "Assigned", "In Progress",
                     "Resolved", "Verified", "Closed", "Wont Fix"],
                    key="sw_filter_status",
                )
            with _fc2:
                _sw_filt_sev = st.selectbox(
                    "Filter by Severity",
                    ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    key="sw_filter_severity",
                )
            with _fc3:
                _sw_all_tables = sorted(set(
                    i.get("table_name", "")
                    for i in _sw.data.get("issues", {}).values()
                    if i.get("table_name")
                ))
                _sw_filt_tbl = st.selectbox(
                    "Filter by Table",
                    ["All"] + _sw_all_tables,
                    key="sw_filter_table",
                )

            _sw_issues = _sw.get_issues(
                status=(
                    _sw_filt_status
                    if _sw_filt_status != "All" else None),
                severity=(
                    _sw_filt_sev
                    if _sw_filt_sev != "All" else None),
                table_name=(
                    _sw_filt_tbl
                    if _sw_filt_tbl != "All" else None),
            )

            if _sw_issues:
                st.write(f"**{len(_sw_issues)} issues**")

                _sev_icon = {
                    "CRITICAL": "[CRIT]", "HIGH": "[HIGH]",
                    "MEDIUM": "[MED]", "LOW": "[LOW]",
                }
                _st_icon = {
                    "Open": "[OPEN]", "Assigned": "[ASGN]",
                    "In Progress": "[WIP]", "Resolved": "[DONE]",
                    "Verified": "[OK]", "Closed": "[CLOSED]",
                    "Wont Fix": "[WFIX]",
                }

                for _swi in _sw_issues:
                    _swi_id = _swi.get("id", "")
                    _swi_sev = _swi.get("severity", "LOW")
                    _swi_st = _swi.get("status", "Open")
                    _si = _sev_icon.get(_swi_sev, "")
                    _sti = _st_icon.get(_swi_st, "")

                    with st.expander(
                        f"{_si} {_sti} {_swi_id} | "
                        f"{_swi.get('title', '')[:70]} | {_swi_st} | "
                        f"{_swi.get('assigned_to', 'Unassigned')}"
                    ):
                        st.write(
                            f"**Table:** {_swi.get('table_name', '')}")
                        if _swi.get("column_name"):
                            st.write(
                                f"**Column:** {_swi['column_name']}")
                        st.write(
                            f"**Dimension:** "
                            f"{_swi.get('dimension', 'N/A')}")
                        st.write(
                            f"**Description:** "
                            f"{_swi.get('description', '')}")
                        st.write(
                            f"**Created:** "
                            f"{_swi.get('created_at', '')[:16]}")
                        if _swi.get("due_at"):
                            st.write(
                                f"**Due:** {_swi['due_at'][:16]}")
                        if _swi.get("regulatory_tag"):
                            st.write(
                                f"**Regulatory:** "
                                f"{_swi['regulatory_tag']}")

                        # Actions
                        _ac1, _ac2, _ac3 = st.columns(3)

                        with _ac1:
                            _stew_list = list(
                                _sw.data.get("stewards", {}).keys())
                            _asgn_opts = (
                                ["Unassigned"] + _stew_list
                                if _stew_list
                                else ["Unassigned", "DQ Team",
                                      "Data Engineering", "Analytics"]
                            )
                            _cur_asgn = _swi.get(
                                "assigned_to", "Unassigned")
                            _asgn_idx = (
                                _asgn_opts.index(_cur_asgn)
                                if _cur_asgn in _asgn_opts else 0
                            )
                            _new_asgn = st.selectbox(
                                "Assign to", _asgn_opts,
                                index=_asgn_idx,
                                key=f"sw_assign_{_swi_id}",
                            )
                            if (_new_asgn != _cur_asgn
                                    and st.button(
                                        "Assign",
                                        key=f"sw_do_assign_{_swi_id}")):
                                _sw.assign_issue(_swi_id, _new_asgn)
                                st.rerun()

                        with _ac2:
                            _valid_tr = {
                                "Open": [
                                    "Assigned", "In Progress",
                                    "Wont Fix"],
                                "Assigned": [
                                    "In Progress", "Resolved",
                                    "Wont Fix"],
                                "In Progress": [
                                    "Resolved", "Wont Fix"],
                                "Resolved": [
                                    "Verified", "Closed",
                                    "In Progress"],
                                "Verified": ["Closed"],
                                "Closed": [],
                                "Wont Fix": ["Open"],
                            }
                            _tr = _valid_tr.get(_swi_st, [])
                            if _tr:
                                _new_st = st.selectbox(
                                    "Change status", _tr,
                                    key=f"sw_status_{_swi_id}",
                                )
                                if st.button(
                                    "Update",
                                    key=f"sw_do_status_{_swi_id}",
                                ):
                                    _notes = None
                                    if _new_st in (
                                        "Resolved", "Closed",
                                        "Wont Fix",
                                    ):
                                        _notes = st.session_state.get(
                                            f"sw_notes_{_swi_id}", "")
                                    _sw.update_status(
                                        _swi_id, _new_st,
                                        notes=_notes)
                                    st.rerun()

                        with _ac3:
                            if _swi_st in (
                                "Open", "Assigned", "In Progress",
                            ):
                                if st.button(
                                    "Escalate",
                                    key=f"sw_escalate_{_swi_id}",
                                ):
                                    _sw.escalate_issue(
                                        _swi_id,
                                        reason="Manual escalation",
                                    )
                                    st.success("Escalated")
                                    st.rerun()

                        # Resolution notes
                        if _swi_st in ("Assigned", "In Progress"):
                            st.text_area(
                                "Resolution notes",
                                key=f"sw_notes_{_swi_id}",
                                height=60,
                                placeholder=(
                                    "Describe what was done to "
                                    "fix this..."),
                            )

                        # Comments
                        if _swi.get("comments"):
                            st.write("**Comments:**")
                            for _c in _swi["comments"]:
                                st.caption(
                                    f"{_c.get('author', 'User')} "
                                    f"({_c.get('at', '')[:16]}): "
                                    f"{_c.get('text', '')}")

                        _cmt = st.text_input(
                            "Add comment",
                            key=f"sw_comment_{_swi_id}",
                            placeholder="Type a comment...",
                        )
                        if _cmt and st.button(
                            "Post", key=f"sw_post_{_swi_id}"
                        ):
                            _sw.add_comment(_swi_id, _cmt)
                            st.rerun()

                        # History
                        with st.expander("Issue History"):
                            for _h in _swi.get("history", []):
                                st.caption(
                                    f"{_h.get('at', '')[:16]} - "
                                    f"{_h.get('action', '')} by "
                                    f"{_h.get('by', 'System')}")
            else:
                st.info(
                    "No issues matching filters. Run a DQ scan and "
                    "click 'Create Issues from Latest DQ Scan' to "
                    "populate.")

            # Steward management
            st.markdown("---")
            st.markdown("##### Manage Data Stewards")

            _existing_stew = _sw.data.get("stewards", {})
            if _existing_stew:
                _stew_rows = [
                    {
                        "Name": name,
                        "Role": info.get("role", ""),
                        "Email": info.get("email", ""),
                        "Tables": len(
                            info.get("assigned_tables", [])),
                        "Domains": ", ".join(
                            info.get("domains", [])),
                    }
                    for name, info in _existing_stew.items()
                ]
                st.dataframe(
                    pd.DataFrame(_stew_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("Add Data Steward"):
                _sn = st.text_input("Name", key="sw_stew_name")
                _se = st.text_input("Email", key="sw_stew_email")
                _sr = st.selectbox(
                    "Role",
                    ["Data Steward", "Data Owner",
                     "Domain Lead", "DQ Analyst"],
                    key="sw_stew_role",
                )
                _sd = st.multiselect(
                    "Domains",
                    ["Customer", "Transaction", "Compliance",
                     "Operations", "Financial", "Reference"],
                    key="sw_stew_domains",
                )
                if st.button("Add Steward", key="sw_stew_add") and _sn:
                    _sw.add_steward(
                        _sn, email=_se, role=_sr, domains=_sd)
                    st.success(f"Added steward: {_sn}")
                    st.rerun()

            # Export
            st.markdown("---")
            if st.button("Export Issues (CSV)", key="sw_export"):
                _sw_csv = _sw.export_issues(fmt="csv")
                if _sw_csv:
                    st.download_button(
                        "Download", _sw_csv,
                        "dq_issues.csv", "text/csv",
                        key="sw_dl_csv",
                    )

        # ============================================================
        # SUBTAB 6: Regulatory Compliance Mapping
        # ============================================================
        with dq_subtab6:
            from core.compliance_map import ComplianceMapper
            if "compliance" not in st.session_state:
                st.session_state.compliance = ComplianceMapper()

            _comp = st.session_state.compliance

            st.markdown("#### Regulatory Compliance Mapping")

            # Framework overview
            _comp_fws = _comp.data.get("frameworks", {})
            if _comp_fws:
                _fw_cols = st.columns(min(len(_comp_fws), 6))
                for _fi, (_fn, _fd) in enumerate(_comp_fws.items()):
                    with _fw_cols[_fi % len(_fw_cols)]:
                        _rc = len(_fd.get("requirements", {}))
                        st.metric(_fn, f"{_rc} reqs")

            # Auto-map
            if _dq_tables:
                if st.button(
                    "Auto-Map Columns to Regulations",
                    type="primary", key="comp_automap",
                ):
                    _comp_total = 0
                    for _ctn, _cdf in _dq_tables.items():
                        _cm = _comp.auto_map_columns(_cdf, _ctn)
                        _comp_total += len(_cm) if _cm else 0
                    st.success(
                        f"Mapped {_comp_total} column-to-regulation "
                        f"links")
                    st.rerun()

            # Assess compliance
            if _dq.scan_results and _dq_tables:
                if st.button("Assess Compliance", key="comp_assess"):
                    with st.status(
                        "Mapping columns to regulatory frameworks...",
                        expanded=True,
                    ) as _cs:
                        for _ctn in _dq.scan_results:
                            _cs.write(f"Assessing {_ctn}...")
                            _crr = (
                                st.session_state.rules_engine
                                .evaluation_results.get(_ctn)
                                if "rules_engine" in st.session_state
                                else None
                            )
                            _comp.assess_compliance(
                                _ctn, _dq.scan_results[_ctn], _crr)
                        # Show per-framework scores
                        _csumm = _comp.get_compliance_summary()
                        _by_fw = _csumm.get("by_framework", {})
                        for _fwn, _fwi in _by_fw.items():
                            _cs.write(
                                f"Checking {_fwn}... "
                                f"{_fwi.get('score', 0):.1f}%")
                        _overall = _csumm.get("overall_score", 0)
                        _cs.update(
                            label=f"Compliance assessment complete "
                                  f"- Overall: {_overall:.1f}%",
                            state="complete",
                        )
                    # Bridge compliance summary to agent
                    if "agent" in st.session_state and st.session_state.agent and hasattr(st.session_state.agent, "set_compliance_summary"):
                        st.session_state.agent.set_compliance_summary(_comp.get_compliance_summary())
                    st.rerun()

            # Results
            _comp_scores = _comp.data.get("compliance_scores", {})
            if _comp_scores:
                _csummary = _comp.get_compliance_summary()

                _cs1, _cs2, _cs3 = st.columns(3)
                _cs1.metric(
                    "Overall Compliance",
                    f"{_csummary.get('overall_score', 0):.1f}%",
                )
                _cs2.metric(
                    "Tables Assessed",
                    _csummary.get("tables_assessed", 0),
                )
                _cs3.metric(
                    "Critical Gaps",
                    len(_csummary.get("critical_gaps", [])),
                )

                # By framework
                st.markdown("##### Compliance by Framework")
                _by_fw = _csummary.get("by_framework", {})
                if _by_fw:
                    _fw_rows = [
                        {
                            "Framework": _fwn,
                            "Score": f"{_fwi.get('score', 0):.1f}%",
                            "Compliant": _fwi.get("compliant", 0),
                            "Non-Compliant": _fwi.get(
                                "non_compliant", 0),
                        }
                        for _fwn, _fwi in _by_fw.items()
                    ]
                    st.dataframe(
                        pd.DataFrame(_fw_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Critical gaps
                _gaps = _csummary.get("critical_gaps", [])
                if _gaps:
                    st.markdown("##### Critical Compliance Gaps")
                    for _g in _gaps[:10]:
                        st.warning(
                            f"**{_g.get('framework', '')} - "
                            f"{_g.get('requirement_id', '')}**: "
                            f"{_g.get('title', '')} "
                            f"(Score: {_g.get('score', 0):.1f}%)")

                # Per-table details
                for _ctn, _ctcomp in _comp_scores.items():
                    with st.expander(
                        f"{_ctn}: "
                        f"{_ctcomp.get('overall_compliance_score', 0):.1f}"
                        f"% compliant"
                    ):
                        for _cfn, _cfd in _ctcomp.get(
                            "frameworks", {}
                        ).items():
                            st.write(
                                f"**{_cfn}**: "
                                f"{_cfd.get('overall_score', 0):.1f}% "
                                f"({_cfd.get('compliant_count', 0)} "
                                f"compliant / "
                                f"{_cfd.get('non_compliant_count', 0)} "
                                f"non-compliant)")
                            for _rid, _rd in _cfd.get(
                                "requirements", {}
                            ).items():
                                if not _rd.get("compliant", True):
                                    st.caption(
                                        f"  {_rid}: "
                                        f"{_rd.get('title', '')} - "
                                        f"{_rd.get('score', 0):.1f}%")
            else:
                st.info(
                    "Run a DQ scan, then click Assess Compliance "
                    "to evaluate regulatory requirements.")

            # Regulatory requirement browser
            st.markdown("---")
            st.markdown("##### Regulatory Requirements Browser")
            _sel_fw = st.selectbox(
                "Framework", list(_comp_fws.keys()),
                key="comp_fw_select",
            )
            if _sel_fw:
                _reqs = _comp.get_requirements_by_framework(_sel_fw)
                if _reqs:
                    _sev_ic = {
                        "CRITICAL": "[CRIT]", "HIGH": "[HIGH]",
                        "MEDIUM": "[MED]", "LOW": "[LOW]",
                    }
                    for _rid, _rdata in _reqs.items():
                        _rsev = _rdata.get("severity", "MEDIUM")
                        with st.expander(
                            f"{_sev_ic.get(_rsev, '')} {_rid}: "
                            f"{_rdata.get('title', '')}"
                        ):
                            st.write(
                                f"**Description:** "
                                f"{_rdata.get('description', '')}")
                            st.write(
                                f"**Data Elements:** "
                                f"{', '.join(_rdata.get('data_elements', []))}")
                            st.write(
                                f"**DQ Dimensions:** "
                                f"{', '.join(_rdata.get('dq_dimensions', []))}")
                            st.write(f"**Severity:** {_rsev}")

            # Export
            st.markdown("---")
            if st.button(
                "Export Compliance Report (Markdown)",
                key="comp_export",
            ):
                _comp_md = _comp.export_compliance_report(
                    fmt="markdown")
                if _comp_md:
                    st.download_button(
                        "Download", _comp_md,
                        "compliance_report.md", "text/markdown",
                        key="comp_dl",
                    )

        # ============================================================
        # SUBTAB 7: Incident Management
        # ============================================================
        with dq_subtab7:
            from core.incidents import IncidentManager
            if "incidents" not in st.session_state:
                st.session_state.incidents = IncidentManager()
            _inc = st.session_state.incidents
            _inc_stats = _inc.get_dashboard_stats()

            st.markdown("#### Incident Management")

            _in1, _in2, _in3, _in4, _in5 = st.columns(5)
            _in1.metric("Open", _inc_stats.get("open", 0))
            _in2.metric("Resolved", _inc_stats.get("resolved", 0))
            _in3.metric("Closed", _inc_stats.get("closed", 0))
            _mttr = _inc_stats.get("mttr")
            _in4.metric("MTTR", f"{_mttr:.1f}h" if _mttr else "N/A")
            _in5.metric("PM Pending", _inc_stats.get("postmortems_pending", 0))

            # -- Create incident manually --
            with st.expander("Create New Incident"):
                _inc_title = st.text_input("Title", key="inc_title")
                _inc_desc = st.text_area(
                    "Description", key="inc_desc", height=80)
                _ic1, _ic2, _ic3 = st.columns(3)
                with _ic1:
                    _inc_sev = st.selectbox(
                        "Severity",
                        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        index=1, key="inc_sev")
                with _ic2:
                    _inc_cat = st.selectbox(
                        "Category",
                        ["Data Quality", "Schema", "Volume",
                         "Timeliness", "Compliance", "Security",
                         "Other"], key="inc_cat")
                with _ic3:
                    _inc_tbl = st.text_input(
                        "Affected Table", key="inc_table")
                if st.button(
                    "Create Incident", key="inc_create"
                ) and _inc_title:
                    _new_iid = _inc.create_incident(
                        _inc_title, _inc_desc, _inc_sev,
                        _inc_cat, table_name=_inc_tbl or None)
                    st.success(f"Created: {_new_iid}")
                    st.rerun()

            # -- Auto-detect from observability --
            if st.button(
                "Auto-Detect Incidents from Observability",
                key="inc_auto",
            ):
                _auto_created = 0
                if "dq_history" in st.session_state:
                    _hist = st.session_state.dq_history
                    for _atn in _hist.get_all_table_latest():
                        _adeg = _hist.detect_degradation(_atn)
                        if _adeg.get("degraded"):
                            _aiid = _inc.auto_create_from_degradation(
                                _atn, _adeg)
                            if _aiid:
                                _auto_created += 1
                if _dq.scan_results:
                    for _atn in _dq.scan_results:
                        if _atn in _dq_tables:
                            _adrift = _dq.detect_schema_drift(
                                _dq_tables[_atn], _atn)
                            if _adrift.get("has_drift"):
                                _aiid = _inc.auto_create_from_schema_drift(
                                    _atn, _adrift)
                                if _aiid:
                                    _auto_created += 1
                            _avol = _dq.detect_volume_anomaly(
                                _dq_tables[_atn], _atn)
                            if _avol.get("is_anomaly"):
                                _aiid = _inc.auto_create_from_volume_anomaly(
                                    _atn, _avol)
                                if _aiid:
                                    _auto_created += 1
                if _auto_created > 0:
                    st.success(
                        f"Created {_auto_created} incidents "
                        f"from observability checks")
                else:
                    st.info("No anomalies detected")
                # Bridge incident stats to agent
                if "agent" in st.session_state and st.session_state.agent and hasattr(st.session_state.agent, "set_incident_stats"):
                    st.session_state.agent.set_incident_stats(_inc.get_dashboard_stats())
                st.rerun()

            # -- Incident list --
            st.markdown("---")
            _all_inc = _inc.get_incidents()

            if _all_inc:
                for _incident in _all_inc:
                    _iid = _incident.get("id", "")
                    _isev = _incident.get("severity", "LOW")
                    _ist = _incident.get("status", "Open")
                    _sev_tag = {"CRITICAL": "[CRIT]", "HIGH": "[HIGH]",
                                "MEDIUM": "[MED]", "LOW": "[LOW]"
                                }.get(_isev, "[--]")

                    with st.expander(
                        f"{_sev_tag} {_iid} | "
                        f"{_incident.get('title', '')[:60]} | {_ist}"
                    ):
                        st.write(
                            f"**Category:** "
                            f"{_incident.get('category', '')}")
                        st.write(
                            f"**Table:** "
                            f"{_incident.get('table_name', 'N/A')}")
                        st.write(
                            f"**Detected:** "
                            f"{_incident.get('detected_at', '')[:16]}")
                        st.write(
                            f"**Description:** "
                            f"{_incident.get('description', '')}")

                        # Status actions
                        _act1, _act2, _act3 = st.columns(3)
                        with _act1:
                            if _ist == "Open":
                                if st.button(
                                    "Acknowledge",
                                    key=f"inc_ack_{_iid}",
                                ):
                                    _inc.acknowledge_incident(
                                        _iid, "User")
                                    st.rerun()
                            elif _ist == "Acknowledged":
                                if st.button(
                                    "Mark Resolved",
                                    key=f"inc_resolve_{_iid}",
                                ):
                                    _inc.resolve_incident(
                                        _iid,
                                        "Resolved via DQ Engine",
                                        ["Identified issue",
                                         "Applied fix"],
                                        "User")
                                    st.rerun()
                            elif _ist == "Resolved":
                                if st.button(
                                    "Close (Postmortem)",
                                    key=f"inc_close_{_iid}",
                                ):
                                    _inc.complete_postmortem(_iid)
                                    st.rerun()

                        # Root cause
                        with _act2:
                            _rca = _incident.get("root_cause", {})
                            if not _rca.get("category"):
                                _rca_cat = st.selectbox(
                                    "Root Cause",
                                    ["ETL Failure", "Schema Change",
                                     "Source System Outage",
                                     "Code Deployment",
                                     "Data Volume Spike",
                                     "Network Issue", "Human Error",
                                     "Third Party", "Unknown"],
                                    key=f"inc_rca_{_iid}")
                                _rca_desc = st.text_input(
                                    "RCA Description",
                                    key=f"inc_rca_desc_{_iid}")
                                if st.button(
                                    "Set RCA",
                                    key=f"inc_set_rca_{_iid}",
                                ) and _rca_desc:
                                    _inc.set_root_cause(
                                        _iid, _rca_cat, _rca_desc,
                                        identified_by="User")
                                    st.rerun()
                            else:
                                st.write(
                                    f"**Root Cause:** "
                                    f"{_rca.get('category', '')}")
                                st.write(_rca.get("description", ""))

                        # RCA prompts
                        with _act3:
                            _prompts = _incident.get("rca_prompts", [])
                            if _prompts:
                                st.write("**Investigation prompts:**")
                                for _p in _prompts:
                                    st.caption(f"- {_p}")

                        # Impact
                        _impact = _incident.get(
                            "impact_assessment", {})
                        if _impact.get("business_impact"):
                            st.write(
                                f"**Business Impact:** "
                                f"{_impact['business_impact']}")

                        # Timeline
                        with st.expander("Timeline"):
                            for _ev in _incident.get("timeline", []):
                                st.caption(
                                    f"{_ev.get('at', '')[:16]} - "
                                    f"{_ev.get('event', '')} "
                                    f"({_ev.get('by', 'System')})")

                        # Postmortem export
                        if (_incident.get("postmortem_complete")
                                or _ist == "Resolved"):
                            if st.button(
                                "Generate Postmortem Report",
                                key=f"inc_pm_{_iid}",
                            ):
                                _pm = _inc.generate_postmortem_report(
                                    _iid)
                                if _pm:
                                    st.download_button(
                                        "Download Postmortem",
                                        _pm,
                                        f"postmortem_{_iid}.md",
                                        "text/markdown",
                                        key=f"inc_pm_dl_{_iid}")
            else:
                st.info(
                    "No incidents recorded. Incidents are created "
                    "automatically from observability checks "
                    "or manually.")

            # Export
            st.markdown("---")
            if st.button("Export All Incidents", key="inc_export"):
                _inc_data = _inc.export_incidents(fmt="json")
                if _inc_data:
                    st.download_button(
                        "Download", _inc_data,
                        "incidents.json", "application/json",
                        key="inc_dl")

        # ============================================================
        # SUBTAB 8: Data Observability (Monte Carlo Style)
        # ============================================================
        with dq_subtab8:
            st.markdown("#### Data Observability (Monte Carlo Style)")

            if not _dq.scan_results:
                st.info("Run a quality scan first (Quality Scanner tab) to enable observability checks.")
            else:
                _obs1, _obs2, _obs3 = st.columns(3)
                with _obs1:
                    st.markdown("**Schema Drift**")
                    if st.button("Check Schema", key="dq_schema_check"):
                        for _otn in list(_dq.scan_results.keys()):
                            if _otn in _dq_tables:
                                _drift = _dq.detect_schema_drift(
                                    _dq_tables[_otn], _otn)
                                if _drift.get("has_drift"):
                                    st.warning(
                                        f"{_otn}: Schema drift detected "
                                        f"({_drift['severity']})")
                                    if _drift.get("new_columns"):
                                        st.write(
                                            f"New columns: "
                                            f"{_drift['new_columns']}")
                                    if _drift.get("removed_columns"):
                                        st.write(
                                            f"Removed: "
                                            f"{_drift['removed_columns']}")
                                    for _tc in _drift.get(
                                            "type_changes", []):
                                        st.write(
                                            f"{_tc['column']}: "
                                            f"{_tc['old_type']} -> "
                                            f"{_tc['new_type']}")
                                else:
                                    st.success(f"{_otn}: No drift")

                with _obs2:
                    st.markdown("**Volume Monitoring**")
                    if st.button("Check Volume", key="dq_volume_check"):
                        for _otn in list(_dq.scan_results.keys()):
                            if _otn in _dq_tables:
                                _vol = _dq.detect_volume_anomaly(
                                    _dq_tables[_otn], _otn)
                                if _vol.get("is_anomaly"):
                                    st.warning(
                                        f"{_otn}: Volume anomaly "
                                        f"({_vol['severity']}) - "
                                        f"{_vol.get('change_pct', 0):.1f}%"
                                        f" change")
                                else:
                                    st.success(
                                        f"{_otn}: Volume normal "
                                        f"({_vol['current_rows']:,} rows)")

                with _obs3:
                    st.markdown("**Distribution Drift**")
                    if st.button(
                        "Check Distributions", key="dq_dist_check"
                    ):
                        for _otn in list(_dq.scan_results.keys()):
                            if _otn in _dq_tables:
                                _dist = _dq.detect_distribution_drift(
                                    _dq_tables[_otn], _otn)
                                _dcols = _dist.get("drifted_columns", [])
                                if _dcols:
                                    st.warning(
                                        f"{_otn}: {len(_dcols)} "
                                        f"columns drifted")
                                    for _dc in _dcols[:5]:
                                        st.write(f"  {_dc}")
                                else:
                                    st.success(
                                        f"{_otn}: No distribution drift")

                if st.button(
                    "Store Current as Baseline", key="dq_baseline"
                ):
                    for _otn in list(_dq.scan_results.keys()):
                        _bl_df = _dq_tables.get(_otn)
                        _dq.store_baseline(_otn, df=_bl_df)
                    st.success(
                        f"Baseline stored for "
                        f"{len(_dq.scan_results)} table(s)")

            st.markdown("---")
            st.markdown("#### Pipeline Integration API")

            from core.dq_api import DQAPIGateway

            st.markdown("Use the DQ API Gateway to integrate quality checks into your data pipelines.")

            with st.expander("API Reference"):
                _api_gw = DQAPIGateway()
                _api_spec = _api_gw.get_api_spec()
                st.json(_api_spec)

            with st.expander("Example: Quality Gate in Airflow"):
                st.code('''from core.dq_api import DQAPIGateway
from core.dq_engine import DataQualityEngine
import pandas as pd

# Initialize
dq = DataQualityEngine()
api = DQAPIGateway(dq_engine=dq)

# Load your data
df = pd.read_csv('transactions.csv')

# Quality gate - pipeline continues only if data passes
result = api.quality_gate(df, 'transactions', min_score=85, fail_on_critical=True)

if not result['passed']:
    raise Exception(f'Quality gate failed: {result["reason"]}')

print(f'Quality gate passed: {result["score"]}/100')
''', language="python")

            with st.expander("Example: Batch Scan in dbt"):
                st.code('''from core.dq_api import DQAPIGateway
from core.dq_engine import DataQualityEngine

api = DQAPIGateway(dq_engine=DataQualityEngine())

# Scan all tables from your dbt run
tables = {
    'orders': orders_df,
    'customers': customers_df,
    'products': products_df,
}

result = api.batch_scan(tables, quality_gate_score=90)

if not result['all_passed']:
    failed = [t for t, d in result['tables'].items() if not d['passed_gate']]
    raise Exception(f'Quality gates failed for: {failed}')
''', language="python")

        # ============================================================
        # SUBTAB 9: Advanced Analytics (Art of the Possible)
        # ============================================================
        with dq_subtab9:
            from core.advanced_dq import (
                benford_analysis, spc_analysis, entity_resolution,
                drift_detection, advanced_profile,
            )

            _safe_html(tab_header(
                "Advanced Analytics",
                "Benford's Law -- SPC Control Charts -- Entity Resolution -- Data Drift -- Deep Profiling",
                accent="#3B82F6",
                icon="//",
            ), "Advanced Analytics")

            _adv_df = None
            if "multi_session" in st.session_state and st.session_state.multi_session:
                try:
                    _adv_df = st.session_state.multi_session.get_primary_dataframe()
                except Exception:
                    pass

            if _adv_df is None or len(_adv_df) == 0:
                st.info("Upload a data file in the sidebar to activate Advanced Analytics.")
            else:
                adv_tab1, adv_tab2, adv_tab3, adv_tab4, adv_tab5 = st.tabs([
                    "Benford's Law", "SPC Charts", "Entity Resolution",
                    "Drift Detection", "Deep Profile",
                ])

                with adv_tab1:
                    st.markdown("### Benford's Law Analysis")
                    st.info(
                        "Benford's Law: In naturally occurring data, leading digits follow "
                        "a predictable distribution. Deviations may indicate data fabrication, "
                        "rounding errors, or systematic biases."
                    )
                    _bf_num_cols = [c for c in _adv_df.columns if pd.api.types.is_numeric_dtype(_adv_df[c])]
                    _bf_selected = st.multiselect("Select numeric columns", _bf_num_cols, default=_bf_num_cols[:4], key="bf_cols")
                    if st.button("Run Benford Analysis", type="primary", key="bf_run"):
                        with st.spinner("Analyzing leading digit distributions..."):
                            _bf_result = benford_analysis(_adv_df, _bf_selected)
                        if _bf_result.get("suspicious_columns"):
                            st.warning(f"{len(_bf_result['suspicious_columns'])} column(s) deviate from Benford's Law: {', '.join(_bf_result['suspicious_columns'])}")
                        else:
                            st.success("All analyzed columns conform to Benford's Law")
                        import plotly.graph_objects as go
                        for _bf_col, _bf_res in _bf_result["column_results"].items():
                            _bf_fig = go.Figure()
                            _bf_digits = list(range(1, 10))
                            _bf_fig.add_trace(go.Bar(x=_bf_digits, y=_bf_res["observed"], name="Observed", marker_color="#3B82F6", opacity=0.8))
                            _bf_fig.add_trace(go.Scatter(x=_bf_digits, y=_bf_res["expected"], name="Expected (Benford)", mode="lines+markers", line=dict(color="#00D47E", width=2)))
                            _bf_fig.update_layout(title=f"{_bf_col} -- {_bf_res['verdict']} (p={_bf_res['p_value']:.4f})", xaxis_title="Leading Digit", yaxis_title="Frequency", template="plotly_dark", height=280)
                            st.plotly_chart(_bf_fig, use_container_width=True)

                with adv_tab2:
                    st.markdown("### Statistical Process Control (SPC)")
                    st.info("SPC detects non-random patterns in time-series data using control limits and Western Electric Rules.")
                    _spc_date_cols = [c for c in _adv_df.columns if "date" in c.lower() or "time" in c.lower() or pd.api.types.is_datetime64_any_dtype(_adv_df[c])]
                    _spc_num_cols = [c for c in _adv_df.columns if pd.api.types.is_numeric_dtype(_adv_df[c])]
                    _spc_c1, _spc_c2 = st.columns(2)
                    with _spc_c1:
                        _spc_date = st.selectbox("Date/Time column", _spc_date_cols if _spc_date_cols else _adv_df.columns.tolist(), key="spc_date")
                    with _spc_c2:
                        _spc_val = st.selectbox("Metric column", _spc_num_cols, key="spc_val")
                    if st.button("Generate Control Chart", type="primary", key="spc_run"):
                        with st.spinner("Computing control limits..."):
                            _spc_r = spc_analysis(_adv_df, _spc_date, _spc_val)
                        if "error" not in _spc_r:
                            import plotly.graph_objects as go
                            _spc_fig = go.Figure()
                            _spc_fig.add_trace(go.Scatter(x=list(range(len(_spc_r["values"]))), y=_spc_r["values"], mode="lines+markers", name="Values", line=dict(color="#3B82F6")))
                            _spc_fig.add_hline(y=_spc_r["ucl"], line_color="#EF4444", line_dash="dash", annotation_text=f"UCL={_spc_r['ucl']:.2f}")
                            _spc_fig.add_hline(y=_spc_r["lcl"], line_color="#EF4444", line_dash="dash", annotation_text=f"LCL={_spc_r['lcl']:.2f}")
                            _spc_fig.add_hline(y=_spc_r["mean"], line_color="#00D47E", annotation_text=f"mean={_spc_r['mean']:.2f}")
                            _spc_fig.add_hline(y=_spc_r["uwl"], line_color="#F59E0B", line_dash="dot")
                            _spc_fig.add_hline(y=_spc_r["lwl"], line_color="#F59E0B", line_dash="dot")
                            _spc_fig.update_layout(title=f"X-bar Control Chart: {_spc_val} | Cpk={_spc_r['cpk']:.2f} | {_spc_r['violation_count']} violations", template="plotly_dark", height=350)
                            st.plotly_chart(_spc_fig, use_container_width=True)
                            if _spc_r["violations"]:
                                st.warning(f"{_spc_r['violation_count']} Western Electric rule violations detected")
                                st.dataframe(pd.DataFrame(_spc_r["violations"][:10]), use_container_width=True)
                            else:
                                st.success("Process appears stable -- no control rule violations")
                        else:
                            st.warning(_spc_r["error"])

                with adv_tab3:
                    st.markdown("### Entity Resolution -- Deduplication Intelligence")
                    st.info("Probabilistic entity matching using blocking + fuzzy similarity scoring (Fellegi-Sunter model).")
                    _er_cols = st.multiselect("Select key columns for matching", _adv_df.columns.tolist(), default=_adv_df.columns[:2].tolist(), key="er_cols")
                    _er_thresh = st.slider("Match threshold (%)", 60, 99, 85, key="er_thresh")
                    if st.button("Run Entity Resolution", type="primary", key="er_run") and _er_cols:
                        with st.spinner(f"Running probabilistic matching on {len(_adv_df)} records..."):
                            _er_r = entity_resolution(_adv_df, _er_cols, _er_thresh)
                        _er_c1, _er_c2, _er_c3 = st.columns(3)
                        _er_c1.metric("Duplicate Pairs Found", _er_r["duplicate_count"])
                        _er_c2.metric("Deduplication Rate", f"{_er_r['dedup_rate']}%")
                        _er_c3.metric("Total Records", _er_r["total_records"])
                        if _er_r["duplicate_pairs"]:
                            st.markdown("**Top Duplicate Pairs:**")
                            st.dataframe(pd.DataFrame(_er_r["duplicate_pairs"][:20]), use_container_width=True)
                        for _er_rec in _er_r["recommendations"]:
                            st.info(_er_rec)

                with adv_tab4:
                    st.markdown("### Data Drift Detection")
                    st.info("Compare current data distribution against a reference baseline using PSI and KS tests.")
                    if st.button("Run Drift Analysis (80/20 split)", type="primary", key="drift_run"):
                        _split_idx = int(len(_adv_df) * 0.8)
                        _ref_df = _adv_df.iloc[:_split_idx]
                        _cur_df = _adv_df.iloc[_split_idx:]
                        with st.spinner("Computing PSI and KS statistics..."):
                            _drift_r = drift_detection(_ref_df, _cur_df)
                        st.markdown(f"**Overall Drift: {_drift_r['drift_verdict']}** -- PSI={_drift_r['overall_psi']:.4f} | {_drift_r['drift_pct']}% columns drifted")
                        _drift_rows = []
                        for _dc, _dr in _drift_r["column_results"].items():
                            _drift_rows.append({"Column": _dc, "Type": _dr["type"], "PSI": _dr["psi"], "Severity": _dr["severity"], "Drifted": "YES" if _dr["drifted"] else "No"})
                        st.dataframe(pd.DataFrame(_drift_rows).sort_values("PSI", ascending=False), use_container_width=True)

                with adv_tab5:
                    st.markdown("### Deep Statistical Profile")
                    if st.button("Run Deep Profile", type="primary", key="deep_run"):
                        with st.spinner("Computing deep statistical profile..."):
                            _dp_r = advanced_profile(_adv_df)
                        _dp_c1, _dp_c2, _dp_c3 = st.columns(3)
                        _dp_c1.metric("Rows", _dp_r["shape"]["rows"])
                        _dp_c2.metric("Columns", _dp_r["shape"]["cols"])
                        _dp_c3.metric("Warnings", len(_dp_r["warnings"]))
                        if _dp_r["warnings"]:
                            for _dpw in _dp_r["warnings"][:10]:
                                st.warning(_dpw)
                        _dp_rows = []
                        for _dpc, _dpi in _dp_r["columns"].items():
                            _row = {"Column": _dpc, "Type": _dpi["dtype"], "Nulls %": _dpi["null_pct"], "Unique": _dpi["unique_count"], "Cardinality": _dpi["cardinality_ratio"]}
                            if "mean" in _dpi:
                                _row.update({"Mean": _dpi["mean"], "Std": _dpi["std"], "Skew": _dpi["skewness"]})
                            _dp_rows.append(_row)
                        st.dataframe(pd.DataFrame(_dp_rows), use_container_width=True)
                        if _dp_r["correlations"].get("high_correlations"):
                            st.markdown("**High Correlations Detected:**")
                            st.dataframe(pd.DataFrame(_dp_r["correlations"]["high_correlations"]), use_container_width=True)

            # DQ Advanced chatbot at bottom of Tab 2
            st.markdown("---")
            render_tab_chatbot(
                tab_key="dq_advanced",
                system_prompt=DQ_ENGINE_PROMPT,
                placeholder="Ask about quality scores, DAMA dimensions, Benford's Law, SPC, remediation strategies...",
                accent_color="#3B82F6",
                tab_label="Data Quality Intelligence",
                height=380,
            )


# ============================================================
# TAB 3: Dashboard Rationalization
# ============================================================
with tab3:
    st.title("Dashboard Rationalization Engine")
    st.caption(
        "Identify zombie dashboards, duplicates, refresh waste, and "
        "millions in hidden costs across your entire BI portfolio."
    )

    # Session state init
    if "dr_engine" not in st.session_state:
        st.session_state.dr_engine = DashboardRationalizationEngine()
    if "dr_inventory" not in st.session_state:
        st.session_state.dr_inventory = None
    if "dr_analysis" not in st.session_state:
        st.session_state.dr_analysis = None
    if "dr_report_html" not in st.session_state:
        st.session_state.dr_report_html = None

    _dr = st.session_state.dr_engine

    # ---- Load Data Section ----
    st.markdown("### Load Dashboard Inventory")
    _dr_col1, _dr_col2 = st.columns(2)

    with _dr_col1:
        _dr_upload = st.file_uploader(
            "Upload Dashboard Inventory CSV",
            type=["csv"],
            key="dr_upload",
        )
        if _dr_upload is not None and st.session_state.dr_inventory is None:
            try:
                _dr_df = pd.read_csv(_dr_upload)
                _dr.load_inventory(_dr_df)
                st.session_state.dr_inventory = _dr.inventory
                st.session_state.dr_analysis = None
                st.session_state.dr_report_html = None
                st.rerun()
            except Exception as _dr_err:
                st.error(f"Failed to load CSV: {_dr_err}")

    with _dr_col2:
        if st.button(
            "Generate Sample Enterprise Data (10,000 dashboards)",
            key="dr_generate_sample",
            type="primary",
        ):
            with st.spinner("Generating 10,000 sample dashboards..."):
                _dr_sample = _dr.generate_sample_data(num_dashboards=10000)
                _dr.load_inventory(_dr_sample)
                st.session_state.dr_inventory = _dr.inventory
                st.session_state.dr_analysis = None
                st.session_state.dr_report_html = None
            st.rerun()

    st.info(
        "CSV should include columns: dashboard_name, workspace, owner_email, "
        "platform, created_date, last_viewed_date, view_count_30d, "
        "refresh_schedule, source_tables, dataset_size_mb, num_visuals, is_certified"
    )

    # ---- Display loaded inventory ----
    if st.session_state.dr_inventory is not None:
        _dr_inv = st.session_state.dr_inventory
        st.markdown("---")

        # Metrics row
        _dr_m1, _dr_m2, _dr_m3, _dr_m4 = st.columns(4)
        _dr_m1.metric("Total Dashboards", f"{len(_dr_inv):,}")
        _dr_m2.metric(
            "Platforms",
            _dr_inv["platform"].nunique() if "platform" in _dr_inv.columns else 0,
        )
        _dr_m3.metric(
            "Workspaces",
            _dr_inv["workspace"].nunique() if "workspace" in _dr_inv.columns else 0,
        )
        _dr_m4.metric(
            "Owners",
            _dr_inv["owner_email"].nunique() if "owner_email" in _dr_inv.columns else 0,
        )

        with st.expander("Preview Inventory Data", expanded=False):
            st.dataframe(_dr_inv.head(200), use_container_width=True)

        # Run analysis button
        if st.button(
            "Run Full Rationalization Analysis",
            key="dr_run_analysis",
            type="primary",
        ):
            with st.status(
                "Analyzing dashboard portfolio...", expanded=True,
            ) as _dr_status:
                _dr_status.write("Detecting zombie dashboards...")
                _dr_status.write("Finding duplicate dashboard groups...")
                _dr_status.write("Calculating refresh waste...")
                _dr_status.write("Computing cost impact...")
                _dr_result = _dr.analyze()
                _dr_status.write("Generating recommendations...")
                _dr_status.write("Building retirement plan...")
                st.session_state.dr_analysis = _dr_result
                st.session_state.dr_report_html = None
                _dr_status.update(
                    label="Analysis complete", state="complete",
                )

    # ---- Display analysis results ----
    if st.session_state.dr_analysis:
        _dra = st.session_state.dr_analysis
        _dra_s = _dra["summary"]
        _dra_z = _dra["zombies"]
        _dra_d = _dra["duplicates"]
        _dra_c = _dra["cost_impact"]
        _dra_rp = _dra["retirement_plan"]
        _dra_rec = _dra["recommendations"]
        _dra_rw = _dra["refresh_waste"]

        st.markdown("---")
        st.markdown("### Analysis Results")

        # Hero metrics
        _drh1, _drh2, _drh3, _drh4 = st.columns(4)
        _drh1.metric(
            "Zombie Dashboards",
            f"{_dra_z['over_90d']['count']:,}",
            f"{_dra_z['over_90d']['pct']}% of portfolio",
        )
        _drh2.metric(
            "Annual Waste",
            f"${_dra_c['total_annual_waste']:,.0f}",
        )
        _drh3.metric(
            "Duplicate Groups",
            f"{_dra_d['total_groups']}",
            f"{_dra_d['total_dashboards_in_groups']:,} dashboards",
        )
        _drh4.metric(
            "Recommended Retirements",
            f"{_dra_rec['retire']['count']:,}",
        )

        st.divider()

        # Expandable detail sections
        with st.expander("Zombie Dashboards", expanded=False):
            _z_df = _dra_z.get("zombie_df")
            if _z_df is not None and len(_z_df) > 0:
                _z_show = _z_df[
                    ["dashboard_name", "workspace", "owner_email",
                     "platform", "days_since_viewed", "view_count_30d"]
                ].head(500)
                st.dataframe(_z_show, use_container_width=True)
                st.caption(f"Showing top 500 of {len(_z_df):,} zombie dashboards")
            else:
                st.write("No zombie dashboards found.")

        with st.expander("Duplicate Groups", expanded=False):
            if _dra_d["groups"]:
                for _dg_idx, _dg in enumerate(_dra_d["groups"][:30]):
                    st.markdown(
                        f"**Group {_dg_idx + 1}** -- {_dg['count']} dashboards "
                        f"sharing: `{_dg['source_tables'][:80]}`"
                    )
                    _dg_df = pd.DataFrame(_dg["dashboards"][:10])
                    st.dataframe(_dg_df, use_container_width=True)
                if _dra_d["total_groups"] > 30:
                    st.caption(
                        f"Showing 30 of {_dra_d['total_groups']} groups")
            else:
                st.write("No duplicate groups found.")

        with st.expander("Cost Impact Breakdown", expanded=False):
            _ci_data = {
                "Category": [
                    "Zombie Refresh Waste",
                    "Duplicate Maintenance",
                    "Analyst Search Time",
                    "Premium Capacity Waste",
                ],
                "Annual Cost": [
                    f"${_dra_c['zombie_refresh_waste']:,.2f}",
                    f"${_dra_c['duplicate_maintenance']:,.2f}",
                    f"${_dra_c['analyst_search_time']:,.2f}",
                    f"${_dra_c['premium_capacity_waste']:,.2f}",
                ],
            }
            st.table(pd.DataFrame(_ci_data))
            st.markdown(
                f"**Total Annual Waste: "
                f"${_dra_c['total_annual_waste']:,.0f}**"
            )

        with st.expander("Retirement Plan", expanded=False):
            for _rp_key in ("phase_1", "phase_2", "phase_3"):
                _rp = _dra_rp[_rp_key]
                st.markdown(
                    f"**{_rp['label']}** -- "
                    f"{_rp['count']:,} dashboards -- "
                    f"Timeline: {_rp['timeline']}"
                )

        st.divider()

        # Generate report button
        if st.button(
            "Generate Rationalization Report",
            key="dr_generate_report",
            type="primary",
        ):
            with st.status(
                "Generating HTML report...", expanded=True,
            ) as _dr_rpt_status:
                _dr_rpt_status.write("Building executive summary...")
                _dr_rpt_status.write("Rendering gauge charts...")
                _dr_rpt_status.write("Building workspace health table...")
                _dr_rpt_status.write("Generating cost breakdown...")
                _dr_rpt_status.write("Formatting retirement plan...")
                _dr_html = _dr.generate_html_report()
                st.session_state.dr_report_html = _dr_html
                _dr_rpt_status.update(
                    label="Report generated", state="complete",
                )

    # ---- Show report download + preview ----
    if st.session_state.dr_report_html:
        st.markdown("---")
        st.download_button(
            "Download Rationalization Report (HTML)",
            data=st.session_state.dr_report_html,
            file_name="dashboard_rationalization_report.html",
            mime="text/html",
            key="dr_download_report",
        )
        with st.expander("Report Preview", expanded=True):
            components.html(
                st.session_state.dr_report_html,
                height=800,
                scrolling=True,
            )

    # Tab 3 -- Rationalization chatbot
    st.markdown("---")
    render_tab_chatbot(
        tab_key="rationalization",
        system_prompt=RATIONALIZATION_EXPERT_PROMPT,
        placeholder="Ask about portfolio scoring, keep/consolidate/retire decisions, governance frameworks...",
        accent_color="#F59E0B",
        tab_label="Portfolio Rationalization Intelligence",
        context_injector=lambda: (
            f"Current portfolio: {len(st.session_state.get('dr_inventory', pd.DataFrame()))} dashboards analyzed"
            if st.session_state.get("dr_inventory") is not None
            else None
        ),
        height=400,
    )


with tab4:
    render_data_modeling()

    # ============================================
    # TAB 4 -- Dimensional Modeling Assistant
    # ============================================
    st.markdown("---")
    _safe_html(tab_header(
        "Dimensional Modeling Assistant",
        "Kimball -- Data Vault 2.0 -- Star Schema -- SQL Analyzer",
        accent="#A78BFA",
        icon="//",
    ), "Dimensional Modeling Assistant")

    modeling_tab1, modeling_tab2, modeling_tab3 = st.tabs([
        "Star Schema (Kimball)", "Data Vault 2.0", "SQL Analyzer",
    ])

    with modeling_tab1:
        _mod_df = None
        if "multi_session" in st.session_state and st.session_state.multi_session:
            try:
                _mod_df = st.session_state.multi_session.get_primary_dataframe()
            except Exception:
                pass

        if _mod_df is not None and len(_mod_df) > 0:
            _mod_tbl = st.text_input("Fact table name", value="fct_transactions", key="kim_tbl")
            if st.button("Infer Star Schema", type="primary", key="kim_run"):
                from core.data_modeling_engine import infer_dimensional_model
                with st.spinner("Analyzing data structure for dimensional model..."):
                    _kim_r = infer_dimensional_model(_mod_df, _mod_tbl)

                _km1, _km2, _km3 = st.columns(3)
                _km1.metric("Measures (Facts)", len(_kim_r["fact_table"]["suggested_measures"]))
                _km2.metric("Dimension Tables", len(_kim_r["dimension_tables"]))
                _km3.metric("Model Type", _kim_r["model_type"].replace("_", " ").title())

                for _kr in _kim_r["kimball_recommendations"]:
                    st.warning(_kr)

                with st.expander("Generated DDL -- Star Schema"):
                    st.code(_kim_r["star_schema_ddl"], language="sql")
        else:
            st.info("Upload a data file to auto-infer a Kimball dimensional model.")

    with modeling_tab2:
        _dv_df = None
        if "multi_session" in st.session_state and st.session_state.multi_session:
            try:
                _dv_df = st.session_state.multi_session.get_primary_dataframe()
            except Exception:
                pass

        if _dv_df is not None and len(_dv_df) > 0:
            _bk_candidates = [c for c in _dv_df.columns if any(kw in c.lower() for kw in ["id", "_key", "code", "number", "no"])]
            _bk_cols = st.multiselect("Select business key columns for Hubs", _dv_df.columns.tolist(), default=_bk_candidates[:2], key="dv_bk")
            if st.button("Generate Data Vault 2.0 Model", type="primary", key="dv_run") and _bk_cols:
                from core.data_modeling_engine import generate_data_vault_model
                with st.spinner("Generating Data Vault 2.0 skeleton..."):
                    _dv_r = generate_data_vault_model(_dv_df, _bk_cols)

                _dv1, _dv2 = st.columns(2)
                _dv1.metric("Hubs", len(_dv_r["hubs"]))
                _dv2.metric("Satellites", len(_dv_r["satellites"]))

                for _hub in _dv_r["hubs"]:
                    with st.expander(f"Hub: {_hub['hub_name']}"):
                        st.code(_hub["ddl"], language="sql")
                for _sat in _dv_r["satellites"]:
                    with st.expander(f"Satellite: {_sat['sat_name']}"):
                        st.code(_sat["ddl"], language="sql")

                st.markdown("**Data Vault Best Practices:**")
                for _dvr in _dv_r["recommendations"]:
                    st.info(_dvr)
        else:
            st.info("Upload a data file to generate a Data Vault model.")

    with modeling_tab3:
        _sql_input = st.text_area("Paste SQL to analyze", height=200, placeholder="SELECT * FROM sales JOIN customers ON ...", key="sql_in")
        if st.button("Analyze SQL", type="primary", key="sql_run") and _sql_input:
            from core.data_modeling_engine import analyze_sql
            _sql_r = analyze_sql(_sql_input)

            if _sql_r.get("issues"):
                for _si in _sql_r["issues"]:
                    _sev = _si.get("severity", "info")
                    if _sev == "error":
                        st.error(f"[ERROR] {_si['issue']}")
                    elif _sev == "warning":
                        st.warning(f"[WARN] {_si['issue']}")
                    else:
                        st.info(f"[INFO] {_si['issue']}")
            else:
                st.success("No SQL anti-patterns detected")

            if _sql_r.get("suggestions"):
                st.markdown("**Optimization suggestions:**")
                for _ss in _sql_r["suggestions"]:
                    st.info(_ss)

            _sq1, _sq2 = st.columns(2)
            _sq1.metric("Complexity Score", f"{_sql_r.get('complexity_score', 0):.1f}/10")
            _sq2.metric("JOIN Count", _sql_r.get("join_count", 0))

    # Data Modeler Chatbot
    st.markdown("---")
    render_tab_chatbot(
        tab_key="modeler",
        system_prompt=MODELER_EXPERT_PROMPT,
        placeholder="Ask about Kimball vs Data Vault, star schema design, DAX semantic models, SCD types...",
        accent_color="#A78BFA",
        tab_label="Data Architecture Intelligence",
        height=450,
    )
