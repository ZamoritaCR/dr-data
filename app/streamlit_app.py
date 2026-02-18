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


def _safe_html(html_str, fallback_text=""):
    """Render HTML via st.markdown with fallback to st.write on error."""
    try:
        st.markdown(html_str, unsafe_allow_html=True)
    except Exception:
        st.write(fallback_text or html_str)


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

WU_LOGO = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 260 40" width="156" height="24"><rect width="260" height="40" rx="4" fill="#FFE600"/><text x="14" y="28" font-family="Arial,Helvetica,sans-serif" font-size="20" font-weight="900" fill="#000000" letter-spacing="1">WESTERN UNION</text></svg>'


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
        st.session_state.agent = DrDataAgent()
    except Exception as e:
        st.error(f"Failed to initialize Dr. Data: {e}")
        st.session_state.agent = None

if "multi_session" not in st.session_state:
    st.session_state.multi_session = MultiFileSession()

if "file_just_uploaded" not in st.session_state:
    st.session_state.file_just_uploaded = None

if "audit_engine" not in st.session_state:
    st.session_state.audit_engine = AuditEngine()

if "audience_mode" not in st.session_state:
    st.session_state.audience_mode = "analyst"


# ============================================
# HEADER (with avatar)
# ============================================
_safe_html(f'<div class="top-header"><div class="header-left">{DR_DATA_AVATAR}<h1>Dr. Data</h1><span class="role">|&nbsp; Chief Data Intelligence Officer</span></div><div>{WU_LOGO}</div></div>', "Dr. Data -- Chief Data Intelligence Officer")


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
    )

    # Handle uploads through MultiFileSession
    if uploaded_files_list:
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
                except Exception:
                    pass

            st.session_state.file_just_uploaded = {"names": new_files}

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
            keep = dict(st.session_state.uploaded_files) if st.session_state.get("uploaded_files") else {}
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.uploaded_files = keep
            st.rerun()
    with col_new:
        if st.button("New Session", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
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
tab1, tab2 = st.tabs(["Dr. Data Agent", "Data Quality Engine"])

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
                    _build_status.write("Starting generation pipeline...")
                    _response = None
                    try:
                        _response = _agent.respond(
                            _export_prompt,
                            st.session_state.messages,
                            st.session_state.uploaded_files,
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
                'The Art of the Possible</div>'
                '<div style="font-size:13px;color:#808080;margin-bottom:24px;">'
                'Upload data in the sidebar, or click a card to get started</div>'
                '</div>',
                "The Art of the Possible -- Upload data or click a card to get started."
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
        _safe_html(f'<div style="padding:4px 0 8px 0;border-bottom:1px solid #4a4a4a;margin-bottom:8px;display:flex;align-items:center;gap:8px;">{DR_DATA_AVATAR}<span style="font-size:13px;font-weight:600;color:#FFFFFF;">Chat with Dr. Data</span><span style="font-size:11px;color:#B0B0B0;">|&nbsp; The Art of the Possible</span></div>', "Chat with Dr. Data -- The Art of the Possible")

        # Chat container with scroll
        chat_container = st.container()

        with chat_container:
            # Opening message if empty
            if not st.session_state.messages:
                with st.chat_message("assistant"):
                    st.markdown(
                        "Hello! Great to have you here. I am Dr. Data, your personal "
                        "data intelligence partner. I believe in **The Art of the Possible** "
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

                    msg_lower = prompt.lower() if isinstance(prompt, str) else ""
                    is_export = any(kw in msg_lower for kw in [
                        "power bi", "powerbi", "pbi", "pbip", "pbix",
                        "dashboard", "html", "interactive",
                        "powerpoint", "pptx", "slides", "presentation", "deck",
                        "pdf", "report", "word", "docx", "document",
                        "all formats", "all three", "everything",
                    ])

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
                        if st.session_state.uploaded_files and agent.dataframe is None:
                            for name, info in st.session_state.uploaded_files.items():
                                path = info.get("path", "")
                                if path and os.path.exists(path):
                                    agent.data_file_path = path
                                    agent.data_path = path
                                    ext = info.get("ext", path.rsplit(".", 1)[-1].lower())
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
                                        if agent.dataframe is not None and agent.data_profile is None:
                                            agent.data_profile = agent.analyzer.profile(agent.dataframe)
                                    except Exception:
                                        pass
                                    break

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
                with st.status(
                    "Running DAMA-DMBOK Quality Assessment...",
                    expanded=True,
                ) as _dq_status:
                    for _tn, _td in _dq_to_scan.items():
                        _dq_status.write(
                            f"Scanning {_tn} ({len(_td):,} rows x "
                            f"{len(_td.columns)} cols)..."
                        )
                        try:
                            _dq.scan_table(_td, _tn)
                            _dq_status.write(
                                f"{_tn}: Score = "
                                f"{_dq.scan_results[_tn]['overall_score']:.1f}"
                                f"/100"
                            )
                        except Exception as _e:
                            _dq_status.write(
                                f"{_tn}: Error - {str(_e)[:200]}")

                    if len(_dq_to_scan) > 1:
                        _dq_status.write("Running cross-table analysis...")
                        try:
                            _dq_cross = _dq.scan_multiple_tables(_dq_to_scan)
                            st.session_state.dq_cross_results = _dq_cross
                        except Exception as _e:
                            _dq_status.write(
                                f"Cross-table error: {str(_e)[:200]}")

                    _dq_status.update(
                        label=f"Scan complete: {len(_dq_to_scan)} table(s) "
                              f"assessed",
                        state="complete",
                    )

                    # Bridge: pass DQ results into Dr. Data agent context
                    if "agent" in st.session_state and st.session_state.agent:
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
                _html = _dq.generate_html_scorecard()
                if _html:
                    st.download_button(
                        "Download HTML Scorecard", _html,
                        "dq_scorecard.html", "text/html",
                        key="dq_dl_html",
                    )
                    import streamlit.components.v1 as _components
                    _components.html(_html, height=800, scrolling=True)

            # ── Data Observability (Monte Carlo Style) ──
            st.markdown("---")
            st.markdown("#### Data Observability (Monte Carlo Style)")

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
