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

WU_LOGO = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 40" width="120" height="24"><rect width="200" height="40" rx="4" fill="#FFDE00"/><text x="12" y="28" font-family="Arial,Helvetica,sans-serif" font-size="20" font-weight="900" fill="#1a1a1a" letter-spacing="1">WESTERN UNION</text></svg>'


# === CUSTOM CSS -- THE ENTIRE LOOK (Western Union) ===
st.markdown("""
<style>
    /* Kill Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Dark foundation */
    .stApp {
        background-color: #1a1a1a;
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
        background: #1a1a1a !important;
        border-top: 2px solid #FFDE00 !important;
        padding: 12px 2rem !important;
        z-index: 999 !important;
    }
    div[data-testid="stBottomBlockContainer"] {
        background: #1a1a1a !important;
        padding: 10px 20px !important;
        border-top: 2px solid #FFDE00 !important;
    }

    /* Chat input wider and styled */
    div[data-testid='stChatInput'] {
        max-width: 100% !important;
    }
    div[data-testid='stChatInput'] textarea,
    div[data-testid='stChatInput'] div[contenteditable='true'] {
        background: #0d1117 !important;
        border: 2px solid #FFDE00 !important;
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
        background: #1a1a1a;
    }

    /* Download buttons inside chat */
    div[data-testid='stChatMessage'] .stDownloadButton button {
        background: #FFDE00 !important;
        color: #1a1a1a !important;
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
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-bottom: 1px solid #4a4a4a;
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
        background: linear-gradient(90deg, #FFDE00, #FFE600, #FFDE00);
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
        background: #1a1a1a !important;
        border-right: 1px solid #4a4a4a;
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
        background-color: #FFDE00 !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 8px 20px !important;
        border-radius: 6px !important;
        width: 100% !important;
        margin-top: 4px !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #FFE600 !important;
    }

    /* Make data tables dark */
    .stDataFrame {
        border: 1px solid #4a4a4a;
        border-radius: 8px;
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
# MAIN LAYOUT: Workspace (left 65%) + Chat (right 35%)
# ============================================
workspace_col, chat_col = st.columns([65, 35])


# ============================================
# LEFT: WORKSPACE -- Shows Dr. Data's work
# ============================================
with workspace_col:
    ws = st.session_state.workspace_content

    if ws["phase"] == "waiting" and ws.get("data_preview") is None:
        # Empty state
        avatar_lg = DR_DATA_AVATAR.replace('width="36" height="36"', 'width="64" height="64"')
        _safe_html(f'<div style="text-align:center;padding:80px 40px;color:#B0B0B0;"><div style="margin-bottom:16px;opacity:0.6;">{avatar_lg}</div><div style="font-size:16px;font-weight:500;color:#FFFFFF;margin-bottom:8px;">Welcome! I am Dr. Data.</div><div style="font-size:13px;max-width:440px;margin:0 auto;line-height:1.6;">I am here to help you unlock the full potential of your data. Upload a file in the sidebar -- CSV, Excel, Tableau, Alteryx, whatever you have -- and let me show you The Art of the Possible. I can build dashboards, reports, PowerPoints, and more. Let us get started!</div></div>', "Welcome! I am Dr. Data. Upload a file in the sidebar to get started.")

    else:
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
                # Re-render download buttons for assistant messages that had them
                if msg["role"] == "assistant" and msg.get("downloads"):
                    _render_downloads(
                        msg["downloads"],
                        key_prefix="hist",
                        ts=msg.get("timestamp", 0),
                    )

    # === HANDLE AUTO-ANALYSIS ON FILE UPLOAD ===
    if st.session_state.file_just_uploaded:
        file_info = st.session_state.file_just_uploaded
        st.session_state.file_just_uploaded = None

        names = file_info.get("names", [])
        name_str = ", ".join(names)

        ws = st.session_state.workspace_content
        ws["phase"] = "analyzing"
        ws["progress_messages"].append(f"Loaded {name_str}")
        ws["progress_messages"].append("Running deep analysis...")

        if st.session_state.agent:
            try:
                response = st.session_state.agent.analyze_uploaded_file()
                if not response or not str(response).strip():
                    response = (
                        f"I have loaded your files: {name_str}. "
                        f"The data looks great and is ready to go! "
                        f"What would you like me to build?"
                    )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": time.time(),
                })
                ws["phase"] = "analyzed"
                ws["progress_messages"].append(
                    "Analysis complete -- audit passed"
                    if ws.get("audit_releasable")
                    else "Analysis complete -- review audit findings"
                )

                agent = st.session_state.agent
                if agent.dataframe is not None and ws.get("data_preview") is None:
                    ws["data_preview"] = agent.dataframe

            except Exception:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"I have loaded your files: {name_str}. "
                        f"The data looks great and is ready to go! "
                        f"What would you like me to build? I can create "
                        f"dashboards, reports, presentations -- whatever "
                        f"helps you most. The Art of the Possible starts here!"
                    ),
                    "timestamp": time.time(),
                })
        else:
            preview_note = ""
            if ws.get("data_preview") is not None:
                df = ws["data_preview"]
                preview_note = (
                    f" I can see {len(df):,} rows and {len(df.columns)} columns: "
                    f"{', '.join(df.columns[:6].tolist())}"
                    f"{'...' if len(df.columns) > 6 else ''}."
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"I have loaded {name_str}.{preview_note} "
                    f"There is a lot we can do with this! I can build "
                    f"interactive HTML dashboards, PDF reports, Power BI "
                    f"projects, PowerPoint presentations, or a full "
                    f"documentation package. Tell me who your audience "
                    f"is and what story you want to tell -- I will take "
                    f"it from there!"
                ),
                "timestamp": time.time(),
            })

        st.rerun()

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
                    # ====== EXPORT PATH -- status widget + blocking respond() ======
                    status = st.status("Dr. Data is working...", expanded=True)
                    response_container = st.empty()
                    download_container = st.container()

                    status.write("Received your request.")

                    if any(kw in msg_lower for kw in ["power bi", "powerbi", "pbi"]):
                        status.write("Preparing Power BI generation pipeline...")
                        status.write("Step 1: Profiling your data...")
                    if any(kw in msg_lower for kw in ["dashboard", "html"]):
                        status.write("Preparing interactive dashboard builder...")
                    if any(kw in msg_lower for kw in ["powerpoint", "pptx", "slides"]):
                        status.write("Preparing PowerPoint generator...")

                    response = None
                    try:
                        response = agent.respond(
                            prompt,
                            st.session_state.messages,
                            st.session_state.uploaded_files,
                        )

                        status.write("Building your deliverables...")

                        if response is None:
                            status.update(label="Something went wrong", state="error", expanded=False)
                            response_container.warning("Dr. Data hit a snag. Try again.")

                        elif isinstance(response, dict):
                            content = response.get("content", "")
                            downloads = response.get("downloads", []) or []
                            engine = response.get("engine", "claude")

                            if downloads:
                                status.write(f"Generated {len(downloads)} file(s).")
                                status.update(label="Done", state="complete", expanded=False)

                                if content:
                                    response_container.markdown(content)

                                with download_container:
                                    for idx, dl in enumerate(downloads):
                                        file_path = dl.get("path", "")
                                        file_name = dl.get("filename", dl.get("name", f"file_{idx}"))
                                        if file_path and os.path.exists(str(file_path)):
                                            with open(file_path, "rb") as f:
                                                file_bytes = f.read()
                                            mime_map = {
                                                ".html": "text/html",
                                                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                                ".pdf": "application/pdf",
                                                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                                ".zip": "application/zip",
                                                ".csv": "text/csv",
                                            }
                                            ext = os.path.splitext(file_name)[1].lower()
                                            st.download_button(
                                                label=f"Download {file_name}",
                                                data=file_bytes,
                                                file_name=file_name,
                                                mime=mime_map.get(ext, "application/octet-stream"),
                                                key=f"dl_{file_name}_{int(time.time() * 1000)}_{idx}",
                                            )
                                        else:
                                            st.warning(f"File not found: {file_path}")
                            else:
                                status.update(label="Done", state="complete", expanded=False)
                                if content:
                                    response_container.markdown(content)
                                else:
                                    response_container.warning("No output generated. Check terminal for errors.")

                            engine_colors = {"claude": "#B39DDB", "openai": "#81C784", "gemini": "#FFB74D"}
                            st.markdown(
                                f'<div style="text-align:right;font-size:10px;'
                                f'color:{engine_colors.get(engine, "#808080")};'
                                f'opacity:0.5;">{engine.title()}</div>',
                                unsafe_allow_html=True,
                            )

                        elif isinstance(response, str) and response.strip():
                            status.update(label="Done", state="complete", expanded=False)
                            response_container.markdown(response)

                        else:
                            status.update(label="No response", state="error", expanded=False)
                            response_container.warning("Dr. Data returned empty. Try rephrasing.")

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        status.update(label="Error", state="error", expanded=False)
                        response_container.error(f"Error: {str(e)[:300]}")

                    # Save export response to history
                    if response:
                        msg_data = {
                            "role": "assistant",
                            "content": (
                                response.get("content", response)
                                if isinstance(response, dict) else response
                            ),
                            "downloads": (
                                response.get("downloads", [])
                                if isinstance(response, dict) else []
                            ),
                            "timestamp": now,
                        }
                        st.session_state.messages.append(msg_data)

                        ws = st.session_state.workspace_content
                        dl_list = msg_data["downloads"] or []
                        if dl_list:
                            for dl in dl_list:
                                if os.path.exists(dl.get("path", "")):
                                    try:
                                        dl_audit = (
                                            st.session_state.audit_engine
                                            .audit_deliverable(
                                                dl["path"],
                                                file_type=dl.get(
                                                    "filename", ""
                                                ).split(".")[-1],
                                            )
                                        )
                                        dl_audit.compute_scores()
                                        dl["audit_score"] = dl_audit.overall_score
                                        dl["audit_releasable"] = dl_audit.is_releasable
                                    except Exception:
                                        pass
                            ws["deliverables"].extend(dl_list)
                            ws["phase"] = "complete"
                            ws["progress_messages"].append(
                                "Deliverables ready -- audited"
                            )
                        if agent.dataframe is not None:
                            if ws.get("data_preview") is None:
                                ws["data_preview"] = agent.dataframe

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
                        for idx, dl in enumerate(chat_downloads):
                            fpath = dl["path"]
                            fname = dl["filename"]
                            with open(fpath, "rb") as f:
                                file_bytes = f.read()
                            mime_map = {
                                ".html": "text/html",
                                ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                ".pdf": "application/pdf",
                                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                ".zip": "application/zip",
                                ".csv": "text/csv",
                            }
                            ext = os.path.splitext(fname)[1].lower()
                            st.download_button(
                                label=f"Download {fname}",
                                data=file_bytes,
                                file_name=fname,
                                mime=mime_map.get(ext, "application/octet-stream"),
                                key=f"dl_{fname}_{int(time.time() * 1000)}_{idx}",
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
