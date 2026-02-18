"""
Dr. Data -- Agentic Data Intelligence Platform
Full-width chat interface with sidebar file management.
"""
import sys
import io
import os
import json
import time
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
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from app.dr_data_agent import DrDataAgent
from app.file_handler import ingest_file, ALL_SUPPORTED
from core.multi_file_handler import MultiFileSession
from core.audit_engine import AuditEngine


# === PAGE CONFIG (must be first Streamlit call) ===
st.set_page_config(
    page_title="Dr. Data -- Dashboard Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# === UNIFIED CSS -- Western Union Enterprise Theme ===
st.markdown('''<style>
/* ===== WU GLOBAL THEME ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

/* Main app background */
.stApp, [data-testid="stAppViewContainer"] {
    background: #0D0D0D !important;
}

/* Remove default Streamlit padding/margins */
section.main > div.block-container {
    max-width: 100% !important;
    padding: 0 1.5rem 140px 1.5rem !important;
}
header[data-testid="stHeader"] { display: none !important; }

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: #1A1A1A !important;
    border-right: 1px solid #3D3D3D !important;
    width: 300px !important;
    min-width: 300px !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem 1rem 2rem 1rem !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #B3B3B3 !important;
    font-size: 13px !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton button {
    background: #FFE600 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #E6CF00 !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #262626 !important;
    border: 1px dashed #3D3D3D !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: #FFE600 !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #B3B3B3 !important;
    font-size: 12px !important;
}

/* Sidebar selectbox / multiselect */
section[data-testid="stSidebar"] [data-testid="stSelectbox"],
section[data-testid="stSidebar"] [data-testid="stMultiSelect"] {
    background: #262626 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #262626 !important;
    border: 1px solid #3D3D3D !important;
    color: #FFFFFF !important;
}

/* Sidebar text input */
section[data-testid="stSidebar"] .stTextInput input {
    background: #262626 !important;
    border: 1px solid #3D3D3D !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #FFE600 !important;
    box-shadow: 0 0 0 1px #FFE600 !important;
}

/* Sidebar slider */
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: #FFE600 !important; }
section[data-testid="stSidebar"] .stSlider [role="slider"] { background: #FFE600 !important; }

/* Sidebar success/info/warning boxes */
section[data-testid="stSidebar"] .stAlert { border-radius: 6px !important; font-size: 12px !important; }
div[data-testid="stNotification"] { border-radius: 6px !important; }

/* Sidebar expander */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #262626 !important;
    border: 1px solid #3D3D3D !important;
    border-radius: 6px !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr { border-color: #3D3D3D !important; margin: 12px 0 !important; }

/* ===== CHAT AREA ===== */
div[data-testid="stChatMessageContainer"] {
    height: calc(100vh - 180px) !important;
    max-height: calc(100vh - 180px) !important;
    overflow-y: auto !important;
    padding: 16px 8px !important;
    scroll-behavior: smooth !important;
}

/* Chat input pinned to bottom */
div[data-testid="stBottom"] {
    position: fixed !important;
    bottom: 0 !important;
    background: linear-gradient(transparent, #0D0D0D 20%) !important;
    padding: 20px 1.5rem 16px 1.5rem !important;
    z-index: 999 !important;
}

/* Chat input box */
div[data-testid="stChatInput"] textarea {
    background: #1A1A1A !important;
    border: 2px solid #3D3D3D !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 14px !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    min-height: 48px !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border-color: #FFE600 !important;
    box-shadow: 0 0 0 1px rgba(255,230,0,0.3) !important;
}
div[data-testid="stChatInput"] textarea::placeholder {
    color: #808080 !important;
    -webkit-text-fill-color: #808080 !important;
}
div[data-testid="stChatInput"] button {
    color: #FFE600 !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 12px 16px !important;
    margin-bottom: 4px !important;
    line-height: 1.65 !important;
    font-size: 14px !important;
    color: #FFFFFF !important;
    max-width: 95% !important;
    border-radius: 0 !important;
    border-bottom: 1px solid rgba(61,61,61,0.3) !important;
}
div[data-testid="stChatMessage"] p { color: #FFFFFF !important; line-height: 1.65 !important; }
div[data-testid="stChatMessage"] li { color: #E0E0E0 !important; }
div[data-testid="stChatMessage"] strong { color: #FFE600 !important; }
div[data-testid="stChatMessage"] code { background: #262626 !important; color: #FFE600 !important; padding: 2px 6px !important; border-radius: 4px !important; font-size: 12px !important; }
div[data-testid="stChatMessage"] a { color: #FFE600 !important; text-decoration: none !important; }
div[data-testid="stChatMessage"] a:hover { text-decoration: underline !important; }

/* User message avatar area */
div[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(255,230,0,0.03) !important;
}

/* ===== STATUS WIDGET (Progress) ===== */
div[data-testid="stStatusWidget"] {
    background: #1A1A1A !important;
    border: 1px solid #3D3D3D !important;
    border-radius: 8px !important;
    margin: 8px 0 !important;
}
div[data-testid="stStatusWidget"] p { color: #B3B3B3 !important; font-size: 13px !important; }
div[data-testid="stStatusWidget"] summary { color: #FFFFFF !important; font-weight: 500 !important; }

/* ===== DOWNLOAD BUTTONS ===== */
.stDownloadButton button {
    background: #FFE600 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(255,230,0,0.15) !important;
}
.stDownloadButton button:hover {
    background: #E6CF00 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(255,230,0,0.25) !important;
}

/* ===== DATA TABLES ===== */
div[data-testid="stDataFrame"] {
    background: #1A1A1A !important;
    border: 1px solid #3D3D3D !important;
    border-radius: 8px !important;
}

/* ===== TABS ===== */
button[data-baseweb="tab"] { color: #B3B3B3 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #FFE600 !important; border-bottom-color: #FFE600 !important; }

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D0D0D; }
::-webkit-scrollbar-thumb { background: #3D3D3D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #FFE600; }

/* ===== METRICS ===== */
div[data-testid="stMetric"] {
    background: #1A1A1A !important;
    border: 1px solid #3D3D3D !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
div[data-testid="stMetric"] label { color: #B3B3B3 !important; font-size: 12px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFE600 !important; }

/* ===== SPINNER ===== */
.stSpinner > div { border-top-color: #FFE600 !important; }

/* ===== RADIO / CHECKBOX ===== */
.stRadio label, .stCheckbox label { color: #FFFFFF !important; }

/* ===== CHAT PANEL BORDER ===== */
div[data-testid="column"]:last-child {
    border-left: 1px solid #3D3D3D;
    padding-left: 16px !important;
}

/* ===== WORKSPACE AREA ===== */
.workspace-card {
    background: #1A1A1A;
    border: 1px solid #3D3D3D;
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
    border-bottom: 1px solid #3D3D3D;
}

/* ===== KPI CARDS ===== */
.kpi-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
}
.kpi-card {
    flex: 1;
    background: linear-gradient(135deg, #1A1A1A, #262626);
    border: 1px solid #3D3D3D;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-card .kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #FFE600;
    line-height: 1.2;
}
.kpi-card .kpi-label {
    font-size: 10px;
    color: #808080;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}

/* ===== SCORE BADGE ===== */
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
.score-green { color: #34D399; border-color: #34D399; }
.score-amber { color: #FBBF24; border-color: #FBBF24; }
.score-red { color: #F87171; border-color: #F87171; }

/* ===== PROGRESS INDICATOR ===== */
.phase-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: #262626;
    border: 1px solid #3D3D3D;
    border-left: 3px solid #FFE600;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 13px;
    color: #FFFFFF;
}
.phase-status.complete {
    border-left-color: #34D399;
}
.phase-status .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #FFE600;
    animation: pulse 1.5s infinite;
}
.phase-status.complete .dot {
    background: #34D399;
    animation: none;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Chat messages custom classes */
.dr-msg {
    background: #1A1A1A;
    border: 1px solid #3D3D3D;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
    color: #FFFFFF;
}
.user-msg {
    background: #262626;
    border: 1px solid #3D3D3D;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 13px;
    color: #B3B3B3;
    text-align: right;
}
.dr-name {
    font-size: 11px;
    font-weight: 600;
    color: #FFE600;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
}

/* Download cards */
.dl-card {
    background: #262626;
    border: 1px solid #FFE600;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
}
.dl-card .dl-name {
    font-size: 12px;
    font-weight: 600;
    color: #FFE600;
}
.dl-card .dl-desc {
    font-size: 11px;
    color: #B3B3B3;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

</style>''', unsafe_allow_html=True)


# ============================================
# SESSION STATE
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Welcome. Upload a file in the sidebar and tell me what you "
            "need -- dashboards, Power BI, PowerPoints, reports. "
            "I will build it."
        ),
        "downloads": [],
        "timestamp": 0,
    }]

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

if "agent" not in st.session_state:
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
# HEADER BAR
# ============================================
st.markdown('''
<div style="background:#0D0D0D;padding:10px 20px;display:flex;align-items:center;gap:14px;border-bottom:2px solid #FFE600;margin:-1rem -1.5rem 0.5rem -1.5rem;flex-wrap:wrap;">
    <div style="width:32px;height:32px;background:#FFE600;border-radius:6px;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </div>
    <span style="font-size:17px;font-weight:700;color:#FFF;letter-spacing:-0.3px;">Dr. Data</span>
    <span style="font-size:11px;color:#666;font-weight:400;">Agentic Data Intelligence Platform</span>
    <span style="margin-left:auto;font-size:10px;color:#666;letter-spacing:0.5px;">WESTERN UNION</span>
</div>
''', unsafe_allow_html=True)


# ============================================
# SIDEBAR -- File Upload + Audience Toggle
# ============================================
with st.sidebar:
    st.markdown('''
<div style="text-align:center;padding:8px 0 16px 0;border-bottom:1px solid #3D3D3D;margin-bottom:16px;">
    <div style="font-size:14px;font-weight:700;color:#FFE600;letter-spacing:0.5px;">DR. DATA</div>
    <div style="font-size:10px;color:#808080;letter-spacing:1px;text-transform:uppercase;margin-top:2px;">Intelligence Suite</div>
</div>
''', unsafe_allow_html=True)
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
# FULL-WIDTH CHAT INTERFACE
# ============================================

# 1. Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("downloads"):
            _render_downloads(
                msg["downloads"],
                key_prefix="hist",
                ts=msg.get("timestamp", 0),
            )

# 2. Handle auto-analysis on file upload
if st.session_state.file_just_uploaded:
    file_info = st.session_state.file_just_uploaded
    st.session_state.file_just_uploaded = None

    names = file_info.get("names", [])
    name_str = ", ".join(names)

    ws = st.session_state.workspace_content
    ws["phase"] = "analyzing"

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

            agent = st.session_state.agent
            if agent.dataframe is not None and ws.get("data_preview") is None:
                ws["data_preview"] = agent.dataframe

        except Exception:
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"I have loaded your files: {name_str}. "
                    f"The data looks great and is ready to go! "
                    f"What would you like me to build?"
                ),
                "timestamp": time.time(),
            })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I have loaded {name_str}. What would you like me to build?",
            "timestamp": time.time(),
        })

    st.rerun()

# 3. Chat input
if prompt := st.chat_input("Ask Dr. Data anything...", key="chat_input"):
    now = time.time()

    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": now,
    })

    if not st.session_state.agent:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I am warming up -- give me one moment and try again.",
            "timestamp": now,
        })
        st.rerun()

    agent = st.session_state.agent

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        status = st.status("Dr. Data is working...", expanded=True)
        response_container = st.empty()
        download_container = st.container()

        status.write("Received your request.")

        msg_lower = prompt.lower() if isinstance(prompt, str) else ""
        is_export = any(kw in msg_lower for kw in [
            "power bi", "powerbi", "pbi", "dashboard", "html",
            "powerpoint", "pptx", "slides", "pdf", "word", "docx",
        ])

        if is_export:
            if any(kw in msg_lower for kw in ["power bi", "powerbi", "pbi"]):
                status.write("Preparing Power BI generation pipeline...")
            if any(kw in msg_lower for kw in ["dashboard", "html"]):
                status.write("Preparing interactive dashboard builder...")
            if any(kw in msg_lower for kw in ["powerpoint", "pptx", "slides"]):
                status.write("Preparing PowerPoint generator...")
        else:
            status.write("Analyzing your question...")

        response = None
        try:
            response = agent.respond(
                prompt,
                st.session_state.messages,
                st.session_state.uploaded_files,
            )

            if is_export:
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
                                ext = os.path.splitext(file_name)[1].lower()
                                st.download_button(
                                    label=f"Download {file_name}",
                                    data=file_bytes,
                                    file_name=file_name,
                                    mime=_MIME_TYPES.get(ext, "application/octet-stream"),
                                    key=f"dl_{file_name}_{int(time.time() * 1000)}_{idx}",
                                )
                            else:
                                st.warning(f"File not found: {file_path}")
                else:
                    status.update(label="Done", state="complete", expanded=False)
                    if content:
                        response_container.markdown(content)
                    else:
                        response_container.warning("No output generated.")

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

        # Save to history
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
                ws["deliverables"].extend(dl_list)
                ws["phase"] = "complete"
            if agent.dataframe is not None:
                if ws.get("data_preview") is None:
                    ws["data_preview"] = agent.dataframe
