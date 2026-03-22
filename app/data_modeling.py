"""
Data Modeler tab for Dr. Data.
Full-fidelity AI chat (same pattern as main Dr. Data Agent tab)
specialized for data modeling, plus visual sub-tabs.
"""
import os
import time

import streamlit as st
import streamlit.components.v1 as components

from app.dr_data_agent import DrDataAgent


# -- Data Modeling context injection ------------------------------------------
# This gets prepended to every user message so the agent stays in modeling mode
# regardless of what the base system prompt says.

_DM_CONTEXT_PREFIX = (
    "[SYSTEM CONTEXT -- DATA MODELER MODE]\n"
    "The user is in the Data Modeler tab. You are operating as a data modeling "
    "specialist. Focus ALL responses on:\n"
    "- Star schema, snowflake schema, galaxy schema design\n"
    "- Fact vs dimension table classification and grain definition\n"
    "- Surrogate keys, natural keys, degenerate dimensions\n"
    "- Slowly changing dimensions (SCD Type 1/2/3/4/6)\n"
    "- Bridge tables, factless fact tables, accumulating snapshots\n"
    "- DAX optimization (CALCULATE, context transition, iterator vs aggregator)\n"
    "- Power Query M transformations and folding\n"
    "- SQL DDL design, indexing strategies, partitioning\n"
    "- Tableau data modeling (relationships, LOD expressions, data blending)\n"
    "- Snowflake warehouse design (clustering keys, micro-partitions, zero-copy clones)\n"
    "- ETL/ELT architecture and incremental load patterns\n"
    "- Platform migration (Tableau to Power BI, on-prem to cloud)\n"
    "- Data vault, anchor modeling, and other advanced patterns\n"
    "- Normalization (1NF through BCNF/5NF) and when to denormalize\n"
    "- Referential integrity, foreign key strategies, orphan record prevention\n"
    "- Columnstore vs rowstore indexes for analytical workloads\n"
    "- Semantic layer design (measures, KPIs, calculated columns)\n"
    "- DAMA-DMBOK Data Architecture and Data Modeling knowledge areas\n"
    "Provide concrete code examples (DAX, SQL, M, DDL) whenever relevant. "
    "Be opinionated about what the RIGHT design is. Challenge bad patterns. "
    "Think like a principal data architect with 25 years in the field.\n"
    "[END CONTEXT]\n\n"
)


def _get_dm_agent():
    """Get or create a dedicated DrDataAgent for the Data Modeler tab."""
    if "dm_agent" not in st.session_state or st.session_state.dm_agent is None:
        try:
            from config.settings import require_at_least_one_engine
            require_at_least_one_engine()
            agent = DrDataAgent()
            st.session_state.dm_agent = agent
            st.session_state.dm_engine_ok = True
        except Exception as e:
            st.session_state.dm_agent = None
            st.session_state.dm_engine_ok = False
            st.session_state.dm_engine_error = str(e)
    return st.session_state.dm_agent


def _render_advisor():
    """AI Modeling Advisor -- full chat with Dr. Data in data modeling mode."""

    # -- Init session state --
    if "dm_messages" not in st.session_state:
        st.session_state.dm_messages = []
    if "dm_platform" not in st.session_state:
        st.session_state.dm_platform = "Power BI"

    # -- Platform selector --
    platforms = ["Power BI", "Tableau", "Snowflake", "Migration"]
    pcols = st.columns(len(platforms))
    for i, plat in enumerate(platforms):
        with pcols[i]:
            if st.button(
                plat,
                key=f"dm_plat_{i}",
                type="primary" if st.session_state.dm_platform == plat else "secondary",
                use_container_width=True,
            ):
                st.session_state.dm_platform = plat
                st.rerun()

    # -- Quick-prompt chips --
    chips = [
        "Star schema best practices",
        "Optimize my DAX measures",
        "Snowflake vs Star schema trade-offs",
        "Tableau-to-Power BI migration plan",
        "SCD Type 2 implementation",
        "Bridge table for many-to-many",
    ]
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        with chip_cols[i]:
            if st.button(chip, key=f"dm_chip_{i}", use_container_width=True):
                st.session_state.dm_pending_chip = chip
                st.rerun()

    st.divider()

    # -- Chat container --
    chat_container = st.container()

    with chat_container:
        # Opening message if empty
        if not st.session_state.dm_messages:
            with st.chat_message("assistant"):
                st.markdown(
                    "I am Dr. Data, and you are in the **Data Modeler**. This is where we "
                    "design schemas that perform, migrate platforms without losing your mind, "
                    "and write DAX that does not make your colleagues weep.\n\n"
                    "Pick a platform above, or just ask me anything -- star schemas, "
                    "slowly changing dimensions, columnstore indexing strategies, "
                    "Tableau-to-Power BI migration, snowflake warehouse design, "
                    "you name it. I will give you concrete code and opinionated advice.\n\n"
                    "What are we building?"
                )

        # Render full chat history
        for msg in st.session_state.dm_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # -- Handle chip click --
    pending = st.session_state.pop("dm_pending_chip", None)

    # -- Chat input --
    prompt = st.chat_input(
        f"Ask Dr. Data about data modeling ({st.session_state.dm_platform})...",
        key="dm_chat_input",
    )
    user_input = pending or prompt

    if user_input:
        now = time.time()

        # Save user message
        st.session_state.dm_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": now,
        })

        # Get agent
        agent = _get_dm_agent()

        if agent is None:
            st.session_state.dm_messages.append({
                "role": "assistant",
                "content": (
                    "I am having trouble connecting to my AI engines right now. "
                    "Check that at least one API key (Anthropic, OpenAI, or Gemini) "
                    "is configured."
                ),
                "timestamp": now,
            })
            st.rerun()
            return

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                # Build enriched message with platform context + DM specialization
                platform_tag = f"[Platform: {st.session_state.dm_platform}] "
                enriched = _DM_CONTEXT_PREFIX + platform_tag + user_input

                # Use _build_context_message if agent has data loaded
                try:
                    enriched = agent._build_context_message(enriched)
                except Exception:
                    pass

                # Stream response
                try:
                    full_text = st.write_stream(agent.chat_stream(enriched))
                except AttributeError:
                    response = agent.chat(enriched)
                    full_text = response.get("text", "") if isinstance(response, dict) else str(response or "")
                    st.markdown(full_text)

                # Check for generated files
                chat_downloads = []
                for fpath in getattr(agent, "generated_files", []):
                    if os.path.exists(fpath):
                        fname = os.path.basename(fpath)
                        chat_downloads.append({"name": fname, "filename": fname, "path": fpath})
                if chat_downloads:
                    file_names = [dl["filename"] for dl in chat_downloads]
                    st.markdown(
                        f"Built {len(chat_downloads)} deliverable(s): "
                        f"{', '.join(file_names)}."
                    )

                # Save assistant message
                st.session_state.dm_messages.append({
                    "role": "assistant",
                    "content": full_text or "",
                    "downloads": chat_downloads,
                    "timestamp": now,
                })


def _render_schema_builder():
    """Visual Schema Builder -- ERD diagram."""
    st.markdown("##### Visual Schema Builder")
    st.caption("Entity-relationship diagram for your data model.")
    components.html(_SCHEMA_BUILDER_HTML, height=580, scrolling=False)


def _render_schema_analyzer():
    """Schema Analyzer -- DDL analysis."""
    st.markdown("##### Schema Analyzer")
    st.caption("Paste DDL for automated analysis and recommendations.")
    components.html(_SCHEMA_ANALYZER_HTML, height=700, scrolling=True)


def _render_mapper():
    """Source-Target Mapper."""
    st.markdown("##### Source to Target Mapper")
    st.caption("Map source columns to target schema.")
    components.html(_MAPPER_HTML, height=560, scrolling=False)


def render_data_modeling():
    """Main entry point -- renders the Data Modeler tab with sub-tabs."""
    dm1, dm2, dm3, dm4 = st.tabs([
        "AI Modeling Advisor",
        "Visual Schema Builder",
        "Schema Analyzer",
        "Source-Target Mapper",
    ])
    with dm1:
        _render_advisor()
    with dm2:
        _render_schema_builder()
    with dm3:
        _render_schema_analyzer()
    with dm4:
        _render_mapper()


# =====================================================================
# Embedded HTML for the visual tabs
# =====================================================================

_COMMON_STYLE = """
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0B0F1A;--sidebar:#060A12;--accent:#06B6D4;--accent-dim:rgba(6,182,212,.15);
  --fact:#F59E0B;--fact-dim:rgba(245,158,11,.15);
  --card:#111827;--card-border:#1F2937;
  --text:#E5E7EB;--text-dim:#9CA3AF;--text-bright:#F9FAFB;
  --critical:#EF4444;--warning:#F59E0B;--optimize:#8B5CF6;--pass:#10B981;
  --radius:10px;--radius-sm:6px;
}
html,body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:16px}
.erd-wrapper{position:relative;background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);height:400px;overflow:hidden}
.erd-table{position:absolute;background:var(--sidebar);border:1px solid var(--card-border);border-radius:var(--radius-sm);min-width:180px;font-size:12px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.erd-table-header{padding:10px 14px;border-bottom:1px solid var(--card-border);font-weight:700;font-size:13px}
.erd-table-header.dim{color:var(--accent)}
.erd-table-header.fact{color:var(--fact)}
.erd-table-row{padding:6px 14px;display:flex;align-items:center;gap:8px;color:var(--text-dim)}
.erd-table-row:last-child{padding-bottom:10px}
.badge{padding:1px 6px;border-radius:3px;font-size:9px;font-weight:700;font-family:monospace}
.badge-pk{background:var(--accent-dim);color:var(--accent)}
.badge-fk{background:var(--fact-dim);color:var(--fact)}
.col-name{font-family:monospace;font-size:11px}
.col-type{margin-left:auto;font-family:monospace;font-size:10px;color:#6B7280}
svg.erd-arrows{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.erd-arrows path{fill:none;stroke:var(--accent);stroke-width:1.5;opacity:.5}
.erd-arrows text{fill:var(--text-dim);font-size:10px;font-family:sans-serif}
.insight-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}
.insight-card{background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);padding:16px}
.insight-card h4{font-size:13px;font-weight:600;color:var(--text-bright);margin-bottom:6px}
.insight-card p{font-size:12px;color:var(--text-dim);line-height:1.5}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-bottom:8px}
.tag-star{background:var(--accent-dim);color:var(--accent)}
.tag-snow{background:rgba(139,92,246,.15);color:#8B5CF6}
.tag-perf{background:rgba(16,185,129,.15);color:#10B981}
.ddl-area{width:100%;min-height:140px;background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);padding:14px;font-family:monospace;font-size:12px;color:var(--text);resize:vertical;outline:none;margin-bottom:14px}
.ddl-area:focus{border-color:var(--accent)}
.format-bar{display:flex;gap:8px;margin-bottom:18px}
.fmt-btn{padding:6px 16px;border-radius:var(--radius-sm);border:1px solid var(--card-border);background:var(--card);color:var(--text-dim);font-size:12px;font-weight:500;cursor:pointer;transition:all .15s}
.fmt-btn:hover{border-color:var(--accent);color:var(--text)}
.fmt-btn.active{background:var(--accent);border-color:var(--accent);color:#000;font-weight:600}
.result-cards{display:flex;flex-direction:column;gap:10px}
.result-card{background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);padding:14px;display:flex;gap:12px;align-items:flex-start}
.sev{width:8px;height:8px;border-radius:50%;margin-top:5px;flex-shrink:0}
.sev-critical{background:var(--critical)}
.sev-warning{background:var(--warning)}
.sev-optimize{background:var(--optimize)}
.sev-pass{background:var(--pass)}
.result-card h4{font-size:13px;font-weight:600;color:var(--text-bright);margin-bottom:3px}
.sev-label{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.sev-label.critical{color:var(--critical)}
.sev-label.warning{color:var(--warning)}
.sev-label.optimize{color:var(--optimize)}
.sev-label.pass{color:var(--pass)}
.result-card p{font-size:12px;color:var(--text-dim);line-height:1.5}
.mapper-layout{display:flex;gap:0;background:var(--card);border:1px solid var(--card-border);border-radius:var(--radius);overflow:hidden;min-height:380px}
.mapper-col{flex:1;padding:20px}
.mapper-col h3{font-size:14px;font-weight:600;margin-bottom:4px}
.mapper-col .col-sub{font-size:11px;color:var(--text-dim);margin-bottom:16px}
.mapper-divider{width:60px;background:var(--sidebar);display:flex;align-items:center;justify-content:center;flex-shrink:0;border-left:1px solid var(--card-border);border-right:1px solid var(--card-border)}
.mapper-divider span{writing-mode:vertical-rl;font-size:12px;font-weight:700;color:var(--accent);letter-spacing:3px}
.map-row{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;border-radius:var(--radius-sm);margin-bottom:6px;background:var(--sidebar);border:1px solid var(--card-border);font-size:12px}
.map-row .field{font-family:monospace;font-size:11px;color:var(--text)}
.map-row .dtype{font-family:monospace;font-size:10px;color:var(--text-dim)}
.map-row.mapped{border-color:rgba(6,182,212,.3);background:var(--accent-dim)}
.export-bar{display:flex;gap:10px;margin-top:18px}
.export-btn{padding:8px 20px;border-radius:var(--radius-sm);border:1px solid var(--accent);background:transparent;color:var(--accent);font-size:12px;font-weight:600;cursor:pointer;transition:all .15s}
.export-btn:hover{background:var(--accent);color:#000}
.export-btn.primary{background:var(--accent);color:#000}
</style>
"""

_SCHEMA_BUILDER_HTML = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">{_COMMON_STYLE}</head><body>
<div class="erd-wrapper">
  <svg class="erd-arrows">
    <path d="M 245 120 C 300 120, 320 100, 370 100"/>
    <text x="295" y="90">1:M</text>
    <path d="M 245 290 C 300 290, 320 180, 370 180"/>
    <text x="295" y="240">1:M</text>
    <path d="M 705 120 C 650 120, 620 140, 570 140"/>
    <text x="625" y="110">1:M</text>
  </svg>
  <div class="erd-table" style="left:40px;top:60px">
    <div class="erd-table-header dim">DIM_DATE</div>
    <div class="erd-table-row"><span class="badge badge-pk">PK</span><span class="col-name">DateKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Date</span><span class="col-type">DATE</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Year</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Quarter</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">MonthName</span><span class="col-type">VARCHAR</span></div>
  </div>
  <div class="erd-table" style="left:370px;top:50px">
    <div class="erd-table-header fact">FACT_SALES</div>
    <div class="erd-table-row"><span class="badge badge-pk">PK</span><span class="col-name">SalesID</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="badge badge-fk">FK</span><span class="col-name">DateKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="badge badge-fk">FK</span><span class="col-name">ProductKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="badge badge-fk">FK</span><span class="col-name">AgentKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">SalesAmount</span><span class="col-type">DECIMAL</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Quantity</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Discount</span><span class="col-type">DECIMAL</span></div>
  </div>
  <div class="erd-table" style="left:40px;top:240px">
    <div class="erd-table-header dim">DIM_PRODUCT</div>
    <div class="erd-table-row"><span class="badge badge-pk">PK</span><span class="col-name">ProductKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">ProductName</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Category</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">SubCategory</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">UnitPrice</span><span class="col-type">DECIMAL</span></div>
  </div>
  <div class="erd-table" style="left:700px;top:60px">
    <div class="erd-table-header dim">DIM_AGENT</div>
    <div class="erd-table-row"><span class="badge badge-pk">PK</span><span class="col-name">AgentKey</span><span class="col-type">INT</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">AgentName</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Region</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">Territory</span><span class="col-type">VARCHAR</span></div>
    <div class="erd-table-row"><span class="col-name" style="margin-left:30px">HireDate</span><span class="col-type">DATE</span></div>
  </div>
</div>
<div class="insight-cards">
  <div class="insight-card">
    <span class="tag tag-star">Star Schema</span>
    <h4>Optimal Grain</h4>
    <p>FACT_SALES at transaction-level grain with 3 dimension foreign keys. Supports drill-down from Year to Date.</p>
  </div>
  <div class="insight-card">
    <span class="tag tag-snow">Normalization</span>
    <h4>Denormalized Dimensions</h4>
    <p>DIM_PRODUCT includes SubCategory inline. Consider a snowflake extension only if category hierarchy exceeds 3 levels.</p>
  </div>
  <div class="insight-card">
    <span class="tag tag-perf">Performance</span>
    <h4>Index Strategy</h4>
    <p>Clustered index on SalesID, non-clustered on DateKey + ProductKey composite for common slice-and-dice queries.</p>
  </div>
</div>
</body></html>"""

_SCHEMA_ANALYZER_HTML = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">{_COMMON_STYLE}
</head><body>
<textarea class="ddl-area" placeholder="-- Paste your CREATE TABLE statements here...">CREATE TABLE FACT_SALES (
    SalesID INT PRIMARY KEY,
    DateKey INT NOT NULL,
    ProductKey INT,
    AgentKey INT,
    SalesAmount DECIMAL(18,2),
    Quantity INT,
    Discount DECIMAL(5,2),
    CreatedAt DATETIME DEFAULT GETDATE()
);

CREATE TABLE DIM_PRODUCT (
    ProductKey INT PRIMARY KEY,
    ProductName VARCHAR(255),
    Category VARCHAR(100),
    SubCategory VARCHAR(100),
    UnitPrice DECIMAL(10,2)
);</textarea>
<div class="format-bar">
  <button class="fmt-btn active" onclick="this.parentElement.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active')">SQL Server</button>
  <button class="fmt-btn" onclick="this.parentElement.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active')">PostgreSQL</button>
  <button class="fmt-btn" onclick="this.parentElement.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active')">Snowflake SQL</button>
  <button class="fmt-btn" onclick="this.parentElement.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active')">BigQuery</button>
  <button class="fmt-btn" onclick="this.parentElement.querySelectorAll('.fmt-btn').forEach(b=>b.classList.remove('active'));this.classList.add('active')">MySQL</button>
</div>
<div class="result-cards">
  <div class="result-card"><div class="sev sev-critical"></div><div>
    <div class="sev-label critical">CRITICAL</div>
    <h4>Missing Foreign Key Constraints</h4>
    <p>FACT_SALES.ProductKey and FACT_SALES.AgentKey lack explicit FOREIGN KEY constraints. Without referential integrity, orphan rows can silently corrupt aggregations.</p>
  </div></div>
  <div class="result-card"><div class="sev sev-warning"></div><div>
    <div class="sev-label warning">WARNING</div>
    <h4>Nullable Foreign Keys in Fact Table</h4>
    <p>ProductKey and AgentKey are nullable. In a star schema, fact-to-dimension relationships should be NOT NULL. Use a sentinel key (e.g., -1 = "Unknown") instead of NULLs.</p>
  </div></div>
  <div class="result-card"><div class="sev sev-optimize"></div><div>
    <div class="sev-label optimize">OPTIMIZE</div>
    <h4>Missing Clustered Columnstore Index</h4>
    <p>For analytical workloads, FACT_SALES would benefit from a clustered columnstore index. Power BI DirectQuery performance improves 3-10x with columnar storage.</p>
  </div></div>
  <div class="result-card"><div class="sev sev-optimize"></div><div>
    <div class="sev-label optimize">OPTIMIZE</div>
    <h4>Consider Partitioning by DateKey</h4>
    <p>If FACT_SALES exceeds 10M rows, partition by DateKey for faster partition pruning on time-range queries and incremental refresh in Power BI.</p>
  </div></div>
  <div class="result-card"><div class="sev sev-pass"></div><div>
    <div class="sev-label pass">PASS</div>
    <h4>Data Type Selection</h4>
    <p>DECIMAL(18,2) for SalesAmount and DECIMAL(5,2) for Discount are appropriate. INT surrogate keys follow best practices for join performance.</p>
  </div></div>
</div>
</body></html>"""

_MAPPER_HTML = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">{_COMMON_STYLE}
</head><body>
<div class="mapper-layout">
  <div class="mapper-col">
    <h3 style="color:var(--fact)">Source: Tableau Extract</h3>
    <div class="col-sub">legacy_sales_extract.hyper</div>
    <div class="map-row mapped"><span class="field">Order Date</span><span class="dtype">DATETIME</span></div>
    <div class="map-row mapped"><span class="field">Product Name</span><span class="dtype">STRING</span></div>
    <div class="map-row mapped"><span class="field">Sales Rep</span><span class="dtype">STRING</span></div>
    <div class="map-row mapped"><span class="field">Revenue</span><span class="dtype">DOUBLE</span></div>
    <div class="map-row mapped"><span class="field">Units Sold</span><span class="dtype">INTEGER</span></div>
    <div class="map-row"><span class="field">Discount Rate</span><span class="dtype">DOUBLE</span></div>
    <div class="map-row"><span class="field">Region Code</span><span class="dtype">STRING</span></div>
    <div class="map-row"><span class="field">Category</span><span class="dtype">STRING</span></div>
  </div>
  <div class="mapper-divider"><span>MAP</span></div>
  <div class="mapper-col">
    <h3 style="color:var(--accent)">Target: Power BI Model</h3>
    <div class="col-sub">FACT_SALES + Dimensions</div>
    <div class="map-row mapped"><span class="field">DateKey</span><span class="dtype">INT (FK)</span></div>
    <div class="map-row mapped"><span class="field">ProductKey</span><span class="dtype">INT (FK)</span></div>
    <div class="map-row mapped"><span class="field">AgentKey</span><span class="dtype">INT (FK)</span></div>
    <div class="map-row mapped"><span class="field">SalesAmount</span><span class="dtype">DECIMAL(18,2)</span></div>
    <div class="map-row mapped"><span class="field">Quantity</span><span class="dtype">INT</span></div>
    <div class="map-row"><span class="field">Discount</span><span class="dtype">DECIMAL(5,2)</span></div>
    <div class="map-row"><span class="field">DIM_AGENT.Region</span><span class="dtype">VARCHAR(50)</span></div>
    <div class="map-row"><span class="field">DIM_PRODUCT.Category</span><span class="dtype">VARCHAR(100)</span></div>
  </div>
</div>
<div class="export-bar">
  <button class="export-btn primary">Export SQL Script</button>
  <button class="export-btn">Export CSV Mapping</button>
  <button class="export-btn">Export JSON Schema</button>
</div>
</body></html>"""
