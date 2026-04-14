"""
Dr. Data v2 -- FastAPI Backend
Tableau to Power BI agentic migration pipeline.
SSE streaming, file upload, Power BI publish, AI chat.
"""

import asyncio
import json
import logging
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Load credentials
load_dotenv(Path.home() / ".env.drdata", override=True)
load_dotenv(Path.home() / "joao-spine" / ".env", override=False)
load_dotenv(Path.home() / ".env", override=False)

# Ensure dr-data modules are importable
DR_DATA_ROOT = Path(__file__).parent.resolve()
if str(DR_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DR_DATA_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("drdata-v2")

# ------------------------------------------------------------------ #
#  App
# ------------------------------------------------------------------ #

app = FastAPI(title="Dr. Data v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = DR_DATA_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Storage
UPLOAD_DIR = Path(tempfile.gettempdir()) / "drdata_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = DR_DATA_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory stores
_uploads: Dict[str, dict] = {}       # file_id -> {path, filename, size}
_jobs: Dict[str, dict] = {}          # job_id -> {file_id, status, queue}
_job_results: Dict[str, dict] = {}   # job_id -> final result
_agents: Dict[str, object] = {}      # session_id -> DrDataAgent


def _get_agent(session_id: str):
    """Get or create a DrDataAgent for a session."""
    if session_id not in _agents:
        try:
            from app.dr_data_agent import DrDataAgent
            _agents[session_id] = DrDataAgent()
            logger.info(f"Agent created for session {session_id}")
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return None
    return _agents[session_id]


# ------------------------------------------------------------------ #
#  Pipeline State Machine (human-gated)
# ------------------------------------------------------------------ #

from core.pipeline_state import PipelineState, StageStatus, StageName
from core.pipeline_runner import (
    run_parse, run_field_mapping, run_formula_translation,
    run_visual_mapping, run_generate, run_synthetic_data,
    dispatch_multi_brain,
)

# Active pipeline states per session
_pipelines: Dict[str, PipelineState] = {}


def _get_pipeline(session_id: str) -> Optional[PipelineState]:
    return _pipelines.get(session_id)


def _create_pipeline(session_id: str, workbook_name: str) -> PipelineState:
    ps = PipelineState(session_id, workbook_name)
    _pipelines[session_id] = ps
    return ps


# ------------------------------------------------------------------ #
#  Models
# ------------------------------------------------------------------ #

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    file_id: str = ""


# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #

@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "No frontend found. Place index.html in static/"})


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex[:12]
    dest_dir = UPLOAD_DIR / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        content_bytes = await file.read()
        f.write(content_bytes)

    info = {
        "path": str(dest_path),
        "filename": file.filename,
        "size": len(content_bytes),
    }
    _uploads[file_id] = info
    logger.info(f"Upload: {file.filename} ({len(content_bytes):,} bytes) -> {file_id}")

    # Quick parse for file info (no LLM call -- instant)
    file_info = {}
    try:
        from core.enhanced_tableau_parser import parse_twb
        spec = parse_twb(str(dest_path))
        file_info = {
            "worksheets": len(spec.get("worksheets", [])),
            "dashboards": len(spec.get("dashboards", [])),
            "calc_fields": len(spec.get("calculated_fields", [])),
        }
    except Exception as pe:
        logger.warning(f"Quick parse error: {pe}")

    # Background agent analysis (non-blocking)
    def _bg_analyze(fid, fpath):
        try:
            agent = _get_agent(fid)
            if agent:
                agent.analyze_uploaded_file(file_path=fpath)
                logger.info(f"Background analysis complete for {fid}")
        except Exception as e:
            logger.error(f"Background analysis error: {e}")

    threading.Thread(target=_bg_analyze, args=(file_id, str(dest_path)), daemon=True).start()

    # Check for existing session with this file (session persistence)
    resumable = False
    resume_session_id = None
    try:
        from core.correction_store import load_session
        # Use file_id as lookup key -- also check any pipeline that used this filename
        for sid, ps in _pipelines.items():
            if ps.workbook_name == file.filename:
                resumable = True
                resume_session_id = sid
                break
        # Also check Supabase for persisted sessions
        if not resumable:
            # Try to find a session with this filename
            from core.correction_store import _get_client
            client = _get_client()
            if client:
                try:
                    result = client.table("drdata_sessions") \
                        .select("session_id, current_stage") \
                        .eq("twbx_filename", file.filename) \
                        .order("updated_at", desc=True) \
                        .limit(1) \
                        .execute()
                    if result.data:
                        resumable = True
                        resume_session_id = result.data[0]["session_id"]
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Session check: {e}")

    resp = {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content_bytes),
        "analysis": f"Uploaded {file.filename}. {file_info.get('worksheets',0)} worksheets, {file_info.get('dashboards',0)} dashboards, {file_info.get('calc_fields',0)} calculated fields detected.",
        "file_info": file_info,
    }
    if resumable and resume_session_id:
        resp["resumable"] = True
        resp["resume_session_id"] = resume_session_id
        resp["analysis"] += " I found a previous session for this file -- you can resume where you left off."
    return resp


@app.post("/api/process/{file_id}")
async def start_process(file_id: str):
    if file_id not in _uploads:
        raise HTTPException(404, f"File {file_id} not found")

    job_id = uuid.uuid4().hex[:12]
    q = queue.Queue()  # Thread-safe stdlib queue

    _jobs[job_id] = {
        "file_id": file_id,
        "status": "queued",
        "queue": q,
        "created": datetime.utcnow().isoformat(),
    }

    # Run pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, file_id, q),
        daemon=True,
    )
    thread.start()

    logger.info(f"Job started: {job_id} for file {file_id}")
    return {"job_id": job_id, "file_id": file_id, "status": "started"}


@app.get("/api/stream/{job_id}")
async def stream_events(job_id: str, request: Request):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    q = _jobs[job_id]["queue"]

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            # Poll the thread-safe queue from the async context
            try:
                event = q.get(block=False)
                yield json.dumps(event, default=str)
                if event.get("phase") in ("done", "error"):
                    break
            except queue.Empty:
                # No event yet -- yield a comment to keep connection alive and retry
                await asyncio.sleep(0.3)
                continue

    return EventSourceResponse(event_generator())


class ChatRequestV2(BaseModel):
    message: str
    session_id: str = "default"
    context: str = ""
    file_id: str = ""


# ------------------------------------------------------------------ #
#  Pipeline Endpoints (human-gated state machine)
# ------------------------------------------------------------------ #

@app.get("/api/pipeline/{session_id}")
async def pipeline_status(session_id: str):
    """Get current pipeline state for a session."""
    ps = _get_pipeline(session_id)
    if not ps:
        return {"active": False, "session_id": session_id}
    return {"active": True, **ps.to_dict()}


@app.post("/api/pipeline/{session_id}/approve")
async def pipeline_approve(session_id: str):
    """Approve the current awaiting stage and advance."""
    ps = _get_pipeline(session_id)
    if not ps:
        raise HTTPException(404, "No active pipeline")
    if not ps.is_awaiting_approval:
        return {"error": "No stage awaiting approval", **ps.to_dict()}

    stage = ps.current_stage
    ps.approve_stage(stage)
    return {"approved": stage, **ps.to_dict()}


class CorrectionRequest(BaseModel):
    corrections: dict = {}


@app.post("/api/pipeline/{session_id}/correct")
async def pipeline_correct(session_id: str, req: dict):
    """Analyst submits corrections to any pipeline stage.

    Body:
    {
        "stage": "field_mapping",
        "corrections": [
            {
                "field_path": "worksheets.0.mark_type",
                "original_value": "bar",
                "corrected_value": "line",
                "correction_type": "chart_type",
                "worksheet_name": "Sales Overview",
                "mark_type": "bar"
            }
        ]
    }
    """
    from core.correction_store import store_correction

    ps = _get_pipeline(session_id)
    stored = 0
    applied = 0

    for c in req.get("corrections", []):
        store_correction(
            session_id=session_id,
            stage=req.get("stage", "unknown"),
            field_path=c.get("field_path", ""),
            original_value=c.get("original_value", ""),
            corrected_value=c.get("corrected_value", ""),
            correction_type=c.get("correction_type", ""),
            worksheet_name=c.get("worksheet_name", ""),
            mark_type=c.get("mark_type", ""),
            context=c.get("context"),
        )
        stored += 1

        if ps:
            _apply_correction(ps, req.get("stage"), c)
            applied += 1

    return {
        "stored": stored,
        "applied": applied,
        "message": f"Dr. Data learned from {stored} corrections. Applied {applied} to current session.",
        **(ps.to_dict() if ps else {}),
    }


@app.get("/api/pipeline/{session_id}/stage/{stage_num}")
async def get_stage_output(session_id: str, stage_num: int):
    """Get the output of a specific pipeline stage for display + editing."""
    ps = _get_pipeline(session_id)
    if not ps:
        raise HTTPException(404, "Session not found")

    stage_names = list(StageName)
    if stage_num < 0 or stage_num >= len(stage_names):
        raise HTTPException(400, f"Invalid stage number: {stage_num}")

    stage_key = stage_names[stage_num].value
    result = ps.results.get(stage_key)
    stage_data = result.data if result else {}

    return {
        "stage": stage_num,
        "stage_name": _STAGE_NAMES.get(stage_num, f"Stage {stage_num}"),
        "stage_key": stage_key,
        "output": stage_data,
        "summary": result.summary if result else "",
        "editable_fields": _get_editable_fields(stage_num, stage_data),
        "status": ps.stages.get(stage_key, "pending"),
    }


@app.get("/api/corrections/stats")
async def correction_stats():
    """Get correction learning statistics."""
    from core.correction_store import get_correction_stats
    stats = get_correction_stats()
    msg = f"Dr. Data has learned from {stats['total']} analyst corrections"
    return {**stats, "message": msg}


_STAGE_NAMES = {
    0: "Parse Tableau Workbook",
    1: "Data Source",
    2: "Field Mapping",
    3: "Formula Translation",
    4: "Visual Mapping",
    5: "Semantic Model",
    6: "Report Layout",
    7: "Generate PBIP",
}


def _get_editable_fields(stage_num: int, stage_data: dict) -> list:
    """Return list of editable field descriptors for a stage."""
    editable = []
    if stage_num == 0:  # Parse
        for i, ws in enumerate(stage_data.get("tableau_spec", {}).get("worksheets", [])):
            editable.append({
                "field_path": f"worksheets.{i}.mark_type",
                "label": ws.get("name", f"Worksheet {i}"),
                "current_value": ws.get("mark_type", "automatic"),
                "correction_type": "chart_type",
                "editor": "dropdown",
                "options": ["bar", "line", "area", "circle", "text", "square",
                            "pie", "gantt", "polygon", "map", "automatic"],
            })
    elif stage_num == 2:  # Field Mapping
        config = stage_data.get("config", {})
        sections = config.get("report_layout", {}).get("sections", [])
        for pi, page in enumerate(sections):
            for vi, vc in enumerate(page.get("visualContainers", [])):
                cfg = vc.get("config", {})
                editable.append({
                    "field_path": f"pages.{pi}.visuals.{vi}",
                    "label": cfg.get("singleVisual", {}).get("title", {}).get("text", f"Visual {vi}"),
                    "current_value": cfg.get("singleVisual", {}).get("visualType", ""),
                    "correction_type": "field_mapping",
                    "editor": "detail",
                })
    elif stage_num == 3:  # Formula Translation
        for i, t in enumerate(stage_data.get("translations", [])):
            editable.append({
                "field_path": f"translations.{i}.dax",
                "label": t.get("name", f"Formula {i}"),
                "current_value": t.get("dax", ""),
                "original_tableau": t.get("tableau_formula", ""),
                "tier": t.get("tier", "AUTO"),
                "correction_type": "dax_formula",
                "editor": "code",
            })
    elif stage_num == 4:  # Visual Mapping
        for i, v in enumerate(stage_data.get("visuals", [])):
            editable.append({
                "field_path": f"visuals.{i}.pbi_type",
                "label": v.get("worksheet_name", f"Visual {i}"),
                "current_value": v.get("pbi_type", ""),
                "correction_type": "chart_type",
                "editor": "dropdown",
                "options": ["clusteredBarChart", "lineChart", "areaChart",
                            "pieChart", "donutChart", "cardVisual", "tableEx",
                            "waterfallChart", "treemap", "scatterChart",
                            "filledMap", "shape", "slicer"],
            })
    return editable


def _apply_correction(ps, stage: str, correction: dict):
    """Apply a single correction to the in-memory pipeline state."""
    ctype = correction.get("correction_type", "")
    field_path = correction.get("field_path", "")
    new_val = correction.get("corrected_value", "")

    if ctype == "chart_type" and field_path.startswith("worksheets."):
        # Update mark_type in tableau_spec
        spec = ps.context.get("tableau_spec", {})
        parts = field_path.split(".")
        try:
            idx = int(parts[1])
            ws_list = spec.get("worksheets", [])
            if idx < len(ws_list):
                ws_list[idx]["mark_type"] = new_val
        except (IndexError, ValueError):
            pass

    elif ctype == "dax_formula" and field_path.startswith("translations."):
        # Update DAX in formula translation result
        result = ps.results.get("formula_trans")
        if result and result.data:
            parts = field_path.split(".")
            try:
                idx = int(parts[1])
                translations = result.data.get("translations", [])
                if idx < len(translations):
                    translations[idx]["dax"] = new_val
            except (IndexError, ValueError):
                pass

    elif ctype == "chart_type" and field_path.startswith("visuals."):
        # Update PBI visual type in visual mapping result
        result = ps.results.get("visual_mapping")
        if result and result.data:
            parts = field_path.split(".")
            try:
                idx = int(parts[1])
                visuals = result.data.get("visuals", [])
                if idx < len(visuals):
                    visuals[idx]["pbi_type"] = new_val
            except (IndexError, ValueError):
                pass

    # Store in pipeline corrections dict
    if stage:
        if stage not in ps.corrections:
            ps.corrections[stage] = {}
        ps.corrections[stage][field_path] = {
            "original": correction.get("original_value"),
            "corrected": new_val,
            "type": ctype,
        }


@app.post("/api/pipeline/{session_id}/run-next")
async def pipeline_run_next(session_id: str):
    """Execute the next pending stage in the pipeline.

    This is called after an approval to actually run the next stage.
    Returns the stage output for presentation in chat.
    """
    ps = _get_pipeline(session_id)
    if not ps:
        raise HTTPException(404, "No active pipeline")
    if ps.is_complete:
        return {"complete": True, **ps.to_dict()}
    if ps.is_awaiting_approval:
        return {"awaiting": ps.current_stage, **ps.to_dict()}

    stage = ps.current_stage
    ps.start_stage(stage)

    try:
        if stage == "parse":
            # Need file_id from context
            file_id = ps.context.get("file_id", "")
            if file_id not in _uploads:
                ps.fail_stage(stage, "No file uploaded")
                return {"error": "No file uploaded", **ps.to_dict()}

            filepath = _uploads[file_id]["path"]
            result = run_parse(filepath)
            ps.context["tableau_spec"] = result["tableau_spec"]
            ps.context["palette"] = result["palette"]
            ps.complete_stage(stage, result["summary"], result)

        elif stage == "data_source":
            # This stage waits for analyst choice
            ps.complete_stage(stage,
                "How should Power BI get its data?\n"
                "  A) Synthetic data (generated from Tableau structure)\n"
                "  B) Upload Excel/CSV\n"
                "  C) Database connection\n"
                "  D) Wireframe (no data)\n\n"
                "Reply with A, B, C, or D.",
                {"choices": ["synthetic", "upload", "database", "wireframe"]})

        elif stage == "field_mapping":
            spec = ps.context.get("tableau_spec")
            profile = ps.context.get("data_profile")
            if not spec or not profile:
                ps.fail_stage(stage, "Missing tableau_spec or data_profile")
                return {"error": "Missing context", **ps.to_dict()}

            result = run_field_mapping(spec, profile)
            ps.context["config"] = result["config"]
            ps.context["dashboard_spec"] = result["dashboard_spec"]
            ps.complete_stage(stage, result["summary"], result,
                              qa_notes=result.get("qa_notes", []))

        elif stage == "formula_trans":
            spec = ps.context.get("tableau_spec", {})
            calcs = spec.get("calculated_fields", [])
            df = ps.context.get("dataframe")

            if not calcs:
                ps.complete_stage(stage,
                    "No calculated fields to translate.", {"translations": []})
                ps.approve_stage(stage)  # auto-approve if nothing to translate
                return await pipeline_run_next(session_id)

            result = run_formula_translation(calcs, spec, df)

            # Multi-brain for REVIEW/BLOCKED
            if result["review_needed"]:
                for r in result["review_needed"][:3]:
                    prompt = (
                        f"Translate this Tableau calculated field to DAX:\n"
                        f"Name: {r['name']}\n"
                        f"Tableau formula: {r['tableau_formula']}\n\n"
                        f"Return ONLY the DAX expression."
                    )
                    brain_result = dispatch_multi_brain(prompt)
                    r["brain_consensus"] = brain_result["results"]

            ps.complete_stage(stage, result["summary"], result,
                              qa_notes=result.get("qa_notes", []))

        elif stage == "visual_mapping":
            spec = ps.context.get("tableau_spec")
            config = ps.context.get("config")
            palette = ps.context.get("palette", [])
            if not config:
                ps.fail_stage(stage, "No config from field mapping")
                return {"error": "Missing config", **ps.to_dict()}

            result = run_visual_mapping(spec, config, palette)
            ps.complete_stage(stage, result["summary"], result)

        elif stage == "semantic_model":
            config = ps.context.get("config", {})
            tmdl = config.get("tmdl_model", {})
            tables = tmdl.get("tables", [])
            measures = tables[0].get("measures", []) if tables else []
            relationships = tmdl.get("relationships", [])

            summary_lines = [
                f"Semantic Model:",
                f"  Tables: {len(tables)}",
                f"  Measures: {len(measures)}",
                f"  Relationships: {len(relationships)}",
            ]
            if measures:
                summary_lines.append("\nDAX Measures:")
                for m in measures[:10]:
                    summary_lines.append(f"  {m.get('name','')}: {m.get('expression','')[:80]}")

            ps.complete_stage(stage, "\n".join(summary_lines),
                              {"tables": len(tables), "measures": len(measures),
                               "relationships": len(relationships)})

        elif stage == "report_layout":
            config = ps.context.get("config", {})
            sections = config.get("report_layout", {}).get("sections", [])
            total_visuals = sum(len(s.get("visualContainers", [])) for s in sections)

            summary_lines = [
                f"Report Layout:",
                f"  {len(sections)} pages, {total_visuals} visuals",
            ]
            for s in sections:
                name = s.get("displayName", "Page")
                vc_count = len(s.get("visualContainers", []))
                summary_lines.append(f"  - {name}: {vc_count} visuals")

            ps.complete_stage(stage, "\n".join(summary_lines),
                              {"pages": len(sections), "visuals": total_visuals})

        elif stage == "generate":
            config = ps.context.get("config")
            profile = ps.context.get("data_profile")
            dspec = ps.context.get("dashboard_spec")
            df = ps.context.get("dataframe")
            palette = ps.context.get("palette", [])

            if not config or not profile:
                ps.fail_stage(stage, "Missing config or profile")
                return {"error": "Missing context", **ps.to_dict()}

            result = run_generate(config, profile, dspec, df, palette, session_id)
            ps.context["pbip_path"] = result["pbip_path"]
            ps.complete_stage(stage, result["summary"], result)

        else:
            ps.fail_stage(stage, f"Unknown stage: {stage}")

    except Exception as e:
        logger.exception(f"Pipeline stage {stage} failed")
        ps.fail_stage(stage, str(e)[:500])
        return {"error": str(e)[:500], **ps.to_dict()}

    return {
        "stage": stage,
        "summary": ps.format_stage_summary_for_chat(stage),
        **ps.to_dict(),
    }


# ------------------------------------------------------------------ #
#  Chat (V1 agent + pipeline awareness)
# ------------------------------------------------------------------ #

@app.post("/api/chat")
async def chat(req: ChatRequestV2):
    """Full conversational chat with Dr. Data agent.

    Pipeline-aware: detects migration intent and starts the human-gated
    pipeline. For all other intents, delegates to the V1 agent.
    """
    session_id = req.session_id or req.file_id or "default"
    msg = req.message.strip()
    msg_lower = msg.lower()
    progress_msgs = []

    def _progress(m):
        progress_msgs.append(m)

    # ---- Check for pipeline approval/correction commands ----
    ps = _get_pipeline(session_id)
    if ps and ps.is_awaiting_approval:
        stage = ps.current_stage

        if msg_lower in ("approve", "yes", "ok", "looks good", "proceed", "continue", "y", "lgtm"):
            ps.approve_stage(stage)
            # If data_source stage, handle the choice
            # Otherwise run next stage automatically
            if stage == "data_source":
                return {
                    "response": "Data source approved. Moving to field mapping...",
                    "pipeline": ps.to_dict(),
                    "files": [],
                    "intent": "pipeline",
                }
            # Auto-run next stage
            return await _run_next_and_respond(session_id)

        elif stage == "data_source":
            # Handle data source choice
            choice = None
            if any(x in msg_lower for x in ["a", "synthetic"]):
                choice = "synthetic"
            elif any(x in msg_lower for x in ["b", "upload", "excel", "csv"]):
                choice = "upload"
            elif any(x in msg_lower for x in ["c", "database", "db", "snowflake", "sql"]):
                choice = "database"
            elif any(x in msg_lower for x in ["d", "wireframe", "no data"]):
                choice = "wireframe"

            if choice:
                ps.context["data_source_choice"] = choice
                ps.log_decision(stage, "analyst", f"chose: {choice}",
                                ["synthetic", "upload", "database", "wireframe"])

                # Generate data based on choice
                if choice == "synthetic":
                    spec = ps.context.get("tableau_spec", {})
                    data_result = run_synthetic_data(spec)
                    ps.context["dataframe"] = data_result["dataframe"]
                    ps.context["data_profile"] = data_result["profile"]
                    ps.approve_stage(stage)
                    return {
                        "response": f"Synthetic data generated.\n\n{data_result['summary']}\n\nMoving to field mapping...",
                        "pipeline": ps.to_dict(),
                        "files": [],
                        "intent": "pipeline",
                    }
                elif choice == "wireframe":
                    ps.context["dataframe"] = None
                    ps.context["data_profile"] = {"table_name": "Data", "row_count": 0, "columns": []}
                    ps.approve_stage(stage)
                    return {
                        "response": "Wireframe mode -- no data will be embedded. Moving to field mapping...",
                        "pipeline": ps.to_dict(),
                        "files": [],
                        "intent": "pipeline",
                    }
                elif choice == "upload":
                    return {
                        "response": "Upload your Excel or CSV file using the upload button. I will use it as the data source.",
                        "pipeline": ps.to_dict(),
                        "files": [],
                        "intent": "pipeline",
                    }
                elif choice == "database":
                    return {
                        "response": "What is your database connection? Provide: type (Snowflake/SQL Server/PostgreSQL), server, database, schema, table.",
                        "pipeline": ps.to_dict(),
                        "files": [],
                        "intent": "pipeline",
                    }

        else:
            # Treat as correction
            ps.correct_stage(stage, {"analyst_note": msg})
            return await _run_next_and_respond(session_id)

    # ---- Detect migration intent ----
    is_migrate = any(x in msg_lower for x in [
        "convert", "migrate", "power bi", "powerbi", "pbip",
        "translate to", "build pbi", "transform to pbi",
    ])

    if is_migrate and req.file_id and req.file_id in _uploads:
        filename = _uploads[req.file_id]["filename"]
        ps = _create_pipeline(session_id, filename)
        ps.context["file_id"] = req.file_id

        # Run parse immediately
        return await _run_next_and_respond(session_id)

    # ---- Default: V1 agent chat ----
    try:
        agent = _get_agent(session_id)
        if agent:
            # Attach uploaded file context if available
            if req.file_id and req.file_id in _uploads:
                info = _uploads[req.file_id]
                filepath = info["path"]
                if not agent.tableau_spec and filepath.endswith((".twbx", ".twb")):
                    try:
                        agent.analyze_uploaded_file(file_path=filepath)
                    except Exception:
                        pass

            result = agent.respond(
                user_message=req.message,
                conversation_history=agent.messages,
                uploaded_files={},
                progress_callback=_progress,
            )

            if isinstance(result, str):
                text = result
                downloads = []
            elif isinstance(result, dict):
                text = result.get("content", result.get("text", ""))
                downloads = result.get("downloads", [])
            else:
                text = str(result)
                downloads = []

            served_files = _serve_downloads(downloads)

            return {
                "response": text,
                "files": served_files,
                "intent": result.get("intent", "chat") if isinstance(result, dict) else "chat",
                "progress": progress_msgs,
            }
    except Exception as agent_err:
        logger.warning(f"Agent chat error: {agent_err}")
        import traceback
        traceback.print_exc()

    # Fallback: direct Claude
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"response": "No ANTHROPIC_API_KEY configured.", "files": [], "intent": "error"}

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        try:
            from config.prompts import DR_DATA_SYSTEM_PROMPT
            system = DR_DATA_SYSTEM_PROMPT
        except ImportError:
            system = (
                "You are Dr. Data, Chief Data Intelligence Officer. "
                "Senior data analyst, 25 years experience. Be direct, give code, no fluff."
            )

        context = req.context
        if req.file_id and req.file_id in _uploads:
            info = _uploads[req.file_id]
            context += f"\nFile: {info['filename']} ({info['size']:,} bytes)"
        if context:
            system += f"\n\nCurrent context:\n{context}"

        msg_resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": req.message}],
        )
        return {"response": msg_resp.content[0].text, "files": [], "intent": "chat"}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)[:200]}", "files": [], "intent": "error"}


async def _run_next_and_respond(session_id: str) -> dict:
    """Helper: run the next pipeline stage and format for chat response."""
    ps = _get_pipeline(session_id)
    if not ps or ps.is_complete:
        return {"response": "Pipeline complete.", "pipeline": ps.to_dict() if ps else {}, "files": [], "intent": "pipeline"}

    # Run stage synchronously (in thread for CPU work)
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _sync_run_next, session_id)
    return result


def _sync_run_next(session_id: str) -> dict:
    """Synchronous pipeline stage runner."""
    ps = _get_pipeline(session_id)
    if not ps:
        return {"response": "No active pipeline.", "files": [], "intent": "error"}

    stage = ps.current_stage
    if not stage:
        # Pipeline complete -- serve final files
        pbip_path = ps.context.get("pbip_path", "")
        served = []
        if pbip_path:
            zip_base = Path(pbip_path).parent / Path(pbip_path).name
            zip_path = str(zip_base) + ".zip"
            if os.path.exists(zip_path):
                served = _serve_downloads([{"path": zip_path, "name": "PBIP Project", "filename": os.path.basename(zip_path)}])
        return {
            "response": "Pipeline complete. All stages approved.\n\n" + _format_audit_log(ps),
            "pipeline": ps.to_dict(),
            "files": served,
            "intent": "pipeline",
        }

    ps.start_stage(stage)

    try:
        if stage == "parse":
            file_id = ps.context.get("file_id", "")
            if file_id not in _uploads:
                ps.fail_stage(stage, "No file uploaded")
                return {"response": "No file uploaded.", "pipeline": ps.to_dict(), "files": [], "intent": "error"}

            filepath = _uploads[file_id]["path"]
            result = run_parse(filepath)
            ps.context["tableau_spec"] = result["tableau_spec"]
            ps.context["palette"] = result["palette"]
            ps.complete_stage(stage, result["summary"], result)

        elif stage == "data_source":
            ps.complete_stage(stage,
                "How should Power BI get its data?\n\n"
                "  A) Synthetic data (generated from Tableau structure)\n"
                "  B) Upload Excel/CSV\n"
                "  C) Database connection\n"
                "  D) Wireframe (no data)\n\n"
                "Reply with your choice.",
                {"choices": ["synthetic", "upload", "database", "wireframe"]})

        elif stage == "field_mapping":
            spec = ps.context.get("tableau_spec")
            profile = ps.context.get("data_profile")
            if not spec or not profile:
                ps.fail_stage(stage, "Missing context")
                return {"response": "Missing tableau spec or data profile.", "pipeline": ps.to_dict(), "files": [], "intent": "error"}
            result = run_field_mapping(spec, profile)
            ps.context["config"] = result["config"]
            ps.context["dashboard_spec"] = result["dashboard_spec"]
            ps.complete_stage(stage, result["summary"], result)

        elif stage == "formula_trans":
            spec = ps.context.get("tableau_spec", {})
            calcs = spec.get("calculated_fields", [])
            if not calcs:
                ps.complete_stage(stage, "No calculated fields to translate.", {})
                ps.approve_stage(stage)
                return _sync_run_next(session_id)
            result = run_formula_translation(calcs, spec, ps.context.get("dataframe"))
            if result["review_needed"]:
                for r in result["review_needed"][:3]:
                    prompt = (
                        f"Translate this Tableau calculated field to DAX:\n"
                        f"Name: {r['name']}\nTableau: {r['tableau_formula']}\n"
                        f"Return ONLY the DAX expression."
                    )
                    r["brain_consensus"] = dispatch_multi_brain(prompt)["results"]
            ps.complete_stage(stage, result["summary"], result, qa_notes=result.get("qa_notes", []))

        elif stage == "visual_mapping":
            config = ps.context.get("config")
            result = run_visual_mapping(ps.context.get("tableau_spec"), config, ps.context.get("palette", []))
            ps.complete_stage(stage, result["summary"], result)

        elif stage == "semantic_model":
            config = ps.context.get("config", {})
            tmdl = config.get("tmdl_model", {})
            tables = tmdl.get("tables", [])
            measures = tables[0].get("measures", []) if tables else []
            summary = f"Semantic Model: {len(tables)} tables, {len(measures)} measures"
            ps.complete_stage(stage, summary, {"tables": len(tables), "measures": len(measures)})

        elif stage == "report_layout":
            config = ps.context.get("config", {})
            sections = config.get("report_layout", {}).get("sections", [])
            total = sum(len(s.get("visualContainers", [])) for s in sections)
            summary = f"Report Layout: {len(sections)} pages, {total} visuals"
            ps.complete_stage(stage, summary, {"pages": len(sections), "visuals": total})

        elif stage == "generate":
            config = ps.context.get("config")
            profile = ps.context.get("data_profile")
            dspec = ps.context.get("dashboard_spec")
            result = run_generate(config, profile, dspec, ps.context.get("dataframe"),
                                  ps.context.get("palette", []), session_id)
            ps.context["pbip_path"] = result["pbip_path"]
            served = []
            if result.get("zip_path") and os.path.exists(result["zip_path"]):
                served = _serve_downloads([{"path": result["zip_path"], "name": "PBIP Project",
                                            "filename": os.path.basename(result["zip_path"])}])
            ps.complete_stage(stage, result["summary"], result)
            return {
                "response": ps.format_stage_summary_for_chat(stage),
                "pipeline": ps.to_dict(),
                "files": served,
                "intent": "pipeline",
            }
        else:
            ps.fail_stage(stage, f"Unknown stage: {stage}")

    except Exception as e:
        logger.exception(f"Stage {stage} failed")
        ps.fail_stage(stage, str(e)[:500])
        _persist_session(session_id, ps)
        return {"response": f"Stage {stage} failed: {str(e)[:300]}", "pipeline": ps.to_dict(), "files": [], "intent": "error"}

    _persist_session(session_id, ps)
    return {
        "response": ps.format_stage_summary_for_chat(stage),
        "pipeline": ps.to_dict(),
        "files": [],
        "intent": "pipeline",
    }


def _persist_session(session_id: str, ps):
    """Save pipeline state to Supabase after each stage."""
    try:
        from core.correction_store import save_session
        # Serialize context, excluding non-serializable items (dataframe)
        ctx = {}
        for k, v in ps.context.items():
            if k == "dataframe":
                continue  # DataFrames are too large / not JSON-serializable
            try:
                json.dumps(v)
                ctx[k] = v
            except (TypeError, ValueError):
                pass

        save_session(session_id, {
            "twbx_filename": ps.workbook_name,
            "tableau_spec": ctx.get("tableau_spec", {}),
            "pipeline_state": {
                "stages": {k: v.value for k, v in ps.stages.items()},
                "current_stage": ps.current_stage,
                "current_stage_index": ps.current_stage_index,
            },
            "current_stage": ps.current_stage_index,
            "config": ctx.get("config", {}),
            "data_profile": ctx.get("data_profile", {}),
            "translations": ctx.get("formula_translations", {}),
        })
    except Exception as e:
        logger.debug(f"Session persist: {e}")


def _serve_downloads(downloads: list) -> list:
    """Copy files to static/downloads and return served entries."""
    served = []
    for dl in downloads:
        if isinstance(dl, dict) and dl.get("path"):
            fpath = dl["path"]
            fname = dl.get("filename", os.path.basename(fpath))
            dl_dir = STATIC_DIR / "downloads"
            dl_dir.mkdir(exist_ok=True)
            dest = dl_dir / fname
            try:
                shutil.copy2(fpath, str(dest))
                served.append({
                    "name": dl.get("name", fname),
                    "filename": fname,
                    "url": f"/static/downloads/{fname}",
                    "description": dl.get("description", ""),
                })
            except Exception as e:
                logger.warning(f"File copy failed: {e}")
    return served


def _format_audit_log(ps: PipelineState) -> str:
    """Format the full audit log for final presentation."""
    lines = ["Decision Audit Log:"]
    for d in ps.audit_log:
        ts = time.strftime("%H:%M:%S", time.localtime(d.timestamp))
        lines.append(f"  [{ts}] {d.stage}: {d.made_by} -- {d.decision}")
    return "\n".join(lines)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    result = _job_results.get(job_id, {})
    return {
        "job_id": job_id,
        "status": _jobs[job_id]["status"],
        "result": result,
    }


# ------------------------------------------------------------------ #
#  Pipeline (runs in background thread)
# ------------------------------------------------------------------ #

def _emit(q, phase, status, detail="", **extra):
    """Thread-safe event emit to stdlib queue."""
    event = {
        "phase": phase,
        "status": status,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat(),
        **extra,
    }
    q.put(event)
    logger.info(f"[{phase}] {status}: {detail[:100]}")


def _run_pipeline(job_id: str, file_id: str, q):
    """Full migration pipeline: parse -> translate -> build -> validate -> publish."""
    import pandas as pd

    info = _uploads[file_id]
    filepath = info["path"]
    filename = info["filename"]
    _jobs[job_id]["status"] = "running"

    try:
        # ---- PHASE: PARSE ----
        _emit(q, "parse", "start", f"Parsing {filename}")

        from core.enhanced_tableau_parser import parse_twb
        spec = parse_twb(filepath)

        ws_count = len(spec.get("worksheets", []))
        db_count = len(spec.get("dashboards", []))
        cf_count = len(spec.get("calculated_fields", []))
        ds_count = len(spec.get("datasources", []))

        ws_names = [w.get("name", "") for w in spec.get("worksheets", [])]
        db_names = [d.get("name", "") for d in spec.get("dashboards", [])]

        _emit(q, "parse", "complete",
              f"{ws_count} worksheets, {db_count} dashboards, {cf_count} calcs, {ds_count} datasources",
              worksheets=ws_names[:10], dashboards=db_names[:5],
              calc_count=cf_count, datasource_count=ds_count)

        # ---- PHASE: TRANSLATE ----
        _emit(q, "translate", "start", "Generating synthetic data + mapping fields")

        from core.synthetic_data import generate_from_tableau_spec
        df, col_map, gen_log = generate_from_tableau_spec(spec)

        profile = {
            "table_name": "Data",
            "row_count": len(df),
            "columns": [
                {
                    "name": c,
                    "dtype": str(df[c].dtype),
                    "semantic_type": (
                        "measure" if pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_datetime64_any_dtype(df[c])
                        else "dimension"
                    ),
                    "unique_count": int(df[c].nunique()),
                }
                for c in df.columns
            ],
        }

        from core.direct_mapper import build_pbip_config_from_tableau
        config, dspec = build_pbip_config_from_tableau(spec, profile, "Data")

        # Extract Tableau colors and inject unified palette for PBI theme
        try:
            from core.enhanced_tableau_parser import get_xml_root
            from core.color_extractor import extract_all_colors, build_unified_palette
            xml_root = get_xml_root(filepath)
            if xml_root is not None:
                extracted = extract_all_colors(xml_root)
                palette = build_unified_palette(extracted, max_colors=12)
                if palette:
                    dspec["_unified_palette"] = palette
                    _emit(q, "translate", "progress",
                          f"Extracted {len(palette)} Tableau colors for PBI theme")
        except Exception as color_err:
            logger.warning(f"Color extraction (non-fatal): {color_err}")

        sections = config.get("report_layout", {}).get("sections", [])
        total_visuals = sum(len(s.get("visualContainers", [])) for s in sections)
        measure_count = len(config.get("tmdl_model", {}).get("tables", [{}])[0].get("measures", []))

        _emit(q, "translate", "complete",
              f"{len(df)} rows, {len(df.columns)} cols, {len(sections)} pages, "
              f"{total_visuals} visuals, {measure_count} DAX measures",
              rows=len(df), cols=len(df.columns), pages=len(sections),
              visuals=total_visuals, measures=measure_count)

        # ---- PHASE: BUILD ----
        _emit(q, "build", "start", "Generating PBIP project files")

        out_dir = str(OUTPUT_DIR / f"v2_{job_id}")
        from generators.pbip_generator import PBIPGenerator
        gen = PBIPGenerator(out_dir)
        gen_result = gen.generate(
            config, profile, dspec,
            data_file_path=None,
            dataframe=df,
        )
        pbip_path = gen_result["path"]
        file_count = gen_result.get("file_count", 0)

        _emit(q, "build", "complete",
              f"{file_count} files generated",
              pbip_path=pbip_path, file_count=file_count)

        # ---- PHASE: VALIDATE ----
        _emit(q, "validate", "start", "Running preflight checks")

        from core.preflight_validator import validate as preflight
        pf = preflight(pbip_path)

        if not pf.all_passed:
            try:
                from core.pbip_healer import heal
                fixes = heal(pbip_path, pf)
                _emit(q, "validate", "progress",
                      f"{pf.fail_count} issues found, {len(fixes)} auto-healed")
                pf = preflight(pbip_path)
            except Exception as heal_err:
                _emit(q, "validate", "progress",
                      f"Healer error (non-fatal): {heal_err}")

        _emit(q, "validate", "complete",
              f"{pf.pass_count} passed, {pf.fail_count} failed",
              passed=pf.pass_count, failed=pf.fail_count)

        # ---- PHASE: PUBLISH ----
        _emit(q, "publish", "start", "Publishing to Power BI")

        report_url = ""
        report_id = ""
        sm_id = ""
        fidelity = {"score": 0, "data": 0, "structure": 0, "quality": 0}

        try:
            from core.powerbi_publisher import get_access_token, list_workspaces, publish_pbip

            token = get_access_token()
            workspaces = list_workspaces(token)

            if not workspaces:
                _emit(q, "publish", "error",
                      "No Power BI workspaces found. Add service principal to a workspace.")
            else:
                target = workspaces[0]
                ws_id = target["id"]
                ws_name = target.get("displayName", "?")

                _emit(q, "publish", "progress",
                      f"Target workspace: {ws_name}")

                display_name = filename.replace(".twbx", "").replace(".twb", "")[:40]
                display_name = f"DrData-{display_name}-{job_id[:6]}"

                pub = publish_pbip(token, ws_id, pbip_path, display_name)

                if pub.get("error"):
                    _emit(q, "publish", "error",
                          f"Publish failed: {pub.get('error')}")
                else:
                    report_id = pub.get("report_id", "")
                    sm_id = pub.get("semantic_model_id", "")
                    report_url = pub.get("report_url", "")

                    _emit(q, "publish", "progress",
                          f"Published. Running QA Agent (3-loop self-heal)...")

                    # Full QA Agent: read-back, score, diagnose, fix, republish
                    try:
                        from core.qa_agent import QAAgent
                        qa = QAAgent(
                            source_spec=spec,
                            dataframe=df,
                            config=config,
                        )
                        qa_result = qa.run_post_publish_qa(
                            token=token,
                            workspace_id=ws_id,
                            report_id=report_id,
                            semantic_model_id=sm_id,
                            pbip_path=pbip_path,
                            data_file_path=os.path.join(
                                os.path.dirname(pbip_path), "Data.csv"),
                            data_profile=profile,
                        )
                        fidelity = qa_result.get("fidelity", {})
                        report_url = qa_result.get("report_url", report_url)
                        report_id = qa_result.get("report_id", report_id)
                        loops = qa_result.get("loops", 1)
                        score = fidelity.get("score", 0)

                        _emit(q, "qa", "complete",
                              f"Fidelity: {score}% "
                              f"(identity={fidelity.get('tab_identity', 0)} "
                              f"charts={fidelity.get('chart_types', 0)} "
                              f"fields={fidelity.get('field_bindings', 0)} "
                              f"layout={fidelity.get('layout', 0)}) "
                              f"[{loops} QA loops]",
                              fidelity=fidelity)
                    except Exception as qa_err:
                        logger.warning(f"QA Agent error (non-fatal): {qa_err}")
                        fidelity = {"score": 50, "data": 20, "structure": 20, "quality": 10}
                        _emit(q, "qa", "complete",
                              f"QA skipped: {qa_err}", fidelity=fidelity)

        except Exception as pub_err:
            _emit(q, "publish", "error", f"Publish error: {pub_err}")

        # ---- PHASE: DONE ----
        result = {
            "report_url": report_url,
            "report_id": report_id,
            "semantic_model_id": sm_id,
            "fidelity": fidelity,
            "pbip_path": pbip_path,
            "file_count": file_count,
            "worksheets": ws_names,
            "dashboards": db_names,
            "rows": len(df),
            "columns": len(df.columns),
            "pages": len(sections),
            "visuals": total_visuals,
            "measures": measure_count,
        }
        _job_results[job_id] = result
        _jobs[job_id]["status"] = "complete"

        _emit(q, "done", "complete", "Pipeline finished", **result)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _emit(q, "error", "failed", str(e)[:500])


# ------------------------------------------------------------------ #
#  Tableau Cloud Bridge Endpoints
# ------------------------------------------------------------------ #

# Stores for cloud bridge jobs (separate from upload pipeline)
_cloud_jobs: Dict[str, dict] = {}
_cloud_results: Dict[str, dict] = {}


@app.get("/api/tableau/workbooks")
async def list_tableau_workbooks():
    """List workbooks from Tableau Cloud."""
    try:
        from core.tableau_connector import TableauCloudConnector
        tc = TableauCloudConnector()
        workbooks = tc.list_workbooks()
        return {"workbooks": workbooks}
    except Exception as e:
        logger.error(f"Tableau workbook list failed: {e}")
        raise HTTPException(500, f"Tableau Cloud error: {str(e)[:300]}")


@app.post("/api/tableau/migrate/{workbook_id}")
async def start_tableau_migration(workbook_id: str, display_name: Optional[str] = None):
    """Start cloud bridge pipeline for a Tableau Cloud workbook."""
    job_id = uuid.uuid4().hex[:12]
    q = queue.Queue()

    _cloud_jobs[job_id] = {
        "workbook_id": workbook_id,
        "status": "queued",
        "queue": q,
        "created": datetime.utcnow().isoformat(),
    }

    thread = threading.Thread(
        target=_run_cloud_bridge,
        args=(job_id, workbook_id, display_name, q),
        daemon=True,
    )
    thread.start()

    logger.info(f"Cloud bridge job started: {job_id} for workbook {workbook_id}")
    return {"job_id": job_id, "workbook_id": workbook_id, "status": "started"}


@app.get("/api/tableau/stream/{job_id}")
async def stream_tableau_migration(job_id: str, request: Request):
    """SSE stream of cloud bridge progress."""
    if job_id not in _cloud_jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    q = _cloud_jobs[job_id]["queue"]

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = q.get(block=False)
                yield json.dumps(event, default=str)
                if event.get("phase") in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.3)
                continue

    return EventSourceResponse(event_generator())


@app.get("/api/tableau/result/{job_id}")
async def get_tableau_result(job_id: str):
    """Get completed cloud bridge result."""
    if job_id not in _cloud_jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    result = _cloud_results.get(job_id, {})
    return {
        "job_id": job_id,
        "status": _cloud_jobs[job_id]["status"],
        "result": result,
    }


@app.get("/api/tableau/images/{job_id}/{view_name}")
async def get_tableau_image(job_id: str, view_name: str):
    """Serve a downloaded Tableau view screenshot."""
    result = _cloud_results.get(job_id, {})
    images = result.get("tableau_images", {})
    img_path = images.get(view_name, "")
    if img_path and os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/png")
    raise HTTPException(404, f"Image not found for view '{view_name}'")


def _run_cloud_bridge(job_id: str, workbook_id: str,
                      display_name: Optional[str], q):
    """Background thread: run cloud bridge and emit SSE events."""
    _cloud_jobs[job_id]["status"] = "running"

    def progress_cb(phase, status, detail=""):
        _emit(q, phase, status, detail)

    try:
        from core.cloud_bridge import run_cloud_bridge
        result = run_cloud_bridge(
            workbook_id,
            display_name=display_name,
            progress_callback=progress_cb,
        )

        _cloud_results[job_id] = result

        if result.get("error"):
            _cloud_jobs[job_id]["status"] = "failed"
            _emit(q, "error", "failed", str(result["error"])[:500])
        else:
            _cloud_jobs[job_id]["status"] = "complete"
            _emit(q, "done", "complete", "Cloud bridge finished",
                  report_url=result.get("report_url", ""),
                  fidelity=result.get("fidelity", {}),
                  tableau_images=list(result.get("tableau_images", {}).keys()),
                  workbook_name=result.get("workbook_name", ""),
                  worksheets=result.get("worksheets", []),
                  dashboards=result.get("dashboards", []),
                  pages=result.get("pages", 0),
                  visuals=result.get("visuals", 0),
                  measures=result.get("measures", 0),
                  rows=result.get("rows", 0),
                  data_source=result.get("data_source", ""),
                  )
    except Exception as e:
        logger.exception(f"Cloud bridge failed: {e}")
        _cloud_jobs[job_id]["status"] = "failed"
        _emit(q, "error", "failed", str(e)[:500])


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8503, log_level="info")
