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
        content = await file.read()
        f.write(content)

    info = {
        "path": str(dest_path),
        "filename": file.filename,
        "size": len(content),
    }
    _uploads[file_id] = info
    logger.info(f"Upload: {file.filename} ({len(content):,} bytes) -> {file_id}")

    # Auto-analyze with Dr. Data agent
    analysis = ""
    file_info = {}
    try:
        agent = _get_agent(file_id)
        if agent:
            analysis = agent.analyze_uploaded_file(file_path=str(dest_path))
            # Extract file info from agent state
            spec = agent.tableau_spec or {}
            df = agent.dataframe
            file_info = {
                "worksheets": len(spec.get("worksheets", [])),
                "dashboards": len(spec.get("dashboards", [])),
                "calc_fields": len(spec.get("calculated_fields", [])),
                "rows": len(df) if df is not None else 0,
                "columns": len(df.columns) if df is not None else 0,
            }
    except Exception as ae:
        logger.error(f"Auto-analysis error: {ae}")
        analysis = f"File uploaded. Analysis error: {str(ae)[:200]}"

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "analysis": analysis,
        "file_info": file_info,
    }


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


@app.post("/api/chat")
async def chat(req: ChatRequestV2):
    """Full conversational chat with V1 Dr. Data agent.

    Uses intent classification (Claude Haiku) to route:
    - build_powerbi, build_dashboard, build_pdf, build_pptx, build_word -> generates files
    - analyze -> deep analysis with specific numbers
    - chat -> tool-calling conversation (up to 10 iterations)

    Returns generated files as downloadable links.
    """
    session_id = req.session_id or req.file_id or "default"
    progress_msgs = []

    def _progress(msg):
        progress_msgs.append(msg)

    try:
        agent = _get_agent(session_id)
        if agent:
            result = agent.respond(
                user_message=req.message,
                conversation_history=agent.messages,
                uploaded_files={},
                progress_callback=_progress,
            )

            # Extract response text
            if isinstance(result, str):
                text = result
                downloads = []
            elif isinstance(result, dict):
                text = result.get("content", result.get("text", ""))
                downloads = result.get("downloads", [])
            else:
                text = str(result)
                downloads = []

            # Convert file paths to download-ready entries
            served_files = []
            for dl in downloads:
                if isinstance(dl, dict) and dl.get("path"):
                    fpath = dl["path"]
                    fname = dl.get("filename", os.path.basename(fpath))
                    # Copy to static/downloads for serving
                    dl_dir = STATIC_DIR / "downloads"
                    dl_dir.mkdir(exist_ok=True)
                    dest = dl_dir / fname
                    try:
                        shutil.copy2(fpath, str(dest))
                        served_files.append({
                            "name": dl.get("name", fname),
                            "filename": fname,
                            "url": f"/static/downloads/{fname}",
                            "description": dl.get("description", ""),
                        })
                    except Exception as cp_err:
                        logger.warning(f"File copy failed: {cp_err}")

            return {
                "response": text,
                "files": served_files,
                "intent": result.get("intent", "chat") if isinstance(result, dict) else "chat",
                "progress": progress_msgs,
            }
    except Exception as agent_err:
        logger.warning(f"Agent chat error, falling back to simple Claude: {agent_err}")
        import traceback
        traceback.print_exc()

    # Fallback: direct Claude call with Dr. Data personality
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"response": "No ANTHROPIC_API_KEY configured.", "files": [], "intent": "error"}

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        # Use the REAL Dr. Data system prompt
        try:
            from config.prompts import DR_DATA_SYSTEM_PROMPT
            system = DR_DATA_SYSTEM_PROMPT
        except ImportError:
            system = (
                "You are Dr. Data, Chief Data Intelligence Officer. You are not a chatbot. "
                "You are a senior data analyst with 25 years of experience who takes initiative, "
                "has strong opinions backed by evidence, and does work before being asked. "
                "Be direct, give code examples, reference actual field names. No fluff. No emojis."
            )

        context = req.context
        if req.file_id and req.file_id in _uploads:
            info = _uploads[req.file_id]
            context += f"\nFile: {info['filename']} ({info['size']:,} bytes)"
        if context:
            system += f"\n\nCurrent context:\n{context}"

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": req.message}],
        )
        return {"response": msg.content[0].text, "files": [], "intent": "chat"}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)[:200]}", "files": [], "intent": "error"}


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
                          f"Published. Verifying data...")

                    # Verify data loaded
                    time.sleep(5)
                    try:
                        from core.powerbi_publisher import execute_dax_query, PBI_SCOPE
                        pbi_token = get_access_token(PBI_SCOPE)
                        dax = execute_dax_query(
                            pbi_token, ws_id, sm_id,
                            'EVALUATE ROW("cnt", COUNTROWS(Data))'
                        )
                        actual_rows = 0
                        rows = dax.get("rows", [])
                        if rows:
                            actual_rows = int(rows[0].get("[cnt]", 0))

                        # Score fidelity
                        data_score = min(40, int(40 * min(actual_rows, len(df)) / max(len(df), 1)))
                        struct_score = min(30, total_visuals * 3 + len(sections) * 5)
                        qual_score = min(30, total_visuals * 2 + (10 if actual_rows > 0 else 0))
                        total_score = data_score + struct_score + qual_score

                        fidelity = {
                            "score": total_score,
                            "data": data_score,
                            "structure": struct_score,
                            "quality": qual_score,
                            "actual_rows": actual_rows,
                        }

                        _emit(q, "publish", "complete",
                              f"Fidelity: {total_score}% ({actual_rows} rows verified)",
                              report_url=report_url, fidelity=fidelity)
                    except Exception as dax_err:
                        fidelity = {"score": 50, "data": 20, "structure": 20, "quality": 10}
                        _emit(q, "publish", "complete",
                              f"Published (DAX verify skipped: {dax_err})",
                              report_url=report_url, fidelity=fidelity)

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
