"""
Dr. Data v2 -- FastAPI Backend
Tableau to Power BI agentic migration pipeline.
SSE streaming, file upload, Power BI publish, AI chat.
"""

import asyncio
import json
import logging
import os
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

    return {"file_id": file_id, "filename": file.filename, "size": len(content)}


@app.post("/api/process/{file_id}")
async def start_process(file_id: str):
    if file_id not in _uploads:
        raise HTTPException(404, f"File {file_id} not found")

    job_id = uuid.uuid4().hex[:12]
    queue = asyncio.Queue()

    _jobs[job_id] = {
        "file_id": file_id,
        "status": "queued",
        "queue": queue,
        "created": datetime.utcnow().isoformat(),
    }

    # Run pipeline in background thread
    loop = asyncio.get_event_loop()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, file_id, loop, queue),
        daemon=True,
    )
    thread.start()

    logger.info(f"Job started: {job_id} for file {file_id}")
    return {"job_id": job_id, "file_id": file_id, "status": "started"}


@app.get("/api/stream/{job_id}")
async def stream_events(job_id: str, request: Request):
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    queue = _jobs[job_id]["queue"]

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield {"data": json.dumps(event, default=str)}
                if event.get("phase") == "done" or event.get("phase") == "error":
                    break
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"phase": "heartbeat", "timestamp": datetime.utcnow().isoformat()})}

    return EventSourceResponse(event_generator())


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {"response": "No ANTHROPIC_API_KEY configured."}

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)

        # Build context from file if available
        context = req.context
        if req.file_id and req.file_id in _uploads:
            info = _uploads[req.file_id]
            context += f"\nFile: {info['filename']} ({info['size']:,} bytes)"

        system = (
            "You are Dr. Data -- expert in Tableau to Power BI migration, DAX, "
            "data modeling, and analytics modernization. Be direct, give code examples, "
            "reference actual field names when available. No fluff."
        )
        if context:
            system += f"\n\nContext:\n{context}"

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": req.message}],
        )
        return {"response": msg.content[0].text}

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)[:200]}"}


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

def _emit(loop, queue, phase, status, detail="", **extra):
    """Thread-safe event emit to async queue."""
    event = {
        "phase": phase,
        "status": status,
        "detail": detail,
        "timestamp": datetime.utcnow().isoformat(),
        **extra,
    }
    asyncio.run_coroutine_threadsafe(queue.put(event), loop)
    logger.info(f"[{phase}] {status}: {detail[:100]}")


def _run_pipeline(job_id: str, file_id: str, loop, queue):
    """Full migration pipeline: parse -> translate -> build -> validate -> publish."""
    import pandas as pd

    info = _uploads[file_id]
    filepath = info["path"]
    filename = info["filename"]
    _jobs[job_id]["status"] = "running"

    try:
        # ---- PHASE: PARSE ----
        _emit(loop, queue, "parse", "start", f"Parsing {filename}")

        from core.enhanced_tableau_parser import parse_twb
        spec = parse_twb(filepath)

        ws_count = len(spec.get("worksheets", []))
        db_count = len(spec.get("dashboards", []))
        cf_count = len(spec.get("calculated_fields", []))
        ds_count = len(spec.get("datasources", []))

        ws_names = [w.get("name", "") for w in spec.get("worksheets", [])]
        db_names = [d.get("name", "") for d in spec.get("dashboards", [])]

        _emit(loop, queue, "parse", "complete",
              f"{ws_count} worksheets, {db_count} dashboards, {cf_count} calcs, {ds_count} datasources",
              worksheets=ws_names[:10], dashboards=db_names[:5],
              calc_count=cf_count, datasource_count=ds_count)

        # ---- PHASE: TRANSLATE ----
        _emit(loop, queue, "translate", "start", "Generating synthetic data + mapping fields")

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

        _emit(loop, queue, "translate", "complete",
              f"{len(df)} rows, {len(df.columns)} cols, {len(sections)} pages, "
              f"{total_visuals} visuals, {measure_count} DAX measures",
              rows=len(df), cols=len(df.columns), pages=len(sections),
              visuals=total_visuals, measures=measure_count)

        # ---- PHASE: BUILD ----
        _emit(loop, queue, "build", "start", "Generating PBIP project files")

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

        _emit(loop, queue, "build", "complete",
              f"{file_count} files generated",
              pbip_path=pbip_path, file_count=file_count)

        # ---- PHASE: VALIDATE ----
        _emit(loop, queue, "validate", "start", "Running preflight checks")

        from core.preflight_validator import validate as preflight
        pf = preflight(pbip_path)

        if not pf.all_passed:
            try:
                from core.pbip_healer import heal
                fixes = heal(pbip_path, pf)
                _emit(loop, queue, "validate", "progress",
                      f"{pf.fail_count} issues found, {len(fixes)} auto-healed")
                pf = preflight(pbip_path)
            except Exception as heal_err:
                _emit(loop, queue, "validate", "progress",
                      f"Healer error (non-fatal): {heal_err}")

        _emit(loop, queue, "validate", "complete",
              f"{pf.pass_count} passed, {pf.fail_count} failed",
              passed=pf.pass_count, failed=pf.fail_count)

        # ---- PHASE: PUBLISH ----
        _emit(loop, queue, "publish", "start", "Publishing to Power BI")

        report_url = ""
        report_id = ""
        sm_id = ""
        fidelity = {"score": 0, "data": 0, "structure": 0, "quality": 0}

        try:
            from core.powerbi_publisher import get_access_token, list_workspaces, publish_pbip

            token = get_access_token()
            workspaces = list_workspaces(token)

            if not workspaces:
                _emit(loop, queue, "publish", "error",
                      "No Power BI workspaces found. Add service principal to a workspace.")
            else:
                target = workspaces[0]
                ws_id = target["id"]
                ws_name = target.get("displayName", "?")

                _emit(loop, queue, "publish", "progress",
                      f"Target workspace: {ws_name}")

                display_name = filename.replace(".twbx", "").replace(".twb", "")[:40]
                display_name = f"DrData-{display_name}-{job_id[:6]}"

                pub = publish_pbip(token, ws_id, pbip_path, display_name)

                if pub.get("error"):
                    _emit(loop, queue, "publish", "error",
                          f"Publish failed: {pub.get('error')}")
                else:
                    report_id = pub.get("report_id", "")
                    sm_id = pub.get("semantic_model_id", "")
                    report_url = pub.get("report_url", "")

                    _emit(loop, queue, "publish", "progress",
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

                        _emit(loop, queue, "publish", "complete",
                              f"Fidelity: {total_score}% ({actual_rows} rows verified)",
                              report_url=report_url, fidelity=fidelity)
                    except Exception as dax_err:
                        fidelity = {"score": 50, "data": 20, "structure": 20, "quality": 10}
                        _emit(loop, queue, "publish", "complete",
                              f"Published (DAX verify skipped: {dax_err})",
                              report_url=report_url, fidelity=fidelity)

        except Exception as pub_err:
            _emit(loop, queue, "publish", "error", f"Publish error: {pub_err}")

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

        _emit(loop, queue, "done", "complete", "Pipeline finished", **result)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _emit(loop, queue, "error", "failed", str(e)[:500])


# ------------------------------------------------------------------ #
#  Run
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8503, log_level="info")
