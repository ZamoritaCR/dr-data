"""Dr. Data V2 -- FastAPI Router.

Wraps existing Dr. Data engines with async job queue, Claude Opus proposals,
synthetic data generation, and SSE progress streaming.

Endpoints:
  POST /drdata/upload              -- upload file, get job_id
  POST /drdata/analyze/{job_id}    -- parse + Claude proposal
  POST /drdata/generate-synthetic/{job_id} -- generate synthetic data
  POST /drdata/build/{job_id}      -- run PBIP generation pipeline
  GET  /drdata/status/{job_id}     -- poll job status
  GET  /drdata/results/{job_id}    -- full results
  GET  /drdata/download/{job_id}   -- download PBIP ZIP
  POST /drdata/chat                -- Dr. Data agent chat (SSE)
  GET  /drdata/health              -- preflight check
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import random
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/drdata", tags=["drdata"])

# -- Paths --
UPLOAD_DIR = Path("/tmp/drdata_uploads")
OUTPUT_DIR = Path("/tmp/drdata_outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DR_DATA_ROOT = Path("/home/zamoritacr/taop-repos/dr-data")
if str(DR_DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DR_DATA_ROOT))

# -- In-memory job store --
JOBS: dict[str, dict[str, Any]] = {}
MAX_JOBS = 100

SUPPORTED_EXTENSIONS = {
    "twb", "twbx", "csv", "xlsx", "xls", "json", "txt",
    "pdf", "docx", "pptx", "parquet", "tsv",
}

CLAUDE_MODEL = "claude-opus-4-20250514"


def _prune_jobs():
    if len(JOBS) > MAX_JOBS:
        oldest = sorted(JOBS.keys(), key=lambda k: JOBS[k].get("created_at", 0))
        for k in oldest[: len(JOBS) - MAX_JOBS]:
            job_dir = UPLOAD_DIR / k
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
            out_dir = OUTPUT_DIR / k
            if out_dir.exists():
                shutil.rmtree(out_dir, ignore_errors=True)
            del JOBS[k]


def _get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


def _log(job: dict, msg: str, level: str = "info"):
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "msg": msg}
    job.setdefault("log", []).append(entry)
    logger.info("[drdata %s] %s", job.get("job_id", "?"), msg)


def _get_claude() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")
    return anthropic.Anthropic(api_key=api_key)


# ============================================================
# POST /drdata/upload
# ============================================================

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a file and get a job_id."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: .{ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    _prune_jobs()
    job_id = str(uuid.uuid4())[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    file_path = job_dir / file.filename
    content = await file.read()

    if len(content) > 200_000_000:
        raise HTTPException(status_code=400, detail="File too large (max 200MB)")

    file_path.write_bytes(content)

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "file_type": ext,
        "size_bytes": len(content),
        "status": "uploaded",
        "stage": "uploaded",
        "progress_pct": 0,
        "log": [],
        "analysis": None,
        "proposal": None,
        "build_result": None,
        "output_zip": None,
        "created_at": time.time(),
    }

    _log(JOBS[job_id], f"Uploaded {file.filename} ({len(content):,} bytes, .{ext})")

    return {
        "job_id": job_id,
        "filename": file.filename,
        "file_type": ext,
        "size_bytes": len(content),
        "status": "uploaded",
    }


# ============================================================
# POST /drdata/analyze/{job_id}
# ============================================================

class AnalyzeRequest(BaseModel):
    sheet_name: str = ""


@router.post("/analyze/{job_id}")
async def analyze(job_id: str, req: AnalyzeRequest = AnalyzeRequest()):
    job = _get_job(job_id)
    file_path = job["file_path"]
    ext = job["file_type"]
    job["status"] = "analyzing"
    job["stage"] = "parsing"
    job["progress_pct"] = 10

    analysis: dict[str, Any] = {"file_type": ext, "filename": job["filename"]}

    try:
        # -- Parse based on file type --
        if ext in ("twb", "twbx"):
            _log(job, "Parsing Tableau workbook...")
            from app.file_handler import ingest_file
            result = ingest_file(file_path)
            structure = result.get("report_structure", {})
            dfs = result.get("dataframes", {})

            analysis["report_structure"] = structure
            analysis["worksheets"] = len(structure.get("worksheets", []))
            analysis["calculated_fields"] = len(structure.get("calculated_fields", []))
            analysis["datasources"] = len(structure.get("datasources", []))
            analysis["relationships"] = structure.get("relationships", [])
            analysis["parameters"] = structure.get("parameters", [])
            analysis["has_data"] = bool(dfs)

            if dfs:
                first_key = next(iter(dfs))
                df = dfs[first_key]
                analysis["columns"] = list(df.columns)
                analysis["row_count"] = len(df)
                analysis["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
                analysis["sample_rows"] = json.loads(df.head(5).to_json(orient="records"))
                job["dataframe_path"] = str(UPLOAD_DIR / job_id / f"{first_key}.csv")
                df.to_csv(job["dataframe_path"], index=False)
            else:
                # Extract column info from Tableau spec
                all_cols = set()
                for ds in structure.get("datasources", []):
                    for col in ds.get("columns", []):
                        all_cols.add(col.get("caption") or col.get("name", ""))
                for ws in structure.get("worksheets", []):
                    for f in ws.get("rows_fields", []) + ws.get("cols_fields", []):
                        all_cols.add(f)
                analysis["columns"] = sorted(c for c in all_cols if c)
                analysis["needs_synthetic_data"] = True

            job["progress_pct"] = 30

        elif ext in ("csv", "tsv", "xlsx", "xls", "parquet"):
            _log(job, f"Loading data file ({ext})...")
            if ext == "csv":
                df = pd.read_csv(file_path)
            elif ext == "tsv":
                df = pd.read_csv(file_path, sep="\t")
            elif ext in ("xlsx", "xls"):
                df = pd.read_excel(file_path, sheet_name=req.sheet_name or 0)
            elif ext == "parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.DataFrame()

            analysis["columns"] = list(df.columns)
            analysis["row_count"] = len(df)
            analysis["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
            analysis["sample_rows"] = json.loads(df.head(5).to_json(orient="records"))
            analysis["null_rates"] = {c: round(df[c].isnull().mean(), 3) for c in df.columns}
            analysis["describe"] = json.loads(df.describe(include="all").to_json())
            job["dataframe_path"] = file_path
            job["progress_pct"] = 30

        elif ext == "json":
            _log(job, "Parsing JSON file...")
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                sample = data[:10]
                keys = set()
                for item in sample:
                    if isinstance(item, dict):
                        keys.update(item.keys())
                analysis["columns"] = sorted(keys)
                analysis["row_count"] = len(data)
                analysis["sample_rows"] = sample[:5]
            elif isinstance(data, dict):
                analysis["keys"] = list(data.keys())[:50]
                analysis["structure"] = "object"
            job["progress_pct"] = 30

        elif ext == "pdf":
            _log(job, "Extracting text from PDF...")
            try:
                import pdfplumber
                text_parts = []
                tables_found = 0
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages[:20]):
                        text_parts.append(page.extract_text() or "")
                        tables = page.extract_tables()
                        tables_found += len(tables)
                analysis["page_count"] = len(pdf.pages) if hasattr(pdf, "pages") else 0
                analysis["text_preview"] = "\n".join(text_parts)[:5000]
                analysis["tables_found"] = tables_found
            except ImportError:
                analysis["error"] = "pdfplumber not installed"
            job["progress_pct"] = 30

        elif ext == "docx":
            _log(job, "Extracting text from DOCX...")
            try:
                from docx import Document
                doc = Document(file_path)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                tables = []
                for table in doc.tables:
                    rows = [[cell.text for cell in row.cells] for row in table.rows]
                    tables.append(rows[:10])
                analysis["paragraphs"] = len(paragraphs)
                analysis["text_preview"] = "\n".join(paragraphs[:50])[:5000]
                analysis["tables"] = tables[:5]
            except ImportError:
                analysis["error"] = "python-docx not installed"
            job["progress_pct"] = 30

        elif ext == "pptx":
            _log(job, "Extracting text from PPTX...")
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                slides_text = []
                for slide in prs.slides:
                    slide_parts = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_parts.append(shape.text)
                    slides_text.append("\n".join(slide_parts))
                analysis["slide_count"] = len(prs.slides)
                analysis["text_preview"] = "\n---\n".join(slides_text[:20])[:5000]
            except ImportError:
                analysis["error"] = "python-pptx not installed"
            job["progress_pct"] = 30

        elif ext == "txt":
            _log(job, "Reading text file...")
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")[:50000]
            analysis["text_preview"] = text[:5000]
            analysis["char_count"] = len(text)
            job["progress_pct"] = 30

        else:
            analysis["error"] = f"No analyzer for .{ext}"

        # -- Claude Opus proposal --
        _log(job, "Generating dashboard proposal with Claude Opus...")
        job["stage"] = "proposing"
        job["progress_pct"] = 50

        summary = json.dumps(
            {k: v for k, v in analysis.items() if k != "describe"},
            default=str,
            indent=2,
        )[:8000]

        client = _get_claude()
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": (
                    "You are Dr. Data, an expert BI analyst AI.\n"
                    "Analyze this data schema and propose a Power BI dashboard.\n\n"
                    f"File type: {ext}\n"
                    f"Schema/Content summary:\n{summary}\n\n"
                    "Return ONLY valid JSON with this exact structure:\n"
                    "{\n"
                    '  "domain": "what this data is about (1 sentence)",\n'
                    '  "business_questions": ["question 1", "question 2", "question 3"],\n'
                    '  "proposed_visuals": [\n'
                    '    {"type": "lineChart|barChart|scatterChart|pieChart|card|treemap|filledMap|matrix", '
                    '"title": "...", "x_field": "...", "y_field": "...", "color_field": "...", "why": "..."}\n'
                    "  ],\n"
                    '  "data_model": {\n'
                    '    "fact_table": "...",\n'
                    '    "dimensions": ["..."],\n'
                    '    "key_measures": [{"name": "...", "dax": "...", "description": "..."}],\n'
                    '    "relationships": [{"from": "...", "to": "...", "type": "many-to-one"}]\n'
                    "  },\n"
                    '  "needs_synthetic_data": true/false,\n'
                    '  "layout_suggestion": "describe the layout"\n'
                    "}"
                ),
            }],
        )

        proposal_text = msg.content[0].text
        # Parse JSON from response (handle markdown code fences)
        proposal_text = proposal_text.strip()
        if proposal_text.startswith("```"):
            proposal_text = proposal_text.split("\n", 1)[1]
            if proposal_text.endswith("```"):
                proposal_text = proposal_text[:-3]

        try:
            proposal = json.loads(proposal_text)
        except json.JSONDecodeError:
            proposal = {"raw": proposal_text, "parse_error": "Claude response was not valid JSON"}

        job["analysis"] = analysis
        job["proposal"] = proposal
        job["status"] = "analyzed"
        job["stage"] = "analyzed"
        job["progress_pct"] = 100

        _log(job, f"Analysis complete. Proposal: {proposal.get('domain', 'unknown domain')}")

        return {
            "job_id": job_id,
            "analysis": analysis,
            "proposal": proposal,
            "needs_synthetic_data": analysis.get("needs_synthetic_data", False),
            "status": "analyzed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Analysis failed for job %s", job_id)
        job["status"] = "failed"
        job["stage"] = "error"
        _log(job, f"Analysis failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ============================================================
# POST /drdata/generate-synthetic/{job_id}
# ============================================================

class SyntheticRequest(BaseModel):
    mode: str = "quick"  # "quick" or "deep"
    rows: int = 100


@router.post("/generate-synthetic/{job_id}")
async def generate_synthetic(job_id: str, req: SyntheticRequest = SyntheticRequest()):
    job = _get_job(job_id)
    analysis = job.get("analysis")
    if not analysis:
        raise HTTPException(status_code=409, detail="Run /analyze first")

    columns = analysis.get("columns", [])
    if not columns:
        raise HTTPException(status_code=422, detail="No columns found in analysis")

    _log(job, f"Generating {req.rows} synthetic rows ({req.mode} mode)...")
    job["status"] = "generating_synthetic"
    job["stage"] = "synthetic"
    job["progress_pct"] = 30
    rows_count = min(req.rows, 10000)

    if req.mode == "deep":
        # Claude generates coherent business data
        client = _get_claude()
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": (
                    f"Generate exactly {rows_count} rows of realistic CSV data with these columns:\n"
                    f"{', '.join(columns)}\n\n"
                    f"Domain: {job.get('proposal', {}).get('domain', 'business data')}\n\n"
                    "Requirements:\n"
                    "- Data should tell a coherent business story with trends\n"
                    "- Include some outliers and null values for realism\n"
                    "- Use realistic names, dates, and values\n"
                    "- Return ONLY the CSV with header row, no explanation\n"
                ),
            }],
        )
        csv_text = msg.content[0].text.strip()
        if csv_text.startswith("```"):
            csv_text = csv_text.split("\n", 1)[1]
            if csv_text.endswith("```"):
                csv_text = csv_text[:-3]

        df = pd.read_csv(io.StringIO(csv_text))

    else:
        # Quick mode: rule-based generation
        from faker import Faker
        fake = Faker()
        data = {}
        for col in columns:
            cl = col.lower()
            if any(k in cl for k in ("date", "time", "day", "month", "year")):
                data[col] = pd.date_range("2023-01-01", periods=rows_count, freq="D").tolist()[:rows_count]
                if rows_count > 365:
                    data[col] = [fake.date_between("-2y", "today") for _ in range(rows_count)]
            elif any(k in cl for k in ("revenue", "sales", "amount", "price", "cost", "profit", "total")):
                data[col] = [round(random.gauss(1000, 500), 2) for _ in range(rows_count)]
                data[col] = [max(0, v) for v in data[col]]
            elif any(k in cl for k in ("quantity", "count", "units", "qty")):
                data[col] = [random.randint(1, 200) for _ in range(rows_count)]
            elif any(k in cl for k in ("region", "country", "state")):
                choices = ["North", "South", "East", "West", "Central"]
                data[col] = [random.choice(choices) for _ in range(rows_count)]
            elif any(k in cl for k in ("city",)):
                data[col] = [fake.city() for _ in range(rows_count)]
            elif any(k in cl for k in ("category", "segment", "type", "class", "group")):
                cats = [fake.word().capitalize() for _ in range(6)]
                data[col] = [random.choice(cats) for _ in range(rows_count)]
            elif any(k in cl for k in ("name", "customer", "employee", "person")):
                data[col] = [fake.name() for _ in range(rows_count)]
            elif any(k in cl for k in ("email",)):
                data[col] = [fake.email() for _ in range(rows_count)]
            elif any(k in cl for k in ("id", "key", "code", "number")):
                data[col] = list(range(1, rows_count + 1))
            elif any(k in cl for k in ("percent", "rate", "ratio", "pct")):
                data[col] = [round(random.uniform(0, 1), 3) for _ in range(rows_count)]
            elif any(k in cl for k in ("status", "flag", "bool")):
                data[col] = [random.choice(["Active", "Inactive", "Pending"]) for _ in range(rows_count)]
            else:
                # Default: categorical with 5-8 unique values
                cats = [fake.word().capitalize() for _ in range(random.randint(5, 8))]
                data[col] = [random.choice(cats) for _ in range(rows_count)]

        df = pd.DataFrame(data)

    # Save
    synth_path = UPLOAD_DIR / job_id / "synthetic_data.csv"
    df.to_csv(synth_path, index=False)
    job["dataframe_path"] = str(synth_path)
    job["status"] = "synthetic_ready"
    job["stage"] = "synthetic_ready"
    job["progress_pct"] = 100

    _log(job, f"Generated {len(df)} rows, {len(df.columns)} columns")

    return {
        "job_id": job_id,
        "rows_generated": len(df),
        "columns": list(df.columns),
        "sample_rows": json.loads(df.head(5).to_json(orient="records")),
        "status": "synthetic_ready",
    }


# ============================================================
# POST /drdata/build/{job_id}
# ============================================================

class BuildRequest(BaseModel):
    confirmed_visuals: list[dict] = []
    confirmed_layout: str = "auto"
    output_format: str = "pbip"
    project_name: str = ""


@router.post("/build/{job_id}")
async def build(job_id: str, req: BuildRequest):
    job = _get_job(job_id)

    if not job.get("analysis"):
        raise HTTPException(status_code=409, detail="Run /analyze first")

    analysis = job["analysis"]
    proposal = job.get("proposal", {})
    visuals = req.confirmed_visuals or proposal.get("proposed_visuals", [])
    project_name = req.project_name or job["filename"].rsplit(".", 1)[0].replace(" ", "_")

    job["status"] = "building"
    job["stage"] = "building"
    job["progress_pct"] = 10

    try:
        # Load data if available
        df_path = job.get("dataframe_path")
        df = None
        if df_path and os.path.isfile(df_path):
            try:
                df = pd.read_csv(df_path)
                _log(job, f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
            except Exception:
                try:
                    df = pd.read_excel(df_path)
                except Exception:
                    pass

        if df is None:
            raise HTTPException(status_code=422, detail="No data available. Generate synthetic data first.")

        # Profile the data for PBIP generator
        _log(job, "Profiling data for PBIP generation...")
        job["progress_pct"] = 20

        from core.deep_analyzer import DeepAnalyzer
        analyzer = DeepAnalyzer()
        data_profile = analyzer.profile(df)
        data_profile["table_name"] = project_name

        # Build dashboard spec from proposal
        _log(job, "Building dashboard specification...")
        job["progress_pct"] = 30
        job["stage"] = "designing"

        measures = proposal.get("data_model", {}).get("key_measures", [])
        pages = [{
            "title": project_name,
            "name": "Page 1",
            "visuals": visuals,
        }]

        dashboard_spec = {
            "title": project_name,
            "pages": pages,
            "measures": [
                {"name": m["name"], "dax": m["dax"], "format": "#,0"}
                for m in measures
            ],
        }

        # Generate PBIP config with Claude
        _log(job, "Claude generating Power BI layout and TMDL model...")
        job["progress_pct"] = 50
        job["stage"] = "generating_pbip"

        from core.openai_engine import OpenAIEngine
        engine = OpenAIEngine()
        pbip_config = engine.generate_pbip_config(dashboard_spec, data_profile)

        # Build PBIP
        _log(job, "Writing PBIP project files...")
        job["progress_pct"] = 70

        out_dir = OUTPUT_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        from generators.pbip_generator import PBIPGenerator
        generator = PBIPGenerator(str(out_dir))
        gen_result = generator.generate(
            config=pbip_config,
            data_profile=data_profile,
            dashboard_spec=dashboard_spec,
            data_file_path=df_path,
        )

        result_path = gen_result["path"]
        valid_measures = gen_result.get("valid_measures", [])
        field_audit = gen_result.get("field_audit", {})

        # Copy data file into project
        if df_path and os.path.isfile(df_path):
            dst = os.path.join(result_path, os.path.basename(df_path))
            if not os.path.exists(dst):
                shutil.copy2(df_path, dst)

        # ZIP it
        _log(job, "Packaging ZIP...")
        job["progress_pct"] = 90
        job["stage"] = "packaging"

        zip_path = shutil.make_archive(str(out_dir / project_name), "zip", result_path)

        job["output_zip"] = zip_path
        job["build_result"] = {
            "project_name": project_name,
            "path": result_path,
            "zip_path": zip_path,
            "file_count": gen_result.get("file_count", 0),
            "measures_validated": len(valid_measures),
            "field_audit": field_audit,
        }
        job["status"] = "completed"
        job["stage"] = "done"
        job["progress_pct"] = 100

        _log(job, f"PBIP build complete: {gen_result.get('file_count', '?')} files, {len(valid_measures)} measures")

        return {
            "job_id": job_id,
            "status": "completed",
            "project_name": project_name,
            "file_count": gen_result.get("file_count", 0),
            "measures_validated": len(valid_measures),
            "field_audit": field_audit,
            "download_url": f"/drdata/download/{job_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Build failed for job %s", job_id)
        job["status"] = "failed"
        job["stage"] = "error"
        _log(job, f"Build failed: {e}", "error")
        raise HTTPException(status_code=500, detail=f"Build failed: {e}")


# ============================================================
# GET /drdata/status/{job_id}
# ============================================================

@router.get("/status/{job_id}")
async def status(job_id: str):
    job = _get_job(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress_pct": job["progress_pct"],
        "log": job.get("log", [])[-20:],
        "partial_results": {
            k: job.get(k)
            for k in ("analysis", "proposal", "build_result")
            if job.get(k) is not None
        },
    }


# ============================================================
# GET /drdata/results/{job_id}
# ============================================================

@router.get("/results/{job_id}")
async def results(job_id: str):
    job = _get_job(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "file_type": job["file_type"],
        "analysis": job.get("analysis"),
        "proposal": job.get("proposal"),
        "build_result": job.get("build_result"),
        "download_url": f"/drdata/download/{job_id}" if job.get("output_zip") else None,
    }


# ============================================================
# GET /drdata/download/{job_id}
# ============================================================

@router.get("/download/{job_id}")
async def download(job_id: str):
    job = _get_job(job_id)
    zip_path = job.get("output_zip")
    if not zip_path or not os.path.isfile(zip_path):
        raise HTTPException(status_code=404, detail="No output file available. Run /build first.")

    filename = Path(zip_path).name
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=filename,
    )


# ============================================================
# POST /drdata/chat
# ============================================================

class ChatRequest(BaseModel):
    job_id: str = ""
    message: str
    history: list[dict] = []


@router.post("/chat")
async def chat(req: ChatRequest):
    """Dr. Data agent chat with SSE streaming."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # Build context from job if available
    context_parts = []
    if req.job_id and req.job_id in JOBS:
        job = JOBS[req.job_id]
        if job.get("analysis"):
            a = job["analysis"]
            context_parts.append(
                f"CURRENT FILE: {job['filename']} ({job['file_type']})\n"
                f"Columns: {', '.join(a.get('columns', [])[:30])}\n"
                f"Rows: {a.get('row_count', 'unknown')}"
            )
        if job.get("proposal"):
            p = job["proposal"]
            context_parts.append(f"PROPOSAL DOMAIN: {p.get('domain', 'unknown')}")
            visuals = p.get("proposed_visuals", [])
            if visuals:
                context_parts.append(
                    "PROPOSED VISUALS:\n" +
                    "\n".join(f"- {v.get('title')}: {v.get('type')}" for v in visuals[:10])
                )
        if job.get("build_result"):
            br = job["build_result"]
            context_parts.append(
                f"BUILD RESULT: {br.get('file_count', 0)} files, "
                f"{br.get('measures_validated', 0)} measures"
            )

    context_block = "\n\n".join(context_parts) if context_parts else "No job context loaded."

    system_prompt = (
        "You are Dr. Data -- the AI intelligence layer of the Dr. Data migration tool "
        "built by The Art of the Possible (TAOP).\n\n"
        "Your personality: Direct, expert, zero fluff. You think like a senior BI architect "
        "who has migrated hundreds of Tableau workbooks to Power BI. You genuinely understand "
        "the user's data and their specific job.\n\n"
        f"Current job context:\n{context_block}\n\n"
        "Rules:\n"
        "- NEVER give canned or generic responses. Always reference the specific job, "
        "specific fields, specific visuals.\n"
        "- If asked about a translation decision, explain exactly what you did and why.\n"
        "- If something was flagged for review, explain the specific DAX limitation and the best fix.\n"
        "- You have expert-level knowledge of Tableau, Power BI, DAX, TMDL, and data modeling.\n"
        "- Be concise. Enterprise users don't want essays."
    )

    messages = []
    for m in req.history[-20:]:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": req.message})

    async def stream():
        client = _get_claude()
        try:
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=messages,
            ) as s:
                for text in s.text_stream:
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("Dr. Data chat error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ============================================================
# GET /drdata/health
# ============================================================

@router.get("/health")
async def health():
    checks = {}

    # Check engines importable
    for name, mod_path in [
        ("parser", "core.enhanced_tableau_parser"),
        ("field_resolver", "core.field_resolver"),
        ("formula_transpiler", "core.formula_transpiler"),
        ("pbip_generator", "generators.pbip_generator"),
        ("deep_analyzer", "core.deep_analyzer"),
    ]:
        try:
            __import__(mod_path)
            checks[name] = True
        except Exception as e:
            checks[name] = f"FAIL: {e}"

    # Check rapidfuzz
    try:
        import rapidfuzz  # noqa: F401
        checks["rapidfuzz"] = True
    except ImportError:
        checks["rapidfuzz"] = False

    # Check API keys
    api_keys = {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY")),
    }

    # Disk space
    import shutil as _shutil
    disk = _shutil.disk_usage("/tmp")
    disk_free_gb = round(disk.free / (1024**3), 1)

    all_ok = all(v is True for v in checks.values()) and api_keys["anthropic"]
    return {
        "status": "ok" if all_ok else "degraded",
        "service": "dr-data-v2",
        "engines": checks,
        "api_keys": api_keys,
        "disk_free_gb": disk_free_gb,
        "active_jobs": len(JOBS),
    }
