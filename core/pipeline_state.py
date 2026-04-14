"""
Dr. Data V2 -- Human-Gated Pipeline State Machine

Every stage produces output, presents it to the analyst in chat, and STOPS.
The pipeline only advances when the analyst explicitly approves or corrects.
All decisions are logged with who made them and when.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    CORRECTED = "corrected"
    SKIPPED = "skipped"
    FAILED = "failed"


class StageName(str, Enum):
    PARSE = "parse"
    DATA_SOURCE = "data_source"
    FIELD_MAPPING = "field_mapping"
    FORMULA_TRANS = "formula_trans"
    VISUAL_MAPPING = "visual_mapping"
    SEMANTIC_MODEL = "semantic_model"
    REPORT_LAYOUT = "report_layout"
    GENERATE = "generate"


STAGE_ORDER = list(StageName)

STAGE_DESCRIPTIONS = {
    StageName.PARSE: "Parse Tableau workbook structure",
    StageName.DATA_SOURCE: "Choose data source (Excel, DB, synthetic, wireframe)",
    StageName.FIELD_MAPPING: "Map Tableau fields to Power BI columns",
    StageName.FORMULA_TRANS: "Translate calculated fields to DAX",
    StageName.VISUAL_MAPPING: "Map Tableau visuals to Power BI chart types",
    StageName.SEMANTIC_MODEL: "Build semantic model (tables, relationships, measures)",
    StageName.REPORT_LAYOUT: "Assemble report pages and visual layout",
    StageName.GENERATE: "Generate PBIP project files and validate",
}


@dataclass
class Decision:
    """A single decision made during the pipeline."""
    stage: str
    made_by: str          # "engine", "brain:deepseek", "brain:claude", "analyst"
    decision: str         # what was decided
    alternatives: list    # what other options existed
    timestamp: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class StageResult:
    """Output of a completed stage."""
    stage: str
    status: str
    summary: str                    # human-readable summary for chat
    data: dict = field(default_factory=dict)    # structured output
    qa_notes: list = field(default_factory=list) # QA findings for this stage
    decisions: list = field(default_factory=list) # Decision dicts


class PipelineState:
    """Manages the state machine for a human-gated migration pipeline.

    Usage:
        state = PipelineState(session_id, workbook_name)

        # Run a stage
        state.start_stage("parse")
        result = do_parsing(...)
        state.complete_stage("parse", result_summary, result_data)
        # Now state is AWAITING_APPROVAL for "parse"

        # On analyst approval
        state.approve_stage("parse")
        # Now next stage can run

        # On analyst correction
        state.correct_stage("parse", corrections={...})
    """

    def __init__(self, session_id: str, workbook_name: str = ""):
        self.session_id = session_id
        self.workbook_name = workbook_name
        self.created_at = time.time()
        self.updated_at = time.time()

        # Stage statuses
        self.stages: Dict[str, StageStatus] = {
            s.value: StageStatus.PENDING for s in StageName
        }

        # Stage results
        self.results: Dict[str, StageResult] = {}

        # All decisions across the pipeline
        self.audit_log: List[Decision] = []

        # Analyst corrections (stage -> corrections dict)
        self.corrections: Dict[str, dict] = {}

        # Shared context passed between stages
        self.context: Dict[str, Any] = {
            "workbook_name": workbook_name,
            "tableau_spec": None,
            "data_source_choice": None,
            "dataframe": None,
            "data_profile": None,
            "field_mappings": None,
            "formula_translations": None,
            "visual_mappings": None,
            "config": None,
            "dashboard_spec": None,
            "pbip_path": None,
            "palette": None,
        }

    @property
    def current_stage(self) -> Optional[str]:
        """Return the current stage (first non-completed stage)."""
        for s in STAGE_ORDER:
            status = self.stages[s.value]
            if status not in (StageStatus.APPROVED, StageStatus.CORRECTED, StageStatus.SKIPPED):
                return s.value
        return None

    @property
    def current_stage_index(self) -> int:
        cs = self.current_stage
        if cs is None:
            return len(STAGE_ORDER)
        return STAGE_ORDER.index(StageName(cs))

    @property
    def is_complete(self) -> bool:
        return self.current_stage is None

    @property
    def is_awaiting_approval(self) -> bool:
        cs = self.current_stage
        return cs is not None and self.stages[cs] == StageStatus.AWAITING_APPROVAL

    @property
    def progress_fraction(self) -> float:
        done = sum(1 for s in self.stages.values()
                   if s in (StageStatus.APPROVED, StageStatus.CORRECTED, StageStatus.SKIPPED))
        return done / len(STAGE_ORDER)

    def start_stage(self, stage: str):
        self.stages[stage] = StageStatus.RUNNING
        self.updated_at = time.time()

    def complete_stage(self, stage: str, summary: str, data: dict = None,
                       qa_notes: list = None):
        """Mark stage as complete and awaiting analyst approval."""
        result = StageResult(
            stage=stage,
            status="awaiting_approval",
            summary=summary,
            data=data or {},
            qa_notes=qa_notes or [],
        )
        self.results[stage] = result
        self.stages[stage] = StageStatus.AWAITING_APPROVAL
        self.updated_at = time.time()

    def approve_stage(self, stage: str):
        self.stages[stage] = StageStatus.APPROVED
        self.log_decision(stage, "analyst", "approved", [])
        self.updated_at = time.time()

    def correct_stage(self, stage: str, corrections: dict):
        self.stages[stage] = StageStatus.CORRECTED
        self.corrections[stage] = corrections
        self.log_decision(stage, "analyst", "corrected", [],
                          detail=corrections)
        self.updated_at = time.time()

    def skip_stage(self, stage: str, reason: str = ""):
        self.stages[stage] = StageStatus.SKIPPED
        self.log_decision(stage, "analyst", f"skipped: {reason}", [])
        self.updated_at = time.time()

    def fail_stage(self, stage: str, error: str):
        self.stages[stage] = StageStatus.FAILED
        if stage in self.results:
            self.results[stage].status = "failed"
        self.updated_at = time.time()

    def log_decision(self, stage: str, made_by: str, decision: str,
                     alternatives: list, detail: dict = None):
        d = Decision(
            stage=stage,
            made_by=made_by,
            decision=decision,
            alternatives=alternatives,
            detail=detail or {},
        )
        self.audit_log.append(d)
        # Also attach to stage result if it exists
        if stage in self.results:
            self.results[stage].decisions.append(d.to_dict())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "workbook_name": self.workbook_name,
            "current_stage": self.current_stage,
            "current_stage_index": self.current_stage_index,
            "progress": self.progress_fraction,
            "is_complete": self.is_complete,
            "is_awaiting_approval": self.is_awaiting_approval,
            "stages": {k: v.value for k, v in self.stages.items()},
            "results": {
                k: {
                    "stage": r.stage,
                    "status": r.status,
                    "summary": r.summary,
                    "data": r.data,
                    "qa_notes": r.qa_notes,
                }
                for k, r in self.results.items()
            },
            "corrections": self.corrections,
            "audit_log_count": len(self.audit_log),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def format_stage_summary_for_chat(self, stage: str) -> str:
        """Format a stage result for presentation in the Dr. Data chat."""
        result = self.results.get(stage)
        if not result:
            return ""

        idx = STAGE_ORDER.index(StageName(stage)) + 1
        total = len(STAGE_ORDER)
        desc = STAGE_DESCRIPTIONS.get(StageName(stage), stage)
        status_label = self.stages[stage].value.replace("_", " ").upper()

        lines = [
            f"--- Step {idx}/{total}: {desc} ---",
            f"Status: {status_label}",
            "",
            result.summary,
        ]

        if result.qa_notes:
            lines.append("")
            lines.append("QA Notes:")
            for note in result.qa_notes:
                lines.append(f"  - {note}")

        if self.stages[stage] == StageStatus.AWAITING_APPROVAL:
            lines.append("")
            lines.append("Reply 'approve' to continue, or describe what to change.")

        return "\n".join(lines)
