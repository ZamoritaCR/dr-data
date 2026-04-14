"""
Correction Store -- Supabase persistence for analyst corrections.

Every correction becomes learning data. When the pipeline encounters
a similar pattern in the future, it checks corrections first.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger("drdata-v2.corrections")

_client = None
_init_attempted = False


def _get_client():
    """Lazy Supabase client initialization (cached)."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = (
            os.environ.get("SUPABASE_KEY", "")
            or os.environ.get("SUPABASE_SERVICE_KEY", "")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        )
        if not url or not key:
            logger.warning("[CORRECTIONS] No Supabase credentials -- corrections will not persist")
            return None
        _client = create_client(url, key)
        logger.info("[CORRECTIONS] Supabase client initialized")
        return _client
    except Exception as e:
        logger.warning(f"[CORRECTIONS] Supabase init failed: {e}")
        return None


def store_correction(
    session_id: str,
    stage: str,
    field_path: str,
    original_value: Any,
    corrected_value: Any,
    correction_type: str,
    worksheet_name: str = "",
    mark_type: str = "",
    context: Optional[Dict] = None,
) -> bool:
    """Store an analyst correction in Supabase.

    Returns True if stored successfully, False if Supabase unavailable.
    """
    client = _get_client()
    if not client:
        logger.info(f"[CORRECTIONS] Local-only: {correction_type} on {field_path}")
        return False

    try:
        row = {
            "session_id": session_id,
            "stage": stage,
            "field_path": field_path,
            "original_value": json.dumps(original_value) if not isinstance(original_value, str) else original_value,
            "corrected_value": json.dumps(corrected_value) if not isinstance(corrected_value, str) else corrected_value,
            "correction_type": correction_type,
            "worksheet_name": worksheet_name,
            "mark_type": mark_type,
            "context": context or {},
        }
        client.table("drdata_corrections").insert(row).execute()
        logger.info(f"[CORRECTIONS] Stored: {correction_type} on {field_path} ({worksheet_name})")
        return True
    except Exception as e:
        logger.warning(f"[CORRECTIONS] Store failed: {e}")
        return False


def lookup_corrections(
    correction_type: str,
    mark_type: str = "",
    field_path: str = "",
    limit: int = 10,
) -> List[Dict]:
    """Look up past corrections for similar patterns.

    Used by the pipeline to learn from previous analyst feedback.
    Returns most recent corrections matching the pattern.
    """
    client = _get_client()
    if not client:
        return []

    try:
        query = client.table("drdata_corrections") \
            .select("*") \
            .eq("correction_type", correction_type) \
            .order("created_at", desc=True) \
            .limit(limit)

        if mark_type:
            query = query.eq("mark_type", mark_type)
        if field_path:
            query = query.ilike("field_path", f"%{field_path}%")

        result = query.execute()
        return result.data or []
    except Exception as e:
        logger.warning(f"[CORRECTIONS] Lookup failed: {e}")
        return []


def get_correction_stats() -> Dict:
    """Get correction learning statistics."""
    client = _get_client()
    if not client:
        return {"total": 0, "by_type": {}, "connected": False}

    try:
        result = client.table("drdata_corrections") \
            .select("correction_type") \
            .execute()
        rows = result.data or []
        by_type = {}
        for r in rows:
            ct = r.get("correction_type", "unknown")
            by_type[ct] = by_type.get(ct, 0) + 1
        return {"total": len(rows), "by_type": by_type, "connected": True}
    except Exception as e:
        logger.warning(f"[CORRECTIONS] Stats failed: {e}")
        return {"total": 0, "by_type": {}, "connected": False}


def save_session(session_id: str, data: Dict) -> bool:
    """Upsert session state to Supabase for persistence across restarts."""
    client = _get_client()
    if not client:
        return False

    try:
        row = {
            "session_id": session_id,
            "twbx_filename": data.get("twbx_filename", ""),
            "tableau_spec": data.get("tableau_spec", {}),
            "pipeline_state": data.get("pipeline_state", {}),
            "current_stage": data.get("current_stage", 0),
            "config": data.get("config", {}),
            "data_profile": data.get("data_profile", {}),
            "translations": data.get("translations", {}),
            "updated_at": datetime.utcnow().isoformat(),
        }
        client.table("drdata_sessions").upsert(
            row, on_conflict="session_id"
        ).execute()
        return True
    except Exception as e:
        logger.warning(f"[SESSIONS] Save failed: {e}")
        return False


def load_session(session_id: str) -> Optional[Dict]:
    """Load session state from Supabase."""
    client = _get_client()
    if not client:
        return None

    try:
        result = client.table("drdata_sessions") \
            .select("*") \
            .eq("session_id", session_id) \
            .limit(1) \
            .execute()
        if result.data:
            return result.data[0]
        return None
    except Exception:
        return None
