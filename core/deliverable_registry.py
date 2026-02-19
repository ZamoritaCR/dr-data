"""
Deliverable Registry -- persistent history of everything Dr. Data has built.

Stores metadata in deliverable_history.json at the project root.
Provides save, search, and retrieval methods.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from difflib import SequenceMatcher

REGISTRY_PATH = Path(__file__).parent.parent / "deliverable_history.json"


def _load():
    """Load the registry from disk. Returns empty list if missing."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save(records):
    """Write the full registry to disk."""
    try:
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    except (IOError, OSError) as e:
        print(f"[REGISTRY] Failed to save: {e}")


def save_deliverable(metadata):
    """
    Append a deliverable record to the registry.

    metadata should contain:
        type, name, source_file, columns_used, row_count,
        file_path, description, insights_found (all optional except type)
    """
    record = {
        "id": str(uuid.uuid4())[:8],
        "type": metadata.get("type", "unknown"),
        "name": metadata.get("name", "Untitled"),
        "source_file": metadata.get("source_file", ""),
        "columns_used": metadata.get("columns_used", []),
        "row_count": metadata.get("row_count", 0),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_path": metadata.get("file_path", ""),
        "description": metadata.get("description", ""),
        "insights_found": metadata.get("insights_found", ""),
    }
    records = _load()
    records.append(record)
    _save(records)
    return record


def get_all():
    """Return all deliverables, newest first."""
    records = _load()
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def get_recent(n=5):
    """Return the last N deliverables."""
    return get_all()[:n]


def get_by_type(dtype):
    """Filter deliverables by type (e.g. 'powerbi', 'dashboard')."""
    return [r for r in get_all() if r.get("type") == dtype]


def search(query):
    """
    Fuzzy search across name, source_file, columns_used, description.
    Returns matches sorted by relevance (best first).
    """
    if not query or not query.strip():
        return []

    query_lower = query.lower()
    tokens = query_lower.split()
    records = _load()
    scored = []

    for r in records:
        # Build a searchable blob from all relevant fields
        blob = " ".join([
            r.get("name", ""),
            r.get("source_file", ""),
            " ".join(r.get("columns_used", [])),
            r.get("description", ""),
            r.get("type", ""),
            r.get("insights_found", ""),
        ]).lower()

        # Score: token hits + sequence similarity
        token_hits = sum(1 for t in tokens if t in blob)
        seq_score = SequenceMatcher(None, query_lower, blob[:200]).ratio()
        score = token_hits * 2 + seq_score

        if token_hits > 0 or seq_score > 0.3:
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored]
