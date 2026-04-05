# Dr. Data -- Session Handoff (2026-04-05)

## What Was Done This Session

### Parser Unification (COMPLETED, PUSHED)
- **Commit:** `86cd299` on `main`, pushed to origin
- **Problem:** Two competing TWB parsers existed -- `app/file_handler.py:parse_twb()` (old, used by agent) and `core/enhanced_tableau_parser.py:parse_twb()` (new, never called).
- **Fix:** Merged best of both into `core/enhanced_tableau_parser.py` as the single authoritative parser. `file_handler.py` now delegates to it.
- **Changes:**
  - `core/enhanced_tableau_parser.py` -- full rewrite with chart type extraction, shelf field parsing, visual encoding extraction (color/size/label/tooltip), dashboard zone deduplication, `.hyper` file detection, join relationship parsing
  - `app/file_handler.py` -- removed 200+ lines of duplicate parsing code; `parse_twb()` and `parse_twbx()` now delegate to enhanced parser; old helper functions (`_extract_mark_type`, `_extract_shelf_fields`, `_extract_ws_filters`) removed
- **Backward compat:** Worksheets now have BOTH `chart_type` and `mark_type`, BOTH `rows`/`cols` (raw strings) and `rows_fields`/`cols_fields` (parsed lists), plus `dimensions`/`measures` from encoding shelves
- **Tests:** 275 passed, 0 failures
- **Backup:** `../dr-data-backup-20260405_*`

---

## Known Bugs to Fix (Synthetic Data)

### BUG 1 -- CRITICAL: Filter format incompatibility in synthetic_data.py

**File:** `core/synthetic_data.py`, lines 196-211
**Problem:** The enhanced parser changed filter format from strings to dicts:
```python
# OLD format (what synthetic_data.py expects):
filters: ["[federated.xxx].[none:Region:nk]", "[federated.xxx].[yr:Date:ok]"]

# NEW format (what enhanced parser now outputs):
filters: [
    {"field": "Region", "column_ref": "[federated.xxx].[none:Region:nk]", "type": "categorical"},
    {"field": "Date", "column_ref": "[federated.xxx].[yr:Date:ok]", "type": "range"}
]
```

The code at line 196-197:
```python
for filt in ws.get("filters", []):
    if isinstance(filt, str):   # <-- NEW dicts silently skipped!
```

**Impact:** Filter-only dimensions (e.g., Region, Year) are silently dropped from synthetic data schema. Data is incomplete but no crash/error.

**Fix:** Add `elif isinstance(filt, dict):` branch to extract `filt.get("field")` or parse `filt.get("column_ref")`.

### BUG 2 -- MODERATE: Hardcoded date range

**File:** `core/synthetic_data.py`, around line 275
**Problem:** Date generation uses `datetime(2023, 1, 1)` to `datetime(2025, 3, 31)` -- stale for 2026.
**Fix:** Use `datetime.now() - timedelta(days=730)` to `datetime.now()`.

### BUG 3 -- MODERATE: Silent error suppression

**File:** `app/dr_data_agent.py`, lines 1334-1337 and 2716-2717
**Problem:** Synthetic data failures are printed to console but never shown to user.
```python
except Exception as synth_err:
    print(f"[SYNTH] Failed to generate synthetic data: {synth_err}")
```
**Fix:** Call `self._report_progress()` or similar to surface errors in the UI.

### BUG 4 -- LOW: Dead code

**File:** `core/synthetic_data_generator.py`
**Problem:** Entire Faker-based generator class (`SyntheticDataGenerator`) with QUICK/DEEP modes is never imported or called by anything. Dead code.
**Fix:** Delete or integrate.

### BUG 5 -- LOW: No CSV path validation

**File:** `app/dr_data_agent.py`, lines 1316-1323
**Problem:** After `generate_from_tableau_spec()` returns `csv_path`, it's assigned to `self.data_file_path` without checking the file actually exists.

---

## Architecture Quick Reference

```
Upload Flow:
  file_handler.ingest_file()
    -> enhanced_tableau_parser.parse_twb()     # FIXED this session
    -> returns {type, worksheets, dashboards, datasources, ...}
    -> stored as agent.tableau_spec

Synthetic Data Flow (when no embedded data):
  dr_data_agent.py line 1300-1337  (export path)
  dr_data_agent.py line 2701-2717  (recovery path)
    -> core/synthetic_data.generate_from_tableau_spec(tableau_spec)
    -> extract_schema_from_tableau()    # <-- BUG 1 HERE
    -> generate_synthetic_dataframe()
    -> returns (df, csv_path, schema)

Translation Flow:
  dr_data_agent._build_tableau_translate_request()  (line ~2580)
    -> builds text prompt from tableau_spec
    -> claude_interpreter.interpret()               (Stage 4)
    -> openai_engine.generate_pbip_config()          (Stage 5)
    -> pbip_generator.generate()                     (Stage 6)
```

## Key Files

| File | Purpose |
|------|---------|
| `core/enhanced_tableau_parser.py` | Single TWB/TWBX parser (FIXED) |
| `core/synthetic_data.py` | Synthetic data generation (BUGGY -- fix next) |
| `app/file_handler.py` | File ingestion, delegates to enhanced parser |
| `app/dr_data_agent.py` | Main agent, calls synthetic data at lines 1316, 2706 |
| `core/visual_intent.py` | Visual intent model (consumes parser output) |
| `config/prompts.py` | System prompts for Claude/OpenAI |
| `generators/pbip_generator.py` | PBIP output generation |
| `ROOT_CAUSE_DIAGNOSTIC.md` | Full P0-P5 diagnostic (still valid) |

## Priority for Next Session

1. **Fix BUG 1** (filter format in synthetic_data.py) -- 5 min fix, critical impact
2. **Fix BUG 2** (date range) -- 2 min fix
3. **Fix BUG 3** (error surfacing) -- 10 min fix
4. Then continue with P1-P5 from ROOT_CAUSE_DIAGNOSTIC.md
