# Dr. Data -- Tableau-to-Power-BI Migration Tool

## Project Overview

Streamlit app that converts Tableau workbooks (.twb/.twbx) into Power BI PBIP projects.
Uses Claude + GPT-4 in a multi-stage pipeline: parse Tableau XML -> build translation prompt -> generate dashboard spec -> generate PBIP files.

## Pipeline Stages

```
Stage 1  enhanced_tableau_parser.parse_twb()        Parse .twb/.twbx XML -> tableau_spec dict
Stage 2  dr_data_agent._build_upload_context()       Store tableau_spec on agent
Stage 3  dr_data_agent._build_tableau_translate_request()  Build text prompt from tableau_spec
Stage 4  claude_interpreter.interpret()              Claude generates dashboard_spec JSON
Stage 5  openai_engine.generate_pbip_config()        GPT-4 generates report_layout + tmdl_model
Stage 6  pbip_generator.generate()                   Write PBIP project files
```

When no data is embedded in the Tableau file, synthetic data is generated at:
- `dr_data_agent.py:1300-1337` (export path)
- `dr_data_agent.py:2701-2717` (recovery path)
Both call `core/synthetic_data.generate_from_tableau_spec(tableau_spec)`.

## Key Files

| File | Purpose |
|------|---------|
| `core/enhanced_tableau_parser.py` | Single authoritative TWB/TWBX parser |
| `core/synthetic_data.py` | Synthetic data generation from tableau_spec |
| `core/visual_intent.py` | Visual intent model (chart type mapping, page layout) |
| `core/field_resolver.py` | Fuzzy field name matching (tableau fields -> data columns) |
| `app/file_handler.py` | File ingestion -- delegates TWB parsing to enhanced parser |
| `app/dr_data_agent.py` | Main agent -- orchestrates the full pipeline |
| `config/prompts.py` | System prompts for Claude/OpenAI stages |
| `generators/pbip_generator.py` | PBIP output file generation |
| `ROOT_CAUSE_DIAGNOSTIC.md` | Full P0-P5 diagnostic of known architecture issues |
| `SESSION_HANDOFF.md` | Detailed handoff from 2026-04-05 session with bug list |

## Active Bugs (as of 2026-04-05)

### BUG 1 -- FIXED: synthetic_data.py filter format mismatch
Fixed. Dict filters from enhanced parser are now handled with `isinstance(filt, dict)` branch.

### BUG 2 -- FIXED: Hardcoded date range in synthetic data
Fixed. Uses `datetime.now()` with 2-year relative window.

### BUG 3 -- FIXED: Silent synthetic data errors
Fixed. Errors now reported to user via `_report_progress()`.

### BUG 4 -- Dead code (NOT fixed -- low priority)
`core/synthetic_data_generator.py` -- entire Faker-based generator is never imported or called.
Kept for potential future use; not causing any runtime issues.

## Recent Changes

- **2026-04-05:** Full pipeline audit and fix (308 tests passing). Key fixes:
  - Shelf field prefix parsing: all Tableau prefixes (tqr:, yr:, mn:, qr:, tmn:, etc.) now resolved
  - Generated fields (Latitude/Longitude (generated)) filtered from synthetic schema
  - Action/Tooltip filters excluded from parser output (were leaking as fake columns)
  - Filter field names cleaned of shelf encoding (none:Region:nk -> Region)
  - Agent prompt builder: fixed wrong keys (columns->cols_fields, worksheets->worksheets_used)
  - Added multipolygon/polygon chart type mappings to visual_intent.py and agent
  - New comprehensive E2E test suite: tests/test_e2e_pipeline.py (33 tests)
- **2026-04-05:** Parser unification (commit `86cd299`). Merged two competing TWB parsers into `core/enhanced_tableau_parser.py`. `file_handler.py` now delegates all TWB/TWBX parsing there.

## Conventions

- No emojis anywhere -- code, comments, UI, docs, conversation
- Python venv: `~/taop-agents-env/`
- Tests: `pytest tests/` (use the venv)
- Output dir: `output/`

## Cross-Machine Context

Global memory (user prefs, infra, project list) is stored in `.claude-memory/` in this repo.
If starting on a new machine without `~/.claude/` memory, read these files first:
- `.claude-memory/MEMORY.md` -- user profile, project list, council architecture, environment
- `.claude-memory/infrastructure.md` -- domains, services, ports, cron, tunnels

## Owner

Johan (zamoritacr). Server: ROG Strix 192.168.0.55. Domain: theartofthepossible.io.
