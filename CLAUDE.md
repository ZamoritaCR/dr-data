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

### BUG 1 -- CRITICAL: synthetic_data.py filter format mismatch
`core/synthetic_data.py:196-211` -- expects filter strings but enhanced parser now outputs filter dicts.
The `isinstance(filt, str)` check silently skips all dict filters, dropping filter-only dimensions from synthetic data.
Fix: add `elif isinstance(filt, dict):` branch using `filt.get("field")` or `filt.get("column_ref")`.

### BUG 2 -- Hardcoded date range in synthetic data
`core/synthetic_data.py:~275` -- dates hardcoded to 2023-2025, stale for 2026.
Fix: use relative dates from `datetime.now()`.

### BUG 3 -- Silent synthetic data errors
`app/dr_data_agent.py:1334-1337, 2716-2717` -- errors printed to console, never shown to user.

### BUG 4 -- Dead code
`core/synthetic_data_generator.py` -- entire Faker-based generator is never imported or called.

## Recent Changes

- **2026-04-05:** Parser unification (commit `86cd299`). Merged two competing TWB parsers into `core/enhanced_tableau_parser.py`. `file_handler.py` now delegates all TWB/TWBX parsing there. 275 tests passing.

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
