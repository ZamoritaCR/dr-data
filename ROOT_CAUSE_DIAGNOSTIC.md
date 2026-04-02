# Dr. Data -- Tableau-to-Power-BI Converter: Root-Cause Diagnostic

## 1. Current Pipeline Stages

```
Stage 1  file_handler.py:parse_twb()     Parse .twb XML -> tableau_spec dict
Stage 2  dr_data_agent.py:_build_upload_context   Store tableau_spec on agent
Stage 3  dr_data_agent.py:_build_tableau_translate_request()  Build text prompt from tableau_spec
Stage 4  claude_interpreter.py:interpret()   Claude generates dashboard_spec JSON
Stage 5  openai_engine.py:generate_pbip_config()  GPT-4 generates report_layout + tmdl_model
Stage 6  pbip_generator.py:generate()     Write PBIP project files (pages, visuals, model)
```

Key observation: the user's natural-language request enters at Stage 3 but is
appended at the very end of the prompt under "ADDITIONAL USER REQUEST" -- after
all the Tableau metadata.  Everything between Stage 1 and Stage 6 is a game of
telephone where information is lost at each handoff.

---

## 2. Where Requirement Loss Occurs

### 2a. parse_twb() extracts almost no visual metadata (file_handler.py:249-269)

- **Marks**: only captures `mark.get("class")` -- misses mark type (bar, line,
  area, circle, etc.) which is the actual chart type signal.
- **Shelves**: only looks at `<encoding attr="columns|rows">` for dimensions
  and `attr="size|color|text"` for measures. This misses:
  - `attr="x"`, `attr="y"` (the real axis encodings in newer TWB XML)
  - `<rows>` and `<cols>` elements (free-text shelf expressions like
    `[Category]`, `SUM([Sales])`)
  - `<mark class="...">` at the worksheet level (the actual chart type)
- **Filters**: worksheet-level filters are not extracted at all.
  `ws_info["filters"]` is always empty.
- **No chart type**: the worksheet dict has no `type` or `chart_type` key,
  so `_build_tableau_translate_request` falls back to "unknown" chart type
  for every worksheet.

### 2b. Dashboard zone layout is extracted as names only (file_handler.py:271-280)

- `parse_twb()` extracts `zone.get("name")` but ignores:
  - `zone` attributes: `x`, `y`, `w`, `h`, `type-v2` (which distinguish
    worksheet zones from filter/blank/text/parameter zones)
  - Zone nesting (tiled vs floating layout)
  - Zone ordering (which determines visual stacking / tab order)
- Result: the `dashboards` list has zone *names* but zero spatial or
  structural information.

### 2c. _build_tableau_translate_request loses dashboard layout (dr_data_agent.py:2580-2614)

- For each dashboard, it builds: `"Dashboard1": contains ['Sheet1','Sheet2'], size=1000x800`
- It says "layout_zones=N zones" but never passes the zone positions, sizes,
  or stacking order.
- The prompt says "map proportionally to PBI 1280x720 canvas" but gives
  Claude no x/y/w/h data to actually do the proportional mapping.

### 2d. INTERPRETER_SYSTEM_PROMPT has no visual-mapping guidance (config/prompts.py:241-308)

- The system prompt tells Claude to generate a generic dashboard spec.
- It does **not** instruct Claude to:
  - Preserve the source Tableau page count
  - Map specific Tableau chart types to specific PBI visual types
  - Preserve slicer/filter placement
  - Maintain visual stacking order
  - Handle navigation tabs / dashboard actions

### 2e. The user request is buried at the end (dr_data_agent.py:2689-2693)

- `_build_tableau_translate_request` appends the user's request under
  "ADDITIONAL USER REQUEST" only if "power bi" is not already in it.
- If the user says "convert to a 3-page Power BI report with slicers",
  Claude sees the Tableau metadata first (which may describe 1 dashboard)
  and the page-count request last. The system prompt gives no weight to
  the user's structural preferences.

---

## 3. Where Visual Intent Is Under-Modeled

### 3a. No Tableau chart type detection

`parse_twb()` collects `mark.get("class")` into `ws_info["marks"]` but the
Tableau XML actually stores chart type on `<mark class="...">` under the
`<pane>` element, and the top-level worksheet type is often inferred from
the combination of mark type + shelf encodings + number of axes.

The `_TABLEAU_CHART_MAP` (dr_data_agent.py:2483-2507) is correct and
comprehensive, but it never gets used effectively because the input
`chart_type` is almost always "unknown".

### 3b. No visual-to-visual mapping guarantee

Claude (Stage 4) receives a text list of worksheets and is told to
"TRANSLATE this into an equivalent Power BI dashboard". But the
`INTERPRETER_SYSTEM_PROMPT` does not enforce a 1:1 worksheet-to-visual
mapping. Claude may:
- Merge multiple worksheets into one visual
- Omit worksheets it considers redundant
- Add visuals that were not in the original

### 3c. No slicer/filter modeling

Tableau quick filters, parameter controls, and filter actions are not
extracted. The dashboard spec schema has a `slicers` array per page,
but the Tableau translation prompt never populates or references it.

### 3d. Unsupported visual differences are never surfaced

When a Tableau visual has no direct PBI equivalent (e.g., Gantt, box plot,
packed bubbles), the `_TABLEAU_CHART_MAP` silently maps them to
`clusteredBarChart`. There is no mechanism to:
- Warn the user about the lossy mapping
- Suggest a workaround (custom visual, decomposition tree, etc.)

---

## 4. Where Layout Intent Is Under-Modeled

### 4a. No spatial layout extraction

`parse_twb()` does not extract x/y/w/h from dashboard zones. Without
coordinates, the pipeline cannot:
- Preserve vertical stacking vs horizontal tiling
- Maintain relative visual sizes
- Reproduce the original aspect ratio per visual

### 4b. No layout guidance in the OpenAI prompt

`OPENAI_PBIP_SYSTEM_PROMPT` says "Visual positions must not overlap" and
"Canvas size: 1280x720" but provides zero information about the intended
layout. GPT-4 invents positions from scratch.

### 4c. PBIPGenerator accepts whatever GPT-4 generates

`_write_pages()` and `_write_visual_json()` in pbip_generator.py faithfully
write whatever coordinates GPT-4 provides. There is no validation that the
layout matches the original Tableau dashboard proportions.

---

## 5. Where Page-Count Intent Is Ignored

### 5a. No explicit page-count instruction

The translation prompt describes Tableau dashboards and worksheets but
never says "create N pages". The `INTERPRETER_SYSTEM_PROMPT` schema
allows multiple pages, but Claude defaults to 1-2 pages for most inputs.

### 5b. User request for multi-page is weakly positioned

If a user says "I want 5 pages", this appears under "ADDITIONAL USER
REQUEST" at the end of the prompt. Claude's system prompt does not
prioritize this directive.

### 5c. Tableau dashboard count != PBI page count by default

Tableau "dashboards" are layout containers that reference worksheets.
A workbook with 8 worksheets and 2 dashboards might need 2 pages
(matching dashboards) or 8 pages (matching worksheets) depending on
user intent. The pipeline makes no attempt to resolve this ambiguity.

---

## 6. Where Prompt Instructions Are Ambiguous or Too Weak

| Prompt | Issue |
|--------|-------|
| `INTERPRETER_SYSTEM_PROMPT` | No Tableau-specific instructions. Same prompt used for fresh design and migration. |
| `_build_tableau_translate_request` | Says "TRANSLATE this" but provides no structural constraints (page count, visual count, layout). |
| `OPENAI_PBIP_SYSTEM_PROMPT` | No awareness that this is a migration. Says "Map all visuals from the specification" but does not enforce 1:1 mapping. |
| Chart type mapping | Embedded as free text in the prompt, not as structured data that Claude must follow. |
| `ADDITIONAL USER REQUEST` | Placed last, after all metadata, giving it minimal prompt weight. |

---

## 7. Recommended Code Changes (Priority Order)

### P0: Fix Tableau XML parsing (file_handler.py)
- Extract `<mark class>` at pane level for actual chart type
- Extract shelf expressions from `<rows>` and `<cols>` text elements
- Extract worksheet-level filters from `<filter>` elements
- Extract dashboard zone x/y/w/h and type from zone attributes
- Extract zone ordering for stacking intent

### P1: Add visual intent model (new: core/visual_intent.py)
- Structured `VisualIntent` and `PageIntent` dataclasses
- Parse user request for explicit page count, layout preferences, slicer requests
- Map each Tableau worksheet to a `VisualIntent` with chart type, fields, filters
- Map each Tableau dashboard to a `PageIntent` with layout zones
- Surface unsupported visual mappings as warnings

### P2: Fix the translation prompt (dr_data_agent.py)
- Move user request to the TOP of the prompt, not the bottom
- Add explicit page-count instruction derived from Tableau dashboards + user request
- Include zone coordinates so Claude can produce proportional PBI positions
- Add structured visual mapping table (not free text)

### P3: Add a migration-specific system prompt (config/prompts.py)
- New `TABLEAU_MIGRATION_SYSTEM_PROMPT` that enforces:
  - 1:1 worksheet-to-visual mapping (unless user says otherwise)
  - Dashboard-to-page mapping
  - Slicer preservation
  - Layout fidelity
  - Warnings for unsupported visuals

### P4: Pass layout intent through to OpenAI (openai_engine.py)
- Include per-page layout zones in the user message
- Add validation that GPT-4's visual positions roughly match the
  intended layout proportions

### P5: Add slicer generation to PBIPGenerator (generators/pbip_generator.py)
- Support `slicer` visual type with proper queryState
- Map Tableau quick filters to PBI slicers in the PBIP output
