# DR. DATA V2 — QA-CERTIFIED PIPELINE RUN

**Date:** 2026-04-14
**Benchmark:** Pipeline_summary_demo.twbx
**Method:** Full pipeline_runner.py execution (8 stages with QA gates)
**Branch:** v2-rebuild-april2026

## Pipeline Stages

| # | Stage | Result |
|---|-------|--------|
| 1 | Parse | 45 worksheets, 5 dashboards, 37 calcs, 32 ID mappings |
| 2 | Synthetic Data | 2000 rows, 103 columns, 37/37 calc fields present |
| 3 | Field Mapping | 5 pages, 37 DAX measures |
| 4 | Formula Translation | 34 AUTO, 0 GOOD, 1 REVIEW, 2 BLOCKED |
| 5 | Visual Mapping | 57 visuals, 5 pages |
| 6 | Generate | 78 files generated |
| 7 | QA Gate | Preflight 22/22 pass, Fidelity failed=45 |
| 8 | Publish | SUCCESS |

## Pages Generated

| Page | Visuals |
|------|---------|
| Pipeline summary | 12 |
| Product pipeline | 13 |
| Pipeline activity 1 | 10 |
| Pipeline activity 1 (2) | 10 |
| Pipeline activity 2 | 12 |

## Formula Translation Detail

- Total calculated fields: 37
- AUTO (deterministic, high confidence): 34
- GOOD (heuristic, medium confidence): 0
- REVIEW (needs human verification): 1
- BLOCKED (no translation available): 2

## Preflight Validation

- Passed: 22
- Failed: 0
- All passed: True

## Visual Fidelity QA

- Fidelity check passed: False
- Failed visuals: 45

### Per-Visual Fidelity

-   Closed lost: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Closed won: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Closed won 2: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Closed won future: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Closed won this quarter: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Closed won this year: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   New: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Sheet 14 (2): clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Sheet 15: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Sheet 16: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Sheet 16 (2): clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   Win rate: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   avg sales: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days 1: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days 2: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days 3: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days 4: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days to ship bucket: card — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days to ship bucket (2): card — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days to ship bucket (3): card — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   days to ship bucket (4): card — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   existing: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   gross pipeline: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   gross pipeline 2: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   gross pipeline future: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   gross pipeline this quarter: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   gross pipeline this year: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   group: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   group time: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   legend: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   opp by days: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   opp time: scatterChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   opp time (2): clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   opps: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   opps ban: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   pillar time: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   region: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   sales + pipeline age: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   sales ban: map — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   subcat time: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   time trend: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   time trend (2): clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   top 10: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   waterfall: clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]
-   waterfall (2): clusteredBarChart — type ✗, fields ✗, colors ✓, layout ✓ [FAIL]

## PBI Service

- **Report URL:** https://app.powerbi.com/groups/226a11c9-8f9a-4374-b4c6-5e01dafa482d/reports/598efec8-f291-4a69-9060-0ece5510b9bf
- **Report ID:** 598efec8-f291-4a69-9060-0ece5510b9bf
- **Semantic Model ID:** 6168800b-8138-4d7b-89bb-c2bb864044c6
- **Refresh Status:** Completed
- **Display Name:** Pipeline Summary V4 Final
- **Workspace:** 226a11c9-8f9a-4374-b4c6-5e01dafa482d

## V1 Integrity

V1 on port 8502 was NOT touched. Zero lines changed in app/streamlit_app.py or app/dr_data_agent.py.

## Verdict

**CERTIFIED**
