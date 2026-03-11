## SPRINT 1-3 DEPLOYMENT -- 2026-03-11

### Libraries Installed
- tableaudocumentapi 0.11
- rapidfuzz 3.14.3
- lark 1.3.1

### Sprint 1 -- Field Resolver
- Status: PASS
- Files created: core/field_resolver.py, core/enhanced_tableau_parser.py
- Files modified: requirements.txt
- TableauFieldResolver class with exact, alias, normalized, and fuzzy matching
- Uses rapidfuzz token_sort_ratio + WRatio scorers with configurable thresholds
- tableaudocumentapi integration with XML fallback
- Bidirectional alias map (caption <-> name)
- All 5 unit tests pass

### Sprint 2 -- Visual Type Mapping
- Status: PASS
- Files created: generators/powerbi_visual_generator.py
- Visual types mapped: 33
- Deneb fallback types: gantt, density, heatmap, boxplot
- Mark type detection from worksheet XML (stacked, percent, heatmap)
- Visual property injection (title, labels)
- Deneb/Vega-Lite spec generation for unmappable chart types

### Sprint 3 -- Formula Transpiler
- Status: PASS
- Files created: core/formula_transpiler.py
- Files modified: generators/pbip_generator.py (import + _transpile_tableau_calcs method)
- Lark earley parser with full Tableau grammar
- DAXEmitter transformer for AST -> DAX conversion
- LOD translation: FIXED -> ALLEXCEPT, EXCLUDE -> ALL, INCLUDE -> KEEPFILTERS
- Special function handling: ZN, IIF, IFNULL, DATEPART, DATETRUNC, ATTR, CONTAINS, STARTSWITH, ENDSWITH
- Table calc detection with warnings (WINDOW_SUM, RUNNING_SUM, etc.)
- Regex fallback for formulas that fail AST parsing
- Confidence scoring (1.0 = full confidence, reduced for table calcs/fallbacks)
- Test formulas passing: 10/10

### Test Results
- Test 1 (Field Resolver Unit): PASS (5/5)
- Test 2 (Visual Map Coverage): PASS (16/16 required types, 33 total)
- Test 3 (Transpiler Unit): PASS (10/10)
- Test 4 (End-to-End Pipeline): PASS (3 calc fields transpiled from synthetic TWB)
- Test 5 (Import Sanity): PASS (all 5 modules)
- Test 6 (Regression): PASS (streamlit_app imports clean)

### Known Gaps / Next Actions
- tableaudocumentapi throws on some TWB versions (NoneType error) -- XML fallback handles this
- Table calculations (WINDOW_SUM, RUNNING_SUM, LOOKUP, etc.) marked for manual review
- REGEXP_MATCH has no DAX equivalent -- emitted as comment
- Deneb templates use placeholder field names -- user must adjust in Power BI Desktop
- DATETRUNC(week) approximation may need locale adjustment
