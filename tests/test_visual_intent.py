"""
Tests for core/visual_intent.py -- migration intent extraction, visual mapping,
page-count parsing, slicer detection, and prompt formatting.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from core.visual_intent import (
    extract_migration_intent,
    format_intent_for_prompt,
    _parse_page_count_from_request,
    TABLEAU_TO_PBI_MAP,
    VisualIntent,
    PageIntent,
    MigrationIntent,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

def _make_tableau_spec(
    worksheets=None, dashboards=None, calculated_fields=None, parameters=None,
):
    """Build a minimal tableau_spec dict for testing."""
    return {
        "type": "tableau_workbook",
        "version": "2024.1",
        "worksheets": worksheets or [],
        "dashboards": dashboards or [],
        "datasources": [],
        "calculated_fields": calculated_fields or [],
        "parameters": parameters or [],
        "relationships": [],
    }


def _make_worksheet(name, chart_type="bar", rows_fields=None, cols_fields=None,
                    filters=None):
    return {
        "name": name,
        "chart_type": chart_type,
        "marks": [chart_type],
        "rows_fields": rows_fields or [],
        "cols_fields": cols_fields or [],
        "dimensions": [],
        "measures": [],
        "filters": filters or [],
    }


def _make_dashboard(name, zone_names=None, width=1000, height=800):
    zones = []
    if zone_names:
        y_offset = 0
        for zn in zone_names:
            zones.append({
                "name": zn,
                "type": "worksheet",
                "x": "0", "y": str(y_offset), "w": str(width), "h": "200",
            })
            y_offset += 200
    return {
        "name": name,
        "size": {"width": str(width), "height": str(height)},
        "zones": zones,
    }


# ------------------------------------------------------------------ #
#  Page count parsing                                                  #
# ------------------------------------------------------------------ #

class TestPageCountParsing:

    def test_numeric_pages(self):
        assert _parse_page_count_from_request("I want 3 pages") == 3

    def test_word_pages(self):
        assert _parse_page_count_from_request("create a two-page report") == 2

    def test_single_page(self):
        assert _parse_page_count_from_request("a single page dashboard") == 1

    def test_no_mention(self):
        assert _parse_page_count_from_request("convert my tableau workbook") is None

    def test_empty(self):
        assert _parse_page_count_from_request("") is None

    def test_five_pages(self):
        assert _parse_page_count_from_request("make it five pages") == 5


# ------------------------------------------------------------------ #
#  Visual type mapping                                                 #
# ------------------------------------------------------------------ #

class TestVisualMapping:

    def test_all_tableau_types_have_mapping(self):
        """Every key in the map produces a pbi_type and fidelity."""
        for key, info in TABLEAU_TO_PBI_MAP.items():
            assert "pbi_type" in info, f"Missing pbi_type for {key}"
            assert "fidelity" in info, f"Missing fidelity for {key}"

    def test_exact_mappings(self):
        exact = [k for k, v in TABLEAU_TO_PBI_MAP.items() if v["fidelity"] == "exact"]
        assert "bar" in exact
        assert "line" in exact
        assert "waterfall" in exact

    def test_lossy_mappings_have_notes(self):
        for key, info in TABLEAU_TO_PBI_MAP.items():
            if info["fidelity"] == "lossy":
                assert "note" in info and info["note"], (
                    f"Lossy mapping '{key}' must have a note explaining the limitation"
                )


# ------------------------------------------------------------------ #
#  Migration intent extraction                                         #
# ------------------------------------------------------------------ #

class TestMigrationIntent:

    def test_single_dashboard_single_worksheet(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("Sales Chart", "bar")],
            dashboards=[_make_dashboard("Overview", zone_names=["Sales Chart"])],
        )
        intent = extract_migration_intent(spec, "convert to power bi")
        assert len(intent.pages) == 1
        assert intent.pages[0].display_name == "Overview"
        assert len(intent.pages[0].visuals) == 1
        assert intent.pages[0].visuals[0].chart_type_pbi == "clusteredBarChart"

    def test_multi_dashboard_creates_multi_page(self):
        spec = _make_tableau_spec(
            worksheets=[
                _make_worksheet("Sales", "bar"),
                _make_worksheet("Trends", "line"),
            ],
            dashboards=[
                _make_dashboard("Page1", zone_names=["Sales"]),
                _make_dashboard("Page2", zone_names=["Trends"]),
            ],
        )
        intent = extract_migration_intent(spec, "convert to power bi")
        assert len(intent.pages) == 2
        assert intent.pages[0].visuals[0].chart_type_pbi == "clusteredBarChart"
        assert intent.pages[1].visuals[0].chart_type_pbi == "lineChart"

    def test_orphan_worksheets_become_pages(self):
        """Worksheets not on any dashboard should become their own pages."""
        spec = _make_tableau_spec(
            worksheets=[
                _make_worksheet("Sheet1", "bar"),
                _make_worksheet("Sheet2", "line"),
            ],
            dashboards=[],  # No dashboards
        )
        intent = extract_migration_intent(spec, "convert")
        assert len(intent.pages) == 2
        assert len(intent.orphan_worksheets) == 0

    def test_user_page_count_override_warning(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("A", "bar")],
            dashboards=[_make_dashboard("D1", zone_names=["A"])],
        )
        intent = extract_migration_intent(spec, "I want 3 pages")
        assert intent.requested_page_count == 3
        assert any("3 pages" in w for w in intent.warnings)

    def test_lossy_mapping_generates_warning(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("Gantt View", "gantt")],
            dashboards=[_make_dashboard("D1", zone_names=["Gantt View"])],
        )
        intent = extract_migration_intent(spec, "convert")
        assert any("Gantt" in w for w in intent.warnings)

    def test_filters_become_slicers(self):
        ws = _make_worksheet("Sales", "bar", filters=[
            {"field": "Region", "column_ref": "[Region]", "type": "categorical"},
        ])
        spec = _make_tableau_spec(
            worksheets=[ws],
            dashboards=[_make_dashboard("D1", zone_names=["Sales"])],
        )
        intent = extract_migration_intent(spec, "convert")
        assert len(intent.pages[0].slicers) == 1
        assert intent.pages[0].slicers[0]["field"] == "Region"

    def test_zone_coordinates_preserved(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("A", "bar")],
            dashboards=[_make_dashboard("D1", zone_names=["A"], width=1000, height=800)],
        )
        intent = extract_migration_intent(spec, "convert")
        zones = intent.pages[0].zones
        assert len(zones) >= 1
        assert zones[0].x == 0
        assert zones[0].w == 1000

    def test_calculated_fields_passed_through(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("A", "bar")],
            dashboards=[_make_dashboard("D1", zone_names=["A"])],
            calculated_fields=[
                {"name": "Profit Ratio", "formula": "SUM([Profit])/SUM([Sales])"},
            ],
        )
        intent = extract_migration_intent(spec, "convert")
        assert len(intent.calculated_fields) == 1
        assert intent.calculated_fields[0]["name"] == "Profit Ratio"


# ------------------------------------------------------------------ #
#  Prompt formatting                                                   #
# ------------------------------------------------------------------ #

class TestPromptFormatting:

    def test_user_request_at_top(self):
        intent = MigrationIntent(user_request="build me a 3-page report")
        prompt = format_intent_for_prompt(intent, "Sales")
        lines = prompt.strip().split("\n")
        assert lines[0] == "USER REQUEST:"
        assert "3-page report" in lines[1]

    def test_page_count_instruction_present(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("A", "bar")],
            dashboards=[_make_dashboard("D1", zone_names=["A"])],
        )
        intent = extract_migration_intent(spec, "I want 5 pages")
        prompt = format_intent_for_prompt(intent, "Data")
        assert "exactly 5 pages" in prompt

    def test_visual_types_in_prompt(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("Sales", "line")],
            dashboards=[_make_dashboard("D1", zone_names=["Sales"])],
        )
        intent = extract_migration_intent(spec, "convert")
        prompt = format_intent_for_prompt(intent, "Data")
        assert "lineChart" in prompt
        assert "Tableau line" in prompt

    def test_warnings_in_prompt(self):
        spec = _make_tableau_spec(
            worksheets=[_make_worksheet("G", "gantt")],
            dashboards=[_make_dashboard("D1", zone_names=["G"])],
        )
        intent = extract_migration_intent(spec, "convert")
        prompt = format_intent_for_prompt(intent, "Data")
        assert "MIGRATION WARNINGS" in prompt
        assert "Gantt" in prompt

    def test_slicers_in_prompt(self):
        ws = _make_worksheet("S", "bar", filters=[
            {"field": "Category", "column_ref": "[Cat]", "type": "categorical"},
        ])
        spec = _make_tableau_spec(
            worksheets=[ws],
            dashboards=[_make_dashboard("D", zone_names=["S"])],
        )
        intent = extract_migration_intent(spec, "convert")
        prompt = format_intent_for_prompt(intent, "Data")
        assert "SLICERS" in prompt
        assert "Category" in prompt


# ------------------------------------------------------------------ #
#  TWB parsing integration                                             #
# ------------------------------------------------------------------ #

class TestTwbParsing:

    def test_parse_twb_extracts_chart_type(self):
        """parse_twb should now populate chart_type from mark class."""
        import tempfile
        import os
        from app.file_handler import parse_twb

        twb_xml = """<?xml version='1.0'?>
        <workbook version='2024.1'>
          <worksheets>
            <worksheet name='Sales Over Time'>
              <table>
                <pane>
                  <mark class='Line'/>
                </pane>
                <rows>[Category]</rows>
                <cols>SUM([Sales])</cols>
              </table>
            </worksheet>
          </worksheets>
          <dashboards>
            <dashboard name='Overview'>
              <size maxwidth='1200' maxheight='900'/>
              <zones>
                <zone name='Sales Over Time' type-v2='worksheet'
                      x='10' y='20' w='580' h='400'/>
              </zones>
            </dashboard>
          </dashboards>
        </workbook>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".twb", delete=False) as f:
            f.write(twb_xml)
            path = f.name

        try:
            structure = parse_twb(path)
            assert structure["type"] == "tableau_workbook"

            # Worksheet chart type
            ws = structure["worksheets"][0]
            assert ws["chart_type"] == "line"
            assert ws["name"] == "Sales Over Time"
            assert "Category" in ws["rows_fields"]
            assert "Sales" in ws["cols_fields"]

            # Dashboard zones with coordinates
            db = structure["dashboards"][0]
            assert db["name"] == "Overview"
            assert db["size"]["width"] == "1200"
            zone = db["zones"][0]
            assert zone["x"] == "10"
            assert zone["w"] == "580"
            assert zone["type"] == "worksheet"
        finally:
            os.unlink(path)

    def test_parse_twb_extracts_filters(self):
        import tempfile
        import os
        from app.file_handler import parse_twb

        twb_xml = """<?xml version='1.0'?>
        <workbook version='2024.1'>
          <worksheets>
            <worksheet name='Filtered'>
              <table>
                <pane><mark class='Bar'/></pane>
              </table>
              <filter class='categorical' column='[Region]'/>
              <filter class='quantitative' column='[Sales]'/>
            </worksheet>
          </worksheets>
        </workbook>"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".twb", delete=False) as f:
            f.write(twb_xml)
            path = f.name

        try:
            structure = parse_twb(path)
            ws = structure["worksheets"][0]
            assert len(ws["filters"]) == 2
            assert ws["filters"][0]["field"] == "Region"
            assert ws["filters"][0]["type"] == "categorical"
            assert ws["filters"][1]["field"] == "Sales"
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  Run with pytest or standalone                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
