"""
End-to-end regression test against a real TWBX file.

Runs the full pipeline: parse -> data extraction -> direct mapper ->
PBIP generation -> preflight validation -> ZIP structure.
Asserts zero failures at every stage.

Uses DigitalAds-Sales-Data.twbx (has embedded CSV data, 5 worksheets,
1 dashboard, 1 calculated field -- exercises the real data path).
"""

import json
import os
import re
import sys
import tempfile
import zipfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Path to real TWBX test file
_TWBX_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "test_data", "DigitalAds-Sales-Data.twbx"),
    "/home/zamoritacr/twbx-work/DigitalAds-Sales-Data.twbx",
]
TWBX_PATH = None
for p in _TWBX_CANDIDATES:
    if os.path.isfile(p):
        TWBX_PATH = p
        break

pytestmark = pytest.mark.skipif(
    TWBX_PATH is None,
    reason="DigitalAds-Sales-Data.twbx not found in test_data/ or twbx-work/",
)


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def tableau_spec():
    from core.enhanced_tableau_parser import parse_twb
    return parse_twb(TWBX_PATH)


@pytest.fixture(scope="module")
def embedded_data():
    from app.file_handler import _extract_twbx_data
    _, dfs = _extract_twbx_data(TWBX_PATH)
    if dfs:
        return list(dfs.values())[0]
    return None


@pytest.fixture(scope="module")
def dataframe(embedded_data, tableau_spec):
    if embedded_data is not None:
        return embedded_data
    from core.synthetic_data import generate_from_tableau_spec
    output_dir = tempfile.mkdtemp()
    df, _, _ = generate_from_tableau_spec(
        tableau_spec, num_rows=100, output_dir=output_dir
    )
    return df


@pytest.fixture(scope="module")
def data_profile(dataframe):
    from core.deep_analyzer import DeepAnalyzer
    analyzer = DeepAnalyzer()
    profile = analyzer.profile(dataframe)
    profile["table_name"] = "Data"
    return profile


@pytest.fixture(scope="module")
def mapper_output(tableau_spec, data_profile):
    from core.direct_mapper import build_pbip_config_from_tableau
    config, dashboard_spec = build_pbip_config_from_tableau(
        tableau_spec, data_profile, "Data"
    )
    return config, dashboard_spec


@pytest.fixture(scope="module")
def pbip_project(mapper_output, data_profile, dataframe):
    config, dashboard_spec = mapper_output
    from generators.pbip_generator import PBIPGenerator
    gen = PBIPGenerator(tempfile.mkdtemp())
    result = gen.generate(config, data_profile, dashboard_spec, dataframe=dataframe)
    return result["path"]


# ------------------------------------------------------------------ #
#  Stage 1: Parser                                                     #
# ------------------------------------------------------------------ #

class TestStage1Parse:
    def test_has_worksheets(self, tableau_spec):
        assert len(tableau_spec.get("worksheets", [])) > 0

    def test_has_dashboards(self, tableau_spec):
        assert len(tableau_spec.get("dashboards", [])) > 0

    def test_worksheets_have_names(self, tableau_spec):
        for ws in tableau_spec["worksheets"]:
            assert ws.get("name"), "Worksheet missing name"

    def test_dashboards_reference_worksheets(self, tableau_spec):
        for db in tableau_spec["dashboards"]:
            assert db.get("worksheets_used"), f"Dashboard '{db.get('name')}' has no worksheets"


# ------------------------------------------------------------------ #
#  Stage 2: Data extraction                                            #
# ------------------------------------------------------------------ #

class TestStage2Data:
    def test_dataframe_not_empty(self, dataframe):
        assert len(dataframe) > 0
        assert len(dataframe.columns) > 0


# ------------------------------------------------------------------ #
#  Stage 3: Direct mapper                                              #
# ------------------------------------------------------------------ #

class TestStage3Mapper:
    def test_has_pages(self, mapper_output):
        config, dashboard_spec = mapper_output
        sections = config.get("report_layout", {}).get("sections", [])
        assert len(sections) > 0

    def test_has_visuals(self, mapper_output):
        config, _ = mapper_output
        sections = config.get("report_layout", {}).get("sections", [])
        total_vc = sum(len(s.get("visualContainers", [])) for s in sections)
        assert total_vc > 0


# ------------------------------------------------------------------ #
#  Stage 4: PBIP generation                                            #
# ------------------------------------------------------------------ #

class TestStage4Generate:
    def test_project_exists(self, pbip_project):
        assert os.path.isdir(pbip_project)

    def test_has_pbip_file(self, pbip_project):
        found = False
        for root, dirs, files in os.walk(pbip_project):
            for f in files:
                if f.endswith(".pbip"):
                    found = True
        assert found, "No .pbip file"

    def test_has_tmdl(self, pbip_project):
        found = False
        for root, dirs, files in os.walk(pbip_project):
            for f in files:
                if f.endswith(".tmdl") and "tables" in root:
                    found = True
        assert found, "No table .tmdl file"


# ------------------------------------------------------------------ #
#  Stage 5: Preflight validation (18 checks)                           #
# ------------------------------------------------------------------ #

class TestStage5Preflight:
    def test_preflight_all_pass(self, pbip_project):
        from core.preflight_validator import validate
        report = validate(pbip_project)
        failed = report.failed
        assert len(failed) == 0, (
            f"Preflight failures: {[str(f) for f in failed]}"
        )

    def test_preflight_count(self, pbip_project):
        from core.preflight_validator import validate
        report = validate(pbip_project)
        assert report.pass_count >= 15, (
            f"Expected 15+ checks, got {report.pass_count}"
        )


# ------------------------------------------------------------------ #
#  Stage 6: ZIP structure                                              #
# ------------------------------------------------------------------ #

class TestStage6Zip:
    @pytest.fixture(scope="class")
    def zip_names(self, pbip_project):
        import shutil
        zip_path = os.path.join(tempfile.mkdtemp(), "test_output.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", pbip_project)
        with zipfile.ZipFile(zip_path, "r") as z:
            return z.namelist()

    def test_has_pbip(self, zip_names):
        assert any(n.endswith(".pbip") for n in zip_names)

    def test_has_pbir(self, zip_names):
        assert any("definition.pbir" in n for n in zip_names)

    def test_has_pbism(self, zip_names):
        assert any("definition.pbism" in n for n in zip_names)

    def test_has_model_tmdl(self, zip_names):
        assert any("model.tmdl" in n for n in zip_names)

    def test_has_table_tmdl(self, zip_names):
        assert any(".tmdl" in n and "tables" in n for n in zip_names)

    def test_has_visual_json(self, zip_names):
        assert any("visual.json" in n for n in zip_names)
