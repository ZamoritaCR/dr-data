"""
Bulletproof PBIP validation tests.

Generates a real PBIP project from a sample DataFrame and verifies
every error class that has been reported by real users.
"""

import json
import os
import re
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generators.pbip_generator import PBIPGenerator
from generators.pbip_reference import PBIP_TEMPLATE, PBIR_TEMPLATE


# ------------------------------------------------------------------ #
#  Shared fixture: generate a PBIP project once for all tests          #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pbip_project():
    """Generate a PBIP project from a sample DataFrame."""
    gen = PBIPGenerator(tempfile.mkdtemp())
    df = pd.DataFrame({
        "Sales": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Quantity": [1, 2, 3, 4, 5],
        "Order Date": pd.to_datetime([
            "2023-01-15", "2023-02-20", "2023-03-10",
            "2023-04-05", "2023-05-25",
        ]),
        "Region": ["North", "South", "East", "West", "Central"],
        "Category": ["Tech", "Office", "Furniture", "Tech", "Office"],
    })

    from core.deep_analyzer import DeepAnalyzer
    analyzer = DeepAnalyzer()
    profile = analyzer.profile(df)
    profile["table_name"] = "Data"

    config = {
        "report_layout": {"sections": []},
        "tmdl_model": {
            "tables": [{
                "name": "Data",
                "measures": [
                    {
                        "name": "Total Sales",
                        "dax": "SUM(Data[Sales])",
                        "format": "#,0",
                    },
                    {
                        "name": "Avg Quantity",
                        "dax": "AVERAGE(Data[Quantity])",
                        "format": "#,0.00",
                    },
                ],
            }],
        },
    }
    dash = {"dashboard_title": "Test Dashboard", "pages": []}
    result = gen.generate(
        config, profile, dash, data_file_path="/tmp/test_data.csv"
    )
    return result["path"]


def _read_file(project_path, *parts):
    """Read a file from the project directory."""
    for root, dirs, files in os.walk(project_path):
        for f in files:
            if all(p in os.path.join(root, f) for p in parts):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                    return fh.read()
    return None


def _read_json(project_path, *parts):
    """Read and parse a JSON file from the project."""
    text = _read_file(project_path, *parts)
    if text:
        return json.loads(text)
    return None


# ------------------------------------------------------------------ #
#  ERROR CLASS 1: PBIP Manifest                                        #
# ------------------------------------------------------------------ #

class TestPbipManifest:
    def test_version_is_1_0(self):
        """Version must be '1.0', not '1.0.0'."""
        assert PBIP_TEMPLATE["version"] == "1.0"

    def test_artifacts_report_only(self):
        """Artifacts must have exactly one entry: the report."""
        arts = PBIP_TEMPLATE["artifacts"]
        assert len(arts) == 1
        assert "report" in arts[0]

    def test_no_semantic_model_in_artifacts(self):
        """semanticModel must NOT appear in artifacts."""
        for art in PBIP_TEMPLATE["artifacts"]:
            assert "semanticModel" not in art
            assert "dataset" not in art


# ------------------------------------------------------------------ #
#  ERROR CLASS 2: definition.pbir                                      #
# ------------------------------------------------------------------ #

class TestDefinitionPbir:
    def test_version_is_1_0(self):
        """PBIR version must be '1.0'."""
        assert PBIR_TEMPLATE["version"] == "1.0"

    def test_schema_present(self):
        """$schema field must be present."""
        assert "$schema" in PBIR_TEMPLATE

    def test_schema_version_1_0_0(self):
        """Schema URL must reference 1.0.0."""
        assert "1.0.0" in PBIR_TEMPLATE["$schema"]

    def test_by_path_present(self):
        """byPath must be present in datasetReference."""
        ds_ref = PBIR_TEMPLATE["datasetReference"]
        assert "byPath" in ds_ref

    def test_by_connection_null(self):
        """byConnection must be present and null."""
        ds_ref = PBIR_TEMPLATE["datasetReference"]
        assert "byConnection" in ds_ref
        assert ds_ref["byConnection"] is None

    def test_generated_pbir(self, pbip_project):
        """Generated definition.pbir must have correct version."""
        content = _read_json(pbip_project, "definition.pbir")
        assert content is not None
        assert content["version"] == "1.0"
        assert content["datasetReference"]["byConnection"] is None


# ------------------------------------------------------------------ #
#  ERROR CLASS 3: TMDL Indentation                                     #
# ------------------------------------------------------------------ #

class TestTmdlIndentation:
    def test_no_space_indentation(self, pbip_project):
        """TMDL files must use tabs, not spaces, for indentation."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        lines = content.split("\n")
        space_indented = [l for l in lines if l.startswith("    ")]
        assert len(space_indented) == 0, (
            f"Found {len(space_indented)} space-indented lines in table TMDL"
        )

    def test_has_tab_indentation(self, pbip_project):
        """Table TMDL must have tab-indented lines."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        tab_lines = [l for l in content.split("\n") if l.startswith("\t")]
        assert len(tab_lines) > 0


# ------------------------------------------------------------------ #
#  ERROR CLASS 4: No Int64.Type                                        #
# ------------------------------------------------------------------ #

class TestNoInt64Type:
    def test_no_int64_type_in_source(self):
        """_build_type_transforms must not use Int64.Type."""
        import inspect
        src = inspect.getsource(PBIPGenerator._build_type_transforms)
        assert "Int64.Type" not in src

    def test_no_int64_type_in_output(self, pbip_project):
        """Generated TMDL must not contain Int64.Type."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        assert "Int64.Type" not in content

    def test_no_type_integer_in_output(self, pbip_project):
        """Generated TMDL must not contain 'type integer'."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        assert "type integer" not in content


# ------------------------------------------------------------------ #
#  ERROR CLASS 5: model.tmdl no table duplication                      #
# ------------------------------------------------------------------ #

class TestModelTmdlClean:
    def test_no_table_block(self, pbip_project):
        """model.tmdl must not contain inline table definitions."""
        content = _read_file(pbip_project, "model.tmdl")
        assert content is not None
        # Should use 'ref table' not 'table Name\n\tcolumn ...'
        assert not re.search(r"^table ", content, re.MULTILINE), (
            "model.tmdl contains an inline table block"
        )

    def test_no_partition(self, pbip_project):
        """model.tmdl must not contain partition definitions."""
        content = _read_file(pbip_project, "model.tmdl")
        assert content is not None
        assert "partition" not in content

    def test_has_ref_table(self, pbip_project):
        """model.tmdl must reference tables via 'ref table'."""
        content = _read_file(pbip_project, "model.tmdl")
        assert content is not None
        assert "ref table" in content


# ------------------------------------------------------------------ #
#  ERROR CLASS 6: Table name consistency                               #
# ------------------------------------------------------------------ #

class TestTableNameConsistency:
    def test_table_named_data(self, pbip_project):
        """All TMDL table definitions must use 'Data'."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        names = re.findall(r"table '?([^'\n]+)'?", content)
        for name in names:
            assert name == "Data", f"Table named '{name}' instead of 'Data'"

    def test_no_synthetic_tableau_data(self, pbip_project):
        """No file should contain 'synthetic_tableau_data'."""
        for root, dirs, files in os.walk(pbip_project):
            for f in files:
                content = open(os.path.join(root, f), "r",
                               errors="replace").read()
                assert "synthetic_tableau_data" not in content, (
                    f"'synthetic_tableau_data' found in {f}"
                )


# ------------------------------------------------------------------ #
#  ERROR CLASS 7: No server paths                                      #
# ------------------------------------------------------------------ #

class TestNoServerPaths:
    def test_no_linux_path(self, pbip_project):
        """No TMDL file should contain Linux server paths."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        assert "/home/" not in content
        assert "/tmp/" not in content

    def test_no_twbx_reference(self, pbip_project):
        """No TMDL file should reference .twbx archives."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        assert ".twbx" not in content.lower()


# ------------------------------------------------------------------ #
#  ERROR CLASS 8: Column consistency                                   #
# ------------------------------------------------------------------ #

class TestColumnConsistency:
    def test_columns_in_tmdl(self, pbip_project):
        """TMDL must define columns matching the input DataFrame."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        for col in ["Sales", "Quantity", "Order Date", "Region", "Category"]:
            assert col in content, f"Column '{col}' not found in TMDL"

    def test_measures_in_tmdl(self, pbip_project):
        """TMDL must contain the defined measures."""
        content = _read_file(pbip_project, "tables", ".tmdl")
        assert content is not None
        assert "Total Sales" in content
        assert "Avg Quantity" in content
