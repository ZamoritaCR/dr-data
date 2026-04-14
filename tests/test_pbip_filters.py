import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generators.pbip_generator import PBIPGenerator


def test_visual_filters_written_to_filter_config():
    outdir = tempfile.mkdtemp()
    viz_dir = Path(outdir) / "visual"
    os.makedirs(viz_dir, exist_ok=True)

    gen = PBIPGenerator(outdir)
    vc = {
        "x": 0,
        "y": 0,
        "width": 320,
        "height": 200,
        "config": {
            "visualType": "clusteredColumnChart",
            "dataRoles": {
                "category": ["Region"],
                "values": ["Sales"],
            },
            "filter_values": ["East", "West"],
        },
    }

    gen._write_visual_json(
        viz_dir=viz_dir,
        vc=vc,
        vidx=0,
        viz_id="viz123",
        table_name="Data",
        measure_names=set(),
        col_types={"Region": "string", "Sales": "number"},
        profile_col_names={"Region", "Sales"},
    )

    visual = json.load(open(viz_dir / "visual.json", "r", encoding="utf-8"))
    assert "filterConfig" in visual
    assert "filterState" not in visual["visual"]["query"]

    filters = visual["filterConfig"]["filters"]
    assert len(filters) == 1
    assert filters[0]["type"] == "Categorical"
    assert filters[0]["field"]["Column"]["Property"] == "Region"
