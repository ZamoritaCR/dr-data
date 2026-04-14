import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.evidence_logger import EvidenceLogger


def test_evidence_logger_summary_counts():
    logger = EvidenceLogger("job123", "demo.twbx")
    logger.log("parse", "start", input_data={"file": "demo.twbx"})
    logger.log("publish", "error", output_data={"error": "403"}, status="error")

    summary = logger.get_summary()
    assert summary["phases_completed"] == 2
    assert summary["actions_logged"] == 2
    assert summary["errors"] == 1


def test_evidence_logger_exports_html():
    logger = EvidenceLogger("job123", "demo.twbx")
    logger.log("parse", "complete", output_data={"worksheets": ["Sheet 1", "Sheet 2"]})
    logger.log("qa", "complete", output_data={"score": 88})

    outdir = tempfile.mkdtemp()
    path = logger.export_html(os.path.join(outdir, "evidence_report.html"))

    assert path.endswith("evidence_report.html")
    assert os.path.isfile(path)
    html = open(path, "r", encoding="utf-8").read()
    assert "Dr. Data V2 Evidence Report" in html
    assert "Parse Evidence" in html
    assert "QA Evidence" in html
