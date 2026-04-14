import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core import powerbi_publisher as publisher


def test_publish_pbip_returns_render_proof(monkeypatch):
    project_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(project_dir, "Demo.SemanticModel"))
    os.makedirs(os.path.join(project_dir, "Demo.Report"))

    calls = {"count": 0}

    def fake_publish_item(token, workspace_id, display_name, item_type, item_dir, file_overrides=None):
        calls["count"] += 1
        if item_type == "SemanticModel":
            return {"id": "sm-1"}
        return {"id": "rp-1"}

    monkeypatch.setattr(publisher, "_publish_item", fake_publish_item)
    monkeypatch.setattr(publisher, "_refresh_dataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        publisher,
        "_capture_render_proof",
        lambda **kwargs: {
            "render_proof": "/tmp/pbi_render_page1.png",
            "render_proof_method": "export_api",
            "render_proof_status": "succeeded",
            "render_proof_error": "",
        },
    )

    result = publisher.publish_pbip(
        token="fabric-token",
        workspace_id="ws-1",
        pbip_folder_path=project_dir,
        display_name="Demo Report",
        proof_output_dir="/tmp/demo-proof",
    )

    assert calls["count"] == 2
    assert result["report_id"] == "rp-1"
    assert result["render_proof"] == "/tmp/pbi_render_page1.png"
    assert result["render_proof_status"] == "succeeded"


def test_capture_render_proof_falls_back_to_playwright(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    expected = os.path.join(tmpdir, "pbi_render_page1.png")

    monkeypatch.setattr(publisher, "get_access_token", lambda scope=None: "pbi-token")
    monkeypatch.setattr(
        publisher,
        "_export_first_page_png",
        lambda **kwargs: {"ok": False, "status_code": 403, "error": "forbidden"},
    )
    monkeypatch.setattr(
        publisher,
        "_capture_playwright_embed_screenshot",
        lambda **kwargs: {"ok": True, "path": expected},
    )

    result = publisher._capture_render_proof(
        workspace_id="ws-1",
        report_id="rp-1",
        output_dir=tmpdir,
        embed_url="https://app.powerbi.com/reportEmbed?reportId=rp-1&groupId=ws-1",
    )

    assert result["render_proof"] == expected
    assert result["render_proof_method"] == "playwright"
    assert result["render_proof_status"] == "succeeded"
