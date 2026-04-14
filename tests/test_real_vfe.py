import os
import sys
import tempfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.real_vfe import compare_images


def _make_chart(path: str, color: str):
    img = Image.new("RGB", (320, 180), "#0d0d22")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 120, 70, 160), fill=color)
    draw.rectangle((90, 80, 140, 160), fill=color)
    draw.rectangle((160, 40, 210, 160), fill=color)
    img.save(path)


def test_real_vfe_scores_similar_images_high():
    tmpdir = tempfile.mkdtemp()
    tab = os.path.join(tmpdir, "tableau.png")
    pbi = os.path.join(tmpdir, "pbi.png")

    _make_chart(tab, "#00ff88")
    _make_chart(pbi, "#00ff88")

    result = compare_images(tab, pbi)

    assert result["composite_score"] >= 0.85
    assert result["verdict"] == "PASS"
