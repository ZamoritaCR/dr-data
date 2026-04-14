"""
Vision QA Module — Dr. Data v2
Compares Tableau dashboard thumbnails with Power BI PBIP previews using GPT-4o vision.
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Schema for structured GPT-4o output
# ---------------------------------------------------------------------------
COMPARISON_SCHEMA = {
    "name": "vision_comparison",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "fidelity_score": {"type": "number"},
            "matched_elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "required": ["element", "status"],
                    "additionalProperties": False,
                },
            },
            "deltas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element": {"type": "string"},
                        "tableau": {"type": "string"},
                        "powerbi": {"type": "string"},
                        "fixable": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["element", "tableau", "powerbi", "fixable", "reason"],
                    "additionalProperties": False,
                },
            },
            "platform_limitations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "summary": {"type": "string"},
        },
        "required": [
            "fidelity_score",
            "matched_elements",
            "deltas",
            "platform_limitations",
            "summary",
        ],
        "additionalProperties": False,
    },
}

COMPARISON_PROMPT = (
    "You are a BI migration QA specialist. Compare these two dashboard images: "
    "the first is the original Tableau dashboard, the second is the Power BI recreation. "
    "Score the fidelity from 0-100. List every element that matched and every delta. "
    "For each delta, state whether it is fixable or a Power BI platform limitation."
)


class VisionQA:
    """Vision-based fidelity QA between Tableau and Power BI dashboards."""

    MAX_FIX_ITERATIONS = 3

    # ------------------------------------------------------------------
    # 1. Extract Tableau thumbnail
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tableau_thumbnail(twbx_path: str | Path) -> bytes | None:
        """
        Extract the thumbnail image from a TWBX archive.

        TWBX files are ZIP archives. Tableau may embed a thumbnail at
        ``thumbnail.png``, ``Thumbnails/*.png``, or similar paths.
        If no pre-rendered thumbnail is found, a synthetic one is generated
        from the embedded worksheet/dashboard metadata and any image assets
        found inside the archive.

        Returns raw PNG bytes, or None if extraction completely fails.
        """
        twbx_path = Path(twbx_path)
        if not twbx_path.exists():
            raise FileNotFoundError(f"TWBX not found: {twbx_path}")

        try:
            with zipfile.ZipFile(twbx_path, "r") as zf:
                names = zf.namelist()

                # — Try standard thumbnail locations first —
                for candidate in names:
                    low = candidate.lower()
                    if ("thumbnail" in low or "thumbnails" in low) and low.endswith(".png"):
                        return zf.read(candidate)

                # — No embedded thumbnail: synthesise from TWB metadata —
                twb_entries = [n for n in names if n.lower().endswith(".twb")]
                if not twb_entries:
                    return None

                twb_xml = zf.read(twb_entries[0]).decode("utf-8", errors="replace")
                root = ET.fromstring(twb_xml)

                worksheets = [w.get("name", "") for w in root.findall(".//worksheet")]
                dashboards = [d.get("name", "") for d in root.findall(".//dashboard")]

                # Collect any embedded PNG images to composite
                image_bytes_list: list[bytes] = []
                for name in names:
                    if name.lower().endswith(".png") and not name.lower().startswith("__"):
                        try:
                            image_bytes_list.append(zf.read(name))
                        except Exception:
                            pass

                return VisionQA._synthesize_thumbnail(
                    worksheets, dashboards, image_bytes_list
                )

        except zipfile.BadZipFile as exc:
            raise ValueError(f"Not a valid ZIP/TWBX archive: {twbx_path}") from exc

    @staticmethod
    def _synthesize_thumbnail(
        worksheets: list[str],
        dashboards: list[str],
        embedded_images: list[bytes],
    ) -> bytes:
        """
        Generate a synthetic thumbnail PNG from dashboard metadata.
        Uses matplotlib for rendering; falls back to a plain PIL image if unavailable.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return VisionQA._pil_thumbnail(worksheets, dashboards)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        title = dashboards[0] if dashboards else (worksheets[0] if worksheets else "Tableau Dashboard")
        ax.text(
            5, 5.5, title,
            ha="center", va="center", fontsize=14, color="white", fontweight="bold"
        )

        # Draw a simple grid of placeholder visuals
        all_sheets = (dashboards + worksheets)[:9]
        cols, rows = 3, 3
        w, h = 2.8, 1.4
        colors = ["#e94560", "#0f3460", "#16213e", "#533483", "#2b2d42", "#8d99ae"]

        for idx, sheet in enumerate(all_sheets):
            col = idx % cols
            row = idx // cols
            x = 0.3 + col * (w + 0.2)
            y = 3.8 - row * (h + 0.2)
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.05",
                facecolor=colors[idx % len(colors)],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
            ax.text(
                x + w / 2, y + h / 2, sheet[:20],
                ha="center", va="center", fontsize=6.5, color="white",
            )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _pil_thumbnail(worksheets: list[str], dashboards: list[str]) -> bytes:
        """Minimal fallback thumbnail using only PIL."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return b""

        img = Image.new("RGB", (800, 500), color=(26, 26, 46))
        draw = ImageDraw.Draw(img)

        title = dashboards[0] if dashboards else (worksheets[0] if worksheets else "Tableau")
        draw.text((400, 30), title, fill=(255, 255, 255), anchor="mm")

        colors = [(233, 69, 96), (15, 52, 96), (22, 33, 62)]
        all_sheets = (dashboards + worksheets)[:6]
        for idx, sheet in enumerate(all_sheets):
            col, row = idx % 3, idx // 3
            x0 = 20 + col * 265
            y0 = 80 + row * 200
            draw.rectangle([x0, y0, x0 + 240, y0 + 170], fill=colors[idx % len(colors)])
            draw.text((x0 + 120, y0 + 85), sheet[:18], fill=(255, 255, 255), anchor="mm")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    # ------------------------------------------------------------------
    # 2. Generate PBIP preview
    # ------------------------------------------------------------------

    @staticmethod
    def generate_pbip_preview(pbip_path: str | Path | None = None,
                              page_names: list[str] | None = None,
                              visual_types: list[str] | None = None) -> bytes:
        """
        Generate a visual preview of a PBIP report structure.

        Parameters
        ----------
        pbip_path:
            Path to the ``.pbip`` file. If provided, page/visual metadata is
            extracted from the adjacent Report folder.
        page_names:
            Fallback list of page names (used when pbip_path is None or unreadable).
        visual_types:
            Fallback list of visual types.

        Returns PNG bytes (base64-encode yourself if needed).
        """
        pages: list[str] = page_names or []
        visuals: list[str] = visual_types or []

        if pbip_path is not None:
            pages, visuals = VisionQA._extract_pbip_metadata(Path(pbip_path))

        if not pages:
            pages = ["Page 1"]
        if not visuals:
            visuals = ["chart", "table", "card"]

        return VisionQA._render_pbip_preview(pages, visuals)

    @staticmethod
    def _extract_pbip_metadata(pbip_path: Path) -> tuple[list[str], list[str]]:
        """Parse page names and visual types from a PBIP Report folder."""
        pages: list[str] = []
        visuals: list[str] = []

        report_dir = pbip_path.parent / (pbip_path.stem + ".Report")
        if not report_dir.exists():
            return pages, visuals

        # pages are sub-directories of the Report/pages folder
        pages_dir = report_dir / "pages"
        if pages_dir.exists():
            for page_dir in sorted(pages_dir.iterdir()):
                if page_dir.is_dir():
                    # displayName is in page.json
                    page_json = page_dir / "page.json"
                    if page_json.exists():
                        try:
                            data = json.loads(page_json.read_text())
                            pages.append(data.get("displayName", page_dir.name))
                        except Exception:
                            pages.append(page_dir.name)

                    # visuals are in visuals/<uuid>/visual.json
                    visuals_dir = page_dir / "visuals"
                    if visuals_dir.exists():
                        for v_dir in visuals_dir.iterdir():
                            vj = v_dir / "visual.json"
                            if vj.exists():
                                try:
                                    vd = json.loads(vj.read_text())
                                    vtype = (
                                        vd.get("visual", {})
                                        .get("visualType", "unknown")
                                    )
                                    visuals.append(vtype)
                                except Exception:
                                    visuals.append("unknown")

        return pages, list(dict.fromkeys(visuals))  # deduplicate preserving order

    @staticmethod
    def _render_pbip_preview(pages: list[str], visuals: list[str]) -> bytes:
        """Render PBIP structure preview using matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return VisionQA._pil_pbip_preview(pages, visuals)

        n_pages = len(pages)
        fig_height = max(4, 2.5 + n_pages * 1.8)
        fig, ax = plt.subplots(figsize=(9, fig_height), facecolor="#f3f2f1")
        ax.set_facecolor("#f3f2f1")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, fig_height)
        ax.axis("off")

        ax.text(5, fig_height - 0.4, "Power BI Report Preview",
                ha="center", va="center", fontsize=13, fontweight="bold", color="#252423")

        vis_colors = {
            "barChart": "#118DFF",
            "columnChart": "#118DFF",
            "lineChart": "#12239E",
            "pieChart": "#E66C37",
            "donutChart": "#E66C37",
            "card": "#498205",
            "multiRowCard": "#498205",
            "tableEx": "#744EC2",
            "pivotTable": "#744EC2",
            "slicer": "#C43F1E",
            "map": "#1AAB40",
            "filledMap": "#1AAB40",
            "image": "#8A8886",
            "textbox": "#8A8886",
            "unknown": "#8A8886",
        }

        vis_cols = 4
        vis_w, vis_h = 1.9, 0.9

        y_cursor = fig_height - 1.1
        for page in pages:
            # Page header
            page_rect = mpatches.FancyBboxPatch(
                (0.1, y_cursor - 0.35), 9.8, 0.55,
                boxstyle="round,pad=0.05",
                facecolor="#0078d4", edgecolor="none",
            )
            ax.add_patch(page_rect)
            ax.text(0.4, y_cursor - 0.07, page,
                    va="center", fontsize=8, color="white", fontweight="bold")
            y_cursor -= 0.6

            # Visual type grid for this page
            for idx, vtype in enumerate(visuals[:vis_cols * 2]):
                col = idx % vis_cols
                row = idx // vis_cols
                x = 0.3 + col * (vis_w + 0.15)
                y = y_cursor - row * (vis_h + 0.1) - vis_h
                color = vis_colors.get(vtype, vis_colors["unknown"])
                rect = mpatches.FancyBboxPatch(
                    (x, y), vis_w, vis_h,
                    boxstyle="round,pad=0.04",
                    facecolor=color, edgecolor="white", linewidth=0.5, alpha=0.85,
                )
                ax.add_patch(rect)
                ax.text(x + vis_w / 2, y + vis_h / 2, vtype[:14],
                        ha="center", va="center", fontsize=5.5, color="white")

            rows_used = max(1, (len(visuals[:vis_cols * 2]) + vis_cols - 1) // vis_cols)
            y_cursor -= rows_used * (vis_h + 0.1) + 0.2

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _pil_pbip_preview(pages: list[str], visuals: list[str]) -> bytes:
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return b""

        img = Image.new("RGB", (900, 500), color=(243, 242, 241))
        draw = ImageDraw.Draw(img)
        draw.text((450, 20), "Power BI Report Preview", fill=(37, 36, 35), anchor="mm")

        for idx, page in enumerate(pages[:4]):
            draw.rectangle([10, 50 + idx * 100, 890, 145 + idx * 100], fill=(0, 120, 212))
            draw.text((20, 78 + idx * 100), page, fill=(255, 255, 255))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    # ------------------------------------------------------------------
    # 3. GPT-4o Vision comparison
    # ------------------------------------------------------------------

    @staticmethod
    def run_comparison(
        tableau_thumbnail_b64: str,
        pbip_preview_b64: str,
        openai_key: str,
    ) -> dict[str, Any]:
        """
        Call GPT-4o with both images and return structured fidelity comparison.

        Parameters
        ----------
        tableau_thumbnail_b64:
            Base64-encoded PNG of the Tableau dashboard.
        pbip_preview_b64:
            Base64-encoded PNG of the Power BI preview.
        openai_key:
            OpenAI API key (sk-...).

        Returns the parsed comparison dict matching COMPARISON_SCHEMA.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package not installed") from exc

        client = OpenAI(api_key=openai_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_schema", "json_schema": COMPARISON_SCHEMA},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": COMPARISON_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{tableau_thumbnail_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{pbip_preview_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=2000,
            timeout=120,
        )

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("GPT-4o returned an empty response for vision comparison")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as _je:
            raise ValueError(f"GPT-4o vision response was not valid JSON: {_je}") from _je

    # ------------------------------------------------------------------
    # 4. Helper: get fidelity score
    # ------------------------------------------------------------------

    @staticmethod
    def get_fidelity_score(comparison_result: dict[str, Any]) -> float:
        """Extract the numeric fidelity score (0–100) from a comparison result."""
        score = comparison_result.get("fidelity_score", 0.0)
        return float(score)

    # ------------------------------------------------------------------
    # 5. Auto-fix loop
    # ------------------------------------------------------------------

    @staticmethod
    def get_fixable_deltas(comparison_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return only the fixable deltas from a comparison result."""
        return [d for d in comparison_result.get("deltas", []) if d.get("fixable")]

    @staticmethod
    def needs_fix_iteration(comparison_result: dict[str, Any]) -> bool:
        """
        Return True if another fix iteration should be attempted.
        Criteria: fidelity < 95 AND at least one fixable delta exists.
        """
        return (
            VisionQA.get_fidelity_score(comparison_result) < 95
            and bool(VisionQA.get_fixable_deltas(comparison_result))
        )

    @classmethod
    def run_fix_loop(
        cls,
        tableau_thumbnail_b64: str,
        pbip_preview_b64_fn,  # callable() -> str  (regenerates preview after fix)
        openai_key: str,
        fix_callback,          # callable(fixable_deltas: list) -> None
    ) -> dict[str, Any]:
        """
        Run comparison + auto-fix loop up to MAX_FIX_ITERATIONS times.

        Parameters
        ----------
        pbip_preview_b64_fn:
            Zero-argument callable that returns the current PBIP preview as
            base64. Called after each fix iteration to regenerate the preview.
        fix_callback:
            Called with the list of fixable deltas before re-comparing.
            Caller is responsible for applying the actual fixes.

        Returns the final comparison result dict.
        """
        result = cls.run_comparison(tableau_thumbnail_b64, pbip_preview_b64_fn(), openai_key)

        for iteration in range(cls.MAX_FIX_ITERATIONS):
            if not cls.needs_fix_iteration(result):
                break
            fixable = cls.get_fixable_deltas(result)
            fix_callback(fixable)
            new_preview = pbip_preview_b64_fn()
            result = cls.run_comparison(tableau_thumbnail_b64, new_preview, openai_key)
            result["_fix_iteration"] = iteration + 1

        return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)
