"""Real-image VFE comparison between Tableau and Power BI renders."""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional

import cv2
import imagehash
import matplotlib
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalized_name(name: str) -> str:
    return "".join(c.lower() for c in Path(name).stem if c.isalnum())


def _extract_twbx_thumbnail(twbx_path: str, output_dir: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(twbx_path) as zf:
            image_names = [
                n for n in zf.namelist()
                if n.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            preferred = sorted(
                image_names,
                key=lambda n: (
                    "thumbnail" not in n.lower(),
                    "preview" not in n.lower(),
                    len(n),
                ),
            )
            if not preferred:
                return None
            out_path = Path(output_dir) / "tableau_thumbnail.png"
            out_path.write_bytes(zf.read(preferred[0]))
            return str(out_path)
    except Exception:
        return None


def _download_tableau_cloud_image(filename: str, output_dir: str) -> Dict[str, str]:
    try:
        from core.tableau_connector import TableauCloudConnector

        connector = TableauCloudConnector()
        normalized_target = _normalized_name(filename)
        workbook = None
        for wb in connector.list_workbooks():
            wb_name = wb.get("name", "")
            if _normalized_name(wb_name) == normalized_target:
                workbook = wb
                break
        if workbook is None:
            for wb in connector.list_workbooks():
                wb_norm = _normalized_name(wb.get("name", ""))
                if normalized_target in wb_norm or wb_norm in normalized_target:
                    workbook = wb
                    break
        if not workbook:
            return {}

        images = connector.get_view_images(workbook["id"], output_dir)
        if not images:
            return {}

        first_view, image_path = next(iter(images.items()))
        return {
            "path": image_path,
            "source": "tableau_cloud",
            "workbook_id": workbook["id"],
            "workbook_name": workbook.get("name", ""),
            "view_name": first_view,
        }
    except Exception:
        return {}


def _render_local_twbx_preview(twbx_path: str, output_dir: str) -> Optional[str]:
    try:
        from app.file_handler import _extract_twbx_data

        _, dfs = _extract_twbx_data(twbx_path)
        if not dfs:
            return None
        df = next(iter(dfs.values())).copy()
        if df.empty:
            return None

        date_col = next((c for c in df.columns if "date" in c.lower()), "")
        sales_col = next((c for c in df.columns if "sale" in c.lower()), "")
        region_col = next((c for c in df.columns if c.lower() == "region"), "")
        subregion_col = next((c for c in df.columns if "sub-region" in c.lower()), "")
        customer_col = next((c for c in df.columns if "customer" in c.lower()), "")
        state_col = next((c for c in df.columns if "state" in c.lower() and "code" not in c.lower()), "")

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0d0d22")
        axes = axes.flatten()
        for ax in axes:
            ax.set_facecolor("#171733")
            ax.tick_params(colors="#e8ecff")
            for spine in ax.spines.values():
                spine.set_color("#394055")

        if customer_col and sales_col:
            by_customer = df.groupby(customer_col)[sales_col].sum().sort_values(ascending=False)
            axes[0].bar(by_customer.index.astype(str), by_customer.values, color="#00ff88")
            axes[0].set_title("Sales by Customer Type", color="#ffd700")
            axes[0].tick_params(axis="x", rotation=20)

        if region_col and sales_col:
            by_region = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
            axes[1].bar(by_region.index.astype(str), by_region.values, color="#ffd700")
            axes[1].set_title("Sales by Region", color="#ffd700")
            axes[1].tick_params(axis="x", rotation=20)

        if state_col and sales_col:
            by_state = df.groupby(state_col)[sales_col].sum().sort_values(ascending=False).head(10)
            axes[2].barh(by_state.index.astype(str), by_state.values, color="#66d9ff")
            axes[2].set_title("Top States by Sales", color="#ffd700")

        if date_col and sales_col:
            series = df[[date_col, sales_col]].copy()
            series[date_col] = np.array(series[date_col], dtype="datetime64[ns]")
            series["Quarter"] = (
                series[date_col].astype("datetime64[ns]")
            )
            series["Quarter"] = series[date_col].dt.to_period("Q").astype(str)
            by_quarter = series.groupby("Quarter")[sales_col].sum().sort_index()
            axes[3].plot(by_quarter.index.astype(str), by_quarter.values, color="#00ff88", linewidth=2.5)
            axes[3].set_title("Quarterly Sales", color="#ffd700")
            axes[3].tick_params(axis="x", rotation=20)

        if subregion_col:
            axes[1].set_xlabel(subregion_col, color="#98a0b3")

        fig.suptitle(Path(twbx_path).stem, color="#e8ecff", fontsize=18)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out_path = Path(output_dir) / "tableau_local_preview.png"
        fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(out_path)
    except Exception:
        return None


def _load_resized_rgb(path: str, size=(1280, 720)):
    img = Image.open(path).convert("RGB").resize(size)
    return img, np.array(img)


def _ssim_score(tab_arr: np.ndarray, pbi_arr: np.ndarray) -> float:
    tab_gray = cv2.cvtColor(tab_arr, cv2.COLOR_RGB2GRAY)
    pbi_gray = cv2.cvtColor(pbi_arr, cv2.COLOR_RGB2GRAY)
    return float(ssim_metric(tab_gray, pbi_gray))


def _phash_score(tab_img: Image.Image, pbi_img: Image.Image) -> float:
    tab_hash = imagehash.phash(tab_img)
    pbi_hash = imagehash.phash(pbi_img)
    return float(1.0 - (tab_hash - pbi_hash) / len(tab_hash.hash.flatten()))


def _orb_score(tab_arr: np.ndarray, pbi_arr: np.ndarray) -> float:
    tab_gray = cv2.cvtColor(tab_arr, cv2.COLOR_RGB2GRAY)
    pbi_gray = cv2.cvtColor(pbi_arr, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(tab_gray, None)
    kp2, des2 = orb.detectAndCompute(pbi_gray, None)
    if des1 is None or des2 is None or not kp1 or not kp2:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance <= 48]
    return float(min(1.0, len(good) / max(1, min(len(kp1), len(kp2)))))


def _histogram_score(tab_arr: np.ndarray, pbi_arr: np.ndarray) -> float:
    score_parts = []
    for channel in range(3):
        hist1 = cv2.calcHist([tab_arr], [channel], None, [256], [0, 256])
        hist2 = cv2.calcHist([pbi_arr], [channel], None, [256], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        score_parts.append(max(0.0, min(1.0, (score + 1.0) / 2.0)))
    return float(sum(score_parts) / len(score_parts))


def _dominant_colors(path: str):
    img = Image.open(path).convert("RGB").resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    counts = {}
    for r, g, b in pixels:
        key = ((r >> 4) << 4, (g >> 4) << 4, (b >> 4) << 4)
        counts[key] = counts.get(key, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]]


def _color_score(tableau_path: str, pbi_path: str) -> float:
    tab = set(_dominant_colors(tableau_path))
    pbi = set(_dominant_colors(pbi_path))
    if not tab or not pbi:
        return 0.0
    return float(len(tab & pbi) / len(tab | pbi))


def compare_images(tableau_path: str, pbi_path: str) -> Dict[str, object]:
    tab_img, tab_arr = _load_resized_rgb(tableau_path)
    pbi_img, pbi_arr = _load_resized_rgb(pbi_path)

    layers = {
        "ssim": round(_ssim_score(tab_arr, pbi_arr), 4),
        "phash": round(_phash_score(tab_img, pbi_img), 4),
        "orb": round(_orb_score(tab_arr, pbi_arr), 4),
        "histogram": round(_histogram_score(tab_arr, pbi_arr), 4),
        "color": round(_color_score(tableau_path, pbi_path), 4),
    }
    composite = round(sum(layers.values()) / len(layers), 4)
    return {
        "composite_score": composite,
        "layers": layers,
        "verdict": "PASS" if composite >= 0.7 else "FAIL",
    }


def run_real_vfe(
    tableau_file_path: str,
    pbi_render_path: str,
    output_dir: str,
) -> Dict[str, object]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tableau_image = ""
    tableau_meta: Dict[str, str] = {}
    cloud_image = _download_tableau_cloud_image(tableau_file_path, output_dir)
    if cloud_image.get("path"):
        tableau_image = cloud_image["path"]
        tableau_meta = cloud_image
    else:
        tableau_image = _extract_twbx_thumbnail(tableau_file_path, output_dir) or ""
        if tableau_image:
            tableau_meta = {"source": "twbx_thumbnail", "path": tableau_image}
        else:
            tableau_image = _render_local_twbx_preview(tableau_file_path, output_dir) or ""
            if tableau_image:
                tableau_meta = {"source": "twbx_local_preview", "path": tableau_image}

    result = {
        "tableau_image": tableau_image,
        "pbi_image": pbi_render_path,
        "source": tableau_meta.get("source", ""),
    }
    if tableau_meta.get("workbook_id"):
        result["tableau_workbook_id"] = tableau_meta["workbook_id"]
    if tableau_meta.get("view_name"):
        result["tableau_view_name"] = tableau_meta["view_name"]

    if not tableau_image or not os.path.exists(tableau_image):
        result.update({
            "error": "No Tableau reference image available from Tableau Cloud or TWBX thumbnail",
            "verdict": "FAIL",
        })
    elif not os.path.exists(pbi_render_path):
        result.update({
            "error": "No Power BI render proof available",
            "verdict": "FAIL",
        })
    else:
        result.update(compare_images(tableau_image, pbi_render_path))

    out_path = Path(output_dir) / "vfe_real_comparison.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_path"] = str(out_path)
    return result
