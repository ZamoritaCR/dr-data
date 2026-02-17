"""PDF report generator using matplotlib and reportlab."""
import os
import json
import base64
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime


class PDFReportBuilder:

    def build(self, dashboard_spec, data_path, profile, output_dir):
        """Generate PDF report. Returns file path."""
        os.makedirs(output_dir, exist_ok=True)

        ext = data_path.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(data_path)
        elif ext in ("xlsx", "xls"):
            from app.file_handler import load_excel_smart
            df, _ = load_excel_smart(data_path)
        else:
            df = pd.read_csv(data_path, sep=None, engine="python")

        title = dashboard_spec.get("title", "Data Report")

        chart_paths = []
        pages = dashboard_spec.get("pages", [])
        for page in pages:
            for i, visual in enumerate(page.get("visuals", [])):
                chart_path = self._render_chart(visual, df, output_dir, i)
                if chart_path:
                    chart_paths.append((visual.get("title", ""), chart_path))

        html = self._build_pdf_html(title, profile, chart_paths, df)

        pdf_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_Report.pdf")

        try:
            from weasyprint import HTML
            HTML(string=html).write_pdf(pdf_path)
        except ImportError:
            # Fallback: save as HTML
            html_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_Report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            return html_path

        return pdf_path

    def build_from_dataframe(self, dashboard_spec, df, profile, output_dir):
        """Generate PDF report from DataFrame (no file path needed)."""
        os.makedirs(output_dir, exist_ok=True)

        title = dashboard_spec.get("title", "Data Report")

        chart_paths = []
        pages = dashboard_spec.get("pages", [])
        for page in pages:
            for i, visual in enumerate(page.get("visuals", [])):
                chart_path = self._render_chart(visual, df, output_dir, i)
                if chart_path:
                    chart_paths.append((visual.get("title", ""), chart_path))

        html = self._build_pdf_html(title, profile, chart_paths, df)

        pdf_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_Report.pdf")

        try:
            from weasyprint import HTML
            HTML(string=html).write_pdf(pdf_path)
        except ImportError:
            html_path = os.path.join(output_dir, f"{title.replace(' ', '_')}_Report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            return html_path

        return pdf_path

    def _render_chart(self, visual, df, output_dir, idx):
        """Render a chart as PNG using matplotlib."""
        try:
            vtype = visual.get("type", "bar_chart")
            data_spec = visual.get("data", {})
            x_col = data_spec.get("x", "")
            y_col = data_spec.get("y", "")
            title = visual.get("title", "")
            agg = data_spec.get("aggregation", "sum")

            if not x_col or not y_col:
                return None
            if x_col not in df.columns or y_col not in df.columns:
                return None

            grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
            grouped = grouped.sort_values(y_col, ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#0d1117")
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            ax.spines["bottom"].set_color("#30363d")
            ax.spines["left"].set_color("#30363d")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if "bar" in vtype.lower():
                ax.bar(grouped[x_col].astype(str), grouped[y_col], color="#1f6feb")
                plt.xticks(rotation=45, ha="right")
            elif "line" in vtype.lower():
                ax.plot(grouped[x_col].astype(str), grouped[y_col],
                        color="#1f6feb", linewidth=2, marker="o", markersize=4)
                plt.xticks(rotation=45, ha="right")
            else:
                ax.bar(grouped[x_col].astype(str), grouped[y_col], color="#1f6feb")
                plt.xticks(rotation=45, ha="right")

            ax.set_title(title, color="#e6edf3", fontsize=12, pad=10)
            ax.yaxis.label.set_color("#8b949e")
            plt.tight_layout()

            path = os.path.join(output_dir, f"chart_{idx}.png")
            fig.savefig(path, dpi=150, facecolor="#0d1117")
            plt.close(fig)
            return path

        except Exception as e:
            print(f"[WARN] Chart render failed: {e}")
            return None

    def _build_pdf_html(self, title, profile, chart_paths, df):
        """Build HTML suitable for PDF conversion."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        chart_html = ""
        for chart_title, path in chart_paths:
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                chart_html += (
                    f'<div style="margin-bottom:24px;">'
                    f'<img src="data:image/png;base64,{b64}" '
                    f'style="width:100%;border-radius:4px;"/>'
                    f'</div>'
                )
            except Exception:
                pass

        row_count = profile.get("row_count", len(df))
        col_count = profile.get("column_count", len(df.columns))

        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: system-ui, sans-serif; color: #1a1a1a;
         padding: 40px; max-width: 800px; margin: 0 auto; }}
  h1 {{ font-size: 24px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 32px; color: #333; }}
  .meta {{ color: #666; font-size: 12px; margin-bottom: 24px; }}
  .stat {{ display: inline-block; margin-right: 32px; }}
  .stat .val {{ font-size: 22px; font-weight: 700; }}
  .stat .lbl {{ font-size: 11px; color: #666; text-transform: uppercase; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Generated {timestamp} -- {row_count:,} rows, {col_count} columns</div>

<div style="margin-bottom:32px;">
  <div class="stat"><div class="val">{row_count:,}</div><div class="lbl">Rows</div></div>
  <div class="stat"><div class="val">{col_count}</div><div class="lbl">Columns</div></div>
</div>

<h2>Visualizations</h2>
{chart_html}

<div style="margin-top:40px; padding-top:16px; border-top:1px solid #ddd;
            font-size:11px; color:#999; text-align:center;">
  Generated by Dr. Data -- Dashboard Intelligence Platform
</div>
</body>
</html>"""
