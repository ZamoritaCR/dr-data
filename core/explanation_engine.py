"""
Explanation Engine - Generates executive summaries, analyst audits, and
visual preview images. Assembles a complete HTML intelligence report.
"""

import sys
import io
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import anthropic
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader

# Windows encoding fix (only if not already wrapped)
if sys.platform == "win32" and getattr(sys.stdout, "encoding", "") != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
    )

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import _get_secret
from config.prompts import EXPLANATION_SYSTEM_PROMPT, ANALYST_REPORT_PROMPT


class ExplanationEngine:
    """Generate intelligence reports for AI-created dashboards."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 8192
    MAX_RETRIES = 3

    def __init__(self):
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.template_dir = Path(__file__).parent
        print(f"[OK] Explanation engine ready (model: {self.MODEL})")

    # ------------------------------------------------------------------ #
    #  Claude API helpers                                                  #
    # ------------------------------------------------------------------ #

    def _call_claude(self, system_prompt, user_message, max_tokens=None):
        """Make a single Claude API call and return the text response."""
        t0 = time.time()
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=max_tokens or self.MAX_TOKENS,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        elapsed = time.time() - t0
        text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        print(f"       Response: {tokens_out} tokens out, "
              f"{tokens_in} tokens in, {elapsed:.1f}s")
        return text

    def _parse_json(self, raw_text):
        """Extract and parse JSON from Claude's response."""
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()
        return json.loads(text)

    # ------------------------------------------------------------------ #
    #  Executive Summary                                                   #
    # ------------------------------------------------------------------ #

    def generate_executive_summary(self, user_request, data_profile,
                                   dashboard_spec):
        """Generate a markdown executive summary of the dashboard design.

        Returns:
            str: Markdown text.
        """
        user_message = (
            f"USER REQUEST:\n{user_request}\n\n"
            f"DATASET PROFILE:\n{json.dumps(data_profile, indent=2, default=str)}\n\n"
            f"DASHBOARD SPECIFICATION:\n{json.dumps(dashboard_spec, indent=2, default=str)}\n\n"
            "Write the executive summary following the structure in your instructions."
        )

        print("\n[CALL] Generating executive summary...")
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                text = self._call_claude(EXPLANATION_SYSTEM_PROMPT, user_message)
                print(f"[OK] Executive summary generated ({len(text)} chars)")
                return text
            except anthropic.APIError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] API error: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)

        raise RuntimeError("Failed to generate executive summary")

    # ------------------------------------------------------------------ #
    #  Analyst Report                                                      #
    # ------------------------------------------------------------------ #

    def generate_analyst_report(self, user_request, data_profile,
                                dashboard_spec, tableau_spec=None):
        """Generate a structured analyst audit with confidence scores.

        Returns:
            dict: Parsed JSON with all audit sections and scores.
        """
        parts = [
            f"USER REQUEST:\n{user_request}\n",
            f"DATASET PROFILE:\n{json.dumps(data_profile, indent=2, default=str)}\n",
            f"DASHBOARD SPECIFICATION:\n{json.dumps(dashboard_spec, indent=2, default=str)}\n",
        ]
        if tableau_spec:
            parts.append(
                f"ORIGINAL TABLEAU SPECIFICATION:\n"
                f"{json.dumps(tableau_spec, indent=2, default=str)}\n"
            )
        else:
            parts.append(
                "No Tableau source was provided. "
                "Skip the migration comparison section -- set all migration "
                "fields to empty/zero.\n"
            )
        parts.append(
            "Perform the full audit and return the JSON object as specified."
        )
        user_message = "\n".join(parts)

        print("\n[CALL] Generating analyst report...")
        raw_text = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_text = self._call_claude(
                    ANALYST_REPORT_PROMPT, user_message, max_tokens=8192
                )
                report = self._parse_json(raw_text)
                print(f"[OK] Analyst report parsed successfully")
                return report
            except json.JSONDecodeError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"JSON parse failed: {e}")
                if attempt < self.MAX_RETRIES and raw_text:
                    # Ask Claude to fix
                    print("       [HEAL] Asking Claude to fix JSON...")
                    fix_msg = (
                        f"Your previous response was invalid JSON.\n"
                        f"Error: {e}\n\n"
                        f"Broken output:\n{raw_text}\n\n"
                        "Return ONLY the corrected JSON. No markdown fences."
                    )
                    try:
                        raw_text = self._call_claude(
                            "Fix the broken JSON. Return ONLY valid JSON.",
                            fix_msg,
                        )
                        report = self._parse_json(raw_text)
                        print(f"[OK] Analyst report parsed (self-healed)")
                        return report
                    except (json.JSONDecodeError, anthropic.APIError):
                        pass
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)
            except anthropic.APIError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] API error: {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(2 ** attempt)

        raise RuntimeError("Failed to generate analyst report")

    # ------------------------------------------------------------------ #
    #  Preview Images (Plotly)                                             #
    # ------------------------------------------------------------------ #

    def _build_measure_map(self, dashboard_spec):
        """Build a dict mapping measure names to their underlying column.

        Parses simple DAX patterns like SUM(Table[Col]) to extract the column.
        For complex measures, returns None (they will be skipped).
        """
        measure_map = {}
        for m in dashboard_spec.get("measures", []):
            name = m.get("name", "")
            dax = m.get("dax", "")
            # Match SUM(Table[Column]), AVERAGE(...), COUNT(...), etc.
            match = re.match(
                r"^(SUM|AVERAGE|COUNT|COUNTROWS|MIN|MAX|DISTINCTCOUNT)\s*\(\s*\w+\[([^\]]+)\]\s*\)$",
                dax.strip(), re.IGNORECASE,
            )
            if match:
                agg_func = match.group(1).upper()
                col_name = match.group(2)
                measure_map[name] = {"column": col_name, "agg": agg_func}
            else:
                # Complex measure -- store the DAX for info but no direct mapping
                measure_map[name] = {"column": None, "agg": None, "dax": dax}
        return measure_map

    def _resolve_measure_scalar(self, name, df, measure_map):
        """Resolve a measure or column name to a single scalar value."""
        # Direct column
        if name in df.columns:
            if df[name].dtype in ("float64", "int64", "float32", "int32"):
                return df[name].sum()
            return df[name].nunique()

        # Known measure
        info = measure_map.get(name, {})
        col = info.get("column")
        agg = info.get("agg")
        if col and col in df.columns and agg:
            if agg in ("SUM",):
                return df[col].sum()
            elif agg in ("AVERAGE",):
                return df[col].mean()
            elif agg in ("COUNT", "COUNTROWS"):
                return len(df)
            elif agg == "DISTINCTCOUNT":
                return df[col].nunique()
            elif agg == "MIN":
                return df[col].min()
            elif agg == "MAX":
                return df[col].max()

        # Special case: Profit Margin = DIVIDE([Total Profit], [Total Sales], 0)
        dax = info.get("dax", "")
        if "DIVIDE" in dax.upper():
            # Try to resolve the two operands
            inner = re.findall(r"\[([^\]]+)\]", dax)
            if len(inner) >= 2:
                num = self._resolve_measure_scalar(inner[0], df, measure_map)
                den = self._resolve_measure_scalar(inner[1], df, measure_map)
                if num is not None and den is not None and den != 0:
                    return num / den
        return None

    def _resolve_measure_grouped(self, name, df, measure_map, group_col):
        """Resolve a measure grouped by a category column. Returns a Series."""
        # Direct column
        if name in df.columns:
            if df[name].dtype in ("float64", "int64", "float32", "int32"):
                return df.groupby(group_col)[name].sum()
            return df.groupby(group_col)[name].nunique()

        # Known measure
        info = measure_map.get(name, {})
        col = info.get("column")
        agg = info.get("agg")
        if col and col in df.columns and agg:
            grouped = df.groupby(group_col)
            if agg == "SUM":
                return grouped[col].sum()
            elif agg == "AVERAGE":
                return grouped[col].mean()
            elif agg == "DISTINCTCOUNT":
                return grouped[col].nunique()
            elif agg in ("COUNT", "COUNTROWS"):
                return grouped[col].count()
        return None

    def generate_preview_images(self, dashboard_spec, dataframe, output_dir):
        """Create Plotly chart PNGs for each visual in the dashboard spec.

        Returns:
            list[dict]: Each dict has 'path', 'title', 'type'.
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "preview_images"
        images_dir.mkdir(parents=True, exist_ok=True)

        measure_map = self._build_measure_map(dashboard_spec)

        results = []
        all_visuals = []
        for page in dashboard_spec.get("pages", []):
            for v in page.get("visuals", []):
                all_visuals.append(v)

        print(f"\n[CALL] Generating preview images for {len(all_visuals)} visuals...")

        for i, visual in enumerate(all_visuals):
            vtype = visual.get("type", "unknown")
            title = visual.get("title", f"Visual {i+1}")
            try:
                fig = self._create_plotly_figure(visual, dataframe, measure_map)
                if fig is None:
                    print(f"       [SKIP] {title} ({vtype}): no data resolved")
                    continue

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    title=dict(text=title, font=dict(size=14, color="#e6edf3")),
                    font=dict(color="#8b949e", size=11),
                    margin=dict(l=50, r=30, t=50, b=40),
                    width=600,
                    height=400,
                )

                fname = f"preview_{i:02d}_{vtype}.png"
                fpath = images_dir / fname
                fig.write_image(str(fpath), scale=2)

                results.append({
                    "path": str(fpath),
                    "title": title,
                    "type": vtype,
                })
                print(f"       [OK] {title} -> {fname}")

            except Exception as e:
                print(f"       [SKIP] {title} ({vtype}): {e}")

        print(f"[OK] Generated {len(results)} preview images")
        return results

    def _create_plotly_figure(self, visual, df, measure_map):
        """Map a dashboard visual spec to a Plotly figure."""
        vtype = visual.get("type", "")
        data_roles = visual.get("data_roles", {})
        categories = data_roles.get("category", [])
        values = data_roles.get("values", [])

        if vtype == "card":
            return self._create_card_figure(visual, values, df, measure_map)
        elif vtype == "lineChart":
            return self._create_line_figure(categories, values, df, measure_map)
        elif vtype in ("clusteredBarChart", "stackedBarChart"):
            return self._create_bar_figure(categories, values, df, vtype, measure_map)
        elif vtype == "donutChart":
            return self._create_donut_figure(categories, values, df, measure_map)
        elif vtype in ("table", "tableEx"):
            return self._create_table_figure(values, df, measure_map)
        else:
            return None

    def _create_card_figure(self, visual, values, df, measure_map):
        """Big number card."""
        if not values:
            return None

        field = values[0]
        val = self._resolve_measure_scalar(field, df, measure_map)
        if val is None:
            return None

        # Format based on value range
        if isinstance(val, float):
            if abs(val) > 1_000_000:
                display = f"${val/1_000_000:,.1f}M"
            elif abs(val) > 1_000:
                display = f"${val:,.0f}"
            elif abs(val) < 1:
                display = f"{val:.1%}"
            else:
                display = f"{val:,.0f}"
        else:
            display = f"{val:,}"

        fig = go.Figure()
        fig.add_annotation(
            text=display,
            x=0.5, y=0.55,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=48, color="#e6edf3"),
        )
        fig.add_annotation(
            text=field,
            x=0.5, y=0.25,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, color="#8b949e"),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=250,
        )
        return fig

    def _create_line_figure(self, categories, values, df, measure_map):
        """Line chart with time series."""
        if not categories or not values:
            return None

        cat_col = categories[0]
        if cat_col not in df.columns:
            return None

        fig = go.Figure()
        colors = ["#1f6feb", "#238636", "#d29922", "#da3633", "#a371f7"]

        for i, val_name in enumerate(values):
            series = self._resolve_measure_grouped(
                val_name, df, measure_map, cat_col
            )
            if series is None:
                continue
            series = series.sort_index()

            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=val_name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        return fig if fig.data else None

    def _create_bar_figure(self, categories, values, df, vtype, measure_map):
        """Clustered or stacked bar chart."""
        if not categories or not values:
            return None

        cat_col = categories[0]
        if cat_col not in df.columns:
            return None

        colors = ["#1f6feb", "#238636", "#d29922", "#da3633"]
        barmode = "stack" if "stacked" in vtype else "group"

        fig = go.Figure()
        for i, val_name in enumerate(values):
            series = self._resolve_measure_grouped(
                val_name, df, measure_map, cat_col
            )
            if series is None:
                continue
            series = series.sort_values(ascending=False)

            fig.add_trace(go.Bar(
                x=series.index,
                y=series.values,
                name=val_name,
                marker_color=colors[i % len(colors)],
            ))

        fig.update_layout(barmode=barmode)
        return fig if fig.data else None

    def _create_donut_figure(self, categories, values, df, measure_map):
        """Donut (pie with hole) chart."""
        if not categories:
            return None

        cat_col = categories[0]
        if cat_col not in df.columns:
            return None

        # Use first resolvable value
        grouped = None
        for v in values:
            grouped = self._resolve_measure_grouped(v, df, measure_map, cat_col)
            if grouped is not None:
                break

        if grouped is None:
            grouped = df[cat_col].value_counts()

        colors = ["#1f6feb", "#238636", "#d29922", "#da3633", "#a371f7"]
        fig = go.Figure(go.Pie(
            labels=grouped.index.tolist(),
            values=grouped.values.tolist(),
            hole=0.5,
            marker=dict(colors=colors[:len(grouped)]),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        return fig

    def _create_table_figure(self, values, df, measure_map):
        """Table visual -- show actual columns only, skip measures."""
        cols = [v for v in values if v in df.columns]
        if not cols:
            return None

        sample = df[cols].head(10)

        fig = go.Figure(go.Table(
            header=dict(
                values=[f"<b>{c}</b>" for c in cols],
                fill_color="#161b22",
                font=dict(color="#e6edf3", size=12),
                align="left",
                line_color="#30363d",
            ),
            cells=dict(
                values=[sample[c].tolist() for c in cols],
                fill_color="#0d1117",
                font=dict(color="#8b949e", size=11),
                align="left",
                line_color="#30363d",
                format=[".2f" if sample[c].dtype == "float64" else "" for c in cols],
            ),
        ))
        fig.update_layout(height=350)
        return fig

    # ------------------------------------------------------------------ #
    #  Full Report Assembly                                                #
    # ------------------------------------------------------------------ #

    def generate_full_report(self, user_request, data_profile, dashboard_spec,
                             dataframe, output_dir, tableau_spec=None,
                             exec_summary=None, analyst=None, previews=None):
        """Generate the complete HTML intelligence report.

        Pre-computed results can be passed to avoid duplicate API calls:
            exec_summary: str from generate_executive_summary()
            analyst: dict from generate_analyst_report()
            previews: list from generate_preview_images()

        Returns:
            str: Path to the generated HTML file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Executive summary
        if exec_summary is None:
            exec_summary = self.generate_executive_summary(
                user_request, data_profile, dashboard_spec
            )

        # 2. Analyst report
        if analyst is None:
            analyst = self.generate_analyst_report(
                user_request, data_profile, dashboard_spec, tableau_spec
            )

        # 3. Preview images
        if previews is None:
            previews = self.generate_preview_images(
                dashboard_spec, dataframe, output_dir
            )

        # 4. Assemble template context
        scorecard = analyst.get("scorecard", {})
        total_score = scorecard.get("total_score", 0)

        # Score color
        if total_score >= 80:
            score_class = "score-green"
            gauge_color = "#238636"
        elif total_score >= 60:
            score_class = "score-amber"
            gauge_color = "#d29922"
        else:
            score_class = "score-red"
            gauge_color = "#da3633"

        # SVG gauge math: circumference of r=80 circle
        circumference = 2 * math.pi * 80  # ~502.65
        filled = circumference * (total_score / 100)
        gauge_dasharray = f"{filled:.1f} {circumference:.1f}"

        # Score bars
        def bar_color(val):
            if val >= 80:
                return "#238636"
            elif val >= 60:
                return "#d29922"
            return "#da3633"

        score_bars = [
            {
                "label": "Data Accuracy",
                "value": scorecard.get("data_accuracy", 0),
                "color": bar_color(scorecard.get("data_accuracy", 0)),
            },
            {
                "label": "Measure Quality",
                "value": scorecard.get("measure_quality", 0),
                "color": bar_color(scorecard.get("measure_quality", 0)),
            },
            {
                "label": "Visual Effectiveness",
                "value": scorecard.get("visual_effectiveness", 0),
                "color": bar_color(scorecard.get("visual_effectiveness", 0)),
            },
            {
                "label": "User Experience",
                "value": scorecard.get("user_experience", 0),
                "color": bar_color(scorecard.get("user_experience", 0)),
            },
        ]

        # Count stats
        total_visuals = sum(
            len(p.get("visuals", []))
            for p in dashboard_spec.get("pages", [])
        )
        total_measures = len(dashboard_spec.get("measures", []))
        total_columns = len(data_profile.get("columns", []))

        # Preview images -- make paths relative to the HTML file
        preview_for_template = []
        for p in previews:
            rel = os.path.relpath(p["path"], str(output_dir))
            preview_for_template.append({
                "path": rel.replace("\\", "/"),
                "title": p["title"],
                "type": p["type"],
            })

        # Migration
        migration = analyst.get("migration_comparison")
        if migration:
            # Check if it's all empty/zero
            has_content = (
                migration.get("replicated")
                or migration.get("not_replicated")
                or migration.get("improvements")
                or migration.get("completeness_pct", 0) > 0
            )
            if not has_content:
                migration = None

        # Render
        env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
        )
        template = env.get_template("report_template.html")

        html = template.render(
            generated_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            classification=dashboard_spec.get("classification", "General"),
            executive_summary=exec_summary,
            total_visuals=total_visuals,
            total_measures=total_measures,
            total_columns=total_columns,
            total_score=total_score,
            score_class=score_class,
            gauge_color=gauge_color,
            gauge_dasharray=gauge_dasharray,
            score_bars=score_bars,
            preview_images=preview_for_template,
            dax_measures=analyst.get("dax_measures", []),
            visuals=analyst.get("visuals", []),
            data_mapping=analyst.get("data_mapping", []),
            migration=migration,
            critical_issues=analyst.get("critical_issues", []),
            recommendations=analyst.get("recommendations", []),
        )

        report_path = output_dir / "intelligence_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"\n[OK] Intelligence report: {report_path}")
        print(f"     Size: {report_path.stat().st_size:,} bytes")
        return str(report_path)
