"""
Dashboard Intelligence Engine -- the BRAIN of Dr. Data V2.

Takes any extracted content (data profile, Tableau workbook, PBIX metadata,
raw CSV) and produces a structured DashboardProposal for user review.

NEVER builds anything. Only proposes. The user reviews and approves before
any PBIP/HTML generation starts.

Usage:
    engine = DashboardIntelligence()
    proposal = engine.propose(data_profile, user_request="executive overview")
    proposal = engine.propose_from_tableau(tableau_extract, data_profile)
    proposal = engine.propose_from_description(description)  # no data yet
"""

import sys
import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import anthropic

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config.settings import _get_secret


# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

@dataclass
class VisualSpec:
    """A single visual in the proposed dashboard."""
    id: str
    chart_type: str              # barChart, lineChart, donutChart, scatterChart, card, filledMap, etc.
    title: str
    x_field: str
    y_field: str
    aggregation: str = "sum"     # sum, avg, count, min, max, distinct_count
    color_field: Optional[str] = None
    size_field: Optional[str] = None
    sort: str = "value_desc"     # value_desc, value_asc, label_asc, none
    top_n: Optional[int] = None
    filters: List[str] = field(default_factory=list)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 400, "h": 300})
    page: int = 0                # which page this visual belongs to
    confidence: float = 1.0      # 0-1, how confident is Claude in this choice
    why: str = ""                # rationale for this visual


@dataclass
class MeasureSpec:
    """A DAX calculated measure."""
    name: str
    dax_expression: str
    description: str
    format_string: str = ""      # e.g. "#,##0", "0.0%", "$#,##0.00"


@dataclass
class RelationshipSpec:
    """A relationship between two tables in the data model."""
    from_table: str
    from_field: str
    to_table: str
    to_field: str
    cardinality: str = "many-to-one"  # many-to-one, one-to-one, many-to-many


@dataclass
class DataModelSpec:
    """The proposed semantic model."""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    # Each table: {"name": str, "columns": [{"name": str, "type": str, "semantic_type": str}]}
    measures: List[MeasureSpec] = field(default_factory=list)
    relationships: List[RelationshipSpec] = field(default_factory=list)


@dataclass
class PageSpec:
    """A page/tab in the dashboard."""
    name: str
    visuals: List[str] = field(default_factory=list)  # list of visual IDs
    layout_hint: str = ""        # "2x3 grid", "kpi row + 2 charts", etc.


@dataclass
class DashboardProposal:
    """The complete proposal shown to the user before anything is built."""
    job_id: str
    domain: str                  # sales, hr, finance, operations, marketing, healthcare, etc.
    title: str
    subtitle: str
    audience: str                # "executive", "analyst", "operational team"
    business_questions: List[str] = field(default_factory=list)
    pages: List[PageSpec] = field(default_factory=list)
    visuals: List[VisualSpec] = field(default_factory=list)
    data_model: DataModelSpec = field(default_factory=DataModelSpec)
    global_filters: List[Dict[str, str]] = field(default_factory=list)
    # Each filter: {"column": str, "type": "dropdown|date_range|slider", "default": "All"}
    layout_grid: Dict[str, Any] = field(default_factory=lambda: {
        "rows": 3, "cols": 3, "canvas": "1280x720"
    })
    color_theme: str = "professional_dark"
    has_data: bool = True
    synthetic_data_needed: bool = False
    synthetic_data_domain: str = ""
    confidence_score: float = 0.0
    requires_user_input: List[str] = field(default_factory=list)
    design_notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        """Serialize for JSON transport / Supabase storage."""
        return asdict(self)

    def summary(self) -> str:
        """One-screen summary for user review."""
        lines = [
            f"DASHBOARD PROPOSAL: {self.title}",
            f"Domain: {self.domain} | Audience: {self.audience}",
            f"Pages: {len(self.pages)} | Visuals: {len(self.visuals)} "
            f"| Measures: {len(self.data_model.measures)}",
            f"Confidence: {self.confidence_score:.0%}",
            "",
            "BUSINESS QUESTIONS:",
        ]
        for i, q in enumerate(self.business_questions, 1):
            lines.append(f"  {i}. {q}")
        lines.append("")
        lines.append("VISUALS:")
        for v in self.visuals:
            flag = f" [confidence: {v.confidence:.0%}]" if v.confidence < 0.8 else ""
            lines.append(f"  [{v.chart_type}] {v.title} -- {v.x_field} x {v.y_field}{flag}")
        if self.requires_user_input:
            lines.append("")
            lines.append("NEEDS YOUR INPUT:")
            for item in self.requires_user_input:
                lines.append(f"  - {item}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Chart selection heuristics                                          #
# ------------------------------------------------------------------ #

CHART_RULES = {
    "time_series":          "lineChart",
    "part_to_whole":        "donutChart",
    "category_comparison":  "barChart",
    "geographic":           "filledMap",
    "correlation":          "scatterChart",
    "single_metric":        "card",
    "ranking":              "barChart",
    "distribution":         "histogram",
    "flow":                 "waterfallChart",
    "funnel":               "funnelChart",
    "table_detail":         "table",
    "kpi":                  "card",
    "trend_with_target":    "lineChart",
    "variance":             "waterfallChart",
}

SEMANTIC_TO_INTENT = {
    # (x_semantic, y_semantic) -> chart intent
    ("date", "measure"):    "time_series",
    ("category", "measure"): "category_comparison",
    ("geo", "measure"):     "geographic",
    ("measure", "measure"): "correlation",
}


# ------------------------------------------------------------------ #
#  System prompt                                                       #
# ------------------------------------------------------------------ #

INTELLIGENCE_SYSTEM_PROMPT = """You are the Dashboard Intelligence Engine inside Dr. Data V2.

YOUR ROLE: Analyze extracted data and propose a dashboard design. You produce a JSON specification that a human will review before anything is built. You NEVER build -- you only propose.

INPUT YOU RECEIVE:
- A data profile (columns, types, semantic roles, statistics, quality scores)
- Optionally: a Tableau workbook extract (existing visuals, layouts, calculated fields)
- Optionally: a user request describing what they want
- Optionally: domain context (industry, audience, business questions)

OUTPUT YOU PRODUCE:
A single JSON object with this exact structure:

{
  "domain": "sales|hr|finance|operations|marketing|healthcare|logistics|custom",
  "title": "Dashboard Title",
  "subtitle": "Context line for the audience",
  "audience": "executive|analyst|operational",
  "business_questions": ["What is our monthly revenue trend?", "Which regions underperform?"],
  "pages": [
    {"name": "Overview", "visual_ids": ["v1", "v2", "v3"], "layout_hint": "kpi row + 2x2 grid"}
  ],
  "visuals": [
    {
      "id": "v1",
      "chart_type": "card|lineChart|barChart|donutChart|scatterChart|filledMap|table|waterfallChart|funnelChart|histogram|gauge",
      "title": "Visual Title",
      "x_field": "column_name or null for cards",
      "y_field": "column_name or measure_name",
      "aggregation": "sum|avg|count|min|max|distinct_count",
      "color_field": "column_name or null",
      "size_field": "column_name or null",
      "sort": "value_desc|value_asc|label_asc|none",
      "top_n": null,
      "filters": [],
      "page": 0,
      "confidence": 0.95,
      "why": "Revenue trend over time is the first thing an exec looks for"
    }
  ],
  "data_model": {
    "tables": [{"name": "Sales", "columns": [{"name": "Revenue", "type": "decimal", "semantic_type": "measure"}]}],
    "measures": [{"name": "Total Revenue", "dax_expression": "SUM(Sales[Revenue])", "description": "Sum of all revenue", "format_string": "$#,##0"}],
    "relationships": [{"from_table": "Sales", "from_field": "ProductID", "to_table": "Products", "to_field": "ProductID", "cardinality": "many-to-one"}]
  },
  "global_filters": [{"column": "Date", "type": "date_range", "default": "Last 12 months"}],
  "layout_grid": {"rows": 3, "cols": 3, "canvas": "1280x720"},
  "color_theme": "professional_dark",
  "has_data": true,
  "synthetic_data_needed": false,
  "synthetic_data_domain": "",
  "confidence_score": 0.87,
  "requires_user_input": ["Should we break down by region or by product line?"],
  "design_notes": "Led with KPIs, then trend, then drill-down. Kept to 6 visuals for exec audience."
}

DESIGN RULES:
1. Lead with KPI cards. Always. The audience needs the health check first.
2. Use the right chart: trends=line, comparison=bar, composition=donut, relationship=scatter, ranking=horizontal bar, single number=card.
3. No pie charts. Use donut or horizontal bar.
4. Max 6-8 visuals per page. Density kills comprehension.
5. If data has a date/time column, include a time series. Always.
6. Sort charts to surface the insight, not alphabetically.
7. Every visual must have a "why" explaining the design choice.
8. Use ONLY columns that exist in the provided data profile.
9. DAX measures must reference the correct table name from the profile.
10. If you are unsure about a visual, set confidence < 0.8 and add a note to requires_user_input.
11. If no data is provided (description-only mode), set has_data=false, synthetic_data_needed=true, and describe the synthetic data domain.
12. For Tableau migrations: preserve the original intent but improve the design. Note what you changed and why.

DOMAIN DETECTION:
Infer the domain from column names, table names, and data patterns:
- Revenue, Sales, Orders, Customers, Products → sales
- Employees, Headcount, Attrition, Salary, Department → hr
- Budget, Expenses, P&L, GL, Accounts → finance
- Shipments, Delivery, Warehouse, Routes → logistics
- Patients, Diagnosis, Claims, Procedures → healthcare
- Campaigns, Clicks, Impressions, Conversions → marketing
- Mixed or unclear → set domain to "custom" and note in requires_user_input

Output ONLY valid JSON. No markdown fences. No commentary outside the JSON."""


DESCRIPTION_ONLY_PROMPT = """You are the Dashboard Intelligence Engine inside Dr. Data V2.

The user has described a dashboard they want but has NOT provided data yet.
Your job: propose a complete dashboard design with synthetic data schema.

Set has_data=false, synthetic_data_needed=true, and describe what synthetic data
should look like in synthetic_data_domain (e.g. "500 rows of sales transactions
with columns: date, region, product, revenue, quantity, customer_segment").

Follow all the same design rules and output the same JSON structure.
Be opinionated about what visuals would best answer their business questions.

Output ONLY valid JSON. No markdown fences. No commentary outside the JSON."""


TABLEAU_MIGRATION_PROMPT = """You are the Dashboard Intelligence Engine inside Dr. Data V2.

The user is migrating a Tableau workbook to Power BI. You receive:
1. The extracted Tableau workbook structure (worksheets, datasources, calculated fields, visual encodings)
2. Optionally: a data profile of the underlying data

Your job: propose a Power BI dashboard that PRESERVES the original Tableau intent
while improving the design where appropriate.

MIGRATION RULES:
- Map every Tableau worksheet to at least one Power BI visual. Do not drop visuals silently.
- Convert Tableau calculated fields to DAX measures. If exact conversion is impossible, set confidence < 0.7 and explain in "why".
- Preserve filter configurations.
- If Tableau uses LOD expressions (FIXED, INCLUDE, EXCLUDE), translate to CALCULATE + ALLEXCEPT or SUMMARIZE patterns. Flag these with lower confidence.
- Note what you changed from the original and why in design_notes.
- If a Tableau visual type has no direct PBI equivalent, pick the closest and explain.

Output ONLY valid JSON. No markdown fences. No commentary outside the JSON."""


# ------------------------------------------------------------------ #
#  Engine                                                              #
# ------------------------------------------------------------------ #

class DashboardIntelligence:
    """
    Core intelligence engine. Takes extracted content, returns a DashboardProposal.
    The proposal is SHOWN TO THE USER before any PBIP is built.
    NEVER builds anything -- only proposes.
    """

    MODEL = "claude-opus-4-20250514"
    FALLBACK_MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 16384
    MAX_RETRIES = 3

    def __init__(self):
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"[OK] Dashboard Intelligence Engine ready (model: {self.MODEL})")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def propose(
        self,
        data_profile: dict,
        user_request: str = "",
        audience: str = "executive",
        job_id: str = "",
    ) -> DashboardProposal:
        """Propose a dashboard from a data profile.

        Args:
            data_profile: dict from DataAnalyzer.analyze().
            user_request: Optional natural language request.
            audience: "executive", "analyst", or "operational".
            job_id: Optional job ID for tracking.

        Returns:
            DashboardProposal ready for user review.
        """
        job_id = job_id or str(uuid.uuid4())
        table_name = data_profile.get("table_name", "Data")
        column_names = {col["name"] for col in data_profile.get("columns", [])}

        user_message = self._build_profile_message(
            data_profile, user_request, audience, table_name
        )

        print(f"\n[PROPOSE] Sending data profile to Claude ({self.MODEL})...")
        print(f"          Table: {table_name} "
              f"({data_profile.get('row_count', '?')} rows, "
              f"{data_profile.get('column_count', '?')} cols)")
        if user_request:
            print(f"          Request: {user_request[:80]}")

        raw_spec = self._call_with_retries(INTELLIGENCE_SYSTEM_PROMPT, user_message)
        warnings = self._validate_columns(raw_spec, column_names)
        if warnings:
            for w in warnings:
                print(f"          [WARN] {w}")

        return self._spec_to_proposal(raw_spec, job_id)

    def propose_from_tableau(
        self,
        tableau_extract: dict,
        data_profile: Optional[dict] = None,
        user_request: str = "",
        job_id: str = "",
    ) -> DashboardProposal:
        """Propose a Power BI dashboard from a Tableau workbook extract.

        Args:
            tableau_extract: dict with keys like "worksheets", "datasources",
                             "calculated_fields", "parameters", "dashboards".
            data_profile: Optional data profile of the underlying data.
            user_request: Optional additional instructions.
            job_id: Optional job ID for tracking.

        Returns:
            DashboardProposal preserving Tableau intent.
        """
        job_id = job_id or str(uuid.uuid4())
        column_names = set()
        if data_profile:
            column_names = {col["name"] for col in data_profile.get("columns", [])}

        user_message = self._build_tableau_message(
            tableau_extract, data_profile, user_request
        )

        print(f"\n[PROPOSE] Sending Tableau extract to Claude ({self.MODEL})...")
        ws_count = len(tableau_extract.get("worksheets", []))
        ds_count = len(tableau_extract.get("datasources", []))
        print(f"          Worksheets: {ws_count} | Datasources: {ds_count}")

        raw_spec = self._call_with_retries(TABLEAU_MIGRATION_PROMPT, user_message)
        if column_names:
            warnings = self._validate_columns(raw_spec, column_names)
            if warnings:
                for w in warnings:
                    print(f"          [WARN] {w}")

        return self._spec_to_proposal(raw_spec, job_id)

    def propose_from_description(
        self,
        description: str,
        domain: str = "",
        audience: str = "executive",
        job_id: str = "",
    ) -> DashboardProposal:
        """Propose a dashboard from a text description only (no data yet).

        Args:
            description: Natural language description of the desired dashboard.
            domain: Optional domain hint.
            audience: Target audience.
            job_id: Optional job ID for tracking.

        Returns:
            DashboardProposal with synthetic_data_needed=True.
        """
        job_id = job_id or str(uuid.uuid4())

        user_message = (
            f"DASHBOARD REQUEST:\n{description}\n\n"
            f"TARGET AUDIENCE: {audience}\n"
        )
        if domain:
            user_message += f"DOMAIN: {domain}\n"
        user_message += (
            "\nNo data has been provided yet. Design the dashboard and describe "
            "what synthetic data should be generated for prototyping."
        )

        print(f"\n[PROPOSE] Description-only mode -- no data provided")
        print(f"          Request: {description[:80]}")

        raw_spec = self._call_with_retries(DESCRIPTION_ONLY_PROMPT, user_message)
        return self._spec_to_proposal(raw_spec, job_id)

    def refine(
        self,
        proposal: DashboardProposal,
        feedback: str,
        data_profile: Optional[dict] = None,
    ) -> DashboardProposal:
        """Refine an existing proposal based on user feedback.

        Args:
            proposal: The current DashboardProposal.
            feedback: User's feedback / change requests.
            data_profile: Optional data profile for column validation.

        Returns:
            Updated DashboardProposal.
        """
        column_names = set()
        if data_profile:
            column_names = {col["name"] for col in data_profile.get("columns", [])}

        user_message = (
            f"CURRENT PROPOSAL:\n{json.dumps(proposal.to_dict(), indent=2, default=str)}\n\n"
            f"USER FEEDBACK:\n{feedback}\n\n"
            "Apply the user's feedback and return the updated proposal JSON. "
            "Keep everything the user did not mention. Change only what they asked for."
        )

        print(f"\n[REFINE] Applying user feedback...")
        print(f"         Feedback: {feedback[:80]}")

        raw_spec = self._call_with_retries(INTELLIGENCE_SYSTEM_PROMPT, user_message)
        if column_names:
            warnings = self._validate_columns(raw_spec, column_names)
            if warnings:
                for w in warnings:
                    print(f"         [WARN] {w}")

        return self._spec_to_proposal(raw_spec, proposal.job_id)

    # ------------------------------------------------------------------ #
    #  Claude API call with retries + self-healing                         #
    # ------------------------------------------------------------------ #

    def _call_with_retries(self, system_prompt: str, user_message: str) -> dict:
        """Call Claude, parse JSON, retry on failure. Returns parsed dict."""
        raw_text = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_text = self._call_claude(system_prompt, user_message)
                spec = self._parse_json(raw_text)
                print(f"[OK] Proposal received on attempt {attempt}")
                return spec

            except json.JSONDecodeError as e:
                print(f"         [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"JSON parse failed: {e}")
                if attempt < self.MAX_RETRIES and raw_text:
                    raw_text = self._self_heal(raw_text, str(e))
                    try:
                        spec = self._parse_json(raw_text)
                        print(f"[OK] Proposal received (self-healed)")
                        return spec
                    except json.JSONDecodeError:
                        pass
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"         Waiting {wait}s before retry...")
                    time.sleep(wait)

            except anthropic.APIError as e:
                print(f"         [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"API error: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"         Waiting {wait}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to get valid proposal after {self.MAX_RETRIES} attempts"
        )

    def _call_claude(self, system_prompt: str, user_message: str) -> str:
        """Single Claude API call using streaming. Returns raw text."""
        t0 = time.time()
        collected_text = []
        tokens_in = 0
        tokens_out = 0

        with self.client.messages.stream(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text_chunk in stream.text_stream:
                collected_text.append(text_chunk)
            response = stream.get_final_message()
            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens

        elapsed = time.time() - t0
        text = "".join(collected_text)
        print(f"         Response: {tokens_out} tokens out, "
              f"{tokens_in} tokens in, {elapsed:.1f}s")
        return text

    def _self_heal(self, broken_json: str, error_msg: str) -> str:
        """Ask Claude to fix broken JSON output."""
        print(f"         [HEAL] Asking Claude to fix its JSON...")
        fix_prompt = (
            f"Your previous response was invalid JSON. Error: {error_msg}\n\n"
            f"Here is the broken output:\n{broken_json[:8000]}\n\n"
            "Fix it and return ONLY the corrected JSON. No commentary."
        )
        return self._call_claude(
            "You fix broken JSON. Return ONLY valid JSON. No markdown. No explanation.",
            fix_prompt,
        )

    # ------------------------------------------------------------------ #
    #  Message builders                                                    #
    # ------------------------------------------------------------------ #

    def _build_profile_message(
        self, data_profile: dict, user_request: str, audience: str, table_name: str
    ) -> str:
        """Build the user message for data-profile-based proposals."""
        profile_json = json.dumps(data_profile, indent=2, default=str)
        parts = []

        if user_request:
            parts.append(f"USER REQUEST:\n{user_request}")

        parts.append(f"TARGET AUDIENCE: {audience}")
        parts.append(f"DATASET PROFILE:\n{profile_json}")
        parts.append(
            f"Use ONLY columns from this profile. "
            f"For DAX measures, reference the table as '{table_name}'."
        )
        return "\n\n".join(parts)

    def _build_tableau_message(
        self, tableau_extract: dict, data_profile: Optional[dict], user_request: str
    ) -> str:
        """Build the user message for Tableau migration proposals."""
        parts = []

        if user_request:
            parts.append(f"ADDITIONAL INSTRUCTIONS:\n{user_request}")

        # Tableau extract -- truncate large sections to stay within context
        extract_json = json.dumps(tableau_extract, indent=2, default=str)
        if len(extract_json) > 40000:
            # Summarize: keep worksheets + calculated fields, trim raw XML
            summary = {
                "worksheets": tableau_extract.get("worksheets", []),
                "datasources": [
                    {
                        "name": ds.get("name"),
                        "caption": ds.get("caption"),
                        "connection": ds.get("connection", {}),
                        "field_count": len(ds.get("fields", [])),
                        "fields_sample": ds.get("fields", [])[:20],
                    }
                    for ds in tableau_extract.get("datasources", [])
                ],
                "calculated_fields": tableau_extract.get("calculated_fields", []),
                "parameters": tableau_extract.get("parameters", []),
                "dashboard_count": len(tableau_extract.get("dashboards", [])),
            }
            extract_json = json.dumps(summary, indent=2, default=str)

        parts.append(f"TABLEAU WORKBOOK EXTRACT:\n{extract_json}")

        if data_profile:
            profile_json = json.dumps(data_profile, indent=2, default=str)
            table_name = data_profile.get("table_name", "Data")
            parts.append(f"UNDERLYING DATA PROFILE:\n{profile_json}")
            parts.append(
                f"For DAX measures, reference the table as '{table_name}'."
            )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  JSON parsing + validation                                           #
    # ------------------------------------------------------------------ #

    def _parse_json(self, raw_text: str) -> dict:
        """Extract and parse JSON from Claude's response."""
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()
        return json.loads(text)

    def _validate_columns(self, spec: dict, valid_columns: set) -> List[str]:
        """Check that all column references in the spec exist in the dataset.

        Returns a list of warning strings (empty if all OK).
        """
        if not valid_columns:
            return []

        warnings = []
        # Collect measure names so we don't flag them as missing columns
        measure_names = set()
        dm = spec.get("data_model", {})
        for m in dm.get("measures", []):
            if isinstance(m, dict):
                measure_names.add(m.get("name", ""))

        for visual in spec.get("visuals", []):
            if not isinstance(visual, dict):
                continue
            for role in ("x_field", "y_field", "color_field", "size_field"):
                field_name = visual.get(role)
                if not field_name or field_name == "null":
                    continue
                if field_name in measure_names:
                    continue
                if field_name not in valid_columns:
                    warnings.append(
                        f"Visual '{visual.get('id', '?')}' references "
                        f"'{field_name}' ({role}) -- not found in dataset"
                    )
        return warnings

    # ------------------------------------------------------------------ #
    #  Spec dict → DashboardProposal                                       #
    # ------------------------------------------------------------------ #

    def _spec_to_proposal(self, spec: dict, job_id: str) -> DashboardProposal:
        """Convert Claude's raw JSON dict into a typed DashboardProposal."""
        now = datetime.now(timezone.utc).isoformat()

        # Parse visuals
        visuals = []
        for v in spec.get("visuals", []):
            visuals.append(VisualSpec(
                id=v.get("id", f"v{len(visuals)+1}"),
                chart_type=v.get("chart_type", "barChart"),
                title=v.get("title", ""),
                x_field=v.get("x_field", ""),
                y_field=v.get("y_field", ""),
                aggregation=v.get("aggregation", "sum"),
                color_field=v.get("color_field"),
                size_field=v.get("size_field"),
                sort=v.get("sort", "value_desc"),
                top_n=v.get("top_n"),
                filters=v.get("filters", []),
                page=v.get("page", 0),
                confidence=v.get("confidence", 1.0),
                why=v.get("why", ""),
            ))

        # Parse pages
        pages = []
        for p in spec.get("pages", []):
            pages.append(PageSpec(
                name=p.get("name", f"Page {len(pages)+1}"),
                visuals=p.get("visual_ids", []),
                layout_hint=p.get("layout_hint", ""),
            ))

        # Parse data model
        dm_raw = spec.get("data_model", {})
        measures = []
        for m in dm_raw.get("measures", []):
            measures.append(MeasureSpec(
                name=m.get("name", ""),
                dax_expression=m.get("dax_expression", ""),
                description=m.get("description", ""),
                format_string=m.get("format_string", ""),
            ))
        relationships = []
        for r in dm_raw.get("relationships", []):
            relationships.append(RelationshipSpec(
                from_table=r.get("from_table", ""),
                from_field=r.get("from_field", ""),
                to_table=r.get("to_table", ""),
                to_field=r.get("to_field", ""),
                cardinality=r.get("cardinality", "many-to-one"),
            ))
        data_model = DataModelSpec(
            tables=dm_raw.get("tables", []),
            measures=measures,
            relationships=relationships,
        )

        # Parse global filters
        global_filters = spec.get("global_filters", spec.get("filters", []))

        return DashboardProposal(
            job_id=job_id,
            domain=spec.get("domain", "custom"),
            title=spec.get("title", "Dashboard"),
            subtitle=spec.get("subtitle", ""),
            audience=spec.get("audience", "executive"),
            business_questions=spec.get("business_questions", []),
            pages=pages,
            visuals=visuals,
            data_model=data_model,
            global_filters=global_filters,
            layout_grid=spec.get("layout_grid", {"rows": 3, "cols": 3, "canvas": "1280x720"}),
            color_theme=spec.get("color_theme", "professional_dark"),
            has_data=spec.get("has_data", True),
            synthetic_data_needed=spec.get("synthetic_data_needed", False),
            synthetic_data_domain=spec.get("synthetic_data_domain", ""),
            confidence_score=spec.get("confidence_score", 0.0),
            requires_user_input=spec.get("requires_user_input", []),
            design_notes=spec.get("design_notes", ""),
            created_at=now,
        )


# ------------------------------------------------------------------ #
#  CLI test                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Dashboard Intelligence Engine Test ===")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.resolve()
    profile_path = project_root / "output" / "data_profile.json"

    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            data_profile = json.load(f)

        engine = DashboardIntelligence()
        proposal = engine.propose(
            data_profile,
            user_request="Create an executive overview dashboard",
            audience="executive",
        )
        print("\n" + proposal.summary())
        print(f"\nFull proposal JSON saved to output/proposal.json")

        out_path = project_root / "output" / "proposal.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(proposal.to_dict(), f, indent=2, default=str)

    else:
        print(f"No data profile at {profile_path}")
        print("Running in description-only mode...\n")

        engine = DashboardIntelligence()
        proposal = engine.propose_from_description(
            "Sales performance dashboard for a retail company with 12 stores, "
            "tracking revenue, units sold, returns, and customer satisfaction by region and month",
            domain="sales",
            audience="executive",
        )
        print("\n" + proposal.summary())
