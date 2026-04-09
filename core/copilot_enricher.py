"""
Copilot Enricher -- AI-powered analysis of Tableau-to-PBI conversions.

Two modes:
1. CopilotEnricher class: heuristic column descriptions + linguistic schema
   with synonyms (no AI, pure Python). Based on phi4 Ollama draft.
2. enrich() function: Ollama-powered conversion analysis with chart suggestions,
   missing KPIs, DAX tips, accessibility checks.

Graceful degradation: returns empty suggestions if Ollama is unavailable.
"""

import json
import logging
import requests

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# CopilotEnricher class (heuristic, no AI)                           #
# Drafted by phi4 via Ollama, reviewed + enhanced by Claude           #
# ------------------------------------------------------------------ #

class CopilotEnricher:
    """Heuristic column enrichment for Power BI linguistic schema."""

    # Common business term synonyms keyed by lowercase stem
    _SYNONYM_MAP = {
        "revenue": ["sales", "income", "earnings", "turnover"],
        "profit": ["gain", "net income", "benefit", "margin"],
        "cost": ["expense", "expenditure", "charge", "spend"],
        "quantity": ["count", "amount", "volume", "units"],
        "price": ["rate", "cost", "value", "unit price"],
        "discount": ["rebate", "reduction", "markdown"],
        "date": ["day", "time", "period", "timestamp"],
        "region": ["area", "territory", "zone", "geography"],
        "category": ["type", "class", "group", "segment"],
        "customer": ["client", "buyer", "account", "consumer"],
        "product": ["item", "good", "sku", "merchandise"],
        "order": ["transaction", "purchase", "sale"],
        "employee": ["staff", "worker", "associate"],
        "country": ["nation", "state", "territory"],
        "city": ["town", "municipality", "metro"],
    }

    def enrich(self, dataframe, column_names):
        """Classify columns and generate linguistic schema with synonyms.

        Args:
            dataframe: pandas DataFrame with the data.
            column_names: list of column name strings to analyze.

        Returns:
            dict with "descriptions" (col->role mapping) and
            "linguistic_schema" (synonyms per column).
        """
        import pandas as pd

        descriptions = {}
        linguistic_schema = {"synonyms": {}}

        for col in column_names:
            if col not in dataframe.columns:
                descriptions[col] = "unknown"
                continue

            series = dataframe[col]

            if self._is_date_column(series):
                descriptions[col] = "date"
            elif pd.api.types.is_numeric_dtype(series):
                descriptions[col] = "measure"
            elif self._is_low_cardinality(series):
                descriptions[col] = "dimension"
            else:
                descriptions[col] = "text"

            linguistic_schema["synonyms"][col] = self._generate_synonyms(col)

        return {
            "descriptions": descriptions,
            "linguistic_schema": linguistic_schema,
        }

    def _is_date_column(self, series):
        """Check if a series contains date-like values."""
        import pandas as pd
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        # Only try parsing string/object columns (not numeric)
        if pd.api.types.is_numeric_dtype(series):
            return False
        try:
            sample = series.dropna().head(20)
            if len(sample) == 0:
                return False
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return parsed.notna().sum() > len(sample) * 0.8
        except Exception:
            return False

    def _is_low_cardinality(self, series):
        """Return True if series has low cardinality (dimension-like).

        Uses absolute threshold for small datasets, ratio for large.
        """
        n = len(series)
        if n == 0:
            return False
        nunique = series.nunique()
        # Small datasets: anything under 50 unique values is a dimension
        if nunique <= 50:
            return True
        # Large datasets: less than 5% unique is a dimension
        return (nunique / n) < 0.05

    def _generate_synonyms(self, column_name):
        """Generate synonyms from column name using stem matching."""
        parts = column_name.lower().replace("-", "_").split("_")
        synonyms = set()

        for part in parts:
            part_clean = part.strip()
            if part_clean in self._SYNONYM_MAP:
                synonyms.update(self._SYNONYM_MAP[part_clean])
            # Partial match for compound names
            for key, syns in self._SYNONYM_MAP.items():
                if key in part_clean or part_clean in key:
                    synonyms.update(syns)
                    break

        return list(synonyms)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "deepseek-coder-v2"


def enrich(tableau_spec, pbip_config, model=None):
    """Analyze a conversion and return enrichment suggestions.

    Args:
        tableau_spec: dict from enhanced_tableau_parser.parse_twb().
        pbip_config: dict with report_layout, tmdl_model, or similar
            output from the conversion pipeline.
        model: Ollama model name (default: deepseek-coder-v2).

    Returns:
        dict with keys: chart_suggestions, missing_kpis, dax_tips,
        accessibility, naming, layout_tips, score, warnings.
    """
    empty = _empty_result()
    model = model or DEFAULT_MODEL

    prompt = _build_analysis_prompt(tableau_spec, pbip_config)
    if not prompt:
        empty["warnings"].append("Could not build analysis prompt from spec.")
        return empty

    raw = _call_ollama(prompt, model)
    if not raw:
        empty["warnings"].append("Ollama unavailable or returned empty response.")
        return empty

    result = _parse_enrichment_response(raw)
    return result


def _empty_result():
    """Return the empty enrichment result structure."""
    return {
        "chart_suggestions": [],
        "missing_kpis": [],
        "dax_tips": [],
        "accessibility": [],
        "naming": [],
        "layout_tips": [],
        "score": 0,
        "warnings": [],
    }


def _build_analysis_prompt(tableau_spec, pbip_config):
    """Build a concise prompt for Ollama analysis.

    Extracts only the key metadata to keep under 2000 tokens.
    """
    if not tableau_spec:
        return ""

    # Extract worksheet summary
    worksheets = tableau_spec.get("worksheets", [])
    ws_summary = []
    for ws in worksheets[:10]:  # Limit to 10 worksheets
        ws_summary.append({
            "name": ws.get("name", ""),
            "chart_type": ws.get("chart_type", ws.get("mark_type", "")),
            "measures": ws.get("measures", [])[:5],
            "dimensions": ws.get("dimensions", [])[:5],
            "filters": len(ws.get("filters", [])),
        })

    # Extract calculated fields
    calc_fields = []
    for cf in tableau_spec.get("calculated_fields", [])[:8]:
        calc_fields.append({
            "name": cf.get("name", ""),
            "formula": cf.get("formula", "")[:100],
        })

    # Extract dashboard info
    dashboards = []
    for d in tableau_spec.get("dashboards", [])[:3]:
        dashboards.append({
            "name": d.get("name", ""),
            "visual_count": len(d.get("worksheets_used", [])),
        })

    # Extract PBIP config summary
    pbip_summary = {}
    if pbip_config:
        if isinstance(pbip_config, dict):
            pbip_summary["page_count"] = pbip_config.get("page_count", 0)
            pbip_summary["field_audit"] = pbip_config.get("field_audit", {})
            pbip_summary["table_names"] = pbip_config.get("table_names", [])[:5]

    context = json.dumps({
        "worksheets": ws_summary,
        "calculated_fields": calc_fields,
        "dashboards": dashboards,
        "datasource_count": len(tableau_spec.get("datasources", [])),
        "pbip_output": pbip_summary,
    }, indent=None, default=str)

    prompt = f"""Analyze this Tableau-to-Power-BI conversion and return a JSON object with improvement suggestions.

CONVERSION METADATA:
{context}

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{{
  "chart_suggestions": [
    {{"worksheet": "name", "current_type": "bar", "suggested_type": "treemap", "reason": "..."}}
  ],
  "missing_kpis": [
    {{"name": "YoY Growth", "dax_formula": "DIVIDE(...)", "reason": "..."}}
  ],
  "dax_tips": [
    {{"field": "name", "current_dax": "...", "optimized_dax": "...", "reason": "..."}}
  ],
  "accessibility": [
    {{"issue": "Low contrast", "fix": "Use darker text color", "severity": "medium"}}
  ],
  "naming": [
    {{"current": "Sheet1", "suggested": "Sales Overview", "reason": "..."}}
  ],
  "layout_tips": [
    {{"tip": "...", "affected_visuals": ["name1"]}}
  ],
  "score": 75
}}

Rules:
- chart_suggestions: only suggest if current type is clearly wrong for the data
- missing_kpis: suggest 1-3 common KPIs based on the measures present
- dax_tips: only if calculated fields have obvious inefficiencies
- accessibility: check for contrast, color-only encoding, missing alt text
- naming: only flag generic names like Sheet1, Sheet2
- score: 0-100 based on overall conversion quality
- Keep each array to max 5 items"""

    return prompt


def _call_ollama(prompt, model, timeout=90):
    """POST to Ollama generate endpoint. Returns response text or empty string."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1500,
                },
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("response", "")
        else:
            logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:200])
            return ""
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not available at %s", OLLAMA_URL)
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out after %ds", timeout)
        return ""
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        return ""


def _parse_enrichment_response(raw_text):
    """Parse Ollama response into structured enrichment dict.

    Expects JSON. Falls back to empty result on parse error.
    """
    result = _empty_result()

    if not raw_text:
        return result

    # Try to extract JSON from response (may have markdown fences)
    text = raw_text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip().startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Find the JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        result["warnings"].append("Could not find JSON in Ollama response.")
        return result

    json_str = text[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        result["warnings"].append(f"JSON parse error: {e}")
        return result

    if not isinstance(parsed, dict):
        result["warnings"].append("Ollama response was not a JSON object.")
        return result

    # Validate and extract each field
    valid_keys = {
        "chart_suggestions", "missing_kpis", "dax_tips",
        "accessibility", "naming", "layout_tips", "score",
    }
    for key in valid_keys:
        val = parsed.get(key)
        if key == "score":
            if isinstance(val, (int, float)):
                result["score"] = max(0, min(100, int(val)))
        elif isinstance(val, list):
            # Validate each item is a dict
            clean = [item for item in val if isinstance(item, dict)]
            result[key] = clean[:5]  # Cap at 5 items

    return result
