"""
Multi-brain consensus engine for DAX formula validation.
Dispatches the same Tableau formula to 5 LLMs, compares outputs,
and synthesizes a consensus DAX translation via Claude Opus judgment.

Execution order (CLAUDE.md canon):
  1. Free Ollama brains draft (qwen2.5-coder, deepseek-coder-v2, phi4)
  2. Paid brains if API keys present (Claude Opus, GPT-4o)
  3. Claude Opus judges all outputs, synthesizes best DAX
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
#  .env loader (home directory)
# ---------------------------------------------------------------------------

def _load_env():
    """Load key=value pairs from ~/.env into os.environ (setdefault)."""
    env_path = Path.home() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_GENERATE = f"{OLLAMA_BASE}/api/generate"
OLLAMA_TAGS = f"{OLLAMA_BASE}/api/tags"

# Target Ollama model names (substring match)
OLLAMA_TARGETS = ["deepseek-coder-v2", "qwen2.5-coder", "phi4", "llama3.1"]

# Judge uses full Opus reasoning
JUDGE_MODEL = "claude-opus-4-6"


# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

_TRANSLATION_PROMPT = """\
You are an expert in translating Tableau calculated fields to Power BI DAX measures.

## Task
Translate the Tableau formula below to a correct, production-ready DAX measure.

## Context
- Table name: {table_name}
- Available columns: {columns}
{rule_engine_section}

## Tableau Formula
```
{tableau_formula}
```

## DAX Rules (MUST follow)
1. Use `VAR` / `RETURN` pattern for readability when the expression is complex.
2. Fully qualify all column references: `'{table_name}'[ColumnName]`.
3. Do NOT use `EARLIER()` — use VAR to capture row context instead.
4. Replace Tableau LOD `{{FIXED dim : agg}}` with `CALCULATE(agg, ALLEXCEPT(table, dim))`.
5. Replace `IIF(cond, t, f)` with `IF(cond, t, f)`.
6. Replace `ZN(x)` with `IF(ISBLANK(x), 0, x)`.
7. Replace `COUNTD(x)` with `DISTINCTCOUNT('{table_name}'[x])`.
8. Replace `AVG(x)` with `AVERAGE('{table_name}'[x])`.
9. Ensure parentheses are balanced.
10. Do NOT include the measure name assignment (e.g., `Measure =`).

## Output Format
Respond with ONLY valid JSON, no markdown fences, no explanation:
{{
  "dax": "<the DAX expression only, no measure name>",
  "confidence": <float 0.0-1.0>,
  "notes": "<brief explanation of translation choices, max 2 sentences>",
  "warnings": ["<any caveats or manual review items>"]
}}
"""

_JUDGE_PROMPT = """\
You are Claude Opus, the final authority on DAX formula quality.

## Original Tableau Formula
```
{tableau_formula}
```

## Table Context
- Table: {table_name}
- Columns: {columns}

## Brain Outputs
{brain_outputs_section}

## Your Job
1. Evaluate each brain's DAX translation for correctness, completeness, and style.
2. Either select the best one or synthesize a better DAX combining the best elements.
3. Identify where brains AGREE (high confidence) and where they DISAGREE (investigate why).
4. Apply the same DAX rules:
   - VAR/RETURN for complex expressions
   - Fully qualified column refs: `'TableName'[Col]`
   - No EARLIER(), no Tableau syntax leaking through
   - Balanced parentheses

## Output Format
Respond with ONLY valid JSON, no markdown fences:
{{
  "best_dax": "<the final, best DAX expression>",
  "confidence": <float 0.0-1.0>,
  "winner": "<brain name or 'synthesis'>",
  "agreement_score": <float 0.0-1.0, how much brains agreed>,
  "reasoning": "<1-3 sentences explaining your choice>",
  "disagreements": ["<key point where brains differed>"],
  "per_brain": {{
    "<brain_name>": {{"score": <0-10>, "issue": "<main problem if any>"}}
  }}
}}
"""


# ---------------------------------------------------------------------------
#  Response parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract and parse JSON from a brain response, tolerating markdown fences."""
    if not text:
        return {}

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"dax": text.strip(), "confidence": 0.3, "notes": "Raw text fallback", "warnings": []}

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to recover common issues: trailing commas
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            # Last resort: return the raw DAX if it looks like code
            return {"dax": text.strip()[:500], "confidence": 0.2, "notes": "JSON parse failed", "warnings": ["Could not parse response JSON"]}


# ---------------------------------------------------------------------------
#  MultiBrainEngine
# ---------------------------------------------------------------------------

class MultiBrainEngine:
    """Dispatch Tableau→DAX translation to all available LLMs, judge consensus."""

    def __init__(self):
        self.ollama_models: list[str] = []
        self._detect_ollama()

    # ── Model detection ──

    def _detect_ollama(self):
        """Discover which target Ollama models are running."""
        try:
            resp = requests.get(OLLAMA_TAGS, timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            for target in OLLAMA_TARGETS:
                match = next((m for m in available if target in m), None)
                if match:
                    self.ollama_models.append(match)
        except Exception as exc:
            print(f"[MULTI_BRAIN] Ollama not reachable: {exc}")

    # ── Public API ──

    def translate_formula(
        self,
        tableau_formula: str,
        table_name: str,
        columns: list[str],
        rule_engine_result: dict | None = None,
    ) -> dict:
        """
        Dispatch to all available brains, judge, return consensus.

        Returns:
            {
              "best_dax": str,
              "confidence": float,
              "agreement_score": float,
              "winner": str,
              "reasoning": str,
              "per_brain": dict,
              "all_results": dict,
              "disagreements": list,
              "original": str,
            }
        """
        translation_prompt = self._build_translation_prompt(
            tableau_formula, table_name, columns, rule_engine_result
        )

        raw_results: dict[str, dict] = {}

        # ── Sequential Ollama dispatch (avoids OOM on 14GB RAM) ──
        for model in self.ollama_models:
            print(f"[MULTI-BRAIN] Calling Ollama model: {model}")
            try:
                raw_results[model] = self._call_ollama(model, translation_prompt)
            except Exception as exc:
                raw_results[model] = {"error": str(exc), "dax": "", "confidence": 0.0}
            time.sleep(5)  # let previous model unload from VRAM

        # ── Parallel paid API calls (network-bound) ──
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures: dict = {}

            if os.getenv("ANTHROPIC_API_KEY"):
                futures[pool.submit(self._call_claude, translation_prompt)] = "claude-opus"
            else:
                print("[MULTI-BRAIN] ANTHROPIC_API_KEY not set -- skipping Claude")

            if os.getenv("OPENAI_API_KEY"):
                futures[pool.submit(self._call_gpt4o, translation_prompt)] = "gpt-4o"
            else:
                print("[MULTI-BRAIN] OPENAI_API_KEY not set -- skipping GPT-4o")

            for future in as_completed(futures, timeout=90):
                brain = futures[future]
                try:
                    raw_results[brain] = future.result()
                except Exception as exc:
                    raw_results[brain] = {"error": str(exc), "dax": "", "confidence": 0.0}

        if not raw_results:
            return {
                "best_dax": "/* No brains available */",
                "confidence": 0.0,
                "agreement_score": 0.0,
                "winner": "none",
                "reasoning": "No LLMs responded.",
                "per_brain": {},
                "all_results": {},
                "disagreements": [],
                "original": tableau_formula,
            }

        # ── Judge ──
        consensus = self._judge(tableau_formula, table_name, columns, raw_results)
        consensus["all_results"] = raw_results
        consensus["original"] = tableau_formula
        return consensus

    # ── Prompt builders ──

    def _build_translation_prompt(
        self,
        formula: str,
        table_name: str,
        columns: list[str],
        rule_engine_result: dict | None,
    ) -> str:
        col_list = ", ".join(columns[:40]) if columns else "unknown"
        if len(columns) > 40:
            col_list += f" ... (+{len(columns)-40} more)"

        if rule_engine_result and rule_engine_result.get("dax"):
            re_conf = rule_engine_result.get("confidence", 0)
            re_section = (
                f"\n## Rule Engine Baseline (confidence: {re_conf:.0%})\n"
                f"The deterministic rule engine produced this translation:\n"
                f"```\n{rule_engine_result['dax']}\n```\n"
                f"You may improve or correct it — do not just echo it back.\n"
            )
        else:
            re_section = ""

        return _TRANSLATION_PROMPT.format(
            table_name=table_name,
            columns=col_list,
            rule_engine_section=re_section,
            tableau_formula=formula,
        )

    # ── Brain callers ──

    def _call_ollama(self, model: str, prompt: str) -> dict:
        start = time.time()
        try:
            resp = requests.post(
                OLLAMA_GENERATE,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=90,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            result = _extract_json(text)
            result["_latency_s"] = round(time.time() - start, 1)
            result["_model"] = model
            return result
        except requests.Timeout:
            return {"error": "timeout", "dax": "", "confidence": 0.0, "_model": model}
        except Exception as exc:
            return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": model}

    def _call_claude(self, prompt: str) -> dict:
        start = time.time()
        try:
            from anthropic import Anthropic
            client = Anthropic()
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            result = _extract_json(text)
            result["_latency_s"] = round(time.time() - start, 1)
            result["_model"] = "claude-opus"
            return result
        except Exception as exc:
            return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": "claude-opus"}

    def _call_gpt4o(self, prompt: str) -> dict:
        start = time.time()
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            result = _extract_json(text)
            result["_latency_s"] = round(time.time() - start, 1)
            result["_model"] = "gpt-4o"
            return result
        except Exception as exc:
            return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": "gpt-4o"}

    # ── Judge ──

    def _judge(
        self,
        tableau_formula: str,
        table_name: str,
        columns: list[str],
        all_results: dict[str, dict],
    ) -> dict:
        """
        Use Claude Opus (via API if available, else local heuristic) to judge.
        Falls back to highest-confidence brain if no API key.
        """
        # Build brain outputs section
        lines = []
        for brain, res in all_results.items():
            if res.get("error"):
                lines.append(f"### {brain}\nERROR: {res['error']}\n")
            else:
                dax = res.get("dax", "")
                conf = res.get("confidence", 0.0)
                notes = res.get("notes", "")
                warnings = res.get("warnings", [])
                lines.append(
                    f"### {brain} (confidence: {conf:.0%})\n"
                    f"DAX:\n```\n{dax}\n```\n"
                    f"Notes: {notes}\n"
                    f"Warnings: {', '.join(warnings) if warnings else 'none'}\n"
                )
        brain_outputs_section = "\n".join(lines)

        col_list = ", ".join(columns[:40]) if columns else "unknown"
        judge_prompt = _JUDGE_PROMPT.format(
            tableau_formula=tableau_formula,
            table_name=table_name,
            columns=col_list,
            brain_outputs_section=brain_outputs_section,
        )

        # Try Anthropic API
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import Anthropic
                client = Anthropic()
                resp = client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": judge_prompt}],
                )
                judgment = _extract_json(resp.content[0].text)
                if judgment.get("best_dax"):
                    return judgment
            except Exception as exc:
                print(f"[MULTI_BRAIN] Judge API call failed: {exc}")

        # Fallback: try Ollama judge (qwen or deepseek)
        for model in self.ollama_models:
            if "deepseek" in model or "qwen" in model:
                try:
                    resp = requests.post(
                        OLLAMA_GENERATE,
                        json={"model": model, "prompt": judge_prompt, "stream": False},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    judgment = _extract_json(resp.json().get("response", ""))
                    if judgment.get("best_dax"):
                        return judgment
                except Exception:
                    pass

        # Final fallback: pick highest-confidence brain
        return self._heuristic_judge(all_results)

    def _heuristic_judge(self, all_results: dict[str, dict]) -> dict:
        """Fallback judge: pick the brain with the highest confidence DAX."""
        valid = {k: v for k, v in all_results.items() if v.get("dax") and not v.get("error")}
        if not valid:
            return {
                "best_dax": "/* Translation failed */",
                "confidence": 0.0,
                "agreement_score": 0.0,
                "winner": "none",
                "reasoning": "All brains failed or returned empty results.",
                "disagreements": [],
                "per_brain": {},
            }

        # Check agreement: count unique normalized DAX outputs
        dax_outputs = [v["dax"].strip() for v in valid.values()]
        unique_dax = set(dax_outputs)
        agreement = 1.0 - (len(unique_dax) - 1) / max(len(dax_outputs), 1)

        best_brain = max(valid, key=lambda k: valid[k].get("confidence", 0.0))
        best = valid[best_brain]

        return {
            "best_dax": best.get("dax", ""),
            "confidence": best.get("confidence", 0.5),
            "agreement_score": round(agreement, 2),
            "winner": best_brain,
            "reasoning": f"Heuristic: picked highest confidence brain ({best_brain}). {len(unique_dax)} unique translations from {len(valid)} brains.",
            "disagreements": list(unique_dax)[:3] if len(unique_dax) > 1 else [],
            "per_brain": {k: {"score": round(v.get("confidence", 0) * 10, 1), "issue": v.get("notes", "")} for k, v in valid.items()},
        }


# ---------------------------------------------------------------------------
#  Batch helper for pipeline integration
# ---------------------------------------------------------------------------

def run_batch(
    formulas: dict[str, str],
    table_name: str,
    columns: list[str],
    rule_engine_results: dict[str, dict] | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Translate a batch of {field_name: tableau_formula} using MultiBrainEngine.

    Args:
        formulas: dict of field_name -> tableau_formula
        table_name: the target Power BI table name
        columns: list of available column names
        rule_engine_results: optional dict of field_name -> rule_engine_result
        verbose: print progress

    Returns:
        dict of field_name -> consensus result
    """
    engine = MultiBrainEngine()
    if verbose:
        print(f"[MULTI_BRAIN] Initialized with {len(engine.ollama_models)} Ollama models: {engine.ollama_models}")

    results = {}
    for i, (name, formula) in enumerate(formulas.items(), 1):
        if verbose:
            print(f"[MULTI_BRAIN] ({i}/{len(formulas)}) Translating: {name!r}")
        re_result = (rule_engine_results or {}).get(name)
        results[name] = engine.translate_formula(formula, table_name, columns, re_result)
        if verbose:
            r = results[name]
            print(f"  → winner={r.get('winner')}, conf={r.get('confidence', 0):.0%}, agreement={r.get('agreement_score', 0):.0%}")
            if r.get("best_dax"):
                preview = r["best_dax"][:80].replace("\n", " ")
                print(f"  → DAX: {preview}...")

    return results


# ---------------------------------------------------------------------------
#  Convenience wrapper (importable as top-level function)
# ---------------------------------------------------------------------------

def dispatch_multi_brain(
    tableau_formula: str,
    table_name: str = "Data",
    columns: list = None,
    rule_engine_result: dict = None,
    timeout: int = 90,
) -> dict:
    """Convenience wrapper around MultiBrainEngine.translate_formula().

    Returns the consensus dict with keys:
        best_dax, confidence, agreement_score, winner, reasoning,
        per_brain, all_results, disagreements, original
    """
    engine = MultiBrainEngine()
    result = engine.translate_formula(
        tableau_formula=tableau_formula,
        table_name=table_name,
        columns=columns or [],
        rule_engine_result=rule_engine_result,
    )
    # Normalize response keys for callers expecting 'responses'/'consensus'
    return {
        "responses": result.get("all_results", {}),
        "consensus": {
            "dax": result.get("best_dax", ""),
            "confidence": result.get("confidence", 0.0),
            "agreement_score": result.get("agreement_score", 0.0),
            "winner": result.get("winner", ""),
            "reasoning": result.get("reasoning", ""),
        },
        "judge_model": JUDGE_MODEL,
        **result,
    }
