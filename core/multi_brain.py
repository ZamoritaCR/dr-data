"""
Multi-brain consensus engine for DAX formula validation.
7+ LLMs across 3 tiers. Claude Opus judges all outputs.

Tier 1 - FREE LOCAL (Ollama, sequential for RAM):
    deepseek-coder-v2, qwen2.5-coder, phi4, llama3.1:8b
Tier 2 - CHEAP API (parallel, network-bound):
    Gemini 2.5 Flash, Grok 3 Fast
Tier 3 - PREMIUM API (parallel, network-bound):
    Claude Opus 4.6, GPT-4o, Gemini 2.5 Pro
Judge: Claude Opus 4.6
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  .env loader
# ---------------------------------------------------------------------------

def _load_env():
    """Load key=value pairs from ~/.env into os.environ (setdefault)."""
    for env_path in [Path.home() / ".env", Path("/home/zamoritacr/.env")]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            break

_load_env()


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_TARGETS = ["deepseek-coder-v2", "qwen2.5-coder", "phi4", "llama3.1"]
OLLAMA_PAUSE = 5  # seconds between sequential calls (RAM management)

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
3. Do NOT use `EARLIER()` -- use VAR to capture row context instead.
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
You are judging outputs from {brain_count} LLMs across 3 tiers (local Ollama, cheap API, premium API).

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
4. Weight premium brains (Claude, GPT-4o, Gemini Pro) higher but do not ignore correct local answers.
5. Apply the same DAX rules:
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

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"dax": text.strip(), "confidence": 0.3, "notes": "Raw text fallback", "warnings": []}

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            return {"dax": text.strip()[:500], "confidence": 0.2, "notes": "JSON parse failed", "warnings": ["Could not parse response JSON"]}


# ---------------------------------------------------------------------------
#  Brain callers (module-level functions)
# ---------------------------------------------------------------------------

def _call_ollama(model: str, prompt: str, timeout: int = 120) -> dict:
    """Call a single Ollama model. Sequential use only (RAM constraint)."""
    start = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
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


def _call_gemini(model: str, prompt: str) -> dict:
    """Call Google Gemini API. Works for both gemini-2.5-flash and gemini-2.5-pro."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"error": "GEMINI_API_KEY not set", "dax": "", "confidence": 0.0, "_model": model}
    start = time.time()
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        response = m.generate_content(prompt)
        result = _extract_json(response.text)
        result["_latency_s"] = round(time.time() - start, 1)
        result["_model"] = model
        return result
    except Exception as exc:
        logger.warning("[BRAIN] Gemini %s failed: %s", model, exc)
        return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": model}


def _call_grok(prompt: str) -> dict:
    """Call xAI Grok via OpenAI-compatible endpoint."""
    api_key = os.environ.get("XAI_API_KEY", "")
    if not api_key:
        return {"error": "XAI_API_KEY not set", "dax": "", "confidence": 0.0, "_model": "grok-3-fast"}
    start = time.time()
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "grok-3-fast",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        result = _extract_json(text)
        result["_latency_s"] = round(time.time() - start, 1)
        result["_model"] = "grok-3-fast"
        return result
    except Exception as exc:
        logger.warning("[BRAIN] Grok failed: %s", exc)
        return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": "grok-3-fast"}


def _call_claude(prompt: str, model: str = JUDGE_MODEL) -> dict:
    """Call Anthropic Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set", "dax": "", "confidence": 0.0, "_model": model}
    start = time.time()
    try:
        from anthropic import Anthropic
        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        result = _extract_json(text)
        result["_latency_s"] = round(time.time() - start, 1)
        result["_model"] = model
        return result
    except Exception as exc:
        logger.warning("[BRAIN] Claude %s failed: %s", model, exc)
        return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": model}


def _call_openai(prompt: str) -> dict:
    """Call OpenAI GPT-4o."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set", "dax": "", "confidence": 0.0, "_model": "gpt-4o"}
    start = time.time()
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        result = _extract_json(text)
        result["_latency_s"] = round(time.time() - start, 1)
        result["_model"] = "gpt-4o"
        return result
    except Exception as exc:
        logger.warning("[BRAIN] GPT-4o failed: %s", exc)
        return {"error": str(exc), "dax": "", "confidence": 0.0, "_model": "gpt-4o"}


# ---------------------------------------------------------------------------
#  MultiBrainEngine
# ---------------------------------------------------------------------------

class MultiBrainEngine:
    """Dispatch Tableau->DAX translation to 7+ LLMs across 3 tiers, judge consensus."""

    def __init__(self):
        self.ollama_models: list[str] = []
        self._detect_ollama()

    def _detect_ollama(self):
        """Discover which target Ollama models are available."""
        try:
            resp = requests.get(OLLAMA_TAGS_URL, timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            for target in OLLAMA_TARGETS:
                match = next((m for m in available if target in m), None)
                if match:
                    self.ollama_models.append(match)
        except Exception as exc:
            print(f"[BRAIN] Ollama not reachable: {exc}")

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

        Returns dict with keys: best_dax, confidence, agreement_score, winner,
        reasoning, per_brain, all_results, disagreements, original, timings
        """
        prompt = self._build_translation_prompt(
            tableau_formula, table_name, columns, rule_engine_result
        )

        results: dict[str, dict] = {}
        timings: dict[str, float] = {}

        # ── TIER 1: Ollama local models (sequential, RAM-bound) ──
        print(f"[BRAIN] Tier 1: {len(self.ollama_models)} Ollama models (sequential)...")
        for i, model in enumerate(self.ollama_models):
            t0 = time.time()
            print(f"  [{i+1}/{len(self.ollama_models)}] {model}...", end="", flush=True)
            results[f"ollama/{model}"] = _call_ollama(model, prompt)
            elapsed = round(time.time() - t0, 1)
            timings[f"ollama/{model}"] = elapsed
            ok = "OK" if results[f"ollama/{model}"].get("dax") else "FAIL"
            print(f" {elapsed}s [{ok}]")
            if i < len(self.ollama_models) - 1:
                time.sleep(OLLAMA_PAUSE)

        # ── TIER 2 + 3: API calls in parallel (network-bound) ──
        api_tasks = {}

        # Tier 2: cheap
        api_tasks["gemini/2.5-flash"] = lambda: _call_gemini("gemini-2.5-flash", prompt)
        api_tasks["grok/3-fast"] = lambda: _call_grok(prompt)

        # Tier 3: premium
        api_tasks["claude/opus"] = lambda: _call_claude(prompt, JUDGE_MODEL)
        api_tasks["openai/gpt-4o"] = lambda: _call_openai(prompt)
        api_tasks["gemini/2.5-pro"] = lambda: _call_gemini("gemini-2.5-pro", prompt)

        print(f"[BRAIN] Tier 2+3: {len(api_tasks)} API models (parallel)...")
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(fn): name for name, fn in api_tasks.items()}
            for future in as_completed(futures, timeout=120):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    results[name] = {"error": str(exc), "dax": "", "confidence": 0.0}
                latency = results[name].get("_latency_s", 0)
                timings[name] = latency
                ok = "OK" if results[name].get("dax") else "SKIP"
                err = results[name].get("error", "")
                if err and "not set" in err:
                    ok = "NO KEY"
                elif err:
                    ok = "FAIL"
                print(f"  {name}: {latency}s [{ok}]")

        # ── Summary ──
        valid = {k: v for k, v in results.items() if v.get("dax") and not v.get("error")}
        failed = {k: v.get("error", "no dax") for k, v in results.items() if k not in valid}

        if failed:
            print(f"[BRAIN] Skipped/failed ({len(failed)}): {', '.join(failed.keys())}")
        print(f"[BRAIN] Got {len(valid)} valid DAX translations from {len(results)} brains")

        if not valid:
            return {
                "best_dax": "/* No brains available */",
                "confidence": 0.0,
                "agreement_score": 0.0,
                "winner": "none",
                "reasoning": "No LLMs responded with valid DAX.",
                "per_brain": {},
                "all_results": results,
                "disagreements": [],
                "original": tableau_formula,
                "timings": timings,
            }

        # ── JUDGE: Claude Opus synthesizes ──
        consensus = self._judge(tableau_formula, table_name, columns, results)
        consensus["all_results"] = results
        consensus["original"] = tableau_formula
        consensus["timings"] = timings
        return consensus

    # ── Prompt builder ──

    def _build_translation_prompt(
        self,
        formula: str,
        table_name: str,
        columns: list[str],
        rule_engine_result: dict | None,
    ) -> str:
        col_list = ", ".join(columns[:40]) if columns else "unknown"
        if columns and len(columns) > 40:
            col_list += f" ... (+{len(columns)-40} more)"

        if rule_engine_result and rule_engine_result.get("dax"):
            re_conf = rule_engine_result.get("confidence", 0)
            re_section = (
                f"\n## Rule Engine Baseline (confidence: {re_conf:.0%})\n"
                f"The deterministic rule engine produced this translation:\n"
                f"```\n{rule_engine_result['dax']}\n```\n"
                f"You may improve or correct it -- do not just echo it back.\n"
            )
        else:
            re_section = ""

        return _TRANSLATION_PROMPT.format(
            table_name=table_name,
            columns=col_list,
            rule_engine_section=re_section,
            tableau_formula=formula,
        )

    # ── Judge ──

    def _judge(
        self,
        tableau_formula: str,
        table_name: str,
        columns: list[str],
        all_results: dict[str, dict],
    ) -> dict:
        """Use Claude Opus to judge all brain outputs. Falls back to heuristic."""
        # Build brain outputs section for the judge prompt
        lines = []
        valid_count = 0
        for brain, res in all_results.items():
            if res.get("error"):
                continue
            dax = res.get("dax", "")
            if not dax:
                continue
            valid_count += 1
            conf = res.get("confidence", 0.0)
            notes = res.get("notes", "")
            warnings = res.get("warnings", [])
            lines.append(
                f"### {brain} (confidence: {conf})\n"
                f"DAX:\n```\n{dax}\n```\n"
                f"Notes: {notes}\n"
                f"Warnings: {', '.join(warnings) if warnings else 'none'}\n"
            )
        brain_outputs_section = "\n".join(lines)

        if not lines:
            return self._heuristic_judge(all_results)

        col_list = ", ".join(columns[:40]) if columns else "unknown"
        judge_prompt = _JUDGE_PROMPT.format(
            tableau_formula=tableau_formula,
            table_name=table_name,
            columns=col_list,
            brain_outputs_section=brain_outputs_section,
            brain_count=valid_count,
        )

        # Try Claude Opus API
        if os.environ.get("ANTHROPIC_API_KEY"):
            print(f"[BRAIN] Judge: Claude Opus evaluating {valid_count} translations...")
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
                    judgment.setdefault("judge_model", JUDGE_MODEL)
                    return judgment
            except Exception as exc:
                print(f"[BRAIN] Judge API failed: {exc}")

        # Fallback: Ollama judge (qwen or deepseek)
        for model in self.ollama_models:
            if "deepseek" in model or "qwen" in model:
                print(f"[BRAIN] Fallback judge: {model}")
                try:
                    resp = requests.post(
                        OLLAMA_URL,
                        json={"model": model, "prompt": judge_prompt, "stream": False},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    judgment = _extract_json(resp.json().get("response", ""))
                    if judgment.get("best_dax"):
                        judgment["judge_model"] = f"ollama/{model}"
                        return judgment
                except Exception:
                    pass

        # Final fallback: heuristic
        return self._heuristic_judge(all_results)

    def _heuristic_judge(self, all_results: dict[str, dict]) -> dict:
        """Fallback: pick the brain with the highest confidence DAX."""
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
                "judge_model": "heuristic",
            }

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
            "judge_model": "heuristic",
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
    """
    engine = MultiBrainEngine()
    if verbose:
        print(f"[BRAIN] Batch: {len(engine.ollama_models)} Ollama + API brains ready")

    results = {}
    for i, (name, formula) in enumerate(formulas.items(), 1):
        if verbose:
            print(f"[BRAIN] ({i}/{len(formulas)}) Translating: {name!r}")
        re_result = (rule_engine_results or {}).get(name)
        results[name] = engine.translate_formula(formula, table_name, columns, re_result)
        if verbose:
            r = results[name]
            print(f"  -> winner={r.get('winner')}, conf={r.get('confidence', 0):.0%}, agreement={r.get('agreement_score', 0):.0%}")
            if r.get("best_dax"):
                preview = r["best_dax"][:80].replace("\n", " ")
                print(f"  -> DAX: {preview}...")

    return results


# ---------------------------------------------------------------------------
#  Convenience wrapper (importable as top-level function)
# ---------------------------------------------------------------------------

def dispatch_multi_brain(
    tableau_formula: str,
    table_name: str = "Data",
    columns: list = None,
    rule_engine_result: dict = None,
    timeout: int = 120,
) -> dict:
    """Convenience wrapper around MultiBrainEngine.translate_formula().

    Returns dict with both the raw engine output and normalized
    'responses'/'consensus' keys for backward compatibility.
    """
    engine = MultiBrainEngine()
    result = engine.translate_formula(
        tableau_formula=tableau_formula,
        table_name=table_name,
        columns=columns or [],
        rule_engine_result=rule_engine_result,
    )
    return {
        "responses": result.get("all_results", {}),
        "consensus": {
            "dax": result.get("best_dax", ""),
            "confidence": result.get("confidence", 0.0),
            "agreement_score": result.get("agreement_score", 0.0),
            "winner": result.get("winner", ""),
            "reasoning": result.get("reasoning", ""),
        },
        "judge_model": result.get("judge_model", JUDGE_MODEL),
        "valid_count": len({k: v for k, v in result.get("all_results", {}).items() if v.get("dax") and not v.get("error")}),
        "failed_count": len({k: v for k, v in result.get("all_results", {}).items() if not v.get("dax") or v.get("error")}),
        "timings": result.get("timings", {}),
        **result,
    }
