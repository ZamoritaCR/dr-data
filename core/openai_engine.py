"""
OpenAI Engine - Converts Claude's dashboard spec into full Power BI configuration.
Generates report_layout (report.json) and tmdl_model (model.bim) structures.
"""

import sys
import json
import time
import re
from pathlib import Path

from openai import OpenAI

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config.settings import _get_secret
from config.prompts import OPENAI_PBIP_SYSTEM_PROMPT


class OpenAIEngine:
    """Convert a dashboard spec into full Power BI configuration via GPT-4."""

    MODEL = "gpt-4o"
    MAX_TOKENS = 16384
    MAX_RETRIES = 3

    def __init__(self):
        api_key = _get_secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        print(f"[OK] OpenAI engine ready (model: {self.MODEL})")

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def generate_pbip_config(self, dashboard_spec, data_profile):
        """Convert Claude's dashboard spec into full Power BI config.

        Args:
            dashboard_spec: dict from ClaudeInterpreter.interpret().
            data_profile: dict from DataAnalyzer.analyze().

        Returns:
            dict with keys: report_layout, tmdl_model
        """
        user_message = self._build_user_message(dashboard_spec, data_profile)

        page_count = len(dashboard_spec.get("pages", []))
        visual_count = sum(
            len(p.get("visuals", [])) for p in dashboard_spec.get("pages", [])
        )
        print(f"\n[CALL] Sending to OpenAI ({self.MODEL})...")
        print(f"       Spec: {page_count} pages, {visual_count} visuals")

        raw_text = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_text = self._call_openai(
                    OPENAI_PBIP_SYSTEM_PROMPT, user_message
                )
                config = self._parse_json(raw_text)
                self._validate_structure(config)
                print(f"[OK] PBIP config received on attempt {attempt}")
                return config

            except json.JSONDecodeError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"JSON parse failed: {e}")
                if attempt < self.MAX_RETRIES and raw_text:
                    raw_text = self._self_heal(raw_text, str(e))
                    try:
                        config = self._parse_json(raw_text)
                        self._validate_structure(config)
                        print(f"[OK] PBIP config received (self-healed)")
                        return config
                    except (json.JSONDecodeError, ValueError):
                        pass
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"       Waiting {wait}s before retry...")
                    time.sleep(wait)

            except ValueError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"Validation failed: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"       Waiting {wait}s before retry...")
                    time.sleep(wait)

            except Exception as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"API error: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"       Waiting {wait}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to get valid PBIP config after {self.MAX_RETRIES} attempts"
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_user_message(self, dashboard_spec, data_profile):
        """Combine dashboard spec + data profile into the prompt."""
        spec_json = json.dumps(dashboard_spec, indent=2, default=str)
        profile_json = json.dumps(data_profile, indent=2, default=str)
        return (
            f"DASHBOARD SPECIFICATION:\n{spec_json}\n\n"
            f"DATASET PROFILE:\n{profile_json}\n\n"
            "Generate the complete Power BI configuration JSON with "
            "report_layout and tmdl_model. Use ONLY columns from the dataset profile. "
            "Map all visuals from the specification to proper Power BI visualContainers "
            "on a 1280x720 canvas with no overlaps."
        )

    def _call_openai(self, system_prompt, user_message):
        """Make a single OpenAI API call and return the text response."""
        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        elapsed = time.time() - t0
        text = response.choices[0].message.content
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        print(f"       Response: {tokens_out} tokens out, "
              f"{tokens_in} tokens in, {elapsed:.1f}s")
        return text

    def _parse_json(self, raw_text):
        """Extract and parse JSON from OpenAI's response."""
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()
        return json.loads(text)

    def _self_heal(self, broken_json, error_msg):
        """Ask OpenAI to fix its own broken JSON output."""
        print(f"       [HEAL] Asking OpenAI to fix its JSON...")
        fix_prompt = (
            f"Your previous response was invalid JSON. Error: {error_msg}\n\n"
            f"Here is the broken output:\n{broken_json[:8000]}\n\n"
            "Fix it and return ONLY the corrected JSON. No commentary."
        )
        return self._call_openai(
            "You fix broken JSON. Return ONLY valid JSON. No markdown. No explanation.",
            fix_prompt,
        )

    def _validate_structure(self, config):
        """Basic structural validation of the PBIP config."""
        if "report_layout" not in config:
            raise ValueError("Missing 'report_layout' in config")
        if "tmdl_model" not in config:
            raise ValueError("Missing 'tmdl_model' in config")

        rl = config["report_layout"]
        sections = rl.get("sections", [])
        if not sections:
            raise ValueError("report_layout has no sections")

        for i, section in enumerate(sections):
            containers = section.get("visualContainers", [])
            if not containers:
                raise ValueError(f"Section {i} has no visualContainers")

        tm = config["tmdl_model"]
        tables = tm.get("tables", [])
        if not tables:
            raise ValueError("tmdl_model has no tables")


# ------------------------------------------------------------------ #
#  CLI test                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== OpenAI Engine Test ===")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.resolve()
    spec_path = project_root / "output" / "claude_dashboard_spec.json"
    profile_path = project_root / "output" / "data_profile.json"

    if not spec_path.exists():
        print(f"[ERROR] No dashboard spec at {spec_path}")
        sys.exit(1)
    if not profile_path.exists():
        print(f"[ERROR] No data profile at {profile_path}")
        sys.exit(1)

    with open(spec_path, "r", encoding="utf-8") as f:
        dashboard_spec = json.load(f)
    with open(profile_path, "r", encoding="utf-8") as f:
        data_profile = json.load(f)

    engine = OpenAIEngine()
    config = engine.generate_pbip_config(dashboard_spec, data_profile)

    out_path = project_root / "output" / "openai_pbip_config.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Saved to {out_path}")

    # Summary
    rl = config["report_layout"]
    sections = rl.get("sections", [])
    print(f"\nReport layout: {len(sections)} pages")
    for s in sections:
        vc = s.get("visualContainers", [])
        print(f"  - {s.get('displayName', s.get('name', '?'))}: {len(vc)} visuals")

    tm = config["tmdl_model"]
    tables = tm.get("tables", [])
    print(f"\nData model: {len(tables)} tables")
    for t in tables:
        cols = t.get("columns", [])
        measures = t.get("measures", [])
        print(f"  - {t.get('name', '?')}: {len(cols)} columns, {len(measures)} measures")
