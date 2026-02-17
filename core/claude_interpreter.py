"""
Claude Interpreter - Sends data profiles to Claude and gets back dashboard specs.
Handles retries, JSON parsing, self-healing, and column validation.
"""

import sys
import json
import time
import re
from pathlib import Path

import anthropic

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from config.settings import _get_secret
from config.prompts import INTERPRETER_SYSTEM_PROMPT, EXPLANATION_SYSTEM_PROMPT


class ClaudeInterpreter:
    """Send data profiles to Claude, get back dashboard specifications."""

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 8192
    MAX_RETRIES = 3

    def __init__(self):
        api_key = _get_secret("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"[OK] Claude interpreter ready (model: {self.MODEL})")

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def interpret(self, user_request, data_profile):
        """Generate a dashboard spec from a user request + data profile.

        Args:
            user_request: Natural language description of what the user wants.
            data_profile: dict from DataAnalyzer.analyze().

        Returns:
            dict: Parsed and validated dashboard specification.
        """
        user_message = self._build_user_message(user_request, data_profile)
        column_names = self._extract_column_names(data_profile)

        print(f"\n[CALL] Sending request to Claude ({self.MODEL})...")
        print(f"       Request: {user_request[:80]}...")
        print(f"       Profile: {data_profile['table_name']} "
              f"({data_profile['row_count']} rows, {data_profile['column_count']} cols)")

        # Attempt with retries
        raw_text = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw_text = self._call_claude(
                    INTERPRETER_SYSTEM_PROMPT, user_message
                )
                spec = self._parse_json(raw_text)
                warnings = self._validate_columns(spec, column_names)
                if warnings:
                    for w in warnings:
                        print(f"       [WARN] {w}")
                print(f"[OK] Dashboard spec received on attempt {attempt}")
                return spec

            except json.JSONDecodeError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"JSON parse failed: {e}")
                if attempt < self.MAX_RETRIES and raw_text:
                    # Ask Claude to fix its own output
                    raw_text = self._self_heal(raw_text, str(e))
                    try:
                        spec = self._parse_json(raw_text)
                        warnings = self._validate_columns(spec, column_names)
                        if warnings:
                            for w in warnings:
                                print(f"       [WARN] {w}")
                        print(f"[OK] Dashboard spec received (self-healed)")
                        return spec
                    except json.JSONDecodeError:
                        pass  # Fall through to retry
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"       Waiting {wait}s before retry...")
                    time.sleep(wait)

            except anthropic.APIError as e:
                print(f"       [RETRY {attempt}/{self.MAX_RETRIES}] "
                      f"API error: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = 2 ** attempt
                    print(f"       Waiting {wait}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to get valid dashboard spec after {self.MAX_RETRIES} attempts"
        )

    # ------------------------------------------------------------------ #
    #  Explanation generation                                              #
    # ------------------------------------------------------------------ #

    def explain(self, dashboard_spec, data_profile):
        """Generate a human-readable explanation of the dashboard design.

        Args:
            dashboard_spec: dict from interpret().
            data_profile: dict from DataAnalyzer.analyze().

        Returns:
            str: Markdown-formatted explanation.
        """
        user_message = (
            "Explain this dashboard design.\n\n"
            f"DATASET:\n{json.dumps(data_profile, indent=2, default=str)}\n\n"
            f"DASHBOARD SPEC:\n{json.dumps(dashboard_spec, indent=2, default=str)}"
        )

        print(f"\n[CALL] Requesting design explanation from Claude...")
        text = self._call_claude(EXPLANATION_SYSTEM_PROMPT, user_message)
        print(f"[OK] Explanation received ({len(text)} chars)")
        return text

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_user_message(self, user_request, data_profile):
        """Combine user request + data profile into the prompt."""
        profile_json = json.dumps(data_profile, indent=2, default=str)
        table_name = data_profile.get("table_name", "Data")
        return (
            f"USER REQUEST:\n{user_request}\n\n"
            f"DATASET PROFILE:\n{profile_json}\n\n"
            "Generate the complete dashboard specification JSON. "
            "Use ONLY columns that exist in the dataset profile above. "
            f"For DAX measures, reference the table as '{table_name}'."
        )

    def _call_claude(self, system_prompt, user_message):
        """Make a single Claude API call and return the text response."""
        t0 = time.time()
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
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

    def _self_heal(self, broken_json, error_msg):
        """Ask Claude to fix its own broken JSON output."""
        print(f"       [HEAL] Asking Claude to fix its JSON...")
        fix_prompt = (
            f"Your previous response was invalid JSON. Error: {error_msg}\n\n"
            f"Here is the broken output:\n{broken_json}\n\n"
            "Fix it and return ONLY the corrected JSON. No commentary."
        )
        return self._call_claude(
            "You fix broken JSON. Return ONLY valid JSON. No markdown. No explanation.",
            fix_prompt,
        )

    def _extract_column_names(self, data_profile):
        """Get the set of all column names from the data profile."""
        return {col["name"] for col in data_profile.get("columns", [])}

    def _validate_columns(self, spec, valid_columns):
        """Check that all column references in the spec exist in the dataset.

        Returns a list of warning strings (empty if all OK).
        """
        warnings = []
        # Collect all column references from visuals
        for page in spec.get("pages", []):
            for visual in page.get("visuals", []):
                data_roles = visual.get("data_roles", {})
                for role_name, fields in data_roles.items():
                    if not isinstance(fields, list):
                        continue
                    for field in fields:
                        # Skip DAX measure names (they won't be in columns)
                        measure_names = {
                            m.get("name", "") for m in spec.get("measures", [])
                        }
                        if field in measure_names:
                            continue
                        if field not in valid_columns:
                            warnings.append(
                                f"Visual '{visual.get('id', '?')}' references "
                                f"'{field}' (role: {role_name}) -- "
                                f"not found in dataset"
                            )
        return warnings


# ------------------------------------------------------------------ #
#  CLI test                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== Claude Interpreter Test ===")
    print("=" * 60)

    project_root = Path(__file__).parent.parent.resolve()
    profile_path = project_root / "output" / "data_profile.json"

    if not profile_path.exists():
        print(f"[ERROR] No data profile at {profile_path}")
        print("       Run core/data_analyzer.py first.")
        sys.exit(1)

    with open(profile_path, "r", encoding="utf-8") as f:
        data_profile = json.load(f)

    interpreter = ClaudeInterpreter()
    spec = interpreter.interpret(
        "Create an executive overview dashboard showing key metrics, "
        "trends over time, and breakdowns by category",
        data_profile,
    )

    out_path = project_root / "output" / "claude_dashboard_spec.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[OK] Saved to {out_path}")
    print(f"\nDashboard: {spec.get('dashboard_title', 'N/A')}")
    print(f"Classification: {spec.get('classification', 'N/A')}")
    print(f"Pages: {len(spec.get('pages', []))}")
    total_visuals = sum(len(p.get("visuals", [])) for p in spec.get("pages", []))
    print(f"Visuals: {total_visuals}")
    print(f"Measures: {len(spec.get('measures', []))}")
